import math
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.calibration import with_absorption
from acoustic_spaces.engine import diffraction as diffraction_module
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine import edge_fade as edge_fade_module
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import (
    EdgeDiffraction,
    Edges,
    along_edge,
    band_frequencies,
    boundary_angles,
    find_edges,
    fresnel_tail,
    transition,
    wedge_prefactor,
    wedge_terms,
)
from acoustic_spaces.engine.edge_fade import EdgeFade
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.reweight import groups_of, reweighted_result
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.scene.model import EarlyStage, Impedance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import edge_diffraction  # noqa: E402

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"

# An L-shaped room: the inner corner at (4, 4) is a vertical edge open over 270 degrees.
# From the source at (8, 2), the shadow boundary of the corner crosses x = 2 at y = 5.
L_PLAN = [[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]]


def _l_room(listeners, **simulation) -> Room:
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"floor_plan": L_PLAN, "height": 3.0},
                "materials": {"m": {"absorption": [0.3] * 8, "scattering": 0.0}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [8.0, 2.0, 1.5]}],
                "listeners": [
                    {"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)
                ],
                "simulation": {"ir_length_s": 0.5, "seed": 1, **simulation},
            }
        )
    )


def _edge_diffraction_alone(monkeypatch) -> None:
    """Leave out the edge fade (:mod:`acoustic_spaces.engine.edge_fade`), to test what the
    edge diffraction does on its own."""
    monkeypatch.setattr(EdgeFade, "terms", lambda self, *args: None)


def _screen(tmp_path: Path, source, listener, obj: Path | None = None) -> Room:
    """A rigid screen in the plane x = 0, so wide and deep that only its top edge (along
    y at z = 0) matters; or the surfaces of ``obj``."""
    if obj is None:
        obj = tmp_path / "screen.obj"
        obj.write_text("v 0 -300 -300\nv 0 300 -300\nv 0 300 0\nv 0 -300 0\ng screen\nf 1 2 3 4\n")
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": list(source)}],
                "listeners": [{"name": "l", "position": list(listener)}],
            }
        )
    )


# -- the coefficient ------------------------------------------------------------------


def test_fresnel_integrals_match_tables() -> None:
    table = {  # z: C(z), S(z) (Abramowitz and Stegun, table 7.7)
        0.5: (0.4923442259, 0.0647324328),
        1.0: (0.7798934004, 0.4382591474),
        2.0: (0.4882534061, 0.3434156784),
        3.0: (0.6057207893, 0.4963129990),
        5.0: (0.5636311887, 0.4991913819),
    }
    z = np.array(list(table))
    tail_c, tail_s = fresnel_tail(z)
    np.testing.assert_allclose(0.5 - tail_c, [c for c, _ in table.values()], atol=1e-9)
    np.testing.assert_allclose(0.5 - tail_s, [s for _, s in table.values()], atol=1e-9)
    # The series and the asymptotic expansion meet smoothly.
    c, s = fresnel_tail(np.array([3.5 - 1e-9, 3.5 + 1e-9]))
    assert abs(c[0] - c[1]) < 1e-7 and abs(s[0] - s[1]) < 1e-7


def test_transition_function_limits() -> None:
    assert transition(np.array([0.0]))[0] == 0
    small = np.array([1e-6])
    np.testing.assert_allclose(
        transition(small), np.sqrt(np.pi * small) * np.exp(0.25j * np.pi), rtol=1e-2
    )
    large = np.array([300.0, 3000.0])
    np.testing.assert_allclose(transition(large), 1 + 0.5j / large - 0.75 / large**2, atol=1e-7)


def _integral_to(a):
    """The integral of exp(-j t^2) from minus infinity to a."""
    z = np.abs(a) * math.sqrt(2 / math.pi)
    tail_c, tail_s = fresnel_tail(z)
    part = math.sqrt(math.pi / 2) * ((0.5 - tail_c) - 1j * (0.5 - tail_s))
    return math.sqrt(math.pi) * np.exp(-0.25j * math.pi) / 2 + np.sign(a) * part


def _half_plane(kr, phi, phi0):
    """Sommerfeld's exact field of a plane wave from phi0 at a rigid half-plane."""
    out = 0
    for psi in (phi - phi0, phi + phi0):
        a = math.sqrt(2 * kr) * np.cos(psi / 2)
        out = out + np.exp(1j * kr * np.cos(psi)) * np.exp(0.25j * math.pi) / math.sqrt(math.pi) * (
            _integral_to(a)
        )
    return out


def _bessel(nu, x, terms=80):
    s = np.arange(terms)
    logs = (2 * s + nu) * math.log(x / 2) - np.array(
        [math.lgamma(k + 1) + math.lgamma(k + nu + 1) for k in s]
    )
    return float(np.sum((-1.0) ** s * np.exp(logs)))


def _wedge(kr, phi, phi0, n):
    """The exact field of a plane wave at a rigid wedge open over n pi (its eigenmodes)."""
    out = np.zeros_like(phi, dtype=complex)
    for m in range(int(3 * kr + 40)):
        nu = m / n
        weight = 1.0 if m == 0 else 2.0
        mode = weight * np.exp(0.5j * math.pi * nu) * _bessel(nu, kr) * math.cos(nu * phi0)
        out += mode * np.cos(nu * phi)
    return 2.0 / n * out


def _utd(kr, phi, phi0, n):
    """UTD for the same plane wave: geometric waves where lit plus the diffracted wave."""
    lit = boundary_angles(n, phi, phi0) >= 0
    terms = wedge_terms(n, phi, phi0, kr, lit)
    total = wedge_prefactor(n, 1.0, 1.0) * terms.sum(axis=-1) * np.exp(-1j * kr) / math.sqrt(kr)
    total += np.where(lit[:, 0], np.exp(1j * kr * np.cos(phi - phi0)), 0)
    total += np.where(lit[:, 1], np.exp(1j * kr * np.cos(phi + phi0)), 0)
    total += np.where(lit[:, 2], np.exp(1j * kr * np.cos(phi + phi0 - 2 * n * math.pi)), 0)
    return total


@pytest.mark.parametrize("kr", [2.0, 10.0, 50.0])
def test_half_plane_matches_the_exact_solution(kr: float) -> None:
    phi0 = math.radians(60)
    phi = np.linspace(0.001, 2 * math.pi - 0.001, 1441)
    np.testing.assert_allclose(_utd(kr, phi, phi0, 2.0), _half_plane(kr, phi, phi0), atol=1e-8)
    # Exactly on the shadow and reflection boundaries, from either side.
    for edge in (phi0 + math.pi, math.pi - phi0):
        on = np.array([edge - 1e-12, edge, edge + 1e-12])
        np.testing.assert_allclose(_utd(kr, on, phi0, 2.0), _half_plane(kr, on, phi0), atol=1e-6)


def test_wedge_reference_is_the_half_plane_for_n_2() -> None:
    phi0 = math.radians(60)
    phi = np.linspace(0.01, 2 * math.pi - 0.01, 361)
    np.testing.assert_allclose(_wedge(5.0, phi, phi0, 2.0), _half_plane(5.0, phi, phi0), atol=1e-9)


@pytest.mark.parametrize("n", [1.25, 1.5, 1.75])
def test_wedges_match_the_exact_solution(n: float) -> None:
    phi0 = math.radians(60)
    phi = np.linspace(0.01, n * math.pi - 0.01, 721)
    for kr in (3.0, 10.0):
        exact = _wedge(kr, phi, phi0, n)
        error_db = 20 * np.log10(np.abs(_utd(kr, phi, phi0, n)) / np.abs(exact))
        assert np.abs(error_db).max() < 0.25


# -- against the exact solution through the whole early stage --------------------------


def test_the_exact_solution_runs_at_the_engines_speed_of_sound(tmp_path: Path) -> None:
    """validation/edge_diffraction.py evaluates BTM at the speed of sound of its scenes,
    whose air is at the default 20 °C, and at the wave numbers of the engine's band
    frequencies at that speed."""
    room = edge_diffraction._room(
        tmp_path, "screen", edge_diffraction.screen_quads(), (-5, 0, -2), (5, 0, -1)
    )
    c = room.speed_of_sound()
    assert edge_diffraction.C == pytest.approx(c, rel=1e-12)
    frequencies = edge_diffraction.band_frequencies()[0].ravel()
    np.testing.assert_allclose(edge_diffraction.K, 2.0 * np.pi * frequencies / c, rtol=1e-12)


def test_the_bands_in_energy_are_low_on_a_shadow_boundary(tmp_path: Path) -> None:
    """On the shadow boundary of a thin screen, of a right-angled corner and of a wedge
    open over 225 degrees, the bands that add in energy lose the cross term between the
    direct sound's wave and the edge's diffracted wave: through ``early_paths`` they are
    as far below the exact solution (BTM, Svensson, Fred and Vanderkooy, J. Acoust. Soc.
    Am. 106, 1999) as the validation report's known issues say, the blunter the edge the
    more; the 63 and 125 Hz bands, in pressure by default, are within 0.08 dB there and
    10.5 degrees into the shadow. A change that keeps the cross term moves these, and
    the report with them."""
    rows = {(row["case"], row["where"]): row for row in edge_diffraction.shadow_boundary(tmp_path)}
    on = "on the shadow boundary"
    screen = rows["thin screen", on]["in_energy_minus_btm_db"]
    corner = rows["right-angled corner", on]["in_energy_minus_btm_db"]
    wedge = rows["wedge open over 225 degrees", on]["in_energy_minus_btm_db"]
    np.testing.assert_allclose(screen[2:5], [-0.69, -0.50, -0.36], atol=0.02)
    np.testing.assert_allclose(corner[2:], [-1.73, -1.32, -0.98, -0.72, -0.52, -0.38], atol=0.02)
    np.testing.assert_allclose([*wedge[2:5], wedge[7]], [-2.22, -1.79, -1.38, -0.56], atol=0.02)
    for row in rows.values():
        assert np.abs(row["default_minus_btm_db"][:2]).max() < 0.08
        assert row["default_minus_btm_db"][2:] == row["in_energy_minus_btm_db"][2:]


# -- edges ----------------------------------------------------------------------------


def test_edges_are_convex_corners_and_free_edges(tmp_path: Path) -> None:
    shoebox = Room.from_scene(Scene.from_yaml(SHOEBOX))
    assert len(find_edges(TraceModel.from_room(shoebox))) == 0  # room corners only

    edges = find_edges(TraceModel.from_room(_l_room([[2.0, 2.0, 1.5]])))
    assert len(edges) == 1
    assert edges.wedge[0] == pytest.approx(1.5)
    np.testing.assert_allclose(np.abs(edges.direction[0]), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(edges.start[0][:2], [4, 4])
    assert edges.length[0] == pytest.approx(3.0)
    # The open side faces the room: its middle points between the two arms.
    np.testing.assert_allclose(edges.bisector()[0][:2], [-math.sqrt(0.5)] * 2, atol=1e-12)

    screen = find_edges(TraceModel.from_room(_screen(tmp_path, [-5, 0, -2], [5, 0, -2])))
    assert len(screen) == 4 and np.all(screen.wedge == 2.0)  # a free edge on each side
    assert np.all(screen.face[:, 0] == screen.face[:, 1])


def test_pieces_of_one_edge_are_joined(tmp_path: Path) -> None:
    # A screen made of two triangles per metre along its top edge.
    obj = tmp_path / "strips.obj"
    lines = []
    for k in range(5):
        lines += [f"v 0 {k} -2", f"v 0 {k} 0"]
    lines.append("g screen")
    for k in range(4):
        a, b = 2 * k + 1, 2 * k + 3
        lines += [f"f {a} {b} {b + 1}", f"f {a} {b + 1} {a + 1}"]
    obj.write_text("\n".join(lines))
    room = _screen(tmp_path, [-1, 2, -1], [1, 2, -1], obj)
    edges = find_edges(TraceModel.from_room(room))
    top = np.flatnonzero(np.isclose(edges.start[:, 2], 0) & np.isclose(edges.end[:, 2], 0))
    assert len(top) == 1 and edges.length[top[0]] == pytest.approx(4.0)


# -- diffracted paths -----------------------------------------------------------------


def _levels(room: Room, **kwargs) -> np.ndarray:
    """Early energy (dB re the direct sound in free field) of every listener, per band."""
    out = []
    source = np.array(room.scene.sources[0].position)
    for listener in room.scene.listeners:
        paths = early_paths(room, listener=listener.name, **kwargs)
        free = 1 / (4 * math.pi * np.sum((np.array(listener.position) - source) ** 2))
        out.append(10 * np.log10(total_energy(paths) / free))
    return np.array(out)


def test_levels_are_continuous_across_the_shadow_boundary(monkeypatch) -> None:
    listeners = [[2.0, y, 1.5] for y in (4.99, 4.999, 5.0, 5.001, 5.01)]
    room = _l_room(listeners)
    # The direct sound and the floor and ceiling reflections all end on the boundary.
    plain = _levels(room, max_order=1, diffraction="none")
    assert np.all(plain[1] - plain[3] > 5.0)
    # With "direct" the edge fade takes the reflections round the corner.
    for mode in ("direct", "first_order"):
        step = np.abs(np.diff(_levels(room, max_order=1, diffraction=mode), axis=0))
        assert step.max() < 0.25
    # Without it only the direct sound bends round the corner: the reflections still jump.
    _edge_diffraction_alone(monkeypatch)
    direct = np.abs(np.diff(_levels(room, max_order=1, diffraction="direct"), axis=0))
    assert 1.0 < direct.max() < np.abs(np.diff(plain, axis=0)).max() - 1.0
    first = np.abs(np.diff(_levels(room, max_order=1, diffraction="first_order"), axis=0))
    assert first.max() < 0.25


def test_levels_are_continuous_across_a_reflection_boundary() -> None:
    # The reflection off the wall y = 4 (x > 4) ends behind the corner on a line through
    # (2, 3), where the direct sound carries on; the two do not interfere there either.
    room = _l_room([[2.0, 3.0 + d, 1.5] for d in (-1e-4, 1e-4)])
    for order, mode in ((1, "direct"), (2, "first_order")):
        step = np.diff(_levels(room, max_order=order, diffraction=mode), axis=0)
        assert np.abs(step).max() < 0.05


def test_shadow_gets_a_diffracted_path_and_lit_side_barely_changes() -> None:
    room = _l_room([[2.0, 8.0, 1.5], [9.0, 1.0, 1.2]])
    shadow = early_paths(room, listener="l0", max_order=0, diffraction="direct")
    assert len(shadow) == 1 and shadow.edge[0] == 0 and shadow.order[0] == 0
    np.testing.assert_allclose(shadow.points[0][1][:2], [4, 4])  # over the corner
    np.testing.assert_allclose(shadow.delay_s[0] * room.speed_of_sound(), shadow.distance_m[0])
    # Low frequencies bend round the corner far better than high ones.
    assert shadow.energy[0, 0] > 30 * shadow.energy[0, -1]
    direction = shadow.points[0][1] - np.array([2.0, 8.0, 1.5])
    np.testing.assert_allclose(shadow.direction[0], direction / np.linalg.norm(direction))

    # In energy: in pressure the low bands add or cancel by where the listener is.
    room = _l_room([[2.0, 8.0, 1.5], [9.0, 1.0, 1.2]], early={"coherent_max_band_hz": None})
    plain = _levels(room, diffraction="none")
    with_edges = _levels(room)
    assert np.all(np.abs(with_edges[1] - plain[1]) < 0.5)  # far from the corner
    assert np.all(with_edges[0][:2] - plain[0][:2] > 1.0)  # behind it, at low frequencies


def test_screen_attenuation_follows_maekawa(tmp_path: Path) -> None:
    source, listener = np.array([-5.0, 0.0, -2.0]), np.array([5.0, 0.0, -1.0])
    room = _screen(tmp_path, source, listener)
    model = TraceModel.from_room(room)
    c = room.speed_of_sound()
    edges = find_edges(model)

    def energy(a, b):
        diffraction = EdgeDiffraction(
            model, a, speed_of_sound=c, air=np.zeros(8), mode="direct", edges=edges
        )
        empty = early_stage.find_paths(
            model, a, b, [], speed_of_sound=c, air=np.zeros(8), max_delay_s=1.0
        )
        assert len(empty) == 0  # the screen blocks the direct sound
        return diffraction.apply(b, empty, 1.0).energy.sum(axis=0)

    e = energy(source, listener)
    np.testing.assert_allclose(energy(listener, source), e, rtol=1e-9)  # reciprocity
    distance = np.linalg.norm(listener - source)
    loss = -10 * np.log10(e * 4 * math.pi * distance**2)
    detour = np.linalg.norm(source) + np.linalg.norm(listener) - distance
    fresnel = 2 * detour * np.array(OCTAVE_BANDS_HZ) / c
    maekawa = 10 * np.log10(3 + 20 * fresnel)
    np.testing.assert_allclose(loss, maekawa, atol=1.5)


def test_a_screen_diffracts_all_four_terms_where_lit(tmp_path: Path) -> None:
    # In front of a thin screen, lit by the direct sound and its reflection off the
    # screen, well off both boundaries: the diffracted wave is all four terms in phase,
    # the reflection off the screen's back included, though that never reaches here.
    source, listener = np.array([-2.0, 0.0, -1.0]), np.array([-1.0, 0.0, -3.0])
    room = _screen(tmp_path, source, listener)
    model = TraceModel.from_room(room)
    c, air = room.speed_of_sound(), np.zeros(8)
    specular = early_stage.find_paths(
        model, source, listener, [np.array([[0]])], speed_of_sound=c, air=air, max_delay_s=1.0
    )
    assert len(specular) == 2  # the direct sound and the reflection
    diffraction = EdgeDiffraction(model, source, speed_of_sound=c, air=air, mode="first_order")
    paths = diffraction.apply(listener, specular, 1.0)
    bent = paths.energy[paths.diffracted]
    assert len(bent) == 1
    frequencies, _ = band_frequencies()
    k = 2 * math.pi * frequencies / c
    r1, r2 = np.linalg.norm(source), np.linalg.norm(listener)
    terms = wedge_terms(2.0, math.atan2(1, 3), math.atan2(2, 1), k * r1 * r2 / (r1 + r2))
    pressure = wedge_prefactor(2.0, k, 1.0) * terms.sum(axis=-1)
    expected = np.mean(np.abs(pressure) ** 2, axis=1) / (4 * math.pi * r1 * r2 * (r1 + r2))
    np.testing.assert_allclose(bent[0][4:], expected[4:], rtol=0.002)  # from 1 kHz up


def test_paths_reflecting_at_the_ends_of_an_edge() -> None:
    # At (2, 8) the listener is as far from the corner as the source and as high: the
    # paths over the corner by the floor or the ceiling touch it at its ends, where
    # reflecting before the edge and after it is the same path.
    room = _l_room([[2.0, 8.0 + d, 1.5] for d in (-1e-4, 0.0, 1e-4)])
    levels = _levels(room, max_order=1, diffraction="first_order")
    assert np.abs(np.diff(levels, axis=0)).max() < 0.01


def test_a_listener_on_the_source() -> None:
    # A map point can fall on a source: next to its direct sound, nothing over an edge
    # counts.
    room = _l_room([[2.0, 8.0, 1.5]])
    source = np.array(room.scene.sources[0].position)
    diffraction = EdgeDiffraction(
        TraceModel.from_room(room), source, speed_of_sound=room.speed_of_sound(), air=np.zeros(8)
    )
    assert len(diffraction.apply(source, early_stage.EarlyReflections.empty(8), 1.0)) == 0


def test_reflections_bend_round_edges_too(monkeypatch) -> None:
    # Behind the corner the floor and ceiling reflections arrive over the edge as well.
    room = _l_room([[2.0, 8.0, 1.5]])
    # With "direct" the edge fade takes the walls' reflections over it.
    direct = early_paths(room, max_order=1, diffraction="direct")
    assert (direct.diffracted & (direct.order == 1)).sum() >= 2
    _edge_diffraction_alone(monkeypatch)
    direct = early_paths(room, max_order=1, diffraction="direct")
    first = early_paths(room, max_order=1, diffraction="first_order")
    assert direct.diffracted.sum() == 1
    bent = first.diffracted & (first.order == 1)
    assert bent.sum() >= 2
    for k in np.flatnonzero(bent):
        assert len(first.points[k]) == 4 and len(first.faces[k]) == 1


def test_simulate_and_settings() -> None:
    assert EarlyStage().diffraction == "first_order"
    with pytest.raises(ValueError):
        EarlyStage(diffraction="all")
    room = _l_room([[2.0, 8.0, 1.5]], late={"method": "statistical"})
    result = simulate(room)
    response = result["s", "l0"]
    assert response.early.diffracted.any()
    assert result.details["s"]["edges"] == 1 and result.details["s"]["diffracted_paths"] >= 1
    off = _l_room([[2.0, 8.0, 1.5]], late={"method": "statistical"}, early={"diffraction": "none"})
    plain = simulate(off)["s", "l0"]
    assert not plain.early.diffracted.any() and "edges" not in simulate(off).details["s"]
    # Where the paths add in energy (in pressure they may cancel).
    count = response.coherent_bands
    assert count == 2 and np.all(response.parameters().g[count:] > plain.parameters().g[count:])
    assert np.all(np.isfinite(response.impulse_response(16_000)))


def test_coverage_maps_hear_round_corners() -> None:
    from acoustic_spaces.engine import coverage_map
    from acoustic_spaces.scene import Coverage

    room = _l_room([[2.0, 8.0, 1.5]], late={"method": "statistical"})
    settings = Coverage(spacing=2.0)
    bent = coverage_map(room, settings=settings)
    off = room.scene.simulation.model_copy(deep=True)
    off.early.diffraction = "none"
    plain = coverage_map(room, settings=settings, simulation=off)
    assert bent.details["s"]["diffracted_paths"] > 0 and "edges" not in plain.details["s"]
    shadow = (bent.grid.points[:, 0] < 4) & (bent.grid.points[:, 1] > 6)
    assert shadow.any()
    # Behind the corner, at 250 Hz: more sound, and more of it early. The statistical
    # tail is Barron's energy less the early paths', so of what bends round the corner
    # only the direct sound adds to the total. (At 63 and 125 Hz the paths add in
    # pressure, and those of the edge fade may cancel others.)
    for name, rise in (("g", 0.01), ("c80", 0.01)):
        values = bent.get().values[name][shadow, 2] - plain.get().values[name][shadow, 2]
        assert np.all(values > rise)


def test_re_weighting_turns_the_phase_of_reflections_on_the_way_round_an_edge(
    monkeypatch,
) -> None:
    # Behind the corner the floor's reflections arrive over the edge. Re-weighted for
    # another floor, their phases at 63 and 125 Hz, where the paths add in pressure, turn
    # as tracing again has them. (Tracing again leaves out the paths that the floor's
    # absorption takes under the energy floor; re-weighting keeps them.)
    for module in (diffraction_module, edge_fade_module):
        monkeypatch.setattr(module, "MIN_RELATIVE_ENERGY", 0.0)
    scene = _l_room([[2.0, 8.0, 1.5]], late={"rays": 300}).scene
    floor = [0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6]
    scene.materials["floor"] = scene.materials["m"].model_copy(update={"absorption": floor})
    scene.assignments = {"floor": "floor"}
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    traced = simulate(room, groups=face_group)
    changed = with_absorption(scene, ["floor"], [np.add(floor, 0.2)])
    guess = reweighted_result(traced, scene, face_group, materials, changed)["s", "l0"].early
    fresh = simulate(Room.from_scene(changed))["s", "l0"].early
    base = traced["s", "l0"].early
    np.testing.assert_array_equal(guess.delay_s, fresh.delay_s)
    floor_group = materials.index("floor")
    on_floor = [
        np.any(face_group[np.asarray(f, dtype=np.int64)] == floor_group) for f in guess.faces
    ]
    bent = guess.diffracted & np.array(on_floor)
    assert bent.sum() >= 10
    assert np.abs(base.spectrum[bent] - fresh.spectrum[bent]).max() > 0.1
    np.testing.assert_allclose(guess.spectrum, fresh.spectrum, rtol=0, atol=1e-12)


# -- over two edges -------------------------------------------------------------------

# The level over a 1 m thick barrier (dB re the direct sound in free field), from the
# exact BTM solution (validation/edge_diffraction.py).
BARRIER_BTM = {
    ((-3.0, 0.0, -1.0), (4.0, 0.0, -1.5)): [
        -9.45,
        -11.79,
        -14.65,
        -18.06,
        -22.05,
        -26.66,
        -31.83,
        -37.46,
    ],
    ((-6.0, -2.0, -0.3), (3.0, 3.0, -4.0)): [
        -11.85,
        -14.66,
        -17.77,
        -21.04,
        -24.40,
        -27.85,
        -31.43,
        -35.23,
    ],
}


def _barrier(tmp_path: Path, source, listeners, material=None, top=None) -> Room:
    """A block from x = 0 to 1 m with its top at z = 0, open underneath, so wide and
    deep that only its two top corners (along y) matter: rigid, or of ``material``, and
    its top of ``top`` if given."""
    a, h = 300, -300
    quads = [
        [(0, -a, 0), (1, -a, 0), (1, a, 0), (0, a, 0)],
        [(0, -a, h), (0, -a, 0), (0, a, 0), (0, a, h)],
        [(1, -a, h), (1, a, h), (1, a, 0), (1, -a, 0)],
        [(0, -a, h), (1, -a, h), (1, -a, 0), (0, -a, 0)],
        [(0, a, h), (0, a, 0), (1, a, 0), (1, a, h)],
    ]
    lines = [f"v {x} {y} {z}" for quad in quads for x, y, z in quad]
    lines += ["g top", "f 1 2 3 4", "g block"]
    lines += [f"f {4 * i + 1} {4 * i + 2} {4 * i + 3} {4 * i + 4}" for i in range(1, 5)]
    obj = tmp_path / "block.obj"
    obj.write_text("\n".join(lines) + "\n")
    scene = {
        "geometry": {"file": str(obj)},
        "materials": {"hard": material or {"absorption": [0.0] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": list(source)}],
        "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
    }
    if top is not None:
        scene["materials"]["top"] = top
        scene["assignments"] = {"top": "top"}
    return Room.from_scene(Scene.model_validate(scene))


def _over_barrier(room: Room, mode: str, coherent: int = 0) -> np.ndarray:
    """The level of every listener over the barrier, without air, as in BARRIER_BTM,
    with the paths in pressure in the lowest ``coherent`` bands."""
    model = TraceModel.from_room(room)
    c, air = room.speed_of_sound(), np.zeros(8)
    source = np.array(room.scene.sources[0].position)
    diffraction = EdgeDiffraction(
        model, source, speed_of_sound=c, air=air, mode=mode, coherent=coherent
    )
    out = []
    for listener in room.scene.listeners:
        at = np.array(listener.position)
        paths = early_stage.find_paths(
            model, source, at, [], speed_of_sound=c, air=air, max_delay_s=1
        )
        paths = diffraction.apply(at, paths, 1.0)
        free = 1 / (4 * math.pi * np.sum((at - source) ** 2))
        with np.errstate(divide="ignore"):
            out.append(10 * np.log10(total_energy(paths) / free))
    return np.array(out)


def test_a_thick_barrier_takes_paths_over_two_edges(tmp_path: Path) -> None:
    source, listener = (-3.0, 0.0, -1.0), (4.0, 0.0, -1.5)
    room = _barrier(tmp_path, source, [listener])
    assert len(early_paths(room, max_order=1, diffraction="first_order")) == 0
    paths = early_paths(room, max_order=1, diffraction="second_order")
    assert len(paths) == 1 and paths.diffracted[0] and paths.order[0] == 0
    np.testing.assert_allclose(paths.points[0], [source, [0, 0, 0], [1, 0, 0], listener], atol=1e-9)
    distance = math.sqrt(10) + 1 + math.sqrt(11.25)
    assert paths.distance_m[0] == pytest.approx(distance)
    assert paths.delay_s[0] == pytest.approx(distance / room.speed_of_sound())
    for (s, r), btm in BARRIER_BTM.items():
        level = _over_barrier(_barrier(tmp_path, s, [r]), "second_order")[0]
        error = level - np.array(btm)
        assert np.abs(error).max() < 1.0 and np.abs(error[3:]).max() < 0.2
        # The same path backwards.
        back = _over_barrier(_barrier(tmp_path, r, [s]), "second_order")[0]
        np.testing.assert_allclose(back, level, atol=1e-9)


@pytest.mark.parametrize("coherent", [0, 8])
def test_paths_over_two_edges_along_an_absorbing_top_lose_what_the_ground_wave_does(
    tmp_path: Path, coherent: int
) -> None:
    # Along the top the wave from the first corner is the Weyl-van der Pol field over a
    # locally reacting face at grazing incidence, 2 F(w) times the free one, which goes
    # to the rigid face's as the admittance does; not its plane wave's R = -1, which
    # cancels the incident wave. So the level over an absorbing top falls short of the
    # rigid one by |F(w)| over the 1 m top, whether the paths add in pressure or in
    # energy, and stays with BTM's for the rigid block.
    from scipy.optimize import brentq
    from scipy.special import wofz

    def level(s, r, absorption):
        material = {"absorption": [absorption] * 8, "scattering": 0.0}
        return _over_barrier(_barrier(tmp_path, s, [r], material), "second_order", coherent)[0]

    def paris(z):  # random-incidence absorption of a real normalised impedance z
        return 8.0 / z * (1.0 + 1.0 / (1.0 + z) - 2.0 / z * math.log(1.0 + z))

    beta = 1.0 / brentq(lambda z: paris(z) - 0.02, 1.0, 1e6)
    w = 0.5 * (1 + 1j) * np.sqrt(2 * np.pi * np.array(OCTAVE_BANDS_HZ) / 343.0) * beta
    loss = 20 * np.log10(np.abs(1 + 1j * math.sqrt(math.pi) * w * wofz(w)))  # -0.02 to -0.24
    for (s, r), btm in BARRIER_BTM.items():
        rigid = level(s, r, 0.0)
        np.testing.assert_allclose(level(s, r, 1e-6), rigid, atol=1e-3)
        absorbing = level(s, r, 0.02)
        if s[1] == r[1]:  # straight across the top, 1 m
            np.testing.assert_allclose(absorbing - rigid, loss, atol=0.05)
        assert np.all(absorbing < rigid) and np.all(absorbing > rigid - 0.3)
        assert np.abs(absorbing - np.array(btm)).max() < 1.0
        back = level(r, s, 0.02)
        np.testing.assert_allclose(back, absorbing, atol=1e-9)
    # Of an impedance (a hard ground's, 20 MPa s/m²) the same, with its own F(w).
    (s, r), _ = next(iter(BARRIER_BTM.items()))
    material = {"impedance": {"flow_resistivity": 20000}, "scattering": 0.0}
    hard = _over_barrier(_barrier(tmp_path, s, [r], material), "second_order", coherent)[0]
    f = np.array(OCTAVE_BANDS_HZ, dtype=float)
    beta = np.conj(Impedance(flow_resistivity=20000).admittance(f))  # as e^(-iwt)
    w = 0.5 * (1 + 1j) * np.sqrt(2 * np.pi * f / 343.0) * beta
    loss = 20 * np.log10(np.abs(1 + 1j * math.sqrt(math.pi) * w * wofz(w)))  # 0.01 to 0.42
    np.testing.assert_allclose(hard - level(s, r, 0.0), loss, atol=0.1)


@pytest.mark.parametrize("mode", ["first_order", "second_order"])
def test_an_absorbing_top_below_its_admittance_in_cosine_is_not_louder_than_a_rigid_one(
    tmp_path: Path, mode: str
) -> None:
    # A source and a listener a little above the plane of the top, 3 m either side: the
    # direct sound, the reflection off the top and the paths over its corners, with the
    # fade. Near grazing incidence a point source's sound over a locally reacting face
    # with a real admittance (here 0.10) is 2 F(w) times the free one, |F(w)| < 1, so an
    # absorbing top is no louder than a rigid one. Checked only where the source sees
    # the top and both corners at cosines below the admittance (h < 0.31 m): from there
    # to h = 0.42 m, where those reflections turn their phase over one after the other,
    # the level is up to 9.6 dB above the rigid one's (a known issue, docs/validation.md).
    # With the path over both corners, only in the bands where the paths arrive nearly
    # in phase, the way over both corners less than a quarter of a wavelength longer
    # than the straight one (all but 4 and 8 kHz at h = 0.3 m): beyond that the path
    # over both lowers the level over the rigid top (by 0.94 dB at 4 kHz there in the
    # exact solution), and less over an absorbing one, which weakens it.
    wavelength = 343.0 / np.array(OCTAVE_BANDS_HZ, dtype=float)
    for h in (0.05, 0.1, 0.3):
        source, listener = (-3.0, 0.0, h), (4.0, 0.0, h)
        rigid = _barrier(tmp_path, source, [listener])
        absorbing = _barrier(
            tmp_path, source, [listener], top={"absorption": [0.5] * 8, "scattering": 0.0}
        )
        levels = [
            10 * np.log10(total_energy(early_paths(room, max_order=1, diffraction=mode)))
            for room in (rigid, absorbing)
        ]
        near = wavelength > 4.0 * (2.0 * math.hypot(3.0, h) + 1.0 - 7.0)
        assert near.sum() == (6 if h == 0.3 else 8)
        if mode == "first_order":
            near[:] = True
        assert np.all((levels[1] - levels[0])[near] < 0.5), (h, levels[1] - levels[0])


def test_levels_are_continuous_over_a_barriers_top(tmp_path: Path) -> None:
    # As the listener rises through the plane of the top behind the barrier, the path
    # over the near corner alone appears and takes over from the one over both; and as
    # the source does in front of it, with the far corner.
    steps = (-1e-3, 0.0, 1e-3)
    behind = _barrier(tmp_path, (-3.0, 0.0, -1.0), [(4.0, 0.0, dz) for dz in steps])
    first = _over_barrier(behind, "first_order")
    assert np.all(np.isinf(first[0])) and np.all(np.isfinite(first[1:]))
    rising = [_barrier(tmp_path, (-3.0, 0.0, dz), [(4.0, 0.0, -1.5)]) for dz in steps]
    for level in (
        np.concatenate([_over_barrier(room, "second_order") for room in rising]),
        _over_barrier(behind, "second_order"),
    ):
        jump = np.abs(np.diff(level, axis=0))
        assert jump.max() < 1.0 and jump[:, 4:].max() < 0.1


#: The level just over a block's top (dB re the direct sound in free field, 63 to 500
#: Hz) from the exact BTM solution with the geometric waves (validation/
#: edge_diffraction.py, ``in_sight``), by the block's width, how far the source stands
#: before it and the listener past it, and their heights above the plane of the top (m).
IN_SIGHT_BTM = {
    (10.0, 10.0, 5.0, (0.01, 0.01)): [-5.220, -5.439, -5.568, -5.647],
    (10.0, 10.0, 5.0, (0.5, 0.5)): [-3.635, -3.180, -2.359, -1.114],
    (10.0, 10.0, 5.0, (0.5, 0.05)): [-4.500, -4.436, -4.162, -3.655],
    (10.0, 10.0, 5.0, (1.0, 1.0)): [-2.107, -1.009, 0.607, 2.547],
    (1.0, 3.0, 3.0, (0.05, 0.05)): [-4.354, -4.723, -4.908, -4.926],
}


def _just_over(tmp_path: Path, width: float, source, listener, mode="second_order") -> np.ndarray:
    """The whole early stage's level (dB re the direct sound in free field, without the
    air) over a rigid block ``width`` m wide with its top at z = 0, with the paths in
    pressure up to 500 Hz."""
    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500}}
    quads = edge_diffraction.barrier_quads(width)
    room = edge_diffraction._room(tmp_path, "over", quads, source, listener, simulation)
    return edge_diffraction.whole_early_stage(room)


@pytest.mark.parametrize("case", list(IN_SIGHT_BTM))
def test_sound_that_just_clears_both_corners_of_a_block_takes_the_path_over_both(
    tmp_path: Path, case
) -> None:
    # The source and the listener a little above the plane of the top, in sight of each
    # other: both corners' terms take their part off the direct sound and its reflection
    # off the top, and the path over both gives back what they take twice. Without it
    # the level was 15 to 23 dB low from 63 to 500 Hz 1 cm above the 10 m top, 6.7 dB at
    # 63 Hz 0.5 m above it, and 11 to 14 dB low 5 cm above the 1 m top, whose corners
    # are 0.18 wavelengths apart at 63 Hz.
    width, before, past, (hs, hr) = case
    source, listener = (-before, 0.0, hs), (width + past, 0.0, hr)
    level = _just_over(tmp_path, width, source, listener)
    error = level[:4] - np.array(IN_SIGHT_BTM[case])
    assert np.abs(error).max() < (0.1 if width == 10.0 else 1.25), error
    # The same path backwards.
    np.testing.assert_allclose(_just_over(tmp_path, width, listener, source), level, atol=1e-6)
    # Paths over one edge alone leave the level low near the plane: by more than 6 dB
    # at 63 Hz up to 0.5 m above the 10 m top, and by 2.6 dB 1 m above it.
    alone = _just_over(tmp_path, width, source, listener, "first_order")[:4]
    assert alone[0] < level[0] - (2.5 if hs == 1.0 else 6.0)


def test_the_level_is_the_same_all_round_the_plane_of_a_block_s_top(tmp_path: Path) -> None:
    # Within a millimetre of the plane of a 10 m block's top, above it or below, on
    # either side of the block, the sound is the same: the path over both corners alone
    # (both below), with the path over one corner (one above), or with the direct sound,
    # its reflection off the top and both corners' paths (both above, also where the
    # source is so near the plane that no ray finds the top and the reflection misses
    # it). Both above, the level was 15 to 38 dB lower, or up to 6 dB higher (3 dB in
    # the bands that add in energy) with the path over both and no term for the
    # reflection that misses the top.
    heights = (-1e-3, -1e-6, 1e-6, 1e-3)
    levels = np.array(
        [
            [_just_over(tmp_path, 10.0, (-10.0, 0.0, hs), (15.0, 0.0, hr)) for hr in heights]
            for hs in heights
        ]
    )
    spread = levels.max(axis=(0, 1)) - levels.min(axis=(0, 1))
    assert spread.max() < 0.2, spread
    np.testing.assert_allclose(levels, levels.transpose(1, 0, 2), atol=0.03)
    # The listener a centimetre up, where the reflection reaches it or misses the top.
    up = np.array(
        [_just_over(tmp_path, 10.0, (-10.0, 0.0, hs), (15.0, 0.0, 1e-2)) for hs in heights]
    )
    assert np.abs(up - levels[:, -1]).max() < 0.2


def test_the_faces_of_diffracting_edges_are_tested_where_no_ray_finds_them(
    tmp_path: Path,
) -> None:
    # A source a millimetre above the plane of a block's top, 10 m before it, sees the
    # top edge-on: no ray hits it. The terms at the top's corners continue the
    # reflection off it, so with edge diffraction its plane is among the planes tested;
    # not without, nor from in the plane or below it, where the source is not in front
    # of the top.
    from acoustic_spaces.engine.simulate import _candidates, _Edges

    quads = edge_diffraction.barrier_quads(10.0)
    room = edge_diffraction._room(tmp_path, "edge_on", quads, (-10.0, 0.0, 1e-3), (15.0, 0.0, 1.0))
    model = TraceModel.from_room(room)
    top = int(model.face_plane[0])
    assert abs(model.plane_normal[top][2]) == pytest.approx(1.0)
    rays = room.scene.simulation.early.rays
    for height, mode, there in (
        (1e-3, "second_order", True),
        (1e-3, "first_order", True),
        (1e-3, "direct", True),
        (1e-3, "none", False),
        (0.0, "second_order", False),
        (-1e-3, "second_order", False),
        (1.0, "none", True),  # in plain sight the rays find it themselves
    ):
        source = np.array([-10.0, 0.0, height])
        found = _candidates(model, source, "ray_tracing", 2, rays, 1, _Edges(model), mode)
        assert (top in found[0][:, 0]) == there, (height, mode)
        assert len(np.unique(found[0][:, 0])) == len(found[0])
        plain = _candidates(model, source, "ray_tracing", 2, rays, 1)
        np.testing.assert_array_equal(found[1], plain[1])
        assert set(plain[0][:, 0]) <= set(found[0][:, 0])


@pytest.mark.parametrize("height", [1e-3, 1e-2])
def test_sound_that_grazes_an_absorbing_top_from_above_loses_what_the_ground_wave_does(
    tmp_path: Path, height: float
) -> None:
    # Just above the plane of a 1 m top absorbing 0.5, as just below it, the sound is
    # the path over both corners along the top: below the rigid top's by the ground
    # wave's loss over the top, 0.9 dB at 63 Hz to 9.3 dB at 8 kHz. Without that path
    # the absorbing top read 0.3 to 1.3 dB louder than the rigid one from above.
    loss = np.array(edge_diffraction.ground_wave(0.5, 1.0))
    assert loss[0] == pytest.approx(-0.86, abs=0.01) and loss[-1] == pytest.approx(-9.33, abs=0.01)
    for h, tolerance in ((height, 0.55), (-height, 0.15)):
        source, listener = (-3.0, 0.0, h), (4.0, 0.0, h)
        rigid = _barrier(tmp_path, source, [listener])
        absorbing = _barrier(
            tmp_path, source, [listener], top={"absorption": [0.5] * 8, "scattering": 0.0}
        )
        levels = [
            10 * np.log10(total_energy(early_paths(room, max_order=1, diffraction="second_order")))
            for room in (rigid, absorbing)
        ]
        np.testing.assert_allclose(levels[1] - levels[0], loss, atol=tolerance)


def test_two_screens_in_a_row_at_grazing_incidence(tmp_path: Path) -> None:
    # The source and the listener a millimetre above the line of two screens' tops, 10 m
    # before the first and 5 m past the second, the screens 10 m apart: of the direct
    # sound less each top's term, the path over both gives back what they take twice.
    # Two knife edges at grazing incidence leave 1/4 + atan(sqrt(a c / (b (a + b +
    # c)))) / (2 pi) of the free field at high frequencies, 0.317 or -9.98 dB here. The
    # product of the two edges' coefficients, each in the other's transition zone (a
    # known issue), gives 1/4, 2.06 dB less: just below the line, where the path over
    # both is all there is, the engine is that far below, and just above it between
    # the two (1.0 to 1.3 dB below from 2 kHz up; it was 7.6 to 9.5 dB below).
    grazing = edge_diffraction.grazing_two_screens(tmp_path)
    assert grazing["closed_form_db"] == pytest.approx(-9.98, abs=0.005)
    product = 20.0 * math.log10(0.25) - grazing["closed_form_db"]
    assert product == pytest.approx(-2.06, abs=0.005)
    above, below = (np.array(row["second_order_db"]) for row in grazing["rows"])
    error = above[5:] - grazing["closed_form_db"]
    assert np.all(error < -0.8) and np.all(error > -1.5), error
    error = below[5:] - grazing["closed_form_db"]
    assert np.all(np.abs(error - product) < 0.1), error
    alone = np.array(grazing["rows"][0]["first_order_db"])
    assert np.all(alone[5:] < -30.0)


def _past(tmp_path: Path, quads, source, listener, mode="second_order", coherent=500) -> np.ndarray:
    """The whole early stage's level (dB re the direct sound in free field, without the
    air) past the rigid ``quads``, with the paths in pressure up to ``coherent`` Hz
    (None: in energy in every band)."""
    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": coherent}}
    room = edge_diffraction._room(tmp_path, "past", quads, source, listener, simulation)
    return edge_diffraction.whole_early_stage(room)


def _past_screens(tmp_path: Path, places, source, listener, mode="second_order", coherent=500):
    """:func:`_past` rigid screens at x = ``places`` with their tops at z = 0."""
    quads = sum((edge_diffraction.screen_quads(x) for x in places), [])
    return _past(tmp_path, quads, source, listener, mode, coherent)


@pytest.mark.parametrize("places", [(0.0, 10.0), (0.0, 5.0, 10.0)])
def test_screens_in_a_row_with_both_ends_level_with_their_tops(tmp_path: Path, places) -> None:
    # Exactly level with the tops of two or three screens, the straight way from the
    # source to the listener is not found, but the paths over the tops are, as their
    # rays start just off the edges. Each path over one top then stood for all the
    # direct sound: 0.6 dB re the free field past two screens and 3.4 dB past three at
    # 63 Hz, where a millimetre higher the level is -9.4 and -9.9 dB. One top keeps the
    # dark side and the others take the lit side: the level is the one just above.
    source, listener = np.array([-10.0, 0.0, 0.0]), np.array([15.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1e-3])
    level = _past_screens(tmp_path, places, source, listener)
    above = _past_screens(tmp_path, places, source + up, listener + up)
    assert np.abs(level - above).max() < (0.15 if len(places) == 2 else 0.7), level - above
    assert -11.5 < level.min() and level.max() < -6.5, level
    # With every band in energy too: a path over one top whose terms cancel there, and
    # is not returned, is still there for the edge fade, which otherwise adds its term
    # again (5.9 dB above the level just above the tops of three screens).
    energy = [
        _past_screens(tmp_path, places, source + h, listener + h, coherent=None) for h in (0.0, up)
    ]
    assert np.abs(energy[0] - energy[1]).max() < (0.15 if len(places) == 2 else 0.7)
    if len(places) == 2:
        # Within the reach of the rays that look for the paths over the tops, a
        # hundredth of a millimetre below their line: as a millimetre below it, where
        # the path over both is all there is (without the sides that follow what those
        # rays found, 5.5 dB lower at 63 Hz and 24 dB at 8 kHz).
        low = [
            _past_screens(tmp_path, places, source - h * up, listener - h * up) for h in (0.01, 1.0)
        ]
        assert np.abs(low[0] - low[1]).max() < 0.25, low[0] - low[1]
    # The same path backwards (within 0.05 dB, on the boundaries of every edge at once),
    # and with paths over one edge alone.
    np.testing.assert_allclose(_past_screens(tmp_path, places, listener, source), level, atol=0.06)
    alone = [
        _past_screens(tmp_path, places, source + h, listener + h, "first_order") for h in (0.0, up)
    ]
    np.testing.assert_allclose(alone[0], alone[1], atol=0.5)


@pytest.mark.parametrize("swap", [False, True])
def test_two_screens_with_one_end_level_with_their_tops(tmp_path: Path, swap: bool) -> None:
    # The source exactly level with two screens' tops, or a millimetre above, and the
    # listener 0.5 m above their line (or the other way round): as with that end a
    # millimetre below the line, where the path over the far top alone is gone and the
    # path over both takes its place. That path was left out with both tops lit, 5 to
    # 5.5 dB low at 63 Hz; a step of up to 1 dB is left there (the way back over both
    # tops off the second screen's face is a path over three edges).
    def level(height: float) -> np.ndarray:
        ends = (-10.0, 0.0, height), (15.0, 0.0, 0.5)
        if swap:
            ends = (-10.0, 0.0, 0.5), (15.0, 0.0, height)
        return _past_screens(tmp_path, (0.0, 10.0), *ends)

    below = level(-1e-3)
    for height in (0.0, 1e-3):
        step = np.abs(level(height) - below)
        assert step.max() < (1.1 if swap else 0.7), step
        assert step[4:].max() < 0.6
    # A tenth of a millimetre below the line, within the reach of the rays that look for
    # the path over the far top: as a millimetre below it (6 dB lower at 63 Hz with the
    # side of the boundary the geometry alone gives).
    assert np.abs(level(-1e-4) - below).max() < 0.15


@pytest.mark.parametrize("gap", [5.0, 0.02])
def test_tops_in_a_row_are_one_wave_in_the_bands_that_add_in_energy(
    tmp_path: Path, gap: float
) -> None:
    # Two flat tops of one height in a row, 5 m wide and 5 m apart, or a 10 m top with
    # a 2 cm slot across it, the source and the listener 1 cm above their plane. At an
    # edge that lies in the plane of the other top, beyond that top, the term of the
    # top's reflection takes off the incident wave's term what it would add: one wave,
    # in the bands that add in energy too. As two waves they read 6.5 to 9.4 dB above
    # the same bands in pressure; without the path over two lit edges, 4.2 to 5.7 dB
    # above the free field.
    width = 5.0 if gap == 5.0 else 5.0 - gap / 2.0
    second = [
        [(x + width + gap, y, z) for x, y, z in quad]
        for quad in edge_diffraction.barrier_quads(width)
    ]
    quads = edge_diffraction.barrier_quads(width) + second
    source, listener = (-10.0, 0.0, 1e-2), (2.0 * width + gap + 5.0, 0.0, 1e-2)
    levels = []
    for coherent in (500, None):
        simulation = {"early": {"diffraction": "second_order", "coherent_max_band_hz": coherent}}
        room = edge_diffraction._room(tmp_path, "tops", quads, source, listener, simulation)
        levels.append(edge_diffraction.whole_early_stage(room))
    pressure, energy = levels
    assert np.abs(energy[:4] - pressure[:4]).max() < 1.2, energy - pressure
    assert -8.0 < min(pressure.min(), energy.min()) and max(pressure.max(), energy.max()) < -5.5


def _screen_from(x: float, y: float) -> list:
    """A screen in the plane x = ``x`` with its top at z = 0, from its end at y = ``y``
    on towards +y."""
    far = edge_diffraction.EXTENT
    return [[(x, y, -far), (x, far, -far), (x, far, 0.0), (x, y, 0.0)]]


@pytest.mark.parametrize("mode", ["first_order", "second_order"])
def test_a_way_beside_the_end_of_a_screen_in_a_row(tmp_path: Path, mode: str) -> None:
    # A long screen and, 10 m behind it, one that ends at y = 0, the source and the
    # listener exactly level with both tops. A centimetre outside that end the second
    # top's path fades out, and takes what is left of its term off the direct sound, as
    # a millimetre above the tops: on the dark side of its boundary it added that to
    # the first top's path, which stands for the direct sound (6.7 to 9.3 dB above the
    # level just above the tops with first_order, 3.7 to 5.8 dB with second_order).
    # With second_order the path over the first top and the top of the second screen's
    # end takes nothing off the first top's wave, which passes level with that top.
    quads = edge_diffraction.screen_quads(0.0) + _screen_from(10.0, 0.0)
    ends = np.array([[-10.0, -0.01, 0.0], [15.0, -0.01, 0.0]])
    level = _past(tmp_path, quads, *ends, mode)
    above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3]), mode)
    # With first_order the terms of the faces at the second screen's end are missing
    # on the line, where the path over that end is not found (1.0 dB at 63 Hz).
    np.testing.assert_allclose(level, above, atol=1.1 if mode == "first_order" else 0.15)
    np.testing.assert_allclose(level[4:], above[4:], atol=0.15)


@pytest.mark.parametrize("mode", ["first_order", "second_order"])
def test_screens_in_a_row_that_end_near_the_way(tmp_path: Path, mode: str) -> None:
    # Two screens that end 15 and 10 cm beside the way, the source and the listener
    # exactly level with their tops and the way inside both. Near its end each top's
    # path is part of the whole edge's: both take their parts off the direct sound, and
    # the one that carries the direct sound carries all of it, not its own part alone.
    # The level is the one a millimetre above the tops (with each top's path standing
    # for all the direct sound it was 4.0 to 7.1 dB above that with second_order).
    quads = _screen_from(0.0, -0.2) + _screen_from(10.0, -0.15)
    ends = np.array([[-10.0, -0.05, 0.0], [15.0, -0.05, 0.0]])
    level = _past(tmp_path, quads, *ends, mode)
    above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3]), mode)
    # With first_order the terms of the faces at the screens' ends are missing on the
    # line (1.0 dB at 63 Hz).
    np.testing.assert_allclose(level, above, atol=1.2 if mode == "first_order" else 0.05)
    np.testing.assert_allclose(level[4:], above[4:], atol=0.05)


@pytest.mark.parametrize("mode", ["first_order", "second_order"])
def test_which_edge_in_a_row_carries_the_direct_sound_changes_nothing(
    tmp_path: Path, mode: str
) -> None:
    # Two screens that both end at y = 0, the way a metre inside them, exactly level
    # with their tops: both tops' paths are near an end, and the edge with the lower
    # index carries the direct sound, in the mirror image of the scene the other
    # screen's. The level is the same, and the same with the source and the listener
    # swapped: the one a millimetre above the tops (5.4 to 18.4 dB below the level that
    # two paths, each standing for all the direct sound, gave with second_order).
    quads = _screen_from(0.0, 0.0) + _screen_from(10.0, 0.0)
    mirrored = [[(-x, y, z) for x, y, z in quad] for quad in quads]
    ends = np.array([[-10.0, 1.0, 0.0], [15.0, 1.0, 0.0]])
    level = _past(tmp_path, quads, *ends, mode)
    np.testing.assert_allclose(
        _past(tmp_path, mirrored, *(ends * [-1.0, 1.0, 1.0]), mode), level, atol=1e-3
    )
    np.testing.assert_allclose(_past(tmp_path, quads, *ends[::-1], mode), level, atol=0.05)
    above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3]), mode)
    # With first_order the terms of the faces at the screens' ends are missing on the
    # line (1.2 dB at 63 Hz).
    np.testing.assert_allclose(level, above, atol=1.4 if mode == "first_order" else 0.2)
    np.testing.assert_allclose(level[4:], above[4:], atol=0.3 if mode == "first_order" else 0.2)


def _upright(start: tuple, end: tuple, top: float = 0.0, top_end: float | None = None) -> list:
    """An upright screen from the plan point ``start`` to ``end``, its top from ``top``
    at the first to ``top_end`` (default ``top``) at the second."""
    far = edge_diffraction.EXTENT
    top_end = top if top_end is None else top_end
    return [[(*start, -far), (*end, -far), (*end, top_end), (*start, top)]]


def test_a_way_through_the_end_of_a_screen_in_a_row(tmp_path: Path) -> None:
    # A long screen and, behind it, an oblique one that ends where the way passes, the
    # source and the listener exactly level with both tops. At its end the oblique
    # top's term is half the whole edge's, and the path over both tops counts by half:
    # the level is the one a millimetre above the tops, between those with the way a
    # millimetre inside the end and outside it (it was that outside the end, 5.5 to
    # 6.0 dB above the level a millimetre above the tops). From 3 mm above the tops the
    # path over both no longer touches the end to within the rounding and counts whole
    # or not at all, as the screen turns: a known issue.
    far = edge_diffraction.EXTENT
    quads = edge_diffraction.screen_quads(0.0) + _upright((10.0, 0.0), (10.0 - far, far))
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends)
    np.testing.assert_allclose(level, _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3])), atol=0.1)
    inside, outside = (_past(tmp_path, quads, *(ends + [0.0, y, 0.0])) for y in (1e-3, -1e-3))
    assert np.all(level < inside - 1.0) and np.all(level > outside + 1.0), (level, inside, outside)


def test_a_way_between_the_ends_of_two_screens_on_either_side(tmp_path: Path) -> None:
    # A screen on one side of the way and, 10 m on, one on the other, both ending at
    # y = 0, the way a millimetre inside the first and outside the second, exactly level
    # with their tops. The path round the first screen's upright end is found too,
    # level with its top: it takes nothing off the direct sound, as just above the
    # line, where the sound passes over the top. The level is the one a millimetre
    # above the tops, each corner taking a quarter of the direct sound (it was 0.34 dB
    # below that at 63 Hz; with the upright end taking its term off the sound, as off
    # one that the screen cuts off there, 12 to 31 dB below it).
    far = edge_diffraction.EXTENT
    quads = _screen_from(0.0, 0.0) + _upright((10.0, -far), (10.0, 0.0))
    ends = np.array([[-10.0, 1e-3, 0.0], [15.0, 1e-3, 0.0]])
    level = _past(tmp_path, quads, *ends)
    np.testing.assert_allclose(level, _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3])), atol=0.1)


@pytest.mark.parametrize("slope", [0.0, 0.137])
@pytest.mark.parametrize("mode", ["first_order", "second_order"])
def test_a_way_through_the_corners_of_two_screens(tmp_path: Path, mode: str, slope: float) -> None:
    # Two screens that both end at y = 0, square to the way or oblique to it, the way
    # through the top corners of both. Each corner is one place (its top and its
    # upright end), and the two are edges in a row: the direct sound passes level with
    # the tops, as just above the line, and the tops' terms stop at the upright ends as
    # they do for it (as for a sound that the screens hide, the level is up to 4.1 dB
    # lower at 63 Hz with first_order). With first_order the level is the one a
    # millimetre clear of both screens, half the direct sound, each corner taking a
    # quarter (it was 8.2 to 9.5 dB above that); with second_order the path over both
    # tops counts by half at each end, between the levels a millimetre clear of the
    # screens and a millimetre inside them. A path touches the end of an oblique top to
    # within the rounding: it counts as one that touches the end, and the scene and its
    # mirror image give one level (without that, 2.2 dB apart with second_order, as the
    # rounding fell; 0.7 dB at 63 Hz before the change).
    far = edge_diffraction.EXTENT
    quads = _upright((0.0, 0.0), (-slope * far, far)) + _upright(
        (10.0, 0.0), (10.0 + slope * far, far)
    )
    mirrored = [[(x, -y, z) for x, y, z in quad] for quad in quads]
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends, mode)
    np.testing.assert_allclose(_past(tmp_path, mirrored, *ends, mode), level, atol=0.02)
    clear = _past(tmp_path, quads, *(ends + [0.0, -1e-3, 1e-3]), mode)
    if mode == "first_order":
        np.testing.assert_allclose(level, clear, atol=0.1)
        np.testing.assert_allclose(level[4:], 20.0 * math.log10(0.5), atol=0.05)
    else:
        inside = _past(tmp_path, quads, *(ends + [0.0, 1e-3, 0.0]), mode)
        assert np.all(level > clear + 0.2) and np.all(level < inside - 1.0), (level, clear, inside)


def test_a_way_that_something_else_hides_beside_the_ends_of_two_screens(tmp_path: Path) -> None:
    # Two screens that end 5 cm beside the way, the source and the listener exactly
    # level with their tops, and a board 2 cm wide across the way between them. The
    # tops' paths fade out past their ends, on the way, but the way touches neither
    # top: it is the board that hides the direct sound, whose edges' paths stand for
    # it. The level is the one a millimetre above the tops (taking the direct sound to
    # be there as well, for the tops, gave it twice: 8 to 9 dB higher).
    quads = _screen_from(0.0, 0.05) + _screen_from(10.0, 0.05)
    quads += [[(5.0, -0.01, -0.5), (5.0, 0.01, -0.5), (5.0, 0.01, 0.5), (5.0, -0.01, 0.5)]]
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends, "first_order")
    above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3]), "first_order")
    np.testing.assert_allclose(level, above, atol=0.1)


def test_two_screens_in_a_row_and_a_wall_beside_the_way(tmp_path: Path) -> None:
    # Two screens that end at a wall 2 m beside the way, the source and the listener
    # exactly level with the tops. The reflection off the wall passes over both tops as
    # the direct sound does, with the wall between them: its paths over one top reflect
    # after the first top and before the second, and the terms that make reflections
    # fade at edges take the product of the two tops' terms for it. The rule for edges
    # in a row leaves that way to them (taking it as well gave the reflection twice, up
    # to 13.6 dB above the level a millimetre higher) and takes the direct way: the
    # level is the one a millimetre above the tops (it was 9.3 to 11.3 dB above that).
    far = edge_diffraction.EXTENT
    quads = _upright((0.0, -far), (0.0, 2.0)) + _upright((10.0, -far), (10.0, 2.0))
    quads += [[(-50.0, 2.0, -far), (50.0, 2.0, -far), (50.0, 2.0, 20.0), (-50.0, 2.0, 20.0)]]
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends)
    np.testing.assert_allclose(level, _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3])), atol=0.15)


def test_a_bent_top_in_a_row_and_its_mirror_image(tmp_path: Path) -> None:
    # A screen whose top is level up to y = 0 and rises from there, a long screen 10 m
    # behind it, and the way through the joint, exactly level with the tops. The paths
    # over the two pieces of the bent top touch them at their common end: each counts
    # by half, the paths over one edge and those over two, at whichever end of a piece
    # the joint is. The scene and its mirror image give one level, within 1.0 dB of
    # that a millimetre above the tops (it was 10.8 to 13.6 dB above that), where the
    # path over the rising piece and the far screen counts whole.
    far = edge_diffraction.EXTENT
    quads = (
        _upright((0.0, -far), (0.0, 0.0))
        + _upright((0.0, 0.0), (0.0, 1.0), 0.0, 1.0)
        + _upright((0.0, 1.0), (0.0, far), 1.0)
        + edge_diffraction.screen_quads(10.0)
    )
    mirrored = [[(x, -y, z) for x, y, z in quad] for quad in quads]
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends)
    np.testing.assert_allclose(_past(tmp_path, mirrored, *ends), level, atol=1e-3)
    np.testing.assert_allclose(level, _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3])), atol=1.1)


def test_three_screens_in_a_row_the_middle_one_ending_beside_the_way(tmp_path: Path) -> None:
    # Three screens 5 m apart, the middle one ending a centimetre beside the way, the
    # source and the listener exactly level with the tops: as a millimetre above them
    # (it was 13.0 to 16.4 dB above that).
    quads = (
        edge_diffraction.screen_quads(0.0)
        + _screen_from(5.0, 0.0)
        + edge_diffraction.screen_quads(10.0)
    )
    ends = np.array([[-10.0, -0.01, 0.0], [15.0, -0.01, 0.0]])
    for coherent in (500, None):
        level = _past(tmp_path, quads, *ends, coherent=coherent)
        above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 1e-3]), coherent=coherent)
        np.testing.assert_allclose(level, above, atol=0.5)


def test_a_way_through_the_corner_of_a_screen(tmp_path: Path) -> None:
    # The quarter of the plane x = 0 below z = 0 and y = 0, the straight way from the
    # source to the listener through its corner: its top and its end meet where the way
    # passes and are one place, not two edges in a row (8 to 10 dB lower as two). The
    # level is the one a millimetre clear of both, three quarters of the direct sound.
    far = edge_diffraction.EXTENT
    quads = [[(0.0, -far, -far), (0.0, 0.0, -far), (0.0, 0.0, 0.0), (0.0, -far, 0.0)]]
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends)
    np.testing.assert_allclose(
        level, _past(tmp_path, quads, *(ends + [0.0, 1e-3, 1e-3])), atol=0.05
    )
    np.testing.assert_allclose(level[4:], 20.0 * math.log10(0.75), atol=0.05)


def test_a_roof_and_then_a_screen_level_with_it_the_same_both_ways_round(tmp_path: Path) -> None:
    # A top 5 m wide and, 5 m past it, a screen whose top lies in the top's plane, the
    # source and the listener a centimetre above it. The path over the top's first
    # corner and the screen leaves the corner along the top, which ends before the
    # screen; backwards it reaches the corner along the top, at grazing incidence. Half
    # the coefficient both ways: with it only on the way back, swapping the source and
    # the listener changed the level by 4.3 to 4.9 dB.
    quads = edge_diffraction.barrier_quads(5.0) + edge_diffraction.screen_quads(10.0)
    source, listener = (-10.0, 0.0, 1e-2), (15.0, 0.0, 1e-2)
    for coherent in (500, None):
        forward = _past(tmp_path, quads, source, listener, coherent=coherent)
        back = _past(tmp_path, quads, listener, source, coherent=coherent)
        np.testing.assert_allclose(back, forward, atol=0.15)


@pytest.mark.parametrize(("mode", "within"), [("second_order", 0.5), ("first_order", 0.5)])
@pytest.mark.parametrize("beside", [0.0, 0.01, 0.5])
@pytest.mark.parametrize("top", [0.0, 1e-12])
def test_a_path_over_a_screens_top_does_not_reflect_off_a_roof_level_with_it_elsewhere(
    tmp_path: Path, mode, within, beside, top
) -> None:
    # A roof from x = 0 to 5 m, a screen at x = 7 m level with it that ends ``beside``
    # the way (y >= 0, the way at y = beside) and a roof from 9 to 19 m, the source and
    # the listener 1 or 2 mm above their plane. A path over the screen's top that
    # reflects off that plane next to it touches the plane at the top itself, where no
    # roof is. The ray that looks for the path starts 0.6 mm off the top (a millionth of
    # the model's size) and met a roof metres away. With that path the level up to
    # 500 Hz was 4.2 to 4.6 dB above that a centimetre above the plane with second_order
    # (through the screen's end 3.2 to 7.8 dB, the two ways round 4.4 dB apart) and 4.4
    # to 5.4 dB below it with first_order (1.9 to 9.5 dB, 7.3 dB apart). Without it the
    # two ways round agree, and with the level a centimetre above within 0.23 dB
    # (through the end 0.39 dB; first_order 0.37 dB). So too with the screen's ``top``
    # a millionth of a micrometre off the plane, as two heights that are sums of other
    # numbers differ.
    extent = edge_diffraction.EXTENT
    screen = [[(7.0, 0.0, -extent), (7.0, extent, -extent), (7.0, extent, top), (7.0, 0.0, top)]]
    far = [[(x + 9.0, y, z) for x, y, z in quad] for quad in edge_diffraction.barrier_quads(10.0)]
    quads = edge_diffraction.barrier_quads(5.0) + screen + far
    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500}}
    levels = {}
    for height in (1e-3, 2e-3, 1e-2):
        source, listener = (-10.0, beside, height), (25.0, beside, height)
        for way, (a, b) in enumerate(((source, listener), (listener, source))):
            room = edge_diffraction._room(tmp_path, "level", quads, a, b, simulation)
            levels[height, way] = edge_diffraction.whole_early_stage(room)
    for height in (1e-3, 2e-3):
        np.testing.assert_allclose(levels[height, 1], levels[height, 0], atol=0.01)
        for way in (0, 1):
            np.testing.assert_allclose(levels[height, way], levels[1e-2, way], atol=within)


@pytest.mark.parametrize(
    ("mode", "method", "top"),
    [
        ("second_order", "image_sources", 1e-6),
        ("second_order", "image_sources", -1e-6),
        ("first_order", "ray_tracing", 1e-6),
        ("first_order", "ray_tracing", -1e-6),
    ],
)
def test_a_screens_top_a_micrometre_off_a_roofs_plane_does_not_reflect_off_the_roof_either(
    tmp_path: Path, mode, method, top
) -> None:
    # A roof and a screen 0.1 m past it whose top is a micrometre above the roof's plane
    # or below it, as a model drawn elsewhere has it, the source and the listener a
    # centimetre above the plane. The path over the screen's top reflects where its own
    # line to the image meets the plane: 2 mm from the top one way, where no roof is,
    # and nowhere with the top below the plane. The ray that looks for it starts 0.6 mm
    # above the top and met the roof. With that path the level up to 500 Hz was 4.8 to
    # 6.5 dB over that of a level top with the top a micrometre above and second_order,
    # and 6.4 to 6.8 dB over it with the top a micrometre below and first_order; with
    # second_order the two ways round were 1.5 to 1.7 dB apart there.
    source, listener = (-10.0, 0.0, 1e-2), (15.0, 0.0, 1e-2)
    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500, "method": method}}
    levels = {}
    for z in (0.0, top):
        quads = edge_diffraction.barrier_quads(9.9) + edge_diffraction.screen_quads(10.0, z)
        for way, (a, b) in enumerate(((source, listener), (listener, source))):
            room = edge_diffraction._room(tmp_path, "nearly", quads, a, b, simulation)
            levels[z, way] = edge_diffraction.whole_early_stage(room)
    np.testing.assert_allclose(levels[top, 1][:4], levels[top, 0][:4], atol=0.05)
    if mode == "first_order" or top > 0.0:
        # Below the plane the level with second_order is not that of a level top (a
        # known issue of docs/validation.md).
        for way in (0, 1):
            np.testing.assert_allclose(levels[top, way][:4], levels[0.0, way][:4], atol=0.25)


def test_a_path_over_a_screens_top_reflects_off_a_roof_its_own_line_meets(tmp_path: Path) -> None:
    # The same with the screen's top a tenth of a millimetre above the roof's plane,
    # closer to it than the ray starts from the top: the path's own line meets the plane
    # 0.2 m from the screen, on the roof, and the path counts as with the top a
    # millimetre above, where the ray alone decides. Up to 500 Hz the two levels are
    # within 0.9 dB (without the path, 4.9 to 7.5 dB apart).
    source, listener = (-10.0, 0.0, 1e-2), (15.0, 0.0, 1e-2)
    simulation = {"early": {"diffraction": "second_order", "coherent_max_band_hz": 500}}
    levels = []
    for top in (1e-4, 1e-3):
        quads = edge_diffraction.barrier_quads(9.9) + edge_diffraction.screen_quads(10.0, top)
        room = edge_diffraction._room(tmp_path, "over", quads, source, listener, simulation)
        levels.append(edge_diffraction.whole_early_stage(room))
    np.testing.assert_allclose(levels[0][:4], levels[1][:4], atol=1.2)


@pytest.mark.parametrize(("mode", "within"), [("second_order", 0.85), ("first_order", 1.6)])
@pytest.mark.parametrize("beside", [0.01, 0.1])
@pytest.mark.parametrize("screen_first", [True, False])
def test_past_the_end_of_a_screen_level_with_a_roof_as_a_centimetre_above(
    tmp_path: Path, mode, within, beside, screen_first
) -> None:
    # A screen that ends ``beside`` the way (y >= 0, the way at y = beside) and a roof
    # 5 m wide level with its top, 5 m before the roof or 5 m past it, the source and the
    # listener a millimetre above their plane. With the path over the screen's top that
    # reflects off the roof's plane, which it touches at the top itself
    # (test_a_path_over_a_screens_top_does_not_reflect_off_a_roof_level_with_it_elsewhere),
    # the level up to 500 Hz was 3.2 to 3.4 dB above that a centimetre above the plane
    # with second_order; with first_order, whose single terms nearly cancel there, it
    # was 1.3 to 8.6 dB off. The two ways round agree, and with the level a
    # centimetre above within 0.64 dB (first_order 0.82 dB up to 500 Hz and 1.23 dB
    # above). With second_order it is 0.19 to 0.64 dB below it up to 500 Hz: the term of
    # the screen's upright end for the direct sound comes in from 1.2 mm above the
    # plane (the screen first) or 1.8 mm (the roof first), as the ray that tests its
    # path starts 0.6 mm inside the end, below the roof's plane (a known issue of
    # docs/validation.md).
    extent = edge_diffraction.EXTENT
    x = -5.0 if screen_first else 10.0
    screen = [[(x, 0.0, -extent), (x, extent, -extent), (x, extent, 0.0), (x, 0.0, 0.0)]]
    quads = screen + edge_diffraction.barrier_quads(5.0)
    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500}}
    levels = {}
    start = -15.0 if screen_first else -10.0
    for height in (1e-3, 1e-2):
        source, listener = (start, beside, height), (start + 25.0, beside, height)
        for way, (a, b) in enumerate(((source, listener), (listener, source))):
            room = edge_diffraction._room(tmp_path, "end", quads, a, b, simulation)
            levels[height, way] = edge_diffraction.whole_early_stage(room)
    np.testing.assert_allclose(levels[1e-3, 1], levels[1e-3, 0], atol=0.05)
    for way in (0, 1):
        np.testing.assert_allclose(levels[1e-3, way], levels[1e-2, way], atol=within)
    if mode == "second_order" and beside == 0.01 and screen_first:
        # Through the screen's end the field is half the sum of the fields without the
        # screen and with an endless one (Kirchhoff's approximation), each from a wave
        # calculation in two dimensions turned into a point source's: a centimetre above
        # the plane, -6.2 to -6.8 dB from 63 to 500 Hz. The engine, a centimetre inside
        # the end, is 0.5 to 1.2 dB above that.
        reference = [-6.24, -6.47, -6.70, -6.84]
        np.testing.assert_allclose(levels[1e-2, 0][:4], reference, atol=1.5)


@pytest.mark.parametrize("beside", [0.01, 0.1])
def test_past_the_end_of_a_screen_level_with_a_roof_with_every_reflection_tested(
    tmp_path: Path, beside
) -> None:
    # The same past a roof 5 m wide and then the screen, with every sequence of
    # reflections tested (method: image_sources): a millimetre above the plane the two
    # ways round agree, and with the level a centimetre above within 0.62 dB (with the
    # path that reflects off the roof's plane at the screen's top, 2.4 to 3.0 dB above
    # it).
    extent = edge_diffraction.EXTENT
    x = 10.0
    screen = [[(x, 0.0, -extent), (x, extent, -extent), (x, extent, 0.0), (x, 0.0, 0.0)]]
    quads = screen + edge_diffraction.barrier_quads(5.0)
    early = {"diffraction": "second_order", "coherent_max_band_hz": 500, "method": "image_sources"}
    levels = {}
    for height in (1e-3, 1e-2):
        source, listener = (-10.0, beside, height), (15.0, beside, height)
        for way, (a, b) in enumerate(((source, listener), (listener, source))):
            room = edge_diffraction._room(tmp_path, "end", quads, a, b, {"early": early})
            levels[height, way] = edge_diffraction.whole_early_stage(room)
    for height in (1e-3, 1e-2):
        np.testing.assert_allclose(levels[height, 1], levels[height, 0], atol=0.05)
    np.testing.assert_allclose(levels[1e-3, 0], levels[1e-2, 0], atol=0.85)


@pytest.mark.parametrize("mode", ["second_order", "first_order"])
@pytest.mark.parametrize(
    ("height", "gap", "method"),
    [(1e-3, 5.0, "ray_tracing"), (1e-2, 0.1, "ray_tracing"), (1e-2, 1.0, "image_sources")],
)
def test_a_roof_and_then_a_screen_level_with_it_with_the_ends_swapped(
    tmp_path: Path, mode, height, gap, method
) -> None:
    # A flat top and a screen ``gap`` past it whose top lies in the top's plane, the
    # source and the listener a millimetre or a centimetre above it. With the path over
    # the screen's top that reflects off the top's plane, which it touches at the
    # screen's top itself
    # (test_a_path_over_a_screens_top_does_not_reflect_off_a_roof_level_with_it_elsewhere),
    # and the screen's term on the leg after the reflection wherever the screen stood
    # (test_a_screen_level_with_a_roof_belongs_to_the_leg_it_stands_beside), swapping
    # the source and the listener changed the level by 5.4 to 7.9 dB (6.5 to 7.4 dB
    # with first_order); without either of the two it does not. With the rays' own
    # search for reflections, the two ways round can still find different reflections
    # between the roof's wall and the screen (a known issue), so the case where that
    # reflection counts takes every sequence.
    quads = edge_diffraction.barrier_quads(10.0 - gap) + edge_diffraction.screen_quads(10.0)
    source, listener = (-10.0, 0.0, height), (15.0, 0.0, height)
    simulation = {"early": {"diffraction": mode, "coherent_max_band_hz": 500, "method": method}}
    levels = []
    for a, b in ((source, listener), (listener, source)):
        room = edge_diffraction._room(tmp_path, "past", quads, a, b, simulation)
        levels.append(edge_diffraction.whole_early_stage(room))
    np.testing.assert_allclose(levels[1], levels[0], atol=0.05)
    if mode == "first_order" and (height, gap) == (1e-2, 1.0):
        # A wave calculation in two dimensions turned into a point source's gives -6.3
        # to -6.9 dB from 63 to 500 Hz there. first_order is 0.2 to 2.0 dB below it; with
        # both, and the source past the screen, it was 4.4 to 6.1 dB above.
        reference = [-6.26, -6.50, -6.74, -6.93]
        for level in levels:
            np.testing.assert_allclose(level[:4], reference, atol=2.5)


@pytest.mark.parametrize("mode", ["second_order", "first_order"])
@pytest.mark.parametrize(("height", "gap"), [(0.1, 0.1), (1.0, 0.1), (1.0, 1.0)])
def test_a_screen_level_with_a_roof_belongs_to_the_leg_it_stands_beside(
    tmp_path: Path, mode, height, gap
) -> None:
    # A flat top and a screen ``gap`` past it whose top lies in the top's plane, with
    # every sequence of reflections tested. The reflection off the top passes the
    # screen's top, an edge in the very plane it reflects off: the screen's term
    # belongs to the leg the screen stands beside, the one after the point of
    # reflection one way round and the one before it the other. Taken for the leg
    # after the plane wherever the edge stood, the term was missing one way round:
    # swapping the source and the listener changed the level by 0.35 to 0.38 dB a
    # tenth of a metre above the plane and by 0.8 to 1.1 dB a metre above it.
    source, listener = (-10.0, 0.0, height), (15.0, 0.0, height)
    simulation = {
        "early": {"diffraction": mode, "coherent_max_band_hz": 500, "method": "image_sources"}
    }
    levels = {}
    for top in (0.0, 1e-4):
        quads = edge_diffraction.barrier_quads(10.0 - gap)
        quads += edge_diffraction.screen_quads(10.0, top)
        for way, (a, b) in enumerate(((source, listener), (listener, source))):
            room = edge_diffraction._room(tmp_path, "past", quads, a, b, simulation)
            levels[top, way] = edge_diffraction.whole_early_stage(room)
    np.testing.assert_allclose(levels[0.0, 1], levels[0.0, 0], atol=0.05)
    if height == 1.0:
        # Which leg: a top a tenth of a millimetre above the plane is on the leg it
        # stands beside by its height, both ways round, and a level top is its limit:
        # up to 500 Hz within 0.18 dB (on the other leg, 0.5 to 1.0 dB off up to
        # 250 Hz). Lower down the level is not even with the top's height (a known
        # issue).
        for way in (0, 1):
            np.testing.assert_allclose(levels[0.0, way][:4], levels[1e-4, way][:4], atol=0.25)


@pytest.mark.parametrize("mode", ["second_order", "first_order"])
def test_a_screens_top_within_rounding_of_a_roofs_plane_is_level_with_it(
    tmp_path: Path, mode
) -> None:
    # The screen's top a millionth of a micrometre below the plane of the top 1 m before
    # it, as two heights that are sums of other numbers differ: it is an edge in that
    # plane all the same, on the leg it stands beside, and the level a metre above is
    # that with the top exactly level, both ways round. Taken for an edge beyond the
    # plane, on no leg of the reflection, the level up to 500 Hz was 0.8 dB off with
    # second_order and 1.0 dB with first_order.
    source, listener = (-10.0, 0.0, 1.0), (15.0, 0.0, 1.0)
    simulation = {
        "early": {"diffraction": mode, "coherent_max_band_hz": 500, "method": "image_sources"}
    }
    levels = []
    for top in (0.0, -1e-12):
        quads = edge_diffraction.barrier_quads(9.0) + edge_diffraction.screen_quads(10.0, top)
        for a, b in ((source, listener), (listener, source)):
            room = edge_diffraction._room(tmp_path, "nearly", quads, a, b, simulation)
            levels.append(edge_diffraction.whole_early_stage(room))
    np.testing.assert_allclose(levels[1:], np.tile(levels[0], (3, 1)), atol=0.02)


def test_two_flat_tops_in_a_row_exactly_in_their_plane_with_first_order(tmp_path: Path) -> None:
    # Two flat tops of one height, 2 and 3 m wide and 6 m apart, the source and the
    # listener exactly in their plane. The path over the far edge of each top is found
    # and the direct sound is not; at those edges the top's reflection is on its
    # boundary as the incident wave is. The rule for edges in a row moves the incident
    # wave's term to the lit side and leaves the reflection's where the row that
    # carries the direct sound has it (with both moved, the two rows cancelled: 16 to
    # 41 dB below the level 3 mm above the plane, which is beyond the reach of the rays
    # that look for paths over edges). Each path stood for all the direct sound before:
    # 3.3 to 8.1 dB above that level.
    second = [[(x + 8.0, y, z) for x, y, z in quad] for quad in edge_diffraction.barrier_quads(3.0)]
    quads = edge_diffraction.barrier_quads(2.0) + second
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends, "first_order")
    above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 3e-3]), "first_order")
    assert np.abs(level - above).max() < 2.6, level - above
    assert np.abs(level - above)[4:].max() < 0.4, level - above


@pytest.mark.parametrize("coherent", [500, None])
def test_a_flat_top_and_then_a_screen_exactly_in_its_plane(tmp_path: Path, coherent) -> None:
    # A flat top 3 m wide and, 5 m past it, a screen whose top lies in the top's plane,
    # the source and the listener exactly in that plane. At the top's far edge the
    # top's reflection is on its boundary with the incident wave: one wave, in the
    # bands that add in energy too (as two waves they read 10 to 11 dB above the level
    # 3 mm above the plane there). The level is that 3 mm above the plane within 1.5 dB
    # (with the source and the listener swapped it is 2.1 to 5.4 dB lower: a known
    # issue).
    quads = edge_diffraction.barrier_quads(3.0) + edge_diffraction.screen_quads(8.0)
    ends = np.array([[-10.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
    level = _past(tmp_path, quads, *ends, coherent=coherent)
    above = _past(tmp_path, quads, *(ends + [0.0, 0.0, 3e-3]), coherent=coherent)
    np.testing.assert_allclose(level, above, atol=1.7)


def test_a_map_tests_the_faces_of_diffracting_edges_too(tmp_path: Path, monkeypatch) -> None:
    # A map takes its candidate reflections as a listener's simulation does: with the
    # planes of the faces at diffracting edges that no ray finds.
    from acoustic_spaces.engine import coverage, coverage_map
    from acoustic_spaces.scene import Coverage

    quads = edge_diffraction.barrier_quads(10.0)
    simulation = {"early": {"diffraction": "second_order"}, "late": {"rays": 50}}
    room = edge_diffraction._room(
        tmp_path, "map", quads, (-10.0, 0.0, 1e-3), (15.0, 0.0, 1.0), simulation
    )
    top = int(TraceModel.from_room(room).face_plane[0])
    seen = []
    plain = coverage._candidates

    def recorded(*args):
        found = plain(*args)
        seen.append(found[0][:, 0])
        return found

    monkeypatch.setattr(coverage, "_candidates", recorded)
    coverage_map(room, settings=Coverage(spacing=100.0, surfaces=["surfaces"]), threads=1)
    assert seen and all(top in planes for planes in seen)


def _over_grass(source, listener, barrier, zones=False) -> tuple[np.ndarray, int]:
    """The level of the early paths (dB) from ``source`` to ``listener`` past a screen
    (path, height) on grass, outdoors, with a track bed of gravel under the source
    (``zones``), and how many specular paths there are."""
    ground = {"extent": [[-200, -200], [200, 200]], "sky": 40}
    if zones:
        ground["zones"] = [{"name": "bed", "polygon": [[-200, -2], [200, -2], [200, 2], [-200, 2]]}]
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": ground,
                    "barriers": [{"name": "screen", "path": barrier[0], "height": barrier[1]}],
                },
                "materials": {
                    "grass": {"impedance": "grass"},
                    "gravel": {"impedance": "gravel"},
                    "panel": {"absorption": [0.05] * 8},
                },
                "assignments": {"ground": "grass", "bed": "gravel", "barriers": "panel"},
                "sources": [{"name": "s", "position": list(source)}],
                "listeners": [{"name": "l", "position": list(listener)}],
                "simulation": {
                    "early": {
                        "method": "image_sources",
                        "max_order": 1,
                        "diffraction": "first_order",
                    },
                    "ir_length_s": 0.5,
                },
            }
        )
    )
    paths = early_paths(room)
    return 10 * np.log10(total_energy(paths)), int(np.sum(paths.edge < 0))


@pytest.mark.parametrize("case", ["end", "top"])
def test_a_ground_reflection_joins_its_edge_term_on_grass(case: str) -> None:
    # Where a screen's end (from a source over a track bed, as in the railway example) or
    # its top (the reflection before the screen) cuts off the reflection off the grass,
    # the term on the dark side takes over from the reflection and its term on the lit
    # side. Their ground factors were those of the leg from the edge, the reflection's
    # that of the whole path, over another stretch of ground (1.2 dB at 250 Hz at the
    # end, 0.3 dB at 63 Hz at the top).
    step = 1e-3
    if case == "end":
        end = -100.0 - 100.0 * 4.0 / 21.0  # the end's shadow boundary on the track
        screen = ([[-100, -4], [100, -4]], 3.0)
        sides = [
            _over_grass((end + d, 0.0, 2.2), (0.0, -25.0, 1.5), screen, True) for d in (step, -step)
        ]
    else:
        top = -1.0 + 4.0 * 30.0 / 10.0  # the line from the source's image over the top
        screen = ([[10, -150], [10, 150]], 3.0)
        sides = [_over_grass((0.0, 0.0, 1.0), (30.0, 0.0, top + d), screen) for d in (-step, step)]
    (dark, hidden), (lit, seen) = sides
    assert seen > hidden  # the reflection off the grass, and at the end the direct sound
    assert np.abs(lit - dark).max() < 0.02


# -- reflections on both sides of an edge ----------------------------------------------


def _ground_barrier(
    both_sides: bool,
    listener=(20.0, 0.0, 1.5),
    absorption: float = 0.0,
    source=(0.0, 0.0, 1.0),
    path=((10, -190), (10, 190)),
) -> Room:
    """A thin barrier 3 m high along ``path``, 10 m from a source 1 m over a ground of
    this absorption."""
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": {"extent": [[-100, -200], [200, 200]], "sky": 40},
                    "barriers": [
                        {"name": "barrier", "path": [list(p) for p in path], "height": 3.0}
                    ],
                },
                "materials": {
                    "ground": {"absorption": [absorption] * 8, "scattering": 0.0},
                    "rigid": {"absorption": [0.0] * 8, "scattering": 0.0},
                },
                "default_material": "ground",
                "assignments": {"barrier": "rigid"},
                "sources": [{"name": "s", "position": list(source)}],
                "listeners": [{"name": "l", "position": list(listener)}],
                "simulation": {
                    "ir_length_s": 0.3,
                    "seed": 1,
                    "late": {"rays": 300},
                    "early": {
                        "method": "image_sources",
                        "max_order": 2,
                        "diffraction": "first_order",
                        "diffraction_reflections": "both_sides" if both_sides else "one_side",
                    },
                },
            }
        )
    )


def _diffracted(room: Room, both_sides: bool, mode: str = "first_order"):
    """The paths of a room's one listener with the waves' terms in energy in every band
    (``coherent=0``), as :func:`_over_barrier` finds them."""
    model = TraceModel.from_room(room)
    c, air = room.speed_of_sound(), room.air_attenuation("octave")
    source = np.array(room.scene.sources[0].position)
    at = np.array(room.scene.listeners[0].position)
    diffraction = EdgeDiffraction(
        model, source, speed_of_sound=c, air=air, mode=mode, both_sides=both_sides
    )
    paths = early_stage.find_paths(model, source, at, [], speed_of_sound=c, air=air, max_delay_s=1)
    return diffraction.apply(at, paths, 1.0)


def test_a_barrier_on_the_ground_gets_its_fourth_path(tmp_path: Path) -> None:
    """One side: over the top, and over it with a reflection off the ground before or
    after. Both sides: also with a reflection before and after, a path of order 2 that
    the mirrored geometry, a screen in free space, has as its direct sound over the
    top."""
    one = early_paths(_ground_barrier(False))
    assert one.diffracted.all() and one.order.tolist() == [0, 1, 1]
    both = early_paths(_ground_barrier(True))
    assert both.diffracted.all() and both.order.tolist() == [0, 1, 1, 2]
    np.testing.assert_array_equal(both.energy[:3], one.energy)
    fourth = len(both) - 1
    assert len(both.points[fourth]) == 5 and len(both.faces[fourth]) == 2
    bounces = both.points[fourth][[1, 3]]
    assert bounces[:, 2] == pytest.approx(0.0) and 0 < bounces[0, 0] < 10 < bounces[1, 0] < 20
    assert both.delay_s[fourth] > both.delay_s.max(initial=0.0, where=np.arange(4) < 3)
    # The same wave as the direct sound over a screen whose top is 4 m over the mirrored
    # source and listener, 1 and 1.5 m below the ground: the terms in energy in both.
    screen = _screen(tmp_path, (-10.0, 0.0, -4.0), (10.0, 0.0, -4.5))
    mirrored = _diffracted(screen, False, "direct")
    assert len(mirrored) == 1 and mirrored.order[0] == 0
    exact = _diffracted(_ground_barrier(True), True)
    assert exact.order.tolist() == [0, 1, 1, 2]
    np.testing.assert_allclose(exact.energy[3], mirrored.energy[0], rtol=1e-9)
    assert exact.distance_m[3] == pytest.approx(mirrored.distance_m[0], abs=1e-9)
    # Over an absorbing ground the reflections carry nothing, so the fourth path is
    # found and dropped, and the level is that of the direct sound over the top.
    absorbing = early_paths(_ground_barrier(True, absorption=1.0))
    assert absorbing.order.tolist() == [0]


@pytest.mark.parametrize("both_sides", [False, True])
def test_no_step_where_the_reflection_off_the_ground_comes_over_the_barrier(
    both_sides: bool,
) -> None:
    """10 m behind the barrier, where the listener rises into sight of the reflection
    off the ground before it (7 m up): the level changes by 0.04 dB or less in 1 cm.
    The term of the path over the top with the reflection after it comes from the same
    image of the source and took that reflection for its own geometric wave, so it was
    left out above, and the level stepped by 0.17 dB (one_side) and 0.20 dB (both_sides)
    at 125 Hz."""
    levels = []
    for z in (6.995, 7.005):
        paths = early_paths(_ground_barrier(both_sides, listener=(20.0, 0.0, z)))
        levels.append(10 * np.log10(total_energy(paths)))
    assert np.abs(levels[1] - levels[0]).max() < 0.06


def test_both_sides_is_off_by_default_and_a_setting() -> None:
    assert EarlyStage().diffraction_reflections == "one_side"
    with pytest.raises(ValueError):
        EarlyStage(diffraction_reflections="all")
    # The simulation takes it from the scene: the level behind the barrier rises.
    one = simulate(_ground_barrier(False))[("s", "l")]
    both = simulate(_ground_barrier(True))[("s", "l")]
    assert one.early.order.max() == 1 and both.early.order.max() == 2
    assert both.early.energy.sum() > one.early.energy.sum()
    assert (both.relative_to_free_field()[4:] > one.relative_to_free_field()[4:]).all()


def test_both_sides_in_a_room(monkeypatch) -> None:
    """Behind the corner of the L-shaped room the floor and ceiling reflections also
    arrive over the edge with a reflection on each side (floor then ceiling, and the
    other way round); the paths that reflect on one side are unchanged. (The edge
    diffraction's own paths: the edge fade's would count among them.)"""
    _edge_diffraction_alone(monkeypatch)
    one = early_paths(_l_room([[2.0, 8.0, 1.5]]), max_order=2, diffraction="first_order")
    room = _l_room([[2.0, 8.0, 1.5]], early={"diffraction_reflections": "both_sides"})
    both = early_paths(room, max_order=2, diffraction="first_order")
    second = both.diffracted & (both.order == 2)
    assert second.sum() >= 2
    for k in np.flatnonzero(second):
        assert len(both.points[k]) == 5 and len(both.faces[k]) == 2
        assert both.faces[k][0] != both.faces[k][1]  # floor and ceiling, not one plane twice
    kept = both[~second] if hasattr(both, "__getitem__") else None
    if kept is not None:
        assert len(kept) == len(one)
        np.testing.assert_allclose(np.sort(kept.delay_s), np.sort(one.delay_s))


# -- many buildings: only the reflections that can happen are tested --------------------


def _town(seed: int = 7, both_sides: bool = True) -> Room:
    """Nine buildings at odd angles on a grass ground, two sources and two listeners
    among them at different heights."""
    rng = np.random.default_rng(seed)
    buildings = []
    for i in range(3):
        for j in range(3):
            centre = np.array([i * 30.0, j * 30.0]) + rng.uniform(-4.0, 4.0, 2)
            w, d, a = rng.uniform(8.0, 18.0), rng.uniform(8.0, 18.0), rng.uniform(0.0, np.pi)
            corners = np.array([[-w, -d], [w, -d], [w, d], [-w, d]]) / 2.0
            turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            buildings.append(
                {
                    "name": f"b{i}{j}",
                    "footprint": (corners @ turn.T + centre).tolist(),
                    "height": float(rng.uniform(5.0, 25.0)),
                }
            )
    early = {"diffraction": "first_order"}
    if both_sides:
        early["diffraction_reflections"] = "both_sides"
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"ground": {"margin": 15.0}, "buildings": buildings},
                "materials": {
                    "lawn": {"impedance": "grass"},
                    "wall": {"absorption": [0.05] * 8},
                },
                "assignments": {"ground": "lawn", "buildings": "wall"},
                "sources": [
                    {"name": "s1", "position": [15.0, 15.0, 1.0]},
                    {"name": "s2", "position": [44.0, 16.0, 30.0]},
                ],
                "listeners": [
                    {"name": "l1", "position": [-8.0, 45.0, 1.5]},
                    {"name": "l2", "position": [46.0, 47.0, 12.0]},
                ],
                "simulation": {"seed": 1, "ir_length_s": 0.6, "early": early},
            }
        )
    )


def _all_reflecting(model, edges, bounds, end, planes, among, offset, threads=0):
    rows, cols = np.nonzero(np.ones((len(among), len(planes)), dtype=bool))
    return rows.astype(np.int64), cols.astype(np.int64)


def _all_reaching(model, edges, bounds, end, planes, among, sides, *rest):
    rows, cols = np.nonzero(np.ones((len(sides[0]), len(planes)), dtype=bool))
    return rows.astype(np.int64), cols.astype(np.int64)


def _same_paths(a, b) -> None:
    import dataclasses

    for field in dataclasses.fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            assert x is not None and y is not None, field.name
            assert x.shape == y.shape and np.array_equal(x, y, equal_nan=True), field.name


@pytest.mark.parametrize("both_sides", [False, True])
def test_culled_pairs_leave_the_paths_bit_identical(monkeypatch, both_sides: bool) -> None:
    """Testing only the pairs of an edge and a plane that a reflection between them can
    take changes nothing: every path, its energy and its spectrum come out bit for bit
    as when every pair goes through the ray tests."""
    from acoustic_spaces.engine import diffraction

    room = _town(both_sides=both_sides)
    kept, tried = [0], [0]
    culled, reaching = diffraction._reflecting_pairs, diffraction._reaching_pairs

    def counting(model, edges, bounds, end, planes, among, offset, threads=0):
        rows, cols = culled(model, edges, bounds, end, planes, among, offset, threads)
        kept[0] += len(rows)
        tried[0] += len(among) * len(planes)
        return rows, cols

    found = 0
    for source in ("s1", "s2"):
        for listener in ("l1", "l2"):
            monkeypatch.setattr(diffraction, "_reflecting_pairs", counting)
            fast = early_paths(room, source, listener, threads=1)
            many = early_paths(room, source, listener, threads=3)
            monkeypatch.setattr(diffraction, "_reflecting_pairs", _all_reflecting)
            monkeypatch.setattr(diffraction, "_reaching_pairs", _all_reaching)
            slow = early_paths(room, source, listener, threads=1)
            monkeypatch.setattr(diffraction, "_reaching_pairs", reaching)
            _same_paths(fast, slow)
            _same_paths(many, slow)
            found += int((fast.diffracted & (fast.order > 0)).sum())
    assert found > 0  # paths that reflect on the way over an edge are there
    assert kept[0] < 0.5 * tried[0]  # and most pairs were never tested


def _plane(normal, offset, low, high, spread=0.0):
    return (
        np.array([normal], float),
        np.array([offset], float),
        np.array([low], float),
        np.array([high], float),
        np.array([spread], float),
    )


def _kept(starts, ends, plane, end, offset=1e-6, threads=0) -> list[int]:
    from acoustic_spaces import _native

    rows, cols = _native.reflecting_pairs(
        np.array(starts, float),
        np.array(ends, float),
        *plane,
        np.array(end, float),
        offset,
        0.01,
        threads,
    )
    assert (cols == 0).all()
    return rows.tolist()


def test_an_edge_is_kept_where_its_shadow_falls_on_the_faces() -> None:
    """The ground's faces from (0, 0) to (10, 10), the listener 2 m over them: an edge
    whose shadow, cast from the listener's image, falls on them is kept; one whose
    shadow falls past them, one under the ground and one standing up out of it beside
    them are not; a post that rises far enough for its shadow to reach them is."""
    ground = _plane([0, 0, 1], 0.0, [0, 0, 0], [10, 10, 0])
    end = [5.0, 5.0, 2.0]
    starts = [
        [4.0, 5.0, 3.0],  # over the faces
        [100.0, 5.0, 1.0],  # its shadow falls at x = 68
        [4.0, 5.0, -1.0],  # under the ground
        [50.0, 5.0, -5.0],  # a post through the ground, its shadow from x = 18 to 50
        [12.0, 5.0, -5.0],  # a post whose top's shadow falls at x = 5.6
    ]
    ends = [[6.0, 5.0, 3.0], [102.0, 5.0, 1.0], [6.0, 5.0, -1.0], [50, 5, 5], [12, 5, 20]]
    assert _kept(starts, ends, ground, end) == [0, 4]
    assert _kept(starts, ends, ground, end, threads=1) == [0, 4]


def test_edges_are_kept_where_the_bounds_do_not_hold() -> None:
    """Rays that graze the plane, faces that stray from it, an image too near it and a
    plane of openings: kept where a reflection cannot be ruled out, never otherwise."""
    end = [5.0, 5.0, 0.002]
    far_edge = ([3000.0, 5.0, 0.5], [3002.0, 5.0, 0.5])  # its shadow falls at x = 17
    ground = _plane([0, 0, 1], 0.0, [0, 0, 0], [10, 10, 0])
    assert _kept([far_edge[0]], [far_edge[1]], ground, end) == [0]  # grazing
    # The edge lies 1 mm under the plane, its faces stray 2 mm from it.
    under = ([4.0, 5.0, -0.001], [6.0, 5.0, -0.001])
    assert _kept([under[0]], [under[1]], ground, [5, 5, 2]) == []
    strayed = _plane([0, 0, 1], 0.0, [0, 0, -0.002], [10, 10, 0.002], spread=0.002)
    assert _kept([under[0]], [under[1]], strayed, [5, 5, 2]) == [0]
    # The listener's image lies within reach of the edge's offset: nothing is ruled out.
    assert _kept([[100.0, 5.0, 1.0]], [[102.0, 5.0, 1.0]], ground, [5, 5, 1e-7]) == [0]
    # A plane with only openings has an empty box: no reflection off it.
    empty = _plane([0, 0, 1], 0.0, [np.inf] * 3, [-np.inf] * 3)
    assert _kept([[4.0, 5.0, 3.0]], [[6.0, 5.0, 3.0]], empty, [5, 5, 2]) == []


def test_culling_is_a_superset_of_rays_that_meet_the_faces() -> None:
    """Random edges and a tilted rectangle of faces: every edge from which a ray (started
    within the offset of a point of the edge, towards the end's image) meets the
    rectangle is kept."""
    rng = np.random.default_rng(3)
    normal = np.array([0.3, -0.2, 1.0])
    normal /= np.linalg.norm(normal)
    u = np.cross(normal, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    centre = np.array([2.0, 1.0, 0.5])
    corners = centre + np.array([[a, b] for a in (-3, 3) for b in (-2, 2)]) @ np.array([u, v])
    plane = _plane(normal, normal @ centre, corners.min(0), corners.max(0))
    offset = 1e-3
    for trial in range(40):
        end = centre + normal * rng.uniform(0.5, 5.0) + rng.normal(0, 3, 3) * [1, 1, 0]
        end += normal * max(0.0, 0.5 - (end - centre) @ normal)
        image = end - 2.0 * ((end - centre) @ normal) * normal
        starts = rng.uniform(-15, 15, (60, 3))
        ends = starts + rng.normal(0, 6, (60, 3))
        kept = set(_kept(starts, ends, plane, end, offset=offset))
        for i in range(60):
            t = rng.uniform(0, 1, (200, 1))
            jitter = rng.normal(0, 1, (200, 3))
            jitter *= offset * rng.uniform(0, 1, (200, 1)) / np.linalg.norm(jitter, axis=1)[:, None]
            start = starts[i] + t * (ends[i] - starts[i]) + jitter
            gap = (start - centre) @ normal
            towards = image - start
            s = gap / np.maximum((-towards) @ normal, 1e-300)
            meets = (gap > 0) & (s > 0) & (s <= 1)
            point = start + s[:, None] * towards - centre
            inside = (np.abs(point @ u) <= 3) & (np.abs(point @ v) <= 2)
            if (meets & inside).any():
                assert i in kept, (trial, i)


def test_reaching_pairs_keep_every_path_that_touches_its_edge_in_time() -> None:
    """The native filter of the listener's images takes every pair the exact tests do
    (the touch on the edge, the length, the distance from the edge's line) and checks its
    arguments."""
    from acoustic_spaces import _native
    from acoustic_spaces.engine.diffraction import _reach

    rng = np.random.default_rng(11)
    starts = rng.uniform(-20, 20, (30, 3))
    ends = starts + rng.normal(0, 8, (30, 3))
    count = 25
    normals = rng.normal(0, 1, (count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    offsets = rng.uniform(-30, 30, count)
    low, high = np.full((count, 3), -1e3), np.full((count, 3), 1e3)
    spread = np.zeros(count)
    listener = np.array([1.0, 2.0, 3.0])
    row_edge = rng.integers(0, 30, 80)
    t_s, rho_s = rng.uniform(-5, 15, 80), rng.uniform(0.5, 20, 80)
    max_distance, tolerance = 60.0, 1e-6
    args = (starts, ends, normals, offsets, low, high, spread, listener, 1e-6, 0.01)
    rows, cols = _native.reaching_pairs(
        *args, row_edge, t_s, rho_s, max_distance, tolerance, 1e-6, 0
    )
    got = set(zip(rows.tolist(), cols.tolist(), strict=True))
    edges_kept, planes_kept = _native.reflecting_pairs(*args, 0)
    for e, j in zip(edges_kept.tolist(), planes_kept.tolist(), strict=True):
        distance = normals[j] @ listener - offsets[j]
        image = listener - 2 * distance * normals[j]
        direction = (ends[e] - starts[e]) / np.linalg.norm(ends[e] - starts[e])
        length = np.linalg.norm(ends[e] - starts[e])
        t_r = (image - starts[e]) @ direction
        rho_r = np.linalg.norm(image - starts[e] - t_r * direction)
        for r in np.flatnonzero(row_edge == e):
            ok, *_ = _reach(
                2,
                length,
                t_s[r : r + 1],
                rho_s[r : r + 1],
                np.array([t_r]),
                np.array([rho_r]),
                1.0,
                max_distance,
            )
            if ok[0] and rho_r > tolerance:
                assert (int(r), j) in got
    assert len(got) > 0
    with pytest.raises(ValueError, match="row_edge"):
        _native.reaching_pairs(
            *args, np.array([30]), t_s[:1], rho_s[:1], max_distance, tolerance, 1e-6, 0
        )
    with pytest.raises(ValueError, match="per plane"):
        _native.reflecting_pairs(
            starts, ends, normals, offsets[:3], low, high, spread, listener, 1e-6, 0.01, 0
        )


def test_lines_that_meet_turn_the_path_down_quietly() -> None:
    """Two edges that meet at a building's corner draw the shortest path over both to
    that corner, where its curvature is beyond the largest number: the path is turned
    down without a floating-point warning (the suite runs with warnings as errors)."""
    from acoustic_spaces.engine.diffraction import _shortest

    corner = [50.914045212647316, 33.62486697879731]
    t1, t2, ok = _shortest(
        np.array([[*corner, 0.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[*corner, 16.799833860212203]]),
        np.array([[0.9977720124720059, -0.06671589861167475, 0.0]]),
        np.array([92.28183640888345, 90.9684650549119, 2.874150844495204]),
        np.array([60.35877251148018, 42.57767948755837, 21.341134114627412]),
    )
    assert not ok[0] and np.isfinite(t1[0]) and np.isfinite(t2[0])


# -- the ends of edges ----------------------------------------------------------------


def _share(t_k, length, fade, k=2.0, r=10.0):
    """along_edge for one path touching an edge's line at ``t_k``, both legs ``r`` long
    and square to it, at wave number ``k``, with no taper."""
    length_param = r / 2.0
    scale = np.full((1, 1, 1), np.sqrt(k / (np.pi * length_param)))
    out = along_edge(
        np.array([t_k], float),
        np.array([length], float),
        scale,
        np.array([fade], float),
        np.array([1e9]),
    )
    return complex(out[0, 0, 0])


def test_a_path_over_an_edge_fades_out_past_its_free_end() -> None:
    """Well inside, the edge gives a path all of it; at a free end half (the Fresnel
    integral along the edge from the end on); well past it almost nothing."""
    assert abs(_share(50.0, 100.0, (1.0, 1.0)) - 1.0) < 0.05
    assert abs(_share(100.0, 100.0, (1.0, 1.0)) - 0.5) < 0.02
    assert abs(_share(120.0, 100.0, (1.0, 1.0))) < 0.05
    # An end that goes on (fade 0) stops the path hard, as an endless edge would.
    assert abs(_share(99.0, 100.0, (1.0, 0.0)) - 1.0) < 0.02
    assert abs(_share(101.0, 100.0, (1.0, 0.0))) < 0.02


def test_the_fades_of_two_pieces_of_an_edge_add_up_to_the_whole() -> None:
    """Two collinear pieces, each fading at the end where they meet, give together what
    the whole edge gives, wherever the path touches the line near the joint."""
    for t_k in (37.0, 39.5, 40.0, 40.3, 43.0):
        whole = _share(t_k, 100.0, (1.0, 1.0))
        pieces = _share(t_k, 40.0, (1.0, 1.0)) + _share(t_k - 40.0, 60.0, (1.0, 1.0))
        assert pieces == pytest.approx(whole, abs=1e-9)


def test_how_a_barrier_s_edges_fade_at_their_ends() -> None:
    """A barrier's top meets its end edges at corners (1/2), and the ground closes the end
    edges' feet, where they go on as their images (0)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_barriers_in_wind import scene

    room = Room.from_scene(scene(atmosphere=None))
    edges = find_edges(TraceModel.from_room(room))
    upright = np.abs(edges.direction[:, 2]) > 0.99
    assert upright.sum() == 2 and (~upright).sum() == 1
    assert np.all(edges.fade[~upright] == 0.5)
    assert np.all(edges.direction[upright, 2] > 0.0)  # from the foot up
    assert np.all(edges.fade[upright] == [0.0, 0.5])
    assert np.all(edges.closed[upright] == [True, False])


def test_a_solid_block_s_edges_stop_at_its_corners(tmp_path: Path) -> None:
    """Three edges of a solid block meet at each corner, where the paths over them stop at
    the ends as on endless edges (fading them came out up to 1.25 dB above BTM at 63 and
    125 Hz near the corner), and the faces behind a convex corner close none of them."""
    x0, x1, y0, y1, z0, z1 = 0.0, 2.0, -10.0, 0.0, -300.0, 0.0  # open underneath
    quads = [
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
        [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
    ]
    room = edge_diffraction._room(tmp_path, "box", quads, (1.0, 3.0, 1.0), (4.0, -2.0, 0.3))
    edges = find_edges(TraceModel.from_room(room))
    assert (np.isclose(edges.wedge, 1.5).sum(), np.isclose(edges.wedge, 2.0).sum()) == (8, 4)
    assert np.all(edges.fade == 0.0)
    assert not np.any(edges.closed)


def _weighted_share(u: float, v: float) -> complex:
    """The integral of exp(-j pi t^2 / 2) / (t^2 + v^2) up to ``u`` over that along the
    whole line, by the midpoint rule out to 300 either side of ``u``."""
    steps = (np.arange(600_000) + 0.5) * 5e-4

    def part(t):
        return np.sum(np.exp(-0.5j * np.pi * t**2) / (t**2 + v**2))

    upto, beyond = part(u - steps), part(u + steps)
    return complex(upto / (upto + beyond))


@pytest.mark.parametrize(
    "u, v",
    [(-0.3, 0.05), (0.4, 1.5), (-1.2, 0.7), (2.5, 0.3), (-2.4, 3.0), (3.2, 2.6), (-4.0, 2.3)],
)
def test_a_term_s_share_of_a_finite_edge_is_weighted_for_its_boundary(u: float, v: float) -> None:
    """The share of one term that the edge gives up to ``u`` past the path's point, its
    boundary ``v`` away (both as arguments of the Fresnel integral along the edge),
    against the integral itself: each of the three ways the engine evaluates it."""
    cumulative = diffraction_module._cumulative
    got = complex(cumulative(np.array([u]), np.array([v]))[0])
    assert abs(got - _weighted_share(u, v)) < 1e-5
    # Half of it lies either side of the point; on the boundary all of it is at the
    # point; far from the boundary it is the Fresnel integral's.
    assert cumulative(np.array([0.0]), np.array([v]))[0] == pytest.approx(0.5, abs=1e-12)
    assert abs(cumulative(np.array([u]), np.array([1e-9]))[0] - (u > 0)) < 1e-6
    far = cumulative(np.array([u]), np.array([1e3]))[0]
    assert abs(far - cumulative(np.array([u]))[0]) < 1e-5


@pytest.mark.parametrize("bend", [10.0, 2.0, -2.0, -10.0])
def test_a_bent_screen_matches_btm_beside_the_bend_and_deep_in_the_shadow(
    tmp_path: Path, bend: float
) -> None:
    """Two screens whose tops meet at a bend of 2 or 10 degrees, away from the source
    (+) and towards it (-): the listener within 2 cm of the top's shadow boundary, the
    line to the source passing the top 0.5 and 0.1 m from the bend, and deep in the
    shadow past the bend, is within 0.1 dB of BTM over both tops in every band (0.03 dB
    here). Each top took the Fresnel integral's share of its terms, whose step on the
    boundary the direct sound did not match: 4.5 to 5.3 dB too high beside a bend away
    from the source, up to 9.9 dB too low beside one towards it (63 Hz)."""
    listeners = [(5.0, -0.5, z) for z in (0.98137, 0.99737, 1.00137, 1.02137)]
    listeners += [(5.0, 0.3, 1.00137), (5.0, 0.003, -2.0), (5.0, 0.553, -2.0), (5.0, 1.503, -2.0)]
    found, exact = edge_diffraction.bent_screen_levels(
        tmp_path, bend, (-5.0, -0.5, -1.0), listeners
    )
    assert np.abs(found - exact).max() < 0.1


def _rhombus(tmp_path: Path, angle: float) -> Edges:
    """The edges of a thin screen with four sides 2 m long that meet at ``angle`` degrees
    and at 180 less."""
    a = np.array([2.0, 0.0, 0.0])
    b = 2.0 * np.array([math.cos(math.radians(angle)), 0.0, math.sin(math.radians(angle))])
    quads = [[tuple(corner) for corner in (0 * a, a, a + b, b)]]
    room = edge_diffraction._room(tmp_path, "rhombus", quads, (1.0, 3.0, 1.0), (1.0, -3.0, 1.0))
    return find_edges(TraceModel.from_room(room))


def test_how_a_screen_s_edges_fade_at_corners_of_other_angles(tmp_path: Path) -> None:
    """The paths over two edges that meet within 1 degree of square fade halfway at the
    corner, where the two halves add up; at any other angle each edge keeps its own part
    of its terms (all the way, 1). They faded halfway from 45 to 135 degrees, which came
    out up to 2.75 dB from BTM at a corner of 45 degrees."""
    assert np.all(_rhombus(tmp_path, 90.0).fade == 0.5)
    assert np.all(_rhombus(tmp_path, 89.5).fade == 0.5)
    for angle in (88.0, 60.0, 45.0, 20.0):
        assert np.all(_rhombus(tmp_path, angle).fade == 1.0)


@pytest.mark.parametrize(
    "angle, turned, ys",
    [(45.0, 45.0, (-0.057, -0.007, 0.003, 0.053)), (120.0, 0.0, (3.6532, 3.9032))],
)
def test_corners_of_other_angles_match_btm(tmp_path: Path, angle: float, turned, ys) -> None:
    """Behind the corner of a thin screen whose edges meet at 45 degrees, one of them
    upright, and at 120 degrees: within 0.15 dB of BTM over both edges in every band,
    and no step that BTM does not have. The acute corner was up to 2.75 dB off (250 Hz)
    with a step of 3 dB at 125 Hz between the two listeners in the middle, where the
    direct sound comes round the upright edge; the obtuse one 0.5 to 0.6 dB off."""
    corner = edge_diffraction.screen_corner(tmp_path, angle, turned, np.array(ys), per_band=96)
    assert max(corner["largest_difference_db"]) < 0.15
    steps = np.subtract(corner["largest_step_db"], corner["exact_largest_step_db"])
    assert np.abs(steps).max() < 0.15


@pytest.mark.parametrize("both_sides", [False, True])
@pytest.mark.parametrize(
    "source, listener",
    [
        ((0.0, -8.0, 3.0), (20.0, -8.0, 3.0)),  # both at the height of the top
        ((0.0, -8.0, 1.0), (20.0, -8.0, 5.0)),  # mirrored in the barrier, the top between
        ((0.0, -10.5, 3.0), (20.0, -10.5, 3.0)),  # 0.5 m past the end
        ((0.0, -12.0, 3.0), (20.0, -12.0, 3.0)),  # 2 m past it
    ],
)
def test_no_hole_where_a_path_touches_the_very_end_of_an_edge(
    both_sides: bool, source, listener
) -> None:
    """A barrier 3 m high and 20 m long: with the source and the listener at the height
    of its top (behind it, and in sight of each other 0.5 and 2 m past its end), or
    mirrored in it at 1 and 5 m, the shortest path round each of its upright ends
    touches the end's top corner exactly. The level there is the mean of those 1 mm
    lower and higher (within 0.1 dB, 0.05 dB here), which are 0.12 dB apart at most. The
    paths that touch the very end of a fading edge were dropped: the level there was 4.8
    to 5.6 dB lower at 63 Hz and 2.1 dB higher at 125 Hz (at the top's height), 3 dB
    lower at 63 and 125 Hz and as much higher at 250 Hz (mirrored), and 0.4 and 0.2 dB
    lower 0.5 and 2 m past the end."""
    path = ((10, -10), (10, 10))
    levels = []
    for dz in (-0.001, 0.0, 0.001):
        at = (listener[0], listener[1], listener[2] + dz)
        paths = early_paths(_ground_barrier(both_sides, at, source=source, path=path))
        levels.append(10 * np.log10(total_energy(paths)))
    assert np.abs(levels[2] - levels[0]).max() < 0.2
    assert np.abs(levels[1] - 0.5 * (levels[0] + levels[2])).max() < 0.1


def test_the_very_end_of_a_fading_edge_counts_once() -> None:
    """A path that touches an end of its edge exactly: nothing reflected (kind 0), with a
    reflection before the edge (1), after it (2) and both (3). An end that fades is part
    of its fade, and keeps them all. Where the end lies in the plane the path reflects
    off, the path that reflects before the edge and the one that reflects after it are
    one path: the first counts at the edge's start only, the second at its end only, and
    the one with both at neither, whether the end fades or not."""
    kind, ones = np.arange(4), np.ones(4)

    def kept(at: float, fade: float, held: bool) -> list[bool]:
        in_plane = np.repeat((kind > 0) & held, 2).reshape(4, 2)
        past = (np.full((4, 2), fade), 1.0, in_plane)
        reach = diffraction_module._reach
        return reach(kind, 3.0 * ones, at * ones, ones, at * ones, ones, 343.0, 1.0, past)[
            0
        ].tolist()

    assert kept(0.0, 1.0, False) == kept(3.0, 1.0, False) == [True, True, True, True]
    for fade, held in ((1.0, True), (0.0, True), (0.0, False)):
        assert kept(0.0, fade, held)[1:] == [True, False, False]
        assert kept(3.0, fade, held)[1:] == [False, True, False]
    # Nothing reflected: the end of an edge that goes on belongs to the next one.
    assert (kept(3.0, 1.0, True)[0], kept(3.0, 0.0, False)[0]) == (True, False)


def _wall_in_line(
    sources,
    listeners,
    side: float = -1.0,
    shift: float = 0.0,
    extent=((-45, -45), (45, 45)),
    start: float = 10.0,
    others=(),
    **simulation,
):
    """A rigid building 10 m deep whose wall runs along y = ``shift`` from x = ``start``
    to 20, the building on the side ``side`` of it, and one like it from x = a to a + 10
    for each a of ``others``, outdoors on ground over ``extent``. Reflections up to the
    first order, unless ``simulation`` gives ``early`` settings, which replace them (and
    with them that order)."""
    boxes = [[[start, 0], [20, 0], [20, 10 * side], [start, 10 * side]]]
    boxes += [[[a, 0], [a + 10, 0], [a + 10, 10 * side], [a, 10 * side]] for a in others]
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": {"extent": [list(c) for c in extent], "sky": 20.0},
                    "buildings": [
                        {
                            "name": f"h{k}",
                            "footprint": [[x, y + shift] for x, y in box],
                            "height": 6,
                        }
                        for k, box in enumerate(boxes)
                    ],
                },
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": f"s{k}", "position": list(p)} for k, p in enumerate(sources)],
                "listeners": [
                    {"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)
                ],
                "simulation": {"early": {"max_order": 1}, **simulation},
            }
        )
    )


def _pairs(room: Room, count: int) -> list:
    return [early_paths(room, f"s{k}", f"l{k}", threads=1) for k in range(count)]


@pytest.mark.parametrize("bands", ["octave", "third_octave"])
@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("swap", [False, True])
def test_a_listener_in_line_with_a_wall_hears_one_corner(swap: bool, side: float, bands) -> None:
    """The source 10 m before a building's wall, in its plane, and the listener 20 m past
    it, in its plane too or 10 um into the wall's shadow: the way runs along the wall
    from one upright corner to the other, and both corners' paths are that one ray. The
    listener on the line has the paths it has just off it in the shadow (over the far
    corner alone), and its level; with the two swapped, those over the near one. Both
    corners' paths were kept on the line, 4 to 8 dB louder than beside it."""
    source, far = (0.0, 0.0, 1.0), (40.0, 0.0, 1.5)
    if swap:
        room = _wall_in_line([far, far], [source, (0.0, 1e-5 * side, 1.0)], side)
    else:
        room = _wall_in_line([source, source], [far, (40.0, 1e-5 * side, 1.5)], side)
    room.scene.simulation.bands = bands
    room = Room.from_scene(room.scene)
    on, off = _pairs(room, 2)
    assert sorted(on.edge.tolist()) == sorted(off.edge.tolist()) and len(on) == 2
    np.testing.assert_allclose(
        10 * np.log10(total_energy(on)), 10 * np.log10(total_energy(off)), atol=0.01
    )


def test_a_listener_in_line_with_a_wall_over_two_edges() -> None:
    """As above with paths over two edges: on the line, as 10 um into the shadow."""
    beside = (40.0, -1e-5, 1.5)
    room = _wall_in_line(
        [(0.0, 0.0, 1.0)] * 2, [(40.0, 0.0, 1.5), beside], early={"diffraction": "second_order"}
    )
    on, off = _pairs(room, 2)
    np.testing.assert_allclose(
        10 * np.log10(total_energy(on)), 10 * np.log10(total_energy(off)), atol=0.01
    )


def test_a_map_point_in_line_with_a_wall_hears_one_corner() -> None:
    """A coverage map whose points at y = 0 lie in the plane of the wall the source
    stands in line with: there as with the wall and the source 10 um aside (the points
    then in the wall's shadow)."""
    from acoustic_spaces.engine import coverage_map
    from acoustic_spaces.scene import Coverage

    settings = Coverage(spacing=10.0)
    maps = []
    for shift in (0.0, 1e-5):
        room = _wall_in_line(
            [(-5.0, shift, 1.0)],
            [(-5.0, 10.0, 1.2)],
            shift=shift,
            extent=((-15, -25), (45, 25)),
            late={"rays": 50},
        )
        maps.append(coverage_map(room, settings=settings, threads=1))
    on, off = (
        m.get().values["spl"][(np.abs(m.grid.points[:, 1]) < 1e-9) & (m.grid.points[:, 0] > 20.0)]
        for m in maps
    )
    assert len(on) == 2
    np.testing.assert_allclose(on, off, atol=0.01)


@pytest.mark.parametrize("order", ["first_order", "second_order"])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_a_listener_on_a_wall_in_line_with_the_source_hears_it(side: float, order) -> None:
    """The source in the plane of a building's wall and the listener on the wall, or at
    its far corner: a point on a face has only the side in front of it. On the wall it
    reads as 10 um in front; at the corner it lies between its neighbours in front of
    the wall and beside the corner, behind the wall's plane. Taking it for a point past
    the wall's end, behind it, dropped every path and left it with none."""
    front, behind = -1e-5 * side, 1e-5 * side
    listeners = [
        (15.0, 0.0, 1.5),
        (15.0, front, 1.5),
        (20.0, 0.0, 1.5),
        (20.0, front, 1.5),
        (20.0 + 1e-5, behind, 1.5),
    ]
    room = _wall_in_line([(0.0, 0.0, 1.0)] * 5, listeners, side, early={"diffraction": order})
    on, ahead, corner, lit, past = (10 * np.log10(total_energy(p)) for p in _pairs(room, 5))
    np.testing.assert_allclose(on, ahead, atol=0.01)
    assert np.all((corner > np.minimum(lit, past)) & (corner < np.maximum(lit, past)))


def test_a_listener_on_another_wall_in_line_is_past_the_first() -> None:
    """Two buildings whose walls lie in one plane with the source, the listener on the
    second's wall: it lies past the first's wall, as a point in the open between them
    does, so it has no path over the first's near corner either (the wave along that
    wall is the far corner's). Taking the second wall for the first's face kept both
    corners' paths there."""
    listeners = [(35.0, 0.0, 1.5), (25.0, 0.0, 1.5)]
    room = _wall_in_line([(0.0, 0.0, 1.0)] * 2, listeners, others=(30.0,))
    edges = find_edges(TraceModel.from_room(room))
    upright = np.abs(edges.direction[:, 2]) > 1.0 - 1e-9
    near = np.flatnonzero(
        upright & (np.linalg.norm(edges.start[:, :2] - [10.0, 0.0], axis=1) < 1e-9)
    )
    far = np.flatnonzero(
        upright & (np.linalg.norm(edges.start[:, :2] - [20.0, 0.0], axis=1) < 1e-9)
    )
    assert len(near) == 1 and len(far) == 1
    for paths in _pairs(room, 2):
        assert near[0] not in paths.edge and far[0] in paths.edge


def test_a_map_point_on_a_wall_in_line_with_the_source_hears_it() -> None:
    """A coverage map whose points at y = 0 lie in the plane of the wall the source
    stands in line with, one on the wall and one at its far corner: the one on the wall
    as with the wall 10 um back (the point then in front of it), the corner's heard too.
    Both had no paths left."""
    from acoustic_spaces.engine import coverage_map
    from acoustic_spaces.scene import Coverage

    settings = Coverage(spacing=10.0)
    maps = []
    for shift in (0.0, -1e-5):
        room = _wall_in_line(
            [(-5.0, 0.0, 1.0)],
            [(-5.0, 10.0, 1.2)],
            shift=shift,
            extent=((-15, -25), (45, 25)),
            start=5.0,
            late={"rays": 50},
        )
        maps.append(coverage_map(room, settings=settings, threads=1))
    on, ahead = (
        m.get().values["spl"][
            (np.abs(m.grid.points[:, 1]) < 1e-9) & np.isin(m.grid.points[:, 0], (10.0, 20.0))
        ]
        for m in maps
    )
    assert len(on) == 2 and np.isfinite(on).all()
    np.testing.assert_allclose(on[0], ahead[0], atol=0.01)


@pytest.mark.parametrize("swap", [False, True])
def test_the_way_along_a_thick_barrier_s_top(tmp_path: Path, swap: bool) -> None:
    """The source and the listener at the height of a 1 m thick barrier's top, on either
    side: just above it and just below the level is the same (within 0.1 dB), and on it
    too. With both corners' paths it was 8 dB louder."""
    steps = (-1e-6, 0.0, 1e-6)
    if swap:
        level = np.concatenate(
            [
                _over_barrier(
                    _barrier(tmp_path, (4.0, 0.3, dz), [(-3.0, 0.0, 0.0)]), "second_order"
                )
                for dz in steps
            ]
        )
    else:
        room = _barrier(tmp_path, (-3.0, 0.0, 0.0), [(4.0, 0.3, dz) for dz in steps])
        level = _over_barrier(room, "second_order")
    assert np.abs(level[2] - level[0]).max() < 0.1
    np.testing.assert_allclose(level[1], level[0], atol=0.01)
