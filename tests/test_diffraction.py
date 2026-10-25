import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.diffraction import (
    EdgeDiffraction,
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
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.scene.model import EarlyStage

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


def _barrier(tmp_path: Path, source, listeners) -> Room:
    """A rigid block from x = 0 to 1 m with its top at z = 0, open underneath, so wide
    and deep that only its two top corners (along y) matter."""
    a, h = 300, -300
    quads = [
        [(0, -a, 0), (1, -a, 0), (1, a, 0), (0, a, 0)],
        [(0, -a, h), (0, -a, 0), (0, a, 0), (0, a, h)],
        [(1, -a, h), (1, a, h), (1, a, 0), (1, -a, 0)],
        [(0, -a, h), (1, -a, h), (1, -a, 0), (0, -a, 0)],
        [(0, a, h), (0, a, 0), (1, a, 0), (1, a, h)],
    ]
    lines = [f"v {x} {y} {z}" for quad in quads for x, y, z in quad]
    lines += ["g block"] + [f"f {4 * i + 1} {4 * i + 2} {4 * i + 3} {4 * i + 4}" for i in range(5)]
    obj = tmp_path / "block.obj"
    obj.write_text("\n".join(lines) + "\n")
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": list(source)}],
                "listeners": [
                    {"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)
                ],
            }
        )
    )


def _over_barrier(room: Room, mode: str) -> np.ndarray:
    """The level of every listener over the barrier, without air, as in BARRIER_BTM."""
    model = TraceModel.from_room(room)
    c, air = room.speed_of_sound(), np.zeros(8)
    source = np.array(room.scene.sources[0].position)
    diffraction = EdgeDiffraction(model, source, speed_of_sound=c, air=air, mode=mode)
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


# -- reflections on both sides of an edge ----------------------------------------------


def _ground_barrier(both_sides: bool, listener=(20.0, 0.0, 1.5), absorption: float = 0.0) -> Room:
    """A thin barrier 3 m high, 10 m from a source 1 m over a ground of this absorption."""
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": {"extent": [[-100, -200], [200, 200]], "sky": 40},
                    "barriers": [
                        {"name": "barrier", "path": [[10, -190], [10, 190]], "height": 3.0}
                    ],
                },
                "materials": {
                    "ground": {"absorption": [absorption] * 8, "scattering": 0.0},
                    "rigid": {"absorption": [0.0] * 8, "scattering": 0.0},
                },
                "default_material": "ground",
                "assignments": {"barrier": "rigid"},
                "sources": [{"name": "s", "position": [0.0, 0.0, 1.0]}],
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
