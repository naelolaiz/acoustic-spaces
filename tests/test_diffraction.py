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


def test_levels_are_continuous_across_the_shadow_boundary() -> None:
    listeners = [[2.0, y, 1.5] for y in (4.99, 4.999, 5.0, 5.001, 5.01)]
    room = _l_room(listeners)
    # The direct sound and the floor and ceiling reflections all end on the boundary.
    plain = _levels(room, max_order=1, diffraction="none")
    assert np.all(plain[1] - plain[3] > 5.0)
    # Only the direct sound bends round the corner: the reflections still jump.
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


def test_reflections_bend_round_edges_too() -> None:
    # Behind the corner the floor and ceiling reflections arrive over the edge as well.
    room = _l_room([[2.0, 8.0, 1.5]])
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
    # Behind the corner, at 125 Hz: more sound, and more of it early. The statistical
    # tail is Barron's energy less the early paths', so of what bends round the corner
    # only the direct sound adds to the total.
    for name, rise in (("g", 0.02), ("c80", 0.02)):
        values = bent.get().values[name][shadow, 1] - plain.get().values[name][shadow, 1]
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
