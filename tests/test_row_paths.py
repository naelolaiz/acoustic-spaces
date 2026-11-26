"""The solver for whole rows among a simulation's early paths (engine/row_paths.py)."""

import math

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import reweight, rows
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.coverage import coverage_map
from acoustic_spaces.engine.diffraction import FADE_REACH, FADE_TAPER, _fade_reach, fade_window
from acoustic_spaces.engine.impedance import fit_admittance
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.refracted import WAVE_MIN_RANGE
from acoustic_spaces.engine.row_paths import MAX_BAND_HZ, ROW, RowPaths
from acoustic_spaces.engine.simulate import coherent_bands, early_paths
from acoustic_spaces.geometry.outdoor import Layout
from acoustic_spaces.scene import Coverage
from acoustic_spaces.walkthrough import simulate_movement, simulate_trajectory

#: A barrier 3 m high across the x axis, 5 m from the source, 500 m long.
LONG = {"name": "screen", "path": [[5, -250], [5, 250]], "height": 3}
SOURCE = (0.0, 0.0, 0.5)
WINDY = {"preset": "windy", "wind": {"direction_deg": 0}}
LISTENER = (25.0, 0.0, 1.5)


def scene(
    listener=LISTENER,
    barriers=(LONG,),
    buildings=(),
    source=SOURCE,
    band=125,
    ground=None,
    roofs=None,
    screens=None,
    atmosphere=None,
    bands="octave",
    extent=((-300.0, -300.0), (300.0, 300.0)),
    **early,
) -> Scene:
    """A source and a listener on a rigid ground (the material ``ground`` if given) with
    barriers (of the material ``screens`` if given) and buildings (their roofs of the
    material ``roofs`` if given), whose early paths add in pressure up to the octave
    band ``band``."""
    materials = {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}}
    assignments = {}
    if ground is not None:
        materials["soil"] = ground
        assignments["ground"] = "soil"
    if roofs is not None:
        materials["cover"] = roofs
        assignments["roofs"] = "cover"
    if screens is not None:
        materials["fence"] = screens
        assignments["barriers"] = "fence"
    data = {
        "geometry": {
            "ground": {"extent": [list(corner) for corner in extent], "sky": 30.0},
            "barriers": list(barriers),
            "buildings": list(buildings),
        },
        "materials": materials,
        "assignments": assignments,
        "default_material": "hard",
        "sources": [{"name": "s", "position": list(source)}],
        "listeners": [{"name": "l", "position": list(listener)}],
        "simulation": {
            "seed": 1,
            "bands": bands,
            "early": {
                "method": "image_sources",
                "max_order": 2,
                "diffraction": "second_order",
                "diffraction_reflections": "both_sides",
                "coherent_max_band_hz": band,
                **early,
            },
            "late": {"rays": 200},
            "ir_length_s": 0.3,
        },
    }
    if atmosphere is not None:
        data["atmosphere"] = atmosphere
    return Scene.model_validate(data)


def levels(room: Room, paths=None) -> np.ndarray:
    """The early paths' level per band, dB re the free field, without the air."""
    paths = early_paths(room) if paths is None else paths
    source = np.array(room.scene.sources[0].position)
    listener = np.array(room.scene.listeners[0].position)
    distance = float(np.linalg.norm(listener - source))
    air = room.air_attenuation(room.scene.simulation.bands)
    energy = total_energy(paths) * np.exp(air * distance)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(energy * 4.0 * np.pi * distance**2)


def solver_of(room: Room) -> RowPaths:
    """The solver for the room's source, as a simulation builds it."""
    model = TraceModel.from_room(room)
    found = RowPaths.of(
        model,
        np.array(room.scene.sources[0].position),
        "solver",
        2,
        speed_of_sound=room.speed_of_sound(),
        air=room.air_attenuation(model.bands),
        coherent=coherent_bands(room.scene.simulation, model.absorption.shape[1]),
    )
    assert found is not None
    return found


def own_level(room: Room, solver: RowPaths, bands, *row, **more) -> np.ndarray:
    """The level (dB re the free field) of :func:`rows.over_row` in ``bands`` at the
    early paths' frequencies, for the room's source and listener; a slice of bands is
    one run, with a level for each band of it."""
    source = np.array(room.scene.sources[0].position)
    listener = np.array(room.scene.listeners[0].position)
    distance = float(np.linalg.norm(listener - source))
    out = []
    for band in bands:
        f = solver.frequencies[band]
        found = rows.over_row(f.ravel(), room.speed_of_sound(), *row, **more)
        heard = np.mean(np.abs(found.pressure.reshape(f.shape)) ** 2, axis=-1)
        out.append(10.0 * np.log10(heard * distance**2))
    return np.array(out)


@pytest.fixture
def asked(monkeypatch):
    """What the solver is asked, call by call, and still answered."""
    calls = []
    real = rows.over_row

    def over_row(frequencies, speed, impedance, zs, reach, zr, obstacles, **more):
        calls.append(
            {
                "frequencies": np.array(frequencies),
                "impedance": impedance,
                "ends": (zs, reach, zr),
                "obstacles": list(obstacles),
                **more,
            }
        )
        return real(frequencies, speed, impedance, zs, reach, zr, obstacles, **more)

    monkeypatch.setattr(rows, "over_row", over_row)
    return calls


def test_the_solver_is_the_default_and_edges_leaves_the_paths() -> None:
    assert Scene.model_validate(scene().to_data()).simulation.early.rows == "solver"
    room = Room.from_scene(scene(rows="edges"))
    assert not np.any(early_paths(room).edge == ROW)
    with pytest.raises(ValueError, match="rows is one of solver, edges"):
        RowPaths.of(TraceModel.from_room(room), np.array(SOURCE), "both")


def test_behind_a_long_barrier_the_level_is_the_solvers(asked) -> None:
    """In the solver's bands the level behind a long barrier is that of the solver
    alone; above them the paths stand as they are without it."""
    room = Room.from_scene(scene(band=250))
    paths = early_paths(room)
    calls = list(asked)
    without = early_paths(Room.from_scene(scene(band=250, rows="edges")))
    solver = solver_of(room)
    assert [group.tolist() for group in solver.groups] == [[0], [1], [2]]
    own = own_level(room, solver, [0, 1, 2], None, 0.5, 25.0, 1.5, [(5.0, 5.0, 3.0)])
    # The solver's own level, but for the air along the string over the top, which is
    # 0.63 m longer than the straight line.
    np.testing.assert_allclose(levels(room, paths)[:3], own, atol=0.01)
    np.testing.assert_allclose(levels(room, paths)[3:], levels(room, without)[3:], atol=1e-9)
    assert [call["obstacles"] for call in calls] == [[(5.0, 5.0, 3.0)]] * 3
    assert all(call["impedance"] is None and call["slants"] is None for call in calls)
    # One path carries it; the row's paths have no energy left in those bands.
    carrier = np.flatnonzero(paths.edge == ROW)
    assert len(carrier) == 1 and len(paths) == len(without) + 1
    others = np.delete(np.arange(len(paths)), carrier)
    assert not paths.energy[others][:, :3].any()
    np.testing.assert_array_equal(paths.energy[others][:, 3:], without.energy[:, 3:])
    assert not paths.energy[carrier][:, 3:].any()


def test_the_solvers_path_follows_the_string_over_the_tops() -> None:
    room = Room.from_scene(scene())
    paths = early_paths(room)
    i = int(np.flatnonzero(paths.edge == ROW)[0])
    top = np.array([5.0, 0.0, 3.0])
    string = np.linalg.norm(top - SOURCE) + np.linalg.norm(np.array(LISTENER) - top)
    np.testing.assert_allclose(paths.points[i], [SOURCE, top, LISTENER], atol=1e-9)
    assert paths.distance_m[i] == pytest.approx(string)
    assert paths.delay_s[i] == pytest.approx(string / room.speed_of_sound())
    # It arrives from the top, and counts as a path over an edge of order 0.
    towards = (top - LISTENER) / np.linalg.norm(top - LISTENER)
    np.testing.assert_allclose(paths.direction[i], towards, atol=1e-12)
    assert paths.order[i] == 0 and paths.diffracted[i] and len(paths.faces[i]) == 0


def _two_buildings(back: float) -> list:
    """Two buildings 6 m high, 500 m long, 2 m apart: from ``back`` to 4 and 6 to 10."""
    return [
        {"name": "a", "footprint": [[back, -250], [4, -250], [4, 250], [back, 250]], "height": 6},
        {"name": "b", "footprint": [[6, -250], [10, -250], [10, 250], [6, 250]], "height": 6},
    ]


def test_a_building_under_the_source_reaches_back_to_its_wall(asked) -> None:
    """The row is given as the solver takes it: a building under the source from its
    wall behind the source. The path over that wall's top, which the way does not
    cross, stays beside the solver's."""
    row = [(-6.0, 4.0, 6.0), (6.0, 10.0, 6.0)]
    on_roof = {"listener": (15.0, 0.0, 5.5), "barriers": (), "source": (0.0, 0.0, 6.5)}
    room = Room.from_scene(scene(buildings=_two_buildings(-6.0), **on_roof))
    paths = early_paths(room)
    assert [call["obstacles"] for call in asked] == [row] * 2
    # The one path left with energy in the solver's bands beside its own goes over the
    # edge of the roof 6 m behind the source, 27 dB and more under the solver's.
    left = np.flatnonzero(paths.energy[:, :2].any(axis=1) & (paths.edge != ROW))
    assert len(left) == 1 and len(paths.faces[left[0]]) == 0
    np.testing.assert_allclose(paths.points[left[0]][1], [-6.0, 0.0, 6.0], atol=1e-9)
    carrier = np.flatnonzero(paths.edge == ROW)[0]
    assert np.all(paths.energy[left[0], :2] < 10.0**-2.7 * paths.energy[carrier, :2])
    own = own_level(room, solver_of(room), [0, 1], None, 6.5, 15.0, 5.5, row)
    np.testing.assert_allclose(levels(room, paths)[:2], own, atol=0.1)


def test_paths_between_the_rows_own_walls_give_way() -> None:
    """The edges' terms that reflect between the two walls of the gap between the
    buildings are the row's, and give way to the solver's sound (the scene is
    validation/rows.py's two roofs on a ground, from the street up to roof level:
    left beside the solver's sound, with reflections up to the third order, they
    raised its 125 Hz band by 5.6 dB)."""
    street = {"listener": (25.0, 0.0, 6.01), "barriers": (), "source": (0.0, 0.0, 1.51)}
    buildings = [
        {"name": "a", "footprint": [[10, -250], [14, -250], [14, 250], [10, 250]], "height": 6},
        {"name": "b", "footprint": [[16, -250], [20, -250], [20, 250], [16, 250]], "height": 6},
    ]
    row = [(10.0, 14.0, 6.0), (16.0, 20.0, 6.0)]
    room = Room.from_scene(scene(buildings=buildings, **street))
    paths = early_paths(room)
    without = early_paths(Room.from_scene(scene(buildings=buildings, rows="edges", **street)))
    solver = solver_of(room)
    model = solver.model
    upright = np.abs(model.plane_normal[model.face_plane][:, 2]) < 1e-9

    def off_walls(found) -> list:
        return [i for i, f in enumerate(found.faces) if upright[np.asarray(f, dtype=int)].any()]

    assert off_walls(without) and without.energy[off_walls(without)][:, :2].any()
    assert off_walls(paths) and not paths.energy[off_walls(paths)][:, :2].any()
    own = own_level(room, solver, [0, 1], None, 1.51, 25.0, 6.01, row)
    np.testing.assert_allclose(levels(room, paths)[:2], own, atol=0.01)


def test_a_wall_past_the_listener_keeps_its_reflection() -> None:
    """A wall that the way does not cross is not the row's: the paths off it stay."""
    behind = {"name": "c", "footprint": [[30, -250], [34, -250], [34, 250], [30, 250]], "height": 9}
    room = Room.from_scene(scene(buildings=[behind]))
    paths = early_paths(room)
    solver = solver_of(room)
    own = own_level(room, solver, [0, 1], None, 0.5, 25.0, 1.5, [(5.0, 5.0, 3.0)])
    with_wall = levels(room, paths)[:2]
    assert np.all(np.abs(with_wall - own) > 0.3)
    model = solver.model
    upright = np.abs(model.plane_normal[model.face_plane][:, 2]) < 1e-9
    off_wall = [
        i for i, faces in enumerate(paths.faces) if upright[np.asarray(faces, dtype=int)].any()
    ]
    assert off_wall and paths.energy[off_wall][:, :2].all()


def test_a_way_at_a_slant_gives_the_solver_the_slant(asked) -> None:
    listener = (25.0, 25.0 * math.tan(math.radians(30.0)), 1.5)
    room = Room.from_scene(scene(listener=listener))
    paths = early_paths(room)
    calls = list(asked)
    reach = 25.0 / math.cos(math.radians(30.0))
    assert len(calls) == 2
    for call in calls:
        (near, far, top), (slant,) = call["obstacles"][0], call["slants"]
        assert (near, far, top) == pytest.approx((5.0 / math.cos(math.radians(30.0)),) * 2 + (3.0,))
        assert call["ends"] == pytest.approx((0.5, reach, 1.5))
        # The barrier's line is turned clockwise from across the way: negative.
        assert slant == pytest.approx(-math.radians(30.0))
    row = (None, 0.5, reach, 1.5, calls[0]["obstacles"])
    own = own_level(room, solver_of(room), [0, 1], *row, slants=calls[0]["slants"])
    np.testing.assert_allclose(levels(room, paths)[:2], own, atol=0.01)


def test_near_a_barriers_end_the_paths_take_over() -> None:
    """Where the way crosses a barrier within reach of its end, the solver's share is
    what the end's taper leaves, band by band, and the row's paths have the rest in
    pressure; from two thirds of the reach on the paths stand alone."""
    short = {"name": "screen", "path": [[10, -15], [10, 15]], "height": 3}
    shares = []
    for y in (0.0, 12.0, 22.0):
        listener = (20.0, y, 1.5)
        room = Room.from_scene(scene(listener=listener, barriers=[short], band=250))
        solver = solver_of(room)
        reach = math.hypot(20.0, y)
        towards = np.array([20.0, y]) / reach
        share = solver._share(np.array(listener), towards, reach)
        # By hand: the shortest way over the top's line touches it where it divides
        # the ends' offset along it as their distances from the line.
        from_source, from_listener = math.hypot(10.0, 2.5), math.hypot(10.0, 1.5)
        touches = y * from_source / (from_source + from_listener)
        at = np.array([10.0, touches, 3.0])
        r1 = float(np.linalg.norm(at - SOURCE))
        r2 = float(np.linalg.norm(np.array(listener) - at))
        k = 2.0 * np.pi * solver.frequencies[:3].min(axis=1) / room.speed_of_sound()
        follows = _fade_reach(r1, r2, from_source / r1, k)
        expected = 1.0 - fade_window(15.0 - touches, follows)
        np.testing.assert_allclose(share[:3], expected, atol=1e-12)
        assert not share[3:].any()
        shares.append(share[:3])
        paths = early_paths(room)
        without = early_paths(
            Room.from_scene(scene(listener=listener, barriers=[short], band=250, rows="edges"))
        )
        carrier = paths.edge == ROW
        if not share.any():
            assert not carrier.any()
            np.testing.assert_array_equal(paths.energy, without.energy)
            continue
        assert carrier.sum() == 1
        # The row's paths keep (1 - share)^2 of their energy; here they are all the
        # paths there are but those round the barrier's ends.
        kept = paths.energy[~carrier][:, :3]
        whole = without.energy[:, :3]
        row = np.flatnonzero(np.any(np.abs(kept - whole) > 1e-15 * whole.max(), axis=1))
        assert len(row)
        np.testing.assert_allclose(kept[row], whole[row] * (1.0 - share[:3]) ** 2, rtol=1e-12)
    shares = np.array(shares)
    # In the middle of the barrier the solver has every band; the lowest band's reach
    # is the longest, so it gives way first; 4 m from the end no band is the solver's.
    assert np.all(shares[0] == 1.0)
    assert shares[1, 0] == 0.0 and 0.0 < shares[1, 1] < 1.0 and shares[1, 2] == 1.0
    assert not shares[2].any()
    assert FADE_TAPER / FADE_REACH == pytest.approx(2.0 / 3.0)


def test_a_band_that_the_solver_refuses_keeps_its_paths(monkeypatch) -> None:
    real = rows.over_row

    def over_row(frequencies, *args, **more):
        if np.min(frequencies) > 150.0:
            raise rows.OutOfReach("not this band")
        return real(frequencies, *args, **more)

    monkeypatch.setattr(rows, "over_row", over_row)
    room = Room.from_scene(scene(band=250))
    paths = early_paths(room)
    without = early_paths(Room.from_scene(scene(band=250, rows="edges")))
    carrier = paths.edge == ROW
    assert carrier.sum() == 1
    # 63 and 125 Hz are the solver's; 250 Hz, refused, has its paths.
    assert not paths.energy[~carrier][:, :2].any() and paths.energy[carrier][:, :2].all()
    assert not paths.energy[carrier][:, 2:].any()
    np.testing.assert_array_equal(paths.energy[~carrier][:, 2:], without.energy[:, 2:])
    np.testing.assert_allclose(levels(room, paths)[2:], levels(room, without)[2:], atol=1e-9)
    solver = solver_of(room)
    solver.apply(np.array(LISTENER), without)
    assert (solver.taken, solver.refused) == (2, 1)


def test_where_the_solver_is_not_asked() -> None:
    def of(data: Scene, order: int = 2, mode: str = "solver"):
        room = Room.from_scene(data)
        model = TraceModel.from_room(room)
        settings = {
            "speed_of_sound": room.speed_of_sound(),
            "air": room.air_attenuation(model.bands),
            "coherent": coherent_bands(room.scene.simulation, model.absorption.shape[1]),
        }
        return RowPaths.of(model, np.array(SOURCE), mode, order, **settings)

    assert of(scene()) is not None
    assert of(scene(), mode="edges") is None
    assert of(scene(atmosphere=WINDY)) is not None
    # The direct sound alone, no barrier or building, no band added in pressure.
    assert of(scene(), order=0) is None
    assert of(scene(barriers=())) is None
    assert of(scene(band=None)) is None
    # Without edge diffraction there are no paths over edges to stand for.
    room = Room.from_scene(scene(diffraction="none"))
    assert not np.any(early_paths(room).edge == ROW)
    # Nothing on the way, and the listener inside the reach of nothing: the paths.
    beside = Room.from_scene(scene(listener=(3.0, 20.0, 1.5)))
    assert not np.any(early_paths(beside).edge == ROW)
    assert MAX_BAND_HZ == 500


def test_third_octave_bands_share_their_octaves_run(asked) -> None:
    room = Room.from_scene(scene(bands="third_octave", band=63))
    paths = early_paths(room)
    solver = solver_of(room)
    # 50, 63 and 80 Hz: the first three of the 24 bands, three frequencies each.
    assert [group.tolist() for group in solver.groups][:1] == [[0, 1, 2]]
    assert len(asked) == 1 and asked[0]["frequencies"].shape == (9,)
    # The octave's nine frequencies in one run, as the simulation asks for them.
    own = own_level(room, solver, [slice(0, 3)], None, 0.5, 25.0, 1.5, [(5.0, 5.0, 3.0)])
    np.testing.assert_allclose(levels(room, paths)[:3], own[0], atol=0.01)


def test_the_ground_has_its_impedance_and_roofs_are_rigid(asked) -> None:
    grass = {"impedance": "grass"}
    room = Room.from_scene(scene(ground=grass, band=63))
    paths = early_paths(room)
    solver = solver_of(room)
    (call,) = asked
    impedance = call["impedance"]
    face = solver._grounds_along(np.array([1.0, 0.0]), 25.0)(12.0)
    np.testing.assert_allclose(impedance(12.0), 1.0 / np.conj(solver._admittance(face)[0]))
    own = own_level(room, solver, [0], impedance, 0.5, 25.0, 1.5, [(5.0, 5.0, 3.0)])
    np.testing.assert_allclose(levels(room, paths)[:1], own, atol=0.01)
    # A ground given by its absorption alone has the admittance that gives it.
    asked.clear()
    soft = {"absorption": [0.1] * 8, "scattering": 0.0}
    room = Room.from_scene(scene(ground=soft, band=63))
    assert np.any(early_paths(room).edge == ROW)
    (call,) = asked
    fitted = fit_admittance([0.1] * 8, 8, None)(solver_of(room).frequencies)[0]
    assert np.all(np.abs(fitted) > 0.0)
    np.testing.assert_allclose(call["impedance"](12.0), 1.0 / np.conj(fitted))
    # A roof that absorbs is rigid to the solver; one with an impedance is not asked.
    house = {"name": "a", "footprint": [[5, -250], [9, -250], [9, 250], [5, 250]], "height": 3}
    asked.clear()
    absorbing = {"absorption": [0.3] * 8, "scattering": 0.0}
    room = Room.from_scene(scene(barriers=(), buildings=[house], roofs=absorbing, band=63))
    assert np.any(early_paths(room).edge == ROW)
    assert [call["obstacles"] for call in asked] == [[(5.0, 9.0, 3.0)]]
    asked.clear()
    room = Room.from_scene(scene(barriers=(), buildings=[house], roofs=grass, band=63))
    assert not np.any(early_paths(room).edge == ROW) and not asked


def test_a_simulation_counts_the_solvers_runs() -> None:
    room = Room.from_scene(scene())
    said = []
    result = simulate(room, progress=said.append)
    assert any("whole rows by the solver" in line for line in said)
    info = result.details["s"]
    assert info["rows_solved"] == 2.0 and info["rows_left_to_edges"] == 0.0
    early = result.responses[("s", "l")].early
    assert np.sum(early.edge == ROW) == 1
    quiet = simulate(Room.from_scene(scene(rows="edges")))
    assert "rows_solved" not in quiet.details["s"]
    assert not np.any(quiet.responses[("s", "l")].early.edge == ROW)


def test_maps_take_the_paths_over_the_edges(asked) -> None:
    small = {
        "barriers": [{"name": "screen", "path": [[5, -15], [5, 15]], "height": 3}],
        "listener": (12.0, 0.0, 1.5),
        "extent": ((-4.0, -16.0), (20.0, 16.0)),
        "band": 500,
    }
    # A listener's paths ask the solver here; the map's points do not.
    assert np.any(early_paths(Room.from_scene(scene(**small))).edge == ROW) and asked
    asked.clear()
    settings = Coverage(spacing=8.0, rays=200)
    maps = [
        coverage_map(Room.from_scene(scene(**small, rows=mode)), settings=settings)
        for mode in ("solver", "edges")
    ]
    assert len(maps[0].grid) > 0 and not asked
    for name, values in maps[0].get().values.items():
        np.testing.assert_array_equal(values, maps[1].get().values[name])


def test_walks_take_the_paths_over_the_edges(asked) -> None:
    small = {
        "barriers": [{"name": "screen", "path": [[5, -15], [5, 15]], "height": 3}],
        "listener": (12.0, 0.0, 1.5),
        "extent": ((-4.0, -16.0), (20.0, 16.0)),
        "band": 500,
    }
    data = scene(**small)
    # Standing, the listener's paths ask the solver; walking, none of its places does.
    assert np.any(early_paths(Room.from_scene(data)).edge == ROW) and asked
    asked.clear()
    walking = data.model_dump()
    walking["listeners"][0]["trajectory"] = {"to": [[12.0, 1.0, 1.5]], "speed_m_s": 1.0}
    room = Room.from_scene(Scene.model_validate(walking))
    walked = simulate_trajectory(room, step_m=0.5)
    assert len(walked.names) == 3 and not asked
    simulate_movement(room, step_m=0.5)
    assert not asked


def test_a_line_s_crossings_have_the_slants_of_the_sides_it_crosses() -> None:
    """A barrier and a building's near wall turned 26.57 degrees clockwise from across
    the x axis, and the building's far wall across it: the same slants from either end
    of the line, and none (nan) for the side of a building that the line starts in."""
    turned = -math.atan2(10.0, 20.0)
    layout = Layout(
        (-100.0, -100.0, 100.0, 100.0),
        0.0,
        30.0,
        buildings=(("b", np.array([[20.0, -10], [40, -10], [40, 10], [30, 10]]), 6.0),),
        barriers=(("s", np.array([[5.0, -10.0, 3.0], [15.0, 10.0, 3.0]])),),
    )
    there = layout.crossed_along(np.zeros(2), np.array([1.0, 0.0]), 50.0)
    assert [found[:3] for found in there] == [(10.0, 10.0, 3.0), (25.0, 40.0, 6.0)]
    assert there[0][3] == pytest.approx((turned, turned))
    assert there[1][3] == pytest.approx((turned, 0.0))
    assert layout.obstacles_along(np.zeros(2), np.array([1.0, 0.0]), 50.0) == [
        found[:3] for found in there
    ]
    back = layout.crossed_along(np.array([50.0, 0.0]), np.array([-1.0, 0.0]), 50.0)
    assert [found[:3] for found in back] == [(10.0, 25.0, 6.0), (40.0, 40.0, 3.0)]
    assert back[0][3] == pytest.approx((0.0, turned))
    assert back[1][3] == pytest.approx((turned, turned))
    inside = layout.crossed_along(np.array([35.0, 0.0]), np.array([1.0, 0.0]), 50.0)
    assert [found[:3] for found in inside] == [(0.0, 5.0, 6.0)]
    assert math.isnan(inside[0][3][0]) and inside[0][3][1] == pytest.approx(0.0)


def test_an_end_s_effect_counts_whole_up_to_two_thirds_of_its_reach() -> None:
    """All of it up to FADE_TAPER / FADE_REACH of the reach, half of it midway from
    there to the reach, none from the reach on, and never more further off."""
    reach = 9.0
    np.testing.assert_array_equal(fade_window(np.array([0.0, 3.0, 6.0]), reach), 1.0)
    assert fade_window(7.5, reach) == pytest.approx(0.5)
    np.testing.assert_array_equal(fade_window(np.array([9.0, 9.5, 40.0]), reach), 0.0)
    assert np.all(np.diff(fade_window(np.linspace(0.0, 12.0, 241), reach)) <= 0.0)


def test_in_wind_the_weather_s_effect_scales_the_solvers_path() -> None:
    """Behind a long barrier in wind the solver's path is one of the plane's: it and
    the paths over the top carry their energies in still air times the weather's
    effect, the same for all of them in a band, with their spectra as they were, and
    come as much sooner. So the wind changes the level by as much with the solver as
    with the paths over the edges."""
    place = {"listener": (40.0, 0.0, 1.5), "band": 500, "barrier_frequencies": 8}
    found = {
        (rows, wind): early_paths(
            Room.from_scene(scene(**place, rows=rows, atmosphere=WINDY if wind else None))
        )
        for rows in ("solver", "edges")
        for wind in (False, True)
    }
    still, windy = found["solver", False], found["solver", True]
    carrier = int(np.flatnonzero(still.edge == ROW)[0])
    plane = [i for i in range(len(still)) if abs(still.points[i][-2][1]) < 1e-6]
    assert carrier in plane and len(plane) >= 5
    # The same paths in the wind, in whatever order: by what they bend over and how far.
    again = [
        int(
            np.flatnonzero(
                (windy.edge == still.edge[i]) & (windy.distance_m == still.distance_m[i])
            )[0]
        )
        for i in plane
    ]
    assert len(set(again)) == len(plane) and len(windy) == len(still)
    assert np.all(still.energy[carrier, :4] > 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = windy.energy[again] / still.energy[plane]
    for band in range(effect.shape[1]):
        held = still.energy[plane][:, band] > 0.0
        np.testing.assert_allclose(effect[held, band], effect[held, band][0], rtol=1e-12)
    # The wind does change the solver's path.
    assert np.all(np.abs(10 * np.log10(effect[plane.index(carrier), :4])) > 0.05)
    np.testing.assert_array_equal(windy.spectrum[again], still.spectrum[plane])
    shift = windy.delay_s[again] - still.delay_s[plane]
    assert np.ptp(shift) < 1e-12 and shift[0] < -1e-5  # downwind: sooner
    changes = [
        10 * np.log10(total_energy(found[rows, True]) / total_energy(found[rows, False]))
        for rows in ("solver", "edges")
    ]
    np.testing.assert_allclose(changes[0], changes[1], atol=1e-9)


def test_in_wind_closer_than_the_equation_goes_the_solvers_path_keeps_its_legs() -> None:
    """A source and a listener 10 m up, 20 m apart either side of a barrier 12 m high:
    nearer than the weather's runs go (``refracted.WAVE_MIN_RANGE`` times their heights
    together), so no run holds the row and the solver's path, like the paths over the
    edges there, keeps its level in still air (no leg of it is in a shadow) and takes
    the travel time of its legs in the wind."""
    high = {
        "source": (0.0, 0.0, 10.0),
        "listener": (20.0, 0.0, 10.0),
        "barriers": [{"name": "screen", "path": [[10, -150], [10, 150]], "height": 12}],
        "band": 500,
    }
    still = early_paths(Room.from_scene(scene(**high)))
    windy = early_paths(Room.from_scene(scene(**high, atmosphere=WINDY)))
    assert 20.0 < WAVE_MIN_RANGE * (10.0 + 10.0)
    carrier = int(np.flatnonzero(still.edge == ROW)[0])
    again = int(np.flatnonzero(windy.edge == ROW)[0])
    np.testing.assert_allclose(windy.energy[again], still.energy[carrier], rtol=1e-12)
    np.testing.assert_array_equal(windy.spectrum[again], still.spectrum[carrier])
    assert -1e-3 < windy.delay_s[again] - still.delay_s[carrier] < -1e-6


def test_in_wind_beside_a_low_wall_the_direct_sound_keeps_its_own_change() -> None:
    """A wall 1 m high between a source and a listener 1.5 m up and 4 m apart, who see
    each other and stand nearer than the weather's runs go: the solver's path is none
    of the specular paths, so the ray of the direct sound in the wind takes the direct
    sound's own share, and the wind changes the level as little as it does with the
    paths over the edges. Taken for the direct sound, the solver's path gave that ray
    its own energy, in the bands above the solver's none."""
    low = {
        "source": (0.0, 0.0, 1.5),
        "listener": (4.0, 0.0, 1.5),
        "barriers": [{"name": "wall", "path": [[2, -150], [2, 150]], "height": 1}],
        "band": 500,
    }
    assert 4.0 < WAVE_MIN_RANGE * (1.5 + 1.5)
    changes = {}
    for mode in ("solver", "edges"):
        still = early_paths(Room.from_scene(scene(**low, rows=mode)))
        windy = early_paths(Room.from_scene(scene(**low, rows=mode, atmosphere=WINDY)))
        assert np.any(still.edge == ROW) == np.any(windy.edge == ROW) == (mode == "solver")
        changes[mode] = 10 * np.log10(total_energy(windy) / total_energy(still))
    assert np.all(np.abs(changes["edges"]) < 0.05)
    np.testing.assert_allclose(changes["solver"], changes["edges"], atol=0.05)


def test_sound_through_a_barrier_that_lets_sound_through_stays() -> None:
    """A fence with a transmission loss of 3 dB: the direct sound and the ground's
    reflection come through it and stay as they are beside the solver's sound, whose
    barriers let none through. One screen's level is then within 0.35 dB of the paths'
    in the solver's bands, where without that sound it is that of a screen that lets
    nothing through."""
    fence = {"scattering": 0.0, "transmission_loss": 3.0}
    room = Room.from_scene(scene(screens=fence, band=500))
    paths = early_paths(room)
    without = early_paths(Room.from_scene(scene(screens=fence, band=500, rows="edges")))
    through = np.flatnonzero(without.crossings > 0)
    assert len(through) == 2 and np.all(without.edge[through] == -1)
    assert np.any(paths.edge == ROW)
    np.testing.assert_array_equal(paths.crossings[through], without.crossings[through])
    np.testing.assert_array_equal(paths.energy[through], without.energy[through])
    np.testing.assert_allclose(levels(room, paths)[:4], levels(room, without)[:4], atol=0.35)
    closed = levels(Room.from_scene(scene(band=500)))
    assert levels(room, paths)[0] > closed[0] + 3.0


def test_a_result_with_the_solvers_path_is_traced_again_for_another_ground() -> None:
    """The solver's sound depends on the absorption of a ground without an impedance,
    through the admittance fitted to it, and its path reflects off no face that a
    re-weighting counts: a result that has it is not re-weighted for another absorption
    of the ground. For another absorption of the barrier, which the solver holds rigid,
    it is, as a result with the paths over the edges is for either."""

    def of(ground: float, screen: float, mode: str) -> Scene:
        soil = {"absorption": [ground] * 8, "scattering": 0.0}
        fence = {"absorption": [screen] * 8, "scattering": 0.0}
        return scene(ground=soil, screens=fence, band=500, rows=mode)

    for mode in ("solver", "edges"):
        traced = of(0.05, 0.1, mode)
        room = Room.from_scene(traced)
        face_group, materials = reweight.groups_of(room)
        result = simulate(room, groups=face_group)
        other_ground = reweight.reweighted_result(
            result, traced, face_group, materials, of(0.5, 0.1, mode)
        )
        other_screen = reweight.reweighted_result(
            result, traced, face_group, materials, of(0.05, 0.3, mode)
        )
        assert other_screen is not None
        if mode == "solver":
            assert reweight.solver_grounds(result) == ["soil"]
            assert other_ground is None
        else:
            assert reweight.solver_grounds(result) == []
            assert other_ground is not None


def test_the_solvers_path_has_the_solvers_phase() -> None:
    """The early paths' time is exp(j omega t) and the solver's exp(-i omega t): the
    path's pressure at each frequency, its spectrum at its delay, is the conjugate of
    the solver's, to a factor above 0."""
    room = Room.from_scene(scene(band=250))
    paths = early_paths(room)
    solver = solver_of(room)
    i = int(np.flatnonzero(paths.edge == ROW)[0])
    for band in range(3):
        f = solver.frequencies[band]
        own = rows.over_row(f, room.speed_of_sound(), None, 0.5, 25.0, 1.5, [(5.0, 5.0, 3.0)])
        mine = paths.spectrum[i, band] * np.exp(-2j * np.pi * f * paths.delay_s[i])
        np.testing.assert_allclose(np.angle(mine * own.pressure), 0.0, atol=1e-6)
        assert np.abs(np.angle(mine * np.conj(own.pressure))).max() > 0.5


def test_a_wall_behind_the_source_keeps_its_reflections() -> None:
    """A building behind the source is not the row's: the paths off its wall, those
    that go on over the barrier's top among them, keep the energy that they have
    without the solver."""
    house = {
        "name": "h",
        "footprint": [[-10, -250], [-6, -250], [-6, 250], [-10, 250]],
        "height": 8,
    }
    place = {"buildings": [house], "source": (-3.0, 0.0, 0.5)}
    paths = early_paths(Room.from_scene(scene(**place)))
    without = early_paths(Room.from_scene(scene(**place, rows="edges")))
    off_the_wall = [
        i for i in range(len(without)) if np.any(np.abs(without.points[i][1:-1, 0] + 6.0) < 1e-9)
    ]
    over_the_top = [i for i in off_the_wall if without.edge[i] >= 0]
    assert len(off_the_wall) >= 3 and over_the_top
    carrier = np.flatnonzero(paths.edge == ROW)
    assert len(carrier) == 1 and len(paths) == len(without) + 1
    kept = np.delete(paths.energy, carrier, axis=0)[off_the_wall]
    np.testing.assert_array_equal(kept, without.energy[off_the_wall])
    assert np.all(kept[:, :2] > 0.0)


def test_the_solvers_path_leaves_a_directional_source_towards_the_first_top() -> None:
    """The solver's source radiates alike all round. The path that carries its sound
    takes the source's directivity in the direction it leaves the source in, towards
    the first top (26.6 degrees up here) and not towards the listener (2.3 degrees): for
    a cardioid that faces along x, 3/4 (1 + cos)^2 of the energy."""

    def carried(directivity: str | None) -> np.ndarray:
        data = scene().to_data()
        if directivity is not None:
            data["sources"][0]["directivity"] = directivity
        paths = early_paths(Room.from_scene(Scene.model_validate(data)))
        return paths.energy[int(np.flatnonzero(paths.edge == ROW)[0]), :2]

    cos = 5.0 / math.hypot(5.0, 2.5)
    np.testing.assert_allclose(
        carried("cardioid") / carried(None), 0.75 * (1.0 + cos) ** 2, rtol=1e-9
    )
