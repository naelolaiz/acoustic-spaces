"""Grazing incidence over the audience: the fit of Jurkiewicz et al. (2025), and where it
applies: the last leg of early paths low over audience surfaces, in the mid bands."""

import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import TraceModel, coverage_map
from acoustic_spaces.engine.grazing import (
    CLEARANCE_M,
    MAX_ELEVATION_RAD,
    STEP_M,
    attenuation,
    audience_faces,
    fit_bands,
)
from acoustic_spaces.engine.simulate import early_paths


def test_the_fit_of_grazing_attenuation() -> None:
    # The fit's own numbers: 17 dB at grazing incidence over 11 m (row 12), 0.6 dB per
    # metre at 5 degrees, about 5 dB at row 12 at 7 degrees, nothing from 17.8 degrees.
    db = 10 * np.log10(attenuation(np.radians([0, 5, 7, 17.8, 40]), 11.0))
    np.testing.assert_allclose(db[:3], [-16.99, -6.29, -4.75], atol=0.01)
    assert db[3] == db[4] == 0.0
    assert math.degrees(MAX_ELEVATION_RAD) == pytest.approx(17.78, abs=0.01)
    # Linear in dB with distance, the first metre free.
    assert attenuation(0.05, 1.0) == 1.0 == attenuation(0.05, 0.4)
    db = 10 * np.log10(attenuation(0.05, [6.0, 11.0]))
    np.testing.assert_allclose(db, [-4.32, -8.63], atol=0.01)


def _scene(tmp_path: Path, audience: bool, listener, panel: bool = False) -> Scene:
    """A 30 x 10 x 8 m hall with seats on its floor (and, with ``panel``, a balcony: a
    seated 6 x 10 m slab 5 m up at the back), a talker at the front."""
    geometry = {"box": [30, 10, 8]}
    assignments = {"floor": "seats", "ceiling": "wall", "walls": "wall"}
    if panel:
        # The hall's shell, then the balcony slab (seated on both sides, for simplicity).
        corners = [(x, y, z) for x in (0, 30) for y in (0, 10) for z in (0, 8)]
        lines = [f"v {x} {y} {z}" for x, y, z in corners]
        lines += ["g floor", "f 1 3 7 5", "g ceiling", "f 2 6 8 4", "g walls"]
        lines += ["f 1 2 4 3", "f 5 7 8 6", "f 1 5 6 2", "f 3 4 8 7"]
        lines += [f"v {x} {y} 5" for x, y in ((24, 0), (30, 0), (30, 10), (24, 10))]
        lines += ["g balcony", "f 9 10 11 12"]
        (tmp_path / "hall.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
        geometry = {"file": str(tmp_path / "hall.obj")}
        assignments["balcony"] = "seats"
    return Scene.model_validate(
        {
            "geometry": geometry,
            "materials": {
                "seats": {"absorption": [0.6] * 8, "scattering": 0.7, "audience": audience},
                "wall": {"absorption": [0.1] * 8, "scattering": 0.1},
            },
            "assignments": assignments,
            "sources": [{"name": "talker", "position": [2.0, 5.0, 1.5]}],
            "listeners": [{"name": "seat", "position": listener}],
            # 125 Hz: the grazing ratios are of energies, not of pressures.
            "simulation": {
                "early": {"max_order": 1, "coherent_max_band_hz": 125},
                "ir_length_s": 0.3,
            },
        }
    )


def _direct_and_ceiling(tmp_path: Path, audience: bool, listener, panel: bool = False):
    room = Room.from_scene(_scene(tmp_path, audience, listener, panel))
    paths = early_paths(room)
    direct = paths.energy[paths.order == 0][0]
    # The ceiling reflection arrives from high up: steeper than the fit's 17.8 degrees.
    ceiling = room.mesh.labels.index(("ceiling",))
    up = [k for k, f in enumerate(paths.faces) if len(f) and room.mesh.face_labels[f[0]] == ceiling]
    return room, direct, paths.energy[up[0]]


def test_sound_low_over_the_seats_is_attenuated(tmp_path: Path) -> None:
    listener = [20.0, 5.0, 1.2]
    room, direct, ceiling = _direct_and_ceiling(tmp_path, True, listener)
    _, plain_direct, plain_ceiling = _direct_and_ceiling(tmp_path, False, listener)
    model = TraceModel.from_room(room)
    bands = fit_bands(model)
    assert [b for b, on in zip(model.band_hz, bands, strict=True) if on] == [500, 1000, 2000]
    assert audience_faces(model).sum() == 2  # the floor's two triangles
    # The direct sound runs 18 m over the seats at 0.95 degrees up from the floor.
    elevation = math.atan2(0.3, 18.0)
    expected = attenuation(elevation, 18.0)
    ratio = direct / plain_direct
    np.testing.assert_allclose(ratio[bands], expected, rtol=0.05)
    np.testing.assert_allclose(ratio[~bands], 1.0)
    assert 10 * np.log10(expected) == pytest.approx(-21.1, abs=0.1)
    np.testing.assert_allclose(ceiling, plain_ceiling)

    # With the statistical tail, what the seats take from the early paths is lost: the
    # tail is the same with them as without.
    totals, early = [], []
    for audience in (True, False):
        data = _scene(tmp_path, audience, listener).model_dump(mode="json")
        data["simulation"].update(late={"method": "statistical"}, transition_ms=300.0)
        room = Room.from_scene(Scene.model_validate(data))
        totals.append(simulate(room)["talker", "seat"].energy().sum(axis=1))
        early.append(early_paths(room).energy.sum(axis=0))
    # (The filters of the low bands, where the paths add in pressure, hear a trace of
    # the mid bands.)
    lost = early[1] - early[0]
    np.testing.assert_allclose(totals[1] - totals[0], lost, rtol=1e-6, atol=1e-3 * lost.max())
    assert np.all((totals[1] - totals[0])[bands] > 0.0)


def test_a_balcony_seat_is_not_shaded_by_the_stalls(tmp_path: Path) -> None:
    # A seat on the balcony, 5 m above the stalls: the direct sound rises at 10 degrees.
    # It runs low over the front stalls, up to 2 m above the seat's own height over its
    # seats (x = 11.4 m), then high over the rest, which do not count, and 4 m over the
    # balcony's own seats.
    listener = [28.0, 5.0, 6.2]
    room, direct, _ = _direct_and_ceiling(tmp_path, True, listener, panel=True)
    _, plain, _ = _direct_and_ceiling(tmp_path, False, listener, panel=True)
    assert len(room.room_volumes) == 1  # the slab stands in the hall
    bands = fit_bands(TraceModel.from_room(room))
    elevation = math.atan2(4.7, 26.0)
    stretch = math.hypot(26.0, 4.7) / 26.0  # the leg's length per metre along the hall
    low = 2.0 + (1.2 + CLEARANCE_M - 1.5) * 26.0 / 4.7
    over = (4.0 + low - 2.0) * stretch
    ratio = direct / plain
    np.testing.assert_allclose(ratio[bands], attenuation(elevation, over), rtol=0.02)
    assert np.all(ratio[bands] > attenuation(elevation, over + 2 * STEP_M))
    assert np.all(ratio[bands] > attenuation(elevation, 26.0 * stretch) * 1.5)
    # A listener high above the seats (on a gallery, say) is left alone.
    _, high, _ = _direct_and_ceiling(tmp_path, True, [10.0, 5.0, 3.5])
    _, high_plain, _ = _direct_and_ceiling(tmp_path, False, [10.0, 5.0, 3.5])
    np.testing.assert_allclose(high, high_plain)


def test_coverage_maps_attenuate_grazing_sound(tmp_path: Path) -> None:
    def run(audience: bool):
        data = _scene(tmp_path, audience, [20.0, 5.0, 1.2]).model_dump(mode="json")
        data["simulation"] = {
            "early": {"method": "image_sources", "max_order": 1, "coherent_max_band_hz": 125},
            "late": {"method": "statistical"},
            "ir_length_s": 1.0,
        }
        data["coverage"] = {"spacing": 3.0}
        result = coverage_map(Room.from_scene(Scene.model_validate(data))).get()
        return result.grid.points, result.values["c50"], result.values["g"]

    points, seated, seated_g = run(True)
    _, plain, plain_g = run(False)
    # 6 to 12 m from the talker the direct sound is much weaker in the mid bands over
    # the seats, so the sound less clear (farther on, the reverberation dominates).
    middle = (points[:, 0] > 8.0) & (points[:, 0] < 14.0)
    assert middle.any() and np.all(seated[middle, 3:6] < plain[middle, 3:6] - 0.5)
    # (The 125 Hz band's filter hears a trace of the mid bands: there the paths add in
    # pressure.)
    np.testing.assert_allclose(seated[:, [2, 6, 7]], plain[:, [2, 6, 7]])
    np.testing.assert_allclose(seated[:, 1], plain[:, 1], atol=0.01)
    # The tail does not make up what the seats take, so the sound is weaker there too.
    assert np.all(seated_g[middle, 3:6] < plain_g[middle, 3:6] - 0.1)
    np.testing.assert_allclose(seated_g[:, [2, 6, 7]], plain_g[:, [2, 6, 7]])
    np.testing.assert_allclose(seated_g[:, 1], plain_g[:, 1], atol=0.01)
