"""The stills of an outdoor scene (:mod:`acoustic_spaces.demos.outdoor_pictures`): its
ground by material, its weather, and the view from the side with the sound's rays,
checked against the simulation's own and against hand calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from acoustic_spaces.room import Room
from acoustic_spaces.scene import Material, Scene

pytest.importorskip("PIL")

from acoustic_spaces.demos import outdoor_pictures, scene_pictures  # noqa: E402
from acoustic_spaces.demos.pipeline import TourStop  # noqa: E402

#: Where the band stands, and the listeners downwind over the pond and upwind behind it.
SOURCE = (0.0, 0.0, 5.0)
FAR = (350.0, 0.0, 1.5)
BACK = (-80.0, 0.0, 1.5)


def _scene(wind: float | None = 4.0, zones: bool = True, **more) -> Scene:
    """A field with a pond across it, a band 5 m up at its west end and two listeners:
    ``far`` 350 m east past the pond, ``back`` 80 m west; with a ``wind`` from the west
    (m/s, 10 m up), calmer and cooler over the pond with ``zones``."""
    data = {
        "geometry": {
            "ground": {
                "extent": [[-100, -40], [400, 40]],
                "sky": 40,
                "zones": [
                    {"name": "pond", "polygon": [[50, -40], [300, -40], [300, 40], [50, 40]]}
                ],
            }
        },
        "materials": {
            "field": {"impedance": "grass"},
            "pond_water": {"impedance": "water"},
        },
        "assignments": {"ground": "field", "pond": "pond_water"},
        "sources": [{"name": "band", "position": list(SOURCE)}],
        "listeners": [
            {"name": "far", "position": list(FAR)},
            {"name": "back", "position": list(BACK)},
        ],
        **more,
    }
    if wind is not None:
        data["atmosphere"] = {"wind": {"speed_m_s": wind, "direction_deg": 0}}
        if zones:
            data["atmosphere"]["zones"] = [
                {
                    "name": "over_pond",
                    "polygon": [[50, -40], [300, -40], [300, 40], [50, 40]],
                    "temperature_c": 15,
                    "wind": {"speed_m_s": 2.0},
                }
            ]
    return Scene.model_validate(data)


def test_the_ground_is_drawn_by_its_materials() -> None:
    outdoors = outdoor_pictures.Outdoors(Room.from_scene(_scene()))
    assert outdoors.materials == ["field", "pond_water"]
    assert outdoors.kinds == ["soft", "water"]
    assert [name for name, _ in outdoors.legend()] == ["field", "pond_water"]
    assert outdoors.names == ["outside the zones", "over_pond"]
    assert outdoors.weather_label(1) == "over_pond: wind 2.0 m/s 10 m up, 15 °C at the ground"
    # A ground of asphalt is hard; one given by its absorption, by how much it absorbs.
    hard = Material.model_validate({"impedance": "asphalt"})
    assert outdoor_pictures.ground_kind("road", hard) == "hard"
    soft = Material.model_validate({"absorption": [0.4] * 8})
    assert outdoor_pictures.ground_kind("bark", soft) == "soft"


def test_a_bounce_keeps_what_the_ground_reflects() -> None:
    """A fan ray's bounce keeps |R|² of the plane wave off the ground's impedance at 1 kHz
    (R = (sin ψ - β) / (sin ψ + β) at the grazing angle ψ), or 1 minus the absorption
    coefficient at 1 kHz of a ground given by its absorption."""
    grass = _scene().materials["field"]
    c = 343.0
    for grazing in (0.02, 0.1, 0.5):
        beta = complex(grass.impedance.admittance(np.array([1000.0]), c, np.array([1.0]))[0])
        sine = math.sin(grazing)
        expected = abs((sine - beta) / (sine + beta)) ** 2
        assert outdoor_pictures.reflected(grass, sine, c) == pytest.approx(expected, rel=1e-12)
    plain = Material.model_validate({"absorption": [0.3] * 8})
    assert outdoor_pictures.reflected(plain, 0.1, c) == pytest.approx(0.7)
    assert outdoor_pictures.reflected(None, 0.1, c) == 1.0


def test_in_still_air_the_rays_are_the_direct_sound_and_its_reflection() -> None:
    """In still air the simulation's rays to a listener are the straight line and the
    one off the ground, which meets it where the heights' ratio has it: at d zs / (zs +
    zr) from the source."""
    outdoors = outdoor_pictures.Outdoors(Room.from_scene(_scene(wind=None)))
    rays = outdoors.eigenrays(SOURCE, FAR, still=False)
    assert len(rays) == 2
    distance = math.dist(SOURCE[:2], FAR[:2])
    direct, reflected = sorted(rays, key=lambda ray: float(ray[:, 1].min()), reverse=True)
    assert direct[0] == pytest.approx([0.0, 5.0])
    assert direct[-1] == pytest.approx([distance, 1.5])
    bounce = reflected[np.argmin(reflected[:, 1])]
    assert bounce == pytest.approx([distance * 5.0 / (5.0 + 1.5), 0.0], abs=1e-6)
    assert outdoors.bending(SOURCE, FAR, still=False) == "in still air the sound goes straight"
    # The scene's weather, heard in still air, the same.
    windy = outdoor_pictures.Outdoors(Room.from_scene(_scene()))
    still = windy.eigenrays(SOURCE, FAR, still=True)
    assert len(still) == 2 and all(np.allclose(a, b) for a, b in zip(still, rays, strict=True))


def test_the_rays_bend_down_downwind_and_up_upwind() -> None:
    """Downwind the effective speed of sound grows with height and the direct ray arcs
    over the straight line; upwind it falls and the ray sags under it."""
    outdoors = outdoor_pictures.Outdoors(Room.from_scene(_scene(zones=False)))
    for listener, words, above in ((FAR, "down", True), (BACK, "up", False)):
        assert outdoors.bending(SOURCE, listener, still=False).startswith(
            f"the sound bends {words}"
        )
        rays = outdoors.eigenrays(SOURCE, listener, still=False)
        direct = next(ray for ray in rays if float(ray[1:-1, 1].min(initial=1e9)) > 0.0)
        chord = np.interp(direct[:, 0], [0.0, direct[-1, 0]], [5.0, 1.5])
        middle = slice(1, -1)
        if above:
            assert np.all(direct[middle, 1] > chord[middle])
        else:
            assert np.all(direct[middle, 1] < chord[middle])


def test_the_rays_drawn_are_the_simulation_s(monkeypatch) -> None:
    """The view from the side draws the eigenrays the simulation finds over the open
    ground: the same profile along the way (the weathers' mean, as it rounds their
    shares) and the same rays from it."""
    from acoustic_spaces.engine import refracted, refraction
    from acoustic_spaces.engine.simulate import early_paths

    found = []
    search = refraction.Rays.eigenrays

    def spy(self, zr, distance, max_bounces, max_time=np.inf):
        rays = search(self, zr, distance, max_bounces, max_time)
        found.append((self.profile, self.zs, zr, distance, rays))
        return rays

    class Stop(Exception):
        pass

    def stop(plans):
        raise Stop

    monkeypatch.setattr(refraction.Rays, "eigenrays", spy)
    monkeypatch.setattr(refracted, "solve", stop)
    room = Room.from_scene(_scene())
    with pytest.raises(Stop):
        early_paths(room, "band", "far", max_order=1, diffraction="none")
    profile, zs, zr, distance, rays = next(f for f in found if f[3] > 300.0)
    outdoors = outdoor_pictures.Outdoors(room)
    mine = outdoors.profile(SOURCE, FAR)
    assert np.array_equal(mine.z, profile.z) and np.array_equal(mine.c, profile.c)
    assert (zs, zr, distance) == (5.0, 1.5, pytest.approx(350.0))
    drawn = outdoors.eigenrays(SOURCE, FAR, still=False)
    assert len(drawn) == len(rays)
    simulated = refraction.Rays(profile, zs)
    for k, curve in enumerate(drawn):
        assert np.allclose(curve, simulated.curve(zr, rays, k))


def test_the_stills_of_an_outdoor_scene_show_its_ground_weather_and_side(tmp_path) -> None:
    """Outdoors a still draws the view from the side under the plan, which ``hide``
    leaves out; its caption says through which source and how the sound bends. The
    reverberation time and clarity of ISO 3382-1, which say nothing outdoors, are not
    given there; indoors the clarity is, where something other than speech plays."""
    from acoustic_spaces.room import outline_edges

    scene = _scene()
    room = Room.from_scene(scene)
    report = {
        "bands_hz": [500, 1000],
        "responses": [
            {
                "source": "band",
                "listener": name,
                "spl_a_db": 60.0,
                "parameters": {"t30": [1.0, 1.2], "c80": [2.0, 4.0]},
            }
            for name in ("far", "back")
        ],
    }
    seats = {name: {"lag_s": 0.1} for name in ("far", "back")}
    stops = [TourStop("far", 0.0, 0.0, 5.0), TourStop("back", 5.0, 0.0, 5.0)]

    def drawing(hide=(), data=None):
        return scene_pictures.Pictures(
            data or scene.to_data(),
            outline_edges(room)[0],
            report,
            seats,
            stops,
            playing=["band"],
            room=room,
            hide=hide,
        )

    shown = drawing()
    assert shown.outdoors is not None and not shown.indoor
    lines = [line if isinstance(line, str) else line[0] for line in shown.lines(0)]
    assert not any("T30" in line or "C80" in line for line in lines)
    img = shown.still(0)
    hidden = drawing(hide=("side",)).still(0)
    left, _top, width, _height = scene_pictures.PLAN_BOX
    below = (left, scene_pictures.SIDE_BOTTOM - 60, left + width, scene_pictures.SIDE_BOTTOM)

    def spread(picture) -> float:
        return float(np.asarray(picture.crop(below))[..., :3].std(axis=(0, 1)).max())

    assert spread(img) > 5.0  # the side view
    assert spread(hidden) == 0.0  # the background alone
    # The listener behind the band is seen with the plan's left on the left: the view
    # runs from the band westwards, drawn flipped.
    assert shown.side_source("back")["name"] == "band"
    # Indoors (no ground): T30 and C80, the band not a talker.
    room_data = {k: v for k, v in scene.to_data().items() if k not in ("geometry", "atmosphere")}
    inside = scene_pictures.Pictures(
        room_data, np.zeros((0, 2, 2)), report, seats, stops, playing=["band"]
    )
    lines = [line if isinstance(line, str) else line[0] for line in inside.lines(0)]
    assert "Reverberation time T30 1.1 s (mean of 500 Hz and 1 kHz)" in lines
    assert "Clarity C80 +3.0 dB (mean of 500 Hz and 1 kHz)" in lines
    talking = {**room_data, "sources": [{**room_data["sources"][0], "spectrum": "speech_male"}]}
    inside = scene_pictures.Pictures(
        talking, np.zeros((0, 2, 2)), report, seats, stops, playing=["band"]
    )
    assert not any("C80" in str(line) for line in inside.lines(0))


def test_hide_names_what_a_still_leaves_out(tmp_path) -> None:
    from acoustic_spaces.demos.scene_demo import DEMO

    path = tmp_path / "scene.yaml"
    path.write_text(_scene(wind=None).model_dump_json(exclude_none=True), encoding="utf-8")
    demo = DEMO.at(tmp_path)

    def check(*argv: str) -> str | None:
        return demo.check(demo.parser().parse_args(["video", "--scene", "scene.yaml", *argv]))

    assert "--hide" not in (check("--hide", "side,wind") or "")
    assert check("--hide", "sky") == "--hide sky: give one or more of side, wind, ground"
