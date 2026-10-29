"""A helicopter flying low over the stadium concert halfway through the song, heard at
each listener over the band and the sound system, its pitch falling as it passes
(Doppler); and a video that tours them, as the stadium demo's does.

The helicopter is the flyover demo's (acoustic_spaces.demos.helicopter): an Airbus
Helicopters AS350 B3 at its maximum mass, flying level at 63 m/s, straight over the
stadium's long axis from behind the stage to past the east stand, 1200 m. It flies 150 m
up (--height), as a police or a television helicopter given leave may: without leave,
the EU's rules of the air keep a flight by sight (VFR) over an open-air assembly of
persons 300 m above the highest obstacle within 600 m (SERA.5005(f)), here the
stadium's roof, 30 m up, so --height 330. It is over the centre spot halfway through
the song (--at, seconds into the song as the sound system plays it). Its sound is its
two rotors', sent each way down each early path, as in the helicopter_flyover example;
the scene gives it the sound power and the directivity of that sound for its late
part.

The concert is the stadium demo's: run it first, in the same work folder, up to its
render step (`python stadium_demo.py render`, or `all`), with the --format and the --hrtf
given here. Then, from the folder of the stadium example, in the Python environment
acoustic-spaces is installed in, with ffmpeg on the PATH:

    python helicopter_demo.py flight      # scene_flight.yaml: the stadium and the flight
    python helicopter_demo.py helicopter  # demo/helicopter/below.wav: its sound straight down
    python helicopter_demo.py scene       # scene_overhead.yaml: the helicopter alone
    python helicopter_demo.py walk        # it every 4 m along its flight, at every listener
    python helicopter_demo.py mix         # demo/crossing/seats: the concert and the helicopter
    python helicopter_demo.py video       # demo/crossing: tour.wav, the stills, tour.mp4

or `python helicopter_demo.py all`; the editor's Simulation > Run demo… runs it too. Use
headphones.

Both play at their simulated levels: the loudspeakers' feeds stand for their sound
power (stadium_demo.py's feeds step), the helicopter's sound is in pascals 1 m from its
hub, and the mix puts one in the units of the other through the loudest loudspeaker's
sound power, in the scene's air. The mixes keep one gain for all; the video turns each
stop up to the loudness of the loudest, as the stadium demo's does. The stadium's air is
still: no wind or eddies bend the helicopter's sound. Its rotors' downwash is left out:
flying past at this speed, their wake trails behind them and sinks a few metres a second
(1.6 m/s through the main rotor's disc in its trim), far above the crowd. With Pegg's
broadband noise, the default, the rotors' model is louder than this helicopter's noise
certificate, and with --broadband bpm quieter (see the helicopter_flyover example).
"""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces.demos import Option, Step, passby, pipeline, sibling
from acoustic_spaces.demos.audio import level_db, loudness_db, mmss, read, write
from acoustic_spaces.demos.helicopter import HELICOPTER, Helicopter

stadium = sibling(__file__, "stadium_demo")

#: The stadium's roof (make_model.py's ROOF), its highest point (m).
ROOF_M = 30.0
#: How high the helicopter flies by default (m): as a police or a television helicopter
#: given leave may, below the 300 m above the roof the rules of the air ask over a crowd
#: without it (SERA.5005(f)).
HEIGHT_M = 150.0
#: How far the flight runs each way from over the centre spot (m), at what speed over
#: the ground (m/s, the flyover example's), and how far the ground runs past its ends.
SPAN_M = 600.0
SPEED_M_S = 63.0
MARGIN_M = 50.0
#: How far the sky stays above the helicopter (m).
SKY_M = 40.0
#: The work folder's folder of the mixes with the helicopter in them, and its video.
CROSSING = "crossing"
FLIGHT_HEADER = (
    "# The stadium concert example with a helicopter's flight over it, written by\n"
    "# helicopter_demo.py: the helicopter is left out of the concert's own simulations.\n"
)


class OverTheConcert(Helicopter):
    """The flyover demo's helicopter, flying over the stadium."""

    demo_name = CROSSING

    def texts(self) -> dict[str, str]:
        return super().texts() | {
            "header": "# The stadium with the helicopter flying over it alone, at the speed and "
            "with the sound\n# power and directivity of its rotors, written by "
            f"{self.script}.\n"
        }


VEHICLE = OverTheConcert("helicopter_demo.py")


# --------------------------------------------------------------------------- the flight


def flight_of(height_m: float) -> dict:
    """The helicopter of the flight's scene: straight over the centre spot along the
    stadium's long axis, from behind the stage (at -x) to past the east stand, at
    ``height_m``, left out of the concert's simulations."""
    return {
        "name": HELICOPTER,
        "group": "Helicopter",
        "enabled": False,
        "position": [-SPAN_M, 0.0, height_m],
        "walk": {"to": [[SPAN_M, 0.0, height_m]], "speed_m_s": SPEED_M_S, "doppler": True},
    }


def flight(args) -> None:
    """``--flight`` (scene_flight.yaml) from ``--stadium`` (scene.yaml): the stadium with
    the helicopter's flight 1200 m over it at ``--height``, the ground and the sky
    widened to take it, and the responses as much longer as the helicopter is farther
    than the stage, so that each keeps the decay after its sound arrives."""
    from acoustic_spaces.air import speed_of_sound
    from acoustic_spaces.scene import Scene

    height = float(args.height)
    data = yaml.safe_load(args.stadium.read_text(encoding="utf-8"))
    ground = data["geometry"]["ground"]
    (x0, y0), (x1, y1) = ground["extent"]
    ground["extent"] = [[min(x0, -SPAN_M - MARGIN_M), y0], [max(x1, SPAN_M + MARGIN_M), y1]]
    ground["sky"] = max(float(ground.get("sky", 0.0)), height + SKY_M)
    ends = np.array([[-SPAN_M, 0.0, height], [SPAN_M, 0.0, height]])
    places = np.array([lis["position"] for lis in data["listeners"]], dtype=float)
    farthest = float(np.linalg.norm(ends[:, None, :] - places[None, :, :], axis=2).max())
    scene = Scene.from_yaml(args.stadium)  # what the file leaves out at its defaults
    c = speed_of_sound(scene.air.temperature_c)
    length = scene.simulation.ir_length_s + farthest / c
    simulation = data.setdefault("simulation", {})
    simulation["ir_length_s"] = math.ceil(length * 10.0) / 10.0
    data["sources"] = [s for s in data["sources"] if s["name"] != HELICOPTER]
    data["sources"].append(flight_of(height))
    pipeline.write_scene(args.source, data, FLIGHT_HEADER, args.stadium.parent)
    print(
        f"wrote {args.source}: the helicopter {height:g} m up, {2 * SPAN_M:g} m over the "
        f"stadium at {SPEED_M_S:g} m/s; responses {simulation['ir_length_s']:g} s long"
    )


def flight_done(args) -> bool:
    """Whether the flight's scene is there, newer than the stadium's, at ``--height``."""
    if not args.source.exists() or not args.stadium.exists():
        return False
    if args.source.stat().st_mtime < args.stadium.stat().st_mtime:
        return False
    data = yaml.safe_load(args.source.read_text(encoding="utf-8")) or {}
    flown = next((s for s in data.get("sources", []) if s.get("name") == HELICOPTER), None)
    return flown == flight_of(float(args.height))


# --------------------------------------------------------------------------- the mix


def band_args(args, crossing: bool = False):
    """The stadium demo's arguments for this run: its work folder (or the crossing's
    folder in it), the scene with the band, its results, and the format, the HRTFs and
    the tour of this run."""
    band = stadium.DEMO.defaults()
    band.work = args.work / CROSSING if crossing else args.work
    band.scene, band.results, band.source = args.band, args.results, args.stadium
    band.format, band.hrtf, band.no_mp4, band.tour = args.format, args.hrtf, args.no_mp4, args.tour
    return band


def crossing_times(args, facts: dict, table: dict) -> tuple[float, float]:
    """When the helicopter is over the centre spot (nearest to it in plan, between the
    places of its walk) along its walk, and in the song: ``--at``, or halfway through
    it (s)."""
    times = np.asarray(facts["times"], dtype=float)
    plan = np.asarray(facts["positions"], dtype=float)[:, :2]
    legs = np.diff(plan, axis=0)
    along = np.clip(
        -np.sum(plan[:-1] * legs, axis=1) / np.maximum(np.sum(legs**2, axis=1), 1e-12), 0, 1
    )
    k = int(np.argmin(np.linalg.norm(plan[:-1] + along[:, None] * legs, axis=1)))
    middle = float(times[k] + along[k] * (times[k + 1] - times[k]))
    at = float(table["length_s"]) / 2.0 if args.at is None else float(args.at)
    if at < middle:
        raise SystemExit(
            f"error: --at is {at:g} s, but the helicopter takes {middle:.1f} s to get over the "
            f"stadium: give --at {math.ceil(middle)} or later"
        )
    return middle, at


def scale_of(args, table: dict) -> float:
    """What the helicopter's sound (in pascals 1 m from its hub) is multiplied by to be
    in the units of the loudspeakers' feeds, where a gated RMS of 0.1 stands for the
    loudest one's sound power W, as its pressure 1 m away, sqrt(W rho c / 4 pi), would
    in the scene's air."""
    from acoustic_spaces.air import density, speed_of_sound
    from acoustic_spaces.scene import Scene

    air = Scene.from_yaml(args.band).air
    rho_c = density(air.temperature_c, air.pressure_kpa) * speed_of_sound(air.temperature_c)
    loudest = max(float(entry["level_db"]) for entry in table["sources"].values())
    return 0.1 / math.sqrt(10.0 ** (loudest / 10.0) * 1e-12 * rho_c / (4.0 * math.pi))


def mix(args) -> None:
    """Each listener's mix of the concert (the stadium demo's, in the work folder's
    ``seats``) with the helicopter's pass there (``walk``) added, over the centre spot at
    ``--at``, both at their simulated levels; one gain for all (the loudest peak at -1
    dBFS), into ``crossing/seats``, and the feeds' table with the song lasting as long
    as the helicopter is heard, into ``crossing/feeds``, for the video."""
    band = band_args(args)
    concert = pipeline.seats_of(band.work / "seats", stadium.DEMO.command("render", band))
    table = stadium.feeds_table(band)
    folder = args.work / "walk"
    passes = pipeline.seats_of(folder, DEMO.command("walk", args))
    facts = pipeline.read_json(folder / passby.WALK)
    if passes.get("label") != pipeline.output_label(args):
        raise SystemExit(
            f"error: the helicopter's passes in {folder} are of another format; run: "
            f"{DEMO.command('walk', args)}"
        )
    if (passes["rate"], passes["format"]) != (concert["rate"], concert["format"]):
        raise SystemExit(
            f"error: the concert's mixes are {concert['format']} at {concert['rate']} Hz, the "
            f"helicopter's passes {passes['format']} at {passes['rate']} Hz: run both with one "
            "--format, the scenes at one sample rate"
        )
    rate, fmt = int(concert["rate"]), concert["format"]
    middle, at = crossing_times(args, facts, table)
    start = round((at - middle) * rate)
    scale = scale_of(args, table) / float(passes["gain"])

    def crossed(key: str, seat: dict) -> np.ndarray:
        """The concert's mix at ``seat`` before its gain, the helicopter's pass added."""
        if seat["listener"] not in passes["seats"]:
            raise SystemExit(
                f"error: no pass of the helicopter at {seat['listener']}; run: "
                f"{DEMO.command('walk', args)}"
            )
        x = read(band.work / "seats" / f"{key}.{fmt}.wav")[0] / float(concert["gain"])
        y = read(folder / f"{seat['listener']}.{fmt}.wav")[0] * scale
        out = np.zeros((max(len(x), start + len(y)), x.shape[1]))
        out[: len(x)] += x
        out[start : start + len(y)] += y
        return out

    peak = max(float(np.abs(crossed(k, s)).max()) for k, s in concert["seats"].items())
    gain = 10.0 ** (-1.0 / 20.0) / peak
    out = args.work / CROSSING
    seats = {}
    for key, seat in concert["seats"].items():
        x = crossed(key, seat) * gain
        write(out / "seats" / f"{key}.{fmt}.wav", x, rate)
        seats[key] = seat | {
            "rms_db": level_db(x, rate),
            "loudness_db": loudness_db(x, rate),
            "length_s": len(x) / rate,
        }
    top = max(seat["rms_db"] for seat in seats.values())
    for seat in seats.values():
        seat["relative_db"] = seat["rms_db"] - top
    pipeline.write_json(
        out / "seats" / pipeline.SEATS,
        {"rate": rate, "format": fmt, "gain": gain, "seats": seats},
    )
    heard = max(float(seat["length_s"]) for seat in passes["seats"].values())
    pipeline.write_json(
        out / "feeds" / "feeds.json",
        table
        | {
            "length_s": max(float(table["length_s"]), start / rate + heard),
            "helicopter": {
                "at_s": round(at, 6),
                "start_s": round(start / rate, 6),
                "height_m": round(float(facts["positions"][0][2]), 3),
            },
        },
    )
    print(
        f"wrote {out / 'seats'}: {len(seats)} listeners, the helicopter over the centre spot "
        f"{mmss(at)} into the song"
    )


def mix_done(args) -> bool:
    """Whether the mixes with the helicopter are there, newer than the concert's mixes
    and the helicopter's passes, and made for ``--at``."""
    out = args.work / CROSSING
    path, feeds = out / "seats" / pipeline.SEATS, out / "feeds" / "feeds.json"
    used = [args.work / "seats" / pipeline.SEATS, args.work / "walk" / pipeline.SEATS]
    if not path.exists() or not feeds.exists() or not all(f.exists() for f in used):
        return False
    if any(f.stat().st_mtime > path.stat().st_mtime for f in used):
        return False
    try:
        table = stadium.feeds_table(band_args(args))
    except SystemExit:
        return False
    at = float(table["length_s"]) / 2.0 if args.at is None else float(args.at)
    return pipeline.read_json(feeds).get("helicopter", {}).get("at_s") == round(at, 6)


def video(args) -> None:
    """The stadium demo's video of the mixes with the helicopter: tour.wav, the stills
    and tour.mp4 in the work folder's ``crossing``."""
    feeds = args.work / CROSSING / "feeds" / "feeds.json"
    if not feeds.exists():
        raise SystemExit(f"error: no {feeds}; run first: {DEMO.command('mix', args)}")
    crossing = pipeline.read_json(feeds)["helicopter"]
    facts = pipeline.read_json(args.work / "walk" / passby.WALK)
    file = passby.car_input(args, VEHICLE)
    plays = f"an {facts['helicopter']} at {facts['speed_kmh']:g} km/h (its rotors' model)"
    also = (
        f" · Helicopter: {file.name if file else plays}, {crossing['height_m']:g} m up, "
        f"overhead at {mmss(crossing['at_s'])}"
    )
    heading = (
        "A helicopter over a stadium concert, simulated",
        f"An {facts['helicopter']} {crossing['height_m']:g} m up, over the stadium example's "
        "sound system, stage monitors and band, heard binaurally",
    )
    stadium.video(band_args(args, crossing=True), run_by=(DEMO, args), also=also, heading=heading)


# --------------------------------------------------------------------------- the demo


def problems(demo, args) -> str | None:
    """The flight is over the stadium (above its roof) and ``--at`` is a time; once the
    flight's scene is there, the helicopter must fly as the flyover demo's does
    (:func:`acoustic_spaces.demos.passby.problems`)."""
    height = float(args.height)
    if not math.isfinite(height) or height <= ROOF_M:
        return f"--height is {args.height!r} m, but the stadium's roof is {ROOF_M:g} m up"
    if args.at is not None and not (math.isfinite(float(args.at)) and float(args.at) >= 0.0):
        return f"--at is a time into the song (s), not {args.at!r}"
    resolved = demo.resolved(args)
    if Path(resolved.source).exists():
        return passby.problems(demo, args, VEHICLE)
    for name in ("speed", "step_m"):
        value = getattr(args, name, None)
        if value is not None and not (math.isfinite(float(value)) and float(value) > 0.0):
            return f"{demo.option(name).option} is a positive number, not {value!r}"
    return None if Path(resolved.stadium).exists() else f"no {resolved.stadium}"


DEMO = passby.demo(
    Path(__file__), "stadium_concert", "A helicopter over the concert", __doc__, VEHICLE
)
DEMO.builds_on = stadium.DEMO.name
DEMO.problems = problems
_FLIGHT = Option(
    "source",
    "the stadium with the helicopter's flight over it, which flight writes",
    Path("scene_flight.yaml"),
    "path",
    steps=("flight", HELICOPTER, "scene"),
    flag="--flight",
)
_REPLACED = {
    "source": _FLIGHT,
    "scene": dataclasses.replace(DEMO.option("scene"), default=Path("scene_overhead.yaml")),
}
DEMO.options = [_REPLACED.get(o.name, o) for o in DEMO.options] + [
    Option(
        "stadium",
        "the stadium example's scene",
        Path("scene.yaml"),
        "file",
        steps=("flight",),
        filter="Scene files (*.yaml *.yml)",
    ),
    Option(
        "height",
        "how high the helicopter flies over the ground (m): by default as a police or a "
        "television helicopter given leave may; without leave the EU's rules of the air keep "
        "it 300 m above the stadium's roof, 330 m",
        HEIGHT_M,
        "float",
        steps=("flight",),
        unit="m",
    ),
    Option(
        "at",
        "when the helicopter is over the centre spot, in seconds into the song as the sound "
        "system plays it",
        None,
        "float",
        steps=("mix",),
        unit="s",
        unset="halfway through the song",
    ),
    Option(
        "band",
        "the stadium demo's scene with the band, which its mixes were made from",
        Path("scene_band.yaml"),
        "file",
        steps=("mix", "video"),
        filter="Scene files (*.yaml *.yml)",
    ),
    Option("results", "the stadium demo's simulation of it", Path("results_band"), "folder"),
    Option(
        "tour",
        "listener:phrases,... (8 bars each, 0.5 is 4; the last runs to the end)",
        None,
        steps=("video",),
    ),
]
_OVERHEAD = "scene_overhead.yaml: the helicopter alone over the stadium, at --speed, with Doppler"
DEMO.steps = [
    Step(
        "flight", flight, "scene_flight.yaml: the stadium with the helicopter's flight", flight_done
    ),
    *(
        dataclasses.replace(step, help=_OVERHEAD) if step.name == "scene" else step
        for step in DEMO.steps
        if step.name != "video"
    ),
    Step("mix", mix, "the concert's mixes with the helicopter's pass in each", mix_done),
    Step("video", video, "tour.wav, the stills and tour.mp4 in the work folder's crossing"),
]

__all__ = ["DEMO"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
