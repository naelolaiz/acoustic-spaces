"""Check the flight demo's aircraft (:mod:`acoustic_spaces.demos.aircraft`) against Doc
29's own levels: its NPD curves in the scene's air, and Doc 29's sum over the flight's
segments.

    python validation/aircraft_noise.py [--json results.json] [--step 20] [--rays 50000]

- **The source against its NPD curves.** JETW flying straight and level at 160 kt (the
  NPD curves' speed) and at one power over a microphone 1.2 m above grass, as its NPD
  curves are measured, at the NPD distances from 200 to 4000 ft right over it: its
  sound power and directivity as the demo gives them
  (:mod:`acoustic_spaces.doc29.source`), simulated in the engine
  (:func:`~acoustic_spaces.walkthrough.simulate_walk`, its places ``STEP`` of the
  distance apart, 4 m at 200 ft). Its maximum level (L_Amax) and sound exposure level
  (SEL) are set against the NPD curves' in the scene's air (Appendix D,
  :func:`acoustic_spaces.doc29.atmosphere.adjusted`) with Doc 29's impedance adjustment
  (eq. 4-6). The source is made to the NPD curves' maximum level 1000 ft away, so that
  distance checks the closure (the spectrum, the ground taken out and put back, the
  air); the others show how far the NPD curves' shapes differ from a point source with
  one spectrum in the engine's air and ground.
- **The airport example's flights.** The demo's departure and arrival, each simulated
  at every listener of the example every ``--step`` along its flight, its level
  following its thrust (:meth:`~acoustic_spaces.demos.aircraft.Aircraft.sounding`):
  L_Amax and SEL against Doc 29's at the same receivers, summed over the same flight
  path's segments with the NPD curves in the scene's air, the engines' installation,
  the lateral attenuation and the start-of-roll directivity
  (:func:`acoustic_spaces.doc29.noise.event`). The engine reckons the ground and the
  directivity behind the take-off roll in its own way; the differences are reported,
  not checked.

The engine's L_Amax is the highest level of the places, each as with the aircraft
standing there; Doc 29's is measured with slow time weighting. The late rays are
seeded (``SEED``): the grass's scattered sound comes from the rays that land near the
microphone, few when the aircraft is high, so it changes from place to place, and
``--rays`` sets how many there are.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces import Room
from acoustic_spaces.demos import aircraft, load
from acoustic_spaces.demos.passby import sounded
from acoustic_spaces.directivity import write_directivity_csv
from acoustic_spaces.doc29 import atmosphere, noise, source
from acoustic_spaces.doc29.tables import FOOT_M
from acoustic_spaces.examples import example_path
from acoustic_spaces.materials.bands import power_levels
from acoustic_spaces.walkthrough import pass_by, simulate_walk

#: How far apart the engine simulates the aircraft in the NPD flyovers, in distances to
#: the microphone (4 m at 200 ft), and how far they reach on either side of it.
STEP = 1.0 / 15.24
REACH = 8.0
#: The NPD distances (ft) of the flyovers, and the power of each operation they are
#: flown at (lb per engine, within each NPD curves' powers).
FEET = (200.0, 400.0, 630.0, 1000.0, 2000.0, 4000.0)
POWERS = {"D": 15000.0, "A": 5000.0}
#: The scene's air: the airport example's.
TEMPERATURE_C, HUMIDITY_PERCENT = 15.0, 70.0
#: The late rays' seed, so that a run gives the same numbers.
SEED = 1


def _late(rays: int | None) -> dict:
    """The late stage: the engine's default rays, or ``rays``."""
    late: dict = {"method": "stochastic_ray_tracing"}
    if rays is not None:
        late["rays"] = rays
    return late


def _flyover_scene(
    folder: Path, operation: str, feet: float, power: float, rays: int | None = None
) -> Path:
    """A scene of JETW flying level at 160 kt ``feet`` over a microphone 1.2 m above
    grass, at ``power``, its sound power and directivity the demo's."""
    plane = aircraft.Aircraft(operation)
    maximum, spectrum = plane.curves()
    installation = aircraft._data()[0].installation
    thirds = source.sound_power_db(maximum, spectrum, power, TEMPERATURE_C)
    level = power_levels(thirds, "octave") + source.below_offset_db(installation)
    table = folder / f"jetw_{operation}.csv"
    write_directivity_csv(table, source.directivity(installation), "JETW's installation")
    d = feet * FOOT_M
    height = d + source.MICROPHONE_M
    reach = REACH * d
    data = {
        "name": f"JETW {feet:g} ft over the NPD microphone",
        "geometry": {
            "ground": {"extent": [[-reach - 50, -50], [reach + 50, 50]], "sky": height + 20}
        },
        "materials": {"meadow": {"impedance": "grass"}},
        "assignments": {"ground": "meadow"},
        "air": {"temperature_c": TEMPERATURE_C, "humidity_percent": HUMIDITY_PERCENT},
        "sources": [
            {
                "name": "jet",
                "position": [-reach, 0.0, height],
                "level_db": [round(float(v), 3) for v in level],
                "directivity": {"file": table.name},
                "walk": {
                    "to": [[reach, 0.0, height]],
                    "speed_m_s": noise.REFERENCE_SPEED_M_S,
                    "face_forward": True,
                    "doppler": True,
                },
            }
        ],
        "listeners": [{"name": "npd", "position": [0.0, 0.0, source.MICROPHONE_M]}],
        "simulation": {
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
            "late": _late(rays),
            "ir_length_s": 1.0,
            "seed": SEED,
        },
        "outputs": ["mono"],
    }
    path = folder / f"npd_{operation}_{feet:g}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _doc29_curves(operation: str):
    """JETW's NPD curves of ``operation`` (SEL, L_Amax) in the scene's air, and the
    impedance adjustment there (dB)."""
    plane = aircraft.Aircraft(operation)
    maximum, spectrum = plane.curves()
    jet, npd, *_ = aircraft._data()
    exposure = npd[(jet.npd, "SEL", operation)]
    adjust = (TEMPERATURE_C, HUMIDITY_PERCENT, 101.325)
    return (
        atmosphere.adjusted(exposure, spectrum, *adjust),
        atmosphere.adjusted(maximum, spectrum, *adjust),
        noise.impedance_db(TEMPERATURE_C, 101.325),
    )


def check_npd(folder: Path, rays: int | None = None) -> list[dict]:
    """The flyovers at the NPD distances, each operation at its power of POWERS."""
    rows = []
    for operation, power in POWERS.items():
        exposure, maximum, impedance = _doc29_curves(operation)
        for feet in FEET:
            room = Room.from_yaml(_flyover_scene(folder, operation, feet, power, rays))
            walked = simulate_walk(room, "jet", listener="npd", step_m=STEP * feet * FOOT_M)
            heard = pass_by(walked)
            d = feet * FOOT_M
            want_max = float(maximum.level(power, d)) + impedance
            want_sel = float(exposure.level(power, d)) + impedance
            row = {
                "operation": operation,
                "feet": feet,
                "power_lb": power,
                "engine_max_a_db": round(heard.max_a, 2),
                "doc29_max_a_db": round(want_max, 2),
                "engine_sel_a_db": round(heard.exposure_a, 2),
                "doc29_sel_a_db": round(want_sel, 2),
            }
            rows.append(row)
            print(
                f"  {operation} {feet:6.0f} ft  L_Amax {heard.max_a:6.1f} against {want_max:6.1f}"
                f" ({heard.max_a - want_max:+.1f})   SEL {heard.exposure_a:6.1f} against "
                f"{want_sel:6.1f} ({heard.exposure_a - want_sel:+.1f})"
            )
    return rows


def check_flights(folder: Path, step_m: float, rays: int | None = None) -> list[dict]:
    """The airport example's departure and arrival at each of its listeners, the late
    rays seeded."""
    from acoustic_spaces.scene import Scene

    work = folder / "airport"
    shutil.copytree(Path(example_path("airport")).parent, work)
    path = work / "scene.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["simulation"].update({"late": _late(rays), "seed": SEED})
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    demo = load(work / "flight_demo.py").at(work)
    rows = []
    for plane in (aircraft.DEPARTURE, aircraft.ARRIVAL):
        assert demo.main(["scene", "--vehicle", plane.name]) == 0
        room = Room.from_yaml(work / "scene_demo.yaml")
        args = argparse.Namespace(source=work / "scene.yaml")
        exposure, maximum, impedance = _doc29_curves(plane.operation)
        segments = plane.segments_of(Scene.from_yaml(work / "scene.yaml"))
        listeners = [lis for lis in room.scene.listeners if lis.in_use]
        receivers = np.array([lis.position for lis in listeners], dtype=float)
        doc29 = noise.event(
            aircraft._data()[0], exposure, maximum, segments, receivers, impedance=impedance
        )
        for k, listener in enumerate(listeners):
            walked = simulate_walk(room, plane.name, listener=listener.name, step_m=step_m)
            loud = 10.0 ** (plane.gains_db(args, walked.points.times) / 10.0)
            heard = sounded(pass_by(walked), loud)
            row = {
                "operation": plane.operation,
                "listener": listener.name,
                "engine_max_a_db": round(heard.max_a, 2),
                "doc29_max_a_db": round(float(doc29.lamax[k]), 2),
                "engine_sel_a_db": round(heard.exposure_a, 2),
                "doc29_sel_a_db": round(float(doc29.sel[k]), 2),
            }
            rows.append(row)
            print(
                f"  {plane.name:9s} {listener.name:15s} L_Amax {heard.max_a:6.1f} against "
                f"{doc29.lamax[k]:6.1f} ({heard.max_a - doc29.lamax[k]:+.1f})   SEL "
                f"{heard.exposure_a:6.1f} against {doc29.sel[k]:6.1f} "
                f"({heard.exposure_a - doc29.sel[k]:+.1f})"
            )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    parser.add_argument(
        "--step", type=float, default=20.0, help="how far apart the flights are simulated (m)"
    )
    parser.add_argument(
        "--rays", type=int, help="late rays per band (default: the engine's, 50000)"
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        print("JETW over the NPD microphone at 160 kt, against its NPD curves in the scene's air")
        npd = check_npd(folder, args.rays)
        print(f"\nThe airport example's flights, every {args.step:g} m, against Doc 29")
        flights = check_flights(folder, args.step, args.rays)
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        args.json.write_text(
            json.dumps({"npd": npd, "flights": flights}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
