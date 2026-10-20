"""Check the wave solver in wind and temperature against a parabolic equation.

    python validation/wave_outdoor.py [--weathers NAME ...] [--json results.json]

The reference is the engine's own parabolic equation (:mod:`acoustic_spaces.engine.
parabolic`, within 0.25 dB of the exact solution over grass in still air: see
``ground_reflection.py``), marched in the vertical plane through the source and the
listener with the effective speed of sound along it. The wave solver
(:mod:`acoustic_spaces.engine.wave`) steps the same scene in three dimensions, each
cell's speed of sound the effective one along the horizontal from the source.

Over grass (Miki's impedance, 200 kPa s/m²), the source 2 m up, the listeners 1.5 and
5 m up: the 63 Hz octave band from a grid for that band, 50, 100 and 200 m away, and
the 125 Hz band from a grid for it, 50 and 100 m away. Five weathers: still air,
downwind and upwind in the windy preset (5 m/s 10 m up), a sunny day (sound bends up)
and a clear night (sound bends down). The ground reaches far enough to either side that
its edges change the level by little (three times the half-width of the reflection's
first Fresnel zone at the farthest listener), and 30 m past it.

Both levels are in dB over the free field: the mean square of the pressure over the
band's slices (the frequencies the engine's parabolic equation takes), the wave
solver's from the spectrum of its response over the source's. The script prints a
Markdown table, and the mean and largest differences per band. A run takes some half an
hour on four cores (the 125 Hz grids have up to 30 million cells).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine.diffraction import band_frequencies
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.parabolic import march
from acoustic_spaces.engine.refracted import WAVE_HEADROOM, WAVE_MIN_HEADROOM_M, WAVE_TOP_SLOPE
from acoustic_spaces.engine.refraction import Profile
from acoustic_spaces.engine.wave import STENCILS, source_signal, wave_responses

SOURCE_M = 2.0
HEIGHTS_M = (1.5, 5.0)
#: Per band (centre, Hz): the listeners' distances (m), the ground's half-width (m) and
#: the sky's height (m).
GRIDS = {63: ((50.0, 100.0, 200.0), 70.0, 30.0), 125: ((50.0, 100.0), 30.0, 22.0)}
WEATHERS = {
    "still": None,
    "downwind": {"preset": "windy", "wind": {"direction_deg": 0}},
    "upwind": {"preset": "windy", "wind": {"direction_deg": 180}},
    "sunny_day": {"preset": "sunny_day"},
    "clear_night": {"preset": "clear_night"},
}
SAMPLE_RATE = 48_000


def scene(weather: str, band: int) -> Room:
    distances, half_width, sky = GRIDS[band]
    data = {
        "geometry": {
            "ground": {"extent": [[-20.0, -half_width], [max(distances) + 30.0, half_width]]},
        },
        "materials": {"lawn": {"impedance": "grass"}},
        "assignments": {"ground": "lawn"},
        "sources": [{"name": "s", "position": [0.0, 0.0, SOURCE_M]}],
        "listeners": [
            {"name": f"{d:g}m_{z:g}m", "position": [d, 0.0, z]}
            for d in distances
            for z in HEIGHTS_M
        ],
    }
    data["geometry"]["ground"]["sky"] = sky
    if WEATHERS[weather] is not None:
        data["atmosphere"] = WEATHERS[weather]
    return Room.from_scene(Scene.model_validate(data))


def wave_levels(room: Room, band: int) -> tuple[dict[str, float], dict]:
    """The wave solver's level (dB over the free field) at each listener in ``band``,
    and the run's details."""
    model = TraceModel.from_room(room)
    distances = GRIDS[band][0]
    listeners = room.scene.listeners
    points = np.array([listener.position for listener in listeners], dtype=float)
    responses, info = wave_responses(
        model,
        np.array([0.0, 0.0, SOURCE_M]),
        points,
        [listener.name for listener in listeners],
        max_band_hz=band,
        length_s=max(distances) / 330.0 + 0.3,
        sample_rate=SAMPLE_RATE,
        speed_of_sound=room.speed_of_sound(),
        say=lambda line: print(line, file=sys.stderr, flush=True),
    )
    slices = _slices(band)
    out = {}
    for listener, point, response in zip(listeners, points, responses, strict=True):
        courant = STENCILS["standard"][1]
        signal, _ = source_signal(response.rate, response.cell, courant)
        impulse = signal / (4.0 * math.pi * courant**2 / response.cell)
        n_fft = 1 << int(math.ceil(math.log2(8 * len(response.signals))))
        f = np.fft.rfftfreq(n_fft, 1.0 / response.rate)
        transfer = np.abs(np.fft.rfft(response.signals[:, 0], n_fft) / np.fft.rfft(impulse, n_fft))
        power = np.interp(slices, f, transfer**2)
        distance = float(np.linalg.norm(point - [0.0, 0.0, SOURCE_M]))
        out[listener.name] = 10.0 * math.log10(power.mean() * distance**2)
    return out, info


def _slices(band: int) -> np.ndarray:
    frequencies, _ = band_frequencies()
    centres = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    return frequencies[centres.index(band)]


def pe_levels(room: Room, band: int) -> dict[str, float]:
    """The parabolic equation's level (dB over the free field) at each listener."""
    scene_ = room.scene
    atmosphere = scene_.atmosphere
    ground_c = float(scene_.air.temperature_c)
    towards = np.array([1.0, 0.0])
    if atmosphere is None:
        profile = Profile.uniform(room.speed_of_sound())
    else:
        profile = Profile.of(atmosphere, ground_c, towards)
    slices = _slices(band)
    material = scene_.materials["lawn"]
    beta = np.asarray(material.impedance.admittance(slices, room.speed_of_sound()), dtype=complex)
    impedance = 1.0 / np.conj(beta)  # time as exp(-i omega t)
    out = {}
    for listener in scene_.listeners:
        x, _, z = (float(v) for v in listener.position)
        top = max(
            WAVE_HEADROOM * max(SOURCE_M, z),
            max(SOURCE_M, z) + WAVE_MIN_HEADROOM_M,
            WAVE_TOP_SLOPE * x,
        )
        pressure = march(slices, profile.speed, lambda r: impedance, SOURCE_M, [x], [z], top)[:, 0]
        distance = math.hypot(x, z - SOURCE_M)
        out[listener.name] = 10.0 * math.log10(np.mean(np.abs(pressure) ** 2) * distance**2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--weathers", nargs="+", default=list(WEATHERS), choices=list(WEATHERS))
    parser.add_argument("--bands", nargs="+", type=int, default=list(GRIDS), choices=list(GRIDS))
    parser.add_argument("--json", help="write the results here")
    args = parser.parse_args()
    rows = []
    for band in args.bands:
        for weather in args.weathers:
            started = time.perf_counter()
            room = scene(weather, band)
            wave, info = wave_levels(room, band)
            pe = pe_levels(room, band)
            print(
                f"{weather}, {band} Hz: {time.perf_counter() - started:.0f} s, "
                f"{info['wave_cells']:,.0f} cells",
                file=sys.stderr,
                flush=True,
            )
            for name in wave:
                rows.append(
                    {
                        "weather": weather,
                        "band_hz": band,
                        "listener": name,
                        "wave_db": round(wave[name], 2),
                        "pe_db": round(pe[name], 2),
                        "difference_db": round(wave[name] - pe[name], 2),
                    }
                )
    print(
        "| weather | band (Hz) | listener | wave solver (dB) | parabolic equation (dB) "
        "| difference |"
    )
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['weather']} | {row['band_hz']} | {row['listener']} | {row['wave_db']:+.1f} "
            f"| {row['pe_db']:+.1f} | {row['difference_db']:+.1f} |"
        )
    print()
    for band in args.bands:
        miss = np.abs([row["difference_db"] for row in rows if row["band_hz"] == band])
        print(f"{band} Hz: mean difference {miss.mean():.2f} dB, largest {miss.max():.2f} dB")
    if args.json:
        with open(args.json, "w") as out:
            json.dump(rows, out, indent=1)


if __name__ == "__main__":
    main()
