"""Check the wave solver's moving air (``simulation.wave.wind: moving``) against exact
solutions in a uniform wind and against the parabolic equation in a wind that grows
with height.

    python validation/wave_wind.py [--json results.json] [--skip-profiles]

The wave solver takes the wind in one of two ways (:mod:`acoustic_spaces.engine.wave`):
``effective``, a speed of sound that the wind's component along the horizontal from the
source changes, as the rays and the parabolic equation take it, or ``moving``, air that
moves, so that sound goes at c + v downwind and c - v upwind whichever way it turned.
Three checks, in the 63 Hz octave band from a grid for that band:

- **Free field over a rigid ground in a uniform wind** of 15 m/s. The source 2 m up,
  listeners 1.5 m up 30 m away downwind, upwind, across the wind and at 45 degrees. In
  a uniform wind the field is known exactly: each path (the direct sound and the
  ground's image, which a wind along the ground leaves an image) is the convected
  Green's function, an impulse of height 1 / R* at tau = (R* - M x) / (c (1 - M^2)), R*
  = sqrt(x^2 + (1 - M^2)(y^2 + z^2)), x along the wind. The grid's own dispersion (its
  waves up to 1.7 % slow along its axes) is the same with and without wind, so the
  sound in the wind is compared with the still air's run as the exact solution
  changes it (each listener's sound moved by the change of its arrival time and scaled
  by R / R*); the level is the energy over the arrival.
- **A facade downwind.** A rigid building 40 m downwind of the source, its facade
  across the wind, 30 m high and 50 m wide, and a listener 10 m upwind of the source:
  the facade sends the sound back against the wind. Ray theory in a uniform wind gives
  when the reflections arrive (a ray goes at c n + v for its wave normal n; at the
  facade the wave keeps its frequency and its wavenumber along the facade, and goes
  back at the speed of sound against the wind). Their arrival is read from the
  correlation with the still air's, against the exact change.
- **A wind that grows with height** over grass (Miki's impedance, 200 kPa s/m²), the
  `windy` preset (5 m/s 10 m up, logarithmic): the source 2 m up, listeners 1.5 and 5 m
  up 50, 100 and 200 m downwind and upwind, against the engine's parabolic equation
  (the reference of ``wave_outdoor.py``), with moving air and with the effective speed
  of sound.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acoustic_spaces import Room, Scene  # noqa: E402
from acoustic_spaces.engine.model import TraceModel  # noqa: E402
from acoustic_spaces.engine.wave import wave_responses  # noqa: E402

RATE = 48_000
#: The uniform wind (m/s), the source's height and the listeners' (m).
WIND = 15.0
ZS, ZR = 2.0, 1.5
#: The listeners of the free field: distance (m), and their bearings from the wind.
DISTANCE = 30.0
BEARINGS = {"downwind": 0.0, "upwind": 180.0, "across": 90.0, "45 degrees": 45.0}
#: The facade: where it stands downwind (m), its height and width (m), and the listener
#: upwind of the source (m), both 2 m up.
FACADE_X, FACADE_HEIGHT, FACADE_WIDTH, FACADE_DEPTH = 40.0, 30.0, 50.0, 10.0
FACADE_LISTENER = (-10.0, 0.0, 2.0)
#: The band (Hz) compared, and where its edges fade (a third of an octave each).
BAND = (40.0, 90.0)
WIND_MODELS = ("effective", "moving")


def _scene(listeners: dict, wind: float, wind_model: str, building: bool = False) -> Room:
    span = DISTANCE + 15.0
    low, high = [-span, -span], [span, span]
    sky = 12.0
    data = {
        "geometry": {"ground": {}},
        "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": [0.0, 0.0, ZS]}],
        "listeners": [{"name": k, "position": list(p)} for k, p in listeners.items()],
        "simulation": {"wave": {"enabled": True, "max_band_hz": 63, "wind": wind_model}},
    }
    if building:
        x0, x1, w = FACADE_X, FACADE_X + FACADE_DEPTH, FACADE_WIDTH / 2
        low, high = [-25.0, -w - 15.0], [x1 + 15.0, w + 15.0]
        sky = FACADE_HEIGHT + 5.0
        data["geometry"]["buildings"] = [
            {
                "name": "facade",
                "footprint": [[x0, -w], [x1, -w], [x1, w], [x0, w]],
                "height": FACADE_HEIGHT,
            }
        ]
    data["geometry"]["ground"] = {"extent": [low, high], "sky": sky}
    if wind:
        data["atmosphere"] = {
            "wind": {"speed_m_s": wind, "direction_deg": 0.0, "profile": "constant"}
        }
    return Room.from_scene(Scene.model_validate(data))


def _run(room: Room, length_s: float, wind_model: str, still: bool = False) -> tuple[list, dict]:
    """The wave solver's responses; ``still`` for the still air's reference, on the grid
    the moving air's run takes (as fine as sound going against the wind needs)."""
    model = TraceModel.from_room(room)
    c = room.speed_of_sound()
    listeners = room.scene.listeners
    points = np.array([listener.position for listener in listeners], dtype=float)
    return wave_responses(
        model,
        np.array([0.0, 0.0, ZS]),
        points,
        [listener.name for listener in listeners],
        max_band_hz=63,
        length_s=length_s,
        sample_rate=RATE,
        speed_of_sound=c,
        wind=wind_model,
        points_per_wavelength=8.0 * c / (c - WIND) if still else 8.0,
    )


#: The frequencies (Hz) the band is summed over, and the time step (s) compared on: the
#: runs with and without wind step at different rates.
FREQ_STEP = 0.02
FREQS = np.arange(34.0, 104.0, FREQ_STEP)
STEP = 1.0 / 4000.0


def _gain() -> np.ndarray:
    """The band's gain at FREQS: flat from BAND[0] to BAND[1], its edges raised cosines a
    third of an octave wide."""
    rise = np.clip((np.log2(FREQS / BAND[0]) + 1 / 6) * 3.0, 0.0, 1.0)
    fall = np.clip((np.log2(BAND[1] / FREQS) + 1 / 6) * 3.0, 0.0, 1.0)
    return np.sin(0.5 * np.pi * rise) ** 2 * np.sin(0.5 * np.pi * fall) ** 2


def spectrum(x: np.ndarray, rate: float) -> np.ndarray:
    """The frequency response at FREQS of the response x sampled at rate (Hz)."""
    return np.exp(-2j * np.pi * np.outer(FREQS, np.arange(len(x)) / rate)) @ x


def in_band(response: np.ndarray, times: np.ndarray, delay: float = 0.0) -> np.ndarray:
    """The band's sound at times (s) for the frequency response response, delayed by
    delay (s)."""
    weights = _gain() * response * np.exp(-2j * np.pi * FREQS * delay)
    return 2.0 * FREQ_STEP * (np.exp(2j * np.pi * np.outer(times, FREQS)) @ weights).real


def convected(c: float, wind: float, x: float, y: float, z: float) -> tuple[float, float]:
    """The convected Green's function's arrival (s) and R* (m) at (x, y, z) from the
    source, x along the wind."""
    m = wind / c
    b2 = 1.0 - m * m
    r_star = math.sqrt(x * x + b2 * (y * y + z * z))
    return (r_star - m * x) / (c * b2), r_star


def _lag(a: np.ndarray, b: np.ndarray, rate: float) -> float:
    """How much later a is than b (s), from the peak of their correlation."""
    xc = np.correlate(a, b, "full")
    k = int(np.argmax(xc))
    frac = 0.5 * (xc[k - 1] - xc[k + 1]) / (xc[k - 1] - 2.0 * xc[k] + xc[k + 1])
    return (k - (len(b) - 1) + frac) / rate


def check_free_field() -> list[dict]:
    listeners = {
        name: (
            DISTANCE * math.cos(math.radians(b)),
            DISTANCE * math.sin(math.radians(b)),
            ZR,
        )
        for name, b in BEARINGS.items()
    }
    length = DISTANCE / 300.0 + 0.25
    still_room = _scene(listeners, 0.0, "effective")
    c = still_room.speed_of_sound()
    still, _ = _run(still_room, length, "effective", still=True)
    rows = []
    calm = {}
    # The grid's own error, the same with and without wind: still air against the exact
    # field.
    for (name, (x, y, z)), response in zip(listeners.items(), still, strict=True):
        calm[name] = spectrum(response.signals[:, 0], response.rate)
        r = math.sqrt(x * x + y * y + (z - ZS) ** 2)
        times = r / c + np.arange(-0.03, 0.05, STEP)
        got = in_band(calm[name], times)
        want = in_band(exact(c, 0.0, x, y, z), times)
        print(
            f"still air {name:10s}: the grid's own error, level "
            f"{10.0 * math.log10(np.sum(got**2) / np.sum(want**2)):+.2f} dB, arrival "
            f"{1e3 * _lag(got, want, 1.0 / STEP):+.3f} ms",
            flush=True,
        )
    for wind_model in WIND_MODELS:
        started = time.perf_counter()
        windy, info = _run(_scene(listeners, WIND, wind_model), length, wind_model)
        seconds = time.perf_counter() - started
        for (name, (x, y, z)), response in zip(listeners.items(), windy, strict=True):
            # The still air's sound as the wind changes it: moved by the direct sound's
            # change of arrival and scaled by R / R* (the ground's image changes alike,
            # to a few microseconds).
            r = math.sqrt(x * x + y * y + (z - ZS) ** 2)
            tau, r_star = convected(c, WIND, x, y, z - ZS)
            times = tau + np.arange(-0.03, 0.05, STEP)
            a = in_band(spectrum(response.signals[:, 0], response.rate), times)
            b = in_band(calm[name], times, tau - r / c) * r / r_star
            row = {
                "wind": wind_model,
                "listener": name,
                "waveform_db": 10.0 * math.log10(np.sum((a - b) ** 2) / np.sum(b**2)),
                "level_db": 10.0 * math.log10(np.sum(a**2) / np.sum(b**2)),
                "arrival_ms": 1e3 * _lag(a, b, 1.0 / STEP),
                "seconds": seconds,
                "cells": info["wave_cells"],
            }
            rows.append(row)
            print(
                f"{wind_model:9s} {name:10s}: waveform {row['waveform_db']:6.1f} dB, level "
                f"{row['level_db']:+.2f} dB, arrival {row['arrival_ms']:+.3f} ms",
                flush=True,
            )
    return rows


def exact(c: float, wind: float, x: float, y: float, z: float) -> np.ndarray:
    """The exact frequency response at FREQS: the direct sound and the ground's image,
    each the convected Green's function."""
    out = np.zeros(len(FREQS), dtype=complex)
    for zs in (ZS, -ZS):
        tau, r_star = convected(c, wind, x, y, z - zs)
        out += np.exp(-2j * np.pi * FREQS * tau) / r_star
    return out


def _facade_path(theta: float, c: float, wind: float, zs: float, zr: float) -> tuple:
    """The ray that leaves the source (height zs) with its wave normal at theta above the
    horizontal, off the facade, back to the listener's distance: how far above zr it
    passes there (m), when (s) and its length (m)."""
    xl = FACADE_LISTENER[0]
    # Incident: the wave normal n, the ray at c n + v.
    nx, nz = math.cos(theta), math.sin(theta)
    t1 = FACADE_X / (c * nx + wind)
    z_hit = zs + c * nz * t1
    # The facade keeps the frequency and the wavenumber along it (per unit frequency):
    # c |k| + v k_x = 1 with k_x < 0 on the way back.
    kz = nz / (c + wind * nx)
    a = c * c - wind * wind
    kx = (-wind - math.sqrt(wind * wind - a * (c * c * kz * kz - 1.0))) / a
    k = math.hypot(kx, kz)
    ux, uz = c * kx / k + wind, c * kz / k
    t2 = (xl - FACADE_X) / ux
    length = math.hypot(FACADE_X, z_hit - zs) + math.hypot(FACADE_X - xl, zr - z_hit)
    return z_hit + uz * t2 - zr, t1 + t2, length


def facade_times(c: float, wind: float) -> list[tuple[float, float]]:
    """The arrivals (s) of the sound the facade sends back to the listener in a wind
    ``wind`` blowing at it (m/s), and each path's length (m): ray theory in a uniform
    wind, the ground unfolded into images (a wind along the ground keeps them)."""
    out = []
    zl = FACADE_LISTENER[2]
    for zs in (ZS, -ZS):
        for zr in (zl, -zl):
            lo, hi = -0.5, 0.5
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                below = _facade_path(lo, c, wind, zs, zr)[0] < 0
                if below == (_facade_path(mid, c, wind, zs, zr)[0] < 0):
                    lo = mid
                else:
                    hi = mid
            _, seconds, length = _facade_path(0.5 * (lo + hi), c, wind, zs, zr)
            out.append((seconds, length))
    return out


def check_facade() -> list[dict]:
    listeners = {"upwind": FACADE_LISTENER}
    length = 0.45
    still_room = _scene(listeners, 0.0, "effective", building=True)
    c = still_room.speed_of_sound()
    (calm,), _ = _run(still_room, length, "effective", still=True)
    still_times = facade_times(c, 0.0)
    windy_times = facade_times(c, WIND)
    change = np.mean([w[0] - s[0] for w, s in zip(windy_times, still_times, strict=True)])
    arrival = min(w[0] for w in windy_times)
    times = arrival + np.arange(-0.03, 0.03, STEP)
    # The still air's facade sound moved by the exact change of its arrival: the windy
    # run's lag behind it is the error.
    expected = in_band(spectrum(calm.signals[:, 0], calm.rate), times, change)
    rows = []
    for wind_model in WIND_MODELS:
        started = time.perf_counter()
        (response,), info = _run(_scene(listeners, WIND, wind_model, True), length, wind_model)
        got = in_band(spectrum(response.signals[:, 0], response.rate), times)
        error = 1e3 * _lag(got, expected, 1.0 / STEP)
        moved = 1e3 * change + error
        row = {
            "wind": wind_model,
            "exact_change_ms": 1e3 * change,
            "change_ms": moved,
            "error_ms": error,
            "seconds": time.perf_counter() - started,
            "cells": info["wave_cells"],
        }
        rows.append(row)
        print(
            f"{wind_model:9s}: the facade's sound arrives {row['change_ms']:+.2f} ms later than "
            f"in still air (exact {row['exact_change_ms']:+.2f} ms): off by "
            f"{row['error_ms']:+.2f} ms",
            flush=True,
        )
    return rows


def check_profiles() -> list[dict]:
    """The windy preset downwind and upwind against the parabolic equation, at 63 Hz."""
    import wave_outdoor

    rows = []
    for weather in ("downwind", "upwind"):
        room = wave_outdoor.scene(weather, 63)
        pe = wave_outdoor.pe_levels(room, 63)
        for wind_model in WIND_MODELS:
            started = time.perf_counter()
            wave, info = wave_outdoor.wave_levels(room, 63, wind=wind_model)
            for name in wave:
                rows.append(
                    {
                        "weather": weather,
                        "wind": wind_model,
                        "listener": name,
                        "wave_db": wave[name],
                        "pe_db": pe[name],
                        "difference_db": wave[name] - pe[name],
                    }
                )
            print(
                f"{weather}, {wind_model}: {time.perf_counter() - started:.0f} s, "
                f"{info['wave_cells']:,.0f} cells",
                flush=True,
            )
    print("\n| weather | listener | parabolic equation | effective | moving |")
    print("|---|---|---|---|---|")
    for weather in ("downwind", "upwind"):
        names = [r["listener"] for r in rows if r["weather"] == weather and r["wind"] == "moving"]
        for name in names:
            pick = {r["wind"]: r for r in rows if r["weather"] == weather and r["listener"] == name}
            effective, moving = pick["effective"], pick["moving"]
            print(
                f"| {weather} | {name} | {moving['pe_db']:+.1f} | "
                f"{effective['wave_db']:+.1f} ({effective['difference_db']:+.1f}) | "
                f"{moving['wave_db']:+.1f} ({moving['difference_db']:+.1f}) |"
            )
    for wind_model in WIND_MODELS:
        miss = np.abs([r["difference_db"] for r in rows if r["wind"] == wind_model])
        print(f"{wind_model}: mean difference {miss.mean():.2f} dB, largest {miss.max():.2f} dB")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    parser.add_argument(
        "--skip-profiles", action="store_true", help="leave out the parabolic equation's check"
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    print(f"Free field over a rigid ground in a uniform wind of {WIND:g} m/s, 63 Hz band\n")
    free = check_free_field()
    print("\nA facade 40 m downwind, the listener 10 m upwind of the source\n")
    facade = check_facade()
    profiles = []
    if not args.skip_profiles:
        print("\nThe windy preset over grass against the parabolic equation, 63 Hz band\n")
        profiles = check_profiles()
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        out = {"free_field": free, "facade": facade, "profiles": profiles}
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
