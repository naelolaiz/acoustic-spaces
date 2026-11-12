"""Check the wave solver's moving air (``simulation.wave.wind: moving``) against exact
solutions in a uniform wind, against ray theory through the wind it makes flow round a
building, and against the parabolic equation in a wind that grows with height.

    python validation/wave_wind.py [--json results.json] [--skip-profiles]

The wave solver takes the wind in one of two ways (:mod:`acoustic_spaces.engine.wave`):
``effective``, a speed of sound that the wind's component along the horizontal from the
source changes, as the rays and the parabolic equation take it, or ``moving``, air that
moves, so that sound goes at c + v downwind and c - v upwind whichever way it turned.
Moving air's wind flows round the model's solids. Seven checks, in the 63 Hz octave band
from a grid for that band (the dome's on the 125 Hz band's grid too, the border's on
a grid of its own):

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
- **The wind round a dome.** A rigid half sphere of 4 m radius on the ground in a
  uniform wind: the wind the solver makes flow round it (read from the solver the run
  made, on the faces between its cells) against the exact flow of an ideal fluid past
  a sphere, U (x + a^3 x / (2 r^3))'s gradient, which the ground, a plane of symmetry,
  leaves as it is. On the 63 Hz band's grid and on the 125 Hz band's.
- **A facade downwind.** A rigid building 40 m downwind of the source, its facade
  across the wind, 30 m high and 50 m wide, and a listener 10 m upwind of the source:
  the facade sends the sound back against the wind. Ray theory in a uniform wind gives
  when the reflections arrive (a ray goes at c n + v for its wave normal n; at the
  facade the wave keeps its frequency and its wavenumber along the facade, and goes
  back at the speed of sound against the wind). The moving air's wind is not uniform:
  it slows towards the facade and flows round the building, so its reference is the
  time along the still air's paths through the wind the solver made, each step of a
  path at the speed of a ray along it, v . s + sqrt(c^2 - |v|^2 + (v . s)^2) (the paths
  themselves change the time only to the second order of their bending). Their arrival
  is read from the correlation with the still air's.
- **A wind that grows with height** over grass (Miki's impedance, 200 kPa s/m²), the
  `windy` preset (5 m/s 10 m up, logarithmic): the source 2 m up, listeners 1.5 and 5 m
  up 50, 100 and 200 m downwind and upwind, against the engine's parabolic equation
  (the reference of ``wave_outdoor.py``), with moving air and with the effective speed
  of sound.
- **A room with an opening**, the nearly closed space that a wind blown through the
  model's solids makes grow without bound. A rigid room on the ground, 4 x 4 x 3 m
  inside, with an opening of 0.8 x 0.8 m in its wall downwind, the source and a
  listener in it, in uniform winds of 5, 15 and 30 m/s: over 3 s, how the listener's
  level changes from the second quarter second to the last, against the effective
  speed of sound's run, the fastest wind round the room and the fastest inside it.
- **The border of the moving air.** The air of the grid's matched layer is still, and
  a wave that leaves the moving air for it comes back in part, by about half the wind's
  Mach number. The rigid 8 x 6 m plate of ``wave_solver.py``'s open model in uniform
  winds of 5 to 40 m/s along the grid and at 45 degrees to it, 25 to 150 Hz on cells of
  30 cm: what differs at its three listeners from a grid 45 cells larger all round,
  whose border sends nothing back in the 70 ms of the runs (their last 10 ms faded
  out), relative to the sound itself.
- **The time it takes**, moving air against the effective speed of sound, from the
  free field's runs and the facade's.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple

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


class Made(NamedTuple):
    """A wave solver a run made: the solver, where its grid starts (m), its cell (m), its
    shape, and how long making it took (s; laying the grid out and, in moving air round
    solids, solving the wind)."""

    solver: object
    origin: np.ndarray
    cell: float
    shape: tuple
    seconds: float


def with_solver(run) -> tuple:
    """What ``run()`` gives, and the wave solver it made (the last one; the ones before
    it find the model's open sides)."""
    from acoustic_spaces import _native

    made = []
    real = _native.WaveSolver

    def make(*args, **kwargs):
        started = time.perf_counter()
        solver = real(*args, **kwargs)
        made.append(
            Made(
                solver,
                np.asarray(args[5], dtype=float),
                float(args[6]),
                tuple(args[7:10]),
                time.perf_counter() - started,
            )
        )
        return solver

    _native.WaveSolver = make
    try:
        out = run()
    finally:
        _native.WaveSolver = real
    return out, made[-1]


class Wind:
    """The wind a run's solver made flow round its model's solids, from the solver (see
    :func:`with_solver`) and the run's sample rate: ``faces[d]`` (nz, ny, nx) the
    wind through each cell's face towards +x, +y or +z (m/s), ``types`` the cells'
    (0 solid, 1 and 2 the air that moves, 3 the matched layer), and each cell's own
    wind, the mean of its two faces' along each axis."""

    def __init__(self, made: Made, rate: float) -> None:
        layout = made.solver.layout()
        nx, ny, nz = (int(n) for n in made.shape)
        self.origin, self.cell = made.origin, made.cell
        cell = made.cell
        self.types = layout["types"].reshape(nz, ny, nx)
        self.faces = [
            layout[f"wind_{axis}"].reshape(nz, ny, nx).astype(float) * cell * rate for axis in "xyz"
        ]
        self.cells = np.zeros((nz, ny, nx, 3))
        for d, face in enumerate(self.faces):
            below = np.zeros_like(face)
            np.moveaxis(below, 2 - d, 0)[1:] = np.moveaxis(face, 2 - d, 0)[:-1]
            self.cells[..., d] = 0.5 * (face + below)

    def centres(self) -> np.ndarray:
        """The cells' centres (nz, ny, nx, 3)."""
        nz, ny, nx = self.types.shape
        z, y, x = np.meshgrid(*(np.arange(n) + 0.5 for n in (nz, ny, nx)), indexing="ij")
        return self.origin + self.cell * np.stack([x, y, z], axis=-1)

    def at(self, points: np.ndarray) -> np.ndarray:
        """The wind (m/s) of the cells that hold ``points`` (n, 3)."""
        k = np.floor((np.asarray(points) - self.origin) / self.cell).astype(np.int64)
        return self.cells[k[:, 2], k[:, 1], k[:, 0]]


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


def spectrum(x: np.ndarray, rate: float, lead_s: float = 0.0) -> np.ndarray:
    """The frequency response at FREQS of the response x sampled at rate (Hz), which
    starts lead_s (s) before emission."""
    return np.exp(-2j * np.pi * np.outer(FREQS, np.arange(len(x)) / rate - lead_s)) @ x


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
        calm[name] = spectrum(response.signals[:, 0], response.rate, response.lead_s)
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
        (windy, info), made = with_solver(
            lambda wind_model=wind_model: _run(
                _scene(listeners, WIND, wind_model), length, wind_model
            )
        )
        seconds = time.perf_counter() - started
        for (name, (x, y, z)), response in zip(listeners.items(), windy, strict=True):
            # The still air's sound as the wind changes it: moved by the direct sound's
            # change of arrival and scaled by R / R* (the ground's image changes alike,
            # to a few microseconds).
            r = math.sqrt(x * x + y * y + (z - ZS) ** 2)
            tau, r_star = convected(c, WIND, x, y, z - ZS)
            times = tau + np.arange(-0.03, 0.05, STEP)
            a = in_band(spectrum(response.signals[:, 0], response.rate, response.lead_s), times)
            b = in_band(calm[name], times, tau - r / c) * r / r_star
            row = {
                "wind": wind_model,
                "listener": name,
                "waveform_db": 10.0 * math.log10(np.sum((a - b) ** 2) / np.sum(b**2)),
                "level_db": 10.0 * math.log10(np.sum(a**2) / np.sum(b**2)),
                "arrival_ms": 1e3 * _lag(a, b, 1.0 / STEP),
                "seconds": seconds,
                "set_up_s": made.seconds,
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


def facade_still_paths() -> list[np.ndarray]:
    """The still air's paths from the source off the facade to the listener, each as its
    corners (n, 3): the source, where it meets the facade and the ground, the listener.
    One for each pair of the source or its image under the ground and the listener or
    its image, as :func:`facade_times` counts them."""
    xl, _, zl = FACADE_LISTENER
    out = []
    for zs in (ZS, -ZS):
        for zr in (zl, -zl):
            # Unfolded: a straight line to the listener's image behind the facade.
            start, end = np.array([0.0, 0.0, zs]), np.array([2.0 * FACADE_X - xl, 0.0, zr])
            turns = {0.0, 1.0, FACADE_X / end[0]}
            if zs * zr < 0.0:
                turns.add(zs / (zs - zr))
            corners = start + np.array(sorted(turns))[:, None] * (end - start)
            corners[:, 0] = np.where(
                corners[:, 0] > FACADE_X, 2.0 * FACADE_X - corners[:, 0], corners[:, 0]
            )
            corners[:, 2] = np.abs(corners[:, 2])
            out.append(corners)
    return out


def path_time(corners: np.ndarray, c: float, wind, step: float) -> float:
    """The time (s) along the path through ``corners`` (n, 3) in the wind ``wind(points)``
    (m/s at points (m, 3)), in pieces of at most ``step`` (m): each at the speed of a
    ray along it, v . s + sqrt(c^2 - |v|^2 + (v . s)^2)."""
    total = 0.0
    for a, b in itertools.pairwise(corners):
        length = float(np.linalg.norm(b - a))
        pieces = max(1, math.ceil(length / step))
        s = (b - a) / length
        v = wind(a + (np.arange(pieces)[:, None] + 0.5) / pieces * (b - a))
        along = v @ s
        speed = along + np.sqrt(c * c - np.sum(v * v, axis=1) + along**2)
        total += float(np.sum(length / pieces / speed))
    return total


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
    calm_spectrum = spectrum(calm.signals[:, 0], calm.rate, calm.lead_s)
    paths = facade_still_paths()
    rows = []
    for wind_model in WIND_MODELS:
        started = time.perf_counter()
        room = _scene(listeners, WIND, wind_model, True)
        ((response,), info), made = with_solver(
            lambda room=room, wind_model=wind_model: _run(room, length, wind_model)
        )
        seconds = time.perf_counter() - started
        row = {"wind": wind_model, "exact_change_ms": 1e3 * change, "set_up_s": made.seconds}
        reference = change
        if wind_model == "moving":
            # Through the wind the solver made: the time along the still air's paths. The
            # same sum through the uniform wind, against its exact rays, says what
            # keeping the still air's paths costs.
            wind = Wind(made, response.rate)
            sums = {
                name: np.mean([path_time(path, c, field, 0.25 * made.cell) for path in paths])
                for name, field in (
                    ("still", lambda q: np.zeros_like(q)),
                    ("uniform", lambda q: np.array([WIND, 0.0, 0.0]) + np.zeros_like(q)),
                    ("round", wind.at),
                )
            }
            reference = sums["round"] - sums["still"]
            row["round_change_ms"] = 1e3 * reference
            row["paths_miss_ms"] = 1e3 * (sums["uniform"] - sums["still"] - change)
            row["fastest_m_s"] = info["wave_wind_fastest_m_s"]
            row["sweeps"] = made.solver.wind_sweeps
            at = wind.at(np.array([[0.0, 0.0, ZS], FACADE_LISTENER, [FACADE_X - 5.0, 0.0, ZS]]))
            row["wind_at_source_m_s"], row["wind_at_listener_m_s"], row["wind_at_facade_m_s"] = (
                float(v) for v in at[:, 0]
            )
        # The still air's facade sound moved by the reference's change of its arrival:
        # the windy run's lag behind it is the error.
        expected = in_band(calm_spectrum, times, reference)
        got = in_band(spectrum(response.signals[:, 0], response.rate, response.lead_s), times)
        error = 1e3 * _lag(got, expected, 1.0 / STEP)
        row |= {
            "change_ms": 1e3 * reference + error,
            "error_ms": error,
            "seconds": seconds,
            "cells": info["wave_cells"],
        }
        rows.append(row)
        if wind_model == "moving":
            print(
                f"moving   : its wind is {row['wind_at_listener_m_s']:.1f} m/s at the listener, "
                f"{row['wind_at_source_m_s']:.1f} at the source, {row['wind_at_facade_m_s']:.1f} "
                f"5 m before the facade and up to {row['fastest_m_s']:.1f} round the building; "
                f"along the still air's paths through it the facade's sound arrives "
                f"{row['round_change_ms']:+.2f} ms later than in still air (the same sum "
                f"through the uniform wind is {row['paths_miss_ms']:+.3f} ms off its exact rays)",
                flush=True,
            )
        against = f"ray theory in the uniform wind {row['exact_change_ms']:+.2f} ms"
        if wind_model == "moving":
            against = f"ray theory through its wind {1e3 * reference:+.2f} ms; {against[11:]}"
        print(
            f"{wind_model:9s}: the facade's sound arrives {row['change_ms']:+.2f} ms later than "
            f"in still air ({against}): off by {row['error_ms']:+.2f} ms",
            flush=True,
        )
    return rows


#: The dome: its radius (m), and the bands (Hz) whose grids its wind is read on.
DOME_RADIUS = 4.0
DOME_BANDS = (63, 125)
#: The shells round the dome the wind is compared over, in radii from its centre; the
#: first ends two cells from its surface.
DOME_SHELLS = (1.5, 2.0, 3.0)


def dome(path: Path, wind: float, span: float = 4.0 * DOME_RADIUS) -> Room:
    """A rigid half sphere of DOME_RADIUS on rigid ground, written to ``path`` (an OBJ
    file), in a uniform wind ``wind`` (m/s) along +x, a source and a listener beside
    it; the ground reaches ``span`` (m) from its centre, and the sky is 2 m less up."""
    rings, around = 24, 64
    lines, faces = [], []
    for i in range(rings):
        latitude = 0.5 * math.pi * i / rings
        for j in range(around):
            longitude = 2.0 * math.pi * j / around
            lines.append(
                f"v {DOME_RADIUS * math.cos(latitude) * math.cos(longitude)} "
                f"{DOME_RADIUS * math.cos(latitude) * math.sin(longitude)} "
                f"{DOME_RADIUS * math.sin(latitude)}"
            )
    lines.append(f"v 0 0 {DOME_RADIUS}")
    top = rings * around + 1
    for i in range(rings):
        for j in range(around):
            a, b = i * around + j + 1, i * around + (j + 1) % around + 1
            last = i == rings - 1
            faces.append(f"f {a} {b} {top}" if last else f"f {a} {b} {b + around} {a + around}")
    path.write_text("\n".join(lines + faces) + "\n")
    data = {
        "geometry": {
            "ground": {"extent": [[-span, -span], [span, span]], "sky": span - 2.0},
            "file": str(path),
            "up_axis": "z",
        },
        "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": [-2.0 * DOME_RADIUS, 0.0, 2.0]}],
        "listeners": [{"name": "l", "position": [2.0 * DOME_RADIUS, 0.0, 2.0]}],
        "atmosphere": {"wind": {"speed_m_s": wind, "direction_deg": 0.0, "profile": "constant"}},
    }
    return Room.from_scene(Scene.model_validate(data))


def sphere_flow(points: np.ndarray, wind: float, radius: float) -> np.ndarray:
    """An ideal fluid's velocity at ``points`` (..., 3) past a sphere of ``radius`` at
    the origin, ``wind`` along +x far from it: the gradient of U (x + a^3 x / (2 r^3))."""
    r = np.linalg.norm(points, axis=-1)[..., None]
    out = 0.5 * wind * radius**3 * (-3.0 * points[..., :1] * points / r**5)
    out[..., 0] += wind + 0.5 * wind * radius**3 / r[..., 0] ** 3
    return out


def check_dome(bands=DOME_BANDS, span: float = 4.0 * DOME_RADIUS) -> list[dict]:
    rows = []
    with tempfile.TemporaryDirectory() as folder:
        room = dome(Path(folder) / "dome.obj", WIND, span)
        source = np.array(room.scene.sources[0].position, dtype=float)
        listener = np.array(room.scene.listeners[0].position, dtype=float)
        for band in bands:
            (responses, info), made = with_solver(
                lambda band=band: wave_responses(
                    TraceModel.from_room(room),
                    source,
                    listener[None, :],
                    ["l"],
                    max_band_hz=band,
                    length_s=0.02,
                    sample_rate=RATE,
                    speed_of_sound=room.speed_of_sound(),
                    wind="moving",
                )
            )
            wind = Wind(made, responses[0].rate)
            cell, solver = made.cell, made.solver
            centres = wind.centres()
            # The faces between two cells of moving air over the ground, each against
            # the exact flow's component across it at its centre.
            air = ((wind.types == 1) | (wind.types == 2)) & (centres[..., 2] > 0.0)
            radii, misses = [], []
            for d, face in enumerate(wind.faces):
                ahead = np.zeros_like(air)
                np.moveaxis(ahead, 2 - d, 0)[:-1] = np.moveaxis(air, 2 - d, 0)[1:]
                at = centres[air & ahead]
                at[:, d] += 0.5 * cell
                radii.append(np.linalg.norm(at, axis=1) / DOME_RADIUS)
                misses.append((face[air & ahead] - sphere_flow(at, WIND, DOME_RADIUS)[:, d]) / WIND)
            r, miss = np.concatenate(radii), np.concatenate(misses)
            near = 1.0 + 2.0 * cell / DOME_RADIUS
            row = {
                "band_hz": band,
                "cell_m": cell,
                "sweeps": solver.wind_sweeps,
                "fastest_over_wind": info["wave_wind_fastest_m_s"] / WIND,
                "shells": [],
            }
            for low, high in itertools.pairwise((0.0, near, *DOME_SHELLS, math.inf)):
                pick = (r >= low) & (r < high)
                row["shells"].append(
                    {
                        "from_radii": low,
                        "to_radii": high,
                        "rms_percent": 100.0 * math.sqrt(float(np.mean(miss[pick] ** 2))),
                        "largest_percent": 100.0 * float(np.abs(miss[pick]).max()),
                        "faces": int(pick.sum()),
                    }
                )
            rows.append(row)
            print(
                f"{band} Hz band's grid (cells of {100 * cell:.0f} cm, {DOME_RADIUS / cell:.0f} "
                f"to the radius; {solver.wind_sweeps} sweeps): the fastest wind is "
                f"{row['fastest_over_wind']:.2f} of the given one (exact 1.50, over the top)"
            )
            for shell in row["shells"]:
                where = (
                    "within two cells of the dome"
                    if shell["from_radii"] == 0.0
                    else f"from {shell['from_radii']:.2f} radii"
                    + ("" if math.isinf(shell["to_radii"]) else f" to {shell['to_radii']:.2f}")
                )
                print(
                    f"  {where}: off the exact flow by {shell['rms_percent']:.2f} % of the "
                    f"wind (rms), {shell['largest_percent']:.1f} % at most "
                    f"({shell['faces']:,} faces)",
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


#: The room with an opening: where its source and its listener are (m).
ROOM_SOURCE = (-0.7, -0.4, 1.2)
ROOM_LISTENER = (1.2, 0.9, 1.5)


def room_with_an_opening(path: Path, wind: float, profile: str = "constant") -> Room:
    """A rigid room on rigid ground, 4 x 4 x 3 m inside with walls and a roof 0.8 m thick
    and an opening 0.8 x 0.8 m in its wall downwind, written to ``path`` (an OBJ file),
    the source and a listener in it, in a wind ``wind`` (m/s) along +x: uniform, or
    with ``profile`` the atmosphere's profile that reaches it 10 m up."""
    boxes = [
        ((-2.8, -2.8, 0.0), (-2.0, 2.8, 3.8)),
        ((-2.0, -2.8, 0.0), (2.0, -2.0, 3.8)),
        ((-2.0, 2.0, 0.0), (2.0, 2.8, 3.8)),
        ((-2.0, -2.0, 3.0), (2.0, 2.0, 3.8)),
        ((2.0, -2.8, 0.0), (2.8, 2.8, 1.0)),
        ((2.0, -2.8, 1.8), (2.8, 2.8, 3.8)),
        ((2.0, -2.8, 1.0), (2.8, -0.4, 1.8)),
        ((2.0, 0.4, 1.0), (2.8, 2.8, 1.8)),
    ]
    sides = ((1, 2, 3, 4), (5, 6, 7, 8), (1, 2, 6, 5), (2, 3, 7, 6), (3, 4, 8, 7), (4, 1, 5, 8))
    lines = []
    for k, ((x0, y0, z0), (x1, y1, z1)) in enumerate(boxes):
        plan = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        lines += [f"v {x} {y} {z}" for z in (z0, z1) for x, y in plan]
        lines += ["f " + " ".join(str(8 * k + corner) for corner in side) for side in sides]
    path.write_text("\n".join(lines) + "\n")
    data = {
        "geometry": {
            "ground": {"extent": [[-5, -5], [5, 5]], "sky": 5},
            "file": str(path),
            "up_axis": "z",
        },
        "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": list(ROOM_SOURCE)}],
        "listeners": [{"name": "in", "position": list(ROOM_LISTENER)}],
        "atmosphere": {"wind": {"speed_m_s": wind, "direction_deg": 0.0, "profile": profile}},
    }
    return Room.from_scene(Scene.model_validate(data))


def check_room_with_an_opening(winds=(5.0, 15.0, 30.0), length_s: float = 3.0) -> list[dict]:
    rows = []
    print(
        "| wind (m/s) | fastest round the room (m/s) | fastest in the room (m/s) | moving "
        "air: level, last quarter second over the second (dB) | effective speed of sound "
        "(dB) |\n|---:|---:|---:|---:|---:|"
    )
    with tempfile.TemporaryDirectory() as folder:
        for wind in winds:
            room = room_with_an_opening(Path(folder) / "room.obj", wind)
            row: dict = {"wind_m_s": wind}
            for wind_model in ("moving", "effective"):
                ((response,), info), solver = with_solver(
                    lambda room=room, wind_model=wind_model: wave_responses(
                        TraceModel.from_room(room),
                        np.array(ROOM_SOURCE),
                        np.array([ROOM_LISTENER]),
                        ["in"],
                        max_band_hz=63,
                        length_s=length_s,
                        sample_rate=RATE,
                        speed_of_sound=room.speed_of_sound(),
                        wind=wind_model,
                    )
                )
                quarter = int(0.25 * response.rate)
                pressure = response.signals[round(response.lead_s * response.rate) :, 0]
                rms = [
                    math.sqrt(float(np.mean(pressure[k * quarter : (k + 1) * quarter] ** 2)))
                    for k in range(int(length_s / 0.25))
                ]
                row[f"{wind_model}_change_db"] = 20.0 * math.log10(rms[-1] / rms[1])
                if wind_model == "moving":
                    wind_made = Wind(solver, response.rate)
                    x, y, z = np.moveaxis(wind_made.centres(), -1, 0)
                    inside = (
                        ((wind_made.types == 1) | (wind_made.types == 2))
                        & (np.abs(x) < 2.0)
                        & (np.abs(y) < 2.0)
                        & (z > 0.0)
                        & (z < 3.0)
                    )
                    row["fastest_m_s"] = info["wave_wind_fastest_m_s"]
                    row["fastest_inside_m_s"] = float(
                        np.linalg.norm(wind_made.cells[inside], axis=1).max()
                    )
            rows.append(row)
            print(
                f"| {wind:g} | {row['fastest_m_s']:.1f} | {row['fastest_inside_m_s']:.2f} | "
                f"{row['moving_change_db']:+.1f} | {row['effective_change_db']:+.1f} |",
                flush=True,
            )
    return rows


#: The border's check: its uniform winds (m/s), its cell (m), its runs (s), its band's
#: edges (Hz, with the widths over which they fade) and how many cells larger all round
#: the grid of reference is.
BORDER_WINDS = (5.0, 10.0, 20.0, 40.0)
BORDER_CELL, BORDER_S, BORDER_FAR = 0.3, 0.07, 45
BORDER_BAND = ((25.0, 10.0), (150.0, 30.0))


def check_border() -> list[dict]:
    """What the border of the moving air sends back: the plate of ``wave_solver.py``'s
    open model in still air and in uniform winds along the grid and at 45 degrees to it,
    on a grid with the default matched layer against one ``BORDER_FAR`` cells larger all
    round, whose border sends nothing back in the time of the runs. What differs at the
    three listeners, relative to the sound itself (the runs' last 10 ms fade out, as in
    that check), and half the wind's Mach number."""
    import wave_solver

    from acoustic_spaces.engine.wave import COURANT, LAYER_CELLS

    c, h = wave_solver.C, BORDER_CELL
    rate = c / (COURANT * h)
    n_fft = 1 << 12
    (low, rise), (high, fall) = BORDER_BAND
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gain = np.clip((f - low) / rise, 0, 1) * np.clip((high - f) / fall, 0, 1)

    def band(pad: int, flow) -> np.ndarray:
        solver, _ = wave_solver.plate_solver(h, pad, LAYER_CELLS, flow=flow)
        x = wave_solver._run(solver, rate, h, BORDER_S)[0]
        x = x * wave_solver._fade(x.shape[1], rate, x.shape[1] / rate, 0.01)
        return np.fft.irfft(np.fft.rfft(x, n_fft, axis=1) * gain, n_fft, axis=1)[:, : x.shape[1]]

    def back(flow) -> list[float]:
        reference = band(BORDER_FAR + LAYER_CELLS, flow)
        miss = band(LAYER_CELLS, flow) - reference
        ratio = np.sum(miss**2, axis=1) / np.sum(reference**2, axis=1)
        return [float(x) for x in 10 * np.log10(ratio)]

    slant = math.sqrt(0.5)
    print(
        "| wind (m/s) | along the grid: listener 1 (dB) | 2 | 3 | at 45 degrees: 1 | 2 | 3 "
        "| half the Mach number (dB) |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    still = back(None)
    rows = [{"wind_m_s": 0.0, "along_db": still, "at_45_db": None, "half_mach_db": None}]
    print(f"| 0 | {' | '.join(f'{x:.1f}' for x in still)} | | | | |")
    for wind in BORDER_WINDS:
        shift = wind * COURANT / c  # cells per step
        along, at_45 = back((shift, 0.0)), back((shift * slant, shift * slant))
        half_mach = 20 * math.log10(wind / (2 * c))
        rows.append(
            {"wind_m_s": wind, "along_db": along, "at_45_db": at_45, "half_mach_db": half_mach}
        )
        print(f"| {wind:g} | {' | '.join(f'{x:.1f}' for x in along + at_45)} | {half_mach:.1f} |")
    return rows


def costs(free: list[dict], facade: list[dict]) -> dict:
    """How long moving air took against the effective speed of sound, from the free
    field's rows (the wind carried as given) and the facade's (the wind made to flow
    round the building, the solve of that wind included)."""
    out = {}
    for name, rows in (("free field", free), ("facade", facade)):
        took = {row["wind"]: row for row in rows}
        out[name] = {
            "effective_s": took["effective"]["seconds"],
            "moving_s": took["moving"]["seconds"],
            "ratio": took["moving"]["seconds"] / took["effective"]["seconds"],
            "effective_set_up_s": took["effective"]["set_up_s"],
            "moving_set_up_s": took["moving"]["set_up_s"],
            "cells": took["moving"]["cells"],
        }
        print(
            f"{name}: moving air {out[name]['moving_s']:.1f} s, the effective speed of sound "
            f"{out[name]['effective_s']:.1f} s ({out[name]['ratio']:.1f} times; "
            f"{out[name]['cells']:,.0f} cells); of those, making the solver "
            f"{out[name]['moving_set_up_s']:.1f} s and {out[name]['effective_set_up_s']:.1f} s"
        )
    return out


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
    print(f"\nThe wind round a dome of {DOME_RADIUS:g} m against an ideal fluid's past a sphere\n")
    round_dome = check_dome()
    print("\nA facade 40 m downwind, the listener 10 m upwind of the source\n")
    facade = check_facade()
    profiles = []
    if not args.skip_profiles:
        print("\nThe windy preset over grass against the parabolic equation, 63 Hz band\n")
        profiles = check_profiles()
    print("\nA room with an opening downwind, over 3 s\n")
    opening = check_room_with_an_opening()
    print("\nWhat the border of the moving air sends back, 25 to 150 Hz\n")
    border = check_border()
    print("\nThe time it takes (alone on the machine, or the times mean nothing)\n")
    took = costs(free, facade)
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        out = {
            "free_field": free,
            "dome": round_dome,
            "facade": facade,
            "profiles": profiles,
            "opening": opening,
            "border": border,
            "costs": took,
        }
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
