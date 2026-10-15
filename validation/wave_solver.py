"""Check the wave-based solver against theory.

    python validation/wave_solver.py [--json results.json] [--plot wave.png]

Nine checks, printed as Markdown tables, each with both of the solver's stencils (the
7-point one and the 19-point one) where both apply:

- the speed of sound along an axis and along a diagonal at about 8 cells per
  wavelength, from the modes of a rigid box, against each stencil's dispersion relation;
- the modes of a lightly damped 7 x 5 x 3 m box (10 cm cells) whose walls' admittance
  changes with frequency, from its ringing: frequency and decay rate of each mode that
  stands apart from the others, against perturbation theory: the decay rate
  delta = c Re(beta) / 2 * sum over the three axes of (4 / L if the mode varies along
  that axis, else 2 / L), beta the walls' admittance at the mode, and the rigid box's
  frequency moved by -c Im(beta) / (4 pi) times the same sum;
- the reflection off the end of a duct at normal incidence, against (1 - beta) / (1 + beta)
  for the end's specific admittance beta;
- the normal-incidence absorption of a duct's end whose admittance is fitted to a porous
  material's absorption from 63 to 500 Hz, against that admittance's, and the
  random-incidence absorption of the fit against the material's;
- the attenuation of a plane wave grazing the walls of a duct, against
  exp(-beta * perimeter / (2 * area) per metre);
- an open model (a plate in the open): what comes back from the grid's edge, with and
  without the matched layer, against a grid too large for anything to come back (the
  7-point stencil, the only one with a matched layer);
- the level of the direct sound in the 63 and 125 Hz bands, in a large and very
  absorbing room, against the free-field level the rays give;
- a 7 x 5 x 3 m box with 15 % absorption on every surface, simulated with the default
  settings (the 63 and 125 Hz bands from the wave solver), against the sum of the box's
  modes with those decay rates: G, EDT and T20 in both bands, also with the 7-point
  stencil, with twice as many cells per wavelength and from the rays alone; with
  ``--plot`` (needs matplotlib), the transfer functions from 15 to 250 Hz;
- the run count and time of a box whose absorption differs in every band up to 500 Hz.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np

from acoustic_spaces import Room, Scene, _native, simulate
from acoustic_spaces.engine.impedance import (
    Admittance,
    admittance,
    fit_admittance,
)
from acoustic_spaces.engine.wave import (
    COURANT,
    LAYER_CELLS,
    LAYER_DAMPING,
    STENCILS,
    source_signal,
)
from acoustic_spaces.geometry import box_mesh
from acoustic_spaces.parameters import from_impulse_response

C = 343.0
FS = 48_000
BOX = (7.0, 5.0, 3.0)
#: Each stencil's Courant number, by its points.
COURANTS = dict(STENCILS.values())


def _solver(size, admittances, source, probes, cell, stencil: int = 7):
    """A wave solver with the ``stencil``-point stencil in the box from (0, 0, 0) to
    ``size``, and its sample rate. ``admittances(mesh)`` gives each face's admittance: a
    number, or an Admittance."""
    mesh = box_mesh(size)
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((np.array(size) - origin) / cell).astype(int) + 2
    rate = C / (COURANTS[stencil] * cell)
    walls = [w if isinstance(w, Admittance) else Admittance(float(w)) for w in admittances(mesh)]
    branches = [np.zeros((0, 3))] + [w.branches(rate) for w in walls]
    owners = [np.zeros(0, np.int64)] + [
        np.full(len(w.conductance), k, np.int64) for k, w in enumerate(walls)
    ]
    solver = _native.WaveSolver(
        bvh,
        np.arange(len(walls), dtype=np.int64),
        np.array([w.beta for w in walls]),
        np.ascontiguousarray(np.vstack(branches)),
        np.concatenate(owners),
        origin,
        cell,
        *map(int, shape),
        np.asarray(source, dtype=float),
        np.asarray(probes, dtype=float).reshape(-1, 3),
        stencil=stencil,
    )
    return solver, rate


def _run(solver, rate: float, cell: float, seconds: float) -> tuple[np.ndarray, int]:
    signal, delay = source_signal(rate, cell, COURANTS[solver.stencil])
    steps = int(seconds * rate)
    drive = np.zeros(steps, np.float32)
    drive[: len(signal)] = signal[:steps]
    return solver.run(drive), delay


def box_modes(size, f_max: float, c: float = C) -> tuple[np.ndarray, np.ndarray]:
    """Mode numbers (modes, 3) and frequencies of a rigid box below ``f_max``."""
    top = [int(2 * f_max * s / c) + 1 for s in size]
    numbers = np.array(list(itertools.product(*(range(t + 1) for t in top))))[1:]
    f = c / 2 * np.sqrt(((numbers / np.array(size)) ** 2).sum(axis=1))
    keep = f < f_max
    return numbers[keep], f[keep]


def modal_decay(numbers: np.ndarray, beta, size, c: float = C) -> np.ndarray:
    """Pressure decay rate (1/s) of each mode of a box whose walls all have the small
    specific admittance ``beta`` (its real part, at each mode; first-order perturbation
    theory)."""
    per_axis = np.where(numbers > 0, 4.0, 2.0) / np.array(size)
    return c * np.real(beta) / 2 * per_axis.sum(axis=1)


def modal_shift(numbers: np.ndarray, beta, size, c: float = C) -> np.ndarray:
    """How far (Hz) walls of the small admittance ``beta`` move each mode of a box: down
    where they give like a spring (positive imaginary part), up where they weigh like a
    mass (first-order perturbation theory, the other half of the decay rate's)."""
    per_axis = np.where(numbers > 0, 4.0, 2.0) / np.array(size)
    return -c * np.imag(beta) / (4 * math.pi) * per_axis.sum(axis=1)


def _peak(f: np.ndarray, magnitude: np.ndarray, centre: float, half_width: float) -> float:
    """The frequency of the highest peak of ``magnitude`` within ``half_width`` of
    ``centre``, between the bins (a parabola through the log magnitude round it)."""
    near = np.flatnonzero(np.abs(f - centre) < half_width)
    j = int(near[np.argmax(magnitude[near])])
    a, b, c = np.log(magnitude[j - 1 : j + 2])
    return float(f[j] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0]))


#: Each stencil's weights: of an axial link and of an edge link (scaled by the Courant
#: number squared, as the update uses them).
WEIGHTS = {7: (1.0 / 3.0, 0.0), 19: (0.25, 0.125)}


def grid_frequency(numbers: np.ndarray, size, cell: float, rate: float, stencil: int):
    """The frequencies of a rigid box's modes on the ``stencil``-point grid with the
    walls half a cell past the last centres, from the stencil's dispersion relation
    2 - 2 cos(w T) = sum over the links of g_k (1 - cos(k . d_k))."""
    axial, edge = WEIGHTS[stencil]
    theta = np.pi * numbers * cell / np.array(size)
    cos = np.cos(theta)
    total = 2 * axial * (1 - cos).sum(axis=1)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        total += 4 * edge * (1 - cos[:, a] * cos[:, b])
    return np.arccos(1 - total / 2) * rate / (2 * math.pi)


def relation_error(direction, points_per_wavelength: float, stencil: int) -> float:
    """How far off the speed of sound is along ``direction`` on the ``stencil``-point
    grid at its Courant limit, with ``points_per_wavelength`` cells per wavelength."""
    unit = np.asarray(direction, float) / np.linalg.norm(direction)
    cell, courant = 1.0, COURANTS[stencil]
    rate = 1.0 / courant  # c = 1
    # A mode of a box so large that k h along each axis is what the direction asks.
    size = np.ones(3)
    numbers = (2 * unit / points_per_wavelength)[None, :]  # k h / pi along each axis
    f = grid_frequency(numbers, size, cell, rate, stencil)[0]
    return float(f * points_per_wavelength - 1)


def dispersion() -> dict:
    """The speed of sound along an axis and along a diagonal at about 8 cells per
    wavelength: modes (6, 0, 0) and (4, 4, 0) of a rigid 2.4 x 2.4 x 0.2 m box (10 cm
    cells), from the peaks of its ringing for 8 s, against the exact frequencies and
    against each stencil's dispersion relation."""
    size, cell = (2.4, 2.4, 0.2), 0.1
    source, probe = (0.13, 0.17, 0.1), (2.27, 2.21, 0.1)  # near corners: every mode rings
    picks = {"axis": (6, 0, 0), "diagonal": (4, 4, 0)}
    numbers, exact = box_modes(size, 700.0)
    rows = []
    for stencil in (7, 19):
        solver, rate = _solver(size, lambda m: [0.0] * len(m.faces), source, probe, cell, stencil)
        pressure = _run(solver, rate, cell, 8.0)[0][0]
        n_fft = 1 << int(math.ceil(math.log2(8 * len(pressure))))
        spectrum = np.abs(np.fft.rfft(pressure * np.hanning(len(pressure)), n_fft))
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        on_grid = grid_frequency(numbers, size, cell, rate, stencil)
        for direction, mode in picks.items():
            k = int(np.flatnonzero((numbers == mode).all(axis=1))[0])
            apart = np.abs(on_grid - on_grid[k])
            assert apart[apart > 1e-6].min() > 1.0, "another mode is too close to tell apart"
            peak = _peak(f, spectrum, on_grid[k], 0.5)
            rows.append(
                {
                    "stencil": stencil,
                    "direction": direction,
                    "mode": list(mode),
                    "cells_per_wavelength": float(C / (exact[k] * cell)),
                    "exact_hz": float(exact[k]),
                    "relation_hz": float(on_grid[k]),
                    "fdtd_hz": float(peak),
                }
            )
    directions = {"axis": (1, 0, 0), "face diagonal": (1, 1, 0), "body diagonal": (1, 1, 1)}
    at_8 = {
        str(stencil): {name: relation_error(d, 8.0, stencil) for name, d in directions.items()}
        for stencil in (7, 19)
    }
    return {"rows": rows, "at_8": at_8}


#: The walls of the lightly damped box: a little constant absorption and a broad
#: resonance at 60 Hz, so the modes from 24 to 90 Hz decay at rates up to five times apart.
MODE_WALLS = Admittance(0.002, np.array([0.008]), np.array([1.5]), np.array([60.0]))


def modes(stencil: int = 7) -> list[dict]:
    cell = 0.1
    solver, rate = _solver(
        BOX,
        lambda m: [MODE_WALLS] * len(m.faces),
        (0.33, 0.27, 0.21),
        (6.61, 4.77, 2.83),
        cell,
        stencil,
    )
    pressure = _run(solver, rate, cell, 8.0)[0][0]
    n_fft = 1 << int(math.ceil(math.log2(2 * len(pressure))))
    spectrum = np.fft.rfft(pressure, n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    t = np.arange(len(pressure)) / rate
    numbers, rigid = box_modes(BOX, 125.0)
    walls = MODE_WALLS(rigid, rate)
    decay = modal_decay(numbers, walls, BOX)
    frequencies = rigid + modal_shift(numbers, walls, BOX)
    rows = []
    for n, f_rigid, f0, d0, y in zip(numbers, rigid, frequencies, decay, walls, strict=True):
        if np.sort(np.abs(rigid - f_rigid))[1] < 2.5:
            continue  # too close to another mode to tell them apart
        peak = _peak(f, np.abs(spectrum), f0, 1.0)
        # The mode alone: a narrow band round it, and the slope of its envelope.
        gain = np.exp(-0.5 * ((f - peak) / 0.4) ** 2)
        analytic = np.fft.ifft(np.concatenate([2 * spectrum * gain, np.zeros(n_fft - len(f))]))
        level = 20 * np.log10(np.abs(analytic[: len(pressure)]) + 1e-30)
        # From 1 s on, over 50 dB at most: the fastest modes reach the float noise
        # floor before 6 s.
        start = int(rate)
        below = np.flatnonzero(level[start:] < level[start] - 50.0)
        fit = (t > 1.0) & (t < min(6.0, 1.0 + (below[0] if len(below) else len(t)) / rate))
        slope = np.polyfit(t[fit], level[fit], 1)[0]
        rows.append(
            {
                "mode": [int(k) for k in n],
                "admittance": float(y.real),
                "rigid_hz": float(f_rigid),
                "theory_hz": float(f0),
                "fdtd_hz": peak,
                "theory_decay": float(d0),
                "fdtd_decay": float(-slope * math.log(10) / 20),
            }
        )
    return sorted(rows, key=lambda row: row["theory_hz"])


def duct(stencil: int = 7) -> list[dict]:
    size, cell = (80.0, 0.6, 0.6), 0.1
    rows = []
    for beta in (0.0, 0.1, 0.25, 0.5, 1.0):

        def admittances(mesh, beta=beta):
            x = mesh.vertices[mesh.faces][:, :, 0]
            far = np.all(np.isclose(x, size[0]), axis=1)
            near = np.all(np.isclose(x, 0.0), axis=1)
            return np.where(far, beta, np.where(near, 1.0, 0.0))

        solver, rate = _solver(size, admittances, (10.0, 0.3, 0.3), (30.0, 0.3, 0.3), cell, stencil)
        recorded, delay = _run(solver, rate, cell, 0.45)
        pressure = recorded[0]
        n_fft = 1 << 14
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
        pressure = np.fft.irfft(np.fft.rfft(pressure, n_fft) * gain, n_fft)[: len(pressure)]
        t = (np.arange(len(pressure)) - delay) / rate

        def gated(path: float, t=t, pressure=pressure) -> float:
            x = t - path / C
            window = np.where(np.abs(x) < 0.04, 0.5 + 0.5 * np.cos(np.pi * x / 0.04), 0.0)
            return float(np.sum((pressure * window) ** 2))

        rows.append(
            {
                "admittance": beta,
                "theory": (1 - beta) / (1 + beta),
                "fdtd": math.sqrt(gated(120.0) / gated(20.0)),
            }
        )
    return rows


#: The porous material whose fitted admittance ends the duct (63 to 500 Hz).
POROUS = (0.05, 0.15, 0.4, 0.7)


def _plane_wave_runs(size, end, cell, source_x, probe_x, seconds, stencil):
    """The pressure at the probe in a duct along x with a matched near end (admittance 1)
    and, at the far end, the Admittance ``end`` (rigid for None)."""
    y = size[1] / 2

    def admittances(mesh):
        x = mesh.vertices[mesh.faces][:, :, 0]
        near = np.all(np.isclose(x, 0.0), axis=1)
        far = np.all(np.isclose(x, size[0]), axis=1)
        out = np.where(near, 1.0, 0.0).astype(object)
        out[far] = end if end is not None else 0.0
        return list(out)

    solver, rate = _solver(size, admittances, (source_x, y, y), (probe_x, y, y), cell, stencil)
    return _run(solver, rate, cell, seconds)[0][0], rate


def grid_reflection(
    f: np.ndarray, rate: float, y: np.ndarray, courant: float = COURANT
) -> np.ndarray:
    """Reflection at normal incidence off a wall of admittance y (the scheme's at that
    rate, with time going as e^(-i w t): the conjugate of Y(i w)) in the grid: a plane
    wave along an axis, the wall half a cell past the last centre. Both stencils carry
    such a wave as the one-dimensional scheme at their Courant number does."""
    phi = 2 * np.pi * f / rate
    shift = np.exp(2j * np.arcsin(np.minimum(np.sin(phi / 2) / courant, 1.0)))  # one cell
    c = 1j * np.sin(phi) * y / courant
    return (c - shift + 1) / (1 / shift - 1 - c)


def impedance_duct(stencil: int = 7) -> dict:
    """Normal-incidence absorption of a duct's end with the fitted admittance of POROUS,
    against that admittance's and against the grid's theory for it.

    The reflected wave is the far end's run less a run in a duct long enough for nothing
    to come back from its far end in time; the incident wave is that run before the sound
    off the near end arrives. Each over a rigid end's, which the grid reflects whole at
    every frequency, so what the grid does on the way cancels out.
    """
    cell, width = 0.05, 0.2
    courant = COURANTS[stencil]
    rate = C / (courant * cell)
    fit = fit_admittance(POROUS, 8, rate)
    source_x, probe_x, seconds = 60.0, 110.0, 0.6
    run = (cell, source_x, probe_x, seconds, stencil)
    reference, _ = _plane_wave_runs((200.0, width, width), None, *run)
    incident = np.where(np.arange(len(reference)) / rate < 0.48, reference, 0.0)
    n_fft = 1 << 15
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    band = (f > 30) & (f < 720)
    f = f[band]

    def reflection(end):
        far, _ = _plane_wave_runs((130.0, width, width), end, *run)
        return np.fft.rfft(far - reference, n_fft)[band] / np.fft.rfft(incident, n_fft)[band]

    measured = 1.0 - np.abs(reflection(fit) / reflection(None)) ** 2
    y = fit(f)
    theory = 1.0 - np.abs((1.0 - y) / (1.0 + y)) ** 2
    grid = 1.0 - np.abs(grid_reflection(f, rate, np.conj(fit(f, rate)), courant)) ** 2
    shown = [31.5, 44, 63, 88, 125, 177, 250, 354, 500, 707]
    rows = []
    for frequency in shown:
        k = int(np.argmin(np.abs(f - frequency)))
        rows.append(
            {
                "frequency_hz": frequency,
                "theory": float(theory[k]),
                "grid": float(grid[k]),
                "fdtd": float(measured[k]),
            }
        )
    below = f < 360
    return {
        "rows": rows,
        "max_difference": float(np.max(np.abs(measured - theory))),
        "max_difference_below_360": float(np.max(np.abs(measured - theory)[below])),
        "max_difference_grid": float(np.max(np.abs(measured - grid))),
        "material": list(POROUS),
        "fitted": [float(a) for a in fit.achieved],
        "branches": len(fit.conductance),
    }


def open_boundary() -> list[dict]:
    """A rigid 8 x 6 m plate in the open, 25 to 250 Hz at three listeners round it: what
    differs from a grid 16 m larger all round (nothing comes back from its edge in the 90
    ms the runs last), with the grid's edge alone and with matched layers."""
    h, seconds = 0.2, 0.09
    vertices = np.array([[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0]], float)
    bvh = _native.Bvh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    source = np.array([2.0, 2.0, 1.5])
    listeners = np.array([[6.0, 4.0, 1.2], [4.0, 3.0, 3.0], [7.5, 1.0, 0.5]])
    points = np.vstack([source, listeners])
    low = np.minimum(vertices.min(0), points.min(0) - 2 * h)
    high = np.maximum(vertices.max(0), points.max(0) + 2 * h)
    rate = C / (COURANT * h)

    def run(pad: int, layer: int, damping: float = LAYER_DAMPING):
        origin = low - h * (2.0 + pad + np.array([0.0137, 0.0219, 0.0291]))
        shape = np.ceil((high - origin) / h).astype(int) + 2 + pad
        solver = _native.WaveSolver(
            bvh,
            np.zeros(2, np.int64),
            np.zeros(1),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            origin,
            h,
            *map(int, shape),
            source,
            listeners,
            layer=layer,
            layer_damping=damping,
        )
        return _run(solver, rate, h, seconds)[0], solver

    n_fft = 1 << 13
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gain = np.clip((f - 25) / 10, 0, 1) * np.clip((250 - f) / 50, 0, 1)

    def band(x):
        return np.fft.irfft(np.fft.rfft(x, n_fft, axis=1) * gain, n_fft, axis=1)[:, : x.shape[1]]

    reference = band(run(80, 0)[0])
    rows = []
    for name, pad, layer in (
        ("the grid's edge (admittance 1)", 0, 0),
        ("matched layer, 6 cells", 6, 6),
        (f"matched layer, {LAYER_CELLS} cells (the default)", LAYER_CELLS, LAYER_CELLS),
        ("matched layer, 12 cells", 12, 12),
    ):
        pressure, solver = run(pad, layer)
        miss = band(pressure) - reference
        db = 10 * np.log10(np.sum(miss**2, axis=1) / np.sum(reference**2, axis=1))
        rows.append({"boundary": name, "db": [float(x) for x in db], "cells": solver.air_cells})
    return rows


def layer_stability(seconds: float = 20.0) -> dict:
    """The plate of open_boundary in the default matched layer for ``seconds``: the
    level of each second against the first's (dB), at the first listener."""
    h = 0.2
    vertices = np.array([[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0]], float)
    bvh = _native.Bvh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    source, listener = np.array([2.0, 2.0, 1.5]), np.array([[6.0, 4.0, 1.2]])
    low = np.minimum(vertices.min(0), np.minimum(source, listener[0]) - 2 * h)
    high = np.maximum(vertices.max(0), np.maximum(source, listener[0]) + 2 * h)
    origin = low - h * (2.0 + LAYER_CELLS + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((high - origin) / h).astype(int) + 2 + LAYER_CELLS
    solver = _native.WaveSolver(
        bvh,
        np.zeros(2, np.int64),
        np.zeros(1),
        np.zeros((0, 3)),
        np.zeros(0, np.int64),
        origin,
        h,
        *map(int, shape),
        source,
        listener,
        layer=LAYER_CELLS,
        layer_damping=LAYER_DAMPING,
    )
    rate = C / (COURANT * h)
    pressure = _run(solver, rate, h, seconds)[0][0]
    second = int(rate)
    power = [np.mean(pressure[k * second : (k + 1) * second] ** 2) for k in range(int(seconds))]
    levels = 10 * np.log10(np.maximum(power, 1e-300) / power[0])
    return {"steps": len(pressure), "levels_db": [float(x) for x in levels]}


def free_field(scheme: str = "auto") -> list[dict]:
    box = (30.0, 24.0, 24.0)
    source = [box[0] / 2 - 4.03, box[1] / 2 + 0.07, box[2] / 2 + 0.02]
    distances = (1.0, 2.0, 4.0, 8.0)
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": list(box)},
                "materials": {"a": {"absorption": [0.95] * 8, "scattering": 0.0}},
                "default_material": "a",
                "sources": [{"name": "s", "position": source}],
                "listeners": [
                    {"name": f"{d:g} m", "position": [source[0] + d, source[1], source[2]]}
                    for d in distances
                ],
                "simulation": {
                    "ir_length_s": 0.3,
                    "seed": 1,
                    "late": {"rays": 2000},
                    "early": {"max_order": 0},
                    "wave": {"enabled": True, "max_band_hz": 125, "scheme": scheme},
                },
            }
        )
    )
    result = simulate(room)
    rows = []
    for (_, name), response in result.responses.items():
        d = float(name.split()[0])
        energy = response.energy()
        t = np.arange(energy.shape[1]) * response.time_step
        direct = energy[:2, t < (d + 6.5) / C].sum(axis=1) * 4 * math.pi * d * d
        rows.append(
            {
                "distance_m": d,
                "db_63": 10 * math.log10(direct[0]),
                "db_125": 10 * math.log10(direct[1]),
            }
        )
    return rows


def grazing(stencil: int = 7) -> list[dict]:
    size, beta = (40.0, 0.6, 0.6), 0.05
    rows = []
    for cell in (0.1, 0.05):

        def admittances(mesh):
            x = mesh.vertices[mesh.faces][:, :, 0]
            ends = np.all(np.isclose(x, 0.0), axis=1) | np.all(np.isclose(x, size[0]), axis=1)
            return np.where(ends, 1.0, beta)

        probes = [(10.0, 0.3, 0.3), (20.0, 0.3, 0.3), (30.0, 0.3, 0.3)]
        solver, rate = _solver(size, admittances, (5.0, 0.3, 0.3), probes, cell, stencil)
        recorded, delay = _run(solver, rate, cell, 0.2)
        n_fft = 1 << 15
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
        t = (np.arange(recorded.shape[1]) - delay) / rate
        energies = []
        for k, (x, _, _) in enumerate(probes):
            band = np.fft.irfft(np.fft.rfft(recorded[k], n_fft) * gain, n_fft)[: len(t)]
            u = t - (x - 5.0) / C
            window = np.where(np.abs(u) < 0.03, 0.5 + 0.5 * np.cos(np.pi * u / 0.03), 0.0)
            energies.append(float(np.sum((band * window) ** 2)))
        rows.append(
            {
                "cell_m": cell,
                "theory": math.exp(-beta * 4 * size[1] / size[1] ** 2 / 2 * 10.0),
                "fdtd": [
                    math.sqrt(energies[1] / energies[0]),
                    math.sqrt(energies[2] / energies[1]),
                ],
            }
        )
    return rows


BOX_SOURCE = (1.3, 1.1, 1.4)
BOX_LISTENER = (5.8, 3.9, 1.7)
BOX_ABSORPTION = 0.15
BOX_LENGTH_S = 1.5


def _box_response(wave: bool, points_per_wavelength: float = 8.0, scheme: str = "auto"):
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": list(BOX)},
                "materials": {"m": {"absorption": [BOX_ABSORPTION] * 8, "scattering": 0.1}},
                "default_material": "m",
                "sources": [{"name": "s", "position": list(BOX_SOURCE)}],
                "listeners": [{"name": "l", "position": list(BOX_LISTENER)}],
                "simulation": {
                    "ir_length_s": BOX_LENGTH_S,
                    "seed": 1,
                    "wave": {
                        "enabled": wave,
                        "points_per_wavelength": points_per_wavelength,
                        "scheme": scheme,
                    },
                },
            }
        )
    )
    result = simulate(room)
    seconds = result.details["s"].get("wave_s", 0.0)
    return result.responses[("s", "l")], seconds, room.speed_of_sound()


def modal_sum(frequencies: np.ndarray, c: float, f_max: float = 1500.0) -> np.ndarray:
    """Transfer function of the box from the sum of its modes up to ``f_max``, each with
    its decay rate for walls of the admittance of BOX_ABSORPTION; the direct sound at r
    metres is 1 / r, as in the simulation (air absorption, negligible here, left out)."""
    beta = float(admittance(np.array([BOX_ABSORPTION]))[0])
    numbers, f_n = box_modes(BOX, f_max, c)
    decay = modal_decay(numbers, beta, BOX, c)
    size = np.array(BOX)

    def shape(point) -> np.ndarray:
        return np.prod(np.cos(np.pi * numbers * np.array(point) / size), axis=1)

    weight = (
        shape(BOX_SOURCE) * shape(BOX_LISTENER) / np.prod(np.where(numbers > 0, 0.5, 1.0), axis=1)
    )
    weight *= 4 * math.pi * c**2 / np.prod(size)
    w = 2 * math.pi * frequencies
    out = np.zeros(len(frequencies), complex)
    w_n = 2 * math.pi * f_n
    for start in range(0, len(w_n), 2000):
        part = slice(start, start + 2000)
        denominator = w_n[part, None] ** 2 - w[None, :] ** 2 + 2j * decay[part, None] * w[None, :]
        out += (weight[part, None] / denominator).sum(axis=0)
    return out


def _band_figures(ir: np.ndarray) -> dict:
    """G, EDT and T20 in the 63 and 125 Hz octaves of an impulse response at FS, by the
    ISO 3382-1 analysis of measured responses."""
    p = from_impulse_response(ir, FS, 0.01, noise=False)
    return {
        f"{name}_{f}": float(getattr(p, name)[b])
        for name in ("g", "edt", "t20")
        for b, f in enumerate((63, 125))
    }


def box() -> tuple[list[dict], dict]:
    wave, seconds, c = _box_response(True)
    standard, standard_seconds, _ = _box_response(True, scheme="standard")
    fine, fine_seconds, _ = _box_response(True, 16.0)
    rays, _, _ = _box_response(False)
    n = int(BOX_LENGTH_S * FS)
    n_fft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    reach = f <= 300
    theory = np.zeros(len(f), complex)
    theory[reach] = modal_sum(f[reach], c)
    theory *= np.clip((300 - f) / 50, 0, 1)  # no edge for the band filters to ring at
    spectra = {
        "modes": theory,
        "wave solver": np.fft.rfft(wave.impulse_response(FS, seed=1), n_fft),
        "wave solver, 7-point stencil": np.fft.rfft(standard.impulse_response(FS, seed=1), n_fft),
        "wave solver, 16 per wavelength": np.fft.rfft(fine.impulse_response(FS, seed=1), n_fft),
        "rays only": np.fft.rfft(rays.impulse_response(FS, seed=1), n_fft),
    }
    rows = [
        {"model": name, **_band_figures(np.fft.irfft(x, n_fft)[:n])} for name, x in spectra.items()
    ]
    shown = (f >= 15) & (f <= 250)
    curves = {"f": f[shown], **{name: np.abs(x[shown]) for name, x in spectra.items()}}
    curves.update(
        seconds=seconds,
        standard_seconds=standard_seconds,
        fine_seconds=fine_seconds,
        cell=wave.wave.cell,
    )
    return rows, curves


def one_run() -> dict:
    """A 7 x 5 x 3 m box whose absorption differs in every band, the wave solver up to
    500 Hz: one run for its four bands (2.0.0 ran one per band), and what the branches
    cost against the same run with constant admittances."""
    absorption = [0.1, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3]
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": list(BOX)},
                "materials": {"m": {"absorption": absorption, "scattering": 0.1}},
                "default_material": "m",
                "sources": [{"name": "s", "position": list(BOX_SOURCE)}],
                "listeners": [{"name": "l", "position": list(BOX_LISTENER)}],
                "simulation": {
                    "ir_length_s": 0.5,
                    "seed": 1,
                    "late": {"rays": 2000},
                    "wave": {"enabled": True, "max_band_hz": 500},
                },
            }
        )
    )
    start = time.perf_counter()
    result = simulate(room)
    details = result.details["s"]
    # The same grid with constant walls, for the cost of the branches.
    cell, stencil = details["wave_cell_m"], int(details["wave_stencil"])
    points = (BOX_SOURCE, BOX_LISTENER, cell, stencil)
    constant, rate = _solver(BOX, lambda m: [0.05] * len(m.faces), *points)
    fitted = fit_admittance(absorption[:4], 8, rate)
    branched, _ = _solver(BOX, lambda m: [fitted] * len(m.faces), *points)
    steps = int(0.5 * rate)
    timings = []
    for solver in (constant, branched):
        begin = time.perf_counter()
        _run(solver, rate, cell, 0.5)
        timings.append(time.perf_counter() - begin)
    return {
        "bands": 4,
        "runs": 1,
        "stencil": stencil,
        "cells": details["wave_cells"],
        "steps": steps,
        "wave_s": details["wave_s"],
        "total_s": time.perf_counter() - start,
        "absorption_miss": details["wave_absorption_miss"],
        "constant_s": timings[0],
        "branches_s": timings[1],
        "branches": len(fitted.conductance),
    }


def plot(curves: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    f = curves["f"]
    styles = {
        "modes": {"color": "0.25", "lw": 2.4, "alpha": 0.45, "label": "sum of the modes"},
        "wave solver": {"color": "tab:blue", "lw": 1.1, "label": "wave solver (default grid)"},
        "rays only": {"color": "tab:orange", "lw": 1.0, "ls": ":", "label": "rays only"},
    }
    for name, style in styles.items():
        ax.plot(f, 20 * np.log10(np.maximum(curves[name], 1e-6)), **style)
    ax.axvspan(15, 22.3, color="0.93", zorder=0)
    ax.axvspan(176.8, 250, color="0.93", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([16, 22.3, 31.5, 44.2, 63, 88, 125, 177, 250])
    ax.set_xticklabels(["16", "22", "31.5", "44", "63", "88", "125", "177", "250"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(15, 250)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("level (dB re the direct sound at 1 m)")
    ax.set_title(
        f"{BOX[0]:g} x {BOX[1]:g} x {BOX[2]:g} m box, {BOX_ABSORPTION:.0%} absorption: the "
        "wave solver covers 22 to 177 Hz"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the numbers to this file")
    parser.add_argument("--plot", help="save the box's transfer functions to this image")
    args = parser.parse_args()

    results: dict = {}
    results["dispersion"] = dispersion()
    print(
        "| direction | mode | cells per wavelength | exact (Hz) | 7-point, relation (%) | "
        "wave solver | 19-point, relation | wave solver |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    rows = results["dispersion"]["rows"]
    for seven, nineteen in zip(rows[:2], rows[2:], strict=True):
        errors = [
            100 * (row[key] / row["exact_hz"] - 1)
            for row in (seven, nineteen)
            for key in ("relation_hz", "fdtd_hz")
        ]
        print(
            f"| {seven['direction']} | {tuple(seven['mode'])} | "
            f"{seven['cells_per_wavelength']:.2f} | {seven['exact_hz']:.2f} | "
            + " | ".join(f"{e:+.3f}" for e in errors)
            + " |"
        )
    for stencil, errors in results["dispersion"]["at_8"].items():
        found = ", ".join(f"{name} {100 * e:+.3f} %" for name, e in errors.items())
        print(f"\n{stencil}-point stencil at 8 cells per wavelength: {found}")

    results["modes"] = {str(stencil): modes(stencil) for stencil in (7, 19)}
    print(
        "\n| mode | rigid box (Hz) | theory (Hz) | 7-point (Hz) | 19-point | Re(admittance) | "
        "decay, theory (1/s) | 7-point | 19-point |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seven, nineteen in zip(results["modes"]["7"], results["modes"]["19"], strict=True):
        assert seven["mode"] == nineteen["mode"]
        print(
            f"| {tuple(seven['mode'])} | {seven['rigid_hz']:.2f} | {seven['theory_hz']:.2f} | "
            f"{seven['fdtd_hz']:.2f} | {nineteen['fdtd_hz']:.2f} | {seven['admittance']:.4f} | "
            f"{seven['theory_decay']:.3f} | {seven['fdtd_decay']:.3f} | "
            f"{nineteen['fdtd_decay']:.3f} |"
        )
    theory_moves = max(
        abs(b["theory_decay"] / a["theory_decay"] - 1)
        for a, b in zip(results["modes"]["7"], results["modes"]["19"], strict=True)
    )
    print(f"\n(the 19-point grid's rate moves the theory's decay rates by {theory_moves:.1e})")

    results["duct"] = {str(stencil): duct(stencil) for stencil in (7, 19)}
    print("\n| end admittance | reflection, theory | 7-point | 19-point |\n|---:|---:|---:|---:|")
    for seven, nineteen in zip(results["duct"]["7"], results["duct"]["19"], strict=True):
        print(
            f"| {seven['admittance']:g} | {seven['theory']:.3f} | {seven['fdtd']:.3f} | "
            f"{nineteen['fdtd']:.3f} |"
        )

    results["impedance"] = {str(stencil): impedance_duct(stencil) for stencil in (7, 19)}
    rows = results["impedance"]["7"]["rows"]
    print("\n| frequency (Hz) | " + " | ".join(f"{row['frequency_hz']:g}" for row in rows) + " |")
    print("|---|" + "---:|" * len(rows))
    for stencil, impedance in results["impedance"].items():
        rows = impedance["rows"]
        for name, key in (
            ("normal-incidence absorption, fitted admittance", "theory"),
            ("the grid's theory", "grid"),
            ("wave solver", "fdtd"),
        ):
            values = " | ".join(f"{row[key]:.3f}" for row in rows)
            print(f"| {stencil}-point: {name} | {values} |")
    for stencil, impedance in results["impedance"].items():
        print(
            f"\n{stencil}-point: largest difference from 30 to 720 Hz: "
            f"{impedance['max_difference']:.4f} ({impedance['max_difference_below_360']:.4f} "
            f"below 360 Hz, {impedance['max_difference_grid']:.4f} from the grid's theory); "
            f"random-incidence absorption, material {impedance['material']}, fitted "
            + ", ".join(f"{a:.3f}" for a in impedance["fitted"])
            + f" ({impedance['branches']} branches)"
        )

    results["grazing"] = {str(stencil): grazing(stencil) for stencil in (7, 19)}
    print("\n| stencil | cell | amplitude after 10 m, theory | wave solver |\n|---|---:|---:|---:|")
    for stencil, rows in results["grazing"].items():
        for row in rows:
            found = ", ".join(f"{x:.4f}" for x in row["fdtd"])
            print(
                f"| {stencil}-point | {100 * row['cell_m']:g} cm | {row['theory']:.4f} | {found} |"
            )

    results["open"] = open_boundary()
    print("\n| boundary of the grid | listener 1 (dB) | listener 2 | listener 3 | air cells |")
    print("|---|---:|---:|---:|---:|")
    for row in results["open"]:
        found = " | ".join(f"{x:.1f}" for x in row["db"])
        print(f"| {row['boundary']} | {found} | {row['cells']:,} |")
    results["layer_stability"] = layer_stability()
    levels = results["layer_stability"]["levels_db"]
    print(
        f"\nmatched layer over {results['layer_stability']['steps']:,} steps: each second's "
        f"level against the first's, {levels[1]:.0f} dB after one second, at most "
        f"{max(levels[len(levels) // 2 :]):.0f} dB in the second half"
    )

    results["free_field"] = {scheme: free_field(scheme) for scheme in ("isotropic", "standard")}
    print(
        "\n| distance | 63 Hz, 19-point (dB re free field) | 125 Hz | 63 Hz, 7-point | 125 Hz |"
        "\n|---:|---:|---:|---:|---:|"
    )
    for wide, narrow in zip(
        results["free_field"]["isotropic"], results["free_field"]["standard"], strict=True
    ):
        print(
            f"| {wide['distance_m']:g} m | {wide['db_63']:+.2f} | {wide['db_125']:+.2f} | "
            f"{narrow['db_63']:+.2f} | {narrow['db_125']:+.2f} |"
        )

    rows, curves = box()
    results["box"] = rows
    print("\n| model | G 63 Hz (dB) | 125 Hz | EDT 63 Hz (s) | 125 Hz | T20 63 Hz (s) | 125 Hz |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['g_63']:.1f} | {row['g_125']:.1f} | {row['edt_63']:.2f} | "
            f"{row['edt_125']:.2f} | {row['t20_63']:.2f} | {row['t20_125']:.2f} |"
        )
    print(
        f"\nwave solver in the box: {curves['seconds']:.1f} s ({100 * curves['cell']:.0f} cm "
        f"cells), {curves['standard_seconds']:.1f} s with the 7-point stencil, "
        f"{curves['fine_seconds']:.1f} s with 16 per wavelength"
    )
    results["one_run"] = one_run()
    run = results["one_run"]
    print(
        f"\nbox up to 500 Hz, absorption 0.10 to 0.25 in its four wave bands: {run['runs']} run "
        f"of {run['cells']:,.0f} cells and {run['steps']:,} steps ({run['stencil']}-point "
        f"stencil), {run['wave_s']:.1f} s (absorption fitted within "
        f"{run['absorption_miss']:.4f}); the same grid {run['constant_s']:.1f} s with constant "
        f"walls, {run['branches_s']:.1f} s with {run['branches']} branches"
    )
    if args.plot:
        plot(curves, args.plot)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
