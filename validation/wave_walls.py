"""Check what the wave solver's walls on cell faces cost, against exact solutions.

    python validation/wave_walls.py [--parts modes levels wall ...] [--bands 63 125 250 ...]
                                    [--json results.json]

The wave solver (:mod:`acoustic_spaces.engine.wave`) puts every wall on the faces between
its cells: the link between two cells' centres is cut where a face of the model crosses
it. A wall along the grid therefore lies on the cell face nearest the model's, up to half
a cell from it, and a wall at an angle to the grid becomes a staircase. Three checks, each
with both stencils (the 19-point one, which closed rooms take by default, and the 7-point
one) on the grids the solver takes for the 63, 125 and 250 Hz bands (``band_grid`` at the
default points per wavelength, sample rate and air):

- the modes of a rigid 4.13 x 3.37 x 2.71 m box, whose sides are no whole number of cells
  on any of the grids, placed on the grid as ``wave_responses`` places a room whose source
  and listeners are more than two cells from its walls, and moved from there by parts of
  a cell along each axis, turned by 3, 8 and 14 degrees about the vertical, and tilted
  (turned by 8 degrees about the vertical, then by 5 about x): the eigenvalues of the
  grid's own update operator, assembled from ``WaveSolver.layout()``, each analytic mode
  c/2 |n/L| matched to the computed eigenvector it correlates with best (the mode's shape
  sampled at the air cells' centres; eigenvalues within 1e-8 of each other taken
  together, as their eigenvectors are any mix of them), in percent off it. A mode that
  correlates less than 0.8 with its eigenvector is mixed with others (modes close
  together, which a box turned against the grid couples): it is counted, its figure is
  that of the eigenvector it correlates with best, and the script also gives the worst
  of the modes that keep their shape and the furthest frequency that holds at least a
  quarter of a mode's shape (bands as below). Beside them, the same box with every side
  a whole number of cells and its walls on cell faces: what is off there is the grid's
  own dispersion, and it must come out at the stencil's dispersion relation
  (``grid_phase``), which the script checks;
- the band levels in that box over the first 100 and 300 ms after emission, at five
  listeners 0.74 to 1.19 m from the nearest wall, the source 0.41 to 0.53 m from three,
  against the exact field of the rigid box (its image sources, with the solver's own
  pulse: the source's signal without the 19-point stencil's taps, the impulse it stands
  for), both windowed alike (the last 6 ms tapered) and summed over sharp-edged bands
  (the synthesis's crossovers, 89.1, 176.8 and 353.6 Hz, the 63 Hz band's from 44.5 Hz,
  an octave below its upper one, which leaves out the box's lowest mode at 41.6 Hz). The
  images reach far enough that every one whose pulse starts within the window is in, and
  the script checks that more of them, and a longer FFT, change nothing. The same box on
  cell faces gives what is not the walls' place: the grid's dispersion, and the source's
  and the listeners' readings;
- a single rigid plane wall in the open (in a rigid box large enough that nothing else
  comes back within the window), along the grid at parts of a cell off a cell face and
  at angles to the grid: the band levels at the cell centres 1.5 to 6.5 cells in front of
  the wall, 12 cells either way along it, against the same grid's free field with no
  wall driven by the source and by its image in the model's wall, so that what is left is
  the wall's alone (the grid's own dispersion and the source's are on both sides). For
  the walls along the grid also against the free field driven by the source and by its
  image in the cell face the grid puts the wall on: all that is left there is the place.
  Root-mean-square over those centres, by octave of cells per wavelength; the wall's
  place fitted to the error (the analytic field of the source and its image, the wall
  moved along its normal) shows where the grid has it. Measured in cells and cells per
  wavelength the grid is the same for every band (its Courant number is the stencil's)
  but for the source's high-pass, which is in seconds: this check runs on the 250 Hz
  band's grid only (on the 125 Hz band's, in a one-off check, it is within 0.02 dB).

The script prints Markdown tables, and its progress to stderr. ``--parts`` chooses the
checks and ``--bands`` the grids of the first two (the single wall's is always the 250 Hz
band's). With two threads on a shared four-core machine all of it took 18 to 25 minutes
(rough): 9 to 11 of them the modes on the 250 Hz band's grid (``--bands 63 125`` leaves
them out) and 7 to 11 the single wall (``--parts modes levels`` leaves it out).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time

import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from acoustic_spaces import _native, synthesis
from acoustic_spaces.air import speed_of_sound
from acoustic_spaces.engine.wave import STENCILS, band_grid, grid_phase, source_signal
from acoustic_spaces.geometry import box_mesh
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.scene.model import Air, Simulation, WaveStage

#: The solver's defaults: the air's speed of sound, the sample rate and the cells per
#: wavelength at the top of the highest band.
C = speed_of_sound(Air().temperature_c)
FS = Simulation().sample_rate
PER_WAVELENGTH = WaveStage().points_per_wavelength
#: Each stencil's Courant number, by its points.
COURANTS = dict(STENCILS.values())
BANDS = (63, 125, 250)
#: The band edges (Hz): the synthesis's crossovers, the lowest band from an octave below.
CROSSOVERS = synthesis.crossover_frequencies(len(OCTAVE_BANDS_HZ))
EDGES = (CROSSOVERS[0] / 2, *CROSSOVERS[:3])
SIZE = np.array([4.13, 3.37, 2.71])
#: Where wave_responses starts its grid, in cells below the model's lowest corner.
OFFSET = 2.0 + np.array([0.0137, 0.0219, 0.0291])
#: The box's placements on the grid: on cell faces (every side a whole number of cells),
#: moved by parts of a cell along x, y and z, turned about z (degrees) and tilted.
CASES = (
    ("faces", None),
    ("shift", (0.0, 0.0, 0.0)),
    ("shift", (0.25, 0.5, 0.75)),
    ("shift", (0.5, 0.25, 0.4)),
    ("turn", 3.0),
    ("turn", 8.0),
    ("turn", 14.0),
    ("tilt", (8.0, 5.0)),
)
#: Source and listeners in the box's own frame (m).
SOURCE = np.array([0.53, 0.47, 0.41])
LISTENERS = np.array(
    [
        [1.71, 1.43, 1.19],
        [2.93, 2.21, 1.53],
        [3.17, 0.97, 0.88],
        [1.07, 2.61, 1.97],
        [2.31, 1.83, 0.79],
    ]
)
WINDOWS_S = (0.1, 0.3)
TAPER_S = 0.006
#: The least correlation of a mode's shape with its eigenvector for the mode to keep its
#: shape; below it the eigenvectors are mixtures of analytic modes close together.
MATCHED = 0.8
#: The least share of a mode's shape (its squared correlation) an eigenvector must hold
#: to count among the frequencies the mode is spread over.
SPREAD_SHARE = 0.25
#: The single wall: its rigid box's side, the source's distance from the wall (cells), the
#: walls along the grid (parts of a cell off a cell face, into the air) and the angles.
WALL_BOX = 150
WALL_SOURCE = 14.3
WALL_SHARES = (0.0, 0.1, 0.25, 0.4)
WALL_ANGLES = (3.0, 8.0, 14.0, 25.0)
#: Octaves of cells per wavelength the single wall's levels are taken in.
PER_WAVELENGTH_OCTAVES = ((8.0, 16.0), (16.0, 32.0), (32.0, 64.0))
#: The degree of the Chebyshev polynomial the modes' eigenvalues are found through, and
#: the most air cells whose eigenvalues are found all at once instead.
FILTER_DEGREE = 30
DENSE_CELLS = 3000
#: Eigenvalues whose frequencies lie closer than this (a fraction) count as one: closer
#: than the eigenvectors can be told apart (an eigenvector's residual is below 1e-12).
CLUSTER = 1e-8
#: The fitted place of the wall: shifts tried (cells, along its normal into the air).
WALL_SHIFTS = np.round(np.arange(-0.8, 0.8001, 0.01), 4)
#: The threads each run of the solver takes.
THREADS = 2


def grid(band: int, stencil: int):
    """The grid the solver takes for a run up to ``band`` (Hz) with the stencil."""
    return band_grid(
        OCTAVE_BANDS_HZ.index(band), FS, C, PER_WAVELENGTH, len(OCTAVE_BANDS_HZ), COURANTS[stencil]
    )


def _turn(case) -> np.ndarray:
    kind, value = case
    if kind not in ("turn", "tilt"):
        return np.eye(3)
    yaw, pitch = (value, 0.0) if kind == "turn" else value
    a, b = math.radians(yaw), math.radians(pitch)
    about_z = np.array(
        [[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1]]
    )
    about_x = np.array([[1.0, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    return about_z @ about_x


def placed_box(case, cell: float) -> dict:
    """The box placed on a grid of ``cell``: its mesh, its size, the maps from its own
    frame to the grid's and back, and the grid's origin and shape (as wave_responses lays
    it round the box before any shift; on cell faces, the box from 2 cells up)."""
    kind, value = case
    size = np.round(SIZE / cell) * cell if kind == "faces" else SIZE.copy()
    turn = _turn(case)
    shift = np.array(value) * cell if kind == "shift" else np.zeros(3)
    centre = size / 2

    def world(p):
        return (np.asarray(p, dtype=float) - centre) @ turn.T + centre + shift

    def own(q):
        return (np.asarray(q, dtype=float) - centre - shift) @ turn + centre

    mesh = box_mesh(tuple(size))
    vertices = world(mesh.vertices)
    if kind == "faces":
        origin = -2.0 * cell * np.ones(3)
    else:
        origin = (vertices - shift).min(axis=0) - cell * OFFSET
    shape = np.ceil((vertices.max(axis=0) - origin) / cell).astype(int) + 2
    return {
        "vertices": vertices,
        "faces": mesh.faces,
        "size": size,
        "world": world,
        "own": own,
        "origin": origin,
        "shape": shape,
    }


def _rigid_solver(vertices, faces, origin, cell, shape, source, probes, stencil):
    return _native.WaveSolver(
        _native.Bvh(vertices, faces),
        np.zeros(len(faces), np.int64),
        np.zeros(1),
        np.zeros((0, 3)),
        np.zeros(0, np.int64),
        origin,
        cell,
        *map(int, shape),
        np.asarray(source, dtype=float),
        np.asarray(probes, dtype=float).reshape(-1, 3),
        threads=THREADS,
        stencil=stencil,
    )


#: The directions of the links, in the solver's order (fdtd.hpp's kSteps).
STEPS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    + [[a, b, 0] for b in (-1, 1) for a in (-1, 1)]
    + [[a, 0, b] for b in (-1, 1) for a in (-1, 1)]
    + [[0, a, b] for b in (-1, 1) for a in (-1, 1)]
)


def update_operator(layout: dict) -> tuple[sparse.csr_matrix, np.ndarray]:
    """The rigid grid's update as p+ = (2 - A) p - p-, A over the air cells, from the
    solver's layout, and the air cells' indices in the grid."""
    nx, ny, _ = (int(x) for x in layout["shape"])
    types = np.asarray(layout["types"])
    air = np.flatnonzero((types == 1) | (types == 2))
    index = -np.ones(types.size, np.int64)
    index[air] = np.arange(air.size)
    offsets = STEPS[:, 0] + STEPS[:, 1] * nx + STEPS[:, 2] * nx * ny
    weights = np.asarray(layout["weights"], dtype=float)
    inner = np.flatnonzero(types == 1)
    rows = [index[inner]]
    columns = [index[inner]]
    values = [np.full(inner.size, 2.0 - float(layout["centre"][0]))]
    for k, w in enumerate(weights):
        rows.append(index[inner])
        columns.append(index[inner + offsets[k]])
        values.append(np.full(inner.size, -w))
    special = np.asarray(layout["special"])
    counts = np.diff(np.asarray(layout["link_start"]).astype(np.int64))
    directions = np.asarray(layout["link_direction"]).astype(np.int64)
    rows.append(np.repeat(index[special], counts))
    columns.append(index[np.repeat(special, counts) + offsets[directions]])
    values.append(-np.asarray(layout["link_weight"], dtype=float))
    rows.append(index[special])
    columns.append(index[special])
    values.append(2.0 - np.asarray(layout["coef_a"], dtype=float))
    r, c, v = (np.concatenate(x) for x in (rows, columns, values))
    assert (c >= 0).all(), "a link reaches a cell that is not air"
    return sparse.csr_matrix((v, (r, c)), shape=(air.size, air.size)), air


def box_modes(size, f_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Mode numbers (modes, 3) and frequencies of a rigid box below ``f_max``, lowest
    first, the uniform mode left out."""
    top = [int(2 * f_max * s / C) + 1 for s in size]
    numbers = np.array(list(itertools.product(*(range(t + 1) for t in top))))[1:]
    f = C / 2 * np.sqrt(((numbers / np.array(size)) ** 2).sum(axis=1))
    order = np.argsort(f)
    numbers, f = numbers[order], f[order]
    return numbers[f < f_max], f[f < f_max]


def eigenpairs(
    a: sparse.csr_matrix, mu_top: float, mu_cut: float, expected: int
) -> tuple[np.ndarray, np.ndarray]:
    """Every eigenvalue of the symmetric ``a`` (none below 0) up to ``mu_top`` and its
    eigenvector, lowest first, ``expected`` about how many of them there are (with no
    more than DENSE_CELLS rows, all of them at once).

    Lanczos's largest of a Chebyshev polynomial of ``a`` (degree FILTER_DEGREE), which maps
    the eigenvalues from ``mu_cut`` up (the highest bound by Gershgorin's) into [-1, 1]
    and those below it, ever larger the lower, above 1; then the Rayleigh-Ritz of ``a``
    on what it found. The polynomial keeps the order of the eigenvalues below ``mu_cut``,
    so with more of them found than lie below ``mu_top`` every one below it is in."""
    highest = float(abs(a).sum(axis=1).max())
    centre, half = (highest + mu_cut) / 2, (highest - mu_cut) / 2

    def filtered(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape(a.shape[0], -1)

        def step(v: np.ndarray) -> np.ndarray:
            return (centre * v - a @ v) / half

        before, now = x, step(x)
        for _ in range(FILTER_DEGREE - 1):
            before, now = now, 2 * step(now) - before
        return now

    k = int(1.15 * expected) + 10
    if a.shape[0] <= DENSE_CELLS:
        mu, vectors = scipy_linalg.eigh(a.toarray())
    else:
        operator = sparse_linalg.LinearOperator(
            a.shape, matvec=filtered, matmat=filtered, dtype=float
        )
        start = np.random.default_rng(0).standard_normal(a.shape[0])
        _, basis = sparse_linalg.eigsh(operator, k=k, which="LA", tol=1e-10, v0=start)
        projected = basis.T @ (a @ basis)
        mu, rotation = scipy_linalg.eigh((projected + projected.T) / 2)
        vectors = basis @ rotation
        assert (mu < mu_top).sum() < k, "the filter's eigenvalues reach no higher than the top"
    below = mu < mu_top
    residual = float(
        np.linalg.norm(a @ vectors[:, below] - vectors[:, below] * mu[below], axis=0).max()
    )
    assert residual < 1e-8, f"an eigenvector is off by {residual:.1e}"
    return mu[below], vectors[:, below]


def grid_modes(case, band: int, stencil: int) -> dict:
    """The box's modes in each band up to ``band`` on that band's grid: per band the
    largest and root-mean-square deviation (%) from the analytic modes, each analytic
    mode matched to the eigenvector of the update it correlates with best (its shape
    sampled at the air cells' centres)."""
    g = grid(band, stencil)
    cell, period = g.cell, 1.0 / g.rate
    box = placed_box(case, cell)
    middle = box["world"](box["size"] / 2)
    solver = _rigid_solver(
        box["vertices"], box["faces"], box["origin"], cell, box["shape"], middle, middle, stencil
    )
    a, air = update_operator(solver.layout())
    asymmetry = float(abs(a - a.T).max())
    assert asymmetry < 1e-6, f"the update is not symmetric ({asymmetry:.1e})"
    top = EDGES[BANDS.index(band) + 1]
    numbers, exact = box_modes(box["size"], top)
    # Every eigenvalue up to a little past the top, and the frequency each stands for.
    reach = [(2 * math.sin(math.pi * x * top * period)) ** 2 for x in (1.15, 1.4)]
    expected = len(box_modes(box["size"], 1.15 * top)[0])
    mu, vectors = eigenpairs(a, *reach, expected)
    f_grid = np.arcsin(np.sqrt(np.clip(mu, 0.0, None)) / 2) / (math.pi * period)
    shape = box["shape"]
    ijk = np.stack([air % shape[0], (air // shape[0]) % shape[1], air // (shape[0] * shape[1])], 1)
    centres = box["own"](box["origin"] + (ijk + 0.5) * cell)
    # Eigenvalues too close for their eigenvectors to be told apart count as one: an
    # analytic mode's correlation with them is that with the space they span, and its
    # frequency their mean (they are the same to within CLUSTER).
    starts = np.flatnonzero(np.r_[True, np.diff(f_grid) > CLUSTER * f_grid[1:]])
    groups = np.add.reduceat(np.ones(len(f_grid)), starts).astype(int)
    f_group = np.add.reduceat(f_grid, starts) / groups
    member = np.repeat(np.arange(len(starts)), groups)

    def shares_of(mode: np.ndarray) -> np.ndarray:
        """The share of ``mode``'s shape each group of eigenvectors holds."""
        shares = np.zeros(len(starts))
        np.add.at(shares, member, (mode @ vectors) ** 2)
        return shares / float(mode @ mode)

    def best(mode: np.ndarray) -> tuple[int, float]:
        """The group of eigenvectors ``mode`` correlates with best, and how well."""
        shares = shares_of(mode)
        j = int(np.argmax(shares))
        return j, float(math.sqrt(shares[j]))

    relation = grid_phase(np.pi * numbers * cell / box["size"], stencil) / (2 * math.pi * period)
    out = {"cell_m": cell, "air_cells": int(air.size), "asymmetry": asymmetry, "bands": {}}
    for b, name in enumerate(BANDS[: BANDS.index(band) + 1]):
        lo, hi = EDGES[b], EDGES[b + 1]
        deviations, correlations, spread = [], [], []
        for n, f in zip(numbers, exact, strict=True):
            if not lo <= f < hi:
                continue
            shares = shares_of(np.prod(np.cos(np.pi * n * centres / box["size"]), axis=1))
            j = int(np.argmax(shares))
            deviations.append(100 * (f_group[j] / f - 1))
            correlations.append(math.sqrt(shares[j]))
            # The furthest frequency holding a sizeable share of the mode's shape.
            holds = shares >= SPREAD_SHARE
            holds[j] = True
            held = 100 * (f_group[holds] / f - 1)
            spread.append(float(held[np.argmax(np.abs(held))]))
        kept = np.array(correlations) >= MATCHED
        every, spread = np.array(deviations), np.array(spread)
        d = every[kept]
        out["bands"][name] = {
            "modes": len(deviations),
            "mixed": int((~kept).sum()),
            "worst_percent": float(every[np.argmax(np.abs(every))]),
            "rms_percent": float(np.sqrt(np.mean(every**2))),
            "worst_kept_percent": float(d[np.argmax(np.abs(d))]) if d.size else math.nan,
            "furthest_percent": float(spread[np.argmax(np.abs(spread))]),
        }
    if case[0] == "faces":
        # The eigenvalues against the dispersion relation, mode by mode.
        below = exact < EDGES[BANDS.index(band) + 1]
        miss = []
        for n, f_r in zip(numbers[below], relation[below], strict=True):
            j, _ = best(np.prod(np.cos(np.pi * n * centres / box["size"]), axis=1))
            miss.append(abs(f_group[j] / f_r - 1))
        out["off_relation"] = 100 * max(miss)
    return out


def _pulse_spectrum(rate: float, cell: float, stencil: int, n_fft: int) -> np.ndarray:
    """The spectrum of the pressure 1 m from the source in free field, as the solver's
    probes record it: the impulse the source's signal stands for, a step late (a probe
    records before each step), and the 19-point stencil's taps a step later still."""
    courant = COURANTS[stencil]
    signal, _ = source_signal(rate, cell, courant)
    meant, _ = source_signal(rate, cell, courant, level=False)
    lag = 1 + (len(signal) - len(meant)) // 2
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    spectrum = np.fft.rfft(meant, n_fft) / (4.0 * math.pi * courant**2 / cell)
    return spectrum * np.exp(-2j * math.pi * f * lag / rate)


def image_field(points, source, size, reach: float, rate: float, pulse: np.ndarray):
    """The field (points, n_fft) at ``points`` of the rigid box from 0 to ``size`` with
    ``source`` in it, all positions in the box's frame: every image source within
    ``reach`` metres, the pulse whose spectrum is ``pulse``."""
    n_fft = 2 * (len(pulse) - 1)
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    size = np.asarray(size, dtype=float)
    most = np.ceil(reach / (2 * size)).astype(int) + 1
    lattice = np.stack(np.meshgrid(*(np.arange(-m, m + 1) for m in most), indexing="ij"), -1)
    lattice = 2 * lattice.reshape(-1, 3) * size
    images = np.vstack(
        [
            lattice + np.where(np.array(bits) == 1, -source, source)
            for bits in itertools.product((0, 1), repeat=3)
        ]
    )
    # The sum over the images of exp(-i k r) / r at k = n dk, n = a B + b, as the product
    # of exp(-i a B dk r) / r and exp(-i b dk r) over the images.
    dk = 2 * math.pi * (f[1] - f[0]) / C
    count = len(f)
    block = math.ceil(math.sqrt(count))
    out = []
    for p in np.asarray(points, dtype=float):
        r = np.linalg.norm(images - p, axis=1)
        r = r[r < reach]
        outer = np.exp(-1j * dk * block * np.outer(np.arange(block), r)) / r
        inner = np.exp(-1j * dk * np.outer(r, np.arange(block)))
        total = (outer @ inner).reshape(-1)[:count]
        spectrum = total * pulse
        out.append(np.fft.irfft(spectrum, n_fft))
    return np.array(out), len(images)


def band_energies(x: np.ndarray, rate: float, length: int, edges, n_fft: int) -> np.ndarray:
    """Energy (..., bands) of the first ``length`` samples of x, the last TAPER_S of them
    tapered (raised cosine), in the bands between ``edges`` (Hz, sharp-edged): the
    spectrum's squared magnitude summed over the band, each bin weighted by how much of
    its width lies in it, on four times n_fft (so that the sum does not change with it)."""
    window = np.ones(length)
    taper = round(TAPER_S * rate)
    window[-taper:] = 0.5 * (1 + np.cos(np.linspace(0.0, math.pi, taper)))
    n_fft *= 4  # (bins a quarter as wide as the response's)
    spectrum = np.abs(np.fft.rfft(x[..., :length] * window, n_fft, axis=-1)) ** 2 / n_fft
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    width = rate / n_fft
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        share = np.clip(np.minimum(hi, f + width / 2) - np.maximum(lo, f - width / 2), 0.0, None)
        bands.append(spectrum @ (share / width))
    return np.stack(bands, -1)


def box_levels(case, band: int, stencil: int, reference: dict) -> dict:
    """The box's band levels at the listeners, dB over the exact field's, per window
    (WINDOWS_S after emission) and band up to ``band``, on that band's grid.
    ``reference`` keeps the exact fields from one call to the next."""
    g = grid(band, stencil)
    cell, rate = g.cell, g.rate
    box = placed_box(case, cell)
    solver = _rigid_solver(
        box["vertices"],
        box["faces"],
        box["origin"],
        cell,
        box["shape"],
        box["world"](SOURCE),
        box["world"](LISTENERS),
        stencil,
    )
    signal, delay = source_signal(rate, cell, COURANTS[stencil])
    lengths = [int(round(w * rate)) + delay for w in WINDOWS_S]
    steps = max(lengths) + 4
    drive = np.zeros(steps, np.float32)
    drive[: min(steps, len(signal))] = signal[:steps]
    heard = np.asarray(solver.run(drive), dtype=float)
    n_fft = 1 << int(math.ceil(math.log2(2 * (steps + len(signal)))))
    key = (stencil, band, tuple(box["size"]))
    if key not in reference:
        # Every image whose pulse starts before the window ends (its low-pass starts
        # half its taps early, and the probes record up to two steps late), and a metre.
        reach = C * (steps + 16 + 4) / rate + 1.0
        pulse = _pulse_spectrum(rate, cell, stencil, n_fft)
        reference[key] = image_field(LISTENERS, SOURCE, box["size"], reach, rate, pulse)[0]
    exact = reference[key]
    edges = EDGES[: BANDS.index(band) + 2]
    out = {"cell_m": cell, "rate_hz": rate}
    for window, length in zip(WINDOWS_S, lengths, strict=True):
        db = 10 * np.log10(
            band_energies(heard, rate, length, edges, n_fft)
            / band_energies(exact, rate, length, edges, n_fft)
        )
        out[f"{1000 * window:.0f}_ms"] = {
            name: db[:, b].tolist() for b, name in enumerate(BANDS[: db.shape[1]])
        }
    return out


def reference_converged(band: int, stencil: int) -> float:
    """How far (dB, the most over listeners, bands and windows) the exact field's band
    levels move with 30 m more of images and twice the FFT's length: none."""
    g = grid(band, stencil)
    cell, rate = g.cell, g.rate
    signal, delay = source_signal(rate, cell, COURANTS[stencil])
    lengths = [int(round(w * rate)) + delay for w in WINDOWS_S]
    steps = max(lengths) + 4
    n_fft = 1 << int(math.ceil(math.log2(2 * (steps + len(signal)))))
    edges = EDGES[: BANDS.index(band) + 2]
    reach = C * (steps + 16 + 4) / rate + 1.0
    levels = []
    for more, length in ((0.0, n_fft), (30.0, 2 * n_fft)):
        pulse = _pulse_spectrum(rate, cell, stencil, length)
        exact = image_field(LISTENERS, SOURCE, SIZE, reach + more, rate, pulse)[0]
        levels.append(
            np.stack([10 * np.log10(band_energies(exact, rate, n, edges, length)) for n in lengths])
        )
    return float(np.abs(levels[1] - levels[0]).max())


def _first_back(points, sources, side: float, origin, normal) -> float:
    """The shortest path (m) from any of ``sources`` off a face of the rigid box from 0 to
    ``side`` to any of ``points`` whose point of reflection lies in front of the wall
    through ``origin`` with ``normal`` (the others the wall stops)."""
    best = math.inf
    for source in sources:
        for axis in range(3):
            for face in (0.0, side):
                mirrored = points.copy()
                mirrored[:, axis] = 2 * face - mirrored[:, axis]
                t = (face - source[axis]) / (mirrored[:, axis] - source[axis])
                hit = source + t[:, None] * (mirrored - source)
                seen = (hit - origin) @ normal > 0.0
                if seen.any():
                    best = min(best, float(np.linalg.norm(mirrored[seen] - source, axis=1).min()))
    return best


def single_wall(stencil: int, angle: float, share: float, band: int = 250) -> dict:
    """One rigid wall at ``angle`` (degrees, about z) to the grid, through a point
    ``share`` of a cell in front of a cell face, on the ``band``'s grid: the rms of
    its band levels at the centres 1.5 to 6.5 cells in front of it over the grid's own
    field of the source and its image, per octave of cells per wavelength; along the grid
    also over that of the source and its image in the cell face; and the wall's place
    fitted to the error."""
    g = grid(band, stencil)
    cell, rate = g.cell, g.rate
    a = math.radians(angle)
    normal = np.array([math.cos(a), math.sin(a), 0.0])
    along = np.array([-math.sin(a), math.cos(a), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    side = WALL_BOX * cell
    origin = -cell * OFFSET
    shape = np.ceil((side - origin) / cell).astype(int) + 2
    # A cell face across x (between centres), mid-cell in y and z, 12 cells from the
    # middle; the wall through the point `share` of a cell in front of it.
    i0 = np.floor((np.array([side / 2 - 12 * cell, side / 2, side / 2]) - origin) / cell)
    face = origin + i0 * cell + np.array([0.0, 0.5, 0.5]) * cell
    c0 = face + share * cell * normal
    box = box_mesh((side,) * 3)
    far = 2 * side
    quad = np.array(
        [
            c0 + far * (along - up),
            c0 - far * (along + up),
            c0 + far * (up - along),
            c0 + far * (along + up),
        ]
    )
    count = len(box.vertices)
    vertices = np.vstack([box.vertices, quad])
    faces = np.vstack([box.faces, [[count, count + 1, count + 2], [count, count + 2, count + 3]]])
    source = c0 + cell * (WALL_SOURCE * normal + 0.37 * along + 0.21 * up)
    image = source - 2.0 * ((source - c0) @ normal) * normal
    # The cell centres 1.5 to 6.5 cells in front of the wall, 12 either way along it, in
    # two layers.
    low = np.floor((c0 - 16 * cell - origin) / cell).astype(int)
    high = np.ceil((c0 + 16 * cell - origin) / cell).astype(int)
    ix, iy = np.meshgrid(np.arange(low[0], high[0]), np.arange(low[1], high[1]), indexing="ij")
    kz = int(np.floor((c0[2] - origin[2]) / cell))
    centres = []
    for z in (kz, kz + 1):
        points = origin + (np.stack([ix.ravel(), iy.ravel(), np.full(ix.size, z)], 1) + 0.5) * cell
        off = (points - c0) @ normal / cell
        keep = (off >= 1.5) & (off <= 6.5) & (np.abs((points - c0) @ along / cell) <= 12.0)
        centres.append(points[keep])
    centres = np.vstack(centres)
    signal, delay = source_signal(rate, cell, COURANTS[stencil])
    reach = float(np.linalg.norm(centres - image, axis=1).max())
    steps = int(math.ceil(delay + 1 + reach / C * rate + 0.020 * rate))
    back = _first_back(centres, [source, image], side, c0, normal)
    assert back / C * rate - delay > steps, "the box's own faces come back within the run"
    drive = np.zeros(steps, np.float32)
    drive[: min(steps, len(signal))] = signal[:steps]
    solver = _rigid_solver(vertices, faces, origin, cell, shape, source, centres, stencil)
    layout = solver.layout()
    # (Each centre's reading, the cubic through the cells round it, takes its own cell.)
    weights = np.asarray(layout["probe_weights"], dtype=float)
    starts = np.asarray(layout["probe_start"]).astype(np.int64)
    own = np.maximum.reduceat(weights, starts[:-1])
    assert np.allclose(own, 1.0, atol=1e-6) and np.allclose(
        np.abs(weights).sum(), len(own), atol=1e-4
    ), "a probe at a cell's centre reads more than that cell"
    heard = np.asarray(solver.run(drive), dtype=float)

    def free(at) -> np.ndarray:
        """The same grid's cells with no wall (20 more round it), the source at ``at``."""
        bigger = box_mesh((side + 40 * cell,) * 3)
        free_solver = _rigid_solver(
            bigger.vertices - 20 * cell,
            bigger.faces,
            origin - 20 * cell,
            cell,
            shape + 40,
            at,
            centres,
            stencil,
        )
        for axis in range(3):
            for wall in (-20 * cell, side + 20 * cell):
                mirrored = centres.copy()
                mirrored[:, axis] = 2 * wall - mirrored[:, axis]
                path = float(np.linalg.norm(mirrored - at, axis=1).min())
                assert path / C * rate - delay > steps, "the free grid's faces come back"
        return np.asarray(free_solver.run(drive), dtype=float)

    n_fft = 1 << int(math.ceil(math.log2(steps + 4 * len(signal))))
    edges = sorted({C / (p * cell) for octave in PER_WAVELENGTH_OCTAVES for p in octave})
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    k = 2 * math.pi * f / C
    pulse = _pulse_spectrum(rate, cell, stencil, n_fft)

    def levels(x: np.ndarray) -> np.ndarray:
        # Octaves of cells per wavelength, the fewest cells first.
        return 10 * np.log10(band_energies(x, rate, steps, edges, n_fft))[:, ::-1]

    def exact(shift: float) -> np.ndarray:
        """The source and its image in the wall moved ``shift`` cells along its normal."""
        moved = source - 2.0 * ((source - c0) @ normal - shift * cell) * normal
        r = np.linalg.norm(centres - source, axis=1)[:, None]
        r2 = np.linalg.norm(centres - moved, axis=1)[:, None]
        spectrum = pulse * (np.exp(-1j * r * k) / r + np.exp(-1j * r2 * k) / r2)
        return np.fft.irfft(spectrum, n_fft, axis=1)[:, :steps]

    direct = free(source)
    wall = levels(heard)
    error = wall - levels(direct + free(image))
    out = {
        "angle_deg": angle,
        "share": share,
        "centres": len(centres),
        "steps": steps,
        "rms_db": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }
    if angle == 0.0:
        on_face = source - 2.0 * ((source - face) @ normal) * normal
        left = wall - levels(direct + free(on_face))
        out["rms_on_face_db"] = np.sqrt(np.mean(left**2, axis=0)).tolist()
    base = levels(exact(0.0))
    misses = [
        float(np.sqrt(np.mean((error[:, 0] - (levels(exact(s))[:, 0] - base[:, 0])) ** 2)))
        for s in WALL_SHIFTS
    ]
    best = int(np.argmin(misses))
    out["fitted_place_cells"] = float(WALL_SHIFTS[best])
    out["rms_fitted_db"] = misses[best]
    return out


def _label(case) -> str:
    kind, value = case
    if kind == "faces":
        return "on cell faces"
    if kind == "shift":
        return "moved " + ", ".join(f"{v:g}" for v in value) + " cells"
    if kind == "turn":
        return f"turned {value:g} deg"
    return f"turned {value[0]:g} deg, tilted {value[1]:g} deg"


def _say(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _stencils() -> tuple[int, int]:
    return (19, 7)


def modes_part(bands) -> list[dict]:
    """The box's modes on each band's grid, both stencils, every placement."""
    rows = []
    for stencil in _stencils():
        for band in bands:
            for case in CASES:
                started = time.perf_counter()
                result = grid_modes(case, band, stencil)
                result.update(stencil=stencil, grid_band_hz=band, case=_label(case), kind=case[0])
                result["seconds"] = time.perf_counter() - started
                _say(
                    f"modes, {stencil}-point, {band} Hz grid, {_label(case)}: "
                    f"{result['seconds']:.0f} s"
                )
                rows.append(result)
    return rows


def levels_part(bands) -> tuple[list[dict], list[dict]]:
    """The box's band levels against its image sources on each band's grid, both
    stencils, every placement; and how far the image sources' levels move with more."""
    rows, converged = [], []
    for stencil in _stencils():
        reference: dict = {}
        for band in bands:
            converged.append(
                {
                    "stencil": stencil,
                    "grid_band_hz": band,
                    "moves_db": reference_converged(band, stencil),
                }
            )
            for case in CASES:
                started = time.perf_counter()
                result = box_levels(case, band, stencil, reference)
                result.update(stencil=stencil, grid_band_hz=band, case=_label(case), kind=case[0])
                _say(
                    f"levels, {stencil}-point, {band} Hz grid, {_label(case)}: "
                    f"{time.perf_counter() - started:.0f} s"
                )
                rows.append(result)
    return rows, converged


def wall_part() -> list[dict]:
    """The single wall along the grid and at angles to it, both stencils."""
    rows = []
    for stencil in _stencils():
        for share in WALL_SHARES:
            rows.append({"stencil": stencil, **single_wall(stencil, 0.0, share)})
            _say(f"wall, {stencil}-point, along the grid, {share:g} cell off a face")
        for angle in WALL_ANGLES:
            rows.append({"stencil": stencil, **single_wall(stencil, angle, 0.0)})
            _say(f"wall, {stencil}-point, at {angle:g} deg")
    return rows


def _octaves() -> str:
    return " | ".join(f"{lo:g}-{hi:g}" for lo, hi in PER_WAVELENGTH_OCTAVES)


def print_modes(rows: list[dict]) -> None:
    print(
        "\nModes of the rigid box, % off the analytic ones, each at the eigenvector it "
        "correlates with best: worst (rms)"
    )
    print(f"[how many correlate less than {MATCHED:g} with it, mixed with others, of how many]")
    print("\n| stencil | grid (cell) | placement | 63 Hz band | 125 Hz band | 250 Hz band |")
    print("|---|---|---|---:|---:|---:|")
    for row in rows:
        cells = []
        for band in BANDS:
            b = row["bands"].get(band)
            if b is None:
                cells.append("")
                continue
            cells.append(
                f"{b['worst_percent']:+.2f} ({b['rms_percent']:.2f}) [{b['mixed']} of {b['modes']}]"
            )
        print(
            f"| {row['stencil']}-point | {row['grid_band_hz']} Hz ({100 * row['cell_m']:.2f} cm) "
            f"| {row['case']} | " + " | ".join(cells) + " |"
        )
    print("\nOn cell faces against the stencil's dispersion relation: the most any mode is off")
    for row in rows:
        if row["kind"] == "faces":
            print(
                f"- {row['stencil']}-point, {row['grid_band_hz']} Hz grid: "
                f"{row['off_relation']:.1e} % "
                f"({row['air_cells']:,} air cells; update symmetric within {row['asymmetry']:.0e})"
            )
    kinds = (("faces",), ("shift",), ("turn", "tilt"))
    print(
        "\nSummary, each grid's own (highest) band, over the placements: worst (largest rms), "
        "every mode"
    )
    print(
        "\n| stencil | grid (cell) | on cell faces (sides in whole cells) | moved within a cell "
        "| turned or tilted |"
    )
    print("|---|---|---:|---:|---:|")
    for stencil, band, mine in _own_bands(rows):
        cells = []
        for kind in kinds:
            picked = [r["bands"][band] for r in mine if r["kind"] in kind]
            worst = max(picked, key=lambda b: abs(b["worst_percent"]))["worst_percent"]
            rms = max(b["rms_percent"] for b in picked)
            cells.append(f"{worst:+.1f} % ({rms:.1f} %)")
        print(
            f"| {stencil}-point | {band} Hz ({100 * mine[0]['cell_m']:.1f} cm) | "
            + " | ".join(cells)
            + " |"
        )
    print(
        f"\nThe same off the cell faces: the modes mixed (correlation below {MATCHED:g}), least "
        "... most of how many; the worst"
    )
    print(
        "of the modes that keep their shape; the furthest frequency holding at least "
        f"{SPREAD_SHARE:g} of a mode's shape"
    )
    print(
        "\n| stencil | grid (cell) | placement | mixed | worst, keeping their shape "
        "| furthest, a quarter of the shape |"
    )
    print("|---|---|---|---:|---:|---:|")
    for stencil, band, mine in _own_bands(rows):
        for kind, name in zip(kinds[1:], ("moved within a cell", "turned or tilted"), strict=True):
            picked = [r["bands"][band] for r in mine if r["kind"] in kind]
            mixed = [b["mixed"] for b in picked]
            kept = max(picked, key=lambda b: abs(b["worst_kept_percent"]))["worst_kept_percent"]
            furthest = max(picked, key=lambda b: abs(b["furthest_percent"]))["furthest_percent"]
            print(
                f"| {stencil}-point | {band} Hz ({100 * mine[0]['cell_m']:.1f} cm) | {name} | "
                f"{min(mixed)} ... {max(mixed)} of {picked[0]['modes']} | {kept:+.1f} % "
                f"| {furthest:+.1f} % |"
            )


def _own_bands(rows: list[dict]):
    """Each stencil and grid that has rows, with them."""
    for stencil in _stencils():
        for band in BANDS:
            mine = [r for r in rows if r["stencil"] == stencil and r["grid_band_hz"] == band]
            if mine:
                yield stencil, band, mine


def print_levels(rows: list[dict], converged: list[dict]) -> None:
    print("\nBand levels in the box, dB over its image sources' (5 listeners: least ... most)")
    print(
        "\n| stencil | grid (cell) | placement | window | 63 Hz band | 125 Hz band | 250 Hz band |"
    )
    print("|---|---|---|---|---:|---:|---:|")
    for row in rows:
        for window in WINDOWS_S:
            key = f"{1000 * window:.0f}_ms"
            cells = [
                f"{min(row[key][b]):+.2f} ... {max(row[key][b]):+.2f}" if b in row[key] else ""
                for b in BANDS
            ]
            print(
                f"| {row['stencil']}-point | {row['grid_band_hz']} Hz "
                f"({100 * row['cell_m']:.2f} cm) | {row['case']} | {key.replace('_', ' ')} | "
                + " | ".join(cells)
                + " |"
            )
    print(
        "\nSummary: the largest |dB| over the listeners and the placements off the cell faces, "
        "first 100 / 300 ms"
    )
    print(
        "\n| stencil | grid (cell) | 63 Hz band | 125 Hz band | 250 Hz band "
        "| on cell faces, grid's own band |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for stencil in _stencils():
        for band in BANDS:
            mine = [r for r in rows if r["stencil"] == stencil and r["grid_band_hz"] == band]
            if not mine:
                continue
            cells = []
            for b in BANDS:
                if b not in mine[0]["100_ms"]:
                    cells.append("")
                    continue
                worst = [
                    max(
                        abs(np.array(r[f"{1000 * w:.0f}_ms"][b])).max()
                        for r in mine
                        if r["kind"] != "faces"
                    )
                    for w in WINDOWS_S
                ]
                cells.append(" / ".join(f"{x:.1f}" for x in worst))
            faces = [r for r in mine if r["kind"] == "faces"][0]
            own = [max(abs(np.array(faces[f"{1000 * w:.0f}_ms"][band]))) for w in WINDOWS_S]
            print(
                f"| {stencil}-point | {band} Hz ({100 * mine[0]['cell_m']:.1f} cm) | "
                + " | ".join(cells)
                + " | "
                + " / ".join(f"{x:.2f}" for x in own)
                + " |"
            )
    print(
        "\nThe image sources' levels with 30 m more of images and twice the FFT's length, the "
        "most they move:"
    )
    for row in converged:
        print(f"- {row['stencil']}-point, {row['grid_band_hz']} Hz grid: {row['moves_db']:.1e} dB")


def print_wall(rows: list[dict]) -> None:
    print(
        "\nA single rigid wall: rms dB over the cell centres 1.5 to 6.5 cells in front of it, "
        "against the grid's"
    )
    print(
        "field of the source and its image in the model's wall (and in the cell face the grid "
        "puts it on)"
    )
    print(
        f"\n| stencil | wall | cells per wavelength {_octaves()} | in the cell face: {_octaves()} "
        "| place fitted (cells) |"
    )
    print("|---|---|" + "---:|" * (2 * len(PER_WAVELENGTH_OCTAVES) + 1))
    for row in rows:
        wall = (
            f"along the grid, {row['share']:g} cell off a face"
            if row["angle_deg"] == 0.0
            else f"at {row['angle_deg']:g} deg"
        )
        face = row.get("rms_on_face_db")
        cells = [f"{x:.2f}" for x in row["rms_db"]]
        cells += [f"{x:.3f}" for x in face] if face else [""] * len(PER_WAVELENGTH_OCTAVES)
        print(
            f"| {row['stencil']}-point | {wall} | "
            + " | ".join(cells)
            + f" | {row['fitted_place_cells']:+.2f} |"
        )


PARTS = ("modes", "levels", "wall")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--bands", nargs="+", type=int, default=list(BANDS), choices=BANDS)
    parser.add_argument("--json", help="write the numbers to this file")
    args = parser.parse_args()
    bands = sorted(args.bands)
    print(
        f"c = {C:.4f} m/s, {FS:g} Hz, {PER_WAVELENGTH:g} cells per wavelength at the top of "
        "the highest band"
    )
    for stencil in _stencils():
        for band in bands:
            g = grid(band, stencil)
            print(
                f"- {stencil}-point, grid for the {band} Hz band: {100 * g.cell:.3f} cm cells, "
                f"{g.rate:.2f} steps/s"
            )
    results: dict = {}
    if "modes" in args.parts:
        started = time.perf_counter()
        results["modes"] = modes_part(bands)
        print_modes(results["modes"])
        print(f"\n(modes: {time.perf_counter() - started:.0f} s)")
    if "levels" in args.parts:
        started = time.perf_counter()
        results["levels"], results["levels_converged"] = levels_part(bands)
        print_levels(results["levels"], results["levels_converged"])
        print(f"\n(levels: {time.perf_counter() - started:.0f} s)")
    if "wall" in args.parts:
        started = time.perf_counter()
        results["wall"] = wall_part()
        print_wall(results["wall"])
        print(f"\n(wall: {time.perf_counter() - started:.0f} s)")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
