"""Check what the wave solver's walls cost, on cell faces and fitted, against exact solutions.

    python validation/wave_walls.py [--parts dispersion modes levels wall lroom absorbing
                                     long cost] [--bands 63 125 250 ...]
                                    [--walls cell_faces fitted] [--json results.json]

The wave solver (:mod:`acoustic_spaces.engine.wave`) puts every wall on the faces between
its cells by default (``walls="cell_faces"``): the link between two cells' centres is cut
where a face of the model crosses it. A wall along the grid therefore lies on the cell
face nearest the model's, up to half a cell from it, and a wall at an angle to the grid
becomes a staircase. With fitted walls (``walls="fitted"``, ``simulation.wave.walls:
fitted``) each wall lies where the model has it: the cells the model's faces cross are
cut, and each cut cell is stepped with its own volume V (its air's, raised where the cell
is so small that its links would weigh more than 2 V) and the open areas of its faces, on
the 7-point stencil, the only one fitted walls take. ``--walls`` chooses which walls are
checked (both by default); with walls on cell faces each check runs with both stencils
(the 19-point one, which closed rooms take by default, and the 7-point one), with fitted
walls with the 7-point one. The grids are the ones the solver takes for the 63, 125 and
250 Hz bands (``band_grid`` at the default points per wavelength, sample rate and air).
Every run of the solver is checked to have the walls it was asked for. The references
are exact solutions written here or in ``validation/wave_solver.py``, except where a part
says it compares with another run of the solver. Eight parts:

- ``dispersion``: a rigid box with every side a whole number of cells and its walls on
  cell faces (with fitted walls, the model's walls lie on those planes): its modes, the
  eigenvalues of the grid's own update operator assembled from ``WaveSolver.layout()``
  (with fitted walls made symmetric with the volumes V the update steps the cells with,
  the update's operator being D^-1 K with K symmetric and D the cells' V), against the
  stencil's dispersion relation (``grid_phase``), mode by mode up to the top of the grid's
  band. Nothing but the grid's dispersion is off there, and the check is that every mode
  follows the relation within 1e-5 (the solver lowers every link's weight by 2^-18, a
  fitted cell's as a plain cell's, which slows its sound by 2^-19). On the 7-point stencil
  the relation is also written out in the script, sin^2(w T / 2) = (sin^2(k_x h / 2) +
  sin^2(k_y h / 2) + sin^2(k_z h / 2)) / 3, and checked to give the same frequencies as
  ``grid_phase`` within 1e-9, so that the reference does not rest on the package's code
  alone.
- ``modes``: the modes of a rigid 4.13 x 3.37 x 2.71 m box, whose sides are no whole
  number of cells on any of the grids, placed on the grid as ``wave_responses`` places a
  room whose source and listeners are more than two cells from its walls, and moved from
  there by parts of a cell along each axis, turned by 3, 8 and 14 degrees about the
  vertical, and tilted (turned by 8 degrees about the vertical, then by 5 about x): each
  analytic mode c/2 |n/L| matched to the computed eigenvector it correlates with best
  (the mode's shape sampled at the air cells' centres, weighted by the square root of
  their V; eigenvalues within 1e-8 of each other taken together, as their eigenvectors
  are any mix of them), in percent off it. A mode that correlates less than 0.8 with its
  eigenvector is mixed with others (modes close together, which a box turned against the
  grid couples): it is counted, its figure is that of the eigenvector it correlates with
  best, and the script also gives the worst of the modes that keep their shape and the
  furthest frequency that holds at least a quarter of a mode's shape (bands as below).
  Then the same box turned by 0, 3, 8 and 14 degrees about the vertical, by 22.5 and 11
  (about the vertical, then about x), 30 and 20, 45 and 0, and 40 and 35 degrees, on the
  125 and 250 Hz bands' 7-point grids, both kinds of walls: its 16 lowest modes against
  the exact ones as the stencil's dispersion moves them (the mean over the mode's eight
  plane waves of the relation's frequency, the grid's own error taken out); with fitted
  walls also the box moved along the grid, and every turn on the 125 Hz band's grid at 24
  more placements (the grid's origin 2 + u cells below the box's lowest corner, u drawn
  uniformly from [0, 1)^3 with a fixed seed), the worst over them printed beside the one
  at ``wave_responses``' placement. Checked: with fitted walls every one of the 16 within
  0.05 % with the box along the grid (at every placement moved by parts of a cell), within
  0.38 % turned up to 14 degrees about the vertical and within 4.3 % turned further (each
  to the precision it is stated in, so 0.05 % is under 0.055 %), at ``wave_responses``'
  placement and at the worst of the 24; with walls on cell faces the same reference finds
  a mode at least 1 % off on each grid, as a wall up to half a cell from the model's moves
  them. The two turned limits are the 125 Hz band's grid's worst (14 degrees, 45 degrees
  at the worst of the 24 placements); on the 250 Hz band's grid the box is half as far
  off or less.
- ``levels``: the band levels in that box over the first 100 and 300 ms after emission,
  at five listeners 0.74 to 1.19 m from the nearest wall, the source 0.41 to 0.53 m from
  three, against the exact field of the rigid box (its image sources, with the solver's
  own pulse: the source's signal without the 19-point stencil's taps, the impulse it
  stands for), both windowed alike (the last 6 ms tapered) and summed over sharp-edged
  bands (the synthesis's crossovers, 89.1, 176.8 and 353.6 Hz, the 63 Hz band's from
  44.5 Hz, an octave below its upper one, which leaves out the box's lowest mode at
  41.6 Hz). The images reach far enough that every one whose pulse starts within the
  window is in, and the script checks that more of them, and a longer FFT, move the
  levels by less than 0.01 dB. Checked: with fitted walls no level, at any placement,
  listener, window and band, more than 1.82 dB off the exact one (the error stated for a
  single wall on cell faces), and with walls on cell faces the worst on each grid at
  least that far off. The levels hold the grid's dispersion and the source's and the
  listeners' readings beside the walls' places, and the same box with every side a whole
  number of cells is another box, so the limit is coarse: it sees a box as far off as
  walls on cell faces put it, or a source weighted wrongly, and leaves the walls' places
  to the modes and the wall parts;
- ``wall``: a single rigid plane wall in the open (in a rigid box large enough that
  nothing else comes back within the window), along the grid at parts of a cell off a
  cell face and at angles to the grid, and on the 7-point stencil also 0.48 of a cell
  behind a cell face (where walls on cell faces put it furthest from the model's) and at
  45 degrees through a cell face and 0.2 of a cell behind one: the band levels at the
  cell centres 1.5 to 6.5 cells in front of the wall, 12 cells either way along it,
  against the same grid's free field with no wall (a run of the solver with walls on cell
  faces) driven by the source and by its image in the model's wall, so that what is left
  is the wall's alone (the grid's own dispersion and the source's are on both sides). For
  the walls along the grid also against the free field driven by the source and by its
  image in the cell face the grid puts the wall on: all that is left there is the place.
  Root-mean-square over those centres, by octave of cells per wavelength; the place of
  the wall that best explains the error (the analytic field of the source and its image,
  the wall moved along its normal) shows where the grid has it. Measured in cells and
  cells per wavelength the grid is the same for every band (its Courant number is the
  stencil's) but for the source's high-pass, which is in seconds: this check runs on the
  250 Hz band's grid only (on the 125 Hz band's, in a one-off check, it is within
  0.02 dB). Checked, in the octave of 8 to 16 cells per wavelength: fitted walls along
  the grid under 0.045 dB (0.04 dB as stated), at 3, 8 and 14 degrees under 0.185 dB
  (0.08 to 0.18 dB as stated); walls on cell faces 0.48 of a cell off at least 1 dB, as a
  wall half a cell off is;
- ``lroom``: an L-shaped room, 2.4 x 2.0 x 1.5 m with a 1.0 x 0.9 m notch cut from one
  corner of its plan, every length rounded to whole cells of the grid, on the 125 and 250
  Hz bands' 7-point grids, moved off the cell faces by (0.311, 0.173, 0.457), 0.05, -0.48
  and +0.48 cells along each axis, and with fitted walls also by k / 40 of a cell along
  each axis (k = 0 to 39) and by 0.499 and 0.501, either side of the planes of the cells'
  centres: its lowest mode, the plan's (the height's lies higher), with each kind of
  walls. The references are the plan's own problem written here (minus the Laplacian on
  the L, no flow through its walls, on the 5-point stencil over squares): on squares of
  the grid's cell, which walls on cell faces must give within 1e-5 (the room's walls lie
  on cell faces, and its mode does not vary with height); and the continuum's, from
  squares of 2.5, 1.2 and 0.6 cm extrapolated to nothing (the error falls as a power of
  the square's side found from the three, about 4/3 at the plan's concave corner), carried
  through the grid's time step. Checked: at each of the four placements, fitted walls'
  mode no further from the continuum's than the cell-faces mode; fitted walls on the cell
  faces (moved by 0) giving the cell-faces mode within 1e-5 (their rows' coefficients are
  rounded to single precision from the fit's volumes and areas, which there lie within
  1e-14 of whole ones, so they can differ from the plain cells' by a unit in single
  precision's last place); and over the scan, fitted walls' mode between 1.4 % below and
  1.0 % above the continuum's on the 125 Hz band's grid and between 0.5 % below and 0.4 %
  above on the 250 Hz band's (each to the precision it is stated in, so 1.4 % is under
  1.45 %). Walls on cell faces lie below it (the concave edge on whole cells); fitted
  walls lie above it or below it with where the concave edge falls in its cell, and their
  mode jumps by 2.3 % on the 125 Hz band's grid and 0.9 % on the 250 Hz band's as the edge
  crosses the planes of the cells' centres, a stated limit of fitted walls (just below
  those planes, on the 250 Hz band's grid, they lie 0.01 % further below the continuum
  than walls on cell faces). At 0.05 the cube round the concave edge keeps a thin L of air
  whose link the fit scales to the L's volume (taken whole, it would put the mode 3.3 %
  above the continuum on the 125 Hz grid);
- ``absorbing``: fitted walls with admittances. A duct 0.3 m square on 10 cm cells, its far
  end 0, 0.25, -0.48 and 0.48 of a cell past a cell face: the reflection off an end of
  admittance 0.25, 0.5 and 1 (the incident and the reflected waves at 40-200 Hz, gated)
  against (1 - beta) / (1 + beta), within 0.012 at admittance 0.25 and 0.5 and within 0.1
  at admittance 1, which reflects nothing in the continuum: there the grid's own
  reflection off a wall half a cell past the last centre is 0.02 to 0.09 at these
  frequencies (``grid_reflection`` in ``validation/wave_solver.py``), and walls on cell
  faces (not checked) give 0.044 at every end. The end of a material with a branch
  resonant at 100 Hz (the admittance written
  out here, and checked against ``Admittance``'s within 1e-12): its reflection over a rigid
  end's at 30-250 Hz against the grid's theory for that admittance (``grid_reflection``),
  within 0.02 at the ends 0 and 0.25, and 0.0278 at -0.48, where the fitted update itself
  lies 0.0228 off it. And the 4.13 m box turned 8 degrees on the 125 and 250 Hz bands'
  grids, its floor and ceiling of admittance 0.05 and its four walls 0.02: each of its 9
  lowest modes' first-order decay from the update's own terms (sum g psi^2 / sum V psi^2,
  psi the rigid update's mode and g = V (1 - coef_b) the cell's wall term) within 4 % of
  modal theory's (``modal_decay`` in ``validation/wave_solver.py``, each pair of walls with
  its own admittance). The same box with rigid floor and ceiling and its walls of the
  material with a branch: the layout's branch states' weights, each times its cell's
  volume, summed, within 1e-6 of L/2 times the walls' area in cells squared (L the
  Courant number; a state weighs (L/2) W / V for a wall of area W in a cell of volume V),
  which no reflection above sees where the wall crosses its cells at a slant. Walls on
  cell faces, whose walls are the staircase's squares, are printed beside;
- ``long``: 200,000 steps with fitted walls of an L-shaped room with plates (the room
  above with four plates, one turned 31 and 57 degrees, one level, two upright with a
  3 cm slit between them, all turned 22.5 and 11 degrees, on 10 cm cells): rigid, every
  wall of admittance 1, every wall of admittance 0.05 with a mass-spring branch resonant
  at 60 Hz, and open (its ceiling left out, in a grid with a matched layer); and the
  4.13 m box turned 22.5 degrees on the 19.8 cm grid, rigid. The rms over 24 points by
  blocks of 10,000 steps, of the whole signal and of its oscillating part (each point's
  mean over the block taken out), and the update's Rayleigh quotient of the uniform
  pressure (its lowest eigenvalue, where a uniform pressure rings). Checked: a rigid
  room's last block within 0.97 to 1.05 of its second, none above 1.1 times it, its
  update's largest eigenvalue under 4, and each of the update's coefficients as stored
  (the centre's and the links', a special cell's or a plain one's) not below 0, the
  centre's and the links' of each cell summing to at most 2 (in float64): together these
  keep every eigenvalue of the update within 0 to 4, so a uniform pressure never grows
  (rounded to the nearest float instead of down, the centre's coefficients take some rows
  above 2); an absorbing or open room's sound below a millionth of the first block from
  the half on, and not growing again.
  The clamp's counts (cells whose volume it raised, and by how much at most) are printed;
- ``cost``: the time the fit and the steps take with fitted walls against walls on cell
  faces, with the 7-point stencil and with the 19-point (isotropic) one: the box above
  on the 250 Hz band's grids (the solver made and run directly, 5 s of sound, about a
  second of steps), and through ``wave_responses`` (its first source to every listener,
  0.25 s of sound) the church example, closed, and the leaky classroom, open
  (``wave_responses`` makes and fits a closed grid first, finds the model open and makes
  a grid with a matched layer, fitted again), on the 125 Hz band's grid; the least of
  three runs, two threads. Checked: fitted walls' steps at most 2.4 times as long as the
  7-point stencil's with walls on cell faces (the solver's own plain run is the
  reference), and the fit (every grid made) at most 2.9 s. The times are this machine's,
  printed beside them; the script is meant to run alone on it.

The script prints Markdown tables, then every check with what it found and its limit, and
its progress to stderr; a check that fails is marked FAILED, and the script then ends with
an error naming the failed checks. Checks that the chosen walls leave out (with ``--walls
fitted`` alone, the controls on cell faces, the L-shaped room's checks, and the cost's
ratio to the plain run) are listed at the end. ``--parts`` chooses the parts
(``modes`` runs ``dispersion``'s boxes too), ``--bands`` the grids of the first three (the
single wall's is always the 250 Hz band's, and the L-shaped room's, the duct's and the long
runs' are their own) and ``--walls`` the kinds of walls. With two threads on a shared
four-core machine the whole script took 41 and 51 minutes in two runs (others running,
load 1.4 to 1.8 and 1.2 to 3.2). The parts, in minutes: ``dispersion`` about 2 run alone
(``modes`` runs its boxes too); ``modes`` 20 to 24, all but a few of them on the 250 Hz
band's grids, which ``--bands 63 125`` leaves out; ``levels`` 2 to 3; ``wall`` 15 to 20;
``lroom`` about a quarter; ``absorbing`` under one; ``long`` about 2, most of it the open
room; ``cost`` under one.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from acoustic_spaces import Room, _native, synthesis
from acoustic_spaces.air import speed_of_sound
from acoustic_spaces.engine.impedance import Admittance
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.wave import (
    LAYER_CELLS,
    LAYER_DAMPING,
    STENCILS,
    band_grid,
    grid_phase,
    source_signal,
    wave_responses,
)
from acoustic_spaces.geometry import box_mesh
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.scene.model import Air, Simulation, WaveStage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wave_solver import grid_reflection, modal_decay  # noqa: E402

#: The solver's defaults: the air's speed of sound, the sample rate and the cells per
#: wavelength at the top of the highest band.
C = speed_of_sound(Air().temperature_c)
FS = Simulation().sample_rate
PER_WAVELENGTH = WaveStage().points_per_wavelength
#: Each stencil's Courant number, by its points.
COURANTS = dict(STENCILS.values())
BANDS = (63, 125, 250)
#: Where the grid's walls lie: on the cell faces nearest the model's, or fitted.
WALLS = ("cell_faces", "fitted")
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
#: The turned box: turned about the vertical, then about x (degrees), on the 7-point
#: grids of TURN_BANDS, its LOWEST lowest modes against the exact ones with the grid's
#: dispersion.
TURNS = (
    (0.0, 0.0),
    (3.0, 0.0),
    (8.0, 0.0),
    (14.0, 0.0),
    (22.5, 11.0),
    (30.0, 20.0),
    (45.0, 0.0),
    (40.0, 35.0),
)
TURN_BANDS = (125, 250)
LOWEST = 16
#: The signs of a mode's eight plane waves.
SIGNS = np.array(list(itertools.product((1, -1), repeat=3)))
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
#: More single walls, on the 7-point stencil with both kinds of walls: (angle, part of a
#: cell off a cell face) along the grid 0.48 of a cell behind a face, so that walls on
#: cell faces lie 0.48 of a cell in front of the model's, and at 45 degrees.
WALL_MORE = ((0.0, -0.48), (45.0, 0.0), (45.0, -0.2))
#: Octaves of cells per wavelength the single wall's levels are taken in.
PER_WAVELENGTH_OCTAVES = ((8.0, 16.0), (16.0, 32.0), (32.0, 64.0))
#: The degree of the Chebyshev polynomial the modes' eigenvalues are found through, and
#: the most air cells whose eigenvalues are found all at once instead.
FILTER_DEGREE = 30
DENSE_CELLS = 3000
#: Eigenvalues whose frequencies lie closer than this (a fraction) count as one: closer
#: than the eigenvectors can be told apart (an eigenvector's residual is below 1e-12).
CLUSTER = 1e-8
#: The wall's place that best explains the error: shifts tried (cells, along its normal
#: into the air).
WALL_SHIFTS = np.round(np.arange(-0.8, 0.8001, 0.01), 4)
#: The threads each run of the solver takes.
THREADS = 2
#: The cost's models: the example models run through wave_responses, the length of
#: sound each run makes, and the runs of each whose least time counts.
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
COST_ROOMS = ("church", "leaky_classroom")
COST_LENGTH_S = 0.25
COST_REPEATS = 3
#: The length of sound the box's solver steps in the cost part: about a second of
#: stepping on the 250 Hz band's grid, long enough for the time per step to hold still.
COST_BOX_LENGTH_S = 5.0
#: The turned box is also placed at SCAN_PLACEMENTS more places on the SCAN_BAND's grid
#: (the 19.8 cm grid, the one its figures were stated on): the grid's origin 2 + u cells
#: below the box's lowest corner, u drawn uniformly from [0, 1)^3 with the seed SCAN_SEED.
SCAN_BAND = 125
SCAN_PLACEMENTS = 24
SCAN_SEED = 0
#: The L-shaped room: its plan's sides and height (m), and the notch cut from the plan's
#: corner at (x, y) = its sides (m); on each grid every length rounded to whole cells.
L_SIZE = np.array([2.4, 2.0, 1.5])
L_NOTCH = np.array([1.0, 0.9])
L_BANDS = (125, 250)
#: Where the L-shaped room lies on the grid: moved off the cell faces by these parts of
#: a cell along x, y and z.
L_PLACEMENTS = (
    (0.311, 0.173, 0.457),
    (0.05, 0.05, 0.05),
    (-0.48, -0.48, -0.48),
    (0.48, 0.48, 0.48),
)
#: With fitted walls the L-shaped room is also moved by k / L_SCAN_STEPS of a cell along
#: each axis (k = 0 ... L_SCAN_STEPS - 1), and by 0.5 - L_SCAN_CROSS and 0.5 + L_SCAN_CROSS,
#: on either side of the planes of the cells' centres, where its mode jumps.
L_SCAN_STEPS = 40
L_SCAN_CROSS = 1e-3
#: The finest square the plan's continuum mode is computed on (m): it and twice and four
#: times it, each a whole part of the grid's cell, extrapolated to nothing.
L_FINEST_M = 0.0062
#: The duct: its cell and cross section (m), how far its far end lies past a cell face
#: (cells), the far end's admittances, and the band its reflection is taken over (Hz).
DUCT_CELL = 0.1
DUCT_WIDTH = 0.3
DUCT_ENDS = (0.0, 0.25, -0.48, 0.48)
DUCT_BETAS = (0.25, 0.5, 1.0)
DUCT_BAND = (40.0, 200.0)
#: The duct's end of a material with a branch: admittance 0.05 and a branch resonant at
#: 100 Hz (conductance 0.6, quality 2), at the ends of DUCT_BRANCH_ENDS; its reflection over
#: a rigid end's, over DUCT_BRANCH_BAND.
RESONANT = Admittance(0.05, np.array([0.6]), np.array([2.0]), np.array([100.0]))
DUCT_BRANCH_ENDS = (0.0, 0.25, -0.48)
DUCT_BRANCH_BAND = (30.0, 250.0)
#: The absorbing box: the room-sized box turned DECAY_TURN degrees about the vertical, its
#: floor and ceiling of admittance DECAY_BETAS[0], its four walls of DECAY_BETAS[1]; the
#: first-order decay of its DECAY_MODES lowest modes, on the grids of DECAY_BANDS.
DECAY_TURN = 8.0
DECAY_BETAS = (0.05, 0.02)
DECAY_MODES = 9
DECAY_BANDS = (125, 250)
#: The long runs: their steps and the blocks the rms is taken over.
LONG_STEPS = 200_000
LONG_BLOCK = 10_000
#: The absorbing wall of the long runs: admittance 0.05 and a mass-spring branch resonant
#: at 60 Hz.
LONG_WALL = Admittance(0.05, np.array([0.08]), np.array([1.5]), np.array([60.0]))
#: Points the long runs are heard at, spread through the room's air.
LONG_POINTS = 24

# The checks' limits. A limit taken from a figure stated for fitted walls is that figure
# to the precision it is stated in: 0.05 % is under 0.055 %.
#: The most a whole-cell box's modes may be off the stencil's dispersion relation (a
#: fraction): the solver lowers every link's weight by 2^-18, a fitted cell's as a plain
#: cell's, which slows its sound by 2^-19 (2e-6).
DISPERSION_LIMIT = 1e-5
#: The most the 7-point stencil's relation written out in the script may differ from
#: ``grid_phase``'s (a fraction): both are the same formula, up to rounding.
CLOSED_FORM_LIMIT = 1e-9
#: The most (%) any of the turned box's lowest modes may be off with fitted walls, at any
#: placement: along the grid (0.05 %), turned up to 14 degrees about the vertical alone
#: (0.38 %), and turned further (4.3 %), the worst on the 125 Hz band's grid.
TURN_LIMITS_PERCENT = (0.055, 0.385, 4.35)
#: The least (%) the worst of the turned box's lowest modes is off on each grid with walls
#: on cell faces: the reference sees a wall up to half a cell from the model's.
CELL_FACES_LEAST_PERCENT = 1.0
#: The most the box's exact levels may move with more images and a longer FFT (dB).
LEVELS_REFERENCE_LIMIT_DB = 0.01
#: The most (dB) any band level in the box may be off its exact one with fitted walls, and
#: the least the worst of them is off on each grid with walls on cell faces: the error the
#: single wall on cell faces was stated to have, 1.82 dB. The box's levels hold the grid's
#: dispersion and the source's and the listeners' readings beside the walls' places, and
#: the box with every side a whole number of cells is another box (its sides moved by up
#: to half a cell), so it gives no bound on that part for this one: the limit is coarse.
LEVELS_LIMIT_DB = 1.825
#: The most the lowest mode of the L-shaped room with its walls on cell faces may be off
#: the plan's own 5-point problem at the grid's cell (a fraction), as DISPERSION_LIMIT.
L_FACES_LIMIT = 1e-5
#: The range the fitted L-shaped room's lowest mode lies in over the scan, % off the
#: continuum's (least, most) on each grid, as stated (1.4 % below to 1.0 % above, 0.5 %
#: below to 0.4 % above), each to the precision it is stated in.
L_SCAN_RANGE = {125: (-1.45, 1.05), 250: (-0.55, 0.45)}
#: The most the duct's reflection may be off (1 - beta) / (1 + beta), wherever its end falls
#: in a cell: an end of admittance up to 0.5, and one of admittance 1, which reflects nothing
#: in the continuum, where the grid's own wall (walls on cell faces) gives back 0.044 of the
#: wave over these frequencies.
DUCT_LIMIT = 0.012
DUCT_MATCHED_LIMIT = 0.1
#: The most the duct's reflection off the end with a branch may be off the grid's theory
#: for that admittance (a wall half a cell past the last centre): at its ends 0 and 0.25,
#: and at -0.48, where the fitted update itself lies 0.0228 off it.
DUCT_BRANCH_LIMIT = 0.02
DUCT_BRANCH_FAR_LIMIT = 0.0278
#: The admittance written out in the script against Admittance's (a fraction).
ADMITTANCE_FORM_LIMIT = 1e-12
#: The most a mode's first-order decay from the layout may be off modal theory's (a
#: fraction).
DECAY_LIMIT = 0.04
#: The most the box's branch weights summed may be off L/2 times its walls' area (a
#: fraction): each weight is a float, rounded within 6e-8 of its value.
BRANCH_WEIGHTS_LIMIT = 1e-6
#: The long runs: a rigid room's last block of the oscillating part within these of its
#: second, and no block above LONG_MOST times the second; an absorbing or open room's
#: oscillating part below LONG_FLOOR of the first block's rms from the half on, and no block
#: of the second half above LONG_REGROW times the largest of the sixth to the tenth (or
#: LONG_DUST of the first block); the rigid room's largest eigenvalue under the bound 4, and
#: each row of its update (the centre's coefficient and the links, as stored) not below 0
#: and summing to at most LONG_ROW_SUM: then no eigenvalue of the update is below 0, where
#: a uniform pressure would grow, nor above 4 (the rows' sums bound a non-negative
#: matrix's eigenvalues).
LONG_KEEPS = (0.97, 1.05)
LONG_MOST = 1.1
LONG_ROW_SUM = 2.0
LONG_FLOOR = 1e-6
LONG_REGROW = 1.1
LONG_DUST = 1e-9
LONG_BOUND = 4.0
#: The single wall's error in the octave of 8 to 16 cells per wavelength (dB): the most
#: with fitted walls along the grid (0.04 dB) and at up to 14 degrees to it (0.18 dB),
#: and the least with walls on cell faces 0.48 of a cell off.
WALL_ALONG_LIMIT_DB = 0.045
WALL_SLANTED_LIMIT_DB = 0.185
WALL_SLANTED_UP_TO_DEG = 14.0
WALL_CELL_FACES_LEAST_DB = 1.0
#: The most fitted walls' time per step may be over the 7-point stencil's with walls on
#: cell faces (a ratio), and the most the fit may take over every grid made (s).
COST_STEP_RATIO = 2.4
COST_FIT_S = 2.9


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


def _turned(yaw: float, pitch: float):
    """The placement of the box turned by ``yaw`` and ``pitch``, as CASES names it."""
    if pitch == 0.0:
        return ("shift", (0.0, 0.0, 0.0)) if yaw == 0.0 else ("turn", yaw)
    return ("tilt", (yaw, pitch))


def placed_box(case, cell: float, offset=OFFSET) -> dict:
    """The box placed on a grid of ``cell``: its mesh, its size, the maps from its own
    frame to the grid's and back, and the grid's origin and shape (as wave_responses lays
    it round the box before any shift, ``offset`` cells below its lowest corner; on cell
    faces, the box from 2 cells up)."""
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
        origin = (vertices - shift).min(axis=0) - cell * np.asarray(offset)
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


def _rigid_solver(vertices, faces, origin, cell, shape, source, probes, stencil, walls: str):
    """The rigid model's solver; with ``walls="fitted"`` its faces fitted to the grid. It
    stops the script when the solver made has other walls than ``walls``."""
    return _solver(vertices, faces, origin, cell, shape, source, probes, stencil, walls)


def _solver(
    vertices,
    faces,
    origin,
    cell,
    shape,
    source,
    probes,
    stencil,
    walls: str,
    materials=None,
    admittances=None,
    layer: int = 0,
):
    """The model's solver: each face's material (an index into ``admittances``, by
    default all 0 and one rigid material), with ``layer`` cells of matched layer round
    the grid; with ``walls="fitted"`` its faces fitted to the grid. It stops the script
    when the solver made has other walls than ``walls``."""
    fitted = walls == "fitted"
    fitting = {}
    if fitted:
        fitting = {
            "fitted_walls": True,
            "vertices": np.ascontiguousarray(vertices, dtype=float),
            "faces": np.ascontiguousarray(faces, dtype=np.int64),
        }
    if layer:
        fitting.update(layer=layer, layer_damping=LAYER_DAMPING)
    if admittances is None:
        admittances = (Admittance(0.0),)
    rate = C / (COURANTS[stencil] * cell)
    branches = [np.zeros((0, 3))] + [w.branches(rate) for w in admittances]
    owners = [np.zeros(0, np.int64)] + [
        np.full(len(w.branches(rate)), k, np.int64) for k, w in enumerate(admittances)
    ]
    solver = _native.WaveSolver(
        _native.Bvh(vertices, faces),
        np.zeros(len(faces), np.int64) if materials is None else np.asarray(materials, np.int64),
        np.array([w.beta for w in admittances], dtype=float),
        np.ascontiguousarray(np.vstack(branches)),
        np.concatenate(owners),
        origin,
        cell,
        *map(int, shape),
        np.asarray(source, dtype=float),
        np.asarray(probes, dtype=float).reshape(-1, 3),
        threads=THREADS,
        stencil=stencil,
        **fitting,
    )
    if solver.fitted_walls != fitted:
        made = "fitted walls" if solver.fitted_walls else "walls on cell faces"
        raise RuntimeError(f"the solver was made with {made}, not the walls asked for ({walls})")
    return solver


#: The directions of the links, in the solver's order (fdtd.hpp's kSteps).
STEPS = np.array(
    [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]]
    + [[a, b, 0] for b in (-1, 1) for a in (-1, 1)]
    + [[a, 0, b] for b in (-1, 1) for a in (-1, 1)]
    + [[0, a, b] for b in (-1, 1) for a in (-1, 1)]
)


def update_operator(layout: dict, volumes=None) -> tuple[sparse.csr_matrix, np.ndarray]:
    """The rigid grid's update as p+ = (2 - A) p - p-, A over the air cells, from the
    solver's layout, and the air cells' indices in the grid. With ``volumes`` (each grid
    cell's volume V as the update steps it, see :func:`cell_volumes`), A is D^-1 K with
    K symmetric and D the air cells' V, and this returns D^1/2 A D^-1/2 instead,
    which is K's scaling and so symmetric, with A's eigenvalues."""
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
    a = sparse.csr_matrix((v, (r, c)), shape=(air.size, air.size))
    if volumes is not None:
        root = np.sqrt(np.asarray(volumes, dtype=float)[air])
        a = (sparse.diags(root) @ a @ sparse.diags(1.0 / root)).tocsr()
    return a, air


def cell_volumes(solver, layout: dict):
    """Each grid cell's volume V as the update steps it (cells^3), x fastest: a fitted
    solver's special cells' V from ``cut_cells()["volume"]`` (the fit's air volume, raised
    by the clamp where a small cell's links would otherwise weigh more than 2 V), 1 for
    every other cell; None with walls on cell faces, where every air cell is a whole one."""
    cut = solver.cut_cells()
    if not cut:
        return None
    special = np.asarray(layout["special"]).astype(np.int64)
    assert np.array_equal(np.asarray(cut["special"]), special), (
        "the fit's cells are not the layout's"
    )
    volumes = np.ones(np.asarray(layout["types"]).size)
    volumes[special] = np.asarray(cut["volume"], dtype=float)
    return volumes


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

    if a.shape[0] <= DENSE_CELLS:
        mu, vectors = scipy_linalg.eigh(a.toarray())
    else:
        operator = sparse_linalg.LinearOperator(
            a.shape, matvec=filtered, matmat=filtered, dtype=float
        )
        start = np.random.default_rng(0).standard_normal(a.shape[0])
        # Half as many again until more are found than lie below the top: walls far from
        # the model's can bring more modes below it than the box has.
        k = int(1.15 * expected) + 10
        while True:
            _, basis = sparse_linalg.eigsh(operator, k=k, which="LA", tol=1e-10, v0=start)
            projected = basis.T @ (a @ basis)
            mu, rotation = scipy_linalg.eigh((projected + projected.T) / 2)
            vectors = basis @ rotation
            if (mu < mu_top).sum() < k:
                break
            k += k // 2
    below = mu < mu_top
    residual = float(
        np.linalg.norm(a @ vectors[:, below] - vectors[:, below] * mu[below], axis=0).max()
    )
    assert residual < 1e-8, f"an eigenvector is off by {residual:.1e}"
    return mu[below], vectors[:, below]


def relation_frequency(numbers: np.ndarray, size, turn: np.ndarray, cell: float, stencil: int):
    """The frequency (in steps, a fraction of the rate) each rigid-box mode ``numbers``
    (modes, 3) has on the stencil's grid, from its dispersion relation: the mean over the
    mode's eight plane waves, their wave vectors (+-n pi / L in the box's frame) turned
    into the grid's by ``turn``. Along the grid all eight have the same."""
    k = (SIGNS[None, :, :] * np.atleast_2d(numbers)[:, None, :] * np.pi / np.asarray(size)) @ turn.T
    return grid_phase(k * cell, stencil).mean(axis=1) / (2 * math.pi)


def grid_modes(case, band: int, stencil: int, walls: str, offset=OFFSET) -> dict:
    """The box's modes in each band up to ``band`` on that band's grid (its origin
    ``offset`` cells below the box's lowest corner): per band the largest and
    root-mean-square deviation (%) from the analytic modes, each analytic mode matched to
    the eigenvector of the update it correlates with best (its shape sampled at the air
    cells' centres, weighted by the square root of the volumes the update steps them
    with); on the grids of TURN_BANDS also the LOWEST lowest modes against the exact ones
    as the stencil's dispersion moves them."""
    g = grid(band, stencil)
    cell, period = g.cell, 1.0 / g.rate
    box = placed_box(case, cell, offset)
    middle = box["world"](box["size"] / 2)
    solver = _rigid_solver(
        box["vertices"],
        box["faces"],
        box["origin"],
        cell,
        box["shape"],
        middle,
        middle,
        stencil,
        walls,
    )
    layout = solver.layout()
    volumes = cell_volumes(solver, layout)
    a, air = update_operator(layout, volumes)
    asymmetry = float(abs(a - a.T).max())
    assert asymmetry < 1e-6, f"the update is not symmetric ({asymmetry:.1e})"
    if volumes is not None:
        a = ((a + a.T) / 2).tocsr()
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
    root_volume = np.ones(air.size) if volumes is None else np.sqrt(volumes[air])
    # Eigenvalues too close for their eigenvectors to be told apart count as one: an
    # analytic mode's correlation with them is that with the space they span, and its
    # frequency their mean (they are the same to within CLUSTER).
    starts = np.flatnonzero(np.r_[True, np.diff(f_grid) > CLUSTER * f_grid[1:]])
    groups = np.add.reduceat(np.ones(len(f_grid)), starts).astype(int)
    f_group = np.add.reduceat(f_grid, starts) / groups
    member = np.repeat(np.arange(len(starts)), groups)

    def sampled(n: np.ndarray) -> np.ndarray:
        """The analytic mode ``n``'s shape at the air cells, as the eigenvectors hold it."""
        return np.prod(np.cos(np.pi * n * centres / box["size"]), axis=1) * root_volume

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
            shares = shares_of(sampled(n))
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
            j, _ = best(sampled(n))
            miss.append(abs(f_group[j] / f_r - 1))
        out["off_relation"] = 100 * max(miss)
        if stencil == 7:
            # The 7-point stencil's relation written out apart from grid_phase:
            # sin^2(w T / 2) is the mean over the axes of sin^2(k h / 2).
            half = np.sin(np.pi * numbers[below] * cell / box["size"] / 2) ** 2
            closed = np.arcsin(np.sqrt(half.sum(axis=1) / 3)) / (math.pi * period)
            out["relation_off_closed_form"] = float(np.abs(relation[below] / closed - 1).max())
    if band in TURN_BANDS and case[0] != "faces":
        # The lowest modes against the exact ones as the grid's dispersion moves them.
        lowest = numbers[:LOWEST]
        moved = relation_frequency(lowest, box["size"], _turn(case), cell, stencil) / period
        off, correlation = [], []
        for n, f_d in zip(lowest, moved, strict=True):
            j, r = best(sampled(n))
            off.append(100 * (f_group[j] / f_d - 1))
            correlation.append(r)
        out["lowest"] = {
            "modes": [n.tolist() for n in lowest],
            "percent": off,
            "min_percent": min(off),
            "max_percent": max(off),
            "rms_percent": float(np.sqrt(np.mean(np.square(off)))),
            "least_correlation": min(correlation),
        }
    if volumes is not None:
        stats = solver.cut_stats
        out["fit"] = {
            "seconds": float(stats["seconds"]),
            "special_cells": len(layout["special"]),
            "raised": int(stats["raised"]),
            "sliver_links": int(stats["sliver_links"]),
        }
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


def box_levels(case, band: int, stencil: int, reference: dict, walls: str) -> dict:
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
        walls,
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


def single_wall(stencil: int, angle: float, share: float, walls: str, band: int = 250) -> dict:
    """One rigid wall at ``angle`` (degrees, about z) to the grid, through a point
    ``share`` of a cell in front of a cell face, on the ``band``'s grid: the rms of
    its band levels at the centres 1.5 to 6.5 cells in front of it over the grid's own
    field of the source and its image, per octave of cells per wavelength; along the grid
    also over that of the source and its image in the cell face; and the wall's place
    that best explains the error. The free fields are always run with walls on cell
    faces: they are the reference, and hold no wall near the centres."""
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
    solver = _rigid_solver(vertices, faces, origin, cell, shape, source, centres, stencil, walls)
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
            "cell_faces",
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
    out["best_place_cells"] = float(WALL_SHIFTS[best])
    out["rms_best_place_db"] = misses[best]
    return out


def _about(yaw: float, pitch: float) -> np.ndarray:
    """Turned by ``yaw`` degrees about the vertical after ``pitch`` degrees about x."""
    return _turn(("tilt", (yaw, pitch)))


def l_plan(size, notch) -> list[tuple[float, float]]:
    """The L-shaped plan's corners, counter-clockwise from the origin: the rectangle of
    ``size`` (x, y) with the ``notch`` cut from its corner at (x, y) = ``size``."""
    x, y = size
    a, b = notch
    return [(0.0, 0.0), (x, 0.0), (x, y - b), (x - a, y - b), (x - a, y), (0.0, y)]


def l_room_mesh(size, notch, ceiling: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The L-shaped room from z = 0 to ``size[2]`` (vertices, faces), every face's normal
    out of the room; without its ceiling when ``ceiling`` is False. The floor and the
    ceiling are fans from the plan's concave corner."""
    plan = l_plan(size[:2], notch)
    n = len(plan)
    vertices = np.array([(x, y, z) for z in (0.0, size[2]) for x, y in plan])
    fan = [(3, (3 + k) % n, (4 + k) % n) for k in range(1, n - 1)]
    faces = [(a, c, b) for a, b, c in fan]  # the floor, its normal down
    if ceiling:
        faces += [(n + a, n + b, n + c) for a, b, c in fan]
    for k in range(n):
        a, b = k, (k + 1) % n
        faces += [(a, b, n + b), (a, n + b, n + a)]
    return vertices, np.array(faces, np.int64)


def l_mode_2d(cells, notch_cells, square: float) -> float:
    """The lowest non-zero eigenvalue (1/m^2) of minus the Laplacian on the L-shaped plan
    of ``cells`` (x, y) squares of side ``square`` with ``notch_cells`` cut from its far
    corner, the walls on the squares' sides (no flow through them): the 5-point stencil
    on the squares' centres, written here."""
    nx, ny = (int(c) for c in cells)
    ax, ay = (int(c) for c in notch_cells)
    inside = np.ones((nx, ny), bool)
    inside[nx - ax :, ny - ay :] = False
    index = -np.ones((nx, ny), np.int64)
    index[inside] = np.arange(int(inside.sum()))
    rows, columns = [], []
    for di, dj in ((1, 0), (0, 1)):
        here = inside[: nx - di, : ny - dj] & inside[di:, dj:]
        a = index[: nx - di, : ny - dj][here]
        b = index[di:, dj:][here]
        rows += [a, b]
        columns += [b, a]
    r, c = np.concatenate(rows), np.concatenate(columns)
    count = int(inside.sum())
    links = sparse.csr_matrix((np.ones(r.size), (r, c)), shape=(count, count))
    laplacian = sparse.diags(np.asarray(links.sum(axis=1)).ravel()) - links
    mu = sparse_linalg.eigsh(
        laplacian.tocsc(), k=2, sigma=-1e-9, which="LM", return_eigenvectors=False, tol=1e-14
    )
    return float(np.max(mu)) / square**2


def l_reference(cell: float) -> dict:
    """The L-shaped room on a grid of ``cell``: its sides and its notch rounded to whole
    cells, the plan's lowest mode (1/m^2) on the 5-point stencil at that cell, and the
    continuum's: the same on squares of L_FINEST_M and twice and four times it (each a
    whole part of the cell), extrapolated to squares of nothing as a power of their side
    found from the three."""
    cells = np.round(L_SIZE / cell).astype(int)
    notch = np.round(L_NOTCH / cell).astype(int)
    finest = 4 * max(1, round(cell / L_FINEST_M / 4))
    parts = (finest // 4, finest // 2, finest)
    levels = [l_mode_2d(cells[:2] * p, notch * p, cell / p) for p in parts]
    a, b, c = levels
    assert (a - b) * (b - c) > 0, "the plan's modes do not converge one way"
    power = math.log2((a - b) / (b - c))
    continuum = c - (b - c) / (2.0**power - 1.0)
    return {
        "cells": cells.tolist(),
        "notch_cells": notch.tolist(),
        "size_m": (cells * cell).tolist(),
        "on_cell": l_mode_2d(cells[:2], notch, cell),
        "parts": list(parts),
        "levels": levels,
        "power": power,
        "continuum": continuum,
    }


def _hz(mu: float, rate: float) -> float:
    """The frequency (Hz) a mode of the update's eigenvalue ``mu`` rings at."""
    return float(np.arcsin(np.sqrt(max(mu, 0.0)) / 2) * rate / math.pi)


def _lowest_mode(solver) -> float:
    """The lowest non-zero eigenvalue of a closed rigid grid's update (its uniform
    pressure's, about 1e-9 or less, left out)."""
    layout = solver.layout()
    volumes = cell_volumes(solver, layout)
    a, _ = update_operator(layout, volumes)
    if volumes is not None:
        a = ((a + a.T) / 2).tocsr()
    mu = sparse_linalg.eigsh(
        a.tocsc(), k=3, sigma=-1e-5, which="LM", return_eigenvectors=False, tol=1e-12
    )
    return float(np.sort(mu[mu > 1e-6])[0])


def l_scan_placements() -> list[tuple[float, float, float]]:
    """The placements of the fitted L-shaped room's scan: k / L_SCAN_STEPS cells along each
    axis, and 0.5 -+ L_SCAN_CROSS."""
    steps = [k / L_SCAN_STEPS for k in range(L_SCAN_STEPS)]
    steps += [0.5 - L_SCAN_CROSS, 0.5 + L_SCAN_CROSS]
    return [(s, s, s) for s in sorted(steps)]


def l_room(band: int, walls: str) -> list[dict]:
    """The L-shaped room's lowest mode on the band's 7-point grid at each placement of
    L_PLACEMENTS (rows with ``scan`` False) and, with fitted walls, of the scan
    (``scan`` True), against the plan's own at the grid's cell and the continuum's."""
    g = grid(band, 7)
    cell, rate = g.cell, g.rate
    reference = l_reference(cell)
    size = np.array(reference["size_m"])
    notch = np.array(reference["notch_cells"]) * cell
    vertices, faces = l_room_mesh(size, notch)
    courant = COURANTS[7]

    def hz(kappa: float) -> float:
        return _hz(courant**2 * cell**2 * kappa, rate)

    rows = []
    placements = [(shift, False) for shift in L_PLACEMENTS]
    if walls == "fitted":
        placements += [(shift, True) for shift in l_scan_placements()]
    for shift, scan in placements:
        moved = vertices + np.array(shift) * cell
        origin = -2.0 * cell * np.ones(3)
        shape = np.ceil((moved.max(axis=0) - origin) / cell).astype(int) + 2
        source = np.array([0.35, 0.3, 0.45]) + np.array(shift) * cell
        started = time.perf_counter()
        solver = _solver(moved, faces, origin, cell, shape, source, source, 7, walls)
        f = _hz(_lowest_mode(solver), rate)
        rows.append(
            {
                "walls": walls,
                "stencil": 7,
                "grid_band_hz": band,
                "cell_m": cell,
                "shift_cells": list(shift),
                "scan": scan,
                "size_m": reference["size_m"],
                "lowest_hz": f,
                "plan_on_cell_hz": hz(reference["on_cell"]),
                "continuum_hz": hz(reference["continuum"]),
                "continuum_free_hz": C * math.sqrt(reference["continuum"]) / (2 * math.pi),
                "extrapolation": {
                    "parts": reference["parts"],
                    "levels_hz": [hz(k) for k in reference["levels"]],
                    "power": reference["power"],
                },
                "seconds": time.perf_counter() - started,
            }
        )
    return rows


def _duct(length: float, far: Admittance, walls: str, source_x: float, at: float, seconds: float):
    """The pressure at x = ``at`` in a duct DUCT_WIDTH square from x = 0 (admittance 1) to
    ``length`` (the admittance ``far``), its sides rigid, the source at ``source_x``, on
    cells of DUCT_CELL whose faces lie 0.0137 of a cell off x = 0; and the sample rate."""
    cell, width = DUCT_CELL, DUCT_WIDTH
    size = np.array([length, width, width])
    mesh = box_mesh(tuple(size))
    x = mesh.vertices[mesh.faces][:, :, 0]
    materials = np.where(np.all(x == 0.0, axis=1), 1, np.where(np.all(x == length, axis=1), 2, 0))
    origin = -cell * (2.0 + OFFSET)
    shape = np.ceil((size - origin) / cell).astype(int) + 2
    middle = width / 2
    solver = _solver(
        mesh.vertices,
        mesh.faces,
        origin,
        cell,
        shape,
        (source_x, middle, middle),
        (at, middle, middle),
        7,
        walls,
        materials,
        (Admittance(0.0), Admittance(1.0), far),
    )
    rate = C / (COURANTS[7] * cell)
    signal, _ = source_signal(rate, cell, COURANTS[7])
    steps = int(seconds * rate)
    drive = np.zeros(steps, np.float32)
    drive[: min(steps, len(signal))] = signal[:steps]
    return np.asarray(solver.run(drive), dtype=float)[0], rate


def _duct_end(e: float, cells: int) -> float:
    """The length of a duct about ``cells`` cells long whose far end lies ``e`` cells past
    a cell face."""
    return DUCT_CELL * (cells + e - OFFSET[0])


def duct_reflection(beta: float, e: float, walls: str) -> dict:
    """The duct's far end of admittance ``beta``, ``e`` cells past a cell face: its
    reflection heard 20 m from the source, the incident and the reflected waves over
    DUCT_BAND gated with Hann windows 80 ms long, the square root of their energies'
    ratio, against (1 - beta) / (1 + beta)."""
    end = _duct_end(e, 800)
    pressure, rate = _duct(end, Admittance(beta), walls, 10.0, 30.0, 0.45)
    _, delay = source_signal(rate, DUCT_CELL, COURANTS[7])
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    lo, hi = DUCT_BAND
    gain = np.clip((f - lo) / 20, 0, 1) * np.clip((hi - f) / 50, 0, 1)
    pressure = np.fft.irfft(np.fft.rfft(pressure, n_fft) * gain, n_fft)[: len(pressure)]
    t = (np.arange(len(pressure)) - delay) / rate

    def gated(path: float) -> float:
        x = t - path / C
        window = np.where(np.abs(x) < 0.04, 0.5 + 0.5 * np.cos(np.pi * x / 0.04), 0.0)
        return float(np.sum((pressure * window) ** 2))

    found = math.sqrt(gated(2 * end - 40.0) / gated(20.0))
    expected = (1 - beta) / (1 + beta)
    return {"walls": walls, "beta": beta, "end_cells": e, "found": found, "off": found - expected}


def admittance_written(f: np.ndarray, wall: Admittance) -> np.ndarray:
    """The admittance of ``wall`` at the frequencies ``f``, written out here: beta plus, per
    resonant branch, its conductance over 1 + i Q (f / f_0 - f_0 / f)."""
    out = np.full(np.shape(f), wall.beta, dtype=complex)
    for g, q, f0 in zip(wall.conductance, wall.quality, wall.frequency_hz, strict=True):
        out = out + g / (1.0 + 1j * q * (f / f0 - f0 / f))
    return out


def duct_branch_reflection(e: float, walls: str, reference: dict) -> dict:
    """The duct's end of RESONANT, ``e`` cells past a cell face: the reflected wave (the
    run less one in a duct long enough that nothing comes back in time) over that off a
    rigid end at the same place, over DUCT_BRANCH_BAND, against the grid's theory for the
    admittance the scheme has (a wall half a cell past the last centre) and the
    continuum's. ``reference`` keeps the long duct's run from one call to the next."""
    if walls not in reference:
        reference[walls] = _duct(200.0, Admittance(0.0), walls, 60.0, 110.0, 0.6)
    incident, rate = reference[walls]
    end = _duct_end(e, 1300)
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    band = (f > DUCT_BRANCH_BAND[0]) & (f < DUCT_BRANCH_BAND[1])
    f = f[band]
    ratio = []
    for far in (RESONANT, Admittance(0.0)):
        pressure, _ = _duct(end, far, walls, 60.0, 110.0, 0.6)
        ratio.append(np.fft.rfft(pressure - incident, n_fft)[band])
    found = np.abs(ratio[0] / ratio[1])
    theory = np.abs(grid_reflection(f, rate, np.conj(RESONANT(f, rate)), COURANTS[7]))
    exact = admittance_written(f, RESONANT)
    continuum = np.abs((1 - exact) / (1 + exact))
    return {
        "walls": walls,
        "end_cells": e,
        "off_grid_theory": float(np.abs(found - theory).max()),
        "off_continuum": float(np.abs(found - continuum).max()),
        "theory_off_continuum": float(np.abs(theory - continuum).max()),
        "admittance_off_written": float(np.abs(RESONANT(f) - exact).max() / np.abs(exact).max()),
    }


def _turned_box(band: int, walls: str, admittances) -> tuple:
    """The room-sized box turned DECAY_TURN degrees on the band's 7-point grid, its floor
    and ceiling of ``admittances[0]`` and its four walls of ``admittances[1]``: the solver,
    the placed box (see :func:`placed_box`) and each face's material."""
    cell = grid(band, 7).cell
    box = placed_box(("turn", DECAY_TURN), cell)
    size = box["size"]
    own = box_mesh(tuple(size)).vertices[box["faces"]][:, :, 2]
    floors = np.all(own == 0.0, axis=1) | np.all(own == size[2], axis=1)
    materials = np.where(floors, 0, 1)
    middle = box["world"](size / 2)
    solver = _solver(
        box["vertices"],
        box["faces"],
        box["origin"],
        cell,
        box["shape"],
        middle,
        middle,
        7,
        walls,
        materials,
        tuple(admittances),
    )
    return solver, box, materials


def box_decay(band: int, walls: str) -> dict:
    """The room-sized box turned DECAY_TURN degrees, its floor and ceiling and its four
    walls of the admittances DECAY_BETAS: each of its DECAY_MODES lowest modes' first-order
    decay rate from the update's own terms, sum over the cells of g psi^2 over sum V
    psi^2 (psi the rigid update's mode, V the cell's volume, g = V (1 - coef_b) its wall
    term), against modal theory's (validation/wave_solver.py's ``modal_decay``, a pair of
    walls at a time)."""
    g = grid(band, 7)
    cell, period = g.cell, 1.0 / g.rate
    solver, box, _ = _turned_box(band, walls, [Admittance(b) for b in DECAY_BETAS])
    size = box["size"]
    layout = solver.layout()
    volumes = cell_volumes(solver, layout)
    a, air = update_operator(layout, volumes)
    if volumes is not None:
        a = ((a + a.T) / 2).tocsr()
    mu, vectors = sparse_linalg.eigsh(a.tocsc(), k=DECAY_MODES + 8, sigma=-1e-4, which="LM")
    order = np.argsort(mu)
    mu, vectors = mu[order], vectors[:, order]
    shape = box["shape"]
    ijk = np.stack([air % shape[0], (air // shape[0]) % shape[1], air // (shape[0] * shape[1])], 1)
    centres = box["own"](box["origin"] + (ijk + 0.5) * cell)
    volume = np.ones(air.size) if volumes is None else volumes[air]
    loss = np.zeros(np.asarray(layout["types"]).size)
    loss[np.asarray(layout["special"]).astype(np.int64)] = 1.0 - np.asarray(
        layout["coef_b"], dtype=float
    )
    loss = loss[air]
    numbers = box_modes(size, 200.0)[0][:DECAY_MODES]
    rows = []
    for n in numbers:
        shape_n = np.prod(np.cos(np.pi * n * centres / size), axis=1) * np.sqrt(volume)
        correlation = np.abs(shape_n @ vectors) / np.linalg.norm(shape_n)
        j = int(np.argmax(correlation))
        y = vectors[:, j]
        found = float(np.sum(loss * y**2) / np.sum(y**2) / period)
        theory = 0.0
        for axis, beta in ((0, DECAY_BETAS[1]), (1, DECAY_BETAS[1]), (2, DECAY_BETAS[0])):
            only = np.full(3, np.inf)
            only[axis] = size[axis]
            theory += float(modal_decay(n[None, :], beta, only, C)[0])
        rows.append(
            {
                "mode": n.tolist(),
                "hz": _hz(float(mu[j]), 1.0 / period),
                "correlation": float(correlation[j]),
                "decay_per_s": found,
                "theory_per_s": theory,
                "ratio": found / theory,
            }
        )
    return {"walls": walls, "stencil": 7, "grid_band_hz": band, "cell_m": cell, "modes": rows}


def branch_weights(band: int, walls: str) -> dict:
    """The box of :func:`_turned_box`, its floor and ceiling rigid and its four walls of
    RESONANT (one branch): the layout's branch states' weights, each times its cell's
    volume V as the update steps it, summed, against L/2 times the four walls' area in
    cells squared (L the stencil's Courant number), which a state weighing (L/2) W / V for
    a wall of area W in its cell sums to. With walls on cell faces W is the staircase's,
    larger than the walls' own."""
    cell = grid(band, 7).cell
    solver, box, materials = _turned_box(band, walls, [Admittance(0.0), RESONANT])
    layout = solver.layout()
    special = np.asarray(layout["special"], dtype=np.int64)
    ends = np.asarray(layout["state_start"], dtype=np.int64)
    volumes = cell_volumes(solver, layout)
    volume = volumes[special] if volumes is not None else np.ones(special.size)
    weights = np.asarray(layout["state_weight"], dtype=float)
    found = float(np.sum(weights * np.repeat(volume, np.diff(ends))))
    corners = box["vertices"][box["faces"][materials == 1]]
    sides = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = 0.5 * float(np.linalg.norm(sides, axis=1).sum())
    expected = COURANTS[7] / 2 * area / cell**2
    return {
        "walls": walls,
        "grid_band_hz": band,
        "cell_m": cell,
        "states": int(weights.size),
        "found": found,
        "expected": expected,
        "off": found / expected - 1,
    }


def long_room(name: str, case: str) -> dict:
    """A long run's model: ``name`` "lroom", the L-shaped room with plates (2.4 x 2.0 x
    1.5 m, its notch 1.0 x 0.9 m; a free plate turned 31 and 57 degrees, a horizontal one,
    two upright ones with a 3 cm slit between them) turned 22.5 and 11 degrees, on 10 cm
    cells; or "box", the room-sized box turned 22.5 degrees on the 125 Hz band's grid, its
    origin 2 + (0.271, 0.613, 0.137) cells below the box. ``case``: "rigid", "strong"
    (every face of admittance 1), "absorbing" (LONG_WALL) or "open" (the ceiling left
    out, in a grid with a matched layer of LAYER_CELLS cells round it and 4 more above)."""
    rng = np.random.default_rng(11)
    if name == "lroom":
        size, notch, cell = L_SIZE, L_NOTCH, 0.1
        turn, offset = _about(22.5, 11.0), np.array([0.311, 0.173, 0.457])
        room, faces = l_room_mesh(size, notch, ceiling=case != "open")
        plates = [
            np.array(
                [[-0.35, -0.25, 0.0], [0.35, -0.25, 0.0], [0.35, 0.25, 0.0], [-0.35, 0.25, 0.0]]
            )
            @ _about(31.0, 57.0).T
            + np.array([0.75, 0.6, 0.8]),
            np.array(
                [[1.3, 0.25, 0.537], [2.0, 0.25, 0.537], [2.0, 0.75, 0.537], [1.3, 0.75, 0.537]]
            ),
            np.array([[0.2, 0.6, 0.1], [0.9, 0.6, 0.1], [0.9, 0.6, 0.9], [0.2, 0.6, 0.9]]),
            np.array([[0.93, 0.6, 0.1], [1.2, 0.6, 0.1], [1.2, 0.6, 0.9], [0.93, 0.6, 0.9]]),
        ]
        parts, triangles = [room], [faces]
        for plate in plates:
            count = sum(len(p) for p in parts)
            parts.append(plate)
            triangles.append(np.array([[0, 1, 2], [0, 2, 3]]) + count)
        own = np.vstack(parts)
        faces = np.vstack(triangles)
        source = np.array([0.35, 0.3, 0.45])
        points = rng.uniform(0.05, 0.95, (8 * LONG_POINTS, 3)) * size
        notched = (points[:, 0] > size[0] - notch[0] - 0.05) & (
            points[:, 1] > size[1] - notch[1] - 0.05
        )
        points = points[~notched][:LONG_POINTS]
    else:
        size, cell = SIZE, grid(125, 7).cell
        turn, offset = _about(22.5, 0.0), np.array([0.271, 0.613, 0.137])
        mesh = box_mesh(tuple(size))
        own, faces = mesh.vertices, mesh.faces
        source = SOURCE
        points = rng.uniform(0.05, 0.95, (LONG_POINTS, 3)) * size
    centre = size / 2

    def world(p):
        return (np.asarray(p, dtype=float) - centre) @ turn.T + centre

    vertices = world(own)
    origin = vertices.min(axis=0) - cell * (2.0 + offset)
    shape = np.ceil((vertices.max(axis=0) - origin) / cell).astype(int) + 2
    layer = 0
    if case == "open":
        layer = LAYER_CELLS
        origin = origin - cell * layer
        shape = shape + 2 * layer + np.array([0, 0, 4])
    admittances = {
        "rigid": (Admittance(0.0),),
        "open": (Admittance(0.0),),
        "strong": (Admittance(1.0),),
        "absorbing": (LONG_WALL,),
    }[case]
    return {
        "vertices": vertices,
        "faces": faces,
        "cell": cell,
        "origin": origin,
        "shape": shape,
        "source": world(source),
        "points": world(points),
        "admittances": admittances,
        "layer": layer,
    }


def update_rows(layout: dict) -> dict:
    """The rigid update p+ = M p - p- read from the layout's coefficients (M's centre and
    links, each a float as stored, summed in float64): the least of them, and the largest
    sum of a row of M (a special cell's or a plain one's)."""
    centre = np.asarray(layout["coef_a"], dtype=float)
    links = np.asarray(layout["link_weight"], dtype=float)
    counts = np.diff(np.asarray(layout["link_start"], dtype=np.int64))
    sums = centre.copy()
    np.add.at(sums, np.repeat(np.arange(centre.size), counts), links)
    plain = np.asarray(layout["weights"], dtype=float)
    plain_centre = float(layout["centre"][0])
    least = min(float(centre.min()), float(links.min()), plain_centre, float(plain.min()))
    most = max(float(sums.max()), plain_centre + float(plain.sum()))
    return {"least_coefficient": least, "row_sum": most, "rows_over": int(np.sum(sums > 2.0))}


def long_run(name: str, case: str) -> dict:
    """LONG_STEPS steps of the model of :func:`long_room` with fitted walls, heard at its
    points: the rms over them per block of LONG_BLOCK steps, of the whole signal and of its
    oscillating part (each point's mean over the block taken out); for a closed room the
    uniform pressure's Rayleigh quotient, and for the rigid ones the update's largest
    eigenvalue, its least coefficient and its rows' largest sum (float64 sums of the stored
    floats); the clamp's counts."""
    model = long_room(name, case)
    started = time.perf_counter()
    solver = _solver(
        model["vertices"],
        model["faces"],
        model["origin"],
        model["cell"],
        model["shape"],
        model["source"],
        model["points"],
        7,
        "fitted",
        admittances=model["admittances"],
        layer=model["layer"],
    )
    made = time.perf_counter() - started
    rate = C / (COURANTS[7] * model["cell"])
    out: dict = {"model": name, "case": case, "walls": "fitted", "steps": LONG_STEPS}
    stats = solver.cut_stats
    out.update(
        cell_m=model["cell"],
        air_cells=int(solver.air_cells),
        special_cells=len(solver.layout()["special"]),
        raised=int(stats["raised"]),
        most_raised=float(stats["most_raised"]),
        open=bool(solver.open),
    )
    mu = math.nan
    if not model["layer"]:
        layout = solver.layout()
        volumes = cell_volumes(solver, layout)
        plain, air = update_operator(layout)
        weight = volumes[air] if volumes is not None else np.ones(air.size)
        rows = np.asarray(plain.sum(axis=1)).ravel()
        mu = float(np.sum(weight * rows) / np.sum(weight))
        out["uniform_mu"] = mu
        if case == "rigid":
            a, _ = update_operator(layout, volumes)
            a = ((a + a.T) / 2).tocsc()
            out["largest_mu"] = float(
                sparse_linalg.eigsh(a, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0]
            )
            out.update(update_rows(layout))
    signal, _ = source_signal(rate, model["cell"], COURANTS[7])
    drive = np.zeros(LONG_STEPS, np.float32)
    drive[: len(signal)] = signal
    ready = time.perf_counter()
    p = np.asarray(solver.run(drive), dtype=float)
    out["run_s"] = time.perf_counter() - ready
    out["set_up_s"] = made
    blocks = p.reshape(len(p), -1, LONG_BLOCK)
    means = blocks.mean(axis=2)
    out["full"] = np.sqrt(np.mean(blocks**2, axis=(0, 2))).tolist()
    out["oscillating"] = np.sqrt(np.mean((blocks - means[:, :, None]) ** 2, axis=(0, 2))).tolist()
    out["finite"] = bool(np.isfinite(p).all())
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


def _stencils(walls: str) -> tuple[int, ...]:
    """The stencils each kind of walls is checked with: fitted walls take the 7-point one."""
    return (7,) if walls == "fitted" else (19, 7)


def _kind(row: dict) -> str:
    """The row's stencil, and its walls when they are fitted."""
    return f"{row['stencil']}-point" + (", fitted walls" if row["walls"] == "fitted" else "")


def modes_part(bands, walls, cases=CASES, turned: bool = True) -> tuple[list[dict], list[dict]]:
    """The box's modes on each band's grid, each kind of walls with its stencils, every
    placement in ``cases``; and with ``turned``, the turned box's lowest modes on the
    7-point grids of TURN_BANDS (sharing the runs of the placements that are the same)."""
    found: dict = {}

    def run(case, band: int, stencil: int, kind: str) -> dict:
        key = (case, band, stencil, kind)
        if key not in found:
            started = time.perf_counter()
            result = grid_modes(case, band, stencil, kind)
            result.update(walls=kind, stencil=stencil, grid_band_hz=band)
            result["seconds"] = time.perf_counter() - started
            _say(
                f"modes, {_kind(result)}, {band} Hz grid, {_label(case)}: {result['seconds']:.0f} s"
            )
            found[key] = result
        return found[key]

    rows, turns = [], []
    for kind in walls:
        for stencil in _stencils(kind):
            for band in bands:
                for case in cases:
                    rows.append(
                        {**run(case, band, stencil, kind), "case": _label(case), "kind": case[0]}
                    )
    if turned:
        for kind in walls:
            for band in (b for b in bands if b in TURN_BANDS):
                for yaw, pitch in TURNS:
                    result = run(_turned(yaw, pitch), band, 7, kind)
                    turns.append(
                        {
                            "walls": kind,
                            "stencil": 7,
                            "grid_band_hz": band,
                            "cell_m": result["cell_m"],
                            "yaw_deg": yaw,
                            "pitch_deg": pitch,
                            **result["lowest"],
                        }
                    )
    return rows, turns


def levels_part(bands, walls) -> tuple[list[dict], list[dict]]:
    """The box's band levels against its image sources on each band's grid, each kind of
    walls with its stencils, every placement; and how far the image sources' levels move
    with more."""
    rows, converged = [], []
    references: dict = {}
    for kind in walls:
        for stencil in _stencils(kind):
            reference = references.setdefault(stencil, {})
            for band in bands:
                if not any(
                    c["stencil"] == stencil and c["grid_band_hz"] == band for c in converged
                ):
                    converged.append(
                        {
                            "stencil": stencil,
                            "grid_band_hz": band,
                            "moves_db": reference_converged(band, stencil),
                        }
                    )
                for case in CASES:
                    started = time.perf_counter()
                    result = box_levels(case, band, stencil, reference, kind)
                    result.update(
                        walls=kind,
                        stencil=stencil,
                        grid_band_hz=band,
                        case=_label(case),
                        kind=case[0],
                    )
                    _say(
                        f"levels, {_kind(result)}, {band} Hz grid, {_label(case)}: "
                        f"{time.perf_counter() - started:.0f} s"
                    )
                    rows.append(result)
    return rows, converged


def wall_part(walls) -> list[dict]:
    """The single wall along the grid and at angles to it, each kind of walls with its
    stencils, and the walls of WALL_MORE on the 7-point stencil."""
    rows = []
    for kind in walls:
        for stencil in _stencils(kind):
            cases = [(0.0, share) for share in WALL_SHARES] + [(a, 0.0) for a in WALL_ANGLES]
            if stencil == 7:
                cases += list(WALL_MORE)
            for angle, share in cases:
                row = {"walls": kind, "stencil": stencil}
                row.update(single_wall(stencil, angle, share, kind))
                rows.append(row)
                _say(f"wall, {_kind(row)}, {_wall_label(row)}")
    return rows


def _wall_label(row: dict) -> str:
    if row["angle_deg"] == 0.0:
        return f"along the grid, {row['share']:g} cell off a face"
    if row["share"] == 0.0:
        return f"at {row['angle_deg']:g} deg"
    return f"at {row['angle_deg']:g} deg, {row['share']:g} cell off a face"


def _box_cost(walls: str, stencil: int) -> dict:
    """The box moved by no part of a cell on the 250 Hz band's grid: its solver made and
    run for COST_BOX_LENGTH_S of sound, the least of COST_REPEATS times."""
    g = grid(250, stencil)
    box = placed_box(("shift", (0.0, 0.0, 0.0)), g.cell)
    middle = box["world"](box["size"] / 2)
    steps = int(round(COST_BOX_LENGTH_S * g.rate))
    drive = np.zeros(steps, np.float32)
    drive[0] = 1.0
    made, stepped, fits = [], [], []
    for _ in range(COST_REPEATS):
        started = time.perf_counter()
        solver = _rigid_solver(
            box["vertices"],
            box["faces"],
            box["origin"],
            g.cell,
            box["shape"],
            middle,
            middle,
            stencil,
            walls,
        )
        ready = time.perf_counter()
        solver.run(drive)
        made.append(ready - started)
        stepped.append(time.perf_counter() - ready)
        if walls == "fitted":
            fits.append(float(solver.cut_stats["seconds"]))
    return {
        "model": "box",
        "scheme": "standard" if stencil == 7 else "isotropic",
        "stencil": stencil,
        "grid_band_hz": 250,
        "cells": int(solver.air_cells),
        "cell_m": g.cell,
        "steps": steps,
        "set_up_s": min(made),
        "fit_s": min(fits) if fits else None,
        "step_ms": 1e3 * min(stepped) / steps,
        "run_s": min(m + s for m, s in zip(made, stepped, strict=True)),
    }


def _room_cost(name: str, walls: str, scheme: str) -> dict:
    """An example's wave bands up to 125 Hz from its first source to every listener,
    through wave_responses: the time to its first step, the steps' and the whole run's,
    the least of COST_REPEATS times."""
    room = Room.from_yaml(EXAMPLES / name / "scene.yaml")
    model = TraceModel.from_room(room)
    source = np.array(room.scene.sources[0].position)
    listeners = room.scene.listeners
    made, stepped, fits = [], [], []
    info: dict = {}
    for _ in range(COST_REPEATS):
        # wave_responses says what it runs just before its first time step.
        marks: list[float] = []
        started = time.perf_counter()
        _, info = wave_responses(
            model,
            source,
            np.array([listener.position for listener in listeners]),
            [listener.name for listener in listeners],
            max_band_hz=125,
            length_s=COST_LENGTH_S,
            sample_rate=room.scene.simulation.sample_rate,
            speed_of_sound=room.speed_of_sound(),
            threads=THREADS,
            scheme=scheme,
            walls=walls,
            say=lambda _, marks=marks: marks.append(time.perf_counter()),
        )
        done = time.perf_counter()
        if ("wave_fitted_walls" in info) != (walls == "fitted"):
            raise RuntimeError(f"{name}: wave_responses ran without the walls asked for ({walls})")
        made.append(marks[0] - started)
        stepped.append(done - marks[0])
        if walls == "fitted":
            fits.append(info["wave_fit_s"])
    steps = int(info["wave_steps"])
    return {
        "model": name,
        "scheme": scheme,
        "grid_band_hz": 125,
        "cells": int(info["wave_cells"]),
        "cell_m": info["wave_cell_m"],
        "steps": steps,
        "open": "wave_open" in info,
        "set_up_s": min(made),
        "fit_s": min(fits) if fits else None,
        "step_ms": 1e3 * min(stepped) / steps,
        "run_s": min(m + s for m, s in zip(made, stepped, strict=True)),
        "stencil": int(info["wave_stencil"]),
    }


def cost_part(walls) -> list[dict]:
    """The fit's and the steps' time with each kind of walls: the box, then the
    examples. Walls on cell faces with the 7-point stencil (``scheme="standard"``) and
    with the 19-point one, which the box and a closed example take by default (an open
    example's default, ``scheme="auto"``, makes a closed grid with the 19-point stencil
    first and steps its grid with a matched layer with the 7-point one); fitted walls
    with ``scheme="auto"``, which takes the 7-point stencil for them."""
    rows = []
    for kind, stencil in (("cell_faces", 7), ("cell_faces", 19), ("fitted", 7)):
        if kind in walls:
            row = {"walls": kind, **_box_cost(kind, stencil)}
            rows.append(row)
            _say(f"cost, {_kind(row)}, box: {row['run_s']:.1f} s")
    for name in COST_ROOMS:
        for kind, scheme in (
            ("cell_faces", "standard"),
            ("cell_faces", "auto"),
            ("fitted", "auto"),
        ):
            if kind in walls:
                row = {"walls": kind, **_room_cost(name, kind, scheme)}
                rows.append(row)
                _say(f"cost, {_kind(row)}, {name}, {scheme}: {row['run_s']:.1f} s")
    return rows


def _check(part: str, what: str, value: float, limit: float, least: bool = False) -> dict:
    """One check: ``value`` at most ``limit``, or with ``least`` at least it."""
    passed = bool(value >= limit) if least else bool(value <= limit)
    return {
        "part": part,
        "check": what,
        "value": float(value),
        "limit": float(limit),
        "least": least,
        "passed": passed,
    }


def dispersion_checks(rows: list[dict]) -> list[dict]:
    """Every whole-cell box's modes on the stencil's dispersion relation, and on the
    7-point stencil that relation the same as the one written out in the script."""
    checks = []
    for row in rows:
        if row["kind"] != "faces":
            continue
        label = f"{_kind(row)}, {row['grid_band_hz']} Hz grid"
        checks.append(
            _check(
                "dispersion",
                f"{label}: modes off the dispersion relation",
                row["off_relation"] / 100,
                DISPERSION_LIMIT,
            )
        )
        if "relation_off_closed_form" in row:
            checks.append(
                _check(
                    "dispersion",
                    f"{label}: the relation off its closed form",
                    row["relation_off_closed_form"],
                    CLOSED_FORM_LIMIT,
                )
            )
    return checks


def _turn_limit(yaw: float, pitch: float) -> float:
    along, turned, beyond = TURN_LIMITS_PERCENT
    if yaw == 0.0 and pitch == 0.0:
        return along
    return turned if pitch == 0.0 and yaw <= 14.0 else beyond


def turned_checks(turns: list[dict]) -> list[dict]:
    """With fitted walls every lowest mode of the turned box within its limit; with walls
    on cell faces, on each grid, a mode at least CELL_FACES_LEAST_PERCENT off."""
    checks = []
    for row in turns:
        if row["walls"] != "fitted":
            continue
        worst = max(abs(row["min_percent"]), abs(row["max_percent"]))
        checks.append(
            _check(
                "modes",
                f"fitted walls, {row['grid_band_hz']} Hz grid, turned {row['yaw_deg']:g}/"
                f"{row['pitch_deg']:g} deg: the worst of the {LOWEST} lowest modes (%)",
                worst,
                _turn_limit(row["yaw_deg"], row["pitch_deg"]),
            )
        )
    for band in TURN_BANDS:
        mine = [r for r in turns if r["walls"] == "cell_faces" and r["grid_band_hz"] == band]
        if mine:
            worst = max(max(abs(r["min_percent"]), abs(r["max_percent"])) for r in mine)
            checks.append(
                _check(
                    "modes",
                    f"walls on cell faces, {band} Hz grid: the worst of the {LOWEST} lowest "
                    "modes over the turns (%)",
                    worst,
                    CELL_FACES_LEAST_PERCENT,
                    least=True,
                )
            )
    return checks


def levels_checks(converged: list[dict]) -> list[dict]:
    """The box's exact levels converged."""
    return [
        _check(
            "levels",
            f"{row['stencil']}-point, {row['grid_band_hz']} Hz grid: the exact levels move "
            "with more images (dB)",
            row["moves_db"],
            LEVELS_REFERENCE_LIMIT_DB,
        )
        for row in converged
    ]


def wall_checks(rows: list[dict]) -> list[dict]:
    """The single wall's error, 8 to 16 cells per wavelength: fitted walls along the grid
    and slanted up to WALL_SLANTED_UP_TO_DEG within their limits, walls on cell faces
    0.48 of a cell off at least WALL_CELL_FACES_LEAST_DB."""
    checks = []
    for row in rows:
        what = f"{_kind(row)}, {_wall_label(row)} (dB)"
        error = row["rms_db"][0]
        if row["walls"] == "fitted" and row["angle_deg"] == 0.0:
            checks.append(_check("wall", what, error, WALL_ALONG_LIMIT_DB))
        elif row["walls"] == "fitted" and row["angle_deg"] <= WALL_SLANTED_UP_TO_DEG:
            checks.append(_check("wall", what, error, WALL_SLANTED_LIMIT_DB))
        elif (
            row["walls"] == "cell_faces"
            and row["stencil"] == 7
            and (row["angle_deg"], row["share"]) == WALL_MORE[0]
        ):
            checks.append(_check("wall", what, error, WALL_CELL_FACES_LEAST_DB, least=True))
    return checks


def cost_checks(rows: list[dict]) -> list[dict]:
    """Fitted walls' steps against the 7-point stencil's on cell faces, and the fit."""
    checks = []
    for row in rows:
        if row["walls"] != "fitted":
            continue
        plain = [
            r
            for r in rows
            if r["model"] == row["model"]
            and r["walls"] == "cell_faces"
            and r["scheme"] == "standard"
        ]
        if plain:
            checks.append(
                _check(
                    "cost",
                    f"{row['model']}: fitted walls' time per step over the 7-point stencil's "
                    "on cell faces",
                    row["step_ms"] / plain[0]["step_ms"],
                    COST_STEP_RATIO,
                )
            )
        checks.append(_check("cost", f"{row['model']}: the fit (s)", row["fit_s"], COST_FIT_S))
    return checks


def turn_scan_placements() -> list[np.ndarray]:
    """The SCAN_PLACEMENTS placements of the turned box's scan: u in [0, 1)^3, the grid's
    origin 2 + u cells below the box's lowest corner."""
    rng = np.random.default_rng(SCAN_SEED)
    return [rng.random(3) for _ in range(SCAN_PLACEMENTS)]


def turn_scan(walls) -> list[dict]:
    """The turned box's LOWEST lowest modes with fitted walls on SCAN_BAND's 7-point grid
    at every placement of :func:`turn_scan_placements`, each turn of TURNS: the least and
    most % off at each, and the worst over them."""
    if "fitted" not in walls:
        return []
    placements = turn_scan_placements()
    rows = []
    for yaw, pitch in TURNS:
        started = time.perf_counter()
        found = []
        for u in placements:
            lowest = grid_modes(_turned(yaw, pitch), SCAN_BAND, 7, "fitted", 2.0 + u)["lowest"]
            found.append((lowest["min_percent"], lowest["max_percent"]))
        worst = [max(abs(a), abs(b)) for a, b in found]
        k = int(np.argmax(worst))
        rows.append(
            {
                "walls": "fitted",
                "stencil": 7,
                "grid_band_hz": SCAN_BAND,
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "placements": [u.tolist() for u in placements],
                "min_percent": [a for a, _ in found],
                "max_percent": [b for _, b in found],
                "worst_percent": worst[k],
                "worst_at": placements[k].tolist(),
                "seconds": time.perf_counter() - started,
            }
        )
        _say(f"modes, scan, {yaw:g}/{pitch:g}: {rows[-1]['seconds']:.0f} s")
    return rows


def lroom_part(walls) -> list[dict]:
    """The L-shaped room's lowest mode on the grids of L_BANDS, each kind of walls."""
    rows = []
    for band in L_BANDS:
        for kind in walls:
            for row in l_room(band, kind):
                rows.append(row)
                if row["scan"]:
                    continue
                _say(
                    f"lroom, {_kind(row)}, {band} Hz grid, moved {row['shift_cells']}: "
                    f"{row['seconds']:.0f} s"
                )
    return rows


def absorbing_part(walls) -> dict:
    """The duct's reflection off ends of DUCT_BETAS and off RESONANT, the absorbing box's
    decay and the branch weights in the box with walls of RESONANT, each kind of walls."""
    out: dict = {"duct": [], "duct_branch": [], "decay": [], "branch_weights": []}
    for kind in walls:
        for beta in DUCT_BETAS:
            for e in DUCT_ENDS:
                out["duct"].append(duct_reflection(beta, e, kind))
        _say(f"absorbing, duct, {kind}")
        reference: dict = {}
        for e in DUCT_BRANCH_ENDS:
            out["duct_branch"].append(duct_branch_reflection(e, kind, reference))
        _say(f"absorbing, duct with a branch, {kind}")
        for band in DECAY_BANDS:
            out["decay"].append(box_decay(band, kind))
            out["branch_weights"].append(branch_weights(band, kind))
            _say(f"absorbing, decay and branch weights, {kind}, {band} Hz grid")
    return out


#: The long runs: (model, case).
LONG_RUNS = (
    ("lroom", "rigid"),
    ("lroom", "strong"),
    ("lroom", "absorbing"),
    ("lroom", "open"),
    ("box", "rigid"),
)


def long_part(walls) -> list[dict]:
    """The long runs of LONG_RUNS, with fitted walls."""
    rows = []
    if "fitted" not in walls:
        return rows
    for name, case in LONG_RUNS:
        row = long_run(name, case)
        rows.append(row)
        _say(f"long, {name}, {case}: {row['run_s']:.0f} s")
    return rows


def scan_checks(scans: list[dict]) -> list[dict]:
    """The turned box's worst lowest mode over the scan's placements within its limit."""
    return [
        _check(
            "modes",
            f"fitted walls, {row['grid_band_hz']} Hz grid, turned {row['yaw_deg']:g}/"
            f"{row['pitch_deg']:g} deg: the worst of the {LOWEST} lowest modes over "
            f"{len(row['placements'])} placements (%)",
            row["worst_percent"],
            _turn_limit(row["yaw_deg"], row["pitch_deg"]),
        )
        for row in scans
    ]


def moved_checks(rows: list[dict]) -> list[dict]:
    """With fitted walls, the box moved along the grid by parts of a cell: its LOWEST
    lowest modes within the limit along the grid."""
    checks = []
    for row in rows:
        if row["walls"] != "fitted" or row["kind"] != "shift" or "lowest" not in row:
            continue
        lowest = row["lowest"]
        checks.append(
            _check(
                "modes",
                f"fitted walls, {row['grid_band_hz']} Hz grid, {row['case']}: the worst of the "
                f"{LOWEST} lowest modes (%)",
                max(abs(lowest["min_percent"]), abs(lowest["max_percent"])),
                TURN_LIMITS_PERCENT[0],
            )
        )
    return checks


def box_levels_checks(rows: list[dict]) -> list[dict]:
    """With fitted walls, on each grid, every band level in the box within LEVELS_LIMIT_DB
    of the exact one (each band and window, the most over the listeners and placements);
    with walls on cell faces (7-point), the most over them on each grid at least that."""
    checks = []
    for walls, stencil in (("fitted", 7), ("cell_faces", 7)):
        for band in BANDS:
            mine = [
                r
                for r in rows
                if r["walls"] == walls and r["stencil"] == stencil and r["grid_band_hz"] == band
            ]
            if not mine:
                continue
            found = {}
            for window in WINDOWS_S:
                key = f"{1000 * window:.0f}_ms"
                for b in BANDS[: BANDS.index(band) + 1]:
                    found[(key, b)] = max(float(np.abs(r[key][b]).max()) for r in mine)
            if walls == "fitted":
                for (key, b), value in found.items():
                    checks.append(
                        _check(
                            "levels",
                            f"fitted walls, {band} Hz grid, {b} Hz band, first "
                            f"{key.replace('_', ' ')}: the most any level is off (dB)",
                            value,
                            LEVELS_LIMIT_DB,
                        )
                    )
            else:
                checks.append(
                    _check(
                        "levels",
                        f"walls on cell faces, 7-point, {band} Hz grid: the most any level is "
                        "off (dB)",
                        max(found.values()),
                        LEVELS_LIMIT_DB,
                        least=True,
                    )
                )
    return checks


def lroom_checks(rows: list[dict]) -> list[dict]:
    """With walls on cell faces, the L-shaped room's lowest mode the plan's own at the
    grid's cell; at each placement, fitted walls' mode no further from the continuum's
    than the cell-faces mode, on either side of it; over the scan, fitted walls on the cell
    faces giving the cell faces' mode, and the least and the most within L_SCAN_RANGE."""
    checks = []
    for row in rows:
        if row["walls"] == "cell_faces":
            checks.append(
                _check(
                    "lroom",
                    f"walls on cell faces, {row['grid_band_hz']} Hz grid, moved "
                    f"{_cells(row['shift_cells'])}: the lowest mode off the plan's own at the "
                    "grid's cell",
                    abs(row["lowest_hz"] / row["plan_on_cell_hz"] - 1),
                    L_FACES_LIMIT,
                )
            )
    for row in rows:
        if row["walls"] != "fitted" or row["scan"]:
            continue
        faces = [
            r
            for r in rows
            if r["walls"] == "cell_faces"
            and r["grid_band_hz"] == row["grid_band_hz"]
            and r["shift_cells"] == row["shift_cells"]
        ]
        if not faces:
            continue
        continuum = row["continuum_hz"]
        checks.append(
            _check(
                "lroom",
                f"{row['grid_band_hz']} Hz grid, moved {_cells(row['shift_cells'])}: fitted "
                "walls' lowest mode off the continuum's over the cell faces' (each as a size)",
                abs(row["lowest_hz"] / continuum - 1) / abs(faces[0]["lowest_hz"] / continuum - 1),
                1.0,
            )
        )
    for band in L_BANDS:
        scan = [
            r for r in rows if r["walls"] == "fitted" and r["scan"] and r["grid_band_hz"] == band
        ]
        faces = [r for r in rows if r["walls"] == "cell_faces" and r["grid_band_hz"] == band]
        if not scan or not faces:
            continue
        at_zero = [r for r in scan if r["shift_cells"] == [0.0, 0.0, 0.0]]
        checks.append(
            _check(
                "lroom",
                f"{band} Hz grid, fitted walls on the cell faces (moved 0 cells): the lowest mode "
                "off the cell faces' (relative)",
                abs(at_zero[0]["lowest_hz"] / faces[0]["lowest_hz"] - 1),
                L_FACES_LIMIT,
            )
        )
    for band in L_BANDS:
        scan = [
            r for r in rows if r["walls"] == "fitted" and r["scan"] and r["grid_band_hz"] == band
        ]
        if not scan:
            continue
        off = [100 * (r["lowest_hz"] / r["continuum_hz"] - 1) for r in scan]
        low, high = L_SCAN_RANGE[band]
        checks.append(
            _check(
                "lroom",
                f"{band} Hz grid, fitted walls over the scan's {len(scan)} placements: the "
                "lowest mode off the continuum's, the least (%)",
                min(off),
                low,
                least=True,
            )
        )
        checks.append(
            _check(
                "lroom",
                f"{band} Hz grid, fitted walls over the scan's {len(scan)} placements: the "
                "lowest mode off the continuum's, the most (%)",
                max(off),
                high,
            )
        )
    return checks


def absorbing_checks(found: dict) -> list[dict]:
    """With fitted walls, the duct's reflection off each end and off the branch's end, the
    admittance written out, the box's decay and its branch weights, each within its
    limit."""
    checks = []
    for row in found["duct"]:
        if row["walls"] != "fitted":
            continue
        checks.append(
            _check(
                "absorbing",
                f"fitted walls, the duct's end of admittance {row['beta']:g}, "
                f"{row['end_cells']:g} cell past a face: its reflection off (1 - b) / (1 + b)",
                abs(row["off"]),
                DUCT_MATCHED_LIMIT if row["beta"] == 1.0 else DUCT_LIMIT,
            )
        )
    for row in found["duct_branch"]:
        if row is found["duct_branch"][0]:
            checks.append(
                _check(
                    "absorbing",
                    "the branch's admittance written out, off Admittance's (a fraction)",
                    row["admittance_off_written"],
                    ADMITTANCE_FORM_LIMIT,
                )
            )
        if row["walls"] != "fitted":
            continue
        checks.append(
            _check(
                "absorbing",
                f"fitted walls, the duct's end with a branch, {row['end_cells']:g} cell past a "
                "face: its reflection off the grid's theory",
                row["off_grid_theory"],
                DUCT_BRANCH_FAR_LIMIT if row["end_cells"] == -0.48 else DUCT_BRANCH_LIMIT,
            )
        )
    for row in found["decay"]:
        if row["walls"] != "fitted":
            continue
        worst = max(abs(m["ratio"] - 1) for m in row["modes"])
        checks.append(
            _check(
                "absorbing",
                f"fitted walls, {row['grid_band_hz']} Hz grid, the box of two materials: the "
                f"worst of its {DECAY_MODES} lowest modes' decay off modal theory's (a fraction)",
                worst,
                DECAY_LIMIT,
            )
        )
    for row in found.get("branch_weights", []):
        if row["walls"] != "fitted":
            continue
        checks.append(
            _check(
                "absorbing",
                f"fitted walls, {row['grid_band_hz']} Hz grid, the box with walls of a branch "
                "material: its branch states' weights times their cells' volumes, summed, off "
                "L/2 times the walls' area (a fraction)",
                abs(row["off"]),
                BRANCH_WEIGHTS_LIMIT,
            )
        )
    return checks


def long_checks(rows: list[dict]) -> list[dict]:
    """Each long run's checks: a rigid room's last block within LONG_KEEPS of its second and none
    above LONG_MOST times it; an absorbing or open room's second half below LONG_FLOOR of its
    first block and not growing again; the rigid update's least coefficient, its rows' largest
    sum and its largest eigenvalue where the run has them."""
    checks = []
    for row in rows:
        label = f"fitted walls, {row['model']}, {row['case']}"
        full, osc = np.array(row["full"]), np.array(row["oscillating"])
        checks.append(
            _check("long", f"{label}: every reading finite", float(row["finite"]), 1.0, least=True)
        )
        if row["case"] == "rigid":
            low, high = LONG_KEEPS
            ratio = float(osc[-1] / osc[1])
            checks.append(
                _check("long", f"{label}: the last block over the second", ratio, low, least=True)
            )
            checks.append(_check("long", f"{label}: the last block over the second", ratio, high))
            checks.append(
                _check(
                    "long",
                    f"{label}: the largest block over the second",
                    float(osc[1:].max() / osc[1]),
                    LONG_MOST,
                )
            )
        else:
            series = full if row["case"] == "open" else osc
            checks.append(
                _check(
                    "long",
                    f"{label}: the second half's largest block over the first block's whole rms",
                    float(series[len(series) // 2 :].max() / full[0]),
                    LONG_FLOOR,
                )
            )
            checks.append(
                _check(
                    "long",
                    f"{label}: the second half's largest block over its allowance",
                    float(
                        series[len(series) // 2 :].max()
                        / max(LONG_REGROW * series[5:10].max(), LONG_DUST * full[0])
                    ),
                    1.0,
                )
            )
        if "row_sum" in row:
            checks.append(
                _check(
                    "long",
                    f"{label}: the update's least coefficient",
                    row["least_coefficient"],
                    0.0,
                    least=True,
                )
            )
            checks.append(
                _check(
                    "long",
                    f"{label}: the largest sum of a row of the update, less {LONG_ROW_SUM:g}",
                    row["row_sum"] - LONG_ROW_SUM,
                    0.0,
                )
            )
        if "largest_mu" in row:
            checks.append(
                _check(
                    "long",
                    f"{label}: the update's largest eigenvalue",
                    row["largest_mu"],
                    LONG_BOUND,
                )
            )
    return checks


def _cells(shift) -> str:
    return ", ".join(f"{v:g}" for v in shift) + " cells"


def print_scan(scans: list[dict], turns: list[dict]) -> None:
    if not scans:
        return
    print(
        f"\nThe turned box with fitted walls on the {SCAN_BAND} Hz band's grid at "
        f"{len(scans[0]['placements'])} more placements (the grid's origin 2 + u cells below "
        f"the box, u uniform in [0, 1)^3, seed {SCAN_SEED}): the worst of its {LOWEST} lowest"
    )
    print("modes, % off the exact ones with the dispersion, beside wave_responses' placement")
    print(
        "\n| turned (deg) | wave_responses' placement | least over the placements | most | at u |"
    )
    print("|---|---:|---:|---:|---|")
    for row in scans:
        own = [
            r
            for r in turns
            if r["walls"] == "fitted"
            and r["grid_band_hz"] == row["grid_band_hz"]
            and (r["yaw_deg"], r["pitch_deg"]) == (row["yaw_deg"], row["pitch_deg"])
        ]
        here = f"{max(abs(own[0]['min_percent']), abs(own[0]['max_percent'])):.3f}" if own else ""
        worst = [
            max(abs(a), abs(b)) for a, b in zip(row["min_percent"], row["max_percent"], strict=True)
        ]
        print(
            f"| {row['yaw_deg']:g} / {row['pitch_deg']:g} | {here} | {min(worst):.3f} | "
            f"{row['worst_percent']:.3f} | {', '.join(f'{x:.4f}' for x in row['worst_at'])} |"
        )


def print_moved(rows: list[dict]) -> None:
    moved = [r for r in rows if r["kind"] == "shift" and "lowest" in r]
    if not moved:
        return
    print(
        f"\nThe box moved along the grid, its {LOWEST} lowest modes, % off the exact ones as "
        "the 7-point stencil's dispersion moves them: least ... most"
    )
    print("\n| stencil, walls | grid (cell) | placement | least ... most (rms) |")
    print("|---|---|---|---:|")
    for r in moved:
        low = r["lowest"]
        print(
            f"| {_kind(r)} | {r['grid_band_hz']} Hz ({100 * r['cell_m']:.2f} cm) | {r['case']} | "
            f"{low['min_percent']:+.3f} ... {low['max_percent']:+.3f} ({low['rms_percent']:.3f}) |"
        )


def print_lroom(rows: list[dict]) -> None:
    print(
        "\nThe L-shaped room's lowest mode (its sides and notch rounded to whole cells), Hz: on "
        "the grid with each kind of walls, the plan's own"
    )
    print(
        "5-point problem at the grid's cell (what walls on cell faces must give) and the "
        "continuum's (extrapolated, then through the grid's time step), % off it"
    )
    print(
        "\n| grid (cell) | room (m) | moved | fitted walls | walls on cell faces | plan at the "
        "cell | continuum | continuum without the time step | extrapolated from (power) |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---|")
    for band in L_BANDS:
        for shift in L_PLACEMENTS:
            mine = {
                r["walls"]: r
                for r in rows
                if r["grid_band_hz"] == band and r["shift_cells"] == list(shift)
            }
            if not mine:
                continue
            any_row = next(iter(mine.values()))
            cont = any_row["continuum_hz"]

            def cell(walls: str, cont: float = cont, mine: dict = mine) -> str:
                if walls not in mine:
                    return ""
                f = mine[walls]["lowest_hz"]
                return f"{f:.4f} ({100 * (f / cont - 1):+.3f} %)"

            levels = any_row["extrapolation"]
            print(
                f"| {band} Hz ({100 * any_row['cell_m']:.2f} cm) | "
                + " x ".join(f"{x:.3f}" for x in any_row["size_m"])
                + f" | {_cells(shift)} | {cell('fitted')} | {cell('cell_faces')} | "
                f"{any_row['plan_on_cell_hz']:.4f} | {cont:.4f} | "
                f"{any_row['continuum_free_hz']:.4f} | "
                + ", ".join(f"{x:.4f}" for x in levels["levels_hz"])
                + f" at {', '.join(map(str, levels['parts']))} parts ({levels['power']:.2f}) |"
            )
    scans = [r for r in rows if r["walls"] == "fitted" and r["scan"]]
    if not scans:
        return
    print(
        f"\nFitted walls, the room moved by s cells along each axis (s = k / {L_SCAN_STEPS}, and "
        f"0.5 -+ {L_SCAN_CROSS:g}, either side of the planes of the cells' centres), % off the "
        "continuum"
    )
    print(
        "\n| grid (cell) | least (at s) | most (at s) | from s = "
        f"{0.5 - L_SCAN_CROSS:g} to {0.5 + L_SCAN_CROSS:g} | walls on cell faces |"
    )
    print("|---|---:|---:|---:|---:|")
    for band in L_BANDS:
        mine = [r for r in scans if r["grid_band_hz"] == band]
        if not mine:
            continue
        cont = mine[0]["continuum_hz"]

        def off(r: dict, cont: float = cont) -> float:
            return 100 * (r["lowest_hz"] / cont - 1)

        least = min(mine, key=off)
        most = max(mine, key=off)
        side = {round(r["shift_cells"][0], 9): r for r in mine}
        jump = off(side[round(0.5 + L_SCAN_CROSS, 9)]) - off(side[round(0.5 - L_SCAN_CROSS, 9)])
        faces = [r for r in rows if r["walls"] == "cell_faces" and r["grid_band_hz"] == band]
        print(
            f"| {band} Hz ({100 * mine[0]['cell_m']:.2f} cm) | {off(least):+.3f} % "
            f"({least['shift_cells'][0]:g}) | {off(most):+.3f} % ({most['shift_cells'][0]:g}) | "
            f"{jump:+.3f} % | " + (f"{off(faces[0]):+.3f} %" if faces else "") + " |"
        )


def print_absorbing(found: dict) -> None:
    print(
        f"\nThe duct ({DUCT_CELL * 100:g} cm cells): its far end's reflection, "
        f"{DUCT_BAND[0]:g}-{DUCT_BAND[1]:g} Hz, off (1 - beta) / (1 + beta)"
    )
    print(
        "\n| walls | beta | " + " | ".join(f"end {e:g} cell past a face" for e in DUCT_ENDS) + " |"
    )
    print("|---|---|" + "---:|" * len(DUCT_ENDS))
    for walls in WALLS:
        for beta in DUCT_BETAS:
            mine = {
                r["end_cells"]: r
                for r in found["duct"]
                if r["walls"] == walls and r["beta"] == beta
            }
            if mine:
                print(
                    f"| {walls} | {beta:g} | "
                    + " | ".join(
                        f"{mine[e]['found']:.4f} ({mine[e]['off']:+.4f})" for e in DUCT_ENDS
                    )
                    + " |"
                )
    print(
        f"\nThe duct's end with a branch resonant at 100 Hz, its reflection over a rigid end's, "
        f"{DUCT_BRANCH_BAND[0]:g}-{DUCT_BRANCH_BAND[1]:g} Hz: the most off the grid's theory "
        "(and off the continuum's; the theory's own off it)"
    )
    print(
        "\n| walls | end (cells past a face) | off the grid's theory | off the continuum "
        "| theory off the continuum |"
    )
    print("|---|---:|---:|---:|---:|")
    for row in found["duct_branch"]:
        print(
            f"| {row['walls']} | {row['end_cells']:g} | {row['off_grid_theory']:.4f} | "
            f"{row['off_continuum']:.4f} | {row['theory_off_continuum']:.4f} |"
        )
    print(
        f"\nThe box turned {DECAY_TURN:g} deg, floor and ceiling of admittance {DECAY_BETAS[0]:g}, "
        f"its walls {DECAY_BETAS[1]:g}: each mode's first-order decay over modal theory's"
    )
    print(
        "\n| walls | grid (cell) | "
        + " | ".join(f"mode {k + 1}" for k in range(DECAY_MODES))
        + " |"
    )
    print("|---|---|" + "---:|" * DECAY_MODES)
    for row in found["decay"]:
        print(
            f"| {row['walls']} | {row['grid_band_hz']} Hz ({100 * row['cell_m']:.2f} cm) | "
            + " | ".join(f"{''.join(map(str, m['mode']))}: {m['ratio']:.3f}" for m in row["modes"])
            + " |"
        )
    rows = found.get("branch_weights", [])
    if rows:
        print(
            f"\nThe box turned {DECAY_TURN:g} deg, floor and ceiling rigid, its walls with a "
            "branch resonant at 100 Hz: the branch states' weights times their cells' volumes, "
            "summed, against L/2 times the walls' area in cells squared"
        )
        print("\n| walls | grid (cell) | states | summed | L/2 times the area | off |")
        print("|---|---|---:|---:|---:|---:|")
        for row in rows:
            print(
                f"| {row['walls']} | {row['grid_band_hz']} Hz ({100 * row['cell_m']:.2f} cm) | "
                f"{row['states']:,} | {row['found']:.7f} | {row['expected']:.7f} | "
                f"{row['off']:+.2e} |"
            )


def print_long(rows: list[dict]) -> None:
    if not rows:
        return
    print(
        f"\nLong runs with fitted walls, {LONG_STEPS:,} steps: the rms over {LONG_POINTS} points "
        f"by blocks of {LONG_BLOCK:,} steps (the oscillating part: each point's mean over the"
    )
    print(
        "block taken out); the uniform pressure's Rayleigh quotient; the rigid update's least "
        "coefficient, its rows' largest sum less 2 and its largest eigenvalue; the clamp's counts"
    )
    print(
        "\n| model | case | cell (cm) | air cells | special | raised (most) | first block | "
        "second | last | last over second | uniform mu | least coefficient | largest row "
        "less 2 | largest mu | run (s) |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        series = r["full"] if r["case"] == "open" else r["oscillating"]
        mu = f"{r['uniform_mu']:.2e}" if "uniform_mu" in r else ""
        least = f"{r['least_coefficient']:.3e}" if "least_coefficient" in r else ""
        row_sum = f"{r['row_sum'] - 2.0:.1e}" if "row_sum" in r else ""
        largest = f"{r['largest_mu']:.6f}" if "largest_mu" in r else ""
        print(
            f"| {r['model']} | {r['case']} | {100 * r['cell_m']:.2f} | {r['air_cells']:,} | "
            f"{r['special_cells']:,} | {r['raised']} ({r['most_raised']:.3f}) | "
            f"{r['full'][0]:.3e} | {series[1]:.3e} | {series[-1]:.3e} | "
            f"{series[-1] / series[1]:.4f} | {mu} | {least} | {row_sum} | {largest} | "
            f"{r['run_s']:.0f} |"
        )


def dropped_checks(walls, parts) -> list[str]:
    """The checks the chosen walls leave out."""
    dropped = []
    if "cell_faces" not in walls:
        if "modes" in parts:
            dropped.append(
                "modes: the controls on cell faces (a mode at least 1 % off on each grid)"
            )
        if "levels" in parts:
            dropped.append("levels: the control on cell faces (a level at least the limit off)")
        if "wall" in parts:
            dropped.append("wall: the control on cell faces (0.48 of a cell off at least 1 dB)")
        if "lroom" in parts:
            dropped.append(
                "lroom: walls on cell faces against the plan's own problem, and fitted walls "
                "against them"
            )
        if "cost" in parts:
            dropped.append("cost: fitted walls' time per step over the plain run's")
    if "fitted" not in walls:
        dropped.append("every check of fitted walls (the long runs run fitted walls only)")
    return dropped


def _octaves() -> str:
    return " | ".join(f"{lo:g}-{hi:g}" for lo, hi in PER_WAVELENGTH_OCTAVES)


def print_dispersion(rows: list[dict]) -> None:
    print(
        "\nSides in whole cells, walls on cell faces, against the stencil's dispersion "
        "relation: the most any mode is off"
    )
    for row in rows:
        if row["kind"] == "faces":
            print(
                f"- {_kind(row)}, {row['grid_band_hz']} Hz grid: "
                f"{row['off_relation']:.1e} % "
                f"({row['air_cells']:,} air cells; update symmetric within {row['asymmetry']:.0e})"
            )


def print_modes(rows: list[dict], turns: list[dict]) -> None:
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
            f"| {_kind(row)} | {row['grid_band_hz']} Hz ({100 * row['cell_m']:.2f} cm) "
            f"| {row['case']} | " + " | ".join(cells) + " |"
        )
    print_dispersion(rows)
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
    for label, band, mine in _own_bands(rows):
        cells = []
        for kind in kinds:
            picked = [r["bands"][band] for r in mine if r["kind"] in kind]
            worst = max(picked, key=lambda b: abs(b["worst_percent"]))["worst_percent"]
            rms = max(b["rms_percent"] for b in picked)
            cells.append(f"{worst:+.1f} % ({rms:.1f} %)")
        print(
            f"| {label} | {band} Hz ({100 * mine[0]['cell_m']:.1f} cm) | "
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
    for label, band, mine in _own_bands(rows):
        for kind, name in zip(kinds[1:], ("moved within a cell", "turned or tilted"), strict=True):
            picked = [r["bands"][band] for r in mine if r["kind"] in kind]
            mixed = [b["mixed"] for b in picked]
            kept = max(picked, key=lambda b: abs(b["worst_kept_percent"]))["worst_kept_percent"]
            furthest = max(picked, key=lambda b: abs(b["furthest_percent"]))["furthest_percent"]
            print(
                f"| {label} | {band} Hz ({100 * mine[0]['cell_m']:.1f} cm) | {name} | "
                f"{min(mixed)} ... {max(mixed)} of {picked[0]['modes']} | {kept:+.1f} % "
                f"| {furthest:+.1f} % |"
            )
    if turns:
        print_turns(turns)


def print_turns(turns: list[dict]) -> None:
    print(
        f"\nThe box turned (about the vertical, then about x), its {LOWEST} lowest modes, % "
        "off the exact ones as the 7-point stencil's dispersion moves them:"
    )
    print("least ... most (rms), and the least correlation of a mode with its eigenvector")
    names = {"cell_faces": "walls on cell faces", "fitted": "fitted walls"}
    present = [w for w in WALLS if any(r["walls"] == w for r in turns)]
    print(
        "\n| grid (cell) | turned (deg) | "
        + " | ".join(f"{names[w]} | correlation" for w in present)
        + " |"
    )
    print("|---|---|" + "---:|---:|" * len(present))
    for band in TURN_BANDS:
        for yaw, pitch in TURNS:
            mine = {
                r["walls"]: r
                for r in turns
                if r["grid_band_hz"] == band and (r["yaw_deg"], r["pitch_deg"]) == (yaw, pitch)
            }
            if not mine:
                continue
            cells = []
            for w in present:
                r = mine[w]
                cells.append(
                    f"{r['min_percent']:+.3f} ... {r['max_percent']:+.3f} ({r['rms_percent']:.3f})"
                )
                cells.append(f"{r['least_correlation']:.3f}")
            cell = next(iter(mine.values()))["cell_m"]
            print(
                f"| {band} Hz ({100 * cell:.2f} cm) | {yaw:g} / {pitch:g} | "
                + " | ".join(cells)
                + " |"
            )


def _own_bands(rows: list[dict]):
    """Each stencil and kind of walls, and grid, that has rows, with them."""
    for walls in WALLS:
        for stencil in _stencils(walls):
            for band in BANDS:
                mine = [
                    r
                    for r in rows
                    if r["walls"] == walls and r["stencil"] == stencil and r["grid_band_hz"] == band
                ]
                if mine:
                    yield _kind(mine[0]), band, mine


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
                f"| {_kind(row)} | {row['grid_band_hz']} Hz "
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
    for label, band, mine in _own_bands(rows):
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
            f"| {label} | {band} Hz ({100 * mine[0]['cell_m']:.1f} cm) | "
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
        "| best place (cells) |"
    )
    print("|---|---|" + "---:|" * (2 * len(PER_WAVELENGTH_OCTAVES) + 1))
    for row in rows:
        face = row.get("rms_on_face_db")
        cells = [f"{x:.2f}" for x in row["rms_db"]]
        cells += [f"{x:.3f}" for x in face] if face else [""] * len(PER_WAVELENGTH_OCTAVES)
        print(
            f"| {_kind(row)} | {_wall_label(row)} | "
            + " | ".join(cells)
            + f" | {row['best_place_cells']:+.2f} |"
        )


def print_cost(rows: list[dict]) -> None:
    try:
        load = ", ".join(f"{x:.1f}" for x in os.getloadavg())
    except OSError:
        load = "not known"
    print(
        f"\nThe time it takes: {platform.machine()}, {platform.system()}, {os.cpu_count()} "
        f"threads, {THREADS} used; load {load} at the end; {COST_BOX_LENGTH_S:g} s of sound "
        f"in the box, {COST_LENGTH_S:g} s in the examples, the least of {COST_REPEATS} runs"
    )
    print(
        "\n| model | stencil | scheme | grid (cell) | air cells | steps | to the first step (s) "
        "| fit (s) | per step (ms) | whole run (s) | per step, over 7-point on cell faces |"
    )
    print("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        plain = [
            r
            for r in rows
            if r["model"] == row["model"]
            and r["walls"] == "cell_faces"
            and r["scheme"] == "standard"
        ]
        ratio = f"{row['step_ms'] / plain[0]['step_ms']:.2f}" if plain else ""
        fit = f"{row['fit_s']:.2f}" if row["fit_s"] is not None else ""
        model = row["model"] + (" (open)" if row.get("open") else "")
        print(
            f"| {model} | {_kind(row)} | {row['scheme']} "
            f"| {row['grid_band_hz']} Hz ({100 * row['cell_m']:.1f} cm) "
            f"| {row['cells']:,} | {row['steps']:,} | {row['set_up_s']:.2f} | {fit} "
            f"| {row['step_ms']:.3f} | {row['run_s']:.2f} | {ratio} |"
        )


def print_checks(checks: list[dict]) -> None:
    print("\nChecks: what each found, and its limit")
    print("\n| part | check | found | limit | |")
    print("|---|---|---:|---:|---|")
    for c in checks:
        limit = ("at least " if c["least"] else "at most ") + f"{c['limit']:.4g}"
        print(
            f"| {c['part']} | {c['check']} | {c['value']:.3g} | {limit} | "
            + ("ok" if c["passed"] else "FAILED")
            + " |"
        )


PARTS = ("dispersion", "modes", "levels", "wall", "lroom", "absorbing", "long", "cost")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", default=list(PARTS), choices=PARTS)
    parser.add_argument("--bands", nargs="+", type=int, default=list(BANDS), choices=BANDS)
    parser.add_argument("--walls", nargs="+", default=list(WALLS), choices=WALLS)
    parser.add_argument("--json", help="write the numbers to this file")
    args = parser.parse_args()
    bands = sorted(args.bands)
    walls = [w for w in WALLS if w in args.walls]
    print(
        f"c = {C:.4f} m/s, {FS:g} Hz, {PER_WAVELENGTH:g} cells per wavelength at the top of "
        "the highest band"
    )
    for stencil in sorted({s for w in walls for s in _stencils(w)}, reverse=True):
        for band in bands:
            g = grid(band, stencil)
            print(
                f"- {stencil}-point, grid for the {band} Hz band: {100 * g.cell:.3f} cm cells, "
                f"{g.rate:.2f} steps/s"
            )
    print(
        "Walls: "
        + " and ".join({"cell_faces": "on cell faces", "fitted": "fitted"}[w] for w in walls)
    )
    results: dict = {"walls": walls}
    checks: list[dict] = []
    if "modes" in args.parts or "dispersion" in args.parts:
        started = time.perf_counter()
        whole = "modes" in args.parts
        results["modes"], results["turned"] = modes_part(
            bands, walls, CASES if whole else CASES[:1], turned=whole
        )
        if whole:
            results["turn_scan"] = turn_scan(walls) if SCAN_BAND in bands else []
            print_modes(results["modes"], results["turned"])
            print_scan(results["turn_scan"], results["turned"])
            print_moved(results["modes"])
            checks += turned_checks(results["turned"])
            checks += scan_checks(results["turn_scan"])
            checks += moved_checks(results["modes"])
        else:
            print_dispersion(results["modes"])
        checks = dispersion_checks(results["modes"]) + checks
        print(f"\n({'modes' if whole else 'dispersion'}: {time.perf_counter() - started:.0f} s)")
    if "levels" in args.parts:
        started = time.perf_counter()
        results["levels"], results["levels_converged"] = levels_part(bands, walls)
        print_levels(results["levels"], results["levels_converged"])
        checks += levels_checks(results["levels_converged"])
        checks += box_levels_checks(results["levels"])
        print(f"\n(levels: {time.perf_counter() - started:.0f} s)")
    if "wall" in args.parts:
        started = time.perf_counter()
        results["wall"] = wall_part(walls)
        print_wall(results["wall"])
        checks += wall_checks(results["wall"])
        print(f"\n(wall: {time.perf_counter() - started:.0f} s)")
    if "lroom" in args.parts:
        started = time.perf_counter()
        results["lroom"] = lroom_part(walls)
        print_lroom(results["lroom"])
        checks += lroom_checks(results["lroom"])
        print(f"\n(lroom: {time.perf_counter() - started:.0f} s)")
    if "absorbing" in args.parts:
        started = time.perf_counter()
        results["absorbing"] = absorbing_part(walls)
        print_absorbing(results["absorbing"])
        checks += absorbing_checks(results["absorbing"])
        print(f"\n(absorbing: {time.perf_counter() - started:.0f} s)")
    if "long" in args.parts:
        started = time.perf_counter()
        results["long"] = long_part(walls)
        print_long(results["long"])
        checks += long_checks(results["long"])
        print(f"\n(long: {time.perf_counter() - started:.0f} s)")
    if "cost" in args.parts:
        started = time.perf_counter()
        results["cost"] = cost_part(walls)
        print_cost(results["cost"])
        checks += cost_checks(results["cost"])
        print(f"\n(cost: {time.perf_counter() - started:.0f} s)")
    results["checks"] = checks
    print_checks(checks)
    results["not_checked"] = dropped_checks(walls, args.parts)
    if results["not_checked"]:
        print("\nNot checked with the walls chosen:")
        for line in results["not_checked"]:
            print(f"- {line}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=float)
    ran = {
        r["walls"]
        for key in ("modes", "levels", "wall", "lroom", "cost")
        for r in results.get(key, [])
    }
    ran |= {r["walls"] for rows in results.get("absorbing", {}).values() for r in rows}
    if ran and ran != set(walls):
        raise SystemExit(f"the parts ran with walls {sorted(ran)}, not the ones asked for {walls}")
    if any(r["walls"] not in walls for r in results.get("long", [])):
        raise SystemExit(f"the long runs ran with walls other than the ones asked for {walls}")
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        raise SystemExit(f"{len(failed)} of {len(checks)} checks failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
