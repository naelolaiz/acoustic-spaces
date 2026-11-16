"""Sound over a row of barriers and buildings whose tops it just clears, in still air.

Where the sound from the source to the listener passes close over several edges in a
row (the roofs along a street, a roof and the parapet behind it, screen after screen),
each edge stands in the others' transition zones: every choice of the edges to bend
over gives a term about as large as the direct sound, and paths over one edge or two
(:mod:`~acoustic_spaces.engine.diffraction`) do not add up to the field, whatever the
number of edges they stop at.

There the field is marched over the whole row at once, in the vertical plane through the
source and the listener, with the parabolic equation
(:func:`~acoustic_spaces.engine.parabolic.march`), which holds the ground and the roofs
as they are and takes each barrier and each wall of a building as a screen with no field
below its top. The march is of a field in two dimensions, u; over edges that cross the
way, a point source's sound is u over the root of the length of its path, the string
pulled taut over the tops from the source to the listener (:func:`string`). A rigid
edge sends on more than the march's screen does. What it adds is the wave the edge
sheds by the uniform theory of diffraction
(:func:`~acoustic_spaces.engine.diffraction.wedge_terms`), less what the march's screen
sheds, lit by the marched field that reaches the edge and weakened on its way on as
that field is on its way there from the other end:

    p = p_march + sum over the edges of  u_S(e) u_L(e) D_e / sqrt(a + b)

with u_S(e) the field marched from the source to the edge over what stands before it,
u_L(e) the field marched from the listener to the edge over what stands after it (by
reciprocity, how the edge's wave reaches the listener), a and b the lengths of the
string from the edge to the source and to the listener, and D_e, at the distance
parameter a b / (a + b) and for the string's directions at the edge:

- for a barrier's top, the two terms of the reflections off the faces of a half-plane,
  and 1 - cos(bend / 2) of its two terms of the incident wave. The march's screen, the
  field kept above the top and none below, sheds cos(out) / cos((in + out) / 2) of what
  a half-plane without a reflection does, for a wave that comes in and goes out at
  those angles from the march's way; the mean of the marches from both ends sheds
  cos(bend / 2) of it;
- for a roof's corner, along whose top the wave runs, the two terms of the incident wave
  of the right-angled wedge, seen from the source (the near corner) or the listener (the
  far one), less cos(bend / 2) of those of the half-plane that the wall is to the
  march. A corner that an end over its roof lights itself takes the end's own wave and
  that of the end's image in the roof apart: the wedge's two terms of the incident
  wave for the one and its two of the reflection off the roof for the other, each less
  cos(bend / 2) of the half-plane's two of the incident wave, for the angle between
  the wave's way to the corner and its way on.

A barrier that stands alone between the ends on a rigid ground, or on a rigid roof
under both, is not a half-plane: with its image in that surface it is a rigid strip in
free space, and sound goes on between its top and the surface. There the barrier adds
what that strip sheds beyond the marches' screen, from the strip's own equation
(:func:`~acoustic_spaces.engine.strip.beyond_screen`), in place of the half-plane's
terms, at the frequencies up to k H = ``strip.MOST_KH`` for its height H at which
that surface is rigid (:func:`_plane_under` finds the surface, :func:`_rigid` the
frequencies). Not where the ends are so close to each other across a high barrier,
or both so close to it, that the strip's density would take more than
``strip.TERMS_MOST`` terms (for a way across the barrier: closer to each other than
about a quarter of its height, both within about a twentieth of it from the barrier, or
between the two; at a slant further off, up to 0.83 of its height from it at 75 degrees):
the half-plane's terms stand there at every frequency, up to 0.27 dB from the strip's
in an octave band's level across the barrier (docs/validation.md, "Rows of roofs and
barriers against a wave calculation").

A way that crosses the row at a slant, seen from above, meets the obstacles' edges
turned from across it (``slants``). The marches still run in the vertical plane along
the way, over the row as that plane cuts it: for a row that is the same all along its
edges, the narrow-angle march along the way is the one across the edges at the
wavenumber times the slant's cosine, which is the wavenumber across the edges of the
sound that goes that way. What an edge adds is taken at its slant (:func:`_slanted`):
the sound that bends at the edge leaves it on the cone of the angle at which it came, by
the shortest way from end to end over the edge, and the uniform theory's terms are those
of that angle, with that way's length for their phase; what the marches' screen sheds is
taken off them as it is in the marches' plane. Each edge is taken as if the row's other
edges were turned as it is: that is so for a row the same all along its edges, and not
measured for edges at different slants. A barrier's strip is summed for ends apart along
it (:func:`~acoustic_spaces.engine.strip.beyond_screen`).

The one edge between the ends, which both light themselves, takes a slant of up to
``SLANT_MOST_DEG`` degrees: what the march's screen sheds there is taken off as it was
marched, whatever its phase. One rigid screen is within 0.10 dB of the exact solution at
30, 60 and 75 degrees, where it is 0.09 to 5.33 dB low as an edge across the way; a
roof's corner within 0.11, 0.36 and 0.61 dB, the most with an end a metre before the
corner; a barrier on a rigid ground that is a strip within 0.08 dB of the strip's exact
level at 30 to 75 degrees.

In a row a march lights the edges, and the march's way over an edge, in the vertical
plane along the way, is longer than the sound's own where an end's way to the row is
steep: at a slant of 60 degrees over two roofs 10 m wide in all, from a source level
with them and 10 m before them to a listener 4.5 m below them and 5 m past them (both
across the edges), by 8 cm, and by 38 cm for the way to the listener's image in the
ground. The edges' terms then have one phase and the marched field another, and no one
phase of the terms holds for ways that differ by different lengths. Where the section's
phase on the terms of the edges that a march lights would change the sound by more than
``SLANT_APART_MOST`` of it, the row is out of reach (:func:`_on_one_way`), as is such an
edge turned more than ``SLANT_ROW_MOST_DEG`` degrees. Against the wave calculation at 30
and 60 degrees (``validation/rows.py``) the solver takes these in every band: rows in
free space that the sound just clears or bends down behind by 22 degrees, within 0.22 dB
of it; a roof under the source up to its edge, within 0.19 dB; a screen on a roof under
both ends, within 0.53 dB. But in the 63 Hz band at 60 degrees it is 0.54 dB low over
three screens and 0.63 dB high over three roofs. (At that slant the row is, in
wavelengths across its edges, what it is across the way an octave lower, and there the
solver is 0.30 to 0.33 dB low over the three screens; over the roofs that is not
measured.) Two roofs and three screens on a rigid ground, between ends from half a metre
below their tops to half a metre above them or 1.5 m above the ground, are taken in the
63 Hz band alone at 30 degrees (at 15 of 18 places) and in no band at 60, where the
level from an end at roof level to one at street level would be up to 6.4 dB off.

The march goes both ways, and the listener takes the mean of the two, so that the sound
from A to B is that from B to A. Each begins at its first obstacle, from the field in
two dimensions that its end and the end's image in the ground have there (a Hankel
function, which a wavelength and more from the end is exp(ikR) / sqrt(R); the image's
with the spherical wave's reflection factor over a ground with an impedance): the
parabolic equation's own starting field holds for sound within some 35 degrees of the
horizontal, and the way from the street up to a roof is steeper. The first obstacle's
edge is lit by the end and by its image apart, each from its own direction. An end
over a roof has its own field up to the first top past it, a barrier on the roof, what
stands against the roof's far side, or that side: its march begins there, with the
end's image in the roof, which is rigid.

Sound that leaves an end almost straight up crosses the march's absorbing layer in too
short a range for the layer to take it, and would come back down from the top of the
march's grid. The layer is made thicker for what a start needs (:func:`_layer`): the
start from the first top up to the upper end of the first Fresnel zone of the way on
(and of the image's), which reaches metres past the top where the way runs almost
along the start, up to ``LAYER_MOST`` times its usual thickness. A start that needs
more is not marched: :func:`over_row` raises :class:`OutOfReach` for an end level
with its first top and closer to it than 3.7 to 3.9 cm (the 63 to 500 Hz octave
bands), for one close before it and high above its top (10 cm before it, more than
2.6 m above in the 500 Hz band, 5.2 m in the 250 Hz band), and for a way from an end
to its first top, or from the end's image in the ground or in the roof under it,
that is too steep (in the 500 Hz band beyond 88.2 degrees from an end a metre before
a wall and 87.4 degrees from one 10 cm before it, in the 250 Hz band beyond 89.1 and
88.7, in the 125 Hz band beyond 89.6 and 89.4, in the 63 Hz band beyond 89.8 and
89.7). Only the reflection off the ground can be left out on demand
(``steep_reflections``), which is how a row high above a ground stands in free
space; a roof's cannot.

Deep in a shadow close behind a high top, where the sound is weak, the level depends
on the height of the march's grid of heights: behind a wall 20 m high on a rigid
ground, both ends 5 m from its foot (46 dB below the free field), by up to 4.6 dB on
the usual grid and 0.72 dB on one four times as high. The cause is not traced. Where
the way from either end to its first bend is steeper than ``STEEP_END_DEG``, the
layer is made so thick that the grid reaches ``GRID_TIMES`` times as high above the
lower end as it does with the layer ``HEADROOM_M`` above everything (a layer that
starts higher, over a long way or as it is given, is made thicker by less). One
rigid screen 3 to 20 m high on a rigid ground, the source 0.5 to 30 m before it and
the listener 0.5 to 40 m past it, each 0.5 to 4 m up, is then within 0.31 dB of the
exact level where the strip is solved, and of the four paths over its top in the
other bands of the 563 places the solver takes in every band; at six places behind
walls 15 and 20 m high a grid twice as high again changes the levels by 0.10 dB at
most.

Just past a top the field changes over less than a cell of the march's grid of
heights: the march that arrives at an end close to a top gets cells of half the
end's distance from the nearest top (``CELLS_TO_TOP``), up to ``FINEST`` times as
many a wavelength. And the root of the path's length makes a point source's sound
of the field in two dimensions far from the ends alone: ends closer to each other
over the tops than ``STRING_WAVELENGTHS`` wavelengths of the lowest frequency are
out of reach too.

Where the ground changes before an end's first obstacle, nothing stands between the
ends, or the roof under an end goes on to the other end with no barrier on it between
the two, that end's march begins at the end, with the parabolic equation's starting
field. That field is the end's own within some 35 degrees of the horizontal and from
about a wavelength on, and :func:`over_row` raises :class:`OutOfReach` beyond: for a
way on from such an end, or from its image in the ground or the roof, steeper than
``FROM_END_DEG``, and for a first top, or the other end with no top before it, closer
than ``FROM_END_WAVELENGTHS`` wavelengths of the lowest frequency
(:func:`_from_its_end`), whatever ``steep_reflections`` is: a march from its end has
the reflection in its starting field. Within those such a march is within 0.5 dB of
the edge integrals of the four paths over a barrier behind a change of the ground,
and of the direct sound and the reflection with nothing between the ends; at 50
degrees it would be 1.0 dB low, and a quarter of a wavelength before a barrier on a
rigid ground up to 1.0 dB low.

The marches take the row's skyline (:func:`skyline`): over buildings that overlap, the
highest roof; roofs level with each other that touch as one roof. A corner that
something at least as high stands against (a lower roof against a wall) sheds nothing
(:func:`edges_of`).

Where the edges' terms stand for a barrier (a parapet on the edge of a roof, a barrier
with anything else between the ends or on a ground with an impedance, a wall given as
a building, and a barrier alone on a rigid ground or roof where the strip is not
solved), they take it for a half-plane, lit by the end and by its image in the surface
it stands on: the sound that goes on between its top and a rigid roof or ground under
it is not in them. With them a barrier 10 cm high alone on a rigid ground, half a
metre past the source, would be 3.0 dB from the exact level, and one 1 m high 2.2 dB,
where the strip is within 0.2 dB (tests/test_rows.py); against a wave calculation, a
parapet on the edge of the roof under an end is within 0.8 dB from a quarter of a
wavelength on and 1.3 dB below that (docs/validation.md, "Rows of roofs and barriers
against a wave calculation").

A march one way has no sound that comes back, and what stands behind an end does
nothing: a roof under an end goes on behind it without end (at the roof's edge there
the level is 3.5 dB too high, with the edge 5 cm behind the end up to 2.6 dB, with
it 20 cm to 1 m behind up to 1.2 dB too high or 0.9 dB too low, from 1.5 m on within
0.35 dB).
Between rigid screens on a rigid ground, where sound goes to and fro, the level comes
out too low; that the sound coming back is what lacks there is likely, not shown (see
docs/validation.md, "Rows of roofs and barriers against a wave calculation").
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import hankel1

from acoustic_spaces.engine import parabolic, strip
from acoustic_spaces.engine.diffraction import wedge_prefactor, wedge_terms
from acoustic_spaces.materials.ground import spherical_reflection

#: The march's absorbing layer starts this far (m) above the source, the listener and
#: the highest top on the way, and at least ``TOP_SLOPE`` times the range up.
HEADROOM_M = 20.0
TOP_SLOPE = 0.15
#: The ground before a march's first obstacle is looked up this often (m), to see
#: whether it is one ground.
GROUND_STEP_M = 0.25
#: A wave that leaves a march's start almost straight up crosses the absorbing layer in
#: too short a range for it to take the wave, and comes back down from the top of the
#: march's grid. The layer is made thick enough to leave this share at most of the
#: waves a march's start needs: those of the end and of its image in the ground (or in
#: the roof under it) to the start from the first top up to the upper end of the first
#: Fresnel zone of their strings over the tops, and round that as far as
#: ``FRESNEL_RADII`` radii of the zone of a string straight across the start, and
#: ``START_WAVELENGTHS`` wavelengths at least (the start of an end closer than a
#: wavelength). It grows to ``LAYER_MOST`` times its usual thickness at most; a start
#: that needs more is out of the solver's reach (:class:`OutOfReach`).
STEEP_LEFT = 1e-3
FRESNEL_RADII = 3.0
START_WAVELENGTHS = 1.0
LAYER_MOST = 8.0
#: A march that begins at its end has the parabolic equation's starting field, which is
#: the end's own within ``FROM_END_DEG`` degrees of the horizontal
#: (:mod:`~acoustic_spaces.engine.parabolic`) and from about a wavelength on: the way
#: on from such an end and from its image in what it stands over is no steeper, and its
#: first top (the other end, with no top before it) no closer than
#: ``FROM_END_WAVELENGTHS`` wavelengths, or the end is out of the solver's reach
#: (:class:`OutOfReach`).
FROM_END_DEG = 35.0
FROM_END_WAVELENGTHS = 1.0
#: The march's field in two dimensions becomes a point source's by the root of the
#: length of the path over the tops, which holds far from the ends: a path shorter
#: than ``STRING_WAVELENGTHS`` wavelengths of the lowest frequency is out of the
#: solver's reach (:class:`OutOfReach`).
STRING_WAVELENGTHS = 0.5
#: Just past a top the field changes over less than a cell of the march's grid of
#: heights. The march that arrives at an end gets cells no larger than
#: 1 / ``CELLS_TO_TOP`` of that end's distance from the nearest top, up to ``FINEST``
#: times as many a wavelength as it is asked for.
CELLS_TO_TOP = 2.0
FINEST = 8.0
#: Deep in a shadow close behind a high top, where the sound is weak, the level
#: depends on the height of the march's grid of heights (the cause is not traced).
#: Where the way from either end to its first bend is steeper than
#: ``STEEP_END_DEG``, the layer is made so thick that the grid reaches ``GRID_TIMES``
#: times as high above the lower end as it does with the layer ``HEADROOM_M`` above
#: everything (behind walls 15 and 20 m high the levels of six places change by
#: 0.10 dB at most with a grid twice as high again).
STEEP_END_DEG = 30.0
GRID_TIMES = 8.0
#: An edge that the way crosses is turned from across it by this at most (degrees): the
#: edges' terms at a slant are measured up to ``SLANT_MOST_DEG`` for an edge that both
#: ends light themselves, and up to ``SLANT_ROW_MOST_DEG`` for one that a march lights
#: (:class:`OutOfReach` beyond).
SLANT_MOST_DEG = 75.0
SLANT_ROW_MOST_DEG = 60.0
#: At a slant the marches' way over an edge, in the vertical plane along the way, is
#: longer than the sound's own, the shortest from end to end over the edge, where an
#: end's way to the row is steep. The terms of an edge that a march lights have the
#: own way's phase and the march the section's: where the section's phase on those
#: terms would change the sound at the frequencies given by more than this share of it
#: (root mean square), the row is out of reach (:class:`OutOfReach`).
SLANT_APART_MOST = 0.1
#: Edges and tops this close (m) are at one place or one height.
_SAME_M = 1e-9
#: The open angles over pi of a half-plane and of a roof's corner, and the sine of the
#: angle between the way and the edges, which cross it at right angles.
_PLANE, _CORNER, _ACROSS = np.asarray(2.0), np.asarray(1.5), np.asarray(1.0)


class OutOfReach(ValueError):
    """Raised by :func:`over_row` for ends closer to each other over the tops than
    ``STRING_WAVELENGTHS`` wavelengths, and for an end whose march cannot hold what
    its start needs. Where the march begins at the end's first obstacle (over a roof:
    at the first top past the end): a thicker absorbing layer than ``LAYER_MOST``
    times the usual one, for an end close to that obstacle with a way past its top
    (or its image's) that is too steep. Where it begins at the end itself
    (:func:`_from_its_end`): a way on from the end, or from its image in the ground
    or the roof under it, steeper than ``FROM_END_DEG``, or a first top, or the other
    end with no top before it, closer than ``FROM_END_WAVELENGTHS`` wavelengths. And
    for a way that crosses the row at a slant: an edge turned more than
    ``SLANT_MOST_DEG`` degrees from across it, or more than ``SLANT_ROW_MOST_DEG``
    where a march lights it, and a row between ends whose ways to it are so steep
    that its sound would depend on the phase its edges' terms are given
    (``SLANT_APART_MOST``)."""


@dataclass(frozen=True)
class Row:
    """The sound from a source to a listener over a row (:func:`over_row`): complex
    pressures (F,), time as exp(-i omega t), free field exp(ikR) / R."""

    #: The parabolic equation's field over the row, the mean of the march from the
    #: source and that from the listener.
    marched: np.ndarray
    #: What the rigid edges add to it.
    edges: np.ndarray

    @property
    def pressure(self) -> np.ndarray:
        """(F,) the pressure at the listener."""
        return self.marched + self.edges


@dataclass(frozen=True)
class _Way:
    """A march from one end of the row, at 0 along it, to the other."""

    #: (F,) the pressure at the other end.
    pressure: np.ndarray
    #: (what, m along the march, top) -> (F,) the march's pressure at each edge it met.
    lit: dict
    #: The end's height (m), and the tops between the ends ((m along the march, m up)).
    height: float
    tops: list
    #: Where the march began from its end's own field (m along it: at its first
    #: obstacle, or over a roof at the first top past the end), or None where it began
    #: at the end with the parabolic equation's.
    first: float | None
    #: What the end stands over up to there (m up: the ground, 0, or a roof), and its
    #: impedance (F,; infinite: rigid, as a roof is).
    under: float
    ground: np.ndarray
    #: What :func:`_kept` takes for the waves of the march's start.
    through: float


def over_row(
    frequencies,
    speed_of_sound: float,
    impedance,
    zs: float,
    reach: float,
    zr: float,
    obstacles,
    top: float | None = None,
    steps_per_wavelength: float = parabolic.STEPS_PER_WAVELENGTH,
    steep_reflections: str = "raise",
    slants=None,
) -> Row:
    """The sound at ``frequencies`` (F,, Hz, marched together: one band's) from a source
    ``zs`` m above the ground to a listener ``zr`` m above it, ``reach`` m away along
    the ground, over the ``obstacles`` between them: (near, far, top) (m along the way
    from the source, and above the ground), a barrier where ``far`` is ``near``, else a
    building with its rigid roof at ``top`` (as
    :func:`~acoustic_spaces.engine.parabolic.march` takes them; one no higher than the
    ground is none). What stands before the source or past the listener does nothing,
    a building's wall just behind an end and a barrier under it included: the march
    goes one way. ``impedance(r)`` gives the ground's normalised impedance at the
    frequencies r m from the source (time as exp(-i omega t)), or None where it is
    rigid, as the march takes it; before each end's first obstacle it is looked up
    every ``GROUND_STEP_M``. ``top`` (m) is where the march's absorbing layer starts
    (:func:`layer_top` by default).

    Raises a ValueError for an obstacle whose far side lies before its near side, for
    a ``top`` that is not above both ends and every obstacle between them, and for an
    end below the top of a building it stands in or of a wall or a barrier it stands
    against on its way to the other end; and :class:`OutOfReach` for ends less than
    ``STRING_WAVELENGTHS`` wavelengths from each other over the tops and for an end
    whose march cannot hold what its start needs, either end before either march
    runs. ``steps_per_wavelength`` is the march's; the march that arrives at an end
    close past a top takes more (``CELLS_TO_TOP``). With ``steep_reflections``
    "leave out" in place of "raise", an end's reflection off the ground before its
    first obstacle that is too steep for the march is left out instead, but for as
    much of it as the march's layer still takes (:func:`_kept`): a row high above a
    ground then stands in free space. That is for a march that begins at its first
    obstacle; one that begins at its end cannot leave its reflection out, and is
    refused as before, as is an end over a roof for its reflection off the roof.

    ``slants`` are the angles (radians) by which the obstacles are turned from across
    the way, seen from above: one for each of ``obstacles``, a number for both its
    sides or, for a roof, a pair (near side, far side); None where all cross it at
    right angles. A slant's sign does not count.
    ``near``, ``far`` and the ends' heights are along the way as before, in the
    vertical plane through the two ends, which is the march's; each edge's terms are
    those of an edge at its slant (:func:`_slanted`). An edge turned more than
    ``SLANT_MOST_DEG`` degrees raises :class:`OutOfReach` before either march runs, as
    does one turned more than ``SLANT_ROW_MOST_DEG`` degrees that the two ends do not
    both light themselves; and after the marches, a row whose sound would depend on
    which way's phase the terms of its slanted edges are given, the sound's own or the
    marches' (``SLANT_APART_MOST``, :func:`_on_one_way`). A slant that is not a number,
    a count of slants that is not the obstacles', or a barrier with slants of different
    sizes for its two sides raises a ValueError."""
    if steep_reflections not in ("raise", "leave out"):
        raise ValueError('steep_reflections is "raise" or "leave out"')
    obstacles = list(obstacles)
    turned = _slants(obstacles, slants)
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    reach = float(reach)
    row = skyline(obstacles)
    mirrored = [(reach - far, reach - near, h) for near, far, h in row]
    _stands_free(row, zs, "source")
    _stands_free(mirrored, zr, "listener")
    back = None if impedance is None else (lambda r: impedance(reach - r))
    # Before either march runs too: an edge turned too far is out of reach.
    slanting = {
        edge: _slant_of(turned, *edge)
        for edge in (edges_of(row) if turned else [])
        if _SAME_M < edge[1] < reach - _SAME_M
    }
    if any(slant > math.radians(SLANT_ROW_MOST_DEG) for slant in slanting.values()):
        _lit_by_both(
            slanting,
            _first(f, impedance, reach, row)[0],
            _first(f, back, reach, mirrored)[0],
            reach,
        )
    between = [o for o in row if o[1] > _SAME_M and o[0] < reach - _SAME_M]
    if top is None:
        top = layer_top(zs, zr, reach, between)
    elif top <= max([zs, zr] + [h for _, _, h in between]):
        raise ValueError("the absorbing layer starts above both ends and every obstacle")
    k = 2.0 * math.pi * f / float(speed_of_sound)
    common = (f, float(speed_of_sound), reach, top, steps_per_wavelength)
    left_out = steep_reflections == "leave out"
    # Both ends are checked before either march runs.
    from_source = _march_from(*common, impedance, zs, zr, row, left_out)
    from_listener = _march_from(*common, back, zr, zs, mirrored, left_out)
    forward, backward = from_source(), from_listener()

    added = np.zeros(len(f), dtype=complex)
    # What the section's phase in place of the own way's would change (a slant).
    apart = np.zeros(len(f), dtype=complex)
    # From the listener's end a near corner is a far one.
    seen_back = {"screen": "screen", "roof": "behind", "behind": "roof"}
    for what, x, h in edges_of(row):
        if not _SAME_M < x < reach - _SAME_M:
            continue
        mirrored_edge = (seen_back[what], reach - x, h)
        slant = slanting.get((what, x, h), 0.0)
        sheds = np.zeros(len(f), dtype=complex)
        for from_source, a, to_source, source_s, before in _lit(forward, k, (what, x, h)):
            for from_listener, b, onwards, listener_s, after in _lit(backward, k, mirrored_edge):
                to_listener = (-onwards[0], onwards[1])
                kl = k * a * b / (a + b)
                at = _slanted(k, a, b, before, after, slant, reach) if slant else None
                ways = (to_source, to_listener, source_s, listener_s)
                shed = _shed(what, k, kl, *ways, at)
                sheds += from_source * from_listener * shed / math.sqrt(a + b)
                if at is not None and not (source_s and listener_s):
                    section = _shed(what, k, kl, *ways, replace(at, turn=np.abs(at.turn)))
                    apart += from_source * from_listener * (shed - section) / math.sqrt(a + b)
        plane = _plane_under(forward, backward, x, h, reach) if what == "screen" else None
        if plane is not None:
            # The barrier and its image in the plane are a strip, whose field is known
            # beyond the first order of the waves between the top and the plane.
            # At the frequencies at which the ground is rigid on both sides of it.
            low = (k * (h - plane) <= strip.MOST_KH) & _rigid(forward) & _rigid(backward)
            # An end on the plane may be given a rounding below it (:func:`_stands_free`).
            ends = (x, max(zs - plane, 0.0)), (reach - x, max(zr - plane, 0.0))
            whole = None
            # The strip's sum is measured up to its own slant (``strip.SLANT_MOST_DEG``).
            # Were that less than ``SLANT_MOST_DEG``, beyond which no edge comes this far,
            # the edges' terms would stand between the two.
            if np.any(low) and slant <= math.radians(strip.SLANT_MOST_DEG):
                whole = strip.beyond_screen(k[low], h - plane, *ends, slant=slant)
            if whole is not None:
                sheds[low] = whole
        added += sheds
    marched = 0.5 * (forward.pressure + backward.pressure)
    _on_one_way(apart, marched + added)
    return Row(marched=marched, edges=added)


def _shed(
    what: str, k: np.ndarray, kl: np.ndarray, to_source, to_listener, source_s, listener_s, slant
) -> np.ndarray:
    """(F,) what an edge sheds beyond the march's screen (``what``: a barrier's top,
    "screen", or a roof's near corner, "roof", or far one, "behind") for one wave from
    each end, by the directions from the edge to where they come from and whether each
    comes straight from its end or the end's image (:func:`_screen`, :func:`_corner`)."""
    if what == "screen":
        return _screen(k, kl, to_source, to_listener, slant)
    if what == "roof":
        return _corner(k, kl, to_source, 1.0, to_listener if listener_s else None, slant)
    return _corner(k, kl, to_listener, -1.0, to_source if source_s else None, slant)


def _lit_by_both(slanting: dict, first: float | None, last: float | None, reach: float) -> None:
    """Raises :class:`OutOfReach` for an edge of ``slanting`` ((what, m along the way,
    top) -> its slant, radians) turned more than ``SLANT_ROW_MOST_DEG`` degrees that the
    two ends do not both light themselves: one at which not both marches begin
    (``first`` m from the source, ``last`` m from the listener; None for a march that
    begins at its end)."""
    for (_, x, _), slant in slanting.items():
        both = (
            first is not None
            and last is not None
            and abs(x - first) <= _SAME_M
            and abs(reach - x - last) <= _SAME_M
        )
        if slant > math.radians(SLANT_ROW_MOST_DEG) and not both:
            raise OutOfReach(
                f"an edge {x:.3g} m along the way that is turned {math.degrees(slant):.1f} "
                f"degrees from across it and that a march lights: the terms of such an edge "
                f"take {SLANT_ROW_MOST_DEG:.0f} degrees"
            )


def _on_one_way(apart: np.ndarray, pressure: np.ndarray) -> None:
    """Raises :class:`OutOfReach` where the section's phase, in place of the own way's,
    on the terms of the slanted edges that a march lights would change the sound by
    ``apart`` (F,) and that is more than ``SLANT_APART_MOST`` of the ``pressure`` (F,),
    as root mean squares over the frequencies: the marches have the section's phase,
    and such a sound depends on which of the two those terms are given."""
    if not np.any(apart):
        return
    share = math.sqrt(float(np.mean(np.abs(apart) ** 2) / np.mean(np.abs(pressure) ** 2)))
    if share > SLANT_APART_MOST:
        raise OutOfReach(
            f"a way that crosses the row at a slant with an end's way to the row so steep "
            f"that the edges' terms on the sound's own way and on the marches' differ by "
            f"{share:.2f} of the sound: the solver takes {SLANT_APART_MOST:.2f}"
        )


def _plane_under(forward: _Way, backward: _Way, x: float, h: float, reach: float) -> float | None:
    """The height (m) of the plane on which the barrier with its top ``h`` m up, ``x``
    m along the way, stands alone between the ends: the first top of both marches, on
    what both ends stand over (the ground, or one roof), each march taking all of its
    end's wave and of that of the end's image (:func:`_kept`). None otherwise. Whether
    the plane is rigid is by the frequency (:func:`_rigid`)."""
    for way, at in ((forward, x), (backward, reach - x)):
        if way.first is None or abs(at - way.first) > _SAME_M:
            return None
        for z0 in (way.height, 2.0 * way.under - way.height):
            if float(_kept(way.through, h - z0)) < 1.0:
                return None
    return forward.under if forward.under == backward.under else None


def _rigid(way: _Way) -> np.ndarray:
    """(F,) whether what a march's end stands over is rigid at each frequency."""
    return np.isinf(way.ground)


def _stands_free(row, height: float, end: str) -> None:
    """Raises a ValueError where the ``end`` at 0 along ``row``, ``height`` m up, is
    below the top of a building it stands in, of a wall it stands against on its way
    along the row, or of a barrier in whose plane it stands."""
    for near, far, h in row:
        at_it = near <= _SAME_M < far if far > near else abs(near) <= _SAME_M
        if at_it and height < h - _SAME_M:
            raise ValueError(f"the {end} stands in or against an obstacle, below its top")


def _march_from(
    f, c, reach, top, steps_per_wavelength, impedance, z0, z1, row, left_out: bool = False
) -> Callable[[], _Way]:
    """The march from an end ``z0`` m up over ``row`` ((near, far, top) from that end)
    to the other end, ``reach`` m on and ``z1`` m up, made ready: what the end's start
    needs is checked here (:class:`OutOfReach`), and the march runs when the result
    is called. It has ``steps_per_wavelength`` height steps a wavelength at least
    (more where it arrives close past a top: ``CELLS_TO_TOP``), and a higher grid
    where the way at either end is steeper than ``STEEP_END_DEG``; ``left_out``,
    whether a reflection off the ground that is too steep for the march is left out
    (:func:`_layer`)."""
    k = 2.0 * math.pi * f / c
    # What stands before this end or past the other does nothing.
    row = [o for o in row if o[1] > _SAME_M and o[0] < reach - _SAME_M]
    tops = sorted(
        (x, h) for near, far, h in row for x in {near, far} if _SAME_M < x < reach - _SAME_M
    )
    whole = string([(0.0, z0), *tops, (reach, z1)])
    length = _length(whole)
    shortest = STRING_WAVELENGTHS * c / float(np.min(f))
    if length < shortest:
        raise OutOfReach(
            f"ends {length:.3g} m from each other over the tops: at {float(np.min(f)):.0f} Hz "
            f"and above the solver takes none closer than {shortest:.3g} m"
        )
    if tops:
        # The top nearest to the end this march arrives at.
        close = min(math.hypot(reach - x, h - z1) for x, h in tops)
        fine = CELLS_TO_TOP * c / float(np.max(f)) / close
        steps_per_wavelength = min(max(steps_per_wavelength, fine), FINEST * steps_per_wavelength)
    first, under = _first(f, impedance, reach, row)
    ground = _impedances(impedance if under == 0.0 else None, 0.0, len(f))
    start, layer, through = None, None, math.inf
    #: The parabolic equation's starting field is a point source's on the way along the
    #: ground; up or down a path it is that times the root of the path's cosine.
    cosine = _cosine(whole)
    if first is not None:
        cosine = 1.0

        # The end's image in what it stands over: the ground, or the roof under it.
        mirror = 2.0 * under - z0
        below = string([(0.0, mirror), *tops, (reach, z1)])
        mirrored = _length(below)

        def field(z):
            # The image's wave by the root of its longer path less, which the path's
            # length at the other end then takes it for.
            direct = np.hypot(first, z - z0)
            image = np.hypot(first, z - mirror)
            reflected = _reflection(ground, k, (z - mirror) / image, image)
            u = _kept(through, z - z0) * _wave(k, direct)
            u = u + (_kept(through, z - mirror) * math.sqrt(length / mirrored)) * reflected * (
                _wave(k, image)
            )
            return u / math.sqrt(first)

        # The layer leaves exp(-2 strength thickness / (3 slope)) at most of a wave that
        # goes up through it at that slope and comes back down from the top of the grid.
        floor = max(h for x, h in tops if x <= first + _SAME_M)
        # Only the ground's reflection can be left out, not a roof's.
        off = "roof" if under > 0.0 else "ground"
        layer = _layer(f, c, first, floor, (whole, below), left_out and under == 0.0, off)
        through = 2.0 * parabolic.LAYER_STRENGTH * layer / 3.0 * first
        start = (first, field)
    else:
        _from_its_end(f, c, z0, z1, reach, row, tops)
    thick = layer
    if max(_steepness(whole[:2]), _steepness(whole[-2:])) > STEEP_END_DEG:
        # From the lower end up: the air up to a layer that starts ``HEADROOM_M`` above
        # everything, and the layer. What a layer that starts higher gives, counts.
        lower = min(z0, z1)
        air = layer_top(z0, z1, 0.0, row) - lower
        least = parabolic.layer_thickness(f, c) if layer is None else layer
        thick = max(least, GRID_TIMES * (air + least) - (top - lower))

    def speed(z):
        return np.full(np.shape(np.asarray(z, dtype=float)), c)

    def run() -> _Way:
        found: list = []
        p = parabolic.march(
            f,
            speed,
            impedance,
            z0,
            np.array([reach]),
            np.array([z1]),
            top,
            obstacles=row,
            refracting=False,
            steps_per_wavelength=steps_per_wavelength,
            edges=found,
            start=start,
            layer=thick,
        )[:, 0]
        return _Way(
            pressure=p * math.sqrt(reach / (length * cosine)),
            lit={(what, at, h): value for at, what, h, value in found},
            height=float(z0),
            tops=tops,
            first=first,
            under=under,
            ground=ground,
            through=through,
        )

    return run


def _wave(k: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(F, n) the field in two dimensions of a source ``distance`` (n,) m away, which
    far from it is exp(ikR) / sqrt(R)."""
    scale = np.sqrt(0.5 * np.pi * k) * np.exp(0.25j * np.pi)
    return scale[:, None] * hankel1(0, k[:, None] * np.atleast_1d(distance)[None, :])


def _layer(
    f, c: float, first: float, floor: float, strings, left_out: bool = False, off: str = "ground"
) -> float:
    """The thickness (m) of the absorbing layer for a march that begins ``first`` m
    from its end, above the top ``floor`` m up there: what leaves ``STEEP_LEFT`` at
    most of the waves from the end and from its image (in the ground, or in the roof
    under it: ``off``, which the error names) to the heights of the start that their
    ``strings`` need. Those are the heights from the top there,
    whose edge the end lights, up to the upper end of each string's first Fresnel zone
    on the start (:func:`_zone`), and the start round them as far as
    ``FRESNEL_RADII`` radii of the zone of a string straight across (sqrt(wavelength
    times ``first``)) and ``START_WAVELENGTHS`` wavelengths at least, at the lowest
    frequency. The usual thickness at least. Where that is more than ``LAYER_MOST``
    times the usual thickness it raises :class:`OutOfReach`; but with ``left_out``
    the image's string then gets the thickest layer, or none where those heights are
    themselves too steep for that."""
    usual = parabolic.layer_thickness(f, c)
    most = LAYER_MOST * usual
    by_slope = 3.0 * math.log(1.0 / STEEP_LEFT) / (2.0 * parabolic.LAYER_STRENGTH)
    longest = c / float(np.min(f))
    around = max(FRESNEL_RADII * math.sqrt(longest * first), START_WAVELENGTHS * longest) / first
    wanted = [usual]
    for of_image, path in enumerate(strings):
        onwards = next(point for point in path[1:] if point[0] > first + _SAME_M)
        high = _zone(first, floor, path[0], onwards, 0.5 * longest)
        slope = max(abs(floor - path[0][1]), abs(high - path[0][1])) / first
        if (slope + around) * by_slope <= most:
            wanted.append((slope + around) * by_slope)
        elif not (of_image and left_out):
            raise OutOfReach(
                f"an end {first:.3g} m before its first "
                f"{'top' if off == 'roof' else 'obstacle'}, with the start that its "
                f"{f'reflection off the {off}' if of_image else 'way on'} needs up to "
                f"{math.degrees(math.atan(slope)):.1f} degrees steep: the march at "
                f"{float(np.min(f)):.0f} Hz and above would need "
                f"{(slope + around) * by_slope / usual:.3g} times its usual absorbing "
                f"layer for that start, and takes {LAYER_MOST:.3g}"
            )
        elif slope * by_slope <= most:
            wanted.append(most)
    return max(wanted)


def _zone(first: float, floor: float, start, onwards, excess: float) -> float:
    """The greatest height (m) of a march's start, ``first`` m along it and from
    ``floor`` up, over which the path from ``start`` to ``onwards`` ((m along, m up),
    before the start and past it) is at most ``excess`` m longer than the shortest
    such path: with half a wavelength, the upper end of the path's first Fresnel zone
    on the start. Where the path runs nearly along the start, almost straight up or
    down, that zone is many times as high as a string straight across has it."""

    def length(z: float) -> float:
        before = math.hypot(first - start[0], z - start[1])
        return before + math.hypot(onwards[0] - first, onwards[1] - z)

    straight = start[1] + (onwards[1] - start[1]) * (first - start[0]) / (onwards[0] - start[0])
    shortest = max(floor, straight)
    most = length(shortest) + excess

    inside, step = shortest, excess
    while length(shortest + step) <= most:
        step *= 2.0
    outside = shortest + step
    for _ in range(60):
        middle = 0.5 * (inside + outside)
        inside, outside = (middle, outside) if length(middle) <= most else (inside, middle)
    return inside


def _from_its_end(f, c: float, z0: float, z1: float, reach: float, row, tops) -> None:
    """Raises :class:`OutOfReach` for a march that begins at its end, ``z0`` m up, with
    the parabolic equation's starting field, where that field is not the end's own:
    where the way on from the end, or from its image in what it stands over (the
    ground, or the roof of ``row`` under it), is steeper than ``FROM_END_DEG`` (the
    first stretch of their strings over ``tops`` to the other end, ``reach`` m on and
    ``z1`` m up), or the first top, or the other end where no top stands between
    them, is closer than ``FROM_END_WAVELENGTHS`` wavelengths of the lowest
    frequency. Whatever ``steep_reflections`` is: the parabolic equation's starting
    field has the reflection in it."""
    under = max((h for near, far, h in row if near <= _SAME_M < far), default=0.0)
    if under > 0.0:
        begins = "an end over a roof"
    elif row:
        begins = "an end with the ground changing before its first obstacle"
    else:
        begins = "an end with nothing between it and the other end"
    ways = (
        ("way on", z0),
        (f"reflection off the {'roof' if under > 0.0 else 'ground'}", 2.0 * under - z0),
    )
    lowest = float(np.min(f))
    for name, height in ways:
        path = string([(0.0, height), *tops, (reach, z1)])
        steep = math.degrees(math.atan2(abs(path[1][1] - path[0][1]), path[1][0] - path[0][0]))
        if steep > FROM_END_DEG:
            raise OutOfReach(
                f"{begins}, whose march begins at the end, with its {name} "
                f"{steep:.1f} degrees steep: such a march takes {FROM_END_DEG:.0f} degrees"
            )
    nearest = FROM_END_WAVELENGTHS * c / lowest
    ahead = tops[0][0] if tops else reach
    if ahead < nearest:
        raise OutOfReach(
            f"{begins}, whose march begins at the end, {ahead:.3g} m before "
            f"{'its first top' if tops else 'the other end'}: such a march at "
            f"{lowest:.0f} Hz and above takes none closer than {nearest:.3g} m"
        )


def _kept(through: float, rise):
    """The share of a wave of a march's start that the march and the first edge take,
    by the wave's ``rise`` (m) on its way to the start: all of it where the absorbing
    layer leaves less than ``STEEP_LEFT`` of the wave (``through`` over the rise is
    what the layer takes of it, as an exponent), else as much as leaves that."""
    with np.errstate(divide="ignore", over="ignore"):
        return np.minimum(1.0, STEEP_LEFT * np.exp(through / np.abs(rise)))


def _first(f, impedance, reach: float, row) -> tuple[float | None, float]:
    """Where a march begins from its end's own field (m along it), and what the end
    stands over up to there (m up). Over the ground (0): at its first obstacle, with
    one ground from the end up to it. Over a roof (the roof's height): at the first
    top past the end, a barrier on the roof or the roof's far side. None where nothing
    stands on the way, the ground changes before the first obstacle, or the roof goes
    on to the other end with no barrier on it between the two."""
    on_the_way = [o for o in row if o[1] > _SAME_M and o[0] < reach - _SAME_M]
    over = [o for o in on_the_way if o[0] <= _SAME_M and o[1] > o[0]]
    if over:
        _, far, under = over[0]
        # The barriers on the roof, what stands against its far side, and that side.
        ahead = [far, *(near for near, _, _ in on_the_way)]
        ahead = [x for x in ahead if _SAME_M < x < reach - _SAME_M]
        return (min(ahead) if ahead else None), under
    if not on_the_way:
        return None, 0.0
    first = min(near for near, _, _ in on_the_way)
    below = _impedances(impedance, 0.0, len(f))
    count = max(1, math.ceil(first / GROUND_STEP_M))
    for i in range(1, count):
        if not np.array_equal(_impedances(impedance, first * i / count, len(f)), below):
            return None, 0.0
    return first, 0.0


def _impedances(impedance, at: float, count: int) -> np.ndarray:
    """(count,) the ground's impedance ``at`` m along a march (infinite: rigid)."""
    value = impedance(at) if impedance is not None else None
    if value is None:
        return np.full(count, np.inf, dtype=complex)
    return np.broadcast_to(np.asarray(value, dtype=complex), (count,))


def _reflection(ground: np.ndarray, k: np.ndarray, cosine, distance) -> np.ndarray:
    """(F, n) the reflection factor of a point source's sound off a ground of impedance
    ``ground`` (F,, time as exp(-i omega t); infinite: rigid, 1), met at the incidence
    cosines ``cosine`` (n,) by paths ``distance`` (n,) m long from the image source."""
    rigid = np.isinf(ground)
    admittance = np.where(rigid, 0.0, np.conj(1.0 / np.where(rigid, 1.0, ground)))
    cosine, distance = np.atleast_1d(cosine), np.atleast_1d(distance)
    return np.conj(
        spherical_reflection(admittance[:, None], cosine[None, :], k[:, None], distance[None, :])
    )


def _lit(way: _Way, k: np.ndarray, edge: tuple) -> list[tuple]:
    """How a march's end lights an ``edge`` (what, m along the march, top): the waves
    that reach it, each as (the two-dimensional field (F,), the length of its path
    (m), the direction (along the march, up) from the edge to where it comes from,
    whether it comes straight from the end or from its image, not from the march,
    and the path's points). The first top of a march that begins there is lit by the
    end and by its image in what the end stands over (the ground, or a roof), each its
    own wave and as much of each as the march takes (:func:`_kept`); any other edge by
    the marched field, along the string from the end."""
    _, x, h = edge
    if way.first is not None and abs(x - way.first) <= _SAME_M:
        out: list[tuple] = []
        for image, z0 in enumerate((way.height, 2.0 * way.under - way.height)):
            a = math.hypot(x, h - z0)
            u = float(_kept(way.through, h - z0)) * np.exp(1j * k * a) / math.sqrt(a)
            if image:
                u = u * _reflection(way.ground, k, (h - z0) / a, a)[:, 0]
            out.append((u, a, (-x, z0 - h), True, [(0.0, z0), (x, h)]))
        return out
    path = string([(0.0, way.height), *(t for t in way.tops if t[0] < x - _SAME_M), (x, h)])
    cosine = 1.0 if way.first is not None else _cosine(path)
    before = path[-2]
    lit = way.lit[edge] * math.sqrt(x / cosine)
    return [(lit, _length(path), (before[0] - x, before[1] - h), False, path)]


def string(points) -> list[tuple[float, float]]:
    """The string pulled taut over ``points`` ((m along the way, m up), in order along
    the way) from the first to the last: those it bends at, the two ends included (the
    upper convex hull)."""
    out: list[tuple[float, float]] = []
    for x, z in points:
        while len(out) >= 2 and (
            (out[-1][0] - out[-2][0]) * (z - out[-2][1])
            - (out[-1][1] - out[-2][1]) * (x - out[-2][0])
            >= 0.0
        ):
            out.pop()
        out.append((float(x), float(z)))
    return out


def _length(path) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:], strict=True)
    )


def _steepness(leg) -> float:
    """The angle (degrees) of a stretch of a string, ``leg`` (two points), from the
    way along the ground, up or down."""
    (x0, z0), (x1, z1) = leg
    return math.degrees(math.atan2(abs(z1 - z0), x1 - x0))


def _cosine(path) -> float:
    """The cosine of the angle at which a path leaves its first point, from the way
    along the ground."""
    (x0, z0), (x1, z1) = path[0], path[1]
    return (x1 - x0) / math.hypot(x1 - x0, z1 - z0)


def layer_top(zs: float, zr: float, reach: float, obstacles) -> float:
    """Where the march's absorbing layer starts (m above the ground) for a source ``zs``
    and a listener ``zr`` up, ``reach`` m apart, over ``obstacles``."""
    highest = max([zs, zr] + [h for _, _, h in obstacles])
    return max(highest + HEADROOM_M, TOP_SLOPE * reach)


def skyline(obstacles) -> list[tuple[float, float, float]]:
    """The ``obstacles`` ((near, far, top), as :func:`over_row` takes them) as the row
    they make, in order along the way: stretches of roof that do not overlap, each as
    high as the highest building over it, those level with each other that touch as
    one; and the barriers that stand higher than a roof they stand on or against (of
    several at one place, the highest). Those no higher than the ground are left
    out, and a building whose two sides are at one place is a barrier there; one
    whose far side lies before its near side raises a ValueError."""
    found = {(float(a), float(b), float(h)) for a, b, h in obstacles if h > 0.0}
    if any(b < a for a, b, _ in found):
        raise ValueError("an obstacle's far side lies before its near side")
    found = {(a, a if b - a <= _SAME_M else b, h) for a, b, h in found}
    buildings = [o for o in found if o[1] > o[0]]
    stops = sorted({x for near, far, _ in buildings for x in (near, far)})
    out: list[tuple[float, float, float]] = []
    for a, b in zip(stops[:-1], stops[1:], strict=True):
        over = [h for near, far, h in buildings if near <= a and far >= b]
        if b - a <= _SAME_M or not over:
            continue
        if out and a - out[-1][1] <= _SAME_M and abs(out[-1][2] - max(over)) <= _SAME_M:
            out[-1] = (out[-1][0], b, out[-1][2])
        else:
            out.append((a, b, max(over)))
    for x, _, h in sorted((o for o in found if o[1] == o[0]), key=lambda o: -o[2]):
        if not _covered(out, (x, x, h), x):
            out.append((x, x, h))
    return sorted(out)


def edges_of(row) -> list[tuple[str, float, float]]:
    """The edges of a row (a :func:`skyline`) that shed sound, as (what, m along the
    way, top), in order: "screen" a barrier's top, "roof" and "behind" the near and
    the far corner of a stretch of roof, where nothing else of the row stands against
    it at least as high (a higher roof, or a barrier on the wall)."""
    out = []
    for near, far, h in row:
        if far == near:
            out.append(("screen", near, h))
            continue
        for what, x in (("roof", near), ("behind", far)):
            if not _covered(row, (near, far, h), x):
                out.append((what, x, h))
    return out


def _covered(row, own: tuple, x: float) -> bool:
    """Whether another obstacle of ``row`` than ``own`` stands at ``x`` m along the way
    with its top at or above ``own``'s."""
    return any(
        other != own
        and other[0] - _SAME_M <= x <= other[1] + _SAME_M
        and other[2] >= own[2] - _SAME_M
        for other in row
    )


def _bend(to_source, to_listener) -> float:
    """The angle (radians) by which the sound turns at an edge, between the directions
    (along the way, up) from the edge to where it comes from and to where it goes."""
    x, z = -to_source[0], -to_source[1]
    return math.atan2(
        x * to_listener[1] - z * to_listener[0], x * to_listener[0] + z * to_listener[1]
    )


@dataclass(frozen=True)
class _Slanted:
    """An edge that the way crosses at a slant, for one wave that reaches it and one
    that goes on from it (:func:`_slanted`)."""

    #: The cosine of the edge's angle from across the way, seen from above: what a step
    #: along the way is across the edge.
    cosine: float
    #: The sine of the angle between the sound's way and the edge.
    sine: float
    #: (F,) the wavenumber times the distance parameter across the edge.
    kl: np.ndarray
    #: (F,) what makes a wave of the section along the way, exp(ik(a + b)) over the
    #: root of a b (a + b), the wave of the edge at its slant.
    turn: np.ndarray


def _slants(obstacles, slants) -> list[tuple[float, float, float, float, float]]:
    """The ``obstacles`` (a list) with their ``slants`` as (near, far, top, the near
    side's angle from across the way, the far side's), the angles as their sizes
    (radians); none where ``slants`` is None. A ValueError where the slants are not one
    for each obstacle, each a real number or a pair of them (not a string, a truth
    value or a complex number), and for a barrier, which has one top, with two slants
    of different sizes."""
    if slants is None:
        return []
    try:
        slants = list(slants)
    except TypeError:
        slants = None
    if slants is None or len(slants) != len(obstacles):
        raise ValueError("one slant for each obstacle (a number, or one for each of its sides)")
    out = []
    for (near, far, h), slant in zip(obstacles, slants, strict=True):
        try:
            given = np.asarray(slant)
        except (TypeError, ValueError):
            given = np.asarray(None)
        # Whole or real numbers alone: numpy would read a string of digits as one.
        if given.dtype.kind not in "iuf" or given.shape not in ((), (2,)):
            raise ValueError("a slant is an angle in radians, or one for each of the two sides")
        sides = np.abs(given.astype(float))
        if not np.all(np.isfinite(sides)):
            raise ValueError("a slant is an angle in radians, or one for each of the two sides")
        near_side, far_side = (float(side) for side in np.broadcast_to(sides, (2,)))
        if abs(far - near) <= _SAME_M and near_side != far_side:
            raise ValueError("a barrier has one top, and one slant")
        out.append((float(near), float(far), float(h), near_side, far_side))
    return out


def _slant_of(turned, what: str, x: float, h: float) -> float:
    """The angle (radians, 0 or more) by which the edge (what, m along the way, top) of
    the row is turned from across the way: that of the side there of the obstacle with
    its top at that height (``turned``, as :func:`_slants` gives them; of several, the
    first given), the far side for a roof's far corner and the near side otherwise.
    :class:`OutOfReach` beyond ``SLANT_MOST_DEG``."""
    far_side = what == "behind"
    for near, far, top, slant_near, slant_far in turned:
        if (far - near > _SAME_M) != (what != "screen") or abs(top - h) > _SAME_M:
            continue
        if abs((far if far_side else near) - x) <= _SAME_M:
            slant = slant_far if far_side else slant_near
            if slant > math.radians(SLANT_MOST_DEG):
                raise OutOfReach(
                    f"an edge {x:.3g} m along the way that is turned "
                    f"{math.degrees(slant):.1f} degrees from across it: the edges' terms "
                    f"take {SLANT_MOST_DEG:.0f} degrees"
                )
            return slant
    raise ValueError(f"no obstacle has a side {x:.6g} m along the way with its top {h:.6g} m up")


def _across(path, cosine: float) -> float:
    """The length (m) of a ``path`` in the vertical plane along the way ((m along it, m
    up) points) as it is across an edge that is turned from across the way by the
    angle with that ``cosine``."""
    return sum(
        math.hypot((b[0] - a[0]) * cosine, b[1] - a[1])
        for a, b in zip(path[:-1], path[1:], strict=True)
    )


def _slanted(
    k: np.ndarray, a: float, b: float, before, after, slant: float, reach: float
) -> _Slanted:
    """What the terms of an edge turned ``slant`` (radians) from across the way take in
    place of the section's (:class:`_Slanted`), for the wave that reaches it over the
    path ``before`` and goes on over the path ``after`` ((m along the way from their
    own end, m up) points; ``a`` and ``b`` m long). The sound that bends at such an
    edge does not stay in the vertical plane along the way: it meets the edge where it
    leaves at the angle to the edge at which it came (Keller's cone), and its way is
    the shortest from end to end over the edge. Across the edge the two paths are
    ``rho_s`` and ``rho_l`` long, and along it the ends are ``reach`` times the
    slant's sine apart, as if the row's other edges were turned as this one is (for a
    row that is the same all along its edges that is the sound's own way; for edges
    at different slants it is a choice). With beta the angle between that way and the
    edge, the uniform theory's terms are those of the angles across the edge, at the
    distance parameter rho_s rho_l sin(beta) / (rho_s + rho_l), over sin(beta)."""
    cosine = math.cos(slant)
    rho_s, rho_l = _across(before, cosine), _across(after, cosine)
    across = rho_s + rho_l
    whole = math.hypot(across, reach * math.sin(slant))
    sine = across / whole
    # exp(ik whole) over the root of s s' (s + s'), with s = rho / sin(beta) the legs.
    spread = sine * math.sqrt(a * b * (a + b) / (rho_s * rho_l * whole))
    return _Slanted(
        cosine=cosine,
        sine=sine,
        kl=k * rho_s * rho_l * sine / across,
        turn=np.exp(1j * k * (whole - a - b)) * spread,
    )


def _screen(
    k: np.ndarray, kl: np.ndarray, to_source, to_listener, slant: _Slanted | None = None
) -> np.ndarray:
    """(F,) what a rigid thin screen's top sheds beyond the march's screen, for the
    directions (along the way, up) from the top to where the sound comes from and to
    where it goes: the two terms of the reflections off the half-plane's faces, and
    the share of its two terms of the incident wave that the mean of the two marches
    lacks, 1 - cos(bend / 2). With a ``slant`` the half-plane's four terms are those
    of the edge at its slant (:func:`_slanted`), less what the marches' screen sheds
    in the section along the way, cos(bend / 2) of the two terms of the incident wave
    there."""

    def angle(direction, cosine: float = 1.0) -> np.ndarray:
        # Round the top from the screen itself, which hangs down from it.
        return np.asarray(math.atan2(-direction[0] * cosine, -direction[1]) % (2.0 * math.pi))

    terms = wedge_terms(_PLANE, angle(to_listener), angle(to_source), kl)
    if slant is None:
        lacking = 1.0 - math.cos(0.5 * _bend(to_source, to_listener))
        return np.conj(
            wedge_prefactor(_PLANE, k, _ACROSS)
            * (terms[..., 2] + terms[..., 3] + lacking * (terms[..., 0] + terms[..., 1]))
        )
    turned = wedge_terms(
        _PLANE, angle(to_listener, slant.cosine), angle(to_source, slant.cosine), slant.kl
    )
    marched = math.cos(0.5 * _bend(to_source, to_listener))
    return slant.turn * np.conj(
        wedge_prefactor(_PLANE, k, np.asarray(slant.sine)) * turned.sum(axis=-1)
    ) - marched * np.conj(wedge_prefactor(_PLANE, k, _ACROSS) * (terms[..., 0] + terms[..., 1]))


def _corner(
    k: np.ndarray,
    kl: np.ndarray,
    outside,
    along: float,
    inside=None,
    slant: _Slanted | None = None,
) -> np.ndarray:
    """(F,) what a roof's rigid corner sheds beyond the screen the march takes its wall
    for, between a wave on the roof's side and the direction ``outside`` (along the
    way, up) from the corner to where the sound comes from (the near corner: ``along``
    1, the roof goes on along the way) or goes to (the far corner: ``along`` -1). The
    wave on the roof's side runs along the roof, where the march brings it or takes
    it on (``inside`` None): the two terms of the incident wave of the wedge open over
    three right angles, which along a rigid face are half its field. Or it is the own
    wave of an end over the roof or of the end's image in the roof, from the corner in
    the direction ``inside`` (below the roof's plane: the image): the wedge's two
    terms of that wave, those of the incident wave for the end and those of the
    reflection off the roof for its image. Less what the march's screen sheds of the
    wave: cos(bend / 2) of the two terms of the incident wave of the half-plane that
    hangs down from the corner. With a ``slant`` the wedge's terms are those of the
    corner at its slant (:func:`_slanted`); what the march's screen sheds stays that
    of the section along the way."""

    def own(cosine: float, distance: np.ndarray) -> tuple[np.ndarray, float, float]:
        # Round the corner from the roof, through the air.
        phi = math.atan2(outside[1], along * outside[0] * cosine) % (2.0 * math.pi)
        # The wave on the roof's side, above the roof (its image in the roof: below).
        raised = 0.0 if inside is None else math.atan2(inside[1], along * inside[0] * cosine)
        wedge = wedge_terms(_CORNER, np.asarray(abs(raised)), np.asarray(phi), distance)
        pair = wedge[..., 0] + wedge[..., 1] if raised >= 0.0 else wedge[..., 2] + wedge[..., 3]
        return pair, phi, raised

    section, phi, raised = own(1.0, kl)
    between = phi - raised
    # The half-plane's terms of the incident wave are of the angle between the two.
    down = 1.5 * math.pi  # the wall, round the corner from the roof
    low = max(down - between, 0.0)
    high = down if low > 0.0 else between
    plane = wedge_terms(_PLANE, np.asarray(high), np.asarray(low), kl)
    # The sound turns by that angle less a straight one.
    shed = math.cos(0.5 * (between - math.pi))
    marched = shed * wedge_prefactor(_PLANE, k, _ACROSS) * (plane[..., 0] + plane[..., 1])
    if slant is None:
        return np.conj(wedge_prefactor(_CORNER, k, _ACROSS) * section - marched)
    turned = own(slant.cosine, slant.kl)[0]
    return slant.turn * np.conj(
        wedge_prefactor(_CORNER, k, np.asarray(slant.sine)) * turned
    ) - np.conj(marched)


__all__ = ["OutOfReach", "Row", "edges_of", "layer_top", "over_row", "skyline", "string"]
