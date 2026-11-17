"""The solver for whole rows among a simulation's early paths.

Where barriers and buildings stand between a source and a listener outdoors, the sound
in the vertical plane through the two is, among the early paths, the direct sound, its
reflections off the ground and the roofs, and the paths over the tops that cross the
plane (:mod:`~acoustic_spaces.engine.diffraction`, over one edge or two). Those do not
add up to the field where the sound passes close over several edges in a row, and a path
over two edges has no copies off the ground. The solver for whole rows
(:func:`~acoustic_spaces.engine.rows.over_row`) gives that plane's sound as one field,
with the ground and the roofs in it. :class:`RowPaths` puts it among a listener's early
paths: in the bands the solver holds (the octave bands up to ``MAX_BAND_HZ`` that add
their paths in pressure), the row's paths in the plane give way to one path that carries
the solver's pressure at the early paths' frequencies, along the string pulled taut over
the tops from the source to the listener. The row's paths are those that bend over
nothing but tops that cross the plane between the two and reflect off nothing but the
ground, the roofs and the walls that the way crosses between the two: sound that goes to
and fro between those walls gives way too, though the solver, which marches one way, has
none of it. The other paths stay as they are: round the ends of barriers and the corners
of buildings, off walls behind the source or past the listener or beside the way, over
the model file's objects, and through a barrier or a wall that lets sound through. The
solver's barriers and roofs are rigid, whatever their materials absorb or let through,
and its ground has each zone's admittance. Its air is still: in wind
and temperature gradients its path is one of the plane's, whose levels the weather's
effect behind the barriers and buildings scales
(:meth:`~acoustic_spaces.engine.refracted.Refraction._barrier`).

The solver's row is the same all along its edges, and each of its edges crosses the way
at the slant of the side that the way meets
(:meth:`~acoustic_spaces.geometry.outdoor.Layout.crossed_along`). That holds where every
top goes on to either side of the way further than the paths over an edge follow its
end: ``diffraction.FADE_REACH`` in the argument of the Fresnel integral along the edge,
at each band's lowest frequency. Closer to an end (of a top that the way crosses, or of
one that stops short of the way) the solver's sound and the paths in the plane are added
in pressure, the solver's by the share that the end's taper leaves
(:func:`~acoustic_spaces.engine.diffraction.fade_window`) and the paths by the rest; up
to ``diffraction.FADE_TAPER`` from the end the paths stand alone, as they do without the
solver. So the level changes smoothly where the way passes an end, and a top that ends
beside the way is the edges' own there.

The paths stand as they are, too, wherever the solver does not take the row
(:class:`~acoustic_spaces.engine.rows.OutOfReach`, or an end that stands against an
obstacle), band by band, and where nothing is asked of it: nothing on the way, an end
inside a building, the listener on a barrier's line or a building's wall, an object of
the model file in the straight line between the two, or a roof on the way whose material
has an impedance.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

from acoustic_spaces.engine import rows
from acoustic_spaces.engine.diffraction import _COLLINEAR_DEG, _fade_reach, fade_window
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel, source_weights
from acoustic_spaces.engine.section import _LAID, GROUND_STEP_M, ROW, Section, _at, _kind
from acoustic_spaces.geometry.outdoor import in_polygon
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, octave_of

#: The highest octave band (centre, Hz) in which the solver's sound takes the place of
#: the paths in the plane: the solver is measured against wave calculations up to it
#: (docs/validation.md, "Rows of roofs and barriers against a wave calculation").
MAX_BAND_HZ = 500
#: The choices of ``simulation.early.rows``: the solver for whole rows, or the paths
#: over the edges alone.
ROWS = ("solver", "edges")


class RowPaths(Section):
    """The solver for whole rows for the early paths of the source at ``source`` in an
    outdoor scene (``model``), whose paths add in pressure in the lowest
    ``coherent`` bands: ``speed_of_sound`` (m/s), ``air`` (bands,, the air's
    attenuation of energy per metre), the source's ``directivity`` and ``orientation``
    as the early stage takes them, and the model's diffracting ``edges`` as the paths
    over them number them (found when needed if not given)."""

    def __init__(
        self,
        model: TraceModel,
        source: np.ndarray,
        *,
        speed_of_sound: float,
        air: np.ndarray,
        directivity=None,
        orientation: np.ndarray | None = None,
        coherent: int = 0,
        edges=None,
    ) -> None:
        super().__init__(model, source, edges)
        self.c = float(speed_of_sound)
        self.air = np.asarray(air, dtype=float)
        self.directivity = directivity
        self.orientation = orientation
        self.coherent = int(coherent)
        octave = octave_of(model.bands)
        held = np.arange(self.bands) < self.coherent
        held &= np.asarray(OCTAVE_BANDS_HZ)[octave] <= MAX_BAND_HZ
        #: The bands the solver gives, each octave band's together (one run).
        self.groups = [
            np.flatnonzero(held & (octave == index)) for index in np.unique(octave[held])
        ]
        self._tops: tuple[np.ndarray, np.ndarray] | None = None
        self._upright: tuple[np.ndarray, np.ndarray] | None = None
        #: How many runs (a listener's octave bands) the solver took, and how many it
        #: refused, so far.
        self.taken = 0
        self.refused = 0

    @classmethod
    def of(
        cls, model: TraceModel, source: np.ndarray, mode: str, max_order: int = 1, **settings
    ) -> RowPaths | None:
        """The solver for the source at ``source``, with ``mode`` "solver"
        (``simulation.early.rows``; "edges" leaves the paths over the edges); None where
        it has nothing to give: indoors, over a terrain, in a scene without barriers and
        buildings, with paths that add in pressure in no band the solver holds, and
        with early paths of no order (``max_order`` 0: the direct sound alone)."""
        if mode not in ROWS:
            raise ValueError(f"rows is one of {', '.join(ROWS)}")
        layout = model.room.outdoor
        if mode != "solver" or layout is None or max_order < 1:
            return None
        if layout.terrain is not None or not (layout.barriers or layout.buildings):
            return None
        found = cls(model, source, **settings)
        return found if found.groups else None

    def apply(self, listener: np.ndarray, paths: EarlyReflections) -> EarlyReflections:
        """``paths``, the early paths from the source to ``listener`` (3,) with the
        edges' own among them, with the solver's sound in the vertical plane through
        the two where the solver takes it (see the module's notes)."""
        if paths.spectrum is None or not len(paths):
            return paths
        listener = np.asarray(listener, dtype=float)
        across = listener[:2] - self.source[:2]
        reach = float(np.linalg.norm(across))
        if reach <= 1e-6:
            return paths
        towards = across / reach
        found = self._row(listener, towards, reach)
        if found is None:
            return paths
        obstacles, slants = found
        plane = self._plane(paths, towards, reach)
        if not len(plane):
            return paths
        share = self._share(listener, towards, reach)
        zr = max(float(listener[2]) - self.ground, 0.0)
        pressure = np.zeros(self.frequencies.shape, dtype=complex)
        for group in self.groups:
            if not np.any(share[group] > 0.0):
                continue
            f = self.frequencies[group].ravel()
            ground = self._ground(group, towards, reach)
            try:
                row = rows.over_row(f, self.c, ground, self.zs, reach, zr, obstacles, slants=slants)
            except ValueError:
                # Out of the solver's reach, or an end against an obstacle: the paths.
                self.refused += 1
                share[group] = 0.0
                continue
            self.taken += 1
            # The early paths' time is exp(j omega t), the solver's exp(-i omega t).
            pressure[group] = np.conj(row.pressure).reshape(len(group), -1)
        if not np.any(share > 0.0):
            return paths
        carrier = self._carrier(paths, share, pressure, listener, towards, reach, zr, obstacles)
        out = paths[np.arange(len(paths))]
        out.energy[plane] *= ((1.0 - share) ** 2)[None, :]
        return EarlyReflections.concatenate([out, carrier], self.bands)

    # -- the row ----------------------------------------------------------------------

    def _row(self, listener: np.ndarray, towards: np.ndarray, reach: float):
        """What stands on the way to ``listener`` (``reach`` m along ``towards``, seen
        from above) as the solver takes it: the obstacles ((near, far, top), m along
        the way and above the ground), a building under the source from its wall behind
        the source and one under the listener to its wall past the listener, and each
        one's slants (None where every side stands across the way). None where the
        solver is not asked: nothing on the way short of the listener, the listener on
        a barrier's line or a building's wall (``section.ON_OBSTACLE_M``), the source
        or the listener inside a building (over its footprint, not above its roof), an
        object of the model file in the straight line between the two, or a roof on the
        way with an impedance (:meth:`_soft_roof`)."""
        layout = self.layout
        x0, y0, x1, y1 = layout.extent
        # Out past the listener to the ground's edge: the building a listener stands on
        # keeps its far wall where it is.
        edge = max(reach, math.hypot(x1 - x0, y1 - y0))
        found = [
            (round(near, 9), round(far, 9), round(top - self.ground, 9), sides)
            for near, far, top, sides in layout.crossed_along(self.source[:2], towards, edge)
            if top > self.ground and near < reach
        ]
        if not found or _at(tuple(o[:3] for o in found), reach):
            return None
        ends = np.array([self.source, listener])
        for _, footprint, roof in layout.buildings:
            if np.any(in_polygon(ends[:, :2], footprint) & (ends[:, 2] <= roof + 1e-9)):
                return None
        if self._blocked(listener) or self._soft_roof(found, towards):
            return None
        if any(near == 0.0 for near, _, _, _ in found):
            # The buildings under the source, back to their walls behind it.
            behind = [
                (round(far, 9), round(top - self.ground, 9), sides[1])
                for near, far, top, sides in layout.crossed_along(self.source[:2], -towards, edge)
                if near == 0.0 and top > self.ground
            ]
            for n, (near, far, top, sides) in enumerate(found):
                back = [o for o in behind if near == 0.0 and far > near and o[1] == top]
                if back:
                    found[n] = (-back[0][0], far, top, (back[0][2], sides[1]))
        obstacles = [o[:3] for o in found]
        # A side that the way does not cross stands past an end, where it does nothing.
        slants = [tuple(0.0 if math.isnan(side) else side for side in o[3]) for o in found]
        if all(side == 0.0 for sides in slants for side in sides):
            return obstacles, None
        return obstacles, [
            sides[0] if far == near else sides
            for (near, far, _), sides in zip(obstacles, slants, strict=True)
        ]

    def _soft_roof(self, found: list, towards: np.ndarray) -> bool:
        """Whether a roof among ``found`` (the obstacles on the way along ``towards``,
        as :meth:`_row` finds them) has a material with an impedance, where the way
        crosses its middle: the solver's roofs are rigid."""
        middles = np.array(
            [
                [*(self.source[:2] + 0.5 * (near + far) * towards), self.ground + top + 0.01]
                for near, far, top, _ in found
                if far > near
            ]
        )
        if not len(middles):
            return False
        down = np.broadcast_to(np.array([0.0, 0.0, -1.0]), middles.shape)
        _, face = self.model.intersect(
            middles, down, np.full(len(middles), 0.02), np.full(len(middles), -1)
        )
        return bool(np.any(self.model.impedance_faces(face[face >= 0])))

    def _plane(self, paths: EarlyReflections, towards: np.ndarray, reach: float) -> np.ndarray:
        """The paths (indices) whose sound the solver's stands for, those of the row in
        the vertical plane through the source and the listener: the paths that reflect
        off nothing but the ground, the buildings' roofs and the row's own walls
        (:meth:`_walls`), which are the direct sound and its specular reflections, and
        the paths over edges that all cross the plane between the two
        (:meth:`~acoustic_spaces.engine.section.Section._in_plane`). None that passes
        through a surface that lets sound through: the solver's barriers let none
        through, so that sound is beside its own."""
        over = cast(np.ndarray, paths.edge)
        through = cast(np.ndarray, paths.crossings)
        walls = self._walls(towards, reach)
        level = self.flat | self._roof_faces() | walls
        plane = []
        for i in range(len(paths)):
            if through[i] > 0:
                continue
            if over[i] >= 0:
                if self._in_plane(paths, i, towards, reach, walls):
                    plane.append(i)
            elif over[i] == -1:
                faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
                if level[faces].all():
                    plane.append(i)
        return np.array(plane, dtype=np.int64)

    def _walls(self, towards: np.ndarray, reach: float) -> np.ndarray:
        """(faces,) bool: the row's walls, the upright faces of the barriers and
        buildings that the way crosses between the source and the listener, seen from
        above. Sound that goes to and fro between them is the row's too."""
        if self._upright is None:
            mesh = self.model.room.mesh
            laid = np.array([_kind(label) in _LAID for label in mesh.labels] or [False])
            normal = self.model.plane_normal[self.model.face_plane]
            index = np.flatnonzero(laid[mesh.face_labels] & (np.abs(normal[:, 2]) < 1e-9))
            # Seen from above, an upright face is a stretch of its plane's line.
            corners = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces)[index]]
            along = np.stack([-normal[index, 1], normal[index, 0]], axis=1)
            t = np.einsum("wkj,wj->wk", corners[:, :, :2], along)
            row = np.arange(len(index))
            ends = np.stack(
                [corners[row, t.argmin(axis=1), :2], corners[row, t.argmax(axis=1), :2]], axis=1
            )
            self._upright = (index, ends)
        index, ends = self._upright
        out = np.zeros(len(self.model.face_plane), dtype=bool)
        if len(index):
            a, b = ends[:, 0] - self.source[:2], ends[:, 1] - self.source[:2]
            left = np.array([-towards[1], towards[0]])
            side_a, side_b = a @ left, b @ left
            crossed = side_a * side_b < 0.0
            # Where the stretch meets the way's line, m from the source.
            share = side_a / np.where(crossed, side_a - side_b, 1.0)
            at = a @ towards + ((b - a) @ towards) * share
            out[index[crossed & (at > 0.0) & (at < reach)]] = True
        return out

    def _top_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """The tops: the edges (indices) of the barriers and buildings that are not
        vertical, and (T, 2) whether each goes on past its start and past its end, in
        another top in line with it (within ``diffraction._COLLINEAR_DEG`` degrees) or
        as its image in a face that closes it
        (:func:`~acoustic_spaces.engine.diffraction.end_fades`)."""
        if self._tops is None:
            from scipy.spatial import cKDTree

            edges = self._edge_set()
            index = np.flatnonzero(
                self._laid_edges() & (np.abs(edges.direction[:, 2]) < 1.0 - 1e-9)
            )
            count = len(index)
            onward = np.zeros(2 * count, dtype=bool)
            if count:
                points = np.concatenate([edges.start[index], edges.end[index]])
                outward = np.concatenate([-edges.direction[index], edges.direction[index]])
                which = np.concatenate([np.arange(count), np.arange(count)])
                size = float(np.ptp(self.model.room.mesh.vertices, axis=0).max()) or 1.0
                pairs = cKDTree(points).query_pairs(1e-6 * size, output_type="ndarray")
                a, b = pairs[:, 0], pairs[:, 1]
                turn = -np.einsum("ij,ij->i", outward[a], outward[b])
                straight = (which[a] != which[b]) & (turn >= math.cos(math.radians(_COLLINEAR_DEG)))
                onward[a[straight]] = True
                onward[b[straight]] = True
            onward = onward.reshape(2, count).T.copy()
            if count and edges.closed is not None:
                onward |= edges.closed[index]
            self._tops = (index, onward)
        return self._tops

    def _share(self, listener: np.ndarray, towards: np.ndarray, reach: float) -> np.ndarray:
        """(bands,) how much of the solver's sound each band takes at ``listener``: 1 in
        the solver's bands where every top whose line the way crosses between the two
        goes on, from the point where the shortest way over its line touches it,
        further than the paths over an edge follow an end
        (:func:`~acoustic_spaces.engine.diffraction._fade_reach` at the band's lowest
        frequency) on both sides, or stops that far short of that point; less within
        that reach of an end, by the end's taper
        (:func:`~acoustic_spaces.engine.diffraction.fade_window`), and 0 up to where
        the taper begins. An end from which the top goes on (:meth:`_top_edges`) is
        none. 0 in the other bands."""
        share = np.zeros(self.bands)
        held = np.concatenate(self.groups)
        share[held] = 1.0
        index, onward = self._top_edges()
        if not len(index):
            return share
        edges = self._edge_set()
        start, along, length = edges.start[index], edges.direction[index], edges.length[index]
        # Seen from above, where the way crosses each top's line.
        e, w = along[:, :2], start[:, :2] - self.source[:2]
        denominator = towards[0] * e[:, 1] - towards[1] * e[:, 0]
        usable = np.abs(denominator) > 1e-12 * np.maximum(np.linalg.norm(e, axis=1), 1e-300)
        at = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(usable, denominator, 1.0)
        crossed = usable & (at > 0.0) & (at < reach)

        def frame(point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """How far along each top's line ``point`` lies, and how far from it."""
            offset = point[None, :] - start
            t = np.einsum("ij,ij->i", offset, along)
            return t, np.linalg.norm(offset - t[:, None] * along, axis=1)

        (ts, rho_s), (tr, rho_r) = frame(self.source), frame(listener)
        both = rho_s + rho_r
        crossed &= both > 0.0
        # Where the shortest way from the source over the line to the listener touches it.
        t = ts + (tr - ts) * rho_s / np.where(both > 0.0, both, 1.0)
        r1, r2 = np.hypot(rho_s, t - ts), np.hypot(rho_r, tr - t)
        sine = both / np.maximum(r1 + r2, 1e-300)
        k = 2.0 * np.pi * self.frequencies[held].min(axis=1) / self.c
        for n in np.flatnonzero(crossed):
            followed = _fade_reach(r1[n], r2[n], sine[n], k)  # (held,)
            for end, distance in ((0, t[n]), (1, length[n] - t[n])):
                if not onward[n, end]:
                    share[held] *= 1.0 - fade_window(abs(distance), followed)
        return share

    def _ground(self, group: np.ndarray, towards: np.ndarray, reach: float):
        """The ground along the way as the solver takes it, at the frequencies of the
        bands ``group``: a function of the distance from the source (m) that gives its
        normalised impedance (time as exp(-i omega t); infinite where its admittance is
        0), by the ground's face there, looked up every ``section.GROUND_STEP_M``; None
        where it is rigid all along."""
        face_at = self._grounds_along(towards, reach)
        tables: dict[int, np.ndarray | None] = {}

        def table(face: int) -> np.ndarray | None:
            if face not in tables:
                beta = self._admittance(face)[group].ravel()
                rigid = beta == 0.0
                tables[face] = (
                    None
                    if rigid.all()
                    else np.where(rigid, np.inf, 1.0 / np.conj(np.where(rigid, 1.0, beta)))
                )
            return tables[face]

        steps = np.arange(0.0, reach + GROUND_STEP_M, GROUND_STEP_M)
        if all(table(face) is None for face in {face_at(float(r)) for r in steps}):
            return None
        return lambda r: table(face_at(float(r)))

    # -- the path that carries the solver's sound -------------------------------------

    def _carrier(
        self,
        paths: EarlyReflections,
        share: np.ndarray,
        pressure: np.ndarray,
        listener: np.ndarray,
        towards: np.ndarray,
        reach: float,
        zr: float,
        obstacles: list,
    ) -> EarlyReflections:
        """The path that carries the solver's sound: ``share`` (bands,) of its
        ``pressure`` (bands, M; time as exp(j omega t), free field exp(-jkR) / R) at
        the early paths' frequencies, as a path's energy and spectrum, along the
        string pulled taut over the tops from the source to ``listener``. The air
        takes its energy over the string's length, and the source's directivity is
        that towards the string's first bend."""
        tops = sorted(
            (x, h) for _, x, h in rows.edges_of(rows.skyline(obstacles)) if 1e-9 < x < reach - 1e-9
        )
        string = rows.string([(0.0, self.zs), *tops, (reach, zr)])
        points = np.array([[*(self.source[:2] + x * towards), self.ground + z] for x, z in string])
        points[0], points[-1] = self.source, listener
        legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
        distance = float(legs.sum())
        delay = distance / self.c
        departure = (points[1] - points[0]) / max(float(legs[0]), 1e-300)
        arrival = (points[-2] - points[-1]) / max(float(legs[-1]), 1e-300)
        weight = source_weights(
            self.directivity,
            self.orientation,
            departure[None],
            self.bands,
            distances=np.array([float(legs[0])]),
        )[0]
        level = np.sqrt(np.mean(np.abs(pressure) ** 2, axis=1))  # (bands,)
        energy = (share * level) ** 2 / (4.0 * np.pi) * np.exp(-self.air * distance) * weight
        coherent = paths.coherent_bands
        turn = np.exp(2j * np.pi * self.frequencies[:coherent] * delay)
        heard = (level[:coherent] > 0.0)[:, None]
        spectrum = np.where(
            heard, pressure[:coherent] * turn / np.where(heard, level[:coherent, None], 1.0), 1.0
        )
        return EarlyReflections(
            order=np.zeros(1, dtype=np.int64),
            delay_s=np.array([delay]),
            distance_m=np.array([distance]),
            direction=arrival[None],
            energy=energy[None],
            points=[points],
            faces=[np.zeros(0, dtype=np.int64)],
            edge=np.array([ROW], dtype=np.int64),
            spectrum=spectrum[None],
        )


__all__ = ["MAX_BAND_HZ", "ROW", "ROWS", "RowPaths"]
