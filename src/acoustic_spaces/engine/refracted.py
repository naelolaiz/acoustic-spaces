"""Early paths outdoors in wind and temperature that change with height.

The early paths are found as in still air (:mod:`~acoustic_spaces.engine.early` and
:mod:`~acoustic_spaces.engine.diffraction`), then bent by the atmosphere
(:class:`~acoustic_spaces.scene.Atmosphere`), whose effective speed of sound depends on
the height and on the way the sound goes (:mod:`~acoustic_spaces.engine.refraction`):

- Over the open ground (the direct sound and its reflections off the ground), the
  paths are the eigenrays from the source to the listener, which may bounce off the
  ground several times (downwind, or under an inversion) or not reach the listener at
  all (upwind, or over ground warmed by the sun: the shadow zone).
- Off vertical walls (buildings, barriers), the same in the vertical planes the wall
  reflections fold the path into: unfolded, the path is one plane, along which the wind
  blows with its mean component over the legs. A ray that passes over the wall, or
  through something on the way, is left out.
- Over an edge, and off any other surface, each leg keeps its straight course and
  takes the travel time of its own heights, and its level falls into a shadow as a
  creeping wave's: by exp(-2 Im kappa_1 d) over the distance d past the shadow's
  start, kappa_1 the first mode of the residue series over a rigid ground
  (:mod:`~acoustic_spaces.engine.shadow`).

Rays fail at low frequencies over the ground: the profiles change fastest within a
wavelength of it, where the direct sound and its reflection meet. In the bands up to
:data:`WAVE_MAX_BAND_HZ`, the sound over the open ground is therefore that of the
parabolic equation (:mod:`~acoustic_spaces.engine.parabolic`) in the vertical plane
through the source and the listener, over the grounds along it, when nothing stands in
the way: the first of the eigenrays carries it, the others nothing in those bands. In the
shadow zone, where no ray arrives, the parabolic equation gives the sound in the bands
up to :data:`SHADOW_WAVE_MAX_BAND_HZ`, and above it the shadow deepens with frequency as
it does over the octave below.

The late stage's rays bend too: the compiled tracer follows each one along its arcs
through the layers, in chords (:func:`bending` gives it the atmosphere).

Where the weather changes from place to place (the atmosphere's ``zones``,
:mod:`~acoustic_spaces.engine.weather`), the rays take the mean profile along their
way, the parabolic equation marches through each weather in turn, and the late rays
take the mean weather over the ground.
"""

from __future__ import annotations

import dataclasses
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.engine.coherent import ground_factors, reflection_phases
from acoustic_spaces.engine.early import _EDGE_TOLERANCE, EarlyReflections, _face_at
from acoustic_spaces.engine.model import TraceModel, _ground_faces, source_weights
from acoustic_spaces.engine.parabolic import footprint, march
from acoustic_spaces.engine.refraction import CHORD_SAGITTA_M, Profile, Rays, _sine, node_heights
from acoustic_spaces.engine.shadow import equivalent_radius, shadow_start
from acoustic_spaces.engine.turbulence import Eddies
from acoustic_spaces.engine.weather import Weather

#: Bands up to this one (centre, Hz) take the sound over the open ground from the
#: parabolic equation.
WAVE_MAX_BAND_HZ = 500
#: In the shadow zone, the parabolic equation's bands go up to this one.
SHADOW_WAVE_MAX_BAND_HZ = 2000
#: Rays come out too loud on the way into the shadow (by 5 dB at 1 kHz three quarters of
#: the way there, upwind in 5 m/s of wind): from this share of the way, the parabolic
#: equation's bands go up to ``SHADOW_WAVE_MAX_BAND_HZ`` as in the shadow.
PENUMBRA = 0.5
#: The parabolic equation applies from this many times the source's and the listener's
#: heights (added) away along the ground, where the sound comes in at 34 degrees from
#: the horizontal or flatter (within 0.2 dB of the exact solution there): nearer, it
#: comes in too steeply for it, and the atmosphere has had little room to bend it.
WAVE_MIN_RANGE = 1.5
#: The parabolic equation's absorbing layer starts this many times higher than the
#: source, the listener and the rays between them, at least ``WAVE_MIN_HEADROOM_M``
#: above the source and the listener and at least ``WAVE_TOP_SLOPE`` times the range up
#: (higher makes no difference: 0.1 dB at 800 m downwind, at 125 Hz, 5 m/s of wind).
WAVE_HEADROOM = 1.5
WAVE_MIN_HEADROOM_M = 20.0
WAVE_TOP_SLOPE = 0.15
#: The directions sound goes in are rounded to this many degrees from the wind's: the
#: rays and the parabolic equation in one serve all within it.
AZIMUTH_STEP_DEG = 1.0
#: The parabolic equation's runs go side by side in at most this much memory (bytes).
WAVE_MEMORY_BYTES = 1 << 30
#: With weather zones, the weathers' shares along a way are rounded to this, so that
#: ways alike share their rays (a share of 0.005 of a 5 m/s wind is 0.025 m/s).
MIX_STEP = 0.005
#: The grounds along the parabolic equation's way are looked up this often (m).
GROUND_STEP_M = 0.25
#: The most times a curved path bounces off the ground.
MAX_GROUND_BOUNCES = 10
#: The first root of Ai' (the first creeping mode over a rigid ground).
_FIRST_MODE = -1.0187929716474710
#: Normals more horizontal than this (|n_z|) are vertical walls.
_VERTICAL = 1e-9


#: How far (m/s) the late tracer's profile may stray from the atmosphere's, in the speed
#: of sound plus the wind's: it drops the nodes it can (each layer is a piece of arc to
#: follow), keeping those where the profile bends.
LATE_SPEED_TOLERANCE = 0.005
#: Rays kept for reuse (about 4 MB each), from the most recently used: the points of a
#: line or surface source share them, and the listeners round a source.
_KEPT_RAYS = 24
_RAYS: OrderedDict = OrderedDict()


@dataclass
class _Wave:
    """A listener's request for the parabolic equation."""

    listener: int
    key: tuple  # the runs it shares: the profile's key, or the direction with weather zones
    mix: object  # the key of the profile of the rays along its way
    direction: np.ndarray  # (2,) along the ground towards the listener
    range: float
    height: float
    top: float
    bands: np.ndarray  # (bands,) bool: those it gives
    shadow: bool
    distance: float
    departure: np.ndarray  # (3,) the direction it leaves the source in
    ratio: np.ndarray  # (bands,) what the edges did to the direct sound
    carrier: int  # the path of the family that carries it, or -1 (shadow)
    family: np.ndarray  # (n,) the family's paths
    delay: float


class Refraction:
    """Bends the early paths of the source at ``source`` in an outdoor scene's
    atmosphere (not still), running the parabolic equation on ``threads`` threads (0:
    every core)."""

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
        max_order: int = 1,
        threads: int = 0,
    ) -> None:
        from acoustic_spaces.engine.diffraction import band_frequencies

        room = model.room
        self.model = model
        self.atmosphere = room.scene.atmosphere
        self.ground_c = float(room.scene.air.temperature_c)
        self.source = np.asarray(source, dtype=float)
        self.ground = float(model.ground)
        self.zs = max(float(self.source[2]) - self.ground, 0.0)
        self.c = float(speed_of_sound)
        self.air = np.asarray(air, dtype=float)
        self.directivity = directivity
        self.orientation = orientation
        self.coherent = int(coherent)
        self.max_order = int(max_order)
        self.bands = model.absorption.shape[1]
        self.flat = _ground_faces(room)
        self.tolerance = _EDGE_TOLERANCE * float(np.ptp(room.mesh.vertices, axis=0).max())
        self.frequencies = band_frequencies(bands=model.bands)[0]  # (bands, M)
        self.centres = np.asarray(model.band_hz, dtype=float)
        z = node_heights(self.atmosphere)
        self._nodes = z
        self._still_speed = np.asarray(self.atmosphere.sound_speed(z, self.ground_c), dtype=float)
        self._wind_speed = np.asarray(self.atmosphere.wind_speed(z), dtype=float)
        self.windy = bool(np.any(self._wind_speed != 0.0))
        self.wind = self.atmosphere.to_wind()
        self.zones = room.outdoor is not None and bool(room.outdoor.zones)
        # The weather where it changes from place to place; None where it is one.
        self.weather = Weather.of(model) if self.atmosphere.zones else None
        self.zoned = self.weather is not None
        if self.zoned:
            self._nodes = self.weather.nodes
            self.windy = self.windy or bool(np.any(self.weather.wind != 0.0))
        self._weather = (self.atmosphere.model_dump_json(), self.ground_c)
        self._profiles: dict[float, Profile] = {}
        self._admittances: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self.threads = int(threads) or os.cpu_count() or 1
        self._starts: dict[tuple, float | None] = {}
        self.eddies = Eddies.of(model, self.c)

    # -- profiles ----------------------------------------------------------------------

    def _key(self, along: float) -> float:
        """The wind's component along a direction (its cosine to the wind) rounded to
        :data:`AZIMUTH_STEP_DEG`; 0 without wind."""
        if not self.windy:
            return 0.0
        angle = math.degrees(math.acos(min(max(along, -1.0), 1.0)))
        angle = round(angle / AZIMUTH_STEP_DEG) * AZIMUTH_STEP_DEG
        return round(math.cos(math.radians(angle)), 12)

    def _towards(self, direction: np.ndarray) -> float:
        return self._key(float(np.dot(self.wind, direction)))

    def _mix(self, corners) -> tuple:
        """With weather zones, the key of the mean profile along the horizontal polyline
        ``corners`` (m + 1, 2): the weathers' shares of its length and of the wind's
        component along it (:meth:`~acoustic_spaces.engine.weather.Weather.shares`),
        rounded so that ways alike share their rays."""
        a, b = self.weather.shares(np.asarray(corners, dtype=float))
        return (tuple(np.round(a / MIX_STEP) * MIX_STEP), tuple(np.round(b / MIX_STEP) * MIX_STEP))

    def profile(self, key) -> Profile:
        """The profile of ``key``: the wind's cosine to the way (:meth:`_key`), or with
        weather zones the mix of the weathers along it (:meth:`_mix`)."""
        if key not in self._profiles:
            if isinstance(key, tuple):
                self._profiles[key] = self.weather.profile(np.array(key[0]), np.array(key[1]))
            else:
                self._profiles[key] = Profile(
                    self._nodes, self._still_speed + key * self._wind_speed
                )
        return self._profiles[key]

    def rays(self, key: float) -> Rays:
        index = (self._weather, key, self.zs)
        rays = _RAYS.get(index)
        if rays is None:
            rays = _RAYS[index] = Rays(self.profile(key), self.zs)
            while len(_RAYS) > _KEPT_RAYS:
                _RAYS.popitem(last=False)
        else:
            _RAYS.move_to_end(index)
        return rays

    # -- all listeners -----------------------------------------------------------------

    def apply(
        self,
        listeners: np.ndarray,
        straight: list[EarlyReflections],
        paths: list[EarlyReflections],
        cutoffs: np.ndarray,
    ) -> list[EarlyReflections]:
        """The paths to each of ``listeners`` (L, 3) bent by the atmosphere: ``paths``
        as found in still air (the edges' diffraction applied to them), ``straight``
        the specular ones before that, up to ``cutoffs`` (L,) s."""
        listeners = np.atleast_2d(np.asarray(listeners, dtype=float))
        cutoffs = np.broadcast_to(np.asarray(cutoffs, dtype=float), (len(listeners),))
        parts: list = [None] * len(listeners)
        requests: list[_Wave] = []
        # The listeners in turn round the wind, which reuses its rays most.
        across = listeners[:, :2] - self.source[:2]
        turn = np.arctan2(across[:, 1], across[:, 0]) - math.atan2(self.wind[1], self.wind[0])
        for k in np.argsort(np.abs(np.angle(np.exp(1j * turn))), kind="stable").tolist():
            found, request = self._listener(
                k, listeners[k], straight[k], paths[k], float(cutoffs[k])
            )
            parts[k] = found
            if request is not None:
                requests.append(request)
        self._waves(requests, parts, listeners)
        return [
            p if isinstance(p, EarlyReflections) else EarlyReflections.concatenate(p, self.bands)
            for p in parts
        ]

    # -- one listener ------------------------------------------------------------------

    def _listener(self, k, listener, straight, paths, cutoff):
        across = listener[:2] - self.source[:2]
        reach = float(np.linalg.norm(across))
        if reach <= 1e-6:
            # Straight above or below the source, the sound goes up or down unbent.
            return paths, None
        towards = across / reach
        zr = max(float(listener[2]) - self.ground, 0.0)
        ratio = _ratios(straight, paths)
        groups: dict[tuple, list[int]] = {}
        bent: list[int] = []
        for i in range(len(paths)):
            if paths.edge[i] >= 0:
                bent.append(i)
                continue
            faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
            walls = faces[~self.flat[faces]] if len(faces) else faces
            planes = tuple(int(p) for p in self.model.face_plane[walls])
            groups.setdefault(planes, []).append(i)
        groups.setdefault((), [])

        out: list[EarlyReflections] = []
        request = None
        for planes, rows in groups.items():
            vertical = all(abs(float(self.model.plane_normal[p][2])) <= _VERTICAL for p in planes)
            if not vertical:
                bent.extend(rows)
                continue
            if planes == ():
                family, request = self._open_ground(
                    k, listener, towards, reach, zr, straight, paths, rows, ratio, cutoff
                )
                out.insert(0, family)
            elif rows:
                out.append(self._off_walls(listener, zr, paths, rows, planes, ratio, cutoff))
        if bent:
            out.append(self._legs(paths[np.array(sorted(bent))]))
        return out, request

    def _open_ground(self, k, listener, towards, reach, zr, straight, paths, rows, ratio, cutoff):
        """The paths over the open ground, and the parabolic equation's request."""
        key = self._mix([self.source[:2], listener[:2]]) if self.zoned else self._towards(towards)
        rays = self.rays(key)
        cap = MAX_GROUND_BOUNCES if self.max_order >= 1 else 0
        found = rays.eigenrays(zr, reach, cap, max_time=cutoff)
        corners = np.array([self.source[:2], listener[:2]])
        r0, r1 = _family_ratios(paths, rows, ratio, self.flat, self.bands)
        family, visible, highest = self._curved(rays, found, zr, corners, (), listener, r0, r1)
        direct = bool(np.any((straight.order == 0) & (straight.edge < 0)))
        clear = direct and bool(np.all(visible))
        shadow = len(found) == 0
        if not clear or reach < WAVE_MIN_RANGE * (self.zs + zr):
            return family, None
        if not shadow and self.max_order < 1:
            return family, None
        start = self._start(key, self.zs, zr)
        near = shadow or (start is not None and reach >= PENUMBRA * start)
        top_band = SHADOW_WAVE_MAX_BAND_HZ if near else WAVE_MAX_BAND_HZ
        bands = self.centres <= top_band * 1.001
        if not bands.any():
            return family, None
        distance = float(np.linalg.norm(listener - self.source))
        if shadow:
            departure = np.append(towards, 0.0)
            delay = distance / float(self.profile(key).c[0])
            carrier = -1
        else:
            carrier = int(np.argmin(family.delay_s))
            departure = _departure(family.points[carrier])
            delay = float(family.delay_s[carrier])
        # Room above the rays for the sound that spreads up from them: deep into the
        # shadow and at the lowest frequencies, more than the rays reach.
        top = max(
            WAVE_HEADROOM * max(self.zs, zr, highest),
            max(self.zs, zr) + WAVE_MIN_HEADROOM_M,
            WAVE_TOP_SLOPE * reach,
        )
        azimuth = None
        if self.zones or self.zoned:
            angle = math.degrees(math.atan2(towards[1], towards[0]))
            azimuth = round(angle / AZIMUTH_STEP_DEG) * AZIMUTH_STEP_DEG % 360.0
        return family, _Wave(
            listener=k,
            key=(None if self.zoned else key, azimuth),
            mix=key,
            direction=towards,
            range=reach,
            height=zr,
            top=top,
            bands=bands,
            shadow=shadow,
            distance=distance,
            departure=departure,
            ratio=r0,
            carrier=carrier,
            family=np.arange(len(family)),
            delay=delay,
        )

    def _off_walls(self, listener, zr, paths, rows, planes, ratio, cutoff):
        """The paths off the vertical walls ``planes`` (in order), unfolded."""
        # The template: the straight path with the fewest bounces off the ground.
        bounces = [int(np.count_nonzero(self.flat[paths.faces[i]])) for i in rows]
        template = rows[int(np.argmin(bounces))]
        points = paths.points[template]
        faces = paths.faces[template]
        corners = [points[0][:2]]
        for j, face in enumerate(faces):
            if not self.flat[face]:
                corners.append(points[j + 1][:2])
        corners.append(listener[:2])
        corners = np.array(corners)
        legs = np.diff(corners, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        reach = float(lengths.sum())
        if reach <= 1e-6 or np.any(lengths <= 1e-9):
            return paths[np.array(rows)]
        # The wind's component along each leg, weighed by the leg's length.
        if self.zoned:
            key = self._mix(corners)
        else:
            key = self._key(float(self.wind @ legs.sum(axis=0)) / reach)
        rays = self.rays(key)
        allowed = self.max_order - len(planes)
        cap = MAX_GROUND_BOUNCES if allowed >= 1 else 0
        found = rays.eigenrays(zr, reach, cap, max_time=cutoff)
        r0, r1 = _family_ratios(paths, rows, ratio, self.flat, self.bands)
        family, _, _ = self._curved(rays, found, zr, corners, planes, listener, r0, r1)
        return family

    # -- curved paths ------------------------------------------------------------------

    def _curved(self, rays, found, zr, corners, planes, listener, r0, r1):
        """The eigenrays ``found`` as paths along the horizontal ``corners`` (k + 2, 2)
        from the source to the listener, reflecting off the vertical ``planes`` (k,) at
        the inner corners; those blocked or passing a wall by are left out. Also (n,)
        whether each could be drawn, and the greatest height they reach (m)."""
        model = self.model
        legs = np.diff(corners, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        directions = legs / np.maximum(lengths, 1e-300)[:, None]
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        total = cumulative[-1]
        n = len(found)
        visible = np.zeros(n, dtype=bool)
        highest = 0.0
        rows = []
        for i in range(n):
            curve = rays.curve(zr, found, i)
            x, z = curve[:, 0], curve[:, 1]
            if x[-1] > 0.0:
                x = x * (total / x[-1])
            kind = np.zeros(len(x), dtype=np.int64)  # 0: in the air, 1: ground, 2 + j: wall j
            if found.ground[i]:
                low = np.flatnonzero(z <= 1e-9)
                low = low[(low > 0) & (low < len(z) - 1)]
                kind[low] = 1
            if planes:
                at = cumulative[1:-1]
                height = np.interp(at, x, z)
                x = np.concatenate([x, at])
                z = np.concatenate([z, height])
                kind = np.concatenate([kind, 2 + np.arange(len(planes))])
                order = np.argsort(x, kind="stable")
                x, z, kind = x[order], z[order], kind[order]
            segment = np.clip(np.searchsorted(cumulative, x, side="right") - 1, 0, len(legs) - 1)
            xy = corners[segment] + (x - cumulative[segment])[:, None] * directions[segment]
            walls = kind >= 2
            xy[walls] = corners[1 + kind[walls] - 2]
            points = np.column_stack([xy, z + self.ground])
            points[0] = self.source
            points[-1] = listener
            face = np.full(len(points), -1, dtype=np.int64)
            ok = True
            for j in np.flatnonzero(walls):
                face[j] = _face_at(model, planes[kind[j] - 2], points[j], self.tolerance)
                ok &= face[j] >= 0
            ground = np.flatnonzero(kind == 1)
            if len(ground):
                face[ground] = self._ground_face(points[ground])
                ok &= bool(np.all(face[ground] >= 0))
            if not ok:
                continue
            share, crossed = self._passes(points, face)
            if not np.any(share > 0.0):
                continue
            visible[i] = True
            highest = max(highest, float(z.max()))
            rows.append((i, points, face, kind, share, crossed))
        if not rows:
            return EarlyReflections.empty(self.bands), visible, highest
        return self._describe(rays, found, rows, directions, r0, r1), visible, highest

    def _ground_face(self, points: np.ndarray) -> np.ndarray:
        """(n,) the face of the ground under each of ``points`` (n, 3), or -1."""
        start = np.asarray(points, dtype=float) + np.array([0.0, 0.0, 0.01])
        down = np.broadcast_to(np.array([0.0, 0.0, -1.0]), start.shape)
        _, face = self.model.intersect(
            start, down, np.full(len(start), 0.02), np.full(len(start), -1)
        )
        return np.where((face >= 0) & self.flat[np.maximum(face, 0)], face, -1)

    def _passes(self, points: np.ndarray, face: np.ndarray):
        """(bands,) the share of sound that gets along the chords between ``points``,
        reflecting off ``face`` at each (-1: in the air), and how many faces it crosses
        that are not openings."""
        start, end = points[:-1], points[1:]
        delta = end - start
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1e-300)[:, None]
        nudge = np.minimum(1e-6, 0.25 * length)
        share, crossed = self.model.passes(
            start + direction * nudge[:, None],
            direction,
            np.maximum(length - 2.0 * nudge, 0.0),
            face[:-1],
        )
        return np.prod(share, axis=0), int(crossed.sum())

    def _describe(self, rays, found, rows, directions, r0, r1) -> EarlyReflections:
        """The early paths of the eigenrays ``rows`` (index, points, faces, kinds,
        share, crossings) of ``found``."""
        model, bands, coherent = self.model, self.bands, self.coherent
        count = len(rows)
        index = np.array([r[0] for r in rows])
        energy = np.empty((count, bands))
        factor = np.ones((count, *self.frequencies.shape), dtype=complex)
        spectrum = (
            np.ones((count, coherent, self.frequencies.shape[1]), dtype=complex)
            if coherent
            else None
        )
        c0 = float(rays.profile.c[0])
        events = []  # (row, face, cosine, incoming, before, at, after, order)
        for row, (i, points, face, kind, share, _) in enumerate(rows):
            length = float(found.length[i])
            bounced = int(np.count_nonzero(kind == 1))
            ratio = r1 if bounced else r0
            energy[row] = (
                found.spreading[i] / (4.0 * np.pi) * np.exp(-self.air * length) * share * ratio
            )
            at = np.flatnonzero(kind > 0)
            chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
            along = np.concatenate([[0.0], np.cumsum(chords)])
            marks = np.concatenate([[0], at, [len(points) - 1]])
            for order, j in enumerate(at, start=1):
                before, after = marks[order - 1], marks[order + 1]
                incoming = points[j] - points[j - 1]
                outgoing = points[j + 1] - points[j]
                tangent_in = incoming / max(np.linalg.norm(incoming), 1e-300)
                tangent_out = outgoing / max(np.linalg.norm(outgoing), 1e-300)
                if kind[j] == 1:
                    # Exactly: the ray meets the ground at the angle its parameter gives.
                    sine = float(_sine(found.p[i], c0))
                    flat = math.sqrt(max(1.0 - sine * sine, 0.0))
                    horizontal_in = tangent_in[:2] / max(np.linalg.norm(tangent_in[:2]), 1e-300)
                    tangent_in = np.append(flat * horizontal_in, -sine)
                    tangent_out = np.append(flat * horizontal_in, sine)
                    cosine = sine
                else:
                    normal = model.plane_normal[model.face_plane[face[j]]]
                    cosine = abs(float(normal @ tangent_in))
                leg_in = along[j] - along[before]
                leg_out = along[after] - along[j]
                events.append(
                    (
                        row,
                        int(face[j]),
                        cosine,
                        tangent_in * leg_in,
                        points[j] - tangent_in * leg_in,
                        points[j],
                        points[j] + tangent_out * leg_out,
                        order,
                        length,
                    )
                )
        if events:
            row_of = np.array([e[0] for e in events])
            faces = np.array([e[1] for e in events], dtype=np.int64)
            cosines = np.array([e[2] for e in events])
            incoming = np.array([e[3] for e in events])
            orders = np.array([e[7] for e in events])
            lengths = np.array([e[8] for e in events])
            for order in np.unique(orders):
                pick = np.flatnonzero(orders == order)
                kept = model.reflection(faces[pick], incoming[pick], int(order), impedance=True)
                np.multiply.at(energy, row_of[pick], kept)
            if coherent:
                np.multiply.at(spectrum, row_of, reflection_phases(model, faces, cosines, coherent))
            grounds = model.factor_faces(faces)
            if grounds.any():
                pick = np.flatnonzero(grounds)
                ends = tuple(np.array([events[e][m] for e in pick]) for m in (4, 5, 6))
                q = ground_factors(model, faces[pick], cosines[pick], lengths[pick], ends=ends)
                np.multiply.at(factor, row_of[pick], q)
                gain = np.mean(np.abs(factor) ** 2, axis=2)
                energy *= gain
                if coherent:
                    level = np.sqrt(gain[:, :coherent, None])
                    spectrum *= np.where(
                        level > 0.0, factor[:, :coherent] / np.where(level > 0.0, level, 1.0), 1.0
                    )
        if coherent:
            # Each caustic delays the phase by a quarter period.
            spectrum *= (1j ** found.caustics[index])[:, None, None]
        departure = np.array([_departure(r[1]) for r in rows])
        energy *= source_weights(self.directivity, self.orientation, departure, bands)
        arrival = np.array([r[1][-2] - r[1][-1] for r in rows])
        arrival /= np.maximum(np.linalg.norm(arrival, axis=1), 1e-300)[:, None]
        # The ray's own direction at the listener rather than its last chord's.
        lean = found.arrival[index]
        horizontal = (
            arrival[:, :2] / np.maximum(np.linalg.norm(arrival[:, :2], axis=1), 1e-300)[:, None]
        )
        arrival = np.column_stack([horizontal * np.cos(lean)[:, None], -np.sin(lean)])
        return EarlyReflections(
            order=np.array([int(np.count_nonzero(r[3] > 0)) for r in rows], dtype=np.int64),
            delay_s=found.time[index],
            distance_m=found.length[index],
            direction=arrival,
            energy=energy,
            points=[r[1] for r in rows],
            faces=[r[2][r[3] > 0] for r in rows],
            crossings=np.array([r[5] for r in rows], dtype=np.int64),
            spectrum=spectrum,
        )

    # -- legs --------------------------------------------------------------------------

    def _legs(self, paths: EarlyReflections) -> EarlyReflections:
        """``paths`` with each straight leg's travel time in the atmosphere, and its
        shadow's attenuation."""
        energy = paths.energy.copy()
        delay = paths.delay_s.copy()
        for i, points in enumerate(paths.points):
            gain = np.ones(self.bands)
            shift = 0.0
            for a, b in zip(points[:-1], points[1:], strict=True):
                g, t = self._leg(a, b)
                gain *= g
                shift += t
            energy[i] *= gain
            delay[i] += shift
        return dataclasses.replace(paths, energy=energy, delay_s=delay)

    def _leg(self, a: np.ndarray, b: np.ndarray):
        """The energy factor (bands,) and the extra travel time (s) of a straight leg
        from ``a`` to ``b``."""
        delta = b - a
        distance = float(np.linalg.norm(delta))
        across = float(np.linalg.norm(delta[:2]))
        za = max(float(a[2]) - self.ground, 0.0)
        zb = max(float(b[2]) - self.ground, 0.0)
        if self.zoned:
            key = self._mix([a[:2], b[:2]]) if across > 1e-9 else self._mix([a[:2]])
        else:
            key = self._towards(delta[:2] / across) if across > 1e-9 else 0.0
            if across <= 1e-9 and self.windy:
                key = 0.0
        profile = self.profile(key)
        heights = za + (zb - za) * (np.arange(16) + 0.5) / 16.0
        shift = distance * float(np.mean(1.0 / profile.speed(heights))) - distance / self.c
        gain = np.ones(self.bands)
        if across > 1e-9 and za + zb > 0.0:
            start = self._start(key, za, zb)
            if start is not None and across > start:
                radius = equivalent_radius(start, za, zb)
                k = 2.0 * np.pi * self.centres / float(profile.c[0])
                ell = (radius / (2.0 * k**2)) ** (1.0 / 3.0)
                kappa = np.sqrt(k**2 - _FIRST_MODE * np.exp(1j * np.pi / 3.0) / ell**2 + 0j)
                gain = np.exp(-2.0 * np.abs(kappa.imag) * (across - start))
                if self.eddies is not None:
                    scattered = self.eddies.scattered(profile, za, zb, across, self.centres, start)
                    gain = np.minimum(gain + scattered, 1.0)
        return gain, shift

    def _start(self, key: float, za: float, zb: float) -> float | None:
        index = (key, round(za, 3), round(zb, 3))
        if index not in self._starts:
            self._starts[index] = shadow_start(self.profile(key), za, zb)
        return self._starts[index]

    # -- the parabolic equation --------------------------------------------------------

    def _admittance(self, face: int) -> np.ndarray:
        """(bands, M) the normalised admittance (time as exp(j omega t)) of ``face``: an
        impedance's at grazing incidence (which differs with the extended reaction)."""
        with self._lock:
            if face not in self._admittances:
                model = self.model
                if model.impedance_faces(np.array([face]))[0]:
                    value = model.grazing_admittance[model.face_impedance[face]]
                else:
                    from acoustic_spaces.engine.impedance import fit_admittance

                    value = fit_admittance(model.absorption[face], self.bands, None)(
                        self.frequencies
                    )
                self._admittances[face] = np.asarray(value, dtype=complex)
            return self._admittances[face]

    def _grounds_along(self, direction: np.ndarray, reach: float):
        """A function giving the face of the ground (its material) at a distance along
        ``direction`` from the source, up to ``reach``."""
        layout = self.model.room.outdoor
        here = self._ground_face(np.array([[*self.source[:2], self.ground]]))[0]
        if here < 0:
            here = int(np.flatnonzero(self.flat)[0])
        if not self.zones or self.model.ground_part is None:
            return lambda r: here
        # The grounds every GROUND_STEP_M along the way, looked up once.
        x0, y0, x1, y1 = layout.extent
        along = np.arange(0.0, reach + 2.0 * GROUND_STEP_M, GROUND_STEP_M)
        points = np.clip(self.source[:2] + along[:, None] * direction, [x0, y0], [x1, y1])
        found = self.model.part_face[layout.zone_at(points) + 1]
        faces = np.where(found < 0, here, found)
        last = len(faces) - 1
        return lambda r: int(faces[min(int(round(r / GROUND_STEP_M)), last)])

    def _waves(self, requests: list[_Wave], parts: list, listeners: np.ndarray) -> None:
        """Runs the parabolic equation for ``requests`` and puts its sound into the
        families (the first of each listener's ``parts``)."""
        groups: dict[tuple, list[_Wave]] = {}
        for request in requests:
            groups.setdefault(request.key, []).append(request)
        runs = []
        for (key, azimuth), members in groups.items():
            if azimuth is None:
                direction = members[0].direction
            else:
                angle = math.radians(azimuth)
                direction = np.array([math.cos(angle), math.sin(angle)])
            reach = max(m.range for m in members)
            face_at = self._grounds_along(direction, reach)
            if self.zoned:
                speed_at = self.weather.along(self.source[:2], direction, reach)
                speed = speed_at(0.0)
            else:
                speed_at = None
                speed = self.profile(key).speed
            top = max(m.top for m in members)
            for band in range(self.bands):
                needing = [m for m in members if m.bands[band]]
                if needing:
                    runs.append((speed, speed_at, face_at, band, needing, top))

        def run(job) -> np.ndarray:
            speed, speed_at, face_at, band, needing, top = job

            def impedance(r):
                beta = self._admittance(face_at(r))[band]
                with np.errstate(divide="ignore"):
                    return np.where(beta == 0.0, np.inf, 1.0 / np.conj(beta))

            ranges = np.array([m.range for m in needing])
            heights = np.array([m.height for m in needing])
            return march(
                self.frequencies[band],
                speed,
                impedance,
                self.zs,
                ranges,
                heights,
                top,
                speed_at=speed_at,
            )

        # The runs side by side, as many as the threads and the memory allow.
        biggest = max(
            (footprint(self.frequencies[job[3]], self.c, job[5]) for job in runs), default=0
        )
        workers = min(self.threads, len(runs), max(1, WAVE_MEMORY_BYTES // max(biggest, 1)))
        if workers > 1:
            with ThreadPoolExecutor(workers) as pool:
                results = list(pool.map(run, runs))
        else:
            results = [run(job) for job in runs]
        pressure = {id(m): np.zeros(self.frequencies.shape, dtype=complex) for m in requests}
        for (_, _, _, band, needing, _), values in zip(runs, results, strict=True):
            for m, value in zip(needing, values.T, strict=True):
                # Time as exp(j omega t), as the early paths have it.
                pressure[id(m)][band] = np.conj(value)
        for members in groups.values():
            for m in members:
                self._place(m, pressure[id(m)], parts, listeners)

    def _place(self, wave: _Wave, pressure: np.ndarray, parts: list, listeners: np.ndarray):
        bands = wave.bands
        excess = np.mean(np.abs(pressure) ** 2, axis=1) * wave.distance**2  # (bands,)
        if wave.shadow:
            # Above the parabolic equation's bands the shadow deepens as it does over the
            # octave below: by the same factor per octave.
            computed = np.flatnonzero(bands)
            last = computed[-1]
            octave = np.flatnonzero(
                np.abs(self.centres - self.centres[last] / 2.0) < 1e-6 * self.centres[last]
            )
            slope = 1.0
            if len(octave) and excess[octave[0]] > 0.0:
                slope = min(excess[last] / excess[octave[0]], 1.0)
            for band in range(last + 1, self.bands):
                octaves = math.log2(self.centres[band] / self.centres[last])
                excess[band] = excess[last] * slope**octaves
            bands = np.ones(self.bands, dtype=bool)
            if self.eddies is not None:
                # What the eddies scatter into the shadow.
                key = wave.mix
                start = self._start(key, self.zs, wave.height)
                if start is not None:
                    excess += self.eddies.scattered(
                        self.profile(key), self.zs, wave.height, wave.range, self.centres, start
                    )
        weight = source_weights(
            self.directivity, self.orientation, wave.departure[None], self.bands
        )[0]
        energy = (
            excess
            / (4.0 * np.pi * wave.distance**2)
            * np.exp(-self.air * wave.distance)
            * weight
            * wave.ratio
        )
        coherent = self.coherent
        spectrum = None
        if coherent:
            level = np.sqrt(np.mean(np.abs(pressure[:coherent]) ** 2, axis=1))[:, None]
            turn = np.exp(2j * np.pi * self.frequencies[:coherent] * wave.delay)
            spectrum = np.where(
                level > 0.0, pressure[:coherent] * turn / np.where(level > 0.0, level, 1.0), 1.0
            )
            computed = wave.bands[:coherent]
            spectrum[~computed] = 1.0
        found = parts[wave.listener]
        family = found[0]
        if self.eddies is not None and wave.carrier >= 0:
            energy = self._blurred(energy, family[wave.family], bands)
        if wave.carrier < 0:
            listener = listeners[wave.listener]
            arrival = self.source - listener
            arrival /= max(float(np.linalg.norm(arrival)), 1e-300)
            path = EarlyReflections(
                order=np.zeros(1, dtype=np.int64),
                delay_s=np.array([wave.delay]),
                distance_m=np.array([wave.distance]),
                direction=arrival[None],
                energy=np.where(bands, energy, 0.0)[None],
                points=[np.vstack([self.source, listener])],
                faces=[np.zeros(0, dtype=np.int64)],
                spectrum=None if spectrum is None else spectrum[None],
            )
            found[0] = EarlyReflections.concatenate([family, path], self.bands)
            return
        others = wave.family[wave.family != wave.carrier]
        family.energy[others[:, None], np.flatnonzero(bands)[None]] = 0.0
        family.energy[wave.carrier, bands] = energy[bands]
        if coherent:
            low = np.flatnonzero(bands[:coherent])
            family.spectrum[wave.carrier, low] = spectrum[low]
            family.spectrum[others[:, None], low[None]] = 1.0

    def _blurred(self, energy: np.ndarray, rays: EarlyReflections, bands: np.ndarray):
        """The parabolic equation's ``energy`` (bands,) in still air, in the ``bands``
        it gives, as the eddies blur it: E = C E_pe + (1 - C) E_rays, E_rays the
        family's ``rays`` added in energy and C their coherence factor, weighted by
        sqrt(E_i E_j) over each two of them."""
        if len(rays) < 2:
            return energy
        wander = self.eddies.wander(rays)
        out = energy.copy()
        off = ~np.eye(len(rays), dtype=bool)
        for band in np.flatnonzero(bands):
            alone = rays.energy[:, band]
            weights = np.sqrt(np.outer(alone, alone))[off]
            if weights.sum() <= 0.0:
                continue
            together = wander.coherence(self.frequencies[band]).mean(axis=2)[off]
            share = float(np.sum(together * weights) / weights.sum())
            out[band] = share * energy[band] + (1.0 - share) * alone.sum()
        return out


def _departure(points: np.ndarray) -> np.ndarray:
    """The unit vector a path along ``points`` leaves its first point in."""
    delta = points[1] - points[0]
    return delta / max(float(np.linalg.norm(delta)), 1e-300)


def _ratios(straight: EarlyReflections, paths: EarlyReflections) -> dict[tuple, np.ndarray]:
    """What the edges' diffraction did to each specular path's energy (bands,), by its
    faces."""
    before = {}
    for i in range(len(straight)):
        if straight.edge[i] < 0:
            before[tuple(int(f) for f in straight.faces[i])] = straight.energy[i]
    out = {}
    for i in range(len(paths)):
        if paths.edge[i] >= 0:
            continue
        key = tuple(int(f) for f in paths.faces[i])
        old = before.get(key)
        if old is None:
            continue
        out[key] = np.where(old > 0.0, paths.energy[i] / np.where(old > 0.0, old, 1.0), 1.0)
    return out


def _family_ratios(paths, rows, ratio, flat, bands):
    """The edges' ratios of a family's straight paths without a bounce off the ground and
    with one (1 where there is none)."""
    r0 = r1 = None
    for i in rows:
        key = tuple(int(f) for f in paths.faces[i])
        value = ratio.get(key)
        if value is None:
            continue
        if np.any(flat[np.array(key, dtype=np.int64)]) if key else False:
            r1 = value if r1 is None else r1
        else:
            r0 = value if r0 is None else r0
    one = np.ones(bands)
    r0 = one if r0 is None else r0
    r1 = r0 if r1 is None else r1
    return r0, r1


def _thinned(rows: np.ndarray, tolerance: float) -> np.ndarray:
    """Which of the profile's ``rows`` (height, speed in still air, wind speed) to keep
    for the speed and the wind, linear between the rows kept, to stay within
    ``tolerance`` (m/s, the two errors added) at every row dropped: the first, the last
    and as few between as that allows, each layer as tall as it can be."""
    z, still, wind = rows.T
    keep = [0]
    i, last = 0, len(z) - 1
    while i < last:
        j = i + 1
        while j < last:
            k = j + 1
            t = (z[i + 1 : k] - z[i]) / (z[k] - z[i])
            off = np.abs(still[i] + t * (still[k] - still[i]) - still[i + 1 : k]).max()
            off += np.abs(wind[i] + t * (wind[k] - wind[i]) - wind[i + 1 : k]).max()
            if off > tolerance:
                break
            j = k
        keep.append(j)
        i = j
    return np.array(keep)


def bending(model: TraceModel) -> dict:
    """The arguments that bend the rays of ``_native.trace_stochastic``, ``trace_map``
    and ``bent_chords`` in an outdoor scene's atmosphere: its profile (rows of height
    above the ground, speed of sound in still air and wind speed), the way the wind
    blows, the ground's height and how far a chord may stray from its arc. Empty where
    sound goes straight: indoors, or outdoors in still air."""
    scene = model.room.scene
    atmosphere = getattr(scene, "atmosphere", None)
    if model.room.outdoor is None or atmosphere is None or atmosphere.still:
        return {}
    if atmosphere.zones:
        # The late rays sweep the whole scene: the mean weather over it, by area.
        rows, towards = Weather.of(model).mean_rows()
    else:
        z = node_heights(atmosphere)
        ground_c = float(scene.air.temperature_c)
        rows = np.column_stack([z, atmosphere.sound_speed(z, ground_c), atmosphere.wind_speed(z)])
        towards = atmosphere.to_wind()
    rows = rows[_thinned(rows, LATE_SPEED_TOLERANCE)]
    return {
        "atmosphere": np.ascontiguousarray(rows, dtype=np.float64),
        "wind_direction": np.ascontiguousarray(towards, dtype=np.float64),
        "ground": float(model.ground),
        "chord_sagitta": CHORD_SAGITTA_M,
    }


def direct_times(times: np.ndarray, paths: list[EarlyReflections]) -> np.ndarray:
    """(n,) when the direct sound reaches each listener along its bent path: the earliest
    of its ``paths`` that neither reflects nor goes round an edge, sooner downwind and
    later upwind than along the straight line; ``times``, the straight line's, where
    there is none (behind a building)."""
    out = np.array(times, dtype=float)
    for k, found in enumerate(paths):
        clear = (found.order == 0) & ~found.diffracted
        if clear.any():
            out[k] = float(found.delay_s[clear].min())
    return out


def refraction_for(model: TraceModel, source: np.ndarray, **kwargs) -> Refraction | None:
    """The :class:`Refraction` of the source at ``source``, or None where sound goes
    straight: indoors, or outdoors in still air."""
    scene = model.room.scene
    atmosphere = getattr(scene, "atmosphere", None)
    if model.room.outdoor is None or atmosphere is None or atmosphere.still:
        return None
    return Refraction(model, source, **kwargs)


__all__ = [
    "AZIMUTH_STEP_DEG",
    "MAX_GROUND_BOUNCES",
    "SHADOW_WAVE_MAX_BAND_HZ",
    "WAVE_MAX_BAND_HZ",
    "WAVE_MIN_RANGE",
    "Refraction",
    "bending",
    "direct_times",
    "refraction_for",
]
