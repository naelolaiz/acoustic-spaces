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

Rays fail over the ground: the profiles change fastest within a wavelength of it, where
the direct sound and its reflection meet. In the bands within the octave bands up to
:data:`WAVE_MAX_BAND_HZ`, the sound over the open ground is therefore that of the
parabolic equation (:mod:`~acoustic_spaces.engine.parabolic`) in the vertical plane
through the source and the listener, over the grounds along it, when nothing stands in
the way: the first of the eigenrays carries it, the others nothing in those bands. In
the shadow zone, where no ray arrives, the shadow deepens above those bands with
frequency as it does over the octave below. ``simulation.early.open_ground_frequencies``
sets how many frequencies an octave the runs take at 1 and 2 kHz, or leaves those bands
to the rays short of half the way to the shadow's start (:data:`LIT_WAVE_MAX_BAND_HZ`,
:data:`PENUMBRA`).

Where the listener's way, seen from above, crosses a barrier or a building
(:meth:`Refraction._barrier`), the paths in the vertical plane through the source and
the listener keep their levels in still air, and in every band the weather changes them
by the ratio of the parabolic equation's sound marched through it to that marched
through still air, both with the barriers and buildings on the way inside (the
``obstacles`` of :func:`~acoustic_spaces.engine.parabolic.march`), along the whole
degrees either side of the way (:meth:`Refraction._cuts_for`). The paths round a
barrier's end or a building's corner alone, where one of them reflects off the ground,
take the open ground's ratio along their own way round the edges in its bands
(:meth:`Refraction._ends`).

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
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.engine import parabolic, run_size, wide_angle
from acoustic_spaces.engine.coherent import ground_factors, reflection_phases
from acoustic_spaces.engine.early import _EDGE_TOLERANCE, EarlyReflections, _face_at
from acoustic_spaces.engine.model import TraceModel, _ground_faces, source_weights
from acoustic_spaces.engine.refraction import (
    CHORD_SAGITTA_M,
    Profile,
    Rays,
    _sine,
    node_heights,
    table_heights,
)
from acoustic_spaces.engine.shadow import equivalent_radius, shadow_start
from acoustic_spaces.engine.turbulence import Eddies
from acoustic_spaces.engine.weather import Weather
from acoustic_spaces.geometry.outdoor import GROUND, SKY, in_polygon, line_crossings
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, band_limits, octave_of

#: The bands within the octave bands up to this one (centre, Hz) take the sound over the
#: open ground from the parabolic equation, in and out of the shadow (with
#: ``simulation.early.open_ground_frequencies: rays`` only up to
#: ``LIT_WAVE_MAX_BAND_HZ`` short of ``PENUMBRA`` of the way to the shadow's start): in
#: a third-octave run the three third octaves of each (to 2.5 kHz). At 1 and 2 kHz, 50
#: to 300 m from sources 0.5 to 5 m up, rays were up to 3.2 dB off a fast field program
#: in the ``windy`` preset downwind and 8.5 dB in ``clear_night``, and stepped by up to
#: 14 dB a centimetre at a caustic, where the parabolic equation is within 0.1 dB of it;
#: on the way into the shadow they come out too loud (by 5 dB at 1 kHz three quarters of
#: the way there, upwind in 5 m/s of wind).
WAVE_MAX_BAND_HZ = 2000
#: The parabolic equation applies from this many times the source's and the listener's
#: heights (added) away along the ground, where the sound comes in at 34 degrees from
#: the horizontal or flatter (where its starting field is within 0.2 dB of the exact
#: field over a rigid ground): nearer, it comes in too steeply for it, and the
#: atmosphere has had little room to bend it.
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
#: ways alike share their rays (a share of 0.005 of a 5 m/s wind is 0.025 m/s), and
#: divided by their sum (:meth:`Refraction.profile`).
MIX_STEP = 0.005
#: The grounds along the parabolic equation's way are looked up this often (m).
GROUND_STEP_M = 0.25
#: The most times a curved path bounces off the ground.
MAX_GROUND_BOUNCES = 10
#: Behind barriers and buildings the parabolic equation runs in every band with them
#: inside (:meth:`Refraction._barrier`). From this band (centre, Hz) up its height step
#: is BARRIER_COARSE times the solver's own: a quarter wavelength in the narrow-angle
#: equation (a tenth below it), which costs 2 to 2.5 times less and changes the
#: weather's effect by up to 0.2 dB at 1 to 8 kHz behind barriers and a building over
#: grass and rigid ground, 25 to 150 m away, but up to 0.3 dB at 1 and 2 kHz and 2.2 dB
#: at 4 and 8 kHz 11 m behind a top 60 degrees up from the source
#: (``validation/barrier_wind.py --only convergence``).
BARRIER_COARSE_FROM_HZ = 1000
BARRIER_COARSE = 0.4
#: From each band (centre, Hz) up, those runs take this many frequencies an octave (8
#: below, as everywhere). The ground's interference fringes behind a barrier are dense
#: in frequency: at 8 kHz 16 an octave left the weather's effect off by up to 1.8 dB
#: over rigid ground 25 to 100 m away and 6.4 dB where the fringes fell in step with the
#: frequencies (27 degrees up from the source, 25 m away; a one-off check); 32 are within
#: 0.18 dB of 64 there and over grass, but 0.51 dB 11 m behind a top 60 degrees up, and
#: at 4 kHz 16 are within 0.21 dB of 32 (``validation/barrier_wind.py --only
#: convergence``).
BARRIER_SAMPLES = ((4000, 16), (8000, 32))
#: Over the open ground (and along the ways round ends) the runs in the octave bands
#: from this one (centre, Hz) up take ``OPEN_DENSE`` times the early paths' frequencies
#: (an odd number, so that theirs are among them, :func:`_early_of`), with
#: ``simulation.early.open_ground_frequencies: 24``, the default (8 and ``rays`` take
#: theirs): far downwind the ground's interference fringes are dense in frequency, and
#: in the benchmark's case 2 1 km away the 1 kHz band of 8 an octave was 2.2 dB below
#: the fast field program's (``validation/benchmark_1995.py``).
OPEN_DENSE_FROM_HZ = 1000
OPEN_DENSE = 3
#: With ``simulation.early.open_ground_frequencies: rays`` the open ground's parabolic
#: equation takes the bands within the octave bands up to this one (centre, Hz), and the
#: rays the bands above, where a ray reaches the listener and the listener is short of
#: ``PENUMBRA`` of the way to the shadow's start; in the shadow and from there on, the
#: bands up to ``WAVE_MAX_BAND_HZ``, at the early paths' frequencies.
LIT_WAVE_MAX_BAND_HZ = 500
#: Rays come out too loud on the way into the shadow (by 5 dB at 1 kHz three quarters of
#: the way there, upwind in 5 m/s of wind): with ``rays``, from this share of the way the
#: parabolic equation's bands go up to ``WAVE_MAX_BAND_HZ`` as in the shadow.
PENUMBRA = 0.5


def _early_of(values: np.ndarray, count: int) -> np.ndarray:
    """``values`` (..., F) at a band's F frequencies, at the ``count`` of them that are
    the early paths': with F an odd multiple of ``count``, both laid out by
    :func:`~acoustic_spaces.engine.diffraction.band_frequencies`, the middle one of each
    run of F / count."""
    step = values.shape[-1] // count
    return values[..., step // 2 :: step]


def barrier_samples(at_8k: int = 32) -> tuple[tuple[int, int], ...]:
    """``BARRIER_SAMPLES`` with ``at_8k`` frequencies an octave at 8 kHz (8, 16 or 32,
    ``simulation.early.barrier_frequencies``) and at 4 kHz no more: 8 takes the early
    paths' frequencies in every band."""
    return tuple((start, min(n, at_8k)) for start, n in BARRIER_SAMPLES if min(n, at_8k) > 8)


#: Their absorbing layer starts at least this many times higher than the highest top
#: on the way.
BARRIER_HEADROOM = 1.5
#: A listener this close (m) to a barrier's line or a building's wall, seen from above,
#: stands on it: the parabolic equation takes it past it, where nothing below its top
#: gets through, so the paths over it keep straight legs there.
ON_OBSTACLE_M = 1e-6
#: A run along a whole degree counts fully for a listener behind barriers while the last
#: of them along it stands at least this share as far before the listener as along its
#: own way, and less as it comes nearer (:meth:`Refraction._cuts_for`).
BARRIER_CLOSE = 0.5
#: Over this share of the step between two whole degrees, round the halfway direction,
#: a listener behind barriers takes both runs' effects, in dB, weighed linearly by its
#: direction (the nearest alone elsewhere).
BARRIER_BLEND = 0.5
#: The first root of Ai' (the first creeping mode over a rigid ground).
_FIRST_MODE = -1.0187929716474710
#: The least weather's effect (energy ratio) a run behind barriers gives, before its log.
_TINY = 1e-300
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


@dataclass
class _Cut:
    """One of the directions whose runs of the parabolic equation behind barriers and
    buildings give a listener the weather's effect: along ``azimuth`` with what stands
    there (``cut``), in the weather and in still air, their ratio counting by
    ``weight``."""

    key: object  # the profile's key along that direction (None with weather zones)
    azimuth: float  # degrees from the x axis
    cut: tuple  # (near, far, top above the ground), as parabolic.march takes them
    weight: float
    range: float
    height: float
    top: float  # the runs' absorbing layer
    bands: np.ndarray  # (bands,) bool: the bands they run in (all)


@dataclass
class _Barrier:
    """A listener's request for the parabolic equation behind barriers and buildings:
    the runs of its ``cuts``, whose ratios (weighed) scale the paths in its vertical
    plane."""

    listener: int
    part: int  # of the listener's parts, the paths in the plane
    cuts: list[_Cut]
    shift: float  # the time the weather adds to the paths in the plane (s)
    scattered: np.ndarray  # (bands,) what the eddies scatter in, over each path's energy


@dataclass
class _Way:
    """Paths round the same ends of barriers and corners of buildings: the open ground's
    runs of the parabolic equation along their way, unfolded round those edges, in the
    weather and in still air, whose ratio scales them in ``bands``."""

    rows: np.ndarray  # theirs, of the listener's part of paths round ends
    key: object  # the profile's key along the way
    ground: tuple | None  # with ground zones, its grounds (:meth:`Refraction._grounds_round`)
    range: float  # the way's length seen from above
    height: float
    top: float  # the runs' absorbing layer
    bands: np.ndarray  # (bands,) bool
    shift: float  # the delay (s) its paths took on, the first of them's along its legs


@dataclass
class _Ends:
    """A listener's paths round the ends of barriers and corners of buildings, by way."""

    listener: int
    part: int  # of the listener's parts
    ways: list[_Way]


class Refraction:
    """Bends the early paths of the source at ``source`` in an outdoor scene's
    atmosphere (not still), running the parabolic equation on ``threads`` threads (0:
    every core): the wide-angle one (:mod:`~acoustic_spaces.engine.wide_angle`) with
    ``wide_angle``. ``edges``: the model's diffracting edges, as the paths over them
    number them (found when needed if not given). ``profiles``: the profiles of the
    model's air found so far by their key (:meth:`profile`), which the sources of one
    model may share, as none depends on the source; its own if not given."""

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
        wide_angle: bool = False,
        edges=None,
        barrier_frequencies: int = 32,
        open_ground_frequencies: int | str = 24,
        profiles: dict | None = None,
    ) -> None:
        from acoustic_spaces.engine.diffraction import band_frequencies

        self.barrier_samples = barrier_samples(barrier_frequencies)
        #: The open ground's runs from ``OPEN_DENSE_FROM_HZ`` up take this many times the
        #: early paths' frequencies (``simulation.early.open_ground_frequencies``: 24 an
        #: octave, else theirs); with ``rays`` their bands end below 1 kHz short of
        #: ``PENUMBRA`` of the way to the shadow's start (``LIT_WAVE_MAX_BAND_HZ``).
        self.open_dense = OPEN_DENSE if open_ground_frequencies == 24 else 1
        self.open_rays = open_ground_frequencies == "rays"
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
        self._rows = table_heights(self.atmosphere)
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
            self._rows = self.weather.rows
            self.windy = self.windy or bool(np.any(self.weather.wind != 0.0))
        self._weather = (self.atmosphere.model_dump_json(), self.ground_c)
        self._profiles: dict[float, Profile] = {} if profiles is None else profiles
        #: The frequencies of each band at so many an octave (:meth:`_samples`).
        self._dense: dict[int, list] = {}
        self._admittances: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self.threads = run_size.worker_threads(int(threads))
        self.wide_angle = bool(wide_angle)
        self._starts: dict[tuple, float | None] = {}
        self.eddies = Eddies.of(model, self.c)
        self.layout = room.outdoor
        self._cuts: dict[float, tuple] = {}
        self._dense: dict[tuple, np.ndarray] = {}
        self._objects: np.ndarray | None = None
        self._edges = edges
        self._laid: np.ndarray | None = None

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
        weather zones the mix of the weathers along it (:meth:`_mix`), its shares divided
        by their sum: rounded one by one, they can add up to 1 +/- 0.005 or further off,
        which would scale the still air's speed of sound by as much."""
        if key not in self._profiles:
            if isinstance(key, tuple):
                a, b = np.array(key[0]), np.array(key[1])
                total = float(a.sum())
                self._profiles[key] = self.weather.profile(a / total, b / total)
            else:
                self._profiles[key] = Profile(
                    self._nodes, self._still_speed + key * self._wind_speed, self._rows
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
        planned = self.plan(listeners, straight, paths, cutoffs)
        return planned.finish(solve([planned])[0])

    def plan(
        self,
        listeners: np.ndarray,
        straight: list[EarlyReflections],
        paths: list[EarlyReflections],
        cutoffs: np.ndarray,
    ) -> Planned:
        """:meth:`apply` up to the parabolic equation: the paths bent so far and what the
        listeners ask of the equation, which :func:`solve` runs (for several sources at
        once, sharing what they can) and :meth:`Planned.finish` puts in."""
        listeners = np.atleast_2d(np.asarray(listeners, dtype=float))
        cutoffs = np.broadcast_to(np.asarray(cutoffs, dtype=float), (len(listeners),))
        parts: list = [None] * len(listeners)
        requests: list[_Wave] = []
        barriers: list[_Barrier] = []
        ends: list[_Ends] = []
        # The listeners in turn round the wind, which reuses its rays most.
        across = listeners[:, :2] - self.source[:2]
        turn = np.arctan2(across[:, 1], across[:, 0]) - math.atan2(self.wind[1], self.wind[0])
        for k in np.argsort(np.abs(np.angle(np.exp(1j * turn))), kind="stable").tolist():
            found, asked = self._listener(k, listeners[k], straight[k], paths[k], float(cutoffs[k]))
            parts[k] = found
            for request in asked:
                if isinstance(request, _Barrier):
                    barriers.append(request)
                elif isinstance(request, _Ends):
                    ends.append(request)
                else:
                    requests.append(request)
        return Planned(self, listeners, parts, requests, barriers, ends)

    def _sharing(self) -> tuple | None:
        """What another source's runs of the parabolic equation must have in common with
        this one's for the two to share them: none with weather or ground zones, where
        the way from the source matters; otherwise the source's height, the ground under
        it and the solver, the weather being the model's."""
        if self.zoned or self.zones:
            return None
        return (
            id(self.model),
            self.zs,
            self.ground,
            self.wide_angle,
            self._admittance(self._here()).tobytes(),
        )

    # -- one listener ------------------------------------------------------------------

    def _listener(self, k, listener, straight, paths, cutoff):
        across = listener[:2] - self.source[:2]
        reach = float(np.linalg.norm(across))
        if reach <= 1e-6:
            # Straight above or below the source, the sound goes up or down unbent.
            return paths, []
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
        # In line with an end or a corner, the way on the side the paths in still air
        # are on (:meth:`_edge_side`).
        way = self._edge_side(listener, towards, reach)
        barrier = self._barrier(k, listener, way, reach, zr, paths, groups[()], bent, cutoff)
        request = None
        for planes, rows in groups.items():
            vertical = all(abs(float(self.model.plane_normal[p][2])) <= _VERTICAL for p in planes)
            if not vertical:
                bent.extend(rows)
                continue
            if planes == ():
                if barrier is not None:
                    continue
                family, request = self._open_ground(
                    k, listener, towards, reach, zr, straight, paths, rows, ratio, cutoff
                )
                out.insert(0, family)
            elif rows:
                out.append(self._off_walls(listener, zr, paths, rows, planes, ratio, cutoff))
        if barrier is not None:
            plane, request = barrier
            taken = set(plane)
            bent = [i for i in bent if i not in taken]
            out.insert(0, paths[np.array(plane)])
            request.part = 0
        asked = [] if request is None else [request]
        # The paths over a barrier's top past its end, where the edge fades them out,
        # take the weather's effect behind it just inside the end (:meth:`_past_ends`).
        if barrier is None and bent:
            for rows, past in self._past_ends(k, listener, zr, paths, bent, cutoff, reach):
                taken = set(rows)
                bent = [i for i in bent if i not in taken]
                out.append(paths[np.array(rows)])
                past.part = len(out) - 1
                asked.append(past)
        # The paths round the ends of barriers and the corners of buildings take the
        # weather's effect on the open ground along their own way (:meth:`_ends`).
        ends = None
        if request is not None and bent:
            ends = self._ends(k, listener, zr, paths, bent, cutoff, towards)
        if ends is not None:
            chosen, high, low, request_ends = ends
            taken = set(chosen)
            bent = [i for i in bent if i not in taken]
        if bent:
            out.append(self._legs(paths[np.array(sorted(bent))]))
        if ends is not None:
            if high is not None:
                out.append(high)
            out.append(low)
            request_ends.part = len(out) - 1
            asked.append(request_ends)
        return out, asked

    def _open_ground(self, k, listener, towards, reach, zr, straight, paths, rows, ratio, cutoff):
        """The paths over the open ground, and the parabolic equation's request."""
        key = self._mix([self.source[:2], listener[:2]]) if self.zoned else self._towards(towards)
        rays = self.rays(key)
        cap = MAX_GROUND_BOUNCES if self.max_order >= 1 else 0
        found = rays.eigenrays(zr, reach, cap, max_time=cutoff)
        corners = np.array([self.source[:2], listener[:2]])
        r0, r1 = _family_ratios(paths, rows, ratio, self.flat, self.bands)
        # The paths over edges are those of the straight geometry, and their terms make
        # up for the straight paths that are there or not: the direct ray and the one
        # off the ground once are there where those are (:meth:`_curved`).
        twins = self._twins(paths, rows) if np.any(paths.edge >= 0) else None
        family, visible, highest = self._curved(
            rays, found, zr, corners, (), listener, r0, r1, twins=twins
        )
        direct = bool(np.any((straight.order == 0) & (straight.edge < 0)))
        clear = direct and bool(np.all(visible))
        shadow = len(found) == 0
        if not clear or reach < WAVE_MIN_RANGE * (self.zs + zr):
            return family, None
        if not shadow and self.max_order < 1:
            return family, None
        top, bands = self._open_run(key, zr, reach, highest, shadow)
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
        azimuth = self._azimuth(towards) if self.zones or self.zoned else None
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

    def _twins(self, paths: EarlyReflections, rows) -> dict[int, tuple]:
        """Of the straight paths ``rows`` of ``paths`` (off nothing but the ground), the
        one with no bounce off the ground (the direct sound) and the one with one, by
        bounces, each as the points and faces (-1: in the air) :meth:`_passes` takes."""
        out: dict[int, tuple] = {}
        for i in rows:
            faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
            bounces = int(np.count_nonzero(self.flat[faces])) if len(faces) else 0
            if bounces <= 1 and bounces not in out:
                face = np.concatenate([[-1], np.asarray(faces, dtype=np.int64), [-1]])
                out[bounces] = (np.asarray(paths.points[i], dtype=float), face)
        return out

    def _curved(self, rays, found, zr, corners, planes, listener, r0, r1, twins=None):
        """The eigenrays ``found`` as paths along the horizontal ``corners`` (k + 2, 2)
        from the source to the listener, reflecting off the vertical ``planes`` (k,) at
        the inner corners; those blocked or passing a wall by are left out. With
        ``twins`` (:meth:`_twins`), a ray with no bounce off the ground or one is there
        where the straight path with as many is, and only there, the edges' terms making
        up for the straight paths: blocked on its own way, it takes the share of sound
        that gets along the straight one. Also (n,) whether each could be drawn, and the
        greatest height they reach (m)."""
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
            bounces = int(np.count_nonzero(kind == 1))
            twin = None
            if twins is not None and bounces <= 1:
                twin = twins.get(bounces)
                if twin is None:
                    continue
            share, crossed = self._passes(points, face)
            if twin is not None and not np.any(share > 0.0):
                share, crossed = self._passes(*twin)
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
        # Each path's reflections in order, with the cosines of those whose phase is
        # their face's absorption's (EarlyReflections.cosines).
        own = [np.full(int(np.count_nonzero(r[3] > 0)), np.nan) for r in rows]
        if events:
            row_of = np.array([e[0] for e in events])
            faces = np.array([e[1] for e in events], dtype=np.int64)
            cosines = np.array([e[2] for e in events])
            plain = ~model.factor_faces(faces)
            for row in np.unique(row_of):
                mine = row_of == row
                own[row] = np.where(plain[mine], cosines[mine], np.nan)
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
            # Each caustic advances the phase by a quarter period: a factor j, time going
            # as exp(j omega t), in which a delay tau is exp(-j omega tau).
            spectrum *= (1j ** found.caustics[index])[:, None, None]
        departure = np.array([_departure(r[1]) for r in rows])
        travelled = np.array([found.length[r[0]] for r in rows])
        energy *= source_weights(
            self.directivity, self.orientation, departure, bands, distances=travelled
        )
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
            cosines=own if coherent else [],
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

    # -- behind barriers and buildings ---------------------------------------------------

    def _barrier(self, k, listener, towards, reach, zr, paths, open_rows, edge_rows, cutoff):
        """Behind barriers and buildings in the weather: the paths in the vertical plane
        through the source and the listener (``open_rows``, those off the ground alone,
        and those of ``edge_rows`` over edges across the plane, see :meth:`_in_plane`)
        keep their level in still air, which the weather changes by the ratio of the
        parabolic equation's sound in it to that in still air, both with the barriers
        and buildings on the way inside (:meth:`_cuts_for`, :meth:`_place_barrier`).
        Returns those paths (indices) and the request; None where the way ``reach``
        along ``towards`` from the source, seen from above (the listener's, or the way
        :meth:`_edge_side` gives in line with an end or a corner), crosses no barrier or
        building, an object of the model file stands in the listener's straight line,
        the source or the listener stands on a building, the listener is nearer than
        the open ground's parabolic equation goes (``WAVE_MIN_RANGE``), stands on a
        barrier's line or a building's wall seen from above (``ON_OBSTACLE_M``), the
        paths have no order, or none lies in the plane."""
        layout = self.layout
        if not layout.barriers and not layout.buildings:
            return None
        if self.max_order < 1 or reach < WAVE_MIN_RANGE * (self.zs + zr):
            return None
        own = self._above(layout.obstacles_along(self.source[:2], towards, reach))
        if not own or _at(own, reach):
            return None
        ends = np.array([self.source[:2], listener[:2]])
        if any(in_polygon(ends, footprint).any() for _, footprint, _ in layout.buildings):
            return None
        if self._blocked(listener):
            return None
        plane = list(open_rows)
        plane += [i for i in edge_rows if self._in_plane(paths, i, towards, reach)]
        if not plane:
            return None
        key = self._mix([self.source[:2], listener[:2]]) if self.zoned else self._towards(towards)
        # The absorbing layer as over the open ground, above every ray that reaches the
        # listener there (as on the open side of an end, where none is blocked), and
        # above what stands on the way.
        rays = self.rays(key)
        found = rays.eigenrays(zr, reach, MAX_GROUND_BOUNCES, max_time=cutoff)
        open_top, _ = self._open_run(key, zr, reach, _highest(rays, found, zr), len(found) == 0)
        cuts = self._cuts_for(towards, reach, zr, own, open_top)
        # The paths in the plane arrive as much sooner or later as the first of them.
        first = paths.points[min(plane, key=lambda i: float(paths.delay_s[i]))]
        shift = sum(self._leg(a, b)[1] for a, b in zip(first[:-1], first[1:], strict=True))
        start = self._start(key, self.zs, zr)
        scattered = np.zeros(self.bands)
        if self.eddies is not None and start is not None and reach > start:
            scattered = self.eddies.scattered(
                self.profile(key), self.zs, zr, reach, self.centres, start
            )
        request = _Barrier(listener=k, part=0, cuts=cuts, shift=shift, scattered=scattered)
        return plane, request

    def _edge_side(self, listener: np.ndarray, towards, reach: float) -> np.ndarray:
        """The way from the source, seen from above, on the side of a vertical edge of a
        barrier or a building (an end or a corner) that the straight line to
        ``listener`` (``reach`` along ``towards``) passes through, within the tolerance
        the paths over edges are found to: that way turned about the source by twice
        the tolerance at the edge, into what the edge belongs to (the turn whose
        crossings come nearest the edge) where the straight line meets a face in the
        planes of the edge's own faces (by the intersection test the paths in still air
        are checked with, :meth:`_meets`), away from it otherwise, so that the paths in
        still air are those of that side whatever else stands in the way and whichever
        way the way, rounded, falls; the edge nearest the source decides where the line
        touches several. ``towards`` itself where it touches none."""
        from acoustic_spaces.engine.diffraction import _closest

        if not self.layout.barriers and not self.layout.buildings:
            return towards
        edges = self._edge_set()
        upright = np.flatnonzero(self._laid_edges() & (np.abs(edges.direction[:, 2]) >= 1.0 - 1e-9))
        if not len(upright):
            return towards
        start = np.broadcast_to(self.source, (len(upright), 3))
        end = np.broadcast_to(np.asarray(listener, dtype=float), (len(upright), 3))
        points = _closest(start, end, edges, upright)
        line = end[0] - start[0]
        shares = (points - start[0]) @ line / max(float(line @ line), 1e-300)
        touched = (shares > 0.0) & (shares < 1.0)
        touched &= _edge_distance(edges, upright, points) <= self.tolerance
        if not touched.any():
            return towards
        first = int(np.argmin(np.where(touched, shares, np.inf)))
        at = float(shares[first]) * reach
        faces = edges.face[upright[first]]
        planes = self.model.face_plane[faces[faces >= 0]]
        inside = self._meets(listener, np.isin(self.model.face_plane, planes))
        angle = 2.0 * self.tolerance / at
        sides = []
        for turn in (angle, -angle):
            c, s = math.cos(turn), math.sin(turn)
            way = np.array([c * towards[0] - s * towards[1], s * towards[0] + c * towards[1]])
            own = self._above(self.layout.obstacles_along(self.source[:2], way, reach))
            # How far from the edge the nearest of them is crossed along the way.
            gap = min((max(near - at, at - far, 0.0) for near, far, _ in own), default=math.inf)
            sides.append((gap, way))
        sides.sort(key=lambda side: side[0])
        return sides[0][1] if inside else sides[1][1]

    def _cuts_for(self, towards, reach: float, zr: float, own: tuple, open_top: float):
        """The runs whose weather's effects, weighed (in dB), give a listener ``reach``
        away along ``towards`` and ``zr`` up the effect behind what stands on its own way
        (``own``, see :meth:`_cut`). Runs along whole degrees (``AZIMUTH_STEP_DEG``) are
        shared by every listener near them: the nearest alone over the middle
        ``1 - BARRIER_BLEND`` of the step between two, and both, weighed linearly, over
        the ``BARRIER_BLEND`` round the halfway direction, so that the effect changes
        smoothly from one to the next. A direction counts for nothing where more or
        fewer obstacles stand short of the listener along it than along its own way, or
        the listener would be under a building's roof or on a barrier or a wall there,
        and for less where the last of them stands nearer the listener than
        ``BARRIER_CLOSE`` times as far as along its own way (a listener close behind a
        slanting barrier or wall); the
        listener's own way, in a run of its own, takes the weight they lose."""
        angle = math.degrees(math.atan2(towards[1], towards[0]))
        lower = math.floor(angle / AZIMUTH_STEP_DEG)
        share = angle / AZIMUTH_STEP_DEG - lower
        upper = min(max((share - 0.5) / BARRIER_BLEND + 0.5, 0.0), 1.0)
        behind = max(reach - max(far for _, far, _ in own), 1e-9)
        picked = []
        for n, weight in ((lower, 1.0 - upper), (lower + 1, upper)):
            if weight <= 0.0:
                continue
            azimuth = (n * AZIMUTH_STEP_DEG) % 360.0
            cut = self._cut(azimuth)
            short = [o for o in cut if o[0] < reach]
            if len(short) != len(own) or _at(cut, reach):
                continue
            fit = (reach - max(far for _, far, _ in short)) / (BARRIER_CLOSE * behind)
            weight *= min(max(fit, 0.0), 1.0)
            if weight > 1e-9:
                picked.append((azimuth, cut, weight))
        rest = 1.0 - sum(weight for *_, weight in picked)
        if rest > 1e-9:
            picked.append((angle % 360.0, own, rest))
        out = []
        for azimuth, cut, weight in picked:
            a = math.radians(azimuth)
            key = None if self.zoned else self._towards(np.array([math.cos(a), math.sin(a)]))
            top = max([open_top] + [BARRIER_HEADROOM * h for near, _, h in cut if near < reach])
            out.append(_Cut(key, azimuth, cut, weight, reach, zr, top, np.ones(self.bands, bool)))
        return out

    def _open_run(self, key, zr: float, reach: float, highest: float, shadow: bool):
        """The absorbing layer (m) and the bands ((bands,) bool) of the open ground's
        parabolic equation for a listener ``zr`` up and ``reach`` away along a way of
        profile ``key``: room above the rays that reach it (the highest ``highest`` up)
        for the sound that spreads up from them, deep into the shadow and at the lowest
        frequencies more than they reach; and the bands within the octave bands up to
        ``WAVE_MAX_BAND_HZ``, with ``open_rays`` up to ``LIT_WAVE_MAX_BAND_HZ`` unless
        the listener is in or near the refraction's shadow (``shadow``: no ray reaches
        it)."""
        top_band = WAVE_MAX_BAND_HZ
        if self.open_rays:
            start = self._start(key, self.zs, zr)
            near = shadow or (start is not None and reach >= PENUMBRA * start)
            top_band = WAVE_MAX_BAND_HZ if near else LIT_WAVE_MAX_BAND_HZ
        bands = octave_of(self.model.bands) <= OCTAVE_BANDS_HZ.index(top_band)
        top = max(
            WAVE_HEADROOM * max(self.zs, zr, highest),
            max(self.zs, zr) + WAVE_MIN_HEADROOM_M,
            WAVE_TOP_SLOPE * reach,
        )
        return top, bands

    def _azimuth(self, towards) -> float:
        """The direction ``towards`` (degrees from the x axis), rounded to
        ``AZIMUTH_STEP_DEG``."""
        angle = math.degrees(math.atan2(towards[1], towards[0]))
        return round(angle / AZIMUTH_STEP_DEG) * AZIMUTH_STEP_DEG % 360.0

    def _above(self, found) -> tuple:
        """The obstacles ``found`` ((near, far, top), as
        :meth:`~acoustic_spaces.geometry.outdoor.Layout.obstacles_along` gives them) with
        their tops above the ground, as :func:`~acoustic_spaces.engine.parabolic.march`
        takes them, to the nanometre (so that mirrored ways share their runs)."""
        return tuple(
            (round(near, 9), round(far, 9), round(top - self.ground, 9))
            for near, far, top in found
            if top > self.ground
        )

    def _past_ends(self, k, listener, zr, paths, rows, cutoff, reach):
        """Of the paths ``rows`` of ``paths``, those over a barrier's or a building's
        edge that is not vertical, touching its line past one of its ends, where the
        edge diffraction fades them out (:func:`~acoustic_spaces.engine.diffraction.
        along_edge`), and reflecting off nothing but the ground: by edge and end, the
        paths and the request that changes them as the weather changes the sound behind
        the barrier just inside that end (:meth:`_barrier`, along the way from the source
        to it), as the paths over the edge just inside it change."""
        edges = self._edge_set()
        laid = self._laid_edges()
        groups: dict[tuple, list[int]] = {}
        for i in rows:
            e = int(paths.edge[i])
            if e < 0 or not laid[e] or abs(float(edges.direction[e][2])) >= 1.0 - 1e-9:
                continue
            faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
            if len(faces) and not np.all(self.flat[faces]):
                continue
            for corner in paths.points[i][1:-1]:
                v = corner - edges.start[e]
                t = float(v @ edges.direction[e])
                if np.linalg.norm(v - t * edges.direction[e]) > self.tolerance:
                    continue
                if t < -self.tolerance or t > edges.length[e] + self.tolerance:
                    groups.setdefault((e, int(t > 0.0)), []).append(i)
                break
        out = []
        for (e, end), members in groups.items():
            inset = min(1e-3, 0.5 * float(edges.length[e]))
            t = inset if end == 0 else float(edges.length[e]) - inset
            point = edges.start[e] + t * edges.direction[e]
            across = point[:2] - self.source[:2]
            if float(np.linalg.norm(across)) <= 1e-6:
                continue
            towards = across / float(np.linalg.norm(across))
            found = self._barrier(k, listener, towards, reach, zr, paths, [], members, cutoff)
            if found is not None:
                out.append(found)
        return out

    def _ends(self, k, listener, zr, paths, rows, cutoff, towards):
        """Of the paths ``rows`` of ``paths``, those round the ends of barriers and the
        corners of buildings alone (:meth:`_round`), where some of those round the
        same edges reflect off the ground; for each such way, the open ground's runs
        of the parabolic equation along it, unfolded round the edges (its length seen
        from above, the wind's component along it as the lengths of its legs weigh
        it), change its paths in the open ground's bands for it as the weather changes
        the field along it (:meth:`_place_ends`), delayed as much as the first of them,
        and above them give them their legs' own times and shadows (:meth:`_legs`). At
        an end's shadow boundary its way is the listener's own, so its paths change as
        the direct sound does there.
        Paths round the same edges none of which reflects off the ground keep their
        legs' (their partners off the ground left out, the ratio over the ground
        holds no longer). Returns the paths taken (indices), those above the bands and
        those in them (parts of the listener's) and the request; None for none."""
        ways: dict[tuple, list[int]] = {}
        for i in rows:
            edges = self._round(paths, i)
            if edges is not None:
                ways.setdefault(edges, []).append(i)
        ways = {e: rows for e, rows in ways.items() if any(self._grounded(paths, i) for i in rows)}
        if not ways:
            return None
        chosen = sorted(i for found in ways.values() for i in found)
        at = {i: n for n, i in enumerate(chosen)}
        part = paths[np.array(chosen)]
        inside = np.zeros((len(chosen), self.bands), dtype=bool)
        delay = part.delay_s.copy()
        edges = self._edge_set()
        requested = []
        for way, found in ways.items():
            corners = np.array([self.source[:2], *(edges.start[e][:2] for e in way), listener[:2]])
            legs = np.diff(corners, axis=0)
            length = float(np.linalg.norm(legs, axis=1).sum())
            if self.zoned:
                key = self._mix(corners)
            else:
                key = self._key(float(self.wind @ legs.sum(axis=0)) / length)
            rays = self.rays(key)
            eigen = rays.eigenrays(zr, length, MAX_GROUND_BOUNCES, max_time=cutoff)
            top, bands = self._open_run(key, zr, length, _highest(rays, eigen, zr), len(eigen) == 0)
            first = paths.points[min(found, key=lambda i: float(paths.delay_s[i]))]
            shift = sum(self._leg(a, b)[1] for a, b in zip(first[:-1], first[1:], strict=True))
            index = np.array([at[i] for i in found])
            inside[index] = bands
            delay[index] += shift
            ground = self._grounds_round(corners)
            requested.append(_Way(index, key, ground, length, zr, top, bands, shift))
        legs = self._legs(part)
        high = None
        if not inside.all():
            high = dataclasses.replace(legs, energy=np.where(inside, 0.0, legs.energy))
        low = dataclasses.replace(
            part,
            energy=np.where(inside, part.energy, 0.0),
            delay_s=delay,
            spectrum=None if part.spectrum is None else part.spectrum.copy(),
        )
        return chosen, high, low, _Ends(listener=k, part=-1, ways=requested)

    def _round(self, paths: EarlyReflections, i: int) -> tuple | None:
        """The edges (indices, in turn) that the path ``i`` of ``paths`` goes round,
        where every one is a vertical edge of a barrier or a building (an end or a
        corner) and it reflects off nothing but the ground; else None."""
        if paths.edge[i] < 0:
            return None
        faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
        if len(faces) and not np.all(self.flat[faces]):
            return None
        edges = self._edge_set()
        laid = self._laid_edges()
        upright = np.abs(edges.direction[:, 2]) >= 1.0 - 1e-9
        out = []
        for corner in paths.points[i][1:-1]:
            on = _on_edges(edges, corner, self.tolerance)
            if not len(on):
                continue  # off the ground
            on = [int(e) for e in on if upright[e]]
            if not on or not laid[on[0]]:
                return None
            out.append(on[0])
        if int(paths.edge[i]) not in out:
            return None
        return tuple(out)

    def _grounded(self, paths: EarlyReflections, i: int) -> bool:
        """Whether the path ``i`` of ``paths`` reflects off the ground."""
        faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
        return bool(len(faces) and np.any(self.flat[faces]))

    def _laid_edges(self) -> np.ndarray:
        """(edges,) bool: whether each of the model's edges is a barrier's or a
        building's."""
        if self._laid is None:
            mesh = self.model.room.mesh
            laid = np.array([_kind(label) in _LAID for label in mesh.labels] or [False])
            faces = laid[mesh.face_labels]
            edge = self._edge_set().face
            self._laid = faces[edge[:, 0]] & faces[edge[:, 1]]
        return self._laid

    def _cut(self, azimuth: float) -> tuple:
        """What stands on the way from the source towards ``azimuth`` (degrees from the
        x axis), out to the ground's edge, for the parabolic equation: (near, far, top),
        m along the ground from the source and above it (a barrier where near is far),
        as :func:`~acoustic_spaces.engine.parabolic.march` takes them."""
        if azimuth not in self._cuts:
            angle = math.radians(azimuth)
            direction = np.array([math.cos(angle), math.sin(angle)])
            x0, y0, x1, y1 = self.layout.extent
            found = self.layout.obstacles_along(
                self.source[:2], direction, math.hypot(x1 - x0, y1 - y0)
            )
            self._cuts[azimuth] = self._above(found)
        return self._cuts[azimuth]

    def _in_plane(self, paths: EarlyReflections, i: int, towards, reach: float) -> bool:
        """Whether the path ``i`` of ``paths`` goes over edges that all cross the vertical
        plane through the source and the listener between them, none vertical, and
        reflects off nothing but the ground."""
        if paths.edge[i] < 0:
            return False
        faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
        if len(faces) and not np.all(self.flat[faces]):
            return False
        edges = self._edge_set()
        # Only the barriers' and buildings' edges: the parabolic equation's runs leave the
        # model file's objects out.
        laid = self._laid_edges()
        if not laid[paths.edge[i]]:
            return False
        if not self._across(edges, np.array([paths.edge[i]]), towards, reach)[0]:
            return False
        # Any other corner on an edge (the second of two) lies on one across the plane.
        tolerance = self.tolerance
        for corner in paths.points[i][1:-1]:
            on = _on_edges(edges, corner, tolerance)
            if (
                len(on)
                and paths.edge[i] not in on
                and not (self._across(edges, on, towards, reach) & laid[on]).any()
            ):
                return False
        return True

    def _across(self, edges, index: np.ndarray, towards, reach: float) -> np.ndarray:
        """Whether each of the edges ``index`` is not vertical and crosses the way from
        the source along ``towards`` (seen from above) short of ``reach``."""
        out = np.zeros(len(index), dtype=bool)
        for n, e in enumerate(index):
            if abs(float(edges.direction[e][2])) >= 1.0 - 1e-9:
                continue
            ends = np.array([edges.start[e][:2], edges.end[e][:2]])
            out[n] = bool(line_crossings(self.source[:2], towards, reach, ends, closed=False))
        return out

    def _edge_set(self):
        """The model's diffracting edges, as the paths over them number them."""
        if self._edges is None:
            from acoustic_spaces.engine.diffraction import find_edges

            self._edges = find_edges(self.model)
        return self._edges

    def _blocked(self, listener: np.ndarray) -> bool:
        """Whether a face of the model file's objects stands in the straight line from
        the source to ``listener``."""
        if self._objects is None:
            mesh = self.model.room.mesh
            laid = np.array(
                [_kind(label) in _LAID or tuple(label) in _OPEN for label in mesh.labels] or [True]
            )
            self._objects = ~laid[mesh.face_labels]
        return self._meets(listener, self._objects)

    def _meets(self, listener: np.ndarray, faces: np.ndarray) -> bool:
        """Whether the straight line from the source to ``listener`` meets one of the
        ``faces`` ((F,) bool), by the model's intersection test."""
        if not faces.any():
            return False
        start = self.source.copy()
        delta = np.asarray(listener, dtype=float) - start
        left = float(np.linalg.norm(delta))
        direction = delta / max(left, 1e-300)
        skip = -1
        for _ in range(64):
            t, face = self.model.intersect(
                start[None], direction[None], np.array([left]), np.array([skip])
            )
            if face[0] < 0:
                return False
            if faces[face[0]]:
                return True
            start = start + direction * float(t[0])
            left -= float(t[0])
            skip = int(face[0])
        return False

    def _place_barrier(self, request: _Barrier, pressures: dict, parts: list) -> None:
        """The paths in the plane of ``request`` scaled by the weather's effect behind the
        barriers and buildings, the ratio of the band energies (``pressures``, by (the
        cut's id, "wind" or "still")) of its cuts' runs, their weighted geometric mean
        (in dB, the weights' mean), plus the eddies' share in the refraction's shadow,
        and delayed as much as the first of them."""
        effect = np.zeros(self.bands)
        for cut in request.cuts:
            still = pressures[(id(cut), "still")]
            if not np.all(still > 0.0):
                raise RuntimeError(
                    "the parabolic equation gave no sound in still air behind the barriers "
                    f"and buildings along {cut.azimuth:g} degrees at {cut.range:g} m"
                )
            ratio = pressures[(id(cut), "wind")] / still
            effect += cut.weight * np.log(np.maximum(ratio, _TINY))
        plane = parts[request.listener][request.part]
        plane.energy *= (np.exp(effect) + request.scattered)[None, :]
        plane.delay_s += request.shift

    def _place_ends(self, request: _Ends, pressures: dict, parts: list) -> None:
        """The paths of ``request`` round ends and corners changed, in the bands of each
        way, as the open ground's field along it changes in the weather: in the coherent
        bands their pressures, at each frequency, by the ratio of its runs' pressures
        (``pressures``, by (the way's id, "wind" or "still", "pressure")), less the delay
        they took on, so that together they change as that field does, its ground's
        fringes and its travel time with it; in the others their energies by the ratio
        of its runs' band energies (by (the way's id, "wind" or "still"))."""
        part = parts[request.listener][request.part]
        coherent = 0 if part.spectrum is None else part.spectrum.shape[1]
        for way in request.ways:
            bands = np.flatnonzero(way.bands)
            ratio = _ratio(pressures[(id(way), "wind")], pressures[(id(way), "still")])
            rows = way.rows
            for band in bands:
                wind = pressures.get((id(way), "wind", "pressure"))
                still = pressures.get((id(way), "still", "pressure"))
                if band >= coherent or wind is None or not np.all(still[band] != 0.0):
                    part.energy[rows, band] *= ratio[band]
                    continue
                change = wind[band] / still[band]  # (M,)
                gain = float(np.mean(np.abs(change) ** 2))
                part.energy[rows, band] *= gain
                if gain > 0.0:
                    turn = np.exp(2j * np.pi * self.frequencies[band] * way.shift)
                    part.spectrum[rows, band] *= (change * turn / math.sqrt(gain))[None, :]

    # -- the parabolic equation --------------------------------------------------------

    def _samples(self, band: int, barrier: bool) -> np.ndarray | None:
        """The frequencies (Hz) a run of the parabolic equation takes in ``band``: behind
        barriers (``barrier``) as many an octave as ``barrier_samples`` gives from its
        bands up; over the open ground, in the octave bands from ``OPEN_DENSE_FROM_HZ``
        up, ``open_dense`` times the early paths' (theirs among them,
        :func:`_early_of`); None for the early paths' own."""
        from acoustic_spaces.engine.diffraction import band_frequencies
        from acoustic_spaces.materials.bands import BANDS_PER_OCTAVE

        if barrier:
            dense = [n for start, n in self.barrier_samples if self.centres[band] >= start * 0.999]
            if not dense:
                return None
            per_band = math.ceil(dense[-1] / BANDS_PER_OCTAVE[self.model.bands])
        elif self.open_dense > 1 and octave_of(self.model.bands)[band] >= OCTAVE_BANDS_HZ.index(
            OPEN_DENSE_FROM_HZ
        ):
            per_band = self.open_dense * self.frequencies.shape[1]
        else:
            return None
        if per_band not in self._dense:
            self._dense[per_band] = band_frequencies(per_band, self.model.bands)[0]
        return self._dense[per_band][band]

    def _admittance_at(self, face: int, band: int, frequencies: np.ndarray) -> np.ndarray:
        """(M,) the admittance of ``face`` as :meth:`_admittance` gives it, at other
        ``frequencies`` (Hz) of ``band`` than the early paths'."""
        index = (face, band, len(frequencies))
        with self._lock:
            if index not in self._dense:
                model = self.model
                if model.impedance_faces(np.array([face]))[0]:
                    row = int(model.face_impedance[face])
                    grazing = (
                        model.impedance_extended is not None and model.impedance_extended.any()
                    )
                    value = model.impedances[row].admittance(
                        frequencies, model.room.speed_of_sound(), cosine=0.0 if grazing else None
                    )
                else:
                    from acoustic_spaces.engine.impedance import fit_admittance

                    value = fit_admittance(model.absorption[face], self.bands, None)(frequencies)
                self._dense[index] = np.asarray(value, dtype=complex)
            return self._dense[index]

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

    def _here(self) -> int:
        """The face of the ground under the source (the first flat one off the ground)."""
        here = self._ground_face(np.array([[*self.source[:2], self.ground]]))[0]
        return int(here) if here >= 0 else int(np.flatnonzero(self.flat)[0])

    def _zone_faces(self, points: np.ndarray) -> np.ndarray:
        """The faces of the ground's zones at the horizontal ``points`` (n, 2), held to
        the layout's extent; the source's own (:meth:`_here`) outside every zone."""
        layout = self.model.room.outdoor
        x0, y0, x1, y1 = layout.extent
        found = self.model.part_face[layout.zone_at(np.clip(points, [x0, y0], [x1, y1])) + 1]
        return np.where(found < 0, self._here(), found)

    def _grounds_along(self, direction: np.ndarray, reach: float):
        """A function giving the face of the ground (its material) at a distance along
        ``direction`` from the source, up to ``reach``."""
        here = self._here()
        if not self.zones or self.model.ground_part is None:
            return lambda r: here
        # The grounds every GROUND_STEP_M along the way, looked up once.
        along = np.arange(0.0, reach + 2.0 * GROUND_STEP_M, GROUND_STEP_M)
        faces = self._zone_faces(self.source[:2] + along[:, None] * direction)
        last = len(faces) - 1
        return lambda r: int(faces[min(int(round(r / GROUND_STEP_M)), last)])

    def _grounds_round(self, corners: np.ndarray) -> tuple | None:
        """With ground zones, the grounds along the horizontal polyline ``corners``
        (m + 1, 2) unfolded, looked up every GROUND_STEP_M as :meth:`_grounds_along`
        does along a line: ((step, face), ...) where each begins, so that ways alike
        share their runs; None without zones."""
        if not self.zones or self.model.ground_part is None:
            return None
        corners = np.asarray(corners, dtype=float)
        legs = np.diff(corners, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        ends = np.cumsum(lengths)
        along = np.arange(0.0, ends[-1] + 2.0 * GROUND_STEP_M, GROUND_STEP_M)
        leg = np.minimum(np.searchsorted(ends, along, side="right"), len(legs) - 1)
        part = (along - (ends[leg] - lengths[leg])) / np.maximum(lengths[leg], 1e-12)
        faces = self._zone_faces(corners[leg] + part[:, None] * legs[leg])
        begins = np.flatnonzero(np.diff(faces, prepend=faces[0] - 1))
        return tuple((int(i), int(faces[i])) for i in begins)

    def _grounds_on(self, ground: tuple | None):
        """A function giving the face of the ground at a distance along a way whose
        grounds are ``ground`` (:meth:`_grounds_round`), the source's own for None."""
        if ground is None:
            here = self._here()
            return lambda r: here
        starts = np.array([step for step, _ in ground])
        faces = [face for _, face in ground]
        return lambda r: faces[int(np.searchsorted(starts, round(r / GROUND_STEP_M), "right")) - 1]

    def _place(
        self,
        wave: _Wave,
        pressure: np.ndarray,
        energy: np.ndarray,
        parts: list,
        listeners: np.ndarray,
    ):
        """Puts the open ground's parabolic equation in the paths of ``wave``'s listener:
        its ``pressure`` (bands, M) at the early paths' frequencies and its band
        ``energy`` (bands,) over all the frequencies it ran at."""
        bands = wave.bands
        excess = np.array(energy, dtype=float) * wave.distance**2  # (bands,)
        if wave.shadow:
            # Above the parabolic equation's bands the shadow deepens as it does over the
            # octave below: by the same factor per octave from the top octave band's
            # energy, taken over the octave band's frequencies (in a third-octave run its
            # three bands' over their widths); in a third-octave run each band's along
            # the way within its octave band, scaled so that the three, as the air lets
            # them through, average to the octave band's. Octave and third-octave runs so
            # deepen it alike.
            computed = np.flatnonzero(bands)
            octave = octave_of(self.model.bands)
            low, high = band_limits(self.model.bands)

            def octave_mean(index: int) -> float:
                inside = computed[octave[computed] == index]
                width = high[inside] - low[inside]
                return float(np.sum(excess[inside] * width) / np.sum(width))

            top = int(octave[computed[-1]])
            anchor = octave_mean(top)
            slope = 1.0
            if top > 0 and octave_mean(top - 1) > 0.0:
                slope = min(anchor / octave_mean(top - 1), 1.0)
            # Each band's place in its octave band (octaves from its centre); a silent
            # top octave band (slope 0) leaves the bands above it silent.
            within = np.zeros(self.bands)
            if slope > 0.0:
                within = slope ** np.log2(self.centres / np.asarray(OCTAVE_BANDS_HZ)[octave])
            heard = np.exp(-self.air * wave.distance)
            for band in range(computed[-1] + 1, self.bands):
                inside = octave == octave[band]
                # The octave band's mean as the air lets its bands through; as the plain
                # mean where the air lets none through (the band is then silent anyway).
                through = np.sum(within[inside] * heard[inside])
                share = (
                    within[band] * np.sum(heard[inside]) / through
                    if through > 0.0
                    else within[band] * np.sum(inside) / max(np.sum(within[inside]), _TINY)
                )
                excess[band] = anchor * slope ** (octave[band] - top) * share
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
            self.directivity,
            self.orientation,
            wave.departure[None],
            self.bands,
            distances=np.array([wave.distance]),
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
            if family.cosines:
                # The parabolic equation's pressure has no reflection phase to turn.
                family.cosines[wave.carrier] = np.full(len(family.cosines[wave.carrier]), np.nan)

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


@dataclass
class Planned:
    """A source's paths to its listeners waiting for the parabolic equation
    (:meth:`Refraction.plan`): each listener's paths bent so far and the requests."""

    refraction: Refraction
    listeners: np.ndarray
    parts: list
    requests: list[_Wave]
    barriers: list[_Barrier] = dataclasses.field(default_factory=list)
    ends: list[_Ends] = dataclasses.field(default_factory=list)

    def finish(self, pressures: dict) -> list[EarlyReflections]:
        """The paths to each listener with the equation's ``pressures`` (by the id of
        each request, as :func:`solve` gives them) put in."""
        refraction = self.refraction
        for request in self.requests:
            refraction._place(
                request,
                pressures[id(request)],
                pressures.get((id(request), None), np.zeros(refraction.bands)),
                self.parts,
                self.listeners,
            )
        for request in self.barriers:
            refraction._place_barrier(request, pressures, self.parts)
        for request in self.ends:
            refraction._place_ends(request, pressures, self.parts)
        return [
            p
            if isinstance(p, EarlyReflections)
            else EarlyReflections.concatenate(p, refraction.bands)
            for p in self.parts
        ]


@dataclass
class _Job:
    """A run of the parabolic equation in one band, and whom it is for."""

    refraction: Refraction
    speed: object
    speed_at: object
    face_at: object
    band: int
    members: list  # (plan, request, slot): slot None for the open ground's pressure
    top: float
    alone: bool
    obstacles: tuple = ()
    #: Whether the air bends sound (None: as the march finds).
    refracting: bool | None = None
    barrier: bool = False  # with the grid and samples of the runs behind barriers
    #: What the run has in common with those that differ from it in what stands on the
    #: way and in their listeners alone (the source's runs behind barriers in one band
    #: and weather, under one absorbing layer), which share the march over the open
    #: ground (:class:`~acoustic_spaces.engine.parabolic.Trunk`); None: none do.
    family: tuple | None = None


def solve(plans: list[Planned]) -> list[dict]:
    """Runs the parabolic equation for ``plans`` (:meth:`Refraction.plan`): for each, the
    pressure (bands, M) of each of its requests by the request's id, and the band
    energies (bands,) of the runs for the paths behind barriers and round ends by (the
    id of the cut or the way, "wind" or "still"). A source's listeners that the sound
    goes the same way to share a run (the same profile; with weather or ground zones the
    same direction too). So do several sources' at one height over one ground, where the
    weather and the grounds have no zones (the places of a walk, say): there a run from
    one source holds for any other alike, the way along the ground being the same from
    each. A run marches out to the farthest of its listeners under an absorbing layer
    above the highest of their ``top``; in a run shared that way each listener is
    reached by a last step of its own, off to the side of the march, so that it gets
    what its own source's run would give it alone.

    Behind barriers the runs in the weather share by the profile and what stands on the
    way (the cut), those in still air by the cut and the absorbing layer alone; the
    open ground's runs along the ways round ends, in the weather and in still air,
    likewise with no cut. Every listener of theirs is reached by a last step of its
    own.

    The runs behind barriers in one band that differ in the cut and the listeners alone
    (in still air those of every direction under one absorbing layer; in the weather,
    where it has no zones, those of one profile) march through the same air before
    their first obstacle: they go one after another, the nearest first obstacle first,
    each taking the field there from the one before where the grounds on its way that
    far are the same (:class:`~acoustic_spaces.engine.parabolic.Trunk`), in as many
    chains as runs go side by side."""
    groups: dict[tuple, list[tuple[int, _Wave]]] = {}
    behind: dict[tuple, list] = {}
    for p, planned in enumerate(plans):
        refraction = planned.refraction
        sharing = refraction._sharing() if len(plans) > 1 else None
        owner = ("own", p) if sharing is None else ("shared", sharing)
        for request in planned.requests:
            groups.setdefault((owner, request.key), []).append((p, request))
        zones = refraction.zones or refraction.zoned
        for request in planned.barriers:
            for cut in request.cuts:
                azimuth = cut.azimuth if zones else None
                behind.setdefault((owner, True, cut.key, azimuth, cut.cut), []).append((p, cut))
        for request in planned.ends:
            for way in request.ways:
                behind.setdefault((owner, False, way.key, way.ground, ()), []).append((p, way))

    def weather(refraction, key, direction, reach):
        if key is None:
            # With weather zones: the weathers along the way.
            speed_at = refraction.weather.along(refraction.source[:2], direction, reach)
            return speed_at(0.0), speed_at
        return refraction.profile(key).speed, None

    def towards(azimuth: float) -> np.ndarray:
        angle = math.radians(azimuth)
        return np.array([math.cos(angle), math.sin(angle)])

    def grounds(refraction, members, reach):
        # Behind barriers along the direction of the cut; round ends along the way.
        first = members[0][1]
        if isinstance(first, _Way):
            return None, refraction._grounds_on(first.ground)
        direction = towards(first.azimuth)
        return direction, refraction._grounds_along(direction, reach)

    jobs: list[_Job] = []
    for (owner, (key, azimuth)), members in groups.items():
        refraction = plans[members[0][0]].refraction
        direction = members[0][1].direction if azimuth is None else towards(azimuth)
        reach = max(m.range for _, m in members)
        face_at = refraction._grounds_along(direction, reach)
        speed, speed_at = weather(refraction, key, direction, reach)
        top = max(m.top for _, m in members)
        # A shared run gives each listener what a run of its own would.
        alone = owner[0] == "shared"
        for band in range(refraction.bands):
            needing = [(p, m, None) for p, m in members if m.bands[band]]
            if needing:
                jobs.append(_Job(refraction, speed, speed_at, face_at, band, needing, top, alone))

    # Behind barriers and along the ways round ends: the runs in the weather, and in
    # still air shared by all with the same cut and absorbing layer.
    calm: dict[tuple, list] = {}
    for (owner, barrier, key, ground, cut), members in behind.items():
        refraction = plans[members[0][0]].refraction
        reach = max(m.range for _, m in members)
        direction, face_at = grounds(refraction, members, reach)
        speed, speed_at = weather(refraction, key, direction, reach)
        top = max(m.top for _, m in members)
        # Behind barriers, in a weather that is one all the way: a family's runs.
        alike = bool(cut) and speed_at is None
        for band in range(refraction.bands):
            needing = [(p, m, "wind") for p, m in members if m.bands[band]]
            if needing:
                jobs.append(
                    _Job(
                        refraction,
                        speed,
                        speed_at,
                        face_at,
                        band,
                        needing,
                        top,
                        True,
                        obstacles=cut,
                        refracting=True,
                        barrier=barrier,
                        family=(owner, barrier, key, top, band) if alike else None,
                    )
                )
        c0 = float(np.asarray(speed(np.array([0.0])))[0])
        if barrier:  # round ends ``ground`` is the way's already
            ground = members[0][1].azimuth if refraction.zones else None
        calm.setdefault((owner, barrier, c0, cut, top, ground), []).extend(members)
    for (owner, barrier, c0, cut, top, _), members in calm.items():
        refraction = plans[members[0][0]].refraction
        reach = max(m.range for _, m in members)
        _, face_at = grounds(refraction, members, reach)
        for band in range(refraction.bands):
            needing = [(p, m, "still") for p, m in members if m.bands[band]]
            if needing:
                jobs.append(
                    _Job(
                        refraction,
                        Profile.uniform(c0).speed,
                        None,
                        face_at,
                        band,
                        needing,
                        top,
                        True,
                        obstacles=cut,
                        refracting=True,
                        barrier=barrier,
                        family=(owner, barrier, "still", c0, top, band) if cut else None,
                    )
                )
    if not jobs:
        return [{} for _ in plans]

    first = plans[0].refraction
    solver = wide_angle if first.wide_angle else parabolic
    share = 1

    def steps(job: _Job) -> float:
        coarse = job.barrier and job.refraction.centres[job.band] >= BARRIER_COARSE_FROM_HZ * 0.999
        return solver.STEPS_PER_WAVELENGTH * (BARRIER_COARSE if coarse else 1.0)

    def frequencies_of(job: _Job) -> np.ndarray:
        found = job.refraction._samples(job.band, job.barrier)
        return job.refraction.frequencies[job.band] if found is None else found

    def run(job: _Job, trunk: parabolic.Trunk | None = None) -> np.ndarray:
        refraction, band = job.refraction, job.band
        dense = refraction._samples(band, job.barrier) is not None
        frequencies = frequencies_of(job)

        def impedance(r):
            if dense:
                beta = refraction._admittance_at(job.face_at(r), band, frequencies)
            else:
                beta = refraction._admittance(job.face_at(r))[band]
            rigid = beta == 0.0
            return np.where(rigid, np.inf, 1.0 / np.conj(np.where(rigid, 1.0, beta)))

        ranges = np.array([m.range for _, m, _ in job.members])
        heights = np.array([m.height for _, m, _ in job.members])
        extra = {}
        if job.obstacles or job.barrier or job.refracting is not None:
            extra = {"obstacles": job.obstacles, "steps_per_wavelength": steps(job)}
            if solver is parabolic:
                extra["refracting"] = job.refracting
                if trunk is not None:
                    extra["trunk"] = trunk
        # The wide-angle equation's frequencies (the narrow-angle one's at steep points)
        # share the threads the runs leave.
        return solver.march(
            frequencies,
            job.speed,
            impedance,
            refraction.zs,
            ranges,
            heights,
            job.top,
            speed_at=job.speed_at,
            independent=job.alone,
            threads=share,
            **extra,
        )

    # The runs side by side, as many as the threads and the memory allow.
    biggest = max(
        solver.footprint(
            frequencies_of(job),
            job.refraction.c,
            job.top,
            steps(job),
        )
        for job in jobs
    )
    workers = min(first.threads, len(jobs), max(1, WAVE_MEMORY_BYTES // max(biggest, 1)))
    # The runs in chains: each family's, the nearest first obstacle first, shared out
    # over the workers (two runs or more to a chain), and every other run by itself.
    families: dict[tuple, list[int]] = {}
    for index, job in enumerate(jobs):
        if solver is parabolic and job.family is not None:
            families.setdefault(job.family, []).append(index)
    chains: list[list[int]] = []
    chained: set[int] = set()
    for members in families.values():
        members.sort(key=lambda index: min(near for near, _, _ in jobs[index].obstacles))
        count = min(workers, len(members) // 2)
        for part in range(count):
            chain = members[part * len(members) // count : (part + 1) * len(members) // count]
            chains.append(chain)
            chained.update(chain)
    chains += [[index] for index in range(len(jobs)) if index not in chained]
    chains.sort(key=min)

    def run_chain(chain: list[int]) -> list[np.ndarray]:
        trunk = parabolic.Trunk() if len(chain) > 1 else None
        return [run(jobs[index], trunk) for index in chain]

    workers = min(workers, len(chains))
    share = max(1, first.threads // max(workers, 1))
    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            done = list(pool.map(run_chain, chains))
    else:
        done = [run_chain(chain) for chain in chains]
    results: list = [None] * len(jobs)
    for chain, values in zip(chains, done, strict=True):
        for index, value in zip(chain, values, strict=True):
            results[index] = value
    out: list[dict] = [
        {
            id(m): np.zeros(planned.refraction.frequencies.shape, dtype=complex)
            for m in planned.requests
        }
        for planned in plans
    ]
    for job, values in zip(jobs, results, strict=True):
        shape = job.refraction.frequencies.shape
        for (p, m, slot), value in zip(job.members, values.T, strict=True):
            # The band's energy over every frequency the run took; the pressures at the
            # early paths' own, time as exp(j omega t), as the early paths have it.
            energy = out[p].setdefault((id(m), slot), np.zeros(job.refraction.bands))
            energy[job.band] = np.mean(np.abs(value) ** 2)
            if slot is None:
                out[p][id(m)][job.band] = np.conj(_early_of(value, shape[1]))
            elif isinstance(m, _Way):
                pressure = out[p].setdefault(
                    (id(m), slot, "pressure"), np.zeros(shape, dtype=complex)
                )
                pressure[job.band] = np.conj(_early_of(value, shape[1]))
    return out


def _highest(rays: Rays, found, zr: float) -> float:
    """The greatest height (m) the eigenrays ``found`` to a listener ``zr`` up reach."""
    return max(
        (float(rays.curve(zr, found, i)[:, 1].max()) for i in range(len(found))), default=0.0
    )


def _ratio(wind: np.ndarray, still: np.ndarray) -> np.ndarray:
    """(bands,) the weather's effect: the energies ``wind`` over ``still`` (1 where both
    are none)."""
    return np.where(still > 0.0, wind / np.where(still > 0.0, still, 1.0), 1.0)


#: The kinds of the faces the parabolic equation's runs hold: barriers and buildings.
_LAID = frozenset({"barriers", "buildings"})
#: The labels of the ground and the sky.
_OPEN = frozenset({(GROUND,), (SKY,)})


def _kind(label) -> str | None:
    """The kind of the faces of ``label``, its last part, as the outdoor geometry writes
    it (a building's or a barrier's own name, or a model file's, may be anything)."""
    return label[-1] if len(label) else None


def _on_edges(edges, point: np.ndarray, tolerance: float) -> np.ndarray:
    """The indices of the ``edges`` that ``point`` (3,) lies on, within ``tolerance``."""
    index = np.arange(len(edges.length))
    points = np.broadcast_to(point, (len(index), 3))
    return np.flatnonzero(_edge_distance(edges, index, points) <= tolerance)


def _edge_distance(edges, index: np.ndarray, points: np.ndarray) -> np.ndarray:
    """(n,) the distance (m) from each of ``points`` (n, 3) to the ``edges`` ``index``
    (n,), between their ends."""
    start = edges.start[index]
    direction = edges.direction[index]
    along = np.clip(np.einsum("ij,ij->i", points - start, direction), 0.0, edges.length[index])
    return np.linalg.norm(points - start - direction * along[:, None], axis=1)


def _at(cut: tuple, reach: float) -> bool:
    """Whether a barrier or a building's wall of ``cut`` ((near, far, top)) stands within
    ``ON_OBSTACLE_M`` of ``reach`` m along the way: the march there gives a listener the
    field past it, none below its top."""
    return any(min(abs(near - reach), abs(far - reach)) <= ON_OBSTACLE_M for near, far, _ in cut)


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


def _bends(model: TraceModel) -> bool:
    """Whether the atmosphere bends the sound: outdoors, in air that is not still."""
    atmosphere = getattr(model.room.scene, "atmosphere", None)
    return model.room.outdoor is not None and atmosphere is not None and not atmosphere.still


def bending(model: TraceModel) -> dict:
    """The arguments that bend the rays of ``_native.trace_stochastic``, ``trace_map``
    and ``bent_chords`` in an outdoor scene's atmosphere: its profile (rows of height
    above the ground, speed of sound in still air and wind speed), the way the wind
    blows, the ground's height and how far a chord may stray from its arc. Empty where
    sound goes straight: indoors, or outdoors in still air."""
    if not _bends(model):
        return {}
    scene = model.room.scene
    atmosphere = getattr(scene, "atmosphere", None)
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


def covering(model: TraceModel) -> dict:
    """The arguments of ``_native.trace_stochastic`` and ``trace_map`` that leave out of
    the late rays the specular paths the early stage holds where the atmosphere bends the
    sound (:meth:`Refraction._listener`): the curved paths off the ground, with up to
    :data:`MAX_GROUND_BOUNCES` bounces whatever the order, alone or unfolded off vertical
    walls, the ground's bounces counting as one reflection towards the order, and the
    paths off any other face as in still air. ``face_kind`` is 1 for the faces of the
    ground (:func:`~acoustic_spaces.engine.model._ground_faces`), 0 for vertical walls and
    -1 for the others, ``covered_ground`` the most bounces. Empty where sound goes
    straight, where every reflection counts towards the order."""
    if not _bends(model):
        return {}
    vertical = np.abs(model.plane_normal[model.face_plane, 2]) <= _VERTICAL
    kind = np.where(_ground_faces(model.room), 1, np.where(vertical, 0, -1))
    return {
        "face_kind": np.ascontiguousarray(kind, dtype=np.int64),
        "covered_ground": MAX_GROUND_BOUNCES,
    }


def direct_times(times: np.ndarray, paths: list[EarlyReflections]) -> np.ndarray:
    """(n,) when the direct sound reaches each listener along its bent path: the earliest
    of its ``paths`` that neither reflects nor goes round an edge, sooner downwind and
    later upwind than along the straight line; where there is none (behind a barrier or
    a building), ``times``, the straight line's, or the earliest path's where it comes
    sooner (over the barrier downwind)."""
    out = np.array(times, dtype=float)
    for k, found in enumerate(paths):
        clear = (found.order == 0) & ~found.diffracted
        if clear.any():
            out[k] = float(found.delay_s[clear].min())
        elif len(found):
            out[k] = min(out[k], float(found.delay_s.min()))
    return out


def refraction_for(model: TraceModel, source: np.ndarray, **kwargs) -> Refraction | None:
    """The :class:`Refraction` of the source at ``source``, or None where sound goes
    straight: indoors, or outdoors in still air."""
    if not _bends(model):
        return None
    return Refraction(model, source, **kwargs)


__all__ = [
    "AZIMUTH_STEP_DEG",
    "LIT_WAVE_MAX_BAND_HZ",
    "MAX_GROUND_BOUNCES",
    "WAVE_MAX_BAND_HZ",
    "WAVE_MIN_RANGE",
    "Planned",
    "Refraction",
    "bending",
    "covering",
    "direct_times",
    "refraction_for",
    "solve",
]
