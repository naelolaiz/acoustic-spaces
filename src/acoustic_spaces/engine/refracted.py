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
change in every band as the open ground's field along their own way round the edges,
unfolded, changes from the straight direct sound and ground reflection in still air to
the weather's (the eigenrays, with the parabolic equation in its bands): at the end's
shadow boundary they jump by the open ground's field, as the direct sound does
(:meth:`Refraction._ends`, :meth:`Refraction._turned`).

The late stage's rays bend too: the compiled tracer follows each one along its arcs
through the layers, in chords (:func:`bending` gives it the atmosphere).

Where the weather changes from place to place (the atmosphere's ``zones``,
:mod:`~acoustic_spaces.engine.weather`), the rays take the mean profile along their
way, the parabolic equation marches through each weather in turn, and the late rays
take the mean weather over the ground.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.engine import parabolic, run_size, wide_angle
from acoustic_spaces.engine.coherent import (
    FILTERED_MAX_BAND_HZ,
    band_count,
    ground_factors,
    reflection_phases,
)
from acoustic_spaces.engine.early import _EDGE_TOLERANCE, EarlyReflections, _face_at
from acoustic_spaces.engine.fine import (
    OPEN_DENSE,
    OPEN_DENSE_FROM_HZ,
    fine_frequencies,
    placed,
    reference_delay,
    skirt_frequencies,
)
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
    #: With eddies, out of the shadow: the family's coherence factor at frequencies of a
    #: band, ``blur(band, f)`` (:meth:`Refraction._coherence_of`); None without.
    blur: Callable[[int, np.ndarray], np.ndarray | None] | None = None


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
    """Paths round the same ends of barriers and corners of buildings, and the open
    ground's field along their way, unfolded round those edges, with nothing in it
    (:meth:`Refraction._open_way`), which changes them in the weather."""

    rows: np.ndarray  # theirs, of the listener's part of paths round ends
    key: object  # the profile's key along the way
    ground: tuple | None  # with ground zones, its grounds (:meth:`Refraction._grounds_round`)
    #: With weather zones, the weathers along it (:meth:`Refraction._weathers_round`).
    weathers: tuple | None
    range: float  # the way's length seen from above
    height: float
    top: float  # the runs' absorbing layer
    bands: np.ndarray  # (bands,) bool: those of the open ground's parabolic equation
    #: The eigenrays along the way in the weather (before the equation is put in) and
    #: the straight direct sound and ground reflection along it in still air.
    weather: EarlyReflections
    still: EarlyReflections
    #: The open ground's request for a listener at the way's end (None: the rays alone),
    #: which :meth:`Refraction._place` puts in ``weather`` with the way's run.
    wave: _Wave | None
    at: np.ndarray  # (3,) the listener
    corners: np.ndarray  # (k + 2, 2) the way seen from above: the source, the edges, the listener
    tops: np.ndarray  # (k,) the height (z) of the top of each of its edges
    edges: tuple  # (k,) its edges (indices of the model's diffracting edges), in turn


@dataclass
class _Ends:
    """A listener's paths round the ends of barriers and corners of buildings, by way."""

    listener: int
    part: int  # of the listener's parts
    ways: list[_Way]
    #: Over a roof, each way's paths (rows of the part) and the request for the runs with
    #: the buildings under the source and the listener inside (:meth:`Refraction._ends`).
    roofs: list[tuple[np.ndarray, _Barrier]] = dataclasses.field(default_factory=list)


class Refraction:
    """Bends the early paths of the source at ``source`` in an outdoor scene's
    atmosphere (not still), running the parabolic equation on ``threads`` threads (0:
    every core): the wide-angle one (:mod:`~acoustic_spaces.engine.wide_angle`) with
    ``wide_angle``. ``edges``: the model's diffracting edges, as the paths over them
    number them (found when needed if not given). ``profiles``: the profiles of the
    model's air found so far by their key (:meth:`profile`), which the sources of one
    model may share, as none depends on the source; its own if not given.
    ``rendered``: whether a response is rendered from the paths' spectra (an impulse
    response). Out of the shadow the open ground's equation runs on through the upper
    crossover of each listener's top band of it where one is, and for a coverage map
    (``rendered`` False) only where the band filters read that crossover, a top band up
    to :data:`~acoustic_spaces.engine.coherent.FILTERED_MAX_BAND_HZ`: a map reads the
    bands up to it through those filters from the paths' spectra, as a listener's
    echogram does, and the bands above from their energies.
    """

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
        rendered: bool = True,
    ) -> None:
        from acoustic_spaces.engine.diffraction import band_frequencies

        self.barrier_samples = barrier_samples(barrier_frequencies)
        #: The open ground's runs from ``OPEN_DENSE_FROM_HZ`` up take this many times the
        #: early paths' frequencies (``simulation.early.open_ground_frequencies``: 24 an
        #: octave, else theirs); with ``rays`` their bands end below 1 kHz short of
        #: ``PENUMBRA`` of the way to the shadow's start (``LIT_WAVE_MAX_BAND_HZ``).
        self.open_dense = OPEN_DENSE if open_ground_frequencies == 24 else 1
        self.open_rays = open_ground_frequencies == "rays"
        self.rendered = bool(rendered)
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
        self._starts: dict[tuple, tuple] = {}
        self.eddies = Eddies.of(model, self.c)
        self.layout = room.outdoor
        self._cuts: dict[tuple, tuple] = {}
        self._dense: dict[tuple, np.ndarray] = {}
        self._objects: np.ndarray | None = None
        self._edges = edges
        self._laid: np.ndarray | None = None
        self._roofs: np.ndarray | None = None

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
        return _kept((self._weather, key, self.zs), lambda: Rays(self.profile(key), self.zs))

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
        # The specular paths off roofs (and the ground) alone, which the runs with the
        # buildings inside hold.
        roof_rows = [
            i
            for planes, rows in groups.items()
            if planes and all(self._roof_plane(p) for p in planes)
            for i in rows
        ]
        barrier = self._barrier(
            k, listener, way, reach, zr, paths, groups[()], bent, cutoff, roof_rows
        )
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
        # take the weather's effect behind it just inside the end (:meth:`_past_ends`);
        # over a roof, beside the lit side's paths in the plane too.
        under = self._under(listener) if self.layout.buildings else ()
        if (barrier is None or under) and bent:
            for rows, past in self._past_ends(k, listener, zr, paths, bent, cutoff, reach):
                taken = set(rows)
                bent = [i for i in bent if i not in taken]
                out.append(paths[np.array(rows)])
                past.part = len(out) - 1
                asked.append(past)
        # The paths round the ends of barriers and the corners of buildings take the
        # weather's effect on the open ground along their own way (:meth:`_ends`), on
        # either side of the shadow boundary, whatever the listener's own field; over a
        # roof the effect of the lit side's runs with the building inside.
        ends = None
        if bent:
            ends = self._ends(k, listener, zr, paths, bent, cutoff, way, reach, under)
        if ends is not None:
            chosen, low, request_ends = ends
            taken = set(chosen)
            bent = [i for i in bent if i not in taken]
        if bent:
            out.append(self._legs(paths[np.array(sorted(bent))]))
        if ends is not None:
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
        if not clear:
            return family, None
        shadow = len(found) == 0
        distance = float(np.linalg.norm(listener - self.source))
        azimuth = self._azimuth(towards) if self.zones or self.zoned else None
        run = (None if self.zoned else key, azimuth)
        wave = self._wave(k, key, run, towards, reach, zr, family, highest, shadow, distance, r0)
        return family, wave

    def _wave(self, k, key, run, towards, reach, zr, family, highest, shadow, distance, ratio):
        """The open ground's request for the parabolic equation for the listener ``k``
        ``zr`` m up, ``reach`` m along the ground ``towards`` (2,) from the source and
        ``distance`` m from it, whose eigenrays of profile ``key`` are the paths
        ``family`` (the highest ``highest`` up; none in the refraction's ``shadow``),
        sharing the runs of ``run``; None where the rays alone give the field: near the
        source, where no path off the ground is asked for and a ray reaches, or in no
        band."""
        if reach < WAVE_MIN_RANGE * (self.zs + zr):
            return None
        if not shadow and self.max_order < 1:
            return None
        top, bands = self._open_run(key, zr, reach, highest, shadow)
        if not bands.any():
            return None
        if shadow:
            departure = np.append(towards, 0.0)
            delay = distance / float(self.profile(key).c[0])
            carrier = -1
        else:
            carrier = int(np.argmin(family.delay_s))
            departure = _departure(family.points[carrier])
            delay = float(family.delay_s[carrier])
        return _Wave(
            listener=k,
            key=run,
            mix=key,
            direction=towards,
            range=reach,
            height=zr,
            top=top,
            bands=bands,
            shadow=shadow,
            distance=distance,
            departure=departure,
            ratio=ratio,
            carrier=carrier,
            family=np.arange(len(family)),
            delay=delay,
            blur=None if self.eddies is None or shadow else self._coherence_of(family),
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

    def _curved(self, rays, found, zr, corners, planes, listener, r0, r1, twins=None, clear=False):
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
            if clear:
                # The open ground's field along a way with nothing in it.
                share, crossed = np.ones(self.bands), 0
            else:
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
        if across > 1e-9:
            # The start and the heights it was found for go together (the cache holds
            # one start for heights within 1 mm, where the start goes as their roots).
            start, za, zb = self._found(key, za, zb)
            if start is not None and across > start:
                radius = equivalent_radius(start, za, zb, profile)
                k = 2.0 * np.pi * self.centres / float(profile.c[0])
                ell = (radius / (2.0 * k**2)) ** (1.0 / 3.0)
                kappa = np.sqrt(k**2 - _FIRST_MODE * np.exp(1j * np.pi / 3.0) / ell**2 + 0j)
                gain = np.exp(-2.0 * np.abs(kappa.imag) * (across - start))
                if self.eddies is not None:
                    scattered = self.eddies.scattered(profile, za, zb, across, self.centres, start)
                    gain = np.minimum(gain + scattered, 1.0)
        return gain, shift

    def _start(self, key: float, za: float, zb: float) -> float | None:
        return self._found(key, za, zb)[0]

    def _found(self, key: float, za: float, zb: float) -> tuple:
        """The start of the shadow of profile ``key`` for heights ``za`` and ``zb``
        (:func:`~acoustic_spaces.engine.shadow.shadow_start`), as the cache holds it for
        those heights to the millimetre, and the heights it was found for."""
        index = (key, round(za, 3), round(zb, 3))
        if index not in self._starts:
            self._starts[index] = (shadow_start(self.profile(key), za, zb), za, zb)
        return self._starts[index]

    # -- behind barriers and buildings ---------------------------------------------------

    def _barrier(
        self,
        k,
        listener,
        towards,
        reach,
        zr,
        paths,
        open_rows,
        edge_rows,
        cutoff,
        roof_rows=(),
        under=None,
    ):
        """Behind barriers and buildings in the weather: the paths in the vertical plane
        through the source and the listener (``open_rows``, those off the ground alone,
        and those of ``edge_rows`` over edges across the plane, see :meth:`_in_plane`)
        keep their level in still air, which the weather changes by the ratio of the
        parabolic equation's sound in it to that in still air, both with the barriers
        and buildings on the way inside (:meth:`_cuts_for`, :meth:`_place_barrier`).
        The plane takes the specular paths off roofs and the ground alone
        (``roof_rows``) too, and the paths over edges that reflect off roofs, which
        the runs hold; where the source or the listener stands over a building's roof,
        the runs start from the roof or end over it.
        With ``under`` (the buildings under the source and the listener, :meth:`_under`)
        the runs hold those alone and the plane is all of ``edge_rows``: the runs a lit
        listener there takes with nothing else on its way, for the paths round ends
        (:meth:`_ends`).
        Returns those paths (indices) and the request; None where the way ``reach``
        along ``towards`` from the source, seen from above (the listener's, or the way
        :meth:`_edge_side` gives in line with an end or a corner), crosses no barrier or
        building, an object of the model file stands in the listener's straight line,
        the source or the listener stands inside a building (over its footprint, not
        above its roof), the listener is nearer than
        the open ground's parabolic equation goes (``WAVE_MIN_RANGE``), stands on a
        barrier's line or a building's wall seen from above (``ON_OBSTACLE_M``), the
        paths have no order, or none lies in the plane."""
        layout = self._layout(under)
        if not layout.barriers and not layout.buildings:
            return None
        if self.max_order < 1 or reach < WAVE_MIN_RANGE * (self.zs + zr):
            return None
        # What stands on the way short of the listener, out past it to the ground's edge:
        # the building a listener stands on keeps its far wall where it is.
        x0, y0, x1, y1 = layout.extent
        edge = max(reach, math.hypot(x1 - x0, y1 - y0))
        own = self._above(layout.obstacles_along(self.source[:2], towards, edge))
        own = tuple(o for o in own if o[0] < reach)
        if not own or _at(own, reach):
            return None
        ends = np.array([self.source, listener])
        for _, footprint, roof in self.layout.buildings:
            if np.any(in_polygon(ends[:, :2], footprint) & (ends[:, 2] <= roof + 1e-9)):
                return None
        if self._blocked(listener):
            return None
        plane = list(open_rows) + list(roof_rows)
        if under is not None:
            plane += list(edge_rows)
        else:
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
        cuts = self._cuts_for(towards, reach, zr, own, open_top, under)
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

    def _cuts_for(self, towards, reach: float, zr: float, own: tuple, open_top: float, under=None):
        """The runs whose weather's effects, weighed (in dB), give a listener ``reach``
        away along ``towards`` and ``zr`` up the effect behind what stands on its own way
        (``own``, see :meth:`_cut`). Runs along whole degrees (``AZIMUTH_STEP_DEG``) are
        shared by every listener near them: the nearest alone over the middle
        ``1 - BARRIER_BLEND`` of the step between two, and both, weighed linearly, over
        the ``BARRIER_BLEND`` round the halfway direction, so that the effect changes
        smoothly from one to the next (with ``under``, along each only the buildings
        ``under``, :meth:`_cut`). A direction counts for nothing where more or
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
            cut = self._cut(azimuth, under)
            short = [o for o in cut if o[0] < reach]
            if len(short) != len(own) or _at(cut, reach):
                continue
            fit = (reach - max(far for _, far, _ in short)) / (BARRIER_CLOSE * behind)
            if max(far for _, far, _ in own) > reach:
                # A listener over a roof: over it along this way too, or not at all.
                fit = 1.0 if max(far for _, far, _ in short) > reach else 0.0
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

    def _ends(self, k, listener, zr, paths, rows, cutoff, towards, reach, under=()):
        """Of the paths ``rows`` of ``paths``, those round the ends of barriers and the
        corners of buildings alone (:meth:`_round`), where some of those round the
        same edges reflect off the ground; for each such way, the open ground's runs
        of the parabolic equation along it, unfolded round the edges (its length seen
        from above, the wind's component along it as the lengths of its legs weigh
        it; with weather zones through the weathers along its legs) and the open
        ground's field along it with nothing in the way
        (:meth:`_open_way`), which change its paths in every band as the weather changes
        that field (:meth:`_place_ends`). At an end's shadow boundary its way is the
        listener's own, so its paths change as the direct sound does there.
        Paths round the same edges none of which reflects off the ground keep their
        legs' (their partners off the ground left out, the ratio over the ground
        holds no longer).

        Where the source or the listener stands over a building's roof (``under``, the
        buildings, :meth:`_under`), the field along
        the way is instead the one a lit listener there takes with nothing but the
        buildings under the two on its way: each way's paths (off the ground and the
        roofs, whether or not one reflects off the ground) take the runs of
        :meth:`_barrier` along the listener's way ``towards`` (seen from above, ``reach``
        long) with those buildings alone inside, in every band (:meth:`_scale`), and
        come as much sooner or later as the first of them along its legs. At an end's
        shadow boundary these are the runs the lit side's paths in the plane take, so
        that they jump by the field those carry there. Where :meth:`_barrier` asks for
        none (nearer than ``WAVE_MIN_RANGE``, say), the way's paths off nothing but the
        ground take the open ground's field as above.

        Returns the paths taken (indices), those paths (a part of the listener's) and
        the request; None for none."""
        ways: dict[tuple, list[int]] = {}
        for i in rows:
            edges = self._round(paths, i, roofs=bool(under))
            if edges is not None:
                ways.setdefault(edges, []).append(i)
        roofed = []
        for way, found in list(ways.items()):
            if not under:
                break
            asked = self._barrier(
                k, listener, towards, reach, zr, paths, [], found, cutoff, under=under
            )
            if asked is not None:
                roofed.append(asked)
                del ways[way]
            else:
                ways[way] = [i for i in found if not self._elsewhere(paths, i)]
        ways = {e: rows for e, rows in ways.items() if any(self._grounded(paths, i) for i in rows)}
        if not ways and not roofed:
            return None
        chosen = sorted(i for found in [*ways.values(), *(r for r, _ in roofed)] for i in found)
        at = {i: n for n, i in enumerate(chosen)}
        part = paths[np.array(chosen)]
        roofs = [(np.array([at[i] for i in found]), asked) for found, asked in roofed]
        edges = self._edge_set()
        requested = []
        for way, found in ways.items():
            corners = np.array([self.source[:2], *(edges.start[e][:2] for e in way), listener[:2]])
            tops = np.array([max(edges.start[e][2], edges.end[e][2]) for e in way], dtype=float)
            index = np.array([at[i] for i in found])
            requested.append(self._open_way(index, way, corners, tops, listener, zr, cutoff))
        return chosen, part, _Ends(listener=k, part=-1, ways=requested, roofs=roofs)

    def _straight_rays(self) -> Rays:
        """The rays in still air of the speed of sound the paths in still air take."""
        return _kept(("straight", self.c, self.zs), lambda: Rays(Profile.uniform(self.c), self.zs))

    def _open_way(self, rows, edges, corners, tops, listener, zr, cutoff) -> _Way:
        """The way round ends and corners along the horizontal ``corners`` (from the
        source, past each of ``edges``, to ``listener``), unfolded: the open ground's paths along
        it in the weather, as :meth:`_open_ground` finds them for a listener at its end
        with nothing in the way (the eigenrays, and the request for the open ground's
        parabolic equation under the same rules), and the straight direct sound and
        ground reflection in still air, both by :meth:`_curved`."""
        legs = np.diff(corners, axis=0)
        length = float(np.linalg.norm(legs, axis=1).sum())
        if self.zoned:
            key = self._mix(corners)
        else:
            key = self._key(float(self.wind @ legs.sum(axis=0)) / length)
        rays = self.rays(key)
        cap = MAX_GROUND_BOUNCES if self.max_order >= 1 else 0
        eigen = rays.eigenrays(zr, length, cap, max_time=cutoff)
        one = np.ones(self.bands)
        weather, _, highest = self._curved(
            rays, eigen, zr, corners, (), listener, one, one, clear=True
        )
        straight = self._straight_rays()
        flat = straight.eigenrays(zr, length, min(cap, 1))
        still, _, _ = self._curved(straight, flat, zr, corners, (), listener, one, one, clear=True)
        shadow = len(weather) == 0
        towards = legs[0] / max(float(np.linalg.norm(legs[0])), 1e-300)
        distance = math.hypot(length, float(listener[2]) - float(self.source[2]))
        wave = self._wave(
            0, key, (key, None), towards, length, zr, weather, highest, shadow, distance, one
        )
        if wave is not None:
            top, bands = wave.top, wave.bands
        else:
            top = self._open_run(key, zr, length, highest, shadow)[0]
            bands = np.zeros(self.bands, dtype=bool)
        ground = self._grounds_round(corners)
        return _Way(
            rows,
            key,
            ground,
            self._weathers_round(corners),
            length,
            zr,
            top,
            bands,
            weather=weather,
            still=still,
            wave=wave,
            at=np.asarray(listener, dtype=float),
            corners=np.asarray(corners, dtype=float),
            tops=tops,
            edges=edges,
        )

    def _round(self, paths: EarlyReflections, i: int, roofs: bool = False) -> tuple | None:
        """The edges (indices, in turn) that the path ``i`` of ``paths`` goes round,
        where every one is a vertical edge of a barrier or a building (an end or a
        corner) and it reflects off nothing but the ground (and the buildings' roofs
        with ``roofs``); else None."""
        if paths.edge[i] < 0:
            return None
        if self._elsewhere(paths, i, roofs):
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

    def _elsewhere(self, paths: EarlyReflections, i: int, roofs: bool = False) -> bool:
        """Whether the path ``i`` of ``paths`` reflects off anything but the ground (with
        ``roofs``, but the ground and the buildings' roofs)."""
        faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
        held = self.flat | self._roof_faces() if roofs else self.flat
        return bool(len(faces) and not np.all(held[faces]))

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

    def _cut(self, azimuth: float, under: tuple | None = None) -> tuple:
        """What stands on the way from the source towards ``azimuth`` (degrees from the
        x axis), out to the ground's edge, for the parabolic equation: (near, far, top),
        m along the ground from the source and above it (a barrier where near is far),
        as :func:`~acoustic_spaces.engine.parabolic.march` takes them; of the buildings
        ``under`` alone (:meth:`_under`) if given."""
        index = (azimuth, under)
        if index not in self._cuts:
            angle = math.radians(azimuth)
            direction = np.array([math.cos(angle), math.sin(angle)])
            layout = self._layout(under)
            x0, y0, x1, y1 = layout.extent
            found = layout.obstacles_along(self.source[:2], direction, math.hypot(x1 - x0, y1 - y0))
            self._cuts[index] = self._above(found)
        return self._cuts[index]

    def _layout(self, under: tuple | None):
        """The scene's layout; with ``under`` (indices of its buildings) those buildings
        alone, no barrier."""
        if under is None:
            return self.layout
        buildings = tuple(self.layout.buildings[i] for i in under)
        return dataclasses.replace(self.layout, barriers=(), buildings=buildings)

    def _under(self, listener: np.ndarray) -> tuple:
        """The buildings (indices, in turn) over whose roofs the source or ``listener``
        stands (over the footprint, above the roof)."""
        ends = np.array([self.source, np.asarray(listener, dtype=float)])
        return tuple(
            i
            for i, (_, footprint, roof) in enumerate(self.layout.buildings)
            if np.any(in_polygon(ends[:, :2], footprint) & (ends[:, 2] > roof + 1e-9))
        )

    def _in_plane(self, paths: EarlyReflections, i: int, towards, reach: float) -> bool:
        """Whether the path ``i`` of ``paths`` goes over edges that all cross the vertical
        plane through the source and the listener between them, none vertical, and
        reflects off nothing but the ground and the buildings' roofs."""
        if paths.edge[i] < 0:
            return False
        faces = paths.faces[i] if paths.faces else np.zeros(0, dtype=np.int64)
        if len(faces) and not np.all((self.flat | self._roof_faces())[faces]):
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

    def _roof_faces(self) -> np.ndarray:
        """(faces,) bool: the buildings' roofs, their faces facing straight up, which the
        parabolic equation's runs hold as rigid floors."""
        if self._roofs is None:
            mesh = self.model.room.mesh
            kinds = np.array([_kind(label) == "buildings" for label in mesh.labels] or [False])
            normal = self.model.plane_normal[self.model.face_plane]
            self._roofs = kinds[mesh.face_labels] & (normal[:, 2] >= 1.0 - 1e-9)
        return self._roofs

    def _roof_plane(self, plane: int) -> bool:
        """Whether the plane ``plane`` holds a building's roof."""
        faces = self.model.face_plane == plane
        return bool(np.any(self._roof_faces() & faces))

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
        self._scale(request, pressures, parts[request.listener][request.part])

    def _scale(self, request: _Barrier, pressures: dict, paths: EarlyReflections) -> None:
        """``paths`` changed in place as :meth:`_place_barrier` changes those of
        ``request``."""
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
        paths.energy *= (np.exp(effect) + request.scattered)[None, :]
        paths.delay_s += request.shift

    def _place_ends(self, request: _Ends, pressures: dict, parts: list) -> None:
        """The paths of ``request`` round ends and corners changed, way by way, as the
        open ground's field along the way changes from the straight paths in still air
        to the weather's (:meth:`_turned`): its eigenrays with the way's run of the
        parabolic equation put in (``pressures``, by (the way's id, "wind") its band
        energies, by (the way's id, "wind", "pressure") its pressures and by (the way's
        id, "wind", "fine") those at every frequency its runs took) as :meth:`_place`
        puts the open ground's in for a listener; over a roof as the runs
        with the buildings under the source and the listener inside change the lit
        side's paths (:meth:`_scale`)."""
        part = parts[request.listener][request.part]
        coherent = 0 if part.spectrum is None else part.spectrum.shape[1]
        pieces = []
        for way in request.ways:
            weather = dataclasses.replace(
                way.weather,
                energy=way.weather.energy.copy(),
                spectrum=None if way.weather.spectrum is None else way.weather.spectrum.copy(),
                cosines=[c.copy() for c in way.weather.cosines],
            )
            collapsed = np.zeros(self.bands, dtype=bool)
            if way.wave is not None:
                virtual = [[weather]]
                self._place(
                    way.wave,
                    pressures[(id(way), "wind", "pressure")],
                    pressures[(id(way), "wind")],
                    virtual,
                    way.at[None],
                    pressures.get((id(way), "wind", "fine")),
                    pressures.get((id(way), "wind", "blur")),
                )
                weather = virtual[0][0]
                # Where one path holds the open ground's field (the equation's, or the
                # shadow's above its bands), the paths round the ends are one too.
                collapsed = np.ones(self.bands, bool) if way.wave.shadow else way.wave.bands
            pieces.append(self._turned(part[way.rows], way, weather, collapsed, coherent))
        for rows, roof in request.roofs:
            piece = part[rows]
            self._scale(roof, pressures, piece)
            pieces.append(piece)
        parts[request.listener][request.part] = EarlyReflections.concatenate(pieces, self.bands)

    def _bounces(self, paths: EarlyReflections) -> np.ndarray:
        """(n,) 0 for each of ``paths`` that reflects off no ground, 1 for those that do."""
        return np.array([int(self._grounded(paths, i)) for i in range(len(paths))], dtype=np.int64)

    def _pressure(self, paths, rows, band: int, ref: float) -> np.ndarray:
        """(M,) the pressure of ``paths`` ``rows`` together at the early paths'
        frequencies of ``band`` (a coherent one), time measured from ``ref`` (s)."""
        f = self.frequencies[band]
        level = np.sqrt(paths.energy[rows, band])[:, None]
        late = (paths.delay_s[rows] - ref)[:, None]
        return np.sum(level * paths.spectrum[rows, band] * np.exp(-2j * np.pi * f[None] * late), 0)

    def _blocks(self, way: _Way, points: np.ndarray) -> bool:
        """Whether the path through ``points`` (along ``way`` unfolded) passes one of
        its edges below the top of what the edge belongs to, which stands in its way
        there (in still air the paths round the edges make up for it)."""
        return bool(np.any(_unfolded(points, way.corners)[1] < way.tops))

    def _edge_shares(self, way: _Way, ends, rows, shift: np.ndarray, straight: float):
        """(len(rows), bands, M) for each of the paths ``rows`` of ``ends`` round the
        edges of ``way``, at the early paths' frequencies: the share of each edge that
        the integral along it gives a path touching it ``shift`` (k,) m higher than the
        path does over the share it gives the path itself, the product over the edges
        (:func:`~acoustic_spaces.engine.diffraction.along_edge`). Each end that fades
        fades all the way, its integrand that of the incident wave's term, whose shadow
        boundary lies sqrt(2 k d / pi) away in the integral's argument, d the path's
        length over ``straight``'s (m, its straight partner's): on the boundary both
        shares are 1 inside the edge, so that the paths jump there by all of the field;
        a few Fresnel zones inside an end both are 1; past a barrier's top the share goes
        to 0 continuously. An end that does not fade
        (:func:`~acoustic_spaces.engine.diffraction.end_fades`: the edge goes on there,
        in another edge, at a box's corner or as its image in the ground) cuts nothing
        off: a point moved past it counts as at it, so that the copies stop where the
        paths round the edge stop, where those in still air do."""
        from acoustic_spaces.engine.diffraction import _fade_reach, along_edge, end_fades

        edges = self._edge_set()
        if edges.fade is None or edges.corner is None or edges.closed is None:
            edges.fade, edges.corner, edges.closed = end_fades(self.model, edges)
        k = 2.0 * np.pi * self.frequencies / self.c  # (bands, M)
        k_min = float(k.min())
        out = np.ones((len(rows), *k.shape), dtype=complex)
        for n, e in enumerate(way.edges):
            if shift[n] == 0.0:
                continue
            direction = edges.direction[e]
            fade = edges.fade[e]
            fading = np.tile(fade > 0.0, (2, 1, 1))  # (2, 1, 2)
            low = -np.inf if fade[0] > 0.0 else 0.0
            high = np.inf if fade[1] > 0.0 else float(edges.length[e])
            inside = np.ones((2, 1, 2), dtype=bool)
            for m, i in enumerate(rows):
                points = np.asarray(ends.points[i], dtype=float)
                inner = points[1:-1]
                index = np.full(len(inner), e)
                c = 1 + int(np.argmin(_edge_distance(edges, index, inner)))
                legs = np.linalg.norm(np.diff(points, axis=0), axis=1)
                r1, r2 = float(legs[:c].sum()), float(legs[c:].sum())
                incoming = (points[c] - points[c - 1]) / max(float(legs[c - 1]), 1e-300)
                sin_beta = float(np.linalg.norm(np.cross(incoming, direction)))
                own = float((points[c] - edges.start[e]) @ direction)
                t_k = np.clip([own, own + shift[n] * float(direction[2])], low, high)
                length = np.full(2, float(edges.length[e]))
                # sin^2 beta sqrt(k / (pi L)), L = r1 r2 / (r1 + r2) sin^2 beta.
                scale = sin_beta * np.sqrt(k * (r1 + r2) / (np.pi * r1 * r2))
                scale = np.broadcast_to(scale, (2, *k.shape))
                reach = _fade_reach(np.full(2, r1), np.full(2, r2), np.full(2, sin_beta), k_min)
                boundary = np.sqrt(2.0 * k * max(r1 + r2 - straight, 0.0) / np.pi)
                boundary = np.broadcast_to(boundary, (2, 1, *k.shape))
                share = along_edge(
                    t_k,
                    length,
                    scale,
                    fading.astype(float)[:, 0],
                    reach,
                    inside=inside,
                    alone=fading,
                    boundary=boundary,
                )[:, 0]  # (2, bands, M): the one wave's
                held = share[0] != 0.0
                out[m] *= np.divide(share[1], share[0], out=np.ones_like(share[1]), where=held)
        return out

    def _turned(self, ends, way: _Way, weather, collapsed, coherent: int) -> EarlyReflections:
        """The paths ``ends`` round the edges of ``way`` changed as the open ground's
        field along the way unfolded changes from the straight direct sound and ground
        reflection in still air (``way.still``, the field the paths round the edges were
        found against) to ``weather`` (the weather's), so that at an end's shadow
        boundary, where they jump by the field the edges take away, they jump by the
        weather's: a vertical edge in air that changes with height alone diffracts each
        of its vertical modes alike. Each copy of them goes along the way at the height
        of the path of ``weather`` it stands for, as the eddies see it.

        In the bands ``collapsed``, where one path holds the weather's field (the
        equation's, or the shadow's), they become one path along that one's way: where
        among them are paths of each kind the straight ones are of (off the ground or
        not), their pressures together at each frequency of the coherent bands times
        the weather's over the straight paths', which they then follow at each of them,
        and so at every frequency the way's runs took (:meth:`_turned_fine`);
        where not (an end below the line of sight: the direct sound passes over the
        top), their energy times the weather's band energy over the straight paths' (a
        frequency where those cancel would otherwise come out without bound), their own
        spectrum kept. In the other bands each path of ``weather`` takes a copy of those
        of ``ends`` of its kind, times its pressure over the straight path's of that kind
        and the share of the edges its height at them gives over its straight partner's
        (:meth:`_edge_shares`), which goes to 0 as it rises past a barrier's top; where none
        of its kind goes round the edges, one that the edges block (:meth:`_blocks`)
        takes a copy of those of the other kind. Each copy is delayed as much as the
        weather's paths it stands for are against the straight ones; in the bands that
        add in energy, the energies change likewise. Their legs' own times and shadows
        where the way has no open ground's field (``weather`` empty)."""
        bands = self.bands
        still = way.still
        if not len(weather) or not len(still):
            return self._legs(ends)
        kinds_end, kinds_still = self._bounces(ends), self._bounces(still)
        kinds_weather = self._bounces(weather)

        def copy(rows_end, rows_weather, rows_still, which, along, whole, shares=None):
            ref_end = int(rows_end[np.argmin(ends.delay_s[rows_end])])
            t_end = float(ends.delay_s[ref_end])
            t_weather = float(weather.delay_s[along])
            t_still = float(still.delay_s[rows_still].min())
            new = ends[np.array([ref_end])]
            new.delay_s = new.delay_s + (t_weather - t_still)
            new.points = [_unfolded(weather.points[along], way.corners)[0]]
            new.energy = np.zeros((1, bands))
            if new.spectrum is not None:
                new.spectrum = np.ones_like(new.spectrum)
            if new.cosines:
                # Their phases are the open ground's now, not their faces'.
                new.cosines = [np.full(len(new.cosines[0]), np.nan)]
            for band in np.flatnonzero(which):
                e_weather = float(weather.energy[rows_weather, band].sum())
                e_still = float(still.energy[rows_still, band].sum())
                if e_still <= 0.0 or e_weather <= 0.0:
                    continue
                if band >= coherent:
                    e_rows = ends.energy[rows_end, band]
                    if shares is not None:
                        e_rows = e_rows * np.mean(np.abs(shares[:, band]) ** 2, axis=1)
                    new.energy[0, band] = float(e_rows.sum()) * e_weather / e_still
                    continue
                if shares is None:
                    p_end = self._pressure(ends, rows_end, band, t_end)
                else:
                    p_end = sum(
                        self._pressure(ends, [r], band, t_end) * shares[m, band]
                        for m, r in enumerate(rows_end)
                    )
                p_weather = self._pressure(weather, rows_weather, band, t_weather)
                p_still = self._pressure(still, rows_still, band, t_still)
                if not np.all(np.abs(p_still) > 0.0):
                    p = p_end * math.sqrt(e_weather / e_still)
                elif whole:
                    p = p_end * p_weather / p_still
                else:
                    # The weather's turn of the field's phase at each frequency, its
                    # level the band's: where the straight paths all but cancel, their
                    # ratio would come out without bound.
                    ratio = p_weather / p_still
                    size = np.abs(ratio)
                    turn = np.where(size > 0.0, ratio / np.where(size > 0.0, size, 1.0), 1.0)
                    level = np.mean(np.abs(p_weather) ** 2) / np.mean(np.abs(p_still) ** 2)
                    p = p_end * turn * math.sqrt(float(level))
                gain = float(np.mean(np.abs(p) ** 2))
                new.energy[0, band] = gain
                if gain > 0.0:
                    new.spectrum[0, band] = p / math.sqrt(gain)
            return new

        out = []
        everything = np.arange(len(weather))
        if collapsed.any():
            heard = everything[weather.energy[:, collapsed].sum(axis=1) > 0.0]
            if len(heard):
                carrier = int(heard[np.argmin(weather.delay_s[heard])])
                whole = set(kinds_still.tolist()) <= set(kinds_end.tolist())
                one = copy(
                    np.arange(len(ends)),
                    everything,
                    np.arange(len(still)),
                    collapsed,
                    carrier,
                    whole,
                )
                # Not whole, the weather's turn of the phase swings near the straight
                # paths' cancellation by up to 118 degrees from one of the way's samples
                # to the next (177 Hz apart at 2 kHz with 8 an octave), which drawn
                # linearly loses up to 5.8 dB between them: there the early paths'
                # frequencies alone, drawn in magnitude and phase.
                fine = (
                    self._turned_fine(ends, weather, carrier, collapsed, still) if whole else None
                )
                if fine is not None:
                    one.fine, one.fine_delay = fine[0][None], np.array([fine[1]])
                out.append(one)
        rest = ~collapsed
        if rest.any():
            for i in everything:
                if not np.any(weather.energy[i, rest] > 0.0):
                    continue
                kind = int(kinds_weather[i])
                shares = None
                if np.any(kinds_end == kind):
                    rows_still = np.flatnonzero(kinds_still == kind)
                    if not len(rows_still):
                        continue
                    # The edges give it as much as they give the paths round them of its
                    # kind, touching them as much higher as it passes them higher than
                    # its straight partner (:meth:`_edge_shares`).
                    partner = int(rows_still[np.argmin(still.delay_s[rows_still])])
                    shift = (
                        _unfolded(weather.points[i], way.corners)[1]
                        - _unfolded(still.points[partner], way.corners)[1]
                    )
                    rows_end = np.flatnonzero(kinds_end == kind)
                    corners = still.points[partner]
                    straight = float(np.linalg.norm(np.diff(corners, axis=0), axis=1).sum())
                    shares = self._edge_shares(way, ends, rows_end, shift, straight)
                else:
                    # None of its kind goes round the edges (an end below the line of
                    # sight of its straight path): where the edges block it, a vertical
                    # edge diffracts the other kind alike.
                    if not self._blocks(way, weather.points[i]):
                        continue
                    kind = int(kinds_end[0])
                    rows_still = np.flatnonzero(kinds_still == kind)
                    if not len(rows_still):
                        continue
                    rows_end = np.flatnonzero(kinds_end == kind)
                out.append(copy(rows_end, np.array([i]), rows_still, rest, i, True, shares))
        return EarlyReflections.concatenate(out, bands)

    def _turned_fine(self, ends, weather, carrier: int, collapsed, still):
        """The :attr:`~acoustic_spaces.engine.early.EarlyReflections.fine` samples (C, K)
        and their reference delay (s) of the one path that the paths ``ends`` round the
        edges become in the bands ``collapsed`` where they are of each kind the straight
        paths ``still`` are of (:meth:`_turned`), or None: at each frequency f where the
        way's ``carrier`` of ``weather`` has the equation's sample F(f), the paths round
        the edges together p_end(f) times F(f) over the straight paths' p_still(f), as
        :meth:`_turned` takes them at the early paths' frequencies. Each path's pressure
        is drawn between the early paths' frequencies (:meth:`_pressure_at`). None in a
        band where the straight paths cancel at one of the early paths' frequencies (its
        energy alone changes there). The times are those of :meth:`_turned`'s path: p_end
        from the earliest of ``ends``, p_still from the earliest of ``still``, F from the
        carrier's own delay."""
        if weather.fine is None or not np.isfinite(weather.fine[carrier].real).any():
            return None
        carried = weather.fine[carrier]
        layout = fine_frequencies(self.bands, carried.shape[0])
        rows_end, rows_still = np.arange(len(ends)), np.arange(len(still))
        t_end, t_still = float(ends.delay_s.min()), float(still.delay_s.min())
        out = np.full(carried.shape, np.nan, dtype=complex)
        for band in np.flatnonzero(collapsed[: carried.shape[0]]):
            has = np.isfinite(carried[band].real)
            if not has.any() or not np.all(
                np.abs(self._pressure(still, rows_still, band, t_still)) > 0.0
            ):
                continue
            f = layout[band][has]
            p_still = self._pressure_at(still, rows_still, f, t_still)
            if np.all(np.abs(p_still) > 0.0):
                p_end = self._pressure_at(ends, rows_end, f, t_end)
                out[band, has] = p_end * carried[band][has] / p_still
        if not np.isfinite(out.real).any():
            return None
        return out, reference_delay(out, self.bands)

    def _pressure_at(self, paths, rows, f: np.ndarray, ref: float) -> np.ndarray:
        """(len(f),) the pressure of ``paths`` ``rows`` together at frequencies ``f``
        (Hz), time measured from ``ref`` (s), as :meth:`_pressure` gives it at the early
        paths' frequencies: each path's there, in every coherent band, its delay taken
        out, drawn between them in magnitude and unwrapped phase linearly in log
        frequency (as :func:`~acoustic_spaces.engine.coherent.shaped` draws a band's)
        and held beyond the first and the last, then delayed by its own delay."""
        count = paths.spectrum.shape[1]
        at = np.log2(self.frequencies[:count].ravel())
        x = np.log2(np.clip(f, 2.0 ** at[0], 2.0 ** at[-1]))
        out = np.zeros(len(f), dtype=complex)
        for r in rows:
            values = (np.sqrt(paths.energy[r, :count, None]) * paths.spectrum[r]).ravel()
            size = np.interp(x, at, np.abs(values))
            phase = np.interp(x, at, np.unwrap(np.angle(values)))
            out += size * np.exp(1j * phase - 2j * np.pi * f * (paths.delay_s[r] - ref))
        return out

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
        index = (face, band, len(frequencies), float(frequencies[0]))
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

    def _weathers_round(self, corners: np.ndarray) -> tuple | None:
        """With weather zones, the legs of the horizontal polyline ``corners`` (m + 1, 2)
        unfolded, as the parabolic equation marches through the weathers along them
        (:meth:`_weathers_on`): ((distance to the leg's start, its start x, y, its
        direction rounded as a listener's run's is, :meth:`_azimuth`), ...), so that ways
        alike share their runs; None without weather zones."""
        if not self.zoned:
            return None
        corners = np.asarray(corners, dtype=float)
        legs = np.diff(corners, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        out = []
        start = 0.0
        for corner, leg, length in zip(corners[:-1], legs, lengths, strict=True):
            if length > 1e-12:
                azimuth = self._azimuth(leg / length)
                out.append((start, float(corner[0]), float(corner[1]), azimuth))
            start += float(length)
        return tuple(out)

    def _weathers_on(self, weathers: tuple, reach: float):
        """A function giving the speed function (of the height) of the weather at a
        distance along a way whose legs are ``weathers`` (:meth:`_weathers_round`), up
        to ``reach``: each leg's from :meth:`~acoustic_spaces.engine.weather.Weather.
        along`, as a listener's run along a line takes it."""
        starts = np.array([leg[0] for leg in weathers])
        ends = np.append(starts[1:], max(float(reach), float(starts[-1])))
        legs = []
        for (start, x, y, azimuth), end in zip(weathers, ends, strict=True):
            angle = math.radians(azimuth)
            direction = np.array([math.cos(angle), math.sin(angle)])
            legs.append(self.weather.along(np.array([x, y]), direction, end - start))

        def speed_at(r: float):
            leg = max(int(np.searchsorted(starts, r, side="right")) - 1, 0)
            return legs[leg](r - float(starts[leg]))

        return speed_at

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
        runs: dict | None = None,
        blur: np.ndarray | None = None,
    ):
        """Puts the open ground's parabolic equation in the paths of ``wave``'s listener:
        its ``pressure`` (bands, M) at the early paths' frequencies and its band
        ``energy`` (bands,) over all the frequencies it ran at; its pressure at every
        frequency it ran at (``runs`` by band, :func:`solve`) in the carrier's
        :attr:`~acoustic_spaces.engine.early.EarlyReflections.fine` (:meth:`_fine`);
        with eddies its band energy blurred by ``blur`` (:meth:`_blurred`)."""
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
        # The one place the carrier's fine samples are filled: the coherent field, which
        # the eddies' blur (:meth:`_blurred`) leaves as it is. None in the shadow, where
        # the bands keep their own samples: the bands above the equation's deepen there
        # from the octave below and are not continuous with its field (18 to 36 dB above
        # it at 3.4 kHz upwind in the windy preset 100 m away and on the sunny day 300 m
        # away; docs/validation.md, Known issues).
        fine = self._fine(wave, runs) if coherent and runs and not wave.shadow else None
        found = parts[wave.listener]
        family = found[0]
        if blur is not None and wave.carrier >= 0:
            energy = self._blurred(energy, family[wave.family], bands, blur)
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
            if fine is not None:
                if family.fine is None:
                    family.fine = np.full((len(family), *fine[0].shape), np.nan, dtype=complex)
                    family.fine_delay = np.zeros(len(family))
                family.fine[wave.family] = np.nan
                family.fine_delay[wave.family] = 0.0
                family.fine[wave.carrier], family.fine_delay[wave.carrier] = fine
            if family.cosines:
                # The parabolic equation's pressure has no reflection phase to turn.
                family.cosines[wave.carrier] = np.full(len(family.cosines[wave.carrier]), np.nan)

    def _fine(self, wave: _Wave, runs: dict) -> tuple[np.ndarray, float]:
        """The carrier's :attr:`~acoustic_spaces.engine.early.EarlyReflections.fine`
        (coherent, K) and its :attr:`~acoustic_spaces.engine.early.EarlyReflections.
        fine_delay` (s): the equation's pressure at every frequency its runs took (``runs``
        by band, as lists of (frequencies, pressures)) relative to the carrier's delay,
        in the bands it gives with a sound."""
        coherent = self.coherent
        kept = {
            band: pieces
            for band, pieces in runs.items()
            if band < coherent and any(np.any(np.abs(p) > 0.0) for _, p in pieces)
        }
        fine = placed(self.bands, coherent, kept, wave.delay)
        return fine, reference_delay(fine, self.bands)

    def _coherence_of(
        self, rays: EarlyReflections
    ) -> Callable[[int, np.ndarray], np.ndarray | None] | None:
        """The coherence factor of the family ``rays`` as the eddies give it, as a
        function ``blur(band, f)`` (len(f),) of frequencies f (Hz) in a band: each two
        rays' (:meth:`~acoustic_spaces.engine.turbulence.Wander.coherence`) weighted by
        sqrt(E_i E_j) of their energies in the band, or None in a band where they carry
        nothing; None for fewer than two rays."""
        if len(rays) < 2:
            return None
        wander = self.eddies.wander(rays)
        off = ~np.eye(len(rays), dtype=bool)
        energy = rays.energy.copy()

        def blur(band: int, f: np.ndarray) -> np.ndarray | None:
            alone = energy[:, band]
            weights = np.sqrt(np.outer(alone, alone))[off]
            if weights.sum() <= 0.0:
                return None
            return weights @ wander.coherence(f)[off] / weights.sum()

        return blur

    def _blurred(
        self, energy: np.ndarray, rays: EarlyReflections, bands: np.ndarray, blur: np.ndarray
    ) -> np.ndarray:
        """The parabolic equation's ``energy`` (bands,) in still air, in the ``bands``
        it gives, as the eddies blur it at each frequency f its run took: E(f) = C(f)
        E_pe(f) + (1 - C(f)) E_rays, E_rays the family's ``rays`` added in energy and
        C(f) their coherence factor (:meth:`_coherence_of`), averaged over the band as
        the rays' own bands average their pairs' cross terms
        (:func:`~acoustic_spaces.engine.coherent.total_energy`). ``blur`` (bands, 2)
        holds for each band the run's mean of C(f) |P(f)|² over that of |P(f)|², and its
        mean of C(f) (:func:`solve`); NaN where it has none."""
        out = energy.copy()
        for band in np.flatnonzero(bands):
            effective, mean = blur[band]
            if np.isfinite(mean):
                out[band] = effective * energy[band] + (1.0 - mean) * rays.energy[:, band].sum()
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
                pressures.get((id(request), "fine")),
                pressures.get((id(request), None, "blur")),
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
    obstacles: tuple = ()
    #: Whether the air bends sound (None: as the march finds).
    refracting: bool | None = None
    barrier: bool = False  # with the grid and samples of the runs behind barriers
    #: What the run has in common with those that differ from it in what stands on the
    #: way and in their listeners alone (the source's runs behind barriers in one band
    #: and weather, under one absorbing layer), which share the march over the open
    #: ground (:class:`~acoustic_spaces.engine.parabolic.Trunk`); None: none do.
    family: tuple | None = None
    #: Over the open ground: the run on past the band's own frequencies through its
    #: upper crossover, at their spacing, for the listeners and the ways round ends whose
    #: top band of the equation it is (:func:`~acoustic_spaces.engine.fine.skirt_frequencies`).
    skirt: bool = False
    #: Over the open ground, in the lowest band: the run takes the frequencies under the
    #: band too, at their spacing, on the band's own grid (its highest frequency's), so
    #: that the field goes on from the band's continuous.
    below: bool = False


def _open_wave(member) -> _Wave | None:
    """The open ground's request of a run's ``member``: a listener's own, a way's round
    ends (None where the rays alone give it), None behind barriers."""
    if isinstance(member, _Wave):
        return member
    return member.wave if isinstance(member, _Way) else None


def _skirt(refraction: Refraction, band: int, below: bool = False) -> np.ndarray:
    """The frequencies (Hz) of ``band``'s skirt over the open ground: above its own, or
    ``below`` them (:func:`~acoustic_spaces.engine.fine.skirt_frequencies`)."""
    dense = refraction._samples(band, False) is not None
    return skirt_frequencies(refraction.bands, band, dense, below)


def solve(plans: list[Planned]) -> list[dict]:
    """Runs the parabolic equation for ``plans`` (:meth:`Refraction.plan`): for each,
    the pressure (bands, M) of each of its requests by the request's id, and the band
    energies (bands,) of the runs for the paths behind barriers by (the id of the cut,
    "wind" or "still") and round ends by (the id of the way, "wind"); by (the request's
    id, "fine") and (the way's id, "wind", "fine"), out of the shadow, for each band the
    open ground's pressure at every frequency its runs took (:meth:`Refraction._fine`):
    the band's own, under the lowest band those down to an octave below it, and through
    the upper crossover of the request's or the way's top band of the equation those
    above, where a response is rendered or the band filters read that crossover; with
    eddies, by (the request's id, None, "blur") and (the way's id, "wind", "blur"), the
    two numbers (bands, 2) of each band that the eddies' blur takes from the band's own
    frequencies (:meth:`Refraction._blurred`). A
    source's listeners that the sound goes the same way to share a run (the same profile;
    with weather or ground zones the same direction too). So do several sources' at one
    height over one ground, where the weather and the grounds have no zones (the places
    of a walk, say): there a run from one source holds for any other alike, the way along
    the ground being the same from each. A run marches out to the farthest of its
    listeners under an absorbing layer above the highest of their ``top``; each listener
    is reached by a last step of its own, off to the side of the march, so that it gets
    what a march to it alone under that layer gives, whoever shares the run.

    Behind barriers the runs in the weather share by the profile and what stands on the
    way (the cut), those in still air by the cut and the absorbing layer alone; the open
    ground's runs along the ways round ends, in the weather alone (the straight paths
    are their reference in still air), likewise with no cut, and with weather zones
    through the weathers along the way's legs (:meth:`Refraction._weathers_on`), with
    their skirts as a listener's (one run above a band for every way of the profile
    whose top band it is). Every listener of theirs is reached by a last step of its
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
        # The runs behind barriers and buildings, and those with the buildings under the
        # source and the listener inside for the paths round ends over a roof.
        roofs = [roof for request in planned.ends for _, roof in request.roofs]
        for request in [*planned.barriers, *roofs]:
            for cut in request.cuts:
                azimuth = cut.azimuth if zones else None
                behind.setdefault((owner, True, cut.key, azimuth, cut.cut), []).append((p, cut))
        for request in planned.ends:
            for way in request.ways:
                # With weather zones, through the weathers along the way, as a listener's
                # run goes through those along its line.
                key = way.key if way.weathers is None else way.weathers
                behind.setdefault((owner, False, key, way.ground, ()), []).append((p, way))

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

    def open_ground(refraction, band, needing, job):
        # The open ground's runs in ``band`` for ``needing`` (a listener's requests or
        # the ways round ends), made by ``job``, and their skirts, none in the shadow,
        # where the carrier keeps no fine samples (:meth:`Refraction._place`): under the
        # lowest band, which its band filter reads down to an octave below it, in the
        # band's own run; through the upper crossover of each one's top band, where a
        # coherent band lies above, in a run of its own, where a response is rendered
        # from the spectra and, for a map, where the band filters read that crossover (a
        # top band up to FILTERED_MAX_BAND_HZ).
        lit = [n for n in needing if not _open_wave(n[1]).shadow]
        if needing:
            below = band == 0 and bool(lit) and refraction.coherent > 0
            jobs.append(job(needing, below=below))
        topmost = [n for n in lit if not n[1].bands[band + 1 :].any()]
        read = refraction.rendered or band < band_count(FILTERED_MAX_BAND_HZ, refraction.bands)
        if topmost and read and band + 1 < refraction.coherent:
            jobs.append(job(topmost, skirt=True))

    for (_, (key, azimuth)), members in groups.items():
        refraction = plans[members[0][0]].refraction
        direction = members[0][1].direction if azimuth is None else towards(azimuth)
        reach = max(m.range for _, m in members)
        face_at = refraction._grounds_along(direction, reach)
        speed, speed_at = weather(refraction, key, direction, reach)
        top = max(m.top for _, m in members)
        for band in range(refraction.bands):
            needing = [(p, m, None) for p, m in members if m.bands[band]]
            open_ground(
                refraction,
                band,
                needing,
                functools.partial(_Job, refraction, speed, speed_at, face_at, band, top=top),
            )

    # Behind barriers and along the ways round ends: the runs in the weather, and behind
    # barriers in still air, shared by all with the same cut and absorbing layer.
    calm: dict[tuple, list] = {}
    for (owner, barrier, key, ground, cut), members in behind.items():
        refraction = plans[members[0][0]].refraction
        reach = max(m.range for _, m in members)
        direction, face_at = grounds(refraction, members, reach)
        first = members[0][1]
        if isinstance(first, _Way) and first.weathers is not None:
            speed_at = refraction._weathers_on(first.weathers, reach)
            speed = speed_at(0.0)
        else:
            speed, speed_at = weather(refraction, key, direction, reach)
        top = max(m.top for _, m in members)
        # Behind barriers, in a weather that is one all the way: a family's runs.
        alike = bool(cut) and speed_at is None
        for band in range(refraction.bands):
            needing = [(p, m, "wind") for p, m in members if m.bands[band]]
            job = functools.partial(
                _Job,
                refraction,
                speed,
                speed_at,
                face_at,
                band,
                top=top,
                obstacles=cut,
                refracting=True,
                barrier=barrier,
                family=(owner, barrier, key, top, band) if alike else None,
            )
            if not barrier:
                # Along the ways round ends as for a listener: the skirts too.
                open_ground(refraction, band, needing, job)
            elif needing:
                jobs.append(job(needing))
        if not barrier:
            # Round ends the weather's runs alone: the straight paths are the reference
            # in still air (Refraction._turned).
            continue
        c0 = float(np.asarray(speed(np.array([0.0])))[0])
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
        if job.skirt:
            return _skirt(job.refraction, job.band)
        found = job.refraction._samples(job.band, job.barrier)
        own = job.refraction.frequencies[job.band] if found is None else found
        return np.concatenate([_skirt(job.refraction, job.band, True), own]) if job.below else own

    def run(job: _Job, trunk: parabolic.Trunk | None = None) -> np.ndarray:
        refraction, band = job.refraction, job.band
        dense = job.skirt or job.below or refraction._samples(band, job.barrier) is not None
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
            # Each listener by a last step of its own: what it gets does not depend on
            # the ranges of the others, which would break up the march's steps.
            independent=True,
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
        under = len(_skirt(job.refraction, job.band, True)) if job.below else 0
        for (p, m, slot), value in zip(job.members, values.T, strict=True):
            # The open ground's request, a listener's or a way's round ends.
            wave = _open_wave(m)
            if wave is not None and not wave.shadow:
                # Every frequency the open ground's runs took, for the carrier's fine
                # samples (none in the shadow, :meth:`Refraction._place`).
                kept = (id(m), "fine") if slot is None else (id(m), slot, "fine")
                fine = out[p].setdefault(kept, {}).setdefault(job.band, [])
                fine.append((frequencies_of(job), np.conj(value)))
            if job.skirt:
                continue
            value = value[under:]  # the band's own frequencies
            # The band's energy over every frequency the run took; the pressures at the
            # early paths' own, time as exp(j omega t), as the early paths have it.
            energy = out[p].setdefault((id(m), slot), np.zeros(job.refraction.bands))
            power = np.abs(value) ** 2
            energy[job.band] = np.mean(power)
            if wave is not None and wave.blur is not None:
                # With eddies, what the blur takes at each frequency of the run
                # (Refraction._blurred): two numbers a band, however many it took.
                together = wave.blur(job.band, frequencies_of(job)[under:])
                if together is not None:
                    blur = out[p].setdefault(
                        (id(m), slot, "blur"), np.full((job.refraction.bands, 2), np.nan)
                    )
                    total = float(np.sum(power))
                    effective = float(together @ power) / total if total > 0.0 else 0.0
                    blur[job.band] = effective, float(np.mean(together))
            if slot is None:
                out[p][id(m)][job.band] = np.conj(_early_of(value, shape[1]))
            elif isinstance(m, _Way):
                pressure = out[p].setdefault(
                    (id(m), slot, "pressure"), np.zeros(shape, dtype=complex)
                )
                pressure[job.band] = np.conj(_early_of(value, shape[1]))
    return out


def _kept(index: tuple, make) -> Rays:
    """The rays kept under ``index`` (made by ``make()`` if not), now the most recently
    used; the least recently used beyond ``_KEPT_RAYS`` let go."""
    rays = _RAYS.get(index)
    if rays is None:
        rays = _RAYS[index] = make()
        while len(_RAYS) > _KEPT_RAYS:
            _RAYS.popitem(last=False)
    else:
        _RAYS.move_to_end(index)
    return rays


def _highest(rays: Rays, found, zr: float) -> float:
    """The greatest height (m) the eigenrays ``found`` to a listener ``zr`` up reach."""
    return max(
        (float(rays.curve(zr, found, i)[:, 1].max()) for i in range(len(found))), default=0.0
    )


def _unfolded(points: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The path through ``points`` (n, 3), which lie along the horizontal polyline
    ``corners`` (k + 2, 2) from its first to its last (or go straight from the one to the
    other), along the polyline with a point at each inner corner, at the path's height
    there; and (k,) those heights."""
    points = np.asarray(points, dtype=float)
    run = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1))])
    legs = np.linalg.norm(np.diff(corners, axis=0), axis=1)
    at = np.cumsum(legs)[:-1] * (run[-1] / max(float(legs.sum()), 1e-300))
    heights = np.interp(at, run, points[:, 2])
    order = np.argsort(np.concatenate([run, at]), kind="stable")
    path = np.concatenate([points, np.column_stack([corners[1:-1], heights])])[order]
    return path, heights


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
