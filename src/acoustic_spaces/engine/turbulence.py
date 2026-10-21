"""Turbulence outdoors: eddies of wind and temperature that make sound flicker.

The eddies (:class:`~acoustic_spaces.scene.Turbulence`) make the air's index of
refraction fluctuate, mu = -T' / (2 T0) - v' / c0 for sound going along the wind's
fluctuation v' (Ostashev and Wilson 2015, ch. 6). With the von Kármán spectrum in its
scalar form for both,

    Phi(kappa) = A Cn² (kappa² + 1 / L²)^(-11/6),   A = Gamma(8/3) sin(pi/3) / (4 pi²),
    Cn² = C_T² / (4 T0²) + (11/6) C_v² / c0²,

(the wind's 11/6 is that of the velocity's spectral tensor across the way the sound
goes, 22/3 of the temperature's 1/4), L the size of the largest eddies. In the Markov
approximation (Tatarskii 1971; Ishimaru 1978, ch. 20) two waves of wavenumber k that
travel a length ds a distance rho apart drift apart in phase by a random amount whose
variance grows by k² G(rho) ds, with the structure function

    G(rho) = 8 pi² int kappa Phi(kappa) (1 - J0(kappa rho)) dkappa
           = 8 pi² A Cn² (3/5) L^(5/3) [1 - 2^(1/6) / Gamma(5/6) x^(5/6) K_5/6(x)],

x = rho / L, which for rho much smaller than L is Kolmogorov's 2.91 Cn² rho^(5/3). Two
paths of one sound are taken a distance rho(s) apart at the same share s of their
lengths, and

    D_ij = k² int G(rho_ij(s)) ds,
    <|p_i + p_j|²> = |p_i|² + |p_j|² + 2 exp(-D_ij / 2) Re(p_i p_j*),

the coherence factor of Clifford and Lataitis (1983) for any spectrum and any two paths:
for the direct sound and its reflection off the ground, D = (3/8) 2.91 k² Cn² R
rho_max^(5/3) with rho_max = 2 h_s h_r / (h_s + h_r) where L is large.

Near the ground the parabolic equation marched through random eddies
(``validation/turbulence.py``) fills the dip between the direct sound and its
reflection far less than that where the eddies that part them, about as big as their
largest separation rho, span few wavelengths: two paths that differ only by reflections
off the ground take D_ij times :func:`ground_share` (k rho), 1 / (1 + (32 / (k rho))^1.7),
fitted to it over five heights and five grounds from 50 to 400 m (the level within 0.7
dB of it on average, 1.9 dB without; most where the dip is deepest).

D_ij is (2 pi f)² times a variance V_ij that depends on the paths alone (s²): the eddies
delay each path at random, and V_ij is the variance of the difference of two paths'
delays (:class:`Wander`). A moment of the turbulent air, as an impulse response has it,
draws those delays: in each band, the symmetric square root of the delays' centred Gram
matrix (classical multidimensional scaling, exact as G is a variogram) times normal
variables shared by the bands.

The eddies may weaken with height (``height_m``, ``falloff``): Cn² at a height z is its
value at ``height_m`` times (z / height_m)^-falloff, as the surface layer's structure
parameters go, C_T² and C_v² both as z^(-2/3) in neutral and stable air and C_T² as
z^(-4/3) in free convection (Monin-Obukhov similarity; Wyngaard, Izumi and Collins
1971). Nearer the ground than the eddies' outer scale the engine takes Cn² at the outer
scale (:meth:`Eddies.strength`): the eddies that part two paths or scatter sound are as
big as that and reach that high, and the parabolic equation through a field that keeps
strengthening right down to the ground shows the thin layer of stronger eddies there
does not tell (the waves average over it; see ``validation/turbulence.py --profile``).
Two paths' D_ij then takes Cn² at the mean height of the two points a distance rho apart
(the eddies between them), the single scattering Cn² at the scattering height, and the
shaking Cn² at the geometric mean of the source's and the listener's heights, where the
refraction is stirred.

Into a shadow of upward refraction the eddies scatter sound (:meth:`Eddies.scattered`):
once, from the air that the source and the listener both see over the shadow's edge
(the Born approximation, the scattering cross-section 2 pi k^4 Phi(2 k sin(theta / 2))
per unit volume), and at high frequencies most by shaking the refraction near the ground
that makes the shadow, by a share of the free field's energy that grows as the square of
the eddies' speed of sound against its fall over the heights.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from acoustic_spaces.engine.early import EarlyReflections
    from acoustic_spaces.engine.refraction import Profile

#: The von Kármán spectrum's constant A.
SPECTRUM_CONSTANT = math.gamma(8.0 / 3.0) * math.sin(math.pi / 3.0) / (4.0 * math.pi**2)
#: Gauss-Legendre samples along two paths, for their separation.
SAMPLES = 24
#: Paths are taken this many at a time (memory: rows x paths x SAMPLES).
_ROWS = 64
#: Two paths that differ only by reflections off the ground take D times
#: 1 / (1 + (GROUND_K_RHO / (k rho))^GROUND_POWER), rho their largest separation (fitted to
#: the parabolic equation through random eddies, ``validation/turbulence.py``).
GROUND_K_RHO = 32.0
GROUND_POWER = 1.7
#: The index of refraction's variance over Cn² L^(2/3) in the von Kármán spectrum:
#: 4 pi A Gamma(3/2) Gamma(1/3) / (2 Gamma(11/6)).
_VARIANCE = (
    4.0
    * math.pi
    * SPECTRUM_CONSTANT
    * math.gamma(1.5)
    * math.gamma(1.0 / 3.0)
    / (2.0 * math.gamma(11.0 / 6.0))
)
#: The two-dimensional spectrum's constant over the three-dimensional one's: Phi
#: integrated across the plane of the source and the listener.
_PLANE = math.sqrt(math.pi) * math.gamma(4.0 / 3.0) / math.gamma(11.0 / 6.0)
#: The share of the free field's energy that eddies shaking the refraction bring deep into
#: a shadow at high frequencies, over (sigma_c / delta_c)², sigma_c the eddies' rms speed
#: of sound and delta_c its fall from the ground to the heights (fitted to the parabolic
#: equation through random eddies deep in the shadows of a sunny day and of the wind,
#: ``validation/turbulence.py``: within 1 dB of it on average, 3.5 dB at most).
SHAKEN_SHARE = 0.15
#: They do so where the creeping wave's height scale (R / (2 k²))^(1/3) is below
#: SHAKEN_HEIGHT times the source's and the listener's heights' geometric mean: a share
#: 1 / (1 + (scale / (SHAKEN_HEIGHT h))^SHAKEN_POWER) of it.
SHAKEN_HEIGHT = 0.4
SHAKEN_POWER = 6.0
#: The single scattering's grid over the air both see (heights x distances along).
_HEIGHTS = 48
_ALONG = 32

_GAMMA_56 = math.gamma(5.0 / 6.0)
#: h(x) = c x^(5/3) - 1.5 x² near zero (Kolmogorov's range).
_SMALL = -math.gamma(-5.0 / 6.0) / (2.0 ** (5.0 / 3.0) * _GAMMA_56)
_TABLE: tuple[np.ndarray, np.ndarray] | None = None


def _table() -> tuple[np.ndarray, np.ndarray]:
    """log x and log h(x) on a grid, h computed from K_5/6(x) = int_0^inf exp(-x
    cosh t) cosh(5 t / 6) dt by the trapezoidal rule (to 1e-9 relative)."""
    global _TABLE
    if _TABLE is None:
        x = np.geomspace(1e-4, 60.0, 600)
        t = np.linspace(0.0, 30.0, 12001)
        values = np.exp(-np.outer(x, np.cosh(t))) * np.cosh(5.0 / 6.0 * t)
        k = (values.sum(axis=1) - 0.5 * (values[:, 0] + values[:, -1])) * (t[1] - t[0])
        h = 1.0 - 2.0 ** (1.0 / 6.0) * x ** (5.0 / 6.0) * k / _GAMMA_56
        _TABLE = (np.log(x), np.log(h))
    return _TABLE


def von_karman(x) -> np.ndarray:
    """h(x) = 1 - 2^(1/6) / Gamma(5/6) x^(5/6) K_5/6(x), x the separation over the outer
    scale: the structure function over its value far apart."""
    x = np.asarray(x, dtype=float)
    log_x, log_h = _table()
    out = np.ones_like(x)
    small = x < 1e-4
    out[small] = _SMALL * x[small] ** (5.0 / 3.0) - 1.5 * x[small] ** 2
    middle = ~small & (x < 60.0)
    out[middle] = np.exp(np.interp(np.log(x[middle]), log_x, log_h))
    return out


def ground_share(k_rho) -> np.ndarray:
    """The share of the Markov approximation's D that two paths differing only by
    reflections off the ground take, k rho their largest separation in wavenumbers."""
    k_rho = np.asarray(k_rho, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / (1.0 + (GROUND_K_RHO / np.maximum(k_rho, 1e-300)) ** GROUND_POWER)


@dataclass(frozen=True)
class Wander:
    """How the eddies delay a set of paths: the variance (s²) of the difference of each
    two paths' delays, ``variance`` (n, n), as rays have it; ``apart`` (n, n) the
    largest separation (m) of two paths that differ only by reflections off the ground
    (0 for the others), which near the ground take less (:func:`ground_share`); the
    speed of sound (m/s); and how long (s) the eddies take to change (``correlation_s``,
    their outer scale over the wind that carries them past)."""

    variance: np.ndarray
    apart: np.ndarray
    speed_of_sound: float
    correlation_s: float = 10.0

    def __len__(self) -> int:
        return len(self.variance)

    def __getitem__(self, rows) -> Wander:
        rows = np.arange(len(self))[rows]
        index = np.ix_(rows, rows)
        return Wander(
            self.variance[index], self.apart[index], self.speed_of_sound, self.correlation_s
        )

    @classmethod
    def stacked(cls, parts: list[Wander | None], sizes: list[int]) -> Wander | None:
        """The parts one after the other, each two paths of different parts apart as
        nothing (they add in energy: different points of a source); None when no part
        has one."""
        known = [p for p in parts if p is not None]
        if not known:
            return None
        n = sum(sizes)
        variance = np.zeros((n, n))
        apart = np.zeros((n, n))
        row = 0
        for part, size in zip(parts, sizes, strict=True):
            if part is not None:
                block = slice(row, row + size)
                variance[block, block] = part.variance
                apart[block, block] = part.apart
            row += size
        return cls(variance, apart, known[0].speed_of_sound, known[0].correlation_s)

    def structure(self, f) -> np.ndarray:
        """(n, n, len(f)) D_ij at frequencies ``f`` (Hz)."""
        f = np.atleast_1d(np.asarray(f, dtype=float))
        share = np.ones(self.apart.shape + f.shape)
        ground = self.apart > 0.0
        if ground.any():
            k = 2.0 * math.pi * f / self.speed_of_sound
            share[ground] = ground_share(self.apart[ground][:, None] * k[None, :])
        return (2.0 * math.pi * f) ** 2 * self.variance[:, :, None] * share

    def coherence(self, f) -> np.ndarray:
        """(n, n, len(f)) the coherence factor exp(-D_ij / 2) of each two paths at
        frequencies ``f`` (Hz)."""
        return np.exp(-0.5 * self.structure(f))

    def roots(self, centres, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each group of paths (those of one point of a source, ``groups`` (n,)
        labels): its rows and (bands, m, m) the symmetric square roots, band by band at
        ``centres`` (Hz), of the centred Gram matrix of its paths' delays."""
        centres = np.asarray(centres, dtype=float)
        variance = self.structure(centres) / (2.0 * math.pi * centres) ** 2  # (n, n, B)
        out = []
        for label in np.unique(groups):
            rows = np.flatnonzero(groups == label)
            m = len(rows)
            square = np.zeros((len(centres), m, m))
            if m > 1:
                block = variance[np.ix_(rows, rows)].transpose(2, 0, 1)  # (B, m, m)
                centring = np.eye(m) - 1.0 / m
                gram = -0.5 * centring @ block @ centring
                values, vectors = np.linalg.eigh(0.5 * (gram + gram.transpose(0, 2, 1)))
                values = np.sqrt(np.maximum(values, 0.0))
                square = (vectors * values[:, None, :]) @ vectors.transpose(0, 2, 1)
            out.append((rows, square))
        return out

    @staticmethod
    def delays(roots, draws: list[np.ndarray], n: int) -> np.ndarray:
        """(n, bands) the paths' delays (s) for normal variables ``draws`` (one array of
        the group's size per group of :meth:`roots`)."""
        bands = roots[0][1].shape[0] if roots else 0
        out = np.zeros((n, bands))
        for (rows, square), xi in zip(roots, draws, strict=True):
            out[rows] = (square @ xi).T
        return out


class Gusts:
    """The eddies drifting past: for each group of paths (each point of a source), a
    normal variable per path that changes over time as an Ornstein-Uhlenbeck process of
    unit variance and correlation time ``correlation_s`` (s), sampled every ``step_s``
    at ``count`` times from 0, drawn with ``seed``. A group's paths take its variables
    in their order of arrival, so that a listener that moves keeps the eddies' state."""

    def __init__(self, correlation_s: float, step_s: float, count: int, seed: int) -> None:
        self.correlation_s = float(correlation_s)
        self.step_s = float(step_s)
        self.count = max(int(count), 1)
        self.seed = int(seed)
        self._columns: dict[tuple[int, int], np.ndarray] = {}

    @property
    def times(self) -> np.ndarray:
        """(count,) the times (s) of the moments."""
        return np.arange(self.count) * self.step_s

    def _column(self, label: int, index: int) -> np.ndarray:
        key = (label, index)
        if key not in self._columns:
            rng = np.random.default_rng([self.seed, label, index])
            kept = math.exp(-self.step_s / self.correlation_s) if self.correlation_s > 0 else 0.0
            noise = rng.standard_normal(self.count)
            out = np.empty(self.count)
            out[0] = noise[0]
            fresh = math.sqrt(1.0 - kept**2)
            for j in range(1, self.count):
                out[j] = kept * out[j - 1] + fresh * noise[j]
            self._columns[key] = out
        return self._columns[key]

    def draws(self, roots, groups: np.ndarray, moment: int) -> list[np.ndarray]:
        """The normal variables of each group of ``roots`` (:meth:`Wander.roots` of
        paths whose groups are ``groups``) at moment ``moment``."""
        moment = min(max(int(moment), 0), self.count - 1)
        out = []
        for rows, _ in roots:
            label = int(groups[rows[0]]) if len(rows) else 0
            out.append(np.array([self._column(label, i)[moment] for i in range(len(rows))]))
        return out


@dataclass(frozen=True)
class Eddies:
    """The turbulence of an outdoor scene: the strength of the index of refraction's
    fluctuations ``cn2`` (Cn², m^-2/3), the outer scale (m), the speed of sound (m/s)
    its delays are reckoned in, and the ground's faces (``flat``, a bool per face of
    the model)."""

    cn2: float
    outer_scale_m: float
    speed_of_sound: float
    flat: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), compare=False)
    #: The eddies' speed (m/s) that carries them past: the time a moment lasts is the
    #: outer scale over it.
    drift_m_s: float = 1.0
    #: The height (m above the ground) ``cn2`` holds at, from which the eddies weaken
    #: with height as (z / height_m)^-falloff; None: the same at every height.
    height_m: float | None = None
    falloff: float = 2.0 / 3.0
    #: The ground's height (z) in the scene's coordinates.
    ground: float = 0.0

    @classmethod
    def of(cls, model, speed_of_sound: float) -> Eddies | None:
        """The eddies of ``model``'s scene (a :class:`~acoustic_spaces.engine.model.
        TraceModel`) outdoors, or None: indoors, without an atmosphere's
        ``turbulence``, or with a calm one."""
        from acoustic_spaces.engine.model import _ground_faces

        room = model.room
        scene = room.scene
        atmosphere = getattr(scene, "atmosphere", None)
        if room.outdoor is None or atmosphere is None:
            return None
        turbulence = atmosphere.turbulence
        if turbulence is None or turbulence.calm:
            return None
        t0 = scene.air.temperature_c + 273.15
        cn2 = (
            turbulence.temperature_structure / (4.0 * t0**2)
            + 11.0 / 6.0 * turbulence.wind_structure / speed_of_sound**2
        )
        wind = atmosphere.wind.speed_m_s if atmosphere.wind is not None else 0.0
        return cls(
            cn2,
            turbulence.outer_scale_m,
            speed_of_sound,
            _ground_faces(room),
            max(float(wind), 1.0),
            turbulence.height_m,
            turbulence.falloff,
            float(room.outdoor.ground),
        )

    def strength(self, height) -> np.ndarray:
        """Cn² at ``height`` (m above the ground) over :attr:`cn2`: 1 everywhere when the
        eddies are the same at every height, else (max(z, outer_scale_m) /
        height_m)^-falloff: no nearer the ground than the eddies' outer scale."""
        height = np.asarray(height, dtype=float)
        if self.height_m is None:
            return np.ones_like(height)
        return (np.maximum(height, self.outer_scale_m) / self.height_m) ** -self.falloff

    # -- coherence ---------------------------------------------------------------------

    def structure(self, rho) -> np.ndarray:
        """G(rho) (m^-1): how fast the phase difference of two waves a distance
        ``rho`` (m) apart grows in variance, per metre and per k² (rad m)²."""
        scale = self.outer_scale_m
        far = 8.0 * math.pi**2 * SPECTRUM_CONSTANT * self.cn2 * 0.6 * scale ** (5.0 / 3.0)
        return far * von_karman(np.asarray(rho, dtype=float) / scale)

    def _apart(self, points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(n, n) the variance (s²) of the difference between the delays of each two of
        the paths through ``points`` (each (m, 3), source first), and (n, n) their
        largest separation (m)."""
        n = len(points)
        nodes, weights = np.polynomial.legendre.leggauss(SAMPLES)
        share = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        samples = np.zeros((n, SAMPLES, 3))
        lengths = np.zeros(n)
        for i, corners in enumerate(points):
            corners = np.asarray(corners, dtype=float)
            legs = np.linalg.norm(np.diff(corners, axis=0), axis=1)
            along = np.concatenate([[0.0], np.cumsum(legs)])
            lengths[i] = along[-1]
            at = share * along[-1]
            for axis in range(3):
                samples[i, :, axis] = np.interp(at, along, corners[:, axis])
        variance = np.zeros((n, n))
        largest = np.zeros((n, n))
        heights = samples[:, :, 2] - self.ground
        for start in range(0, n, _ROWS):
            rows = slice(start, start + _ROWS)
            apart = np.linalg.norm(samples[rows, None] - samples[None], axis=3)
            mean = 0.5 * (lengths[rows, None] + lengths[None])
            growth = self.structure(apart)
            if self.height_m is not None:
                # The eddies between the two points: Cn² at their mean height (no nearer
                # the ground than the outer scale, see strength()).
                growth = growth * self.strength(0.5 * (heights[rows, None] + heights[None]))
            variance[rows] = mean * (growth @ weights)
            largest[rows] = apart.max(axis=2)
        return variance / self.speed_of_sound**2, largest

    def variances(self, points: list[np.ndarray]) -> np.ndarray:
        """(n, n) the variance (s²) of the difference between the delays of each two of
        the paths through ``points`` (each (m, 3), source first), as rays have it."""
        return self._apart(points)[0]

    def wander(self, paths: EarlyReflections) -> Wander:
        """How the eddies delay ``paths``."""
        variance, largest = self._apart(paths.points)
        flat = self.flat
        signature = []
        for i, faces in enumerate(paths.faces or [np.zeros(0, dtype=np.int64)] * len(paths)):
            faces = np.asarray(faces, dtype=np.int64)
            walls = faces[~flat[faces]] if len(flat) and len(faces) else faces
            signature.append((tuple(walls.tolist()), int(paths.edge[i])))
        same = np.array([[a == b for b in signature] for a in signature], dtype=bool)
        np.fill_diagonal(same, False)
        return Wander(
            variance, np.where(same, largest, 0.0), self.speed_of_sound, self.correlation_s
        )

    @property
    def correlation_s(self) -> float:
        """How long (s) the eddies take to change: their outer scale over the wind's
        speed that carries them past."""
        return self.outer_scale_m / self.drift_m_s

    def attach(self, paths: EarlyReflections) -> EarlyReflections:
        """``paths`` with their :attr:`~EarlyReflections.wander`."""
        if len(paths) == 0 or not paths.points:
            return paths
        return dataclasses.replace(paths, wander=self.wander(paths))

    # -- scattering into a shadow ------------------------------------------------------

    @property
    def speed_spread(self) -> float:
        """The eddies' rms speed of sound (m/s), at :attr:`height_m` when they weaken
        with height."""
        return self.speed_of_sound * math.sqrt(
            _VARIANCE * self.cn2 * self.outer_scale_m ** (2.0 / 3.0)
        )

    def scattered(
        self, profile: Profile, zs: float, zr: float, reach: float, centres, start: float
    ) -> np.ndarray:
        """(bands,) the energy the eddies bring to a listener ``zr`` m up, ``reach`` m
        along the ground from a source ``zs`` m up, in the shadow of ``profile`` that
        starts ``start`` m away, over the free field's, at the bands' ``centres`` (Hz):
        once scattered from the air both see, and shaken in by the refraction they
        stir near the ground."""
        centres = np.asarray(centres, dtype=float)
        out = self._single(profile, zs, zr, reach, centres) + self._shaken(
            profile, zs, zr, centres, start
        )
        return np.minimum(out, 1.0)

    def _single(self, profile, zs: float, zr: float, reach: float, centres) -> np.ndarray:
        """Single scattering (Born) from the air above both the source's and the
        listener's limiting rays: R² int 2 pi k³ Phi2(q) / (d1 d2 (d1 + d2)) dx dz in
        the vertical plane, Phi integrated across it, q = 2 k sin(theta / 2)."""
        from acoustic_spaces.engine.refraction import _Fan

        c0 = float(profile.c[0])
        k = 2.0 * math.pi * centres / c0
        fan = _Fan(profile, np.array([1.0 / c0]), 0.0)
        top = float(fan.hi[0]) if np.isfinite(fan.hi[0]) else math.inf
        top = min(top, 0.5 * reach + 50.0)
        if top <= max(zs, zr):
            return np.zeros(len(centres))
        foot = float(fan.at(np.array([[0.0]]))[0, 0, 0])

        def grazing(z: np.ndarray) -> np.ndarray:
            # How far the ray launched along the ground goes to reach heights z.
            return fan.at(np.asarray(z, dtype=float)[None])[0, :, 0] - foot

        x_s, x_r = grazing(np.array([zs, zr]))
        gap = reach - x_s - x_r
        if gap <= 0.0:
            return np.zeros(len(centres))
        heights = np.geomspace(1e-3, top, 512)
        reached = grazing(heights)
        if reached[-1] <= 0.5 * gap:
            return np.zeros(len(centres))
        lowest = float(np.interp(0.5 * gap, reached, heights))
        z = lowest + (top - lowest) * (np.arange(1, _HEIGHTS + 1) / _HEIGHTS) ** 2
        dz = np.gradient(np.concatenate([[lowest], z]))[1:]
        along = grazing(z)
        climb = np.arccos(np.clip(profile.speed(z) / c0, -1.0, 1.0))
        first = np.maximum(reach - x_r - along, 0.0)
        last = np.minimum(x_s + along, reach)
        width = np.maximum(last - first, 0.0)
        x = first[:, None] + width[:, None] * (np.arange(_ALONG) + 0.5)[None] / _ALONG
        zz = z[:, None]
        # Each ray's angle: the limiting ray's, plus how much steeper the chord is than
        # the chord to the limiting ray at that height.
        arrive = climb[:, None] + np.arctan((zz - zs) / np.maximum(x, 1e-9))
        arrive -= np.arctan((z - zs) / np.maximum(x_s + along, 1e-9))[:, None]
        leave = climb[:, None] + np.arctan((zz - zr) / np.maximum(reach - x, 1e-9))
        leave -= np.arctan((z - zr) / np.maximum(x_r + along, 1e-9))[:, None]
        theta = arrive + leave
        d1 = np.hypot(x, zz - zs)
        d2 = np.hypot(reach - x, zz - zr)
        area = (width / _ALONG * dz)[:, None] / (d1 * d2 * (d1 + d2))  # (heights, along)
        q = 2.0 * k[:, None, None] * np.sin(0.5 * theta)[None]
        spectrum = SPECTRUM_CONSTANT * _PLANE * self.cn2 * self.strength(z)[:, None]
        spectrum = spectrum * (q**2 + 1.0 / self.outer_scale_m**2) ** (-4.0 / 3.0)
        total = np.sum(spectrum * area[None], axis=(1, 2))
        return reach**2 * 2.0 * math.pi * k**3 * total

    def _shaken(self, profile, zs: float, zr: float, centres, start: float) -> np.ndarray:
        """The share the eddies bring deep into the shadow by shaking the refraction
        near the ground: :data:`SHAKEN_SHARE` (sigma_c / delta_c)², where the creeping
        wave's height scale is below the heights (see :data:`SHAKEN_HEIGHT`)."""
        from acoustic_spaces.engine.shadow import equivalent_radius

        height = max(math.sqrt(max(zs, 0.0) * max(zr, 0.0)), 0.1)
        fall = float(profile.c[0] - profile.speed(np.array([height]))[0])
        if fall <= 0.0 or start <= 0.0:
            return np.zeros(len(centres))
        spread = self.speed_spread * math.sqrt(float(self.strength(height)))
        share = SHAKEN_SHARE * (spread / fall) ** 2
        radius = equivalent_radius(start, max(zs, 0.0), max(zr, 0.0))
        k = 2.0 * math.pi * centres / float(profile.c[0])
        scale = (radius / (2.0 * k**2)) ** (1.0 / 3.0)
        return share / (1.0 + (scale / (SHAKEN_HEIGHT * height)) ** SHAKEN_POWER)


__all__ = [
    "GROUND_K_RHO",
    "GROUND_POWER",
    "SHAKEN_HEIGHT",
    "SHAKEN_POWER",
    "SHAKEN_SHARE",
    "SPECTRUM_CONSTANT",
    "Eddies",
    "Gusts",
    "Wander",
    "ground_share",
    "von_karman",
]
