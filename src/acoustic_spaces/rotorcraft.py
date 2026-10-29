"""Helicopters: the sound their rotors send each way, for a walk with Doppler.

A :class:`Helicopter` is its mass, its fuselage's drag and two rotors (:class:`Rotor`).
Flying level at an airspeed, :func:`trim` balances it the way momentum theory does
(Leishman, *Principles of Helicopter Aerodynamics*, 2nd ed., chapters 2 and 5): the main
rotor's thrust holds the weight up and the fuselage's drag back, its disc tilting
forward by the angle between them; Glauert's inflow gives the induced velocity through
the disc; the shaft power is the induced power (times ``induced``), the blades' profile
power (Leishman's ``1 + 4.65 mu^2``) and the parasite power of the drag; and the tail
rotor's thrust balances the main rotor's torque. :func:`emission` then gives the sound
the two rotors send, as an :class:`~acoustic_spaces.emission.Emission` from the main
rotor's hub, in the helicopter's frame: x along its flight, y to its left, z up.

**The tones.** Each blade is a row of stations at Gauss points (split at 97 % of the
radius, where the lift ends), each with a steady lift and drag (the lift growing with
the radius, as a uniform inflow through an ideally twisted blade gives it, up to 97 %
of the radius; the drag an induced part, even along the blade, and a profile part
growing with the square of the radius, sized to the trimmed torque) and a NACA
four-digit section's thickness. Their far field is that of the Ffowcs Williams and
Hawkings equation (Farassat's formulation 1A) for a rotor moving through still air:
the loading compact, the thickness a row of sources and sinks at Gauss points along
each straight chord, each where it truly is (its radius a little more than the
station's, taken to second order). With loads that do not change round the turn, each
harmonic of the blade passing frequency comes in closed form, from Bessel functions,
in every direction; its convective amplification is in it, and the walk adds the
Doppler shift. They are kept as one period of the sound in each direction
(:class:`~acoustic_spaces.emission.Periodic`).

**The broadband noise**, by the helicopter's ``broadband`` model:

- ``"pegg"`` (the default): Pegg's semi-empirical method (NASA TM-80200, 1979,
  equations 7 and 8), a fit to measured rotors' broadband noise: the third-octave band
  levels ``20 log (V_T / c)^3 + 10 log (A_b / r^2) (cos^2 theta + 0.1) + S + f(C_L) +
  130`` round a peak at ``-240 log T + 2.448 V_T + 942`` Hz, with the spectrum's shape
  ``S`` of its figure 9, for each rotor (``theta`` from the thrust's axis)
  (:func:`pegg_broadband`). It is a level for hovering; in flight it is sent with the
  convective amplification of a moving dipole, as the tones are. Its bands run from
  1/32 to 128 times the peak, the range of the figure.
- ``"bpm"``: each blade's own noise in the flow past it, Brooks, Pope and
  Marcolini's airfoil self-noise (:mod:`acoustic_spaces.airfoil_noise`), summed over 24
  sections from the blade's root to its tip at 144 places round the turn
  (:func:`bpm_bands`). The blades' pitch is trimmed (:func:`pitch`): rigid blades whose
  sections' lift is linear in their angle of attack (``lift_slope``), twisted by
  ``twist_deg``, in the trim's inflow, even over the disc, carry its thrust and do not
  flap. Each section's speed and angle of attack follow round the turn; its levels hold
  the convective amplification of its own motion through the air, its bands are shifted
  by its Doppler shift over the hub's (the walk adds the hub's), and each place round
  the turn counts as long as its sound takes to arrive. The boundary layers turn
  turbulent by themselves unless ``tripped``, the tips are square unless
  ``rounded_tip``, the trailing edges ``trailing_edge_m`` thick; sections whose flow
  reverses or whose angle of attack is beyond ``BPM_ALPHA_DEG`` are left out, and so is
  the bluntness noise of sections faster than ``BLUNT_MACH``. It is the blades' noise in
  a smooth flow only: the noise of the blades cutting through the wake of those before
  (blade-wake interaction) and through turbulence is not in it.

**What it leaves out.** For the tones, loads that change round the turn (in forward
flight a blade's lift changes as it goes round, and the blades flap; BPM's sections
follow their angle of attack round the turn, but do not flap), the blades meeting the
tip vortices of those before (blade-vortex interaction), noise from the engine and the
gearbox, the fuselage's reflection and shielding, and wind: the air is still, and the
airspeed is the speed over the ground.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import jv

from acoustic_spaces.air import density, speed_of_sound
from acoustic_spaces.directivity import _from_angles, _grid
from acoustic_spaces.emission import Mixture, Periodic, Sum

#: Standard gravity (m/s²).
GRAVITY = 9.80665
#: The share of the radius out to which a blade carries lift (Leishman's tip loss).
TIP_LOSS = 0.97
#: A rotor's tones keep the harmonics of the blade passing frequency up to the last one
#: within this many dB of the loudest harmonic in any direction.
QUIET_DB = 90.0
#: Pegg's figure 9: third-octave band levels (dB, over the overall level) at 1/32, 1/16,
#: ..., 64 and 128 times the peak frequency, one octave apart.
PEGG_SPECTRUM_DB = (
    -29.0,
    -24.5,
    -19.5,
    -15.3,
    -11.7,
    -7.5,
    -11.5,
    -12.1,
    -16.5,
    -17.0,
    -21.8,
    -26.4,
    -30.0,
)
#: The octave of the first of ``PEGG_SPECTRUM_DB`` from the peak.
PEGG_FIRST_OCTAVE = -5
#: The standard pressure the levels are over (Pa).
REFERENCE_PA = 2e-5
#: The ways a helicopter's broadband noise can be found (:attr:`Helicopter.broadband`).
BROADBAND_MODELS = ("bpm", "pegg")
#: Brooks, Pope and Marcolini's bluntness noise is left out of blade sections faster
#: than this Mach number, as their appendix C recommends for rotors.
BLUNT_MACH = 0.45
#: Their fits hold for angles of attack up to this (degrees); sections beyond it, in
#: the slow flow next to where it reverses, are left out.
BPM_ALPHA_DEG = 25.0


@dataclass(frozen=True)
class Rotor:
    """A rotor: its blades, its speed and where it sits on the helicopter."""

    blades: int
    radius_m: float
    #: Each blade's chord (m), the same along it.
    chord_m: float
    #: Turns a minute.
    rpm: float
    #: Where its hub is (m), in the helicopter's frame (x along its flight, y to its
    #: left, z up), from the helicopter's place: the main rotor's hub is best at the
    #: origin, where the sound is sent from.
    hub_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    #: The way its thrust pushes the helicopter, before the trim tilts it: up for a main
    #: rotor, sideways for a tail rotor.
    thrust: tuple[float, float, float] = (0.0, 0.0, 1.0)
    #: How its blades turn, seen from the side its thrust points to: "counter-clockwise"
    #: or "clockwise".
    turning: str = "counter-clockwise"
    #: The share of the radius where the blades begin.
    root: float = 0.2
    #: The blades' thickness over their chord (a NACA four-digit symmetric section).
    thickness: float = 0.12
    #: The blade sections' mean profile drag coefficient.
    profile_drag: float = 0.01
    #: How much the blades' pitch changes from the hub to the tip (degrees, linearly;
    #: negative, the tip pitched down), for the blade sections' angles of attack.
    twist_deg: float = 0.0
    #: The sections' lift curve slope (per radian).
    lift_slope: float = 5.73
    #: The trailing edge's thickness (m; 0 for a sharp edge, which sends none) and the
    #: angle its surfaces meet at (0 to 14 degrees), for Brooks, Pope and Marcolini's
    #: bluntness noise.
    trailing_edge_m: float = 0.0005
    trailing_edge_deg: float = 14.0
    #: Whether the blades' boundary layers are tripped (else they go turbulent by
    #: themselves), and whether their tips are rounded (else square).
    tripped: bool = False
    rounded_tip: bool = False

    def __post_init__(self) -> None:
        if self.turning not in ("counter-clockwise", "clockwise"):
            raise ValueError(f"a rotor turns counter-clockwise or clockwise, not {self.turning!r}")
        if not (self.blades >= 2 and self.radius_m > 0 and self.chord_m > 0 and self.rpm > 0):
            raise ValueError("a rotor needs two blades or more, a radius, a chord and a speed")
        if not (0.0 <= self.root < TIP_LOSS and 0.0 < self.thickness < 0.4):
            raise ValueError("the blades begin inside the tip and are thinner than 40 %")
        if np.linalg.norm(self.thrust) == 0.0:
            raise ValueError("the thrust needs a direction")
        if not self.lift_slope > 0.0:
            raise ValueError("the sections' lift rises with their angle of attack")
        if not (self.trailing_edge_m >= 0.0 and 0.0 <= self.trailing_edge_deg <= 14.0):
            raise ValueError("the trailing edge's surfaces meet at 0 to 14 degrees")

    @property
    def omega(self) -> float:
        """How fast it turns (rad/s)."""
        return self.rpm * 2.0 * math.pi / 60.0

    @property
    def tip_speed(self) -> float:
        """The blade tips' speed round the hub (m/s)."""
        return self.omega * self.radius_m

    @property
    def disc_m2(self) -> float:
        return math.pi * self.radius_m**2

    @property
    def blade_area_m2(self) -> float:
        """All its blades' area, as Pegg's ``A_b`` and the solidity count it."""
        return self.blades * self.chord_m * self.radius_m

    @property
    def solidity(self) -> float:
        return self.blade_area_m2 / self.disc_m2

    @property
    def spin(self) -> int:
        """+1 for a turn counter-clockwise seen from where the thrust points, -1 the
        other way."""
        return 1 if self.turning == "counter-clockwise" else -1


@dataclass(frozen=True)
class Helicopter:
    """A single main rotor helicopter with a tail rotor."""

    mass_kg: float
    main: Rotor
    tail: Rotor
    #: The fuselage's drag as the area of a flat plate (m²): ``D = rho V^2 f / 2``.
    drag_area_m2: float
    #: The induced power over that of momentum theory (Leishman's ``kappa``).
    induced: float = 1.15
    name: str = ""
    #: How its rotors' broadband noise is found: "pegg" (Pegg's semi-empirical method
    #: for the whole rotor) or "bpm" (Brooks, Pope and Marcolini's airfoil self-noise of
    #: each blade section round the turn).
    broadband: str = "pegg"

    def __post_init__(self) -> None:
        if self.broadband not in BROADBAND_MODELS:
            raise ValueError(
                f"a helicopter's broadband noise is found by {' or '.join(BROADBAND_MODELS)}, "
                f"not {self.broadband!r}"
            )

    @property
    def tail_arm_m(self) -> float:
        """How far the tail rotor's thrust acts from the main rotor's shaft (m)."""
        offset = np.subtract(self.tail.hub_m, self.main.hub_m)
        push = np.asarray(self.tail.thrust, dtype=float)
        push /= np.linalg.norm(push)
        return float(abs(np.cross(offset, push)[2]))


@dataclass(frozen=True)
class Load:
    """What a rotor carries in a trim."""

    #: Its thrust (N).
    thrust_n: float
    #: (3,) the way the thrust points, in the helicopter's frame.
    axis: np.ndarray = field(repr=False)
    #: The induced velocity through the disc (m/s).
    induced_m_s: float = 0.0
    #: The torque (N m) of its induced and parasite power, and of its blades' profile
    #: drag.
    induced_torque_nm: float = 0.0
    profile_torque_nm: float = 0.0

    @property
    def torque_nm(self) -> float:
        return self.induced_torque_nm + self.profile_torque_nm


@dataclass(frozen=True)
class Trim:
    """A helicopter balanced in level flight."""

    helicopter: Helicopter
    #: Its speed through the air (m/s).
    airspeed_m_s: float
    temperature_c: float
    pressure_kpa: float
    main: Load
    tail: Load

    @property
    def density(self) -> float:
        return density(self.temperature_c, self.pressure_kpa)

    @property
    def sound_speed(self) -> float:
        return speed_of_sound(self.temperature_c)

    @property
    def power_w(self) -> float:
        """The power both rotors take (W)."""
        h = self.helicopter
        return self.main.torque_nm * h.main.omega + self.tail.torque_nm * h.tail.omega

    def lift_coefficient(self, which: str) -> float:
        """A rotor's mean lift coefficient, ``6 C_T / sigma``: "main" or "tail"."""
        rotor, load = self._rotor(which)
        return 6.0 * load.thrust_n / (self.density * rotor.blade_area_m2 * rotor.tip_speed**2)

    def _rotor(self, which: str) -> tuple[Rotor, Load]:
        if which not in ("main", "tail"):
            raise ValueError(f"a helicopter's rotors are 'main' and 'tail', not {which!r}")
        return getattr(self.helicopter, which), getattr(self, which)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _inflow(thrust: float, rotor: Rotor, rho: float, edgewise: float, tilt: float) -> float:
    """Glauert's induced velocity (m/s) through a disc carrying ``thrust`` with the air
    crossing it at ``edgewise`` m/s, tilted ``tilt`` rad into it."""
    tip = rotor.tip_speed
    ct = thrust / (rho * rotor.disc_m2 * tip**2)
    mu, lam = edgewise / tip, math.sqrt(ct / 2.0)
    for _ in range(100):
        new = mu * math.tan(tilt) + ct / (2.0 * math.hypot(mu, lam))
        if abs(new - lam) < 1e-12:
            break
        lam = 0.5 * (lam + new)
    return (lam - mu * math.tan(tilt)) * tip


def _load(
    rotor: Rotor,
    thrust: float,
    axis,
    rho: float,
    speed: float,
    tilt: float,
    kappa: float,
    parasite: float = 0.0,
) -> Load:
    edgewise = speed * math.cos(tilt)
    induced = _inflow(thrust, rotor, rho, edgewise, tilt)
    mu = edgewise / rotor.tip_speed
    profile = (
        rotor.solidity
        * rotor.profile_drag
        / 8.0
        * rho
        * rotor.disc_m2
        * rotor.tip_speed**3
        * (1.0 + 4.65 * mu**2)
    )
    return Load(
        thrust,
        _unit(axis),
        induced,
        (kappa * thrust * induced + parasite) / rotor.omega,
        profile / rotor.omega,
    )


def trim(
    helicopter: Helicopter,
    airspeed_m_s: float,
    temperature_c: float = 15.0,
    pressure_kpa: float = 101.325,
) -> Trim:
    """The helicopter balanced flying level at ``airspeed_m_s`` (see the module's
    description)."""
    if not airspeed_m_s >= 0.0:
        raise ValueError("the airspeed must be zero or more")
    h, rho = helicopter, density(temperature_c, pressure_kpa)
    weight = h.mass_kg * GRAVITY
    drag = 0.5 * rho * airspeed_m_s**2 * h.drag_area_m2
    tilt = math.atan2(drag, weight)
    up = _unit(h.main.thrust)
    axis = math.cos(tilt) * up + math.sin(tilt) * np.array([1.0, 0.0, 0.0])
    main = _load(
        h.main,
        math.hypot(weight, drag),
        axis,
        rho,
        airspeed_m_s,
        tilt,
        h.induced,
        drag * airspeed_m_s,
    )
    if h.tail_arm_m == 0.0:
        raise ValueError("the tail rotor's thrust must act away from the main rotor's shaft")
    tail = _load(
        h.tail, main.torque_nm / h.tail_arm_m, h.tail.thrust, rho, airspeed_m_s, 0.0, h.induced
    )
    return Trim(h, float(airspeed_m_s), float(temperature_c), float(pressure_kpa), main, tail)


def _basis(axis) -> np.ndarray:
    """(3, 3) a rotor's frame as rows: two ways across its disc, then its axis."""
    a = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _unit(helper - a * (a @ helper))
    return np.array([e1, np.cross(a, e1), a])


@dataclass(frozen=True)
class _Blade:
    """A blade as Gauss points: S stations along it, their radii (S,) and the lift and
    drag each stands for (N); and the thickness as sources and sinks at G points along
    each station's chord (S, G): how much each puts out (the section's area changing
    across it, m^2), and where it is from the hub, as its angle ahead of the station's
    radius (rad) and how far it lies out beyond the station's radius (m)."""

    radii: np.ndarray
    lift: np.ndarray
    drag: np.ndarray
    areas: np.ndarray
    angles: np.ndarray
    outward: np.ndarray


def _gauss(lo: float, hi: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return lo + (hi - lo) * (x + 1.0) / 2.0, w * (hi - lo) / 2.0


def _blade(rotor: Rotor, load: Load, stations: int = 24, chord_points: int = 12) -> _Blade:
    root, lifting, tip = rotor.root * rotor.radius_m, TIP_LOSS * rotor.radius_m, rotor.radius_m
    inner, outer = _gauss(root, lifting, stations - 3), _gauss(lifting, tip, 3)
    r, w = np.concatenate([inner[0], outer[0]]), np.concatenate([inner[1], outer[1]])
    # Lift growing with the radius out to the tip loss; drag even along the blade for the
    # induced part and growing as the radius squared for the profile part, each giving
    # its share of the torque.
    lift = np.where(
        r <= lifting, 2.0 * load.thrust_n / (rotor.blades * (lifting**2 - root**2)) * r, 0.0
    )
    induced = load.induced_torque_nm / (rotor.blades * (tip**2 - root**2) / 2.0)
    profile = load.profile_torque_nm / (rotor.blades * (tip**4 - root**4) / 4.0)
    # The section's thickness changing along the chord, from the leading edge (x = 0)
    # as x = (1 - cos theta) / 2, which keeps it smooth at the nose.
    theta, weight = _gauss(0.0, math.pi, chord_points)
    x = (1.0 - np.cos(theta)) / 2.0
    slope = (
        5.0
        * rotor.thickness
        * (
            0.14845 * np.cos(theta / 2.0)
            + (-0.1260 - 0.7032 * x + 0.8529 * x**2 - 0.4144 * x**3) * np.sin(theta) / 2.0
        )
    )
    ahead = (0.25 - x) * rotor.chord_m
    return _Blade(
        r,
        lift * w,
        (induced + profile * r**2) * w,
        np.outer(w, 2.0 * rotor.chord_m * slope * weight),
        np.arctan2(ahead[None, :], r[:, None]),
        np.hypot(r[:, None], ahead[None, :]) - r[:, None],
    )


def _harmonics(
    own: np.ndarray,
    ways: np.ndarray,
    flight: np.ndarray,
    *,
    blades: int,
    omega: float,
    spin: int,
    blade: _Blade,
    speed: float,
    rho: float,
    c: float,
    count: int,
) -> np.ndarray:
    """(n, count) the complex amplitudes ``a_m`` of the far field ``R p`` (Pa m) of the
    first ``count`` harmonics of the blade passing frequency, ``R p = sum 2 Re(a_m
    exp(i m B Omega t))``, towards directions ``own`` (n, 3) in the rotor's frame
    (:func:`_basis`), ``ways`` the same in the helicopter's frame, for a rotor moving at
    ``speed`` along ``flight`` (its x in the rotor's frame); each direction's sound
    shifted a share ``phi / 2 pi`` of a turn later, ``phi`` its azimuth round the axis,
    as :class:`~acoustic_spaces.emission.Periodic` holds it.

    A load at radius ``r`` turning with the blade sends harmonic ``k`` (of the turn) as
    ``J_n(k Omega r sin(theta) / (c (1 - M_d)))``, ``n = k s - q`` for its mode ``q``
    round the turn (Jacobi and Anger). The thickness's points along the chord lie a
    little further out than their station; their Bessel functions are the station's
    and its first two derivatives (Taylor, to second order in how far further out)."""
    cos_t = own[:, 2]
    sin_t = np.hypot(own[:, 0], own[:, 1])
    turn = np.exp(1j * np.arctan2(own[:, 1], own[:, 0]))[:, None]
    doppler = 1.0 - speed * ways[:, 0] / c
    k = blades * np.arange(1, count + 1)
    # The order n = k s - q of each mode q of the loads round the turn (+1, 0, -1), and
    # the order p = |n| from which J_n comes (J_-p = (-1)^p J_p).
    orders = {q: k * spin - q for q in (1, 0, -1)}
    signs = {q: np.where(n < 0, (-1.0) ** np.abs(n), 1.0) for q, n in orders.items()}
    rows = {q: 1 - q * spin for q in (1, 0, -1)}  # p - k + 1: 0, 1 or 2
    powers_of_i = np.array([1.0, 1j, -1.0, -1j])
    x1, x2 = flight[0], flight[1]
    total = np.zeros((len(own), count), dtype=complex)
    for r, lift, drag, areas, angles, outward in zip(
        blade.radii, blade.lift, blade.drag, blade.areas, blade.angles, blade.outward, strict=True
    ):
        z = k * (omega * r * sin_t / (c * doppler))[:, None]
        # J_p for p = k - 2 ... k + 1, down from the two top ones.
        top, above = jv(k, z), jv(k + 1, z)
        below = np.divide(2.0 * k * top, z, out=np.zeros_like(z), where=z > 0) - above
        lowest = np.divide(2.0 * (k - 1) * below, z, out=np.zeros_like(z), where=z > 0) - top
        values = (below, top, above)
        under = (lowest, below, top)
        # kappa J_p' and kappa^2 J_p'' (kappa = z / r), from the recurrence and Bessel's
        # equation.
        slopes, curves = [], []
        for j, p in enumerate((k - 1, k, k + 1)):
            slope = z / r * under[j] - p / r * values[j]
            slopes.append(slope)
            curves.append(-slope / r - (z / r) ** 2 * values[j] + (p / r) ** 2 * values[j])
        for q, n in orders.items():
            j = rows[q]
            phase = np.exp(1j * np.outer(angles, n * spin))
            chord = (
                areas @ phase * values[j]
                + (areas * outward) @ phase * slopes[j]
                + (areas * outward**2 / 2.0) @ phase * curves[j]
            )
            if q == 0:
                mode = -lift * cos_t[:, None] / c * values[j] + rho * omega * r * chord
            else:
                mode = (
                    -q * spin * drag * sin_t[:, None] / (2j * c) * values[j]
                    + rho * speed * spin * (x2 + q * 1j * x1) / 2.0 * chord * turn**q
                )
            total += signs[q] * powers_of_i[n % 4] * mode
    return blades * 1j * k * omega / (4.0 * math.pi * doppler[:, None] ** 2) * total


def _count(own, ways, flight, rotor_args: dict[str, Any]) -> int:
    """How many harmonics to keep: up to the last within ``QUIET_DB`` of the loudest."""
    trial = 80
    levels = np.max(np.abs(_harmonics(own, ways, flight, count=trial, **rotor_args)), axis=0)
    loud = levels >= levels.max() * 10.0 ** (-QUIET_DB / 20.0)
    return int(np.nonzero(loud)[0][-1]) + 1


def tones(
    trim: Trim,
    which: str,
    *,
    samples: int,
    sample_rate: int = 48_000,
    harmonics: int | None = None,
    grid_deg: float = 5.0,
    stations: int = 24,
) -> Periodic:
    """The tones a rotor ("main" or "tail") sends in a trim, ``samples`` of them, as one
    period in every direction ``grid_deg`` apart round its axis; ``harmonics`` of the
    blade passing frequency (by default up to the last within ``QUIET_DB`` of the
    loudest)."""
    rotor, load = trim._rotor(which)
    axes = _basis(load.axis)
    elevation, azimuth = _grid(grid_deg)
    own = _from_angles(azimuth[None, :], elevation[:, None]).reshape(-1, 3)
    ways = own @ axes
    flight = axes @ np.array([1.0, 0.0, 0.0])
    rotor_args: dict[str, Any] = dict(
        blades=rotor.blades,
        omega=rotor.omega,
        spin=rotor.spin,
        blade=_blade(rotor, load, stations),
        speed=trim.airspeed_m_s,
        rho=trim.density,
        c=trim.sound_speed,
    )
    if harmonics is None:
        coarse_el, coarse_az = _grid(15.0)
        coarse = _from_angles(coarse_az[None, :], coarse_el[:, None]).reshape(-1, 3)
        harmonics = _count(coarse, coarse @ axes, flight, rotor_args)
    amplitudes = _harmonics(own, ways, flight, count=harmonics, **rotor_args)
    # One blade passage, 16 samples to the period of the highest harmonic.
    n = max(64, 16 * harmonics)
    spectrum = np.zeros((len(own), n // 2 + 1), dtype=complex)
    spectrum[:, 1 : harmonics + 1] = n * amplitudes
    table = np.fft.irfft(spectrum, n, axis=1).reshape(len(elevation), len(azimuth), n)
    return Periodic(
        table,
        period_s=2.0 * math.pi / (rotor.blades * rotor.omega),
        sample_rate=sample_rate,
        samples=samples,
        axes=axes,
        turn_s=rotor.spin * 2.0 * math.pi / rotor.omega,
        centre=np.asarray(rotor.hub_m, dtype=float),
        speed=trim.airspeed_m_s,
        sound_speed=trim.sound_speed,
    )


def pegg_peak_hz(thrust_n: float, tip_speed: float) -> float:
    """Pegg's peak frequency of a rotor's broadband noise (Hz), his equation 7 in SI
    units (thrust in N, tip speed in m/s)."""
    return -240.0 * math.log10(thrust_n) + 2.448 * tip_speed + 942.0


def pegg_lift_db(lift_coefficient: float) -> float:
    """Pegg's ``f(C_L)`` (dB), his appendix B."""
    if lift_coefficient <= 0.48:
        return 10.0 * math.log10(lift_coefficient / 0.4)
    return 0.8 + 80.0 * math.log10(lift_coefficient / 0.48)


def pegg_level_db(
    tip_speed: float, sound_speed: float, blade_area_m2: float, lift_coefficient: float
) -> float:
    """Pegg's level (dB) of a rotor's broadband noise at 1 m, before its pattern
    (``10 log (cos^2 theta + 0.1)``) and its spectrum (``S``): ``20 log (V_T / c)^3 +
    10 log A_b + f(C_L) + 130``."""
    return (
        60.0 * math.log10(tip_speed / sound_speed)
        + 10.0 * math.log10(blade_area_m2)
        + pegg_lift_db(lift_coefficient)
        + 130.0
    )


def pegg_bands(peak_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """The third-octave bands of Pegg's spectrum round a peak: their centres (Hz,
    base-10 exact) and their levels ``S`` (dB) from his figure 9, the peak in the
    standard band it falls in."""
    peak = round(10.0 * math.log10(peak_hz / 1000.0))
    steps = np.arange(
        3 * PEGG_FIRST_OCTAVE, 3 * (PEGG_FIRST_OCTAVE + len(PEGG_SPECTRUM_DB) - 1) + 1
    )
    octaves = PEGG_FIRST_OCTAVE + np.arange(len(PEGG_SPECTRUM_DB))
    return 1000.0 * 10.0 ** ((peak + steps) / 10.0), np.interp(
        steps / 3.0, octaves, PEGG_SPECTRUM_DB
    )


def _noise(peak_hz: float, samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """A noise whose third-octave bands carry Pegg's spectrum, mean squares ``10^(S/10)``:
    even within each band, in random phases."""
    centres, levels = pegg_bands(peak_hz)
    f = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    magnitude = np.zeros(len(f))
    for centre, level in zip(centres, levels, strict=True):
        inside = (f >= centre * 10.0**-0.05) & (f < centre * 10.0**0.05) & (f > 0)
        if np.any(inside):
            # Mean square 2 sum |X|^2 / n^2 of the band's bins.
            magnitude[inside] = samples * math.sqrt(10.0 ** (level / 10.0) / (2.0 * inside.sum()))
    phase = np.exp(2j * math.pi * rng.random(len(f)))
    return np.fft.irfft(magnitude * phase, samples)


def pegg_broadband(
    trim: Trim,
    *,
    samples: int,
    sample_rate: int = 48_000,
    grid_deg: float = 5.0,
    seed: int = 0,
    rotors: tuple[str, ...] = ("main", "tail"),
) -> Mixture:
    """The rotors' broadband noise in a trim by Pegg's method (see the module's
    description), ``samples`` of it, its pattern on a grid ``grid_deg`` apart in the
    helicopter's frame: one noise for each of ``rotors``."""
    rng = np.random.default_rng(seed)
    elevation, azimuth = _grid(grid_deg)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    doppler = 1.0 - trim.airspeed_m_s * ways[..., 0] / trim.sound_speed
    signals, weights = [], []
    for which in ("main", "tail"):
        rotor, load = trim._rotor(which)
        noise = _noise(pegg_peak_hz(load.thrust_n, rotor.tip_speed), samples, sample_rate, rng)
        if which not in rotors:
            continue
        signals.append(noise)
        pattern = np.sqrt((ways @ load.axis) ** 2 + 0.1)
        level_db = pegg_level_db(
            rotor.tip_speed, trim.sound_speed, rotor.blade_area_m2, trim.lift_coefficient(which)
        )
        level = REFERENCE_PA * 10.0 ** (level_db / 20.0)
        weights.append(level * pattern / doppler**2)
    return Mixture(np.array(signals), np.stack(weights, axis=-1), sample_rate)


def pitch(trim: Trim, which: str, stations: int = 24, azimuths: int = 64) -> np.ndarray:
    """A rotor's blade pitch in a trim (radians at the hub, by ``twist_deg`` less at the
    tip): its collective ``theta_0`` and its cyclic ``theta_1c``, ``theta_1s`` (with
    ``cos psi`` and ``sin psi``, ``psi`` the blade's azimuth from straight downwind,
    turning with the rotor), such that its blades carry the trim's thrust and do not
    flap round the turn: their lift's moment about the hub has no part that goes once
    round with it. The blades are rigid, their sections' lift linear in the angle of
    attack (``lift_slope``), the inflow even over the disc (the trim's); the lift runs
    from the root to the tip loss, as the tones' does."""
    rotor, load = trim._rotor(which)
    r, w = _gauss(rotor.root * rotor.radius_m, TIP_LOSS * rotor.radius_m, stations)
    psi = 2.0 * math.pi * np.arange(azimuths) / azimuths
    u_t, u_p = _flow(trim, which, r[None, :], psi[:, None])
    shape = r[None, :] / rotor.radius_m * math.radians(rotor.twist_deg)
    # The lift per length, rho c a (theta U_T^2 - U_P U_T) / 2, is linear in the three
    # controls: rows thrust, moment with cos psi and with sin psi.
    trig = [np.ones_like(psi), np.cos(psi), np.sin(psi)]
    arms = [np.ones_like(r), r, r]
    half = 0.5 * trim.density * rotor.chord_m * rotor.lift_slope
    a = np.zeros((3, 3))
    b = np.zeros(3)
    for i in range(3):
        mean = trig[i][:, None] * (arms[i] * w)[None, :] / azimuths
        for j in range(3):
            a[i, j] = half * np.sum(mean * trig[j][:, None] * u_t**2)
        b[i] = -half * np.sum(mean * (shape * u_t**2 - u_p * u_t))
    b[0] += load.thrust_n / rotor.blades
    return np.linalg.solve(a, b)


def _flow(trim: Trim, which: str, r, psi) -> tuple[np.ndarray, np.ndarray]:
    """The air's speed past a blade section at radius ``r`` and azimuth ``psi``, in a
    trim: along the turn (towards its leading edge, ``U_T``) and down through the disc
    (``U_P``), the inflow even over the disc."""
    rotor, load = trim._rotor(which)
    across = trim.airspeed_m_s * math.sqrt(max(0.0, 1.0 - load.axis[0] ** 2))
    through = trim.airspeed_m_s * load.axis[0] + load.induced_m_s
    u_t = rotor.omega * np.asarray(r) + across * np.sin(psi)
    return u_t, np.broadcast_to(through, np.broadcast(u_t, psi).shape).astype(float)


def _frame(load: Load, rotor: Rotor) -> tuple[np.ndarray, np.ndarray]:
    """A rotor's disc as two unit vectors in the helicopter's frame: straight downwind
    across it (where ``psi`` is 0) and a quarter turn on, with the rotor's turn."""
    a = load.axis
    downwind = np.array([-1.0, 0.0, 0.0]) - a * (-a[0])
    if np.linalg.norm(downwind) < 1e-9:
        downwind = _basis(a)[0]
    downwind = _unit(downwind)
    return downwind, rotor.spin * np.cross(a, downwind)


def _third_octaves(lo: int, hi: int) -> np.ndarray:
    """Base-10 third-octave band centres ``10^(k / 10)`` Hz, for ``k`` from ``lo`` to
    ``hi``."""
    return 10.0 ** (np.arange(lo, hi + 1) / 10.0)


#: The bands (``10^(k/10)`` Hz) BPM's levels are found in, in a blade section's frame,
#: and those the noise is sent in from the hub, after each section's Doppler shift.
BPM_SOURCE_BANDS = (7, 46)
BPM_BANDS = (12, 43)


def bpm_bands(
    trim: Trim,
    which: str,
    ways: np.ndarray,
    *,
    segments: int = 24,
    azimuths: int = 144,
) -> tuple[np.ndarray, np.ndarray]:
    """A rotor's broadband noise in a trim by Brooks, Pope and Marcolini's airfoil
    self-noise (:mod:`acoustic_spaces.airfoil_noise`), sent towards ``ways`` (..., 3)
    in the helicopter's frame: the third-octave bands' centres (Hz) and their mean
    square pressures 1 m from the hub (Pa², ``(..., bands)``), averaged over a turn.

    Each blade is ``segments`` sections from its root to its tip, each at ``azimuths``
    places round the turn, with the flow past it and its angle of attack from the
    trim's pitch (:func:`pitch`). A section's levels hold the convective amplification
    of its motion through the air; its bands are shifted by its own Doppler shift over
    the hub's (the walk adds the hub's), and each place round the turn counts as long
    as its sound takes to arrive. The bluntness noise is left out of sections faster
    than ``BLUNT_MACH`` and sections beyond ``BPM_ALPHA_DEG``, or where the flow
    reverses, are left out; the tip's vortex is the outermost section's."""
    from acoustic_spaces import airfoil_noise as bpm
    from acoustic_spaces.air import viscosity

    rotor, load = trim._rotor(which)
    c, v = trim.sound_speed, trim.airspeed_m_s
    nu = viscosity(trim.temperature_c, trim.pressure_kpa)
    trip = bpm.TRIPPED if rotor.tripped else bpm.UNTRIPPED
    ways = np.asarray(ways, dtype=float)
    flat = ways.reshape(-1, 3)
    flat = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    edges = np.linspace(rotor.root * rotor.radius_m, rotor.radius_m, segments + 1)
    r, span = (edges[1:] + edges[:-1]) / 2.0, np.diff(edges)
    theta0, theta1c, theta1s = pitch(trim, which)
    downwind, quarter = _frame(load, rotor)
    lo, hi = BPM_SOURCE_BANDS
    f = _third_octaves(lo, hi)
    out_lo, out_hi = BPM_BANDS
    flight = 1.0 - v * flat[:, 0] / c
    total = np.zeros((len(flat), out_hi - out_lo + 1))
    for psi in 2.0 * math.pi * np.arange(azimuths) / azimuths:
        u_t, u_p = _flow(trim, which, r, psi)
        theta = (
            theta0
            + r / rotor.radius_m * math.radians(rotor.twist_deg)
            + theta1c * math.cos(psi)
            + theta1s * math.sin(psi)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = np.degrees(theta - u_p / u_t)
        speed = np.hypot(u_t, u_p)
        kept = (u_t > 0.0) & (np.abs(alpha) <= BPM_ALPHA_DEG)
        if not np.any(kept):
            continue
        # The section's frame: its chord from the leading edge back to the trailing
        # edge, its span out along the blade, and the normal to its chord.
        out = math.cos(psi) * downwind + math.sin(psi) * quarter
        ahead = rotor.spin * np.cross(load.axis, out)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        chord = -cos_t[:, None] * ahead - sin_t[:, None] * load.axis
        normal = -sin_t[:, None] * ahead + cos_t[:, None] * load.axis
        dx, dz = flat @ chord.T, flat @ normal.T
        dy = (flat @ out)[:, None]
        # As the report's figure B3 has them: theta from the chord to the way, phi from
        # the span to the way's projection across the chord.
        theta_e = np.degrees(np.arctan2(np.hypot(dy, dz), dx))
        phi_e = np.degrees(np.arctan2(np.abs(dz), dy))
        moving = 1.0 - (v * flat[:, :1] + rotor.omega * r[None, :] * (flat @ ahead)[:, None]) / c
        a, u = np.abs(alpha[kept]), speed[kept]
        m = u / c
        common: dict[str, Any] = {"trip": trip, "viscosity": nu, "sound_speed": c}
        # Each mechanism's level 1 m away at theta = phi = 90 degrees, where both
        # directivities are 1: the stalled sections' over D_l, the others' over D_h.
        args = (f, rotor.chord_m, span[kept], 1.0, 90.0, 90.0, a, u)
        pressure, suction, angle = bpm.tbl_te(*args, **common)
        tbl = 10.0 ** (pressure / 10.0) + 10.0 ** (suction / 10.0) + 10.0 ** (angle / 10.0)
        stall = bpm.stalled(a, m)[:, None]
        high, low = np.where(stall, 0.0, tbl), np.where(stall, tbl, 0.0)
        if trip == bpm.UNTRIPPED:
            high = high + 10.0 ** (bpm.lbl_vs(*args, **common) / 10.0)
        blunt = (m <= BLUNT_MACH)[:, None]
        if rotor.trailing_edge_m > 0.0 and np.any(blunt):
            level = bpm.bluntness(*args, rotor.trailing_edge_m, rotor.trailing_edge_deg, **common)
            high = high + np.where(blunt, 10.0 ** (level / 10.0), 0.0)
        index = np.nonzero(kept)[0]
        if kept[-1]:
            level = bpm.tip(
                f,
                rotor.chord_m,
                1.0,
                90.0,
                90.0,
                a[-1],
                u[-1],
                rounded=rotor.rounded_tip,
                sound_speed=c,
            )
            high[-1] += 10.0 ** (level / 10.0)
        # The section's Doppler shift over the hub's, in third-octave bands, each band
        # shared between the two it lands across; and the time its sound takes to arrive
        # over the hub's.
        ratio = flight[:, None] / moving[:, index]
        d_high = bpm.directivity_high(m[None, :], theta_e[:, index], phi_e[:, index]) / ratio
        d_low = bpm.directivity_low(m[None, :], theta_e[:, index], phi_e[:, index]) / ratio
        steps = 10.0 * np.log10(ratio)
        whole = np.floor(steps).astype(int)
        part = steps - whole
        # A band ``shift`` bands up takes 1 - part of the sections whose shift's whole
        # part it is and part of those one below.
        for shift in range(int(whole.min()), int(whole.max()) + 2):
            share = np.where(whole == shift, 1.0 - part, 0.0)
            share += np.where(whole == shift - 1, part, 0.0)
            source = np.arange(out_lo, out_hi + 1) - shift - lo
            inside = (source >= 0) & (source <= hi - lo)
            total[:, inside] += (share * d_high) @ high[:, source[inside]]
            total[:, inside] += (share * d_low) @ low[:, source[inside]]
    centres = _third_octaves(out_lo, out_hi)
    total *= rotor.blades / azimuths * REFERENCE_PA**2
    return centres, total.reshape(ways.shape[:-1] + (len(centres),))


def _band_noises(centres: np.ndarray, samples: int, sample_rate: int, rng) -> np.ndarray:
    """(bands, samples) noises of unit mean square, each even over its third-octave
    band (base-10 edges) in random phases; a band with no frequency of the length in
    it, or above half the rate, is silent."""
    f = np.fft.rfftfreq(samples, 1.0 / sample_rate)
    out = np.zeros((len(centres), samples))
    for i, centre in enumerate(centres):
        inside = (f >= centre * 10.0**-0.05) & (f < centre * 10.0**0.05) & (f > 0)
        inside &= f < sample_rate / 2.0
        if not np.any(inside):
            continue
        magnitude = np.where(inside, samples * math.sqrt(1.0 / (2.0 * inside.sum())), 0.0)
        phase = np.exp(2j * math.pi * rng.random(len(f)))
        out[i] = np.fft.irfft(magnitude * phase, samples)
    return out


def broadband(
    trim: Trim,
    *,
    samples: int,
    sample_rate: int = 48_000,
    grid_deg: float = 5.0,
    seed: int = 0,
    rotors: tuple[str, ...] = ("main", "tail"),
    model: str | None = None,
) -> Mixture:
    """The rotors' broadband noise in a trim, ``samples`` of it, its pattern on a grid
    ``grid_deg`` apart in the helicopter's frame, by the helicopter's ``broadband``
    model or ``model``: "bpm" (:func:`bpm_bands`, a noise in each third-octave band
    with its level in every direction) or "pegg" (:func:`pegg_broadband`)."""
    model = trim.helicopter.broadband if model is None else model
    if model not in BROADBAND_MODELS:
        raise ValueError(f"broadband noise by {' or '.join(BROADBAND_MODELS)}, not {model!r}")
    if model == "pegg":
        return pegg_broadband(
            trim,
            samples=samples,
            sample_rate=sample_rate,
            grid_deg=grid_deg,
            seed=seed,
            rotors=rotors,
        )
    elevation, azimuth = _grid(grid_deg)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    powers = None
    for which in rotors:
        centres, part = bpm_bands(trim, which, ways)
        powers = part if powers is None else powers + part
    assert powers is not None
    keep = centres * 10.0**-0.05 < sample_rate / 2.0
    signals = _band_noises(centres[keep], samples, sample_rate, np.random.default_rng(seed))
    return Mixture(signals, np.sqrt(powers[..., keep]), sample_rate)


def emission(
    trim: Trim, *, seconds: float, sample_rate: int = 48_000, grid_deg: float = 5.0, seed: int = 0
) -> Sum:
    """All the helicopter sends in a trim for ``seconds``: both rotors' tones and their
    broadband noise, from the main rotor's hub."""
    samples = int(round(seconds * sample_rate))
    return Sum(
        (
            tones(trim, "main", samples=samples, sample_rate=sample_rate, grid_deg=grid_deg),
            tones(trim, "tail", samples=samples, sample_rate=sample_rate, grid_deg=grid_deg),
            broadband(trim, samples=samples, sample_rate=sample_rate, grid_deg=grid_deg, seed=seed),
        )
    )


#: An Airbus Helicopters AS350 B3 (Ecureuil, sold as the H125) at its maximum mass of
#: 2,250 kg. From its type certificate (EASA.R.008): the mass, the rotors' diameters
#: (10.69 m and 1.86 m) and blades (three and two), and the main rotor turning between 375
#: and 405 turns a minute in powered flight. From NASA's flight tests of an AS350 SD1
#: (Greenwood, Sim and Boyd, AHS Forum 72, 2016): the drag area (f / A = 0.0071), the
#: main rotor's speed (394 turns a minute, its tip at 220.6 m/s from their weight
#: coefficient) and its OA209 blades (9 % thick). Assumed, from no source to hand: the
#: main blades' chord (0.35 m) and twist (-10 degrees), the tail rotor's speed (2,086
#: turns a minute, its tip at 203 m/s), chord (0.185 m), thickness (12 %), untwisted
#: blades and the way it turns, both rotors' trailing edges (a rotor's default), and its
#: hub 6.67 m behind the main rotor's shaft (from Airbus's 12.94 m overall with the rotors
#: turning, less both rotors' radii) and 1 m lower. The main rotor turns clockwise seen
#: from above.
AS350_B3 = Helicopter(
    mass_kg=2250.0,
    main=Rotor(3, 5.345, 0.35, 394.0, turning="clockwise", thickness=0.09, twist_deg=-10.0),
    tail=Rotor(2, 0.93, 0.185, 2086.0, hub_m=(-6.665, 0.0, -1.0), thrust=(0.0, 1.0, 0.0)),
    drag_area_m2=0.674,
    name="AS350 B3",
)

__all__ = [
    "AS350_B3",
    "BLUNT_MACH",
    "BPM_ALPHA_DEG",
    "BPM_BANDS",
    "BPM_SOURCE_BANDS",
    "BROADBAND_MODELS",
    "GRAVITY",
    "PEGG_SPECTRUM_DB",
    "QUIET_DB",
    "TIP_LOSS",
    "Helicopter",
    "Load",
    "Rotor",
    "Trim",
    "bpm_bands",
    "broadband",
    "emission",
    "pegg_bands",
    "pegg_broadband",
    "pegg_level_db",
    "pegg_lift_db",
    "pegg_peak_hz",
    "pitch",
    "tones",
    "trim",
]
