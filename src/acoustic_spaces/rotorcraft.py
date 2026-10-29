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

**The broadband noise.** Pegg's semi-empirical method (NASA TM-80200, 1979, equations
7 and 8): the third-octave band levels ``20 log (V_T / c)^3 + 10 log (A_b / r^2)
(cos^2 theta + 0.1) + S + f(C_L) + 130`` round a peak at ``-240 log T + 2.448 V_T +
942`` Hz, with the spectrum's shape ``S`` of its figure 9, for each rotor (``theta``
from the thrust's axis). It is a level for hovering; in flight it is sent with the
convective amplification of a moving dipole, as the tones are. Its bands run from 1/32
to 128 times the peak, the range of the figure.

**What it leaves out.** Loads that change round the turn (the rotor in forward flight
carries less lift on its advancing side, and the blades flap), the blades meeting the
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

    def __post_init__(self) -> None:
        if self.turning not in ("counter-clockwise", "clockwise"):
            raise ValueError(f"a rotor turns counter-clockwise or clockwise, not {self.turning!r}")
        if not (self.blades >= 2 and self.radius_m > 0 and self.chord_m > 0 and self.rpm > 0):
            raise ValueError("a rotor needs two blades or more, a radius, a chord and a speed")
        if not (0.0 <= self.root < TIP_LOSS and 0.0 < self.thickness < 0.4):
            raise ValueError("the blades begin inside the tip and are thinner than 40 %")
        if np.linalg.norm(self.thrust) == 0.0:
            raise ValueError("the thrust needs a direction")

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


def broadband(
    trim: Trim, *, samples: int, sample_rate: int = 48_000, grid_deg: float = 5.0, seed: int = 0
) -> Mixture:
    """Both rotors' broadband noise in a trim (Pegg's method, see the module's
    description), ``samples`` of it, its pattern on a grid ``grid_deg`` apart in the
    helicopter's frame."""
    rng = np.random.default_rng(seed)
    elevation, azimuth = _grid(grid_deg)
    ways = _from_angles(azimuth[None, :], elevation[:, None])
    doppler = 1.0 - trim.airspeed_m_s * ways[..., 0] / trim.sound_speed
    signals, weights = [], []
    for which in ("main", "tail"):
        rotor, load = trim._rotor(which)
        signals.append(
            _noise(pegg_peak_hz(load.thrust_n, rotor.tip_speed), samples, sample_rate, rng)
        )
        pattern = np.sqrt((ways @ load.axis) ** 2 + 0.1)
        level_db = pegg_level_db(
            rotor.tip_speed, trim.sound_speed, rotor.blade_area_m2, trim.lift_coefficient(which)
        )
        level = REFERENCE_PA * 10.0 ** (level_db / 20.0)
        weights.append(level * pattern / doppler**2)
    return Mixture(np.array(signals), np.stack(weights, axis=-1), sample_rate)


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
#: main blades' chord (0.35 m), the tail rotor's speed (2,086 turns a minute, its tip at
#: 203 m/s), chord (0.185 m), thickness (12 %) and the way it turns, and its hub 6.67 m
#: behind the main rotor's shaft (from Airbus's 12.94 m overall with the rotors turning,
#: less both rotors' radii) and 1 m lower. The main rotor turns clockwise seen from above.
AS350_B3 = Helicopter(
    mass_kg=2250.0,
    main=Rotor(3, 5.345, 0.35, 394.0, turning="clockwise", thickness=0.09),
    tail=Rotor(2, 0.93, 0.185, 2086.0, hub_m=(-6.665, 0.0, -1.0), thrust=(0.0, 1.0, 0.0)),
    drag_area_m2=0.674,
    name="AS350 B3",
)

__all__ = [
    "AS350_B3",
    "GRAVITY",
    "PEGG_SPECTRUM_DB",
    "QUIET_DB",
    "TIP_LOSS",
    "Helicopter",
    "Load",
    "Rotor",
    "Trim",
    "broadband",
    "emission",
    "pegg_bands",
    "pegg_level_db",
    "pegg_lift_db",
    "pegg_peak_hz",
    "tones",
    "trim",
]
