"""Airfoil self-noise: the noise an airfoil makes in a smooth flow, by Brooks, Pope
and Marcolini's method (NASA Reference Publication 1218, 1989, "BPM").

BPM scaled wind tunnel measurements of NACA 0012 sections, 2.5 to 30 cm in chord, at
speeds up to 71 m/s, into third-octave band levels from five mechanisms:

- **turbulent boundary layer, trailing edge** (TBL-TE): the turbulence of each side's
  boundary layer passing the trailing edge, and at an angle of attack the flow
  separating from the suction side, which takes over past stall;
- **laminar boundary layer vortex shedding** (LBL-VS): a laminar boundary layer's
  instabilities feeding back from the trailing edge, at low Reynolds numbers;
- **trailing-edge bluntness**: vortices shed from a trailing edge of thickness ``h``;
- **tip vortex formation**: the turbulence where the tip's vortex forms.

Each segment of an airfoil is its chord, its span, the speed and the angle of attack of
the flow, and where the observer is: ``r`` away, at an angle ``theta`` from the chord
line (pointing downstream, from the leading edge to the trailing edge), and ``phi``
from the span to the observer's projection on the plane of the span and the normal,
across the chord (the report's figure B3). The levels are those of the report's code (its
appendix D), checked against its two sample outputs, tables D2 and D3, and are for a
segment moving through still air at the speed of the flow: the high-frequency
directivity ``D_h`` (its equation B1) and the low-frequency ``D_l`` (B2) hold the
convective amplification of that motion. The angles of attack are in degrees, from
the angle of zero lift.
"""

from __future__ import annotations

import numpy as np

#: The report's third-octave band centres (Hz), as its code gives them.
FREQUENCIES_HZ = (
    100.0,
    125.0,
    160.0,
    200.0,
    250.0,
    315.0,
    400.0,
    500.0,
    630.0,
    800.0,
    1000.0,
    1250.0,
    1600.0,
    2000.0,
    2500.0,
    3150.0,
    4000.0,
    5000.0,
    6300.0,
    8000.0,
    10000.0,
    12500.0,
    16000.0,
    20000.0,
    25000.0,
    31500.0,
    40000.0,
)
#: The level (dB) the code puts a mechanism at when it gives less.
FLOOR_DB = -100.0
#: Boundary layers: natural transition, tripped heavily, and tripped lightly.
UNTRIPPED, TRIPPED, LIGHTLY_TRIPPED = 0, 1, 2


def _log(x):
    with np.errstate(divide="ignore"):
        return np.log10(x)


def thicknesses(
    chord: np.ndarray, speed: np.ndarray, alpha: np.ndarray, trip: int, viscosity: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The boundary layer at the trailing edge (the report's section 3, as its code
    has it): the pressure side's thickness, and the suction and pressure sides'
    displacement thicknesses (m)."""
    c, alpha = np.asarray(chord, dtype=float), np.asarray(alpha, dtype=float)
    rc = speed * c / viscosity
    lr = np.log10(rc)
    delta0 = 10.0 ** (1.6569 - 0.9045 * lr + 0.0596 * lr**2) * c
    if trip == LIGHTLY_TRIPPED:
        delta0 = 0.6 * delta0
    delta_p = 10.0 ** (-0.04175 * alpha + 0.00106 * alpha**2) * delta0
    if trip in (TRIPPED, LIGHTLY_TRIPPED):
        dstar0 = np.where(
            rc <= 0.3e6,
            0.0601 * rc**-0.114 * c,
            10.0 ** (3.411 - 1.5397 * lr + 0.1059 * lr**2) * c,
        )
        if trip == LIGHTLY_TRIPPED:
            dstar0 = 0.6 * dstar0
        suction = np.select(
            [alpha <= 5.0, alpha <= 12.5],
            [10.0 ** (0.0679 * alpha), 0.381 * 10.0 ** (0.1516 * alpha)],
            14.296 * 10.0 ** (0.0258 * alpha),
        )
    else:
        dstar0 = 10.0 ** (3.0187 - 1.5397 * lr + 0.1059 * lr**2) * c
        suction = np.select(
            [alpha <= 7.5, alpha <= 12.5],
            [10.0 ** (0.0679 * alpha), 0.0162 * 10.0 ** (0.3066 * alpha)],
            52.42 * 10.0 ** (0.0258 * alpha),
        )
    dstar_p = 10.0 ** (-0.0432 * alpha + 0.00113 * alpha**2) * dstar0
    return delta_p, suction * dstar0, dstar_p


def directivity_high(mach, theta_deg, phi_deg) -> np.ndarray:
    """``D_h``: a trailing edge's noise at high frequencies, 1 at ``theta`` = ``phi``
    = 90 degrees, with the eddies convected at 0.8 of the flow's Mach number."""
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    mc = 0.8 * mach
    return (
        2.0
        * np.sin(t / 2.0) ** 2
        * np.sin(p) ** 2
        / ((1.0 + mach * np.cos(t)) * (1.0 + (mach - mc) * np.cos(t)) ** 2)
    )


def directivity_low(mach, theta_deg, phi_deg) -> np.ndarray:
    """``D_l``: a translating dipole's, for a stalled airfoil's noise."""
    t, p = np.radians(theta_deg), np.radians(phi_deg)
    return (np.sin(t) * np.sin(p)) ** 2 / (1.0 + mach * np.cos(t)) ** 4


def _a_min(a):
    x = np.abs(a)
    with np.errstate(invalid="ignore"):
        return np.select(
            [x <= 0.204, x <= 0.244],
            [np.sqrt(67.552 - 886.788 * x**2) - 8.219, -32.665 * x + 3.981],
            -142.795 * x**3 + 103.656 * x**2 - 57.757 * x + 6.006,
        )


def _a_max(a):
    x = np.abs(a)
    with np.errstate(invalid="ignore"):
        return np.select(
            [x <= 0.13, x <= 0.321],
            [np.sqrt(67.552 - 886.788 * x**2) - 8.219, -15.901 * x + 1.098],
            -4.669 * x**3 + 3.491 * x**2 - 16.699 * x + 1.149,
        )


def _b_min(b):
    x = np.abs(b)
    with np.errstate(invalid="ignore"):
        return np.select(
            [x <= 0.13, x <= 0.145],
            [np.sqrt(16.888 - 886.788 * x**2) - 4.109, -83.607 * x + 8.138],
            -817.81 * x**3 + 355.21 * x**2 - 135.024 * x + 10.619,
        )


def _b_max(b):
    x = np.abs(b)
    with np.errstate(invalid="ignore"):
        return np.select(
            [x <= 0.1, x <= 0.187],
            [np.sqrt(16.888 - 886.788 * x**2) - 4.109, -31.313 * x + 1.854],
            -80.541 * x**3 + 44.174 * x**2 - 39.381 * x + 2.344,
        )


def _a0(rc):
    """Where the A curve is 20 dB down."""
    return np.select(
        [rc < 9.52e4, rc < 8.57e5], [0.57, -9.57e-13 * (rc - 8.57e5) ** 2 + 1.13], 1.13
    )


def _ratio(lo, hi):
    """How far between the curves for the lowest and the highest Reynolds numbers."""
    return (20.0 + lo) / (lo - hi)


def stalled(alpha, mach) -> np.ndarray:
    """Whether an airfoil at ``alpha`` (degrees) in a flow at ``mach`` has stalled for
    the TBL-TE noise (the report's ``gamma_0`` or 12.5 degrees)."""
    alpha = np.asarray(alpha, dtype=float)
    return (alpha >= 23.430 * np.asarray(mach) + 4.651) | (alpha > 12.5)


def tbl_te(
    frequencies,
    chord,
    span,
    distance,
    theta_deg,
    phi_deg,
    alpha,
    speed,
    *,
    trip: int,
    viscosity: float,
    sound_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TBL-TE and separation noise (the report's equations 24 to 30): the band levels
    (dB, ``(..., frequencies)``) of the pressure side, the suction side and the angle
    of attack's part (the flow separating). Stalled, the sides' levels go and the angle
    of attack's part is that of a translating dipole."""
    f = np.asarray(frequencies, dtype=float)
    p = [np.asarray(v, dtype=float)[..., None] for v in (chord, span, distance)]
    c, span_, r = p
    theta = np.asarray(theta_deg, dtype=float)[..., None]
    phi = np.asarray(phi_deg, dtype=float)[..., None]
    alpha = np.asarray(alpha, dtype=float)[..., None]
    u = np.asarray(speed, dtype=float)[..., None]
    rc = u * c / viscosity
    m = u / sound_speed
    _, dstar_s, dstar_p = thicknesses(c, u, alpha, trip, viscosity)
    low, high = directivity_low(m, theta, phi), directivity_high(m, theta, phi)
    r_dstar_p = dstar_p * u / viscosity
    st1 = 0.02 * m**-0.6
    st2 = np.select(
        [alpha <= 1.333, alpha <= 12.5],
        [st1, st1 * 10.0 ** (0.0054 * (alpha - 1.333) ** 2)],
        4.72 * st1,
    )
    st1_prime = (st1 + st2) / 2.0
    a0, a02 = _a0(rc), _a0(3.0 * rc)
    ra0 = _ratio(_a_min(a0), _a_max(a0))
    ra02 = _ratio(_a_min(a02), _a_max(a02))
    b0 = np.select([rc < 9.52e4, rc < 8.57e5], [0.30, -4.48e-13 * (rc - 8.57e5) ** 2 + 0.56], 0.56)
    rb0 = _ratio(_b_min(b0), _b_max(b0))

    def curve_a(x, ratio):
        lo = _a_min(x)
        return lo + ratio * (_a_max(x) - lo)

    lr = np.log10(rc)
    k1 = np.select([rc < 2.47e5, rc < 8.0e5], [-4.31 * lr + 156.3, -9.0 * lr + 181.6], 128.5)
    with np.errstate(divide="ignore"):
        dk1 = np.where(r_dstar_p <= 5000.0, -alpha * (5.29 - 1.43 * np.log10(r_dstar_p)), 0.0)
    gamma, beta = 27.094 * m + 3.31, 72.650 * m + 10.74
    gamma0, beta0 = 23.430 * m + 4.651, -34.190 * m - 13.820
    with np.errstate(invalid="ignore"):
        k2 = np.select(
            [alpha <= gamma0 - gamma, alpha <= gamma0 + gamma],
            [-1000.0, np.sqrt(beta**2 - (beta / gamma) ** 2 * (alpha - gamma0) ** 2) + beta0],
            -12.0,
        )
    k2 = k2 + k1
    st_p, st_s = f * dstar_p / u, f * dstar_s / u
    stall = stalled(alpha, m)
    b = np.abs(np.log10(st_s / st2))
    scale_s = _log(dstar_s * m**5 * span_ / r**2)
    attached_p = curve_a(np.log10(st_p / st1), ra0) + k1 - 3.0
    attached_p = attached_p + _log(dstar_p * m**5 * high * span_ / r**2) * 10.0 + dk1
    attached_s = curve_a(np.log10(st_s / st1_prime), ra0) + k1 - 3.0 + 10.0 * (scale_s + _log(high))
    attached_alpha = _b_min(b) + rb0 * (_b_max(b) - _b_min(b)) + k2 + 10.0 * (scale_s + _log(high))
    stalled_side = 10.0 * (scale_s + _log(low))
    stalled_alpha = curve_a(b, ra02) + k2 + stalled_side
    out = (
        np.where(stall, stalled_side, attached_p),
        np.where(stall, stalled_side, attached_s),
        np.where(stall, stalled_alpha, attached_alpha),
    )
    pressure, suction, angle = (np.maximum(level, FLOOR_DB) for level in out)
    return pressure, suction, angle


def lbl_vs(
    frequencies,
    chord,
    span,
    distance,
    theta_deg,
    phi_deg,
    alpha,
    speed,
    *,
    trip: int,
    viscosity: float,
    sound_speed: float,
) -> np.ndarray:
    """LBL-VS noise (the report's equations 53 to 60), band levels (dB)."""
    f = np.asarray(frequencies, dtype=float)
    c = np.asarray(chord, dtype=float)[..., None]
    span_ = np.asarray(span, dtype=float)[..., None]
    r = np.asarray(distance, dtype=float)[..., None]
    alpha = np.asarray(alpha, dtype=float)[..., None]
    u = np.asarray(speed, dtype=float)[..., None]
    theta = np.asarray(theta_deg, dtype=float)[..., None]
    phi = np.asarray(phi_deg, dtype=float)[..., None]
    rc = u * c / viscosity
    m = u / sound_speed
    delta_p, _, _ = thicknesses(c, u, alpha, trip, viscosity)
    high = directivity_high(m, theta, phi)
    st1_prime = np.select([rc <= 1.3e5, rc <= 4.0e5], [0.18, 0.001756 * rc**0.3931], 0.28)
    st_peak = 10.0 ** (-0.04 * alpha) * st1_prime
    rc0 = np.where(alpha <= 3.0, 10.0 ** (0.215 * alpha + 4.978), 10.0 ** (0.120 * alpha + 5.263))
    ld = np.log10(rc / rc0)
    d = rc / rc0
    g2 = np.select(
        [d <= 0.3237, d <= 0.5689, d <= 1.7579, d <= 3.0889],
        [77.852 * ld + 15.328, 65.188 * ld + 9.125, -114.052 * ld**2, -65.188 * ld + 9.125],
        -77.852 * ld + 15.328,
    )
    g3 = 171.04 - 3.03 * alpha
    scale = 10.0 * _log(delta_p * m**5 * high * span_ / r**2)
    e = (f * delta_p / u) / st_peak
    le = np.log10(e)
    with np.errstate(invalid="ignore"):
        g1 = np.select(
            [e < 0.5974, e <= 0.8545, e < 1.17, e < 1.674],
            [
                39.8 * le - 11.12,
                98.409 * le + 2.0,
                -5.076 + np.sqrt(2.484 - 506.25 * le**2),
                -98.409 * le + 2.0,
            ],
            -39.8 * le - 11.12,
        )
    return g1 + g2 + g3 + scale


def _g5(hd, eta):
    """The bluntness spectrum's shape at ``h / delta*`` (the report's equations 82 to
    85, as its code has them)."""
    mu = np.select(
        [hd < 0.25, hd <= 0.62, hd < 1.15],
        [0.1211, -0.2175 * hd + 0.1755, -0.0308 * hd + 0.0596],
        0.0242,
    )
    m = np.select(
        [hd <= 0.02, hd <= 0.5, hd <= 0.62, hd <= 1.15, hd <= 1.2],
        [
            0.0,
            68.724 * hd - 1.35,
            308.475 * hd - 121.23,
            224.811 * hd - 69.354,
            1583.28 * hd - 1631.592,
        ],
        268.344,
    )
    m = np.maximum(m, 0.0)
    eta0 = -np.sqrt((m * m * mu**4) / (6.25 + m * m * mu * mu))
    k = 2.5 * np.sqrt(1.0 - (eta0 / mu) ** 2) - 2.5 - m * eta0
    with np.errstate(invalid="ignore"):
        return np.select(
            [eta <= eta0, eta <= 0.0, eta <= 0.03616],
            [
                m * eta + k,
                2.5 * np.sqrt(1.0 - (eta / mu) ** 2) - 2.5,
                np.sqrt(1.5625 - 1194.99 * eta**2) - 1.25,
            ],
            -155.543 * eta + 4.375,
        )


def bluntness(
    frequencies,
    chord,
    span,
    distance,
    theta_deg,
    phi_deg,
    alpha,
    speed,
    thickness,
    angle_deg,
    *,
    trip: int,
    viscosity: float,
    sound_speed: float,
) -> np.ndarray:
    """Trailing-edge bluntness noise (the report's equations 73 to 85), band levels
    (dB), for a trailing edge ``thickness`` thick (m) whose surfaces meet at
    ``angle_deg`` (14 degrees for a NACA 0012)."""
    f = np.asarray(frequencies, dtype=float)
    c = np.asarray(chord, dtype=float)[..., None]
    span_ = np.asarray(span, dtype=float)[..., None]
    r = np.asarray(distance, dtype=float)[..., None]
    alpha = np.asarray(alpha, dtype=float)[..., None]
    u = np.asarray(speed, dtype=float)[..., None]
    h = np.asarray(thickness, dtype=float)[..., None]
    psi = np.asarray(angle_deg, dtype=float)[..., None]
    theta = np.asarray(theta_deg, dtype=float)[..., None]
    phi = np.asarray(phi_deg, dtype=float)[..., None]
    m = u / sound_speed
    _, dstar_s, dstar_p = thicknesses(c, u, alpha, trip, viscosity)
    hd = h / ((dstar_s + dstar_p) / 2.0)
    dh = 1.0 / hd
    high = directivity_high(m, theta, phi)
    st_peak = np.where(
        hd >= 0.2,
        (0.212 - 0.0045 * psi) / (1.0 + 0.235 * dh - 0.0132 * dh**2),
        0.1 * hd + 0.095 - 0.00243 * psi,
    )
    g4 = np.where(hd <= 5.0, 17.5 * np.log10(hd) + 157.5 - 1.114 * psi, 169.7 - 1.114 * psi)
    eta = np.log10(f * h / u / st_peak)
    g514 = _g5(hd, eta)
    g50 = _g5(6.724 * hd**2 - 4.019 * hd + 1.107, eta)
    g5 = np.minimum(np.minimum(g50 + 0.0714 * psi * (g514 - g50), 0.0), _g5(0.25, eta))
    return g4 + g5 + 10.0 * _log(m**5.5 * h * high * span_ / r**2)


def tip(
    frequencies,
    chord,
    distance,
    theta_deg,
    phi_deg,
    alpha_tip,
    speed,
    *,
    rounded: bool,
    sound_speed: float,
    lift_ratio: float = 1.0,
) -> np.ndarray:
    """Tip vortex formation noise (the report's equations 61 to 67), band levels (dB),
    for a tip at ``alpha_tip`` (its lift's slope near the tip ``lift_ratio`` times that
    of the report's untwisted blade), rounded or square."""
    f = np.asarray(frequencies, dtype=float)
    c = np.asarray(chord, dtype=float)[..., None]
    r = np.asarray(distance, dtype=float)[..., None]
    a = np.asarray(alpha_tip, dtype=float)[..., None] * lift_ratio
    u = np.asarray(speed, dtype=float)[..., None]
    theta = np.asarray(theta_deg, dtype=float)[..., None]
    phi = np.asarray(phi_deg, dtype=float)[..., None]
    m = u / sound_speed
    high = directivity_high(m, theta, phi)
    if rounded:
        length = 0.008 * a * c
    else:
        length = np.where(np.abs(a) <= 2.0, 0.023 + 0.0169 * a, 0.0378 + 0.0095 * a) * c
    m_max = (1.0 + 0.036 * a) * m
    scale = 10.0 * _log(m * m * m_max**3 * length**2 * high / r**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = f * length / (m_max * sound_speed)
        return 126.0 - 30.5 * (np.log10(st) + 0.3) ** 2 + scale


__all__ = [
    "FLOOR_DB",
    "FREQUENCIES_HZ",
    "LIGHTLY_TRIPPED",
    "TRIPPED",
    "UNTRIPPED",
    "bluntness",
    "directivity_high",
    "directivity_low",
    "lbl_vs",
    "stalled",
    "tbl_te",
    "thicknesses",
    "tip",
]
