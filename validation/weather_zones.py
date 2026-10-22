"""Check the weather zones against a Crank-Nicholson parabolic equation.

    python validation/weather_zones.py [--only pe|mean] [--json results.json]

An atmosphere's ``zones`` give parts of a scene weather of their own, so that the wind
and the temperature change along the way from the source to the listener (see
:mod:`acoustic_spaces.engine.weather`). Two checks, over grass (Miki's impedance, 200
kPa s/m²), the source 2 m up and the listener 1.5 m up, 100 to 800 m away along the
wind, the weather changing 150 m from the source:

- The engine's parabolic equation marched through the zones (the Green's function
  parabolic equation, each step taking the profile of the zone it is over) against a
  Crank-Nicholson parabolic equation (Salomons 2001, app. G) written here, which takes
  the profile at each step as a matter of course: an independent discretisation of the
  same narrow-angle equation, on a grid of a tenth of a wavelength in height and a
  quarter of one in range, at 63 to 500 Hz, the bands the engine takes from its
  parabolic equation over open ground. In still air the Crank-Nicholson equation is
  first checked against the exact solution, and in wind without zones against the
  engine's parabolic equation, which the fast field program validated
  (``benchmark_1995.py``). The engine's wide-angle parabolic equation
  (``simulation.early.parabolic_equation: wide_angle``) is marched through the zones
  too, against the same reference.
- The mean profile along the way, which the early paths take (each weather weighted by
  the length of the way under it, the wind's component along the way with it), against
  the weathers in turn: the engine's parabolic equation through that mean profile
  against the same through the zones, in the 500 Hz, 1 kHz and 2 kHz bands, at the
  ranges past the change. The parabolic equation stands in for the early paths, so
  the difference is the equivalent profile's alone, apart from the paths' own
  accuracy (checked against an independent ray tracer, ``outdoor_propagation.py``).

Levels are in dB over the free field at the direct distance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acoustic_spaces.engine import wide_angle  # noqa: E402
from acoustic_spaces.engine.diffraction import band_frequencies  # noqa: E402
from acoustic_spaces.engine.parabolic import march  # noqa: E402
from acoustic_spaces.engine.weather import Weather  # noqa: E402
from acoustic_spaces.materials.ground import spherical_reflection, surface_impedance  # noqa: E402
from acoustic_spaces.scene import Atmosphere  # noqa: E402

GROUND_C = 20.0
SIGMA = 200.0  # kPa s/m², grass
ZS, ZR = 2.0, 1.5
#: The pure tones of the parabolic equations' check: the bands the engine takes over
#: open ground from its parabolic equation.
TONES = (63.0, 125.0, 250.0, 500.0)
RANGES = (100.0, 200.0, 400.0, 800.0)
#: Where the weather changes (m from the source), and the zones' reach.
CHANGE_M = 150.0
FAR_M = 5000.0
#: The bands of the mean profile's check.
MEAN_BANDS = {"500 Hz": 3, "1 kHz": 5, "2 kHz": 6}
#: Levels re free field under this are deep in a shadow: inaudible next to any other
#: sound, and at the parabolic equation's own floor; they are shown but not summed up.
SHADOW_DB = -40.0
#: The absorbing layer of the parabolic equations starts this high (m): the engine's
#: rule for its own (WAVE_TOP_SLOPE times the range, at least the headroom).
TOP_M = 120.0


def zone(from_m: float, to_m: float, **weather) -> dict:
    return {
        "polygon": [[from_m, -FAR_M], [to_m, -FAR_M], [to_m, FAR_M], [from_m, FAR_M]],
        **weather,
    }


#: The cases: the weather near the source, and the zone it changes into.
CASES: dict[str, dict] = {
    "shelter downwind": {
        "preset": "windy",
        "zones": [zone(CHANGE_M, FAR_M, wind={"speed_m_s": 0.5})],
    },
    "gust downwind": {
        "wind": {"speed_m_s": 0.5},
        "zones": [zone(CHANGE_M, FAR_M, wind={"speed_m_s": 5.0})],
    },
    "shelter upwind": {
        "wind": {"speed_m_s": 5.0, "direction_deg": 180.0},
        "zones": [zone(CHANGE_M, FAR_M, wind={"speed_m_s": 0.5})],
    },
    "inversion past a lake shore": {
        "preset": "neutral",
        "zones": [zone(CHANGE_M, FAR_M, preset="clear_night")],
    },
    "cool water downwind": {
        "preset": "windy",
        "zones": [zone(100.0, 300.0, temperature_c=12.0)],
    },
}
#: Without zones, for the Crank-Nicholson equation's own checks.
PLAIN: dict[str, dict | None] = {"still air": None, "windy": {"preset": "windy"}}


def impedance(f) -> np.ndarray:
    """The ground's normalised impedance, time as exp(-i omega t)."""
    return np.conj(surface_impedance(np.asarray(f, dtype=float), SIGMA))


def weather_of(atmosphere: dict | None) -> Weather:
    """The weathers along +x of ``atmosphere`` (still air for None)."""
    found = Atmosphere.model_validate(atmosphere or {})
    return Weather(found, GROUND_C, (-50.0, -FAR_M, FAR_M, FAR_M))


def free_field_db(p: np.ndarray, ranges) -> np.ndarray:
    ranges = np.asarray(ranges, dtype=float)
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.abs(p) * np.hypot(ranges, ZS - ZR))


# -- the Crank-Nicholson parabolic equation ---------------------------------------------


def cnpe(f: float, speed_at, zg: complex, zs: float, ranges, zr: float, top: float) -> np.ndarray:
    """(n,) the pressure (time as exp(-i omega t), free field exp(ikR) / R) at the
    ranges (m along the ground) and height ``zr`` of a tone at ``f`` Hz from a source
    ``zs`` up, over a ground of normalised impedance ``zg``, by the Crank-Nicholson
    parabolic equation: q = p sqrt(r) exp(-i ka r) stepped by

        (1 - dr M / 2) q(r + dr) = (1 + dr M / 2) q(r),   M = i / (2 ka) (d²/dz² + k² - ka²),

    second-order differences in height (a tenth of a wavelength), the ground's
    impedance in a ghost point (dq/dz + i k beta q = 0), an absorbing layer above
    ``top`` and q = 0 at its top, Salomons' Gaussian starting field (eq. G.75) with the
    ground's image at normal incidence. ``speed_at(r)`` gives the speed function (of
    the height) that holds at range r; the profile is rebuilt where it changes."""
    ranges = np.asarray(ranges, dtype=float)
    c0 = float(np.asarray(speed_at(0.0)(np.array([0.0])))[0])
    ka = 2.0 * math.pi * f / c0
    wavelength = c0 / f
    dz = wavelength / 10.0
    dr = wavelength / 4.0
    thickness = max(30.0 * wavelength, 20.0)
    n = int(math.ceil((top + thickness) / dz)) + 1
    z = np.arange(n) * dz
    depth = np.clip((z - top) / thickness, 0.0, 1.0)
    beta = 1.0 / zg

    def gaussian(x):
        return (1.3717 - 0.3701 * x**2) * np.exp(-(x**2) / 3.0)

    q = np.sqrt(1j * ka) * (
        gaussian(ka * (z - zs)) + (zg - 1.0) / (zg + 1.0) * gaussian(ka * (z + zs))
    )
    current = None
    matrices = None

    def build(speed, step):
        """The tridiagonals of A = 1 - dr M / 2 (its Thomas sweep ready) and B = 1 +
        dr M / 2 for the profile ``speed`` and the range step ``step``."""
        k = 2.0 * math.pi * f / np.asarray(speed(z), dtype=float) + 1j * depth**2
        g = 1j * step / (4.0 * ka)
        off = g / dz**2
        diagonal = g * (-2.0 / dz**2 + k * k - ka * ka)
        # The ground's row: the ghost point folds into the diagonal and the upper term.
        kg = 2.0 * math.pi * f / float(np.asarray(speed(np.array([0.0])))[0])
        diagonal[0] = g * (-2.0 / dz**2 + 2j * kg * beta / dz + k[0] * k[0] - ka * ka)
        upper = np.full(n - 1, off, dtype=complex)
        upper[0] = 2.0 * off
        lower = np.full(n - 1, off, dtype=complex)
        a_lower, a_diag, a_upper = -lower, 1.0 - diagonal, -upper
        b_lower, b_diag, b_upper = lower, 1.0 + diagonal, upper
        # The top: q stays 0.
        a_lower[-1] = b_lower[-1] = 0.0
        a_diag[-1] = b_diag[-1] = 1.0
        # Thomas: the pivots and the multipliers of the forward sweep.
        pivots = np.empty(n, dtype=complex)
        multipliers = np.empty(n - 1, dtype=complex)
        pivots[0] = a_diag[0]
        for i in range(1, n):
            multipliers[i - 1] = a_lower[i - 1] / pivots[i - 1]
            pivots[i] = a_diag[i] - multipliers[i - 1] * a_upper[i - 1]
        return (a_upper, pivots, multipliers), (b_lower, b_diag, b_upper)

    def solve(a, d):
        a_upper, pivots, multipliers = a
        y = np.empty_like(d)
        y[0] = d[0]
        for i in range(1, n):
            y[i] = d[i] - multipliers[i - 1] * y[i - 1]
        x = np.empty_like(d)
        x[-1] = y[-1] / pivots[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (y[i] - a_upper[i] * x[i + 1]) / pivots[i]
        return x

    out = np.zeros(len(ranges), dtype=complex)
    r = 0.0
    for index in np.argsort(ranges, kind="stable"):
        goal = ranges[index]
        while goal - r > 1e-9:
            step = min(dr, goal - r)
            speed = speed_at(r + 0.5 * step)
            if speed is not current or matrices is None or matrices[0] != step:
                current = speed
                matrices = (step, *build(speed, step))
            _, a, (b_lower, b_diag, b_upper) = matrices
            rhs = b_diag * q
            rhs[:-1] += b_upper * q[1:]
            rhs[1:] += b_lower * q[:-1]
            q = solve(a, rhs)
            r += step
        p = q * np.exp(1j * ka * r) / math.sqrt(r)
        out[index] = np.interp(zr, z, p.real) + 1j * np.interp(zr, z, p.imag)
    return out


def exact_still(f: float, ranges) -> np.ndarray:
    """The exact pressure in still air (time as exp(-i omega t))."""
    c = float(weather_of(None).still[0, 0])
    k = 2.0 * math.pi * f / c
    out = []
    for r in np.asarray(ranges, dtype=float):
        r1, r2 = math.hypot(r, ZS - ZR), math.hypot(r, ZS + ZR)
        # The engine's convention is exp(j omega t): conjugate in and out.
        q = spherical_reflection(1.0 / surface_impedance(np.array(f), SIGMA), (ZS + ZR) / r2, k, r2)
        out.append(np.conj(np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2))
    return np.array(out, dtype=complex)


def gfpe(f: float, speed_at, ranges, solver=march) -> np.ndarray:
    """The engine's parabolic equation through the weathers along +x (``solver``: its
    march, or the wide-angle equation's)."""
    z = complex(impedance(f))
    return solver(
        [f],
        speed_at(0.0),
        lambda r: z,
        ZS,
        np.asarray(ranges, dtype=float),
        np.full(len(ranges), ZR),
        TOP_M,
        speed_at=speed_at,
    )[0]


def check_parabolic(tones=TONES, ranges=RANGES) -> dict:
    """The engine's parabolic equation through the zones against the Crank-Nicholson
    one, after the Crank-Nicholson one's own checks."""
    ranges = np.asarray(ranges, dtype=float)
    print("The parabolic equation through weather zones against a Crank-Nicholson parabolic")
    print("equation: the level re free field (dB), and the difference\n")
    print(f"{'':34}" + "".join(f"{f'{r:g} m':>9}" for r in ranges))
    out: dict = {}
    start = time.time()
    for name, atmosphere in PLAIN.items():
        weather = weather_of(atmosphere)
        speed_at = weather.along(np.zeros(2), np.array([1.0, 0.0]), ranges.max())
        for f in tones:
            cn = free_field_db(
                cnpe(f, speed_at, complex(impedance(f)), ZS, ranges, ZR, TOP_M), ranges
            )
            if atmosphere is None:
                reference = free_field_db(exact_still(f, ranges), ranges)
                against = "exact"
            else:
                reference = free_field_db(gfpe(f, speed_at, ranges), ranges)
                against = "engine"
            label = f"{name}, {f:g} Hz"
            out[label] = {
                "reference_db": reference.tolist(),
                "cnpe_minus_reference_db": (cn - reference).tolist(),
            }
            print(f"{label:34}" + "".join(f"{v:9.2f}" for v in reference) + f"  {against}")
            print(
                f"{'':34}" + "".join(f"{v:+9.2f}" for v in cn - reference) + "  cnpe - " + against
            )
    print()
    misses, wide_misses = [], []
    for name, atmosphere in CASES.items():
        weather = weather_of(atmosphere)
        speed_at = weather.along(np.zeros(2), np.array([1.0, 0.0]), ranges.max())
        for f in tones:
            engine = free_field_db(gfpe(f, speed_at, ranges), ranges)
            wide = free_field_db(gfpe(f, speed_at, ranges, wide_angle.march), ranges)
            cn = free_field_db(
                cnpe(f, speed_at, complex(impedance(f)), ZS, ranges, ZR, TOP_M), ranges
            )
            miss = engine - cn
            misses.extend(miss.tolist())
            wide_misses.extend((wide - cn).tolist())
            label = f"{name}, {f:g} Hz"
            out[label] = {
                "cnpe_db": cn.tolist(),
                "engine_minus_cnpe_db": miss.tolist(),
                "wide_minus_cnpe_db": (wide - cn).tolist(),
            }
            print(f"{label:34}" + "".join(f"{v:9.2f}" for v in cn) + "  cnpe")
            print(f"{'':34}" + "".join(f"{v:+9.2f}" for v in miss) + "  engine - cnpe")
            print(f"{'':34}" + "".join(f"{v:+9.2f}" for v in wide - cn) + "  wide - cnpe")
    misses, wide_misses = np.abs(misses), np.abs(wide_misses)
    out["summary"] = {
        "mean_abs_db": float(misses.mean()),
        "max_abs_db": float(misses.max()),
        "wide_mean_abs_db": float(wide_misses.mean()),
        "wide_max_abs_db": float(wide_misses.max()),
    }
    print(
        f"\nengine - cnpe over the zones: {misses.mean():.2f} dB on average, {misses.max():.2f} "
        f"at most; the wide-angle equation {wide_misses.mean():.2f} on average, "
        f"{wide_misses.max():.2f} at most ({time.time() - start:.0f} s)\n"
    )
    return out


# -- the mean profile -----------------------------------------------------------------


def check_mean_profile(ranges=RANGES) -> dict:
    """The parabolic equation through the mean profile along the way, as the early
    paths take it, against the parabolic equation through the zones, band by band."""
    ranges = np.asarray(ranges, dtype=float)
    ranges = ranges[ranges > CHANGE_M]  # nearer, the way lies under one weather
    frequencies = band_frequencies()[0]  # (bands, M)
    print("The parabolic equation through the mean profile along the way (the early")
    print("paths' profile) against the parabolic equation through the zones: the level re")
    print("free field (dB), and the difference\n")
    print(f"{'':34}" + "".join(f"{f'{r:g} m':>9}" for r in ranges))
    out: dict = {}
    misses = []
    shadowed = 0
    start = time.time()
    for name, atmosphere in CASES.items():
        weather = weather_of(atmosphere)
        speed_at = weather.along(np.zeros(2), np.array([1.0, 0.0]), ranges.max())
        for band_name, band in MEAN_BANDS.items():
            z = impedance(frequencies[band])
            distances = np.hypot(ranges, ZS - ZR)
            zoned = march(
                frequencies[band],
                speed_at(0.0),
                lambda r, z=z: z,
                ZS,
                ranges,
                np.full(len(ranges), ZR),
                TOP_M,
                speed_at=speed_at,
            )  # (M, n)
            zoned = 10.0 * np.log10(np.mean(np.abs(zoned) ** 2, axis=0) * distances**2)
            mean = []
            for r, distance in zip(ranges, distances, strict=True):
                profile = weather.profile(*weather.shares([[0.0, 0.0], [r, 0.0]]))
                wave = march(
                    frequencies[band],
                    profile.speed,
                    lambda r, z=z: z,
                    ZS,
                    np.array([r]),
                    np.array([ZR]),
                    TOP_M,
                )[:, 0]
                mean.append(10.0 * math.log10(float(np.mean(np.abs(wave) ** 2)) * distance**2))
            miss = np.array(mean) - zoned
            audible = zoned > SHADOW_DB
            misses.extend(miss[audible].tolist())
            shadowed += int((~audible).sum())
            label = f"{name}, {band_name}"
            out[label] = {"zones_db": zoned.tolist(), "mean_minus_zones_db": miss.tolist()}
            print(f"{label:34}" + "".join(f"{v:9.2f}" for v in zoned) + "  zones")
            print(f"{'':34}" + "".join(f"{v:+9.2f}" for v in miss) + "  mean - zones")
    misses = np.abs(misses)
    out["summary"] = {
        "mean_abs_db": float(misses.mean()),
        "max_abs_db": float(misses.max()),
        "shadow_floor_db": SHADOW_DB,
        "left_out": shadowed,
    }
    print(
        f"\nmean - zones: {misses.mean():.2f} dB on average, {misses.max():.2f} at most, over "
        f"the {misses.size} points above {SHADOW_DB:g} dB re free field ({shadowed} deeper in "
        f"a shadow left out; {time.time() - start:.0f} s)\n"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["pe", "mean"], help="run one check")
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    results = {}
    if args.only in (None, "pe"):
        results["parabolic"] = check_parabolic()
    if args.only in (None, "mean"):
        results["mean_profile"] = check_mean_profile()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
