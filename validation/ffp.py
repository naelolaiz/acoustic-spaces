"""A fast field program: the sound over a ground in a layered atmosphere, by the wave
equation in full.

    python validation/ffp.py [--json results.json]

A reference solution, written apart from the engine, for the outdoor sound in wind and
temperature gradients (:mod:`validation.benchmark_1995` compares the engine with it).
The pressure of a point source at height zs over a locally reacting ground of
normalised impedance Z, in air whose effective speed of sound c(z) changes with the
height only, is the Hankel transform of the solution of the one-dimensional wave
equation in the height (time as exp(-i omega t); Raspet et al. 1985, Lee et al. 1986):

    p(r, z) = 2 * integral from 0 to infinity of P(kr, z) J0(kr r) kr dkr,
    P'' + (k(z)^2 - kr^2) P = -delta(z - zs),   P'(0) = -i k(0) P(0) / Z,

with P outgoing at the top. The air is cut into thin layers, each of one wavenumber,
and P is found from the ratio of the upgoing to the downgoing wave in each layer,
carried up from the ground (where the impedance sets it) and down from the top (where
nothing comes down), the two met at the source: ratios stay bounded where the field
grows or decays, so the recursion is stable however thick the evanescent layers. The
transform is summed along a contour that dips below the real axis past the steepest
waves, at the spacing of the farthest range, which keeps it clear of the poles of the
trapped and surface waves (the dip's width is what the trapezoidal rule resolves) and
lets the field alias from ranges four times the farthest only through a factor of
exp(-3 pi). The Bessel function is evaluated exactly (its series near the origin, the
Hankel expansion beyond), so the result holds at short ranges too. The air's
absorption, when given, is the imaginary part of the wavenumber.

The checks (``python validation/ffp.py``): the Bessel function against its integral
representation and the series against the expansion where they meet; still air over a
rigid ground against the image source, and over grass against the exact solution for
the locally reacting ground (:mod:`validation.ground_reflection`), 10 Hz to 1 kHz,
100 m to 10 km; and the layers against themselves, halved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

#: The contour dips to this many times the wavenumber spacing below the real axis.
DIP = 1.5
#: The spacing of the wavenumbers resolves ranges this many times the farthest asked.
REACH = 4.0
#: Where the contour dips, as a share of the smallest wavenumber in the air (the
#: steepest trapped wave's horizontal wavenumber is never below it), and the width of
#: the dip's slope, the same way.
DIP_AT = 0.7
DIP_WIDTH = 0.08
#: The transform runs to where the evanescent waves between the source and the
#: nearest listener height have fallen by this many nepers, tapered over the last part.
TAIL_NEPERS = 25.0
TAPER_FROM = 16.0
#: Default layer thickness: this share of the shortest wavelength, at most LAYER_MAX_M.
LAYER_WAVELENGTHS = 0.25
LAYER_MAX_M = 1.0
#: Towards the ground, the top, the source and the listener the layers halve this many
#: times (see :func:`_interfaces`): the floor of the field in a deep shadow, which the
#: layers' residual scatters, falls as the thinnest of them.
REFINEMENTS = 16

_EULER_GAMMA = 0.5772156649015329
_SERIES_UP_TO = 17.0


def hankel0(z) -> np.ndarray:
    """H0^(1)(z) for complex ``z`` (the principal branch): the power series of J0 and
    Y0 within |z| < 17, Hankel's expansion beyond (both to about 1e-12)."""
    z = np.asarray(z, dtype=complex)
    out = np.empty(z.shape, dtype=complex)
    near = np.abs(z) < _SERIES_UP_TO
    if near.any():
        j0, y0 = _series(z[near])
        out[near] = j0 + 1j * y0
    far = ~near
    if far.any():
        out[far] = _expansion(z[far])
    return out


def bessel_j0(z) -> np.ndarray:
    """J0(z) for complex ``z``."""
    z = np.asarray(z, dtype=complex)
    out = np.empty(z.shape, dtype=complex)
    near = np.abs(z) < _SERIES_UP_TO
    if near.any():
        out[near] = _series(z[near])[0]
    far = ~near
    if far.any():
        zf = z[far]
        out[far] = 0.5 * (_expansion(zf) + np.conj(_expansion(np.conj(zf))))
    return out


def _series(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = (0.5 * z) ** 2
    term = np.ones_like(w)
    j0 = np.ones_like(w)
    rest = np.zeros_like(w)
    harmonic = 0.0
    for m in range(1, 60):
        term = -term * w / (m * m)
        harmonic += 1.0 / m
        j0 = j0 + term
        rest = rest - harmonic * term
    y0 = (2.0 / math.pi) * ((np.log(0.5 * z) + _EULER_GAMMA) * j0 + rest)
    return j0, y0


def _expansion(z: np.ndarray) -> np.ndarray:
    total = np.ones_like(z)
    coefficient = 1.0
    power = np.ones_like(z)
    for k in range(1, 25):
        coefficient *= -((2 * k - 1) ** 2) / (8.0 * k)
        power = power * 1j / z
        total = total + coefficient * power
    return np.sqrt(2.0 / (math.pi * z)) * np.exp(1j * (z - 0.25 * math.pi)) * total


def pressure(
    frequency: float,
    speed,
    zs: float,
    ranges,
    zr,
    impedance: complex | None = None,
    top: float = 1000.0,
    absorption_db_per_m: float = 0.0,
    layer: float | None = None,
) -> np.ndarray:
    """(n,) the complex pressure at ``frequency`` (Hz; time as exp(-i omega t), free
    field exp(ikR) / R) at the ranges ``ranges`` (n,, m along the ground) and heights
    ``zr`` (n, or one; m) from a source ``zs`` m up, in air of effective speed of
    sound ``speed(z)`` (m/s, an array function of the height) up to ``top`` and the
    same above, over a ground of normalised impedance ``impedance`` (time as
    exp(-i omega t); None: rigid), the air absorbing ``absorption_db_per_m``. The
    layers are ``layer`` m thick (default: a quarter wavelength, at most 1 m). No
    listener may be at the source's height: the transform's tail is set by the
    nearest."""
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
    heights = np.broadcast_to(np.asarray(zr, dtype=float), ranges.shape).copy()
    if np.any(np.abs(heights - zs) < 1e-9):
        raise ValueError("a listener at the source's height has no transform tail")
    omega = 2.0 * math.pi * frequency
    alpha = absorption_db_per_m / (20.0 * math.log10(math.e))  # nepers per metre
    c0 = float(np.asarray(speed(np.array([0.0])))[0])
    k0 = omega / c0
    if layer is None:
        layer = min(LAYER_WAVELENGTHS * 2.0 * math.pi / k0, LAYER_MAX_M)
    out = np.zeros(len(ranges), dtype=complex)
    for height in np.unique(heights):
        at = heights == height
        # The listener is always below the source (reciprocity when it is not).
        lower, upper = (float(height), float(zs)) if height < zs else (float(zs), float(height))
        interfaces = _interfaces(layer, top, (lower, upper))
        middles = 0.5 * (interfaces[:-1] + interfaces[1:])
        thickness = np.diff(interfaces)
        k_layers = omega / np.asarray(speed(middles), dtype=float) + 1j * alpha
        k_top = omega / float(np.asarray(speed(np.array([top])))[0]) + 1j * alpha
        k_min = float(min(k_layers.real.min(), k_top.real))
        k_max_air = float(max(k_layers.real.max(), k_top.real))
        i_lower = int(np.searchsorted(interfaces, lower))
        i_upper = int(np.searchsorted(interfaces, upper))

        # The contour: t along the real axis, dipping by DIP spacings past DIP_AT.
        farthest = float(ranges[at].max())
        dk = 2.0 * math.pi / (REACH * farthest)
        dip = DIP * dk
        separation = upper - lower
        k_end = math.sqrt(k_max_air**2 + (TAIL_NEPERS / separation) ** 2)
        k_taper = math.sqrt(k_max_air**2 + (TAPER_FROM / separation) ** 2)
        t = (np.arange(int(math.ceil(k_end / dk))) + 0.5) * dk
        slope = np.tanh((t - DIP_AT * k_min) / (DIP_WIDTH * k_min))
        kr = t - 1j * dip * 0.5 * (1.0 + slope)
        dkr = 1.0 - 1j * dip * 0.5 * (1.0 - slope**2) / (DIP_WIDTH * k_min)
        taper = np.ones_like(t)
        tail = t > k_taper
        taper[tail] = 0.5 * (1.0 + np.cos(math.pi * (t[tail] - k_taper) / (k_end - k_taper)))

        amplitude = _depth_solution(kr, k_layers, k_top, thickness, i_lower, i_upper, k0, impedance)
        weights = amplitude * kr * dkr * taper * (2.0 * dk)
        for index in np.flatnonzero(at):
            out[index] = np.sum(weights * bessel_j0(kr * ranges[index]))
    return out


def _interfaces(layer: float, top: float, heights: tuple[float, ...]) -> np.ndarray:
    """The heights of the layers' interfaces: every ``layer`` m from the ground to
    ``top``, the ``heights`` of the source and the listener, and layers halving in
    thickness down to ``layer`` / 2 ** REFINEMENTS towards the ground, the top and each
    of those heights from both sides: the field at an interface is that of the layer
    it ends, whose wavenumber is the one at its middle, so the error there is of the
    first order in the layer's thickness where the wave runs grazing (within the
    first layers the error is amplified by 1 / (k^2 - kr^2)); in the layers between,
    the middle's wavenumber integrates the phase to the second order."""
    special = [0.0, top, *heights]
    offsets = layer / 2.0 ** np.arange(1, REFINEMENTS + 1)
    found = [np.arange(0.0, top, layer), np.asarray(special)]
    for height in special:
        found.append(height + offsets)
        found.append(height - offsets)
    z = np.unique(np.concatenate(found))
    z = z[(z >= 0.0) & (z <= top)]
    return z[np.concatenate([[True], np.diff(z) > 1e-9 * max(layer, 1.0)])]


def _depth_solution(kr, k_layers, k_top, thickness, i_lower, i_upper, k0, impedance):
    """P(kr) at the lower of the two heights (the interfaces ``i_lower`` and
    ``i_upper``), the source at the upper, for the layers ``k_layers`` (N,) of
    ``thickness`` (N,) under a half-space of wavenumber ``k_top``."""
    kr2 = kr**2
    n = len(k_layers)

    def vertical(j: int) -> np.ndarray:
        """The vertical wavenumber in layer ``j`` (the half-space for N): the principal
        root, which has a positive imaginary part all along the contour."""
        k = k_top if j >= n else k_layers[j]
        return np.sqrt(k * k - kr2)

    def across(ratio, q_from, q_into):
        """The ratio of the wave going on to the wave coming back, carried across an
        interface from the layer of ``q_from`` into that of ``q_into``."""
        total, difference = q_into + q_from, q_into - q_from
        return (total * ratio + difference) / (difference * ratio + total)

    # Up from the ground: gamma = U / D, where the ground's admittance sets it.
    q_here = vertical(0)
    if impedance is None:
        gamma = np.ones_like(kr)
    else:
        y = -1j * k0 / impedance
        gamma = (1j * q_here + y) / (1j * q_here - y)
    kept = []  # gamma at the bottom of each layer between the listener and the source
    for j in range(i_upper):
        if j >= i_lower:
            kept.append(gamma)
        gamma = gamma * np.exp(2j * q_here * thickness[j])
        q_next = vertical(j + 1)
        gamma = across(gamma, q_here, q_next)
        q_here = q_next
    q_above = q_here  # just above the source
    q_below = vertical(i_upper - 1)
    # Down from the top: lambda = D / U, nothing coming down above the top.
    lam = np.zeros_like(kr)
    q_here = vertical(n)
    for j in range(n - 1, i_upper - 1, -1):
        q_next = vertical(j)
        lam = across(lam, q_here, q_next)
        lam = lam * np.exp(2j * q_next * thickness[j])
        q_here = q_next
    # The source: P continuous, P' jumps by -1; D just below it.
    denominator = -1j * (
        q_below * (1.0 - gamma) * (1.0 + lam) + q_above * (1.0 - lam) * (1.0 + gamma)
    )
    d = (1.0 + lam) / denominator
    # Down to the listener, carrying the downgoing wave (it decays that way).
    q_here = q_below
    for j in range(i_upper - 1, i_lower - 1, -1):
        gamma_bottom = kept[j - i_lower]
        d = d * np.exp(1j * q_here * thickness[j])
        if j == i_lower:
            return d * (1.0 + gamma_bottom)
        # Into the layer below, from the ratio just above the interface.
        q_next = vertical(j - 1)
        total, difference = q_next + q_here, q_next - q_here
        d = d * (total + difference * gamma_bottom) / (2.0 * q_next)
        q_here = q_next
    raise AssertionError("the listener lies below the source")


# -- checks ---------------------------------------------------------------------------


def check_bessel() -> dict:
    """J0 and Y0 against their integral representations on the real axis (the
    trapezoidal rule with its end corrections), and the series against the expansion
    where they meet, round the circle up to 36 degrees off the real axis (nearer the
    imaginary axis J0 and Y0 cancel in H0 and the series loses digits)."""
    theta = np.linspace(0.0, math.pi, 20001)
    h = theta[1] - theta[0]
    worst = 0.0
    for x in (0.5, 1.0, 5.0, 10.0, 16.9, 17.1, 25.0, 60.0, 200.0):
        j0 = np.trapezoid(np.cos(x * np.sin(theta)), theta) / math.pi
        got = bessel_j0(np.array([x]))[0]
        worst = max(worst, abs(got - j0))
    # Y0(x) = (1/pi) int_0^pi sin(x sin t) dt - (2/pi) int_0^inf exp(-x sinh t) dt.
    u = np.linspace(0.0, 12.0, 200001)
    hu = u[1] - u[0]
    for x in (1.0, 5.0, 10.0, 17.1, 25.0, 60.0):
        y0 = (np.trapezoid(np.sin(x * np.sin(theta)), theta) + h * h * x / 6.0) / math.pi
        y0 -= 2.0 / math.pi * (np.trapezoid(np.exp(-x * np.sinh(u)), u) + hu * hu * x / 12.0)
        got = hankel0(np.array([x]))[0].imag
        worst = max(worst, abs(got - y0))
    angles = np.linspace(-0.2 * math.pi, 0.2 * math.pi, 41)
    ring = _SERIES_UP_TO * np.exp(1j * angles)
    j0_series = _series(ring)[0]
    j0_expansion = 0.5 * (_expansion(ring) + np.conj(_expansion(np.conj(ring))))
    seam = float(np.max(np.abs(j0_series - j0_expansion) / np.abs(j0_expansion)))
    print(
        f"Bessel functions: {worst:.1e} off the integrals, series and expansion "
        f"{seam:.1e} apart at |z| = {_SERIES_UP_TO:g}\n"
    )
    return {"integrals": worst, "seam": seam}


def check_still_air(quick: bool = False) -> dict:
    """Still air: a rigid ground against the image source and grass against the
    exact solution, as levels over the free field's, 10 Hz to 1 kHz."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ground_reflection import exact_reflection, field

    from acoustic_spaces.materials.ground import PRESETS, surface_impedance

    c = 343.0
    still = lambda z: np.full(np.shape(z), c)  # noqa: E731
    zs, zr = 5.0, 1.0
    ranges = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0][: 3 if quick else 5])
    r1, r2 = np.hypot(ranges, zs - zr), np.hypot(ranges, zs + zr)
    out = {}
    print("Still air, source 5 m and listener 1 m up: the level over the free field's")
    print("(dB) and the fast field program's difference from the exact solution\n")
    print(f"{'':24}" + "".join(f"{f'{r:g} m':>12}" for r in ranges))
    start = time.time()
    for f in (10.0, 100.0, 1000.0):
        k = 2.0 * math.pi * f / c
        for name, z in (
            ("rigid", None),
            (
                "grass",
                complex(surface_impedance(np.array(f), PRESETS["grass"][0], speed_of_sound=c)),
            ),
        ):
            if z is None:
                exact = np.exp(1j * k * r1) / r1 + np.exp(1j * k * r2) / r2
                exact = 20.0 * np.log10(np.abs(exact) * r1)
            else:
                exact = np.array(
                    [
                        20.0
                        * math.log10(abs(field(exact_reflection(1.0 / z, k, r, zs, zr), k, a, b)))
                        for r, a, b in zip(ranges, r1, r2, strict=True)
                    ]
                )
            got = pressure(f, still, zs, ranges, zr, None if z is None else np.conj(z))
            level = 20.0 * np.log10(np.abs(got) * r1)
            out[f"{f:g} Hz, {name}"] = {"exact_db": exact.tolist(), "ffp_db": level.tolist()}
            print(f"{f'{f:g} Hz, {name}':24}" + "".join(f"{v:12.2f}" for v in exact) + "  exact")
            print(f"{'':24}" + "".join(f"{v:+12.3f}" for v in level - exact) + "  ffp - exact")
    print(f"\n({time.time() - start:.0f} s)\n")
    return out


def check_layers() -> dict:
    """The layers against themselves, halved: downward refraction of 0.1/s to 1000 m
    over a ground of flow resistivity 200 kPa s/m² (Delany and Bazley), the benchmark
    case that bends the sound most, at 1 and 10 km."""
    from acoustic_spaces.materials.ground import surface_impedance

    zs, zr = 5.0, 1.0
    ranges = np.array([1000.0, 10000.0])
    speed = lambda z: 343.0 + 0.1 * np.minimum(np.asarray(z, dtype=float), 1000.0)  # noqa: E731
    print("Downward refraction, 0.1/s to 1000 m: the level over the free field's (dB)")
    print("with the default layers, and the change when they are halved\n")
    out = {}
    start = time.time()
    for f in (100.0, 1000.0):
        z = np.conj(complex(surface_impedance(np.array(f), 200.0, "delany_bazley")))
        layer = min(LAYER_WAVELENGTHS * 343.0 / f, LAYER_MAX_M)
        levels = []
        for thickness in (layer, 0.5 * layer):
            p = pressure(f, speed, zs, ranges, zr, z, layer=thickness)
            levels.append(20.0 * np.log10(np.abs(p) * np.hypot(ranges, zs - zr)))
        out[f"{f:g} Hz"] = {
            "layer_m": layer,
            "level_db": levels[0].tolist(),
            "halved_change_db": (levels[1] - levels[0]).tolist(),
        }
        print(
            f"{f:5g} Hz, layers {layer:.3g} m:"
            + "".join(f"{v:9.2f}" for v in levels[0])
            + "   halved:"
            + "".join(f"{v:+8.3f}" for v in levels[1] - levels[0])
        )
    print(f"\n({time.time() - start:.0f} s)\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="ranges to 1 km only")
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args(argv)
    results = {"bessel": check_bessel(), "still_air": check_still_air(args.quick)}
    if not args.quick:
        results["layers"] = check_layers()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
