"""Check the helicopter's sound (:mod:`acoustic_spaces.rotorcraft`) against Gutin's
formula, the far field summed over the blades in two independent ways, the sample
outputs of Brooks, Pope and Marcolini's code, the blades' pitch in closed form, Pegg's
worked example and the certified noise level of the helicopter it describes.

    python validation/rotorcraft.py [--json results.json]

- **Gutin.** A hovering rotor with its thrust and torque on one radius: each harmonic
  of the blade passing frequency against Gutin's (1936) formula.
- **Round the turn.** The AS350 B3's main and tail rotors at 63 m/s, each harmonic
  against the far field's integral over a turn summed numerically over the blades'
  points where they truly are (the straight chord), as moving forces and sources.
- **In time.** The same rotors' sound at an observer 20,000 km away in eight directions,
  each blade point a moving force or source of Farassat's formulation 1A, heard at the
  time it left (solved by Newton's method): its harmonics against the tones'.
- **The store.** What :class:`~acoustic_spaces.emission.Periodic` sends in random
  directions at random times, against the harmonics summed there, on grids 5 and 2.5
  degrees apart.
- **Airfoil self-noise.** :mod:`acoustic_spaces.airfoil_noise` against the two sample
  outputs of Brooks, Pope and Marcolini's code (NASA RP-1218, tables D2 and D3).
- **The blades' pitch.** :func:`~acoustic_spaces.rotorcraft.pitch` against its closed
  form (Johnson, *Helicopter Theory*, 5.6): rigid blades from the hub to the tip, in
  an even inflow, with no flapping.
- **Pegg.** His example I (NASA TM-80200): its broadband levels and peak frequencies.
- **The certified level.** The AS350 B3 at 2,250 kg flying the overflight of ICAO Annex
  16 chapter 11 (14 CFR 36 appendix J): level at 150 m over a microphone 1.2 m above
  grass (Miki's impedance, 200 kPa s/m²), at 0.9 V_H (V_H taken as Airbus's fast cruise
  speed, 136 kt: 63.0 m/s), 25 °C, 70 % humidity, sea level. The direct sound and the
  ground's reflection (Weyl and van der Pol), each with the air's absorption, heard at
  the time it left the helicopter, A-weighted, with the slow time weighting to find the
  10 dB down points; its sound exposure level against the certified 84.6 dB(A) (EASA
  type certificate data sheet for noise EASA.R.008), each part of the sound alone too,
  with the broadband noise by each model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.special import jv

from acoustic_spaces import airfoil_noise as bpm
from acoustic_spaces import rotorcraft as rc
from acoustic_spaces.air import absorption_db_per_m
from acoustic_spaces.directivity import _sphere
from acoustic_spaces.doppler import read
from acoustic_spaces.emission import Sum
from acoustic_spaces.materials.ground import PRESETS, spherical_reflection, surface_impedance

SPEED = 63.0
TEMPERATURE, HUMIDITY = 25.0, 70.0
HEIGHT, MICROPHONE = 150.0, 1.2
CERTIFIED_SEL = 84.6
RATE = 48_000
#: The report's sample outputs, as the tests read them.
BPM_TABLES = Path(__file__).resolve().parent.parent / "tests" / "reference" / "bpm_tables.json"


def _args(trim: rc.Trim, which: str) -> tuple[np.ndarray, dict]:
    rotor, load = trim._rotor(which)
    axes = rc._basis(load.axis)
    return axes, dict(
        blades=rotor.blades,
        omega=rotor.omega,
        spin=rotor.spin,
        blade=rc._blade(rotor, load),
        speed=trim.airspeed_m_s,
        rho=trim.density,
        c=trim.sound_speed,
    )


def _ours(trim: rc.Trim, which: str, ways: np.ndarray, count: int) -> np.ndarray:
    axes, args = _args(trim, which)
    return rc._harmonics(ways @ axes.T, ways, axes @ np.array([1.0, 0.0, 0.0]), count=count, **args)


def check_gutin() -> dict:
    blades, omega, c, radius, thrust, torque = 3, 41.3, 343.0, 4.28, 22000.0, 9000.0
    theta = np.radians(np.arange(5.0, 180.0, 10.0))
    own = np.column_stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    none = np.zeros((1, 1))
    blade = rc._Blade(
        np.array([radius]),
        np.array([thrust / blades]),
        np.array([torque / (blades * radius)]),
        none,
        none,
        none,
    )
    ours = rc._harmonics(
        own,
        own,
        np.array([1.0, 0.0, 0.0]),
        blades=blades,
        omega=omega,
        spin=1,
        blade=blade,
        speed=0.0,
        rho=1.2,
        c=c,
        count=6,
    )
    m = np.arange(1, 7)[None, :]
    gutin = (
        m
        * blades
        * omega
        / (2.0 * math.sqrt(2.0) * math.pi * c)
        * np.abs(-thrust * np.cos(theta)[:, None] + torque * c / (omega * radius**2))
        * np.abs(jv(m * blades, m * blades * omega * radius * np.sin(theta)[:, None] / c))
    )
    error = float(np.max(np.abs(math.sqrt(2.0) * np.abs(ours) - gutin) / gutin.max(axis=0)))
    print(f"  6 harmonics, 18 directions: within {error:.1e} of each harmonic's loudest")
    return {"error": error}


def _points(trim: rc.Trim, which: str):
    """Each blade point: (force (3,) or None, mass rate per air speed or 0, radius,
    distance ahead along the chord), the force's from the lift and drag."""
    rotor, load = trim._rotor(which)
    blade = rc._blade(rotor, load)
    for i, r in enumerate(blade.radii):
        yield "force", blade.lift[i], blade.drag[i], r, 0.0
        for g in range(blade.areas.shape[1]):
            yield "mass", blade.areas[i, g], 0.0, r, math.tan(blade.angles[i, g]) * r


def check_turn(trim: rc.Trim, count: int = 12) -> list[dict]:
    ways = _sphere(24)
    rows = []
    for which in ("main", "tail"):
        rotor, load = trim._rotor(which)
        axes = rc._basis(load.axis)
        e1, e2, a = axes
        c, v, omega, s = trim.sound_speed, trim.airspeed_m_s, rotor.omega, rotor.spin
        n_t = 4096
        t = np.arange(n_t) / n_t * 2.0 * math.pi / omega
        psi = s * omega * t
        radial = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
        ahead = s * np.cross(a, radial)
        doppler = 1.0 - v * ways[:, 0] / c
        k = rotor.blades * np.arange(1, count + 1)
        total = np.zeros((len(ways), count), dtype=complex)
        for kind, first, second, r, xi in _points(trim, which):
            where = r * radial + xi * ahead
            if kind == "force":
                strength = ((-first * a[None, :] + second * ahead) @ ways.T).T / c
            else:
                strength = np.broadcast_to(
                    trim.density * (omega * r + v * ahead[:, 0]) * first, (len(ways), n_t)
                )
            tau = t[None, :] - (where @ ways.T).T / (c * doppler[:, None])
            total += (
                np.einsum(
                    "nt,nmt->nm", strength, np.exp(-1j * k[None, :, None] * omega * tau[:, None, :])
                )
                / n_t
            )
        exact = rotor.blades * 1j * k * omega / (4.0 * math.pi * doppler[:, None] ** 2) * total
        own = ways @ axes.T
        exact *= np.exp(1j * k * s * np.arctan2(own[:, 1], own[:, 0])[:, None])
        error = np.max(np.abs(_ours(trim, which, ways, count) - exact), axis=0) / np.max(
            np.abs(exact), axis=0
        )
        rows.append({"rotor": which, "worst": float(error.max()), "first": float(error[0])})
        print(
            f"  {which}: {count} harmonics, 24 directions: "
            f"within {error.max():.1e} of each harmonic's loudest"
        )
    return rows


def _kinematics(tau, *, axes, spin, omega, speed, turned, radius, ahead):
    """A blade point's place, velocity and acceleration at times ``tau``, and the
    blade's radial and forward directions: the hub at the origin at time 0 flying
    along x, the point ``radius`` out and ``ahead`` along the chord."""
    e1, e2, a = axes
    psi = spin * omega * tau + turned
    er = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
    et = spin * np.cross(a, er)
    flight = np.array([speed, 0.0, 0.0])
    x = radius * er + ahead * et + tau[:, None] * flight
    u = omega * (radius * et - ahead * er) + flight
    acc = -(omega**2) * (radius * er + ahead * et)
    return x, u, acc, er, et


def check_time(trim: rc.Trim, count: int = 12, far: float = 2.0e7) -> list[dict]:
    """Farassat 1A, blade point by blade point, at an observer ``far`` away: so far
    that the near field's terms and the Fresnel term (the blade's radius squared over
    twice the distance, in the phase: 7.5 x 10⁻⁴ of the main rotor's loudest harmonic
    at 200 km, falling as one over the distance) are left far below the check."""
    directions = []
    for el, az in (
        (-30, 0),
        (-60, 90),
        (-10, 180),
        (-45, 270),
        (-89, 0),
        (20, 30),
        (-5, 0),
        (0, 200),
    ):
        e, z = math.radians(el), math.radians(az)
        directions.append([math.cos(e) * math.cos(z), math.cos(e) * math.sin(z), math.sin(e)])
    directions = np.array(directions)
    rows = []
    for which in ("main", "tail"):
        rotor, load = trim._rotor(which)
        e1, e2, a = rc._basis(load.axis)
        c, v, omega, s, rho = (
            trim.sound_speed,
            trim.airspeed_m_s,
            rotor.omega,
            rotor.spin,
            trim.density,
        )
        ours = _ours(trim, which, directions, count)
        k = rotor.blades * np.arange(1, count + 1)
        worst = 0.0
        for n, d in enumerate(directions):
            hub_times = np.linspace(0.0, 2.0 * math.pi / omega, 1024, endpoint=False)
            observer = far * d
            hub = v * hub_times[:, None] * np.array([1.0, 0.0, 0.0])
            reach = np.linalg.norm(observer - hub, axis=1)
            heard = hub_times + reach / c
            p = np.zeros_like(heard)
            for blade in range(rotor.blades):
                turned = 2.0 * math.pi * blade / rotor.blades
                for kind, first, second, r, xi in _points(trim, which):
                    move = {"axes": (e1, e2, a), "spin": s, "omega": omega, "speed": v}
                    move |= {"turned": turned, "radius": r, "ahead": xi}
                    tau = heard - reach / c
                    for _ in range(20):
                        x, u, _, _, _ = _kinematics(tau, **move)
                        gap = np.linalg.norm(observer - x, axis=1)
                        rhat = (observer - x) / gap[:, None]
                        tau -= (tau + gap / c - heard) / (1.0 - np.einsum("ij,ij->i", rhat, u) / c)
                    x, u, acc, er, et = _kinematics(tau, **move)
                    gap = np.linalg.norm(observer - x, axis=1)
                    rhat = (observer - x) / gap[:, None]
                    mr = np.einsum("ij,ij->i", rhat, u) / c
                    mdot = np.einsum("ij,ij->i", rhat, acc) / c
                    if kind == "force":
                        force = -first * a[None, :] + second * et
                        fdot = -second * omega * er
                        fr = np.einsum("ij,ij->i", force, rhat)
                        p += (
                            np.einsum("ij,ij->i", fdot, rhat) / (c * gap * (1 - mr) ** 2)
                            + fr * mdot / (c * gap * (1 - mr) ** 3)
                        ) / (4.0 * math.pi)
                    else:
                        speed = omega * r + v * et[:, 0]
                        mass, mass_dot = (
                            rho * speed * first,
                            -rho * v * omega * er[:, 0] * first,
                        )
                        p += (
                            mass_dot / (gap * (1 - mr) ** 2) + mass * mdot / (gap * (1 - mr) ** 3)
                        ) / (4.0 * math.pi)
            spectrum = np.fft.fft(reach * p) / len(p)
            exact = spectrum[k]
            own = d @ rc._basis(load.axis).T
            exact = exact * np.exp(1j * k * s * math.atan2(own[1], own[0]))
            worst = max(worst, float(np.max(np.abs(exact - ours[n]) / np.max(np.abs(exact)))))
        rows.append({"rotor": which, "worst": worst})
        print(f"  {which}: {count} harmonics, 8 directions: within {worst:.1e} of the loudest")
    return rows


def check_store(trim: rc.Trim) -> list[dict]:
    rows = []
    rng = np.random.default_rng(3)
    ways = rng.normal(size=(4000, 3))
    ways /= np.linalg.norm(ways, axis=1, keepdims=True)
    start = 400_000
    for which in ("main", "tail"):
        rotor, load = trim._rotor(which)
        for grid in (5.0, 2.5):
            began = time.perf_counter()
            sent = rc.tones(trim, which, samples=start + len(ways), grid_deg=grid)
            took = time.perf_counter() - began
            count = sent.table.shape[-1] // 16
            amplitudes = _ours(trim, which, ways, count)
            own = ways @ sent.axes.T
            k = rotor.blades * np.arange(1, count + 1)
            tau = np.arange(start, start + len(ways)) / RATE + ways @ np.asarray(rotor.hub_m) / (
                trim.sound_speed - trim.airspeed_m_s * ways[:, 0]
            )
            phase = (
                rotor.omega * tau[:, None] - rotor.spin * np.arctan2(own[:, 1], own[:, 0])[:, None]
            )
            exact = np.sum(2.0 * np.real(amplitudes * np.exp(1j * k * phase)), axis=1)
            got = sent.sent(start, start + len(ways), ways)
            error = float(np.sqrt(np.mean((got - exact) ** 2) / np.mean(exact**2)))
            rows.append(
                {
                    "rotor": which,
                    "grid_deg": grid,
                    "error_rms": error,
                    "harmonics": count,
                    "seconds": took,
                    "megabytes": sent.table.nbytes / 1e6,
                }
            )
            print(
                f"  {which}, {grid:g} degrees: {count} harmonics, rms error {error:.1e} "
                f"({20 * math.log10(error):.0f} dB), {sent.table.nbytes / 1e6:.0f} MB, {took:.1f} s"
            )
    return rows


def check_pegg() -> dict:
    main = rc.pegg_level_db(208.0, 344.0, 18.58, 0.438) + 10 * math.log10(
        (math.cos(math.radians(85.0)) ** 2 + 0.1) / 61.6**2
    )
    tail = rc.pegg_level_db(202.0, 344.0, 3.44, 0.182) + 10 * math.log10(
        (math.cos(math.radians(10.0)) ** 2 + 0.1) / 62.2**2
    )
    peaks = rc.pegg_peak_hz(69420.0, 208.0), rc.pegg_peak_hz(5206.5, 202.0)
    print(
        f"  main rotor {main:.2f} + S dB (Pegg: 84.5), tail rotor {tail:.2f} + S dB (82.5); peaks "
        f"{peaks[0]:.0f} Hz (289) and {peaks[1]:.0f} Hz (544)"
    )
    return {"main": main, "tail": tail, "peaks": list(peaks)}


def check_bpm() -> dict:
    """The two sample outputs: one untripped segment (D2) and ten tripped ones with a
    rounded tip (D3), the largest difference of each mechanism's band levels."""
    tables = json.loads(BPM_TABLES.read_text())
    f = np.array(bpm.FREQUENCIES_HZ)
    air = {"viscosity": 1.4529e-5, "sound_speed": 340.46}

    def total(*levels):
        return 10 * np.log10(sum(10 ** (level / 10) for level in levels))

    out = {}
    d2 = np.array(tables["D2"])
    args = (f, 0.3048, 0.4572, 1.22, 90.0, 90.0, 1.516, 71.3)
    p, s, a = bpm.tbl_te(*args, trip=bpm.UNTRIPPED, **air)
    lam = bpm.lbl_vs(*args, trip=bpm.UNTRIPPED, **air)
    names = ("pressure side", "suction side", "angle of attack", "vortex shedding", "total")
    for name, ours, col in zip(
        names, (p, s, a, lam, total(p, s, a, lam)), (1, 2, 3, 4, 7), strict=True
    ):
        out[f"D2 {name}"] = float(np.max(np.abs(ours - d2[:, col])))
    d3 = np.array(tables["D3"])
    n = 10
    args = (f, [0.1524] * n, [0.0305] * n, [1.22] * n, [90.0] * n, [90.0] * n, [5.4] * n)
    parts = bpm.tbl_te(*args, [71.3] * n, trip=bpm.TRIPPED, **air)
    p, s, a = (10 * np.log10(np.sum(10 ** (x / 10), axis=0)) for x in parts)
    for alpha in (7.7, 7.668):
        tip = bpm.tip(f, 0.1524, 1.22, 90.0, 90.0, alpha, 71.3, rounded=True, sound_speed=340.46)
        out[f"D3 tip vortex at {alpha} deg"] = float(np.max(np.abs(tip - d3[:, 6])))
    for name, ours, col in zip(names[:3], (p, s, a), (1, 2, 3), strict=True):
        out[f"D3 {name}"] = float(np.max(np.abs(ours - d3[:, col])))
    for name, off in out.items():
        print(f"  {name:28s} most off {off:.4f} dB")
    return out


def check_pitch() -> list[dict]:
    """The pitch of rigid blades from the hub to the tip against its closed form."""
    tip_loss, rc.TIP_LOSS = rc.TIP_LOSS, 1.0 - 1e-12
    rows = []
    try:
        for speed in (0.0, 30.0, 63.0):
            main = replace(rc.AS350_B3.main, root=0.0)
            trim = rc.trim(replace(rc.AS350_B3, main=main), speed, TEMPERATURE)
            theta0, theta1c, theta1s = rc.pitch(trim, "main", stations=48)
            load, tip = trim.main, main.tip_speed
            mu = speed * math.sqrt(1.0 - load.axis[0] ** 2) / tip
            inflow = (speed * load.axis[0] + load.induced_m_s) / tip
            twist = math.radians(main.twist_deg)
            ct = load.thrust_n / (trim.density * main.disc_m2 * tip**2)
            a = [[(1 + 1.5 * mu**2) / 3, mu / 2], [8 / 3 * mu / (1 + 1.5 * mu**2), 1.0]]
            b = [
                2 * ct / (main.solidity * main.lift_slope) - twist / 4 * (1 + mu**2) + inflow / 2,
                -8 / 3 * mu * 0.75 * (twist - inflow) / (1 + 1.5 * mu**2),
            ]
            exact = np.degrees(np.linalg.solve(a, b))
            row = {
                "speed": speed,
                "theta0": math.degrees(theta0),
                "theta1s": math.degrees(theta1s),
                "theta1c": math.degrees(theta1c),
                "off": float(
                    max(abs(math.degrees(theta0) - exact[0]), abs(math.degrees(theta1s) - exact[1]))
                ),
            }
            rows.append(row)
            print(
                f"  {speed:4.0f} m/s: theta_0 {row['theta0']:.4f} deg, theta_1s "
                f"{row['theta1s']:.4f} deg, theta_1c {row['theta1c']:.1e}; most off "
                f"{row['off']:.1e} deg"
            )
    finally:
        rc.TIP_LOSS = tip_loss
    return rows


def a_weighting_db(f: np.ndarray) -> np.ndarray:
    """IEC 61672's A weighting (dB)."""
    f2 = np.asarray(f, dtype=float) ** 2
    ra = (
        12194.0**2
        * f2**2
        / ((f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    )
    return 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0


def _heard(sent, trim: rc.Trim, seconds: float, image: bool):
    """The pressure at the microphone along the direct path or the ground's image path,
    the path's length and its cosine of incidence at each sample, the helicopter right
    overhead half way."""
    c, v = trim.sound_speed, trim.airspeed_m_s
    t = np.arange(int(seconds * RATE)) / RATE
    observer = np.array([0.0, 0.0, -MICROPHONE if image else MICROPHONE])

    def at(tau):
        return np.stack(
            [v * (tau - seconds / 2), np.zeros_like(tau), np.full_like(tau, HEIGHT)], axis=-1
        )

    tau = t - HEIGHT / c
    for _ in range(60):
        tau = t - np.linalg.norm(observer - at(tau), axis=1) / c
    reach = np.linalg.norm(observer - at(tau), axis=1)
    j = np.arange(max(0, int(tau.min() * RATE) - 64), int(tau.max() * RATE) + 64)
    ways = observer - at(j / RATE)
    ways /= np.linalg.norm(ways, axis=1, keepdims=True)
    samples = sent.sent(int(j[0]), int(j[-1]) + 1, ways)
    return read(samples, tau * RATE - j[0], 0.8) / reach, reach, (HEIGHT + MICROPHONE) / reach


def exposure(sent, trim: rc.Trim, seconds: float = 24.0) -> dict:
    paths = [_heard(sent, trim, seconds, image) for image in (False, True)]
    size, hop = 4096, 2048
    window = np.hanning(size + 1)[:-1]
    f = np.fft.rfftfreq(size, 1.0 / RATE)
    absorption = absorption_db_per_m(np.maximum(f, 1.0), TEMPERATURE, HUMIDITY)
    admittance = 1.0 / surface_impedance(np.maximum(f, 1.0), PRESETS["grass"][0])
    out = np.zeros(len(paths[0][0]))
    for start in range(0, len(out) - size, hop):
        middle = start + size // 2
        total = np.zeros(len(f), dtype=complex)
        for n, (p, reach, cosine) in enumerate(paths):
            part = np.fft.rfft(p[start : start + size] * window) * 10 ** (
                -absorption * reach[middle] / 20
            )
            if n:
                k = 2 * math.pi * f / trim.sound_speed
                part *= spherical_reflection(admittance, cosine[middle], k, reach[middle])
            total += part
        out[start : start + size] += np.fft.irfft(total, size)
    spectrum = np.fft.rfft(out)
    weighted = np.fft.irfft(
        spectrum
        * 10 ** (a_weighting_db(np.maximum(np.fft.rfftfreq(len(out), 1 / RATE), 1e-3)) / 20),
        len(out),
    )
    energy = weighted**2
    step = 1.0 / RATE  # the slow time weighting's 1 s
    slow = lfilter([step], [1.0, -(1.0 - step)], energy)
    level = 10 * np.log10(np.maximum(slow, 1e-30) / 4e-10)
    inside = np.nonzero(level >= level.max() - 10.0)[0]
    sel = 10 * math.log10(np.sum(energy[inside[0] : inside[-1]]) / RATE / 4e-10)
    return {"sel": sel, "max": float(level.max()), "seconds": (inside[-1] - inside[0]) / RATE}


def check_certified(model: str) -> dict:
    """The overflight's exposure with the rotors' broadband noise by ``model``, and
    each part of the sound alone."""
    trim = rc.trim(replace(rc.AS350_B3, broadband=model), SPEED, TEMPERATURE)
    sent = rc.emission(trim, seconds=26.0)
    main, tail = sent.parts[:2]
    rows = {"all": exposure(sent, trim)}
    quiet = main.scaled(0.0)
    samples = int(26.0 * RATE)
    parts = {
        "main rotor tones": Sum((main,)),
        "tail rotor tones": Sum((tail,)),
    }
    for which in ("main", "tail"):
        noise = rc.broadband(trim, samples=samples, sample_rate=RATE, rotors=(which,))
        parts[f"{which} rotor broadband"] = Sum((quiet, noise))
    for name, part in parts.items():
        rows[name] = exposure(part, trim)
    for name, row in rows.items():
        print(
            f"  {name:22s} SEL {row['sel']:5.1f} dB(A), L_ASmax {row['max']:5.1f} dB(A), "
            f"10 dB down for {row['seconds']:.1f} s"
        )
    off = rows["all"]["sel"] - CERTIFIED_SEL
    print(f"  certified: SEL {CERTIFIED_SEL} dB(A); the model {off:+.1f} dB")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    trim = rc.trim(rc.AS350_B3, SPEED, TEMPERATURE)
    print("Gutin's formula, a hovering rotor's thrust and torque on one radius")
    gutin = check_gutin()
    print("\nThe integral round the turn over the blades' points, the AS350 B3 at 63 m/s")
    turn = check_turn(trim)
    print("\nIn time: Farassat's formulation 1A, blade point by blade point, 20,000 km away")
    heard = check_time(trim)
    print("\nThe store: one period in every direction, against the harmonics summed")
    store = check_store(trim)
    print("\nAirfoil self-noise against the sample outputs of its code (tables D2, D3)")
    airfoil = check_bpm()
    print("\nThe blades' pitch against its closed form, the AS350 B3's main rotor")
    pitch = check_pitch()
    print("\nPegg's example I")
    pegg = check_pegg()
    certified = {}
    for model, name in (("bpm", "airfoil self-noise (BPM)"), ("pegg", "Pegg's method")):
        print(f"\nThe AS350 B3's overflight (ICAO Annex 16 chapter 11) over grass, {name}")
        certified[model] = check_certified(model)
    print(f"\n{time.perf_counter() - started:.0f} s")
    if args.json:
        out = {
            "gutin": gutin,
            "turn": turn,
            "time": heard,
            "store": store,
            "airfoil": airfoil,
            "pitch": pitch,
            "pegg": pegg,
            "certified": certified,
        }
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
