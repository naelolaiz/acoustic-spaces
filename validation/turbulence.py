"""Check the engine's turbulence against a parabolic equation through turbulent air.

    python validation/turbulence.py [--realisations N] [--profile] [--json results.json]

The reference is the engine's own parabolic equation (:mod:`acoustic_spaces.engine.
parabolic`, checked against the exact solution in ``ground_reflection.py``) marched
through random realisations of the turbulent air, its sound's energy averaged over them
(Gilbert, Raspet and Di 1990; Salomons 2001, app. I). Each realisation is a field of the
index of refraction's fluctuation mu(x, z) in the vertical plane through the source and
the listener, drawn by FFT with the two-dimensional spectrum

    Phi2(kx, kz) = A Cn² sqrt(pi) Gamma(4/3) / Gamma(11/6) (kx² + kz² + 1 / L²)^(-4/3),

the von Kármán spectrum of :mod:`acoustic_spaces.engine.turbulence` integrated across the
plane, so that the phase a wave gathers along x has the three-dimensional field's
structure function in height. Each step of the march takes the phase k times mu
integrated over the step. The outer scale L is 10 m. Two checks:

- The coherence factor. In still air, over five soft grounds and three hard ones with
  the source and the listener at several heights, 50 to 400 m apart, at 13 frequencies
  from 125 Hz to 2 kHz through the ground dip, with the engine's strong preset: the
  level of the direct sound and its reflection (the Weyl-van der Pol factor Q) added
  with the coherence factor of the engine's :class:`~acoustic_spaces.engine.turbulence.
  Wander`, and with that of the Markov approximation alone (Clifford and Lataitis), over
  the parabolic equation's.
- Scattering into a shadow. On a sunny day and upwind in the windy preset, with the
  moderate and the strong presets, the parabolic equation's level deep in the shadow,
  2.5 to 9 times as far as it starts, without the eddies, plus what
  :meth:`~acoustic_spaces.engine.turbulence.Eddies.scattered` adds, over the parabolic
  equation's through the eddies, from 250 Hz to 2 kHz.

Levels are in dB over the free field. Both checks fitted the engine's constants
(:data:`~acoustic_spaces.engine.turbulence.GROUND_K_RHO` and the shadow's), so they show
how well those fits hold over the cases, not an independent test.

With ``--profile`` the eddies weaken with height as the engine's ``height_m`` and
``falloff`` have it (the presets' strength at 2 m, falling as z^(-2/3)): each
realisation's field is scaled in height by the square root of that, strengthening right
down to the roughness length 5 cm above the ground, while the engine takes Cn² no nearer
the ground than the outer scale (:meth:`~acoustic_spaces.engine.turbulence.Eddies.
strength`). Those fits were made with eddies the same at every height, so this shows how
they carry over.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from acoustic_spaces.engine.parabolic import march
from acoustic_spaces.engine.refraction import Profile
from acoustic_spaces.engine.shadow import shadow_start
from acoustic_spaces.engine.turbulence import SPECTRUM_CONSTANT, Eddies, Wander
from acoustic_spaces.materials.ground import PRESETS, spherical_reflection, surface_impedance
from acoustic_spaces.scene import Atmosphere
from acoustic_spaces.scene.model import TURBULENCE_PRESETS

SPEED_OF_SOUND = 343.0
GROUND_C = 15.0
OUTER_SCALE_M = 10.0
#: The profile of ``--profile``: the strength holds at this height (m) and falls with
#: height by this power; the field keeps it as at the roughness length below that.
PROFILE_HEIGHT_M = 2.0
PROFILE_FALLOFF = 2.0 / 3.0
PROFILE_FLOOR_M = 0.05
#: The field's grid (m): along the ground and in height, fine enough for the angles the
#: parabolic equation takes.
FIELD_DX = 0.5
FIELD_DZ = 0.1
#: Source and listener heights (m) and grounds of the coherence check.
COHERENCE_CASES = [
    (2.0, 1.5, "grass"),
    (5.0, 1.5, "grass"),
    (1.0, 1.0, "loose_soil"),
    (2.0, 1.5, "gravel"),
    (10.0, 2.0, "grass"),
    (10.0, 2.0, "asphalt"),
    (5.0, 1.5, "asphalt"),
    (3.0, 3.0, "compacted_soil"),
]
COHERENCE_RANGES = (50.0, 100.0, 200.0, 400.0)
COHERENCE_FREQUENCIES = np.geomspace(125.0, 2000.0, 13)
#: The shadow check's atmospheres (preset, the way the sound goes along the wind) and
#: its distances, in multiples of where the shadow starts.
SHADOW_CASES = [("sunny_day", None), ("windy", np.array([-1.0, 0.0]))]
SHADOW_SHARES = (1.5, 2.5, 4.0, 6.0, 9.0)
SHADOW_FREQUENCIES = (250.0, 500.0, 1000.0, 2000.0)
#: From this many times as far as the shadow starts, a listener is deep in it.
DEEP = 2.5


def cn2_of(preset: str) -> float:
    values = TURBULENCE_PRESETS[preset]
    t0 = GROUND_C + 273.15
    return (
        values["temperature_structure"] / (4.0 * t0**2)
        + 11.0 / 6.0 * values["wind_structure"] / SPEED_OF_SOUND**2
    )


def make_eddies(cn2: float, c: float, profile: bool) -> Eddies:
    """The engine's eddies, the same at every height or weakening with it."""
    if not profile:
        return Eddies(cn2, OUTER_SCALE_M, c)
    return Eddies(cn2, OUTER_SCALE_M, c, height_m=PROFILE_HEIGHT_M, falloff=PROFILE_FALLOFF)


def field_strength(z: np.ndarray) -> np.ndarray:
    """Cn² over its value at :data:`PROFILE_HEIGHT_M` in the field of ``--profile``:
    falling as z^-:data:`PROFILE_FALLOFF`, as at :data:`PROFILE_FLOOR_M` below it."""
    return (np.maximum(z, PROFILE_FLOOR_M) / PROFILE_HEIGHT_M) ** -PROFILE_FALLOFF


def realisation(
    length: float, height: float, cn2: float, scale: float, rng, eddies: Eddies | None = None
) -> np.ndarray:
    """(nx, nz) a field mu on the grid FIELD_DX x FIELD_DZ from 0 to ``length`` and
    ``height`` (m), periodic over twice each (so that its ends do not meet); scaled in
    height by the square root of :func:`field_strength` when ``eddies`` weaken with
    height."""
    nx = 1 << int(math.ceil(math.log2(2.0 * length / FIELD_DX)))
    nz = 1 << int(math.ceil(math.log2(2.0 * height / FIELD_DZ)))
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, FIELD_DX)
    kz = 2.0 * math.pi * np.fft.fftfreq(nz, FIELD_DZ)
    constant = SPECTRUM_CONSTANT * cn2 * math.sqrt(math.pi) * math.gamma(4 / 3) / math.gamma(11 / 6)
    spectrum = constant * (kx[:, None] ** 2 + kz[None, :] ** 2 + 1.0 / scale**2) ** (-4.0 / 3.0)
    amplitude = np.sqrt(spectrum * (2.0 * math.pi) ** 2 / (nx * FIELD_DX * nz * FIELD_DZ))
    noise = rng.standard_normal((nx, nz)) + 1j * rng.standard_normal((nx, nz))
    mu = np.fft.ifft2(amplitude * noise).real * nx * nz
    mu = mu[: int(length / FIELD_DX) + 2, : int(height / FIELD_DZ) + 2]
    if eddies is not None and eddies.height_m is not None:
        mu = mu * np.sqrt(field_strength(np.arange(mu.shape[1]) * FIELD_DZ))[None, :]
    return mu


def screens(mu: np.ndarray):
    """The ``screen(r, dr, z)`` of :func:`march` through the field ``mu``: mu integrated
    over the step, at heights z (zero above the field)."""
    heights = np.arange(mu.shape[1]) * FIELD_DZ
    cumulative = np.concatenate([np.zeros((1, mu.shape[1])), np.cumsum(mu, axis=0)]) * FIELD_DX

    def screen(r: float, dr: float, z: np.ndarray) -> np.ndarray:
        a, b = r / FIELD_DX, (r + dr) / FIELD_DX
        ia, ib = int(a), int(b)
        row = cumulative[min(ib, len(cumulative) - 1)] - cumulative[min(ia, len(cumulative) - 1)]
        return np.interp(z, heights, row, right=0.0)

    return screen


def two_paths(f, sigma, zs, zr, r):
    """The direct sound and its reflection off a ground of flow resistivity ``sigma``
    (kPa s/m²), each over the free field's pressure at r1, at frequencies ``f``."""
    f = np.asarray(f, dtype=float)
    k = 2.0 * math.pi * f / SPEED_OF_SOUND
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    beta = 1.0 / surface_impedance(f, sigma)
    q = spherical_reflection(beta, (zs + zr) / r2, k, r2)
    direct = np.ones_like(k, dtype=complex)
    reflected = q * np.exp(-1j * k * (r2 - r1)) * r1 / r2
    return direct, reflected


def _impedance(f: float, sigma: float, c: float = SPEED_OF_SOUND):
    z = complex(surface_impedance(np.array(f), sigma, speed_of_sound=c))
    return lambda r: np.conj(z)


def _db(x) -> float:
    return 10.0 * math.log10(max(float(x), 1e-30))


def check_coherence(realisations: int, seed: int = 21, profile: bool = False) -> dict:
    """The coherence check: per case, the mean and largest |difference| (dB) of the
    engine's and of the Markov approximation's level from the parabolic equation's."""
    preset = "strong"
    cn2 = cn2_of(preset)
    eddies = make_eddies(cn2, SPEED_OF_SOUND, profile)
    ranges = np.array(COHERENCE_RANGES)
    still = lambda z: np.full(np.shape(z), SPEED_OF_SOUND)  # noqa: E731
    out = {}
    how = f", weakening with height from {PROFILE_HEIGHT_M:g} m up" if profile else ""
    print(f"coherence ({preset}, Cn² {cn2:.3g} m^-2/3{how}, {realisations} realisations)")
    for zs, zr, ground in COHERENCE_CASES:
        sigma = PRESETS[ground][0]
        top = max(40.0, 3.0 * max(zs, zr) + 20.0)
        rng = np.random.default_rng(seed)
        fields = [
            realisation(ranges.max() + 10.0, top + 5.0, cn2, OUTER_SCALE_M, rng, eddies)
            for _ in range(realisations)
        ]
        engine, markov = [], []
        for f in COHERENCE_FREQUENCIES:
            impedance = _impedance(f, sigma)
            energy = np.zeros(len(ranges))
            for mu in fields:
                p = march(
                    [f], still, impedance, zs, ranges, np.full(len(ranges), zr), top, screens(mu)
                )
                energy += np.abs(p[0]) ** 2
            energy /= len(fields)
            for i, r in enumerate(ranges):
                r1 = math.hypot(r, zs - zr)
                direct, reflected = (complex(x) for x in two_paths(f, sigma, zs, zr, r))
                points = [
                    np.array([[0.0, 0.0, zs], [r, 0.0, zr]]),
                    np.array([[0.0, 0.0, zs], [r * zs / (zs + zr), 0.0, 0.0], [r, 0.0, zr]]),
                ]
                variance, apart = eddies._apart(points)
                ground_pair = np.where(np.eye(2, dtype=bool), 0.0, apart)
                wander = Wander(variance, ground_pair, SPEED_OF_SOUND)
                apart_energy = abs(direct) ** 2 + abs(reflected) ** 2
                cross = 2.0 * (direct * reflected.conjugate()).real
                reference = _db(energy[i] * r1**2)
                together = float(wander.coherence(f)[0, 1, 0])
                plain = math.exp(-0.5 * (2.0 * math.pi * f) ** 2 * variance[0, 1])
                engine.append(_db(apart_energy + together * cross) - reference)
                markov.append(_db(apart_energy + plain * cross) - reference)
        engine, markov = np.abs(engine), np.abs(markov)
        name = f"{ground} {zs:g}/{zr:g} m"
        out[name] = {
            "engine_mean_db": float(engine.mean()),
            "engine_max_db": float(engine.max()),
            "markov_mean_db": float(markov.mean()),
            "markov_max_db": float(markov.max()),
        }
        print(
            f"  {name:24s} engine {engine.mean():4.2f} dB (largest {engine.max():4.1f}), "
            f"Markov alone {markov.mean():4.2f} dB (largest {markov.max():4.1f})",
            flush=True,
        )
    return out


def check_shadow(realisations: int, seed: int = 31, profile: bool = False) -> dict:
    """The shadow check: per atmosphere and preset, the levels deep in the shadow (dB
    over the free field) of the parabolic equation without and with the eddies, and of
    the engine."""
    zs, zr = 2.0, 1.5
    sigma = PRESETS["grass"][0]
    top = 120.0
    out = {}
    how = f", weakening with height from {PROFILE_HEIGHT_M:g} m up" if profile else ""
    print(f"shadows ({realisations} realisations{how})")
    for name, towards in SHADOW_CASES:
        atmosphere = Atmosphere.model_validate({"preset": name})
        weather = Profile.of(atmosphere, GROUND_C, towards)
        start = shadow_start(weather, zs, zr)
        ranges = np.array([share * start for share in SHADOW_SHARES])
        ranges = ranges[ranges <= 1000.0]
        c0 = float(weather.c[0])
        plain = {
            f: march(
                [f],
                weather.speed,
                _impedance(f, sigma, c0),
                zs,
                ranges,
                np.full(len(ranges), zr),
                top,
            )[0]
            for f in SHADOW_FREQUENCIES
        }
        for preset in ("moderate", "strong"):
            cn2 = cn2_of(preset)
            eddies = make_eddies(cn2, c0, profile)
            rng = np.random.default_rng(seed)
            energy = {f: np.zeros(len(ranges)) for f in SHADOW_FREQUENCIES}
            for _ in range(realisations):
                mu = realisation(ranges.max() + 10.0, top + 5.0, cn2, OUTER_SCALE_M, rng, eddies)
                for f in SHADOW_FREQUENCIES:
                    p = march(
                        [f],
                        weather.speed,
                        _impedance(f, sigma, c0),
                        zs,
                        ranges,
                        np.full(len(ranges), zr),
                        top,
                        screens(mu),
                    )[0]
                    energy[f] += np.abs(p) ** 2
            rows = []
            for f in SHADOW_FREQUENCIES:
                for i, r in enumerate(ranges):
                    if r < DEEP * start:
                        continue
                    d2 = r**2 + (zs - zr) ** 2
                    still = abs(plain[f][i]) ** 2 * d2
                    added = float(eddies.scattered(weather, zs, zr, r, [f], start)[0])
                    rows.append(
                        {
                            "f": f,
                            "range_m": float(r),
                            "still_db": _db(still),
                            "eddies_db": _db(energy[f][i] / realisations * d2),
                            "engine_db": _db(still + added),
                        }
                    )
            error = np.array([abs(row["engine_db"] - row["eddies_db"]) for row in rows])
            out[f"{name} {preset}"] = {"start_m": start, "rows": rows}
            print(
                f"  {name:9s} {preset:8s} shadow from {start:5.1f} m: engine within "
                f"{error.mean():4.1f} dB on average (largest {error.max():4.1f})",
                flush=True,
            )
            for row in rows:
                print(
                    f"    {row['f']:5.0f} Hz {row['range_m']:6.0f} m: still {row['still_db']:6.1f}"
                    f"  eddies {row['eddies_db']:6.1f}  engine {row['engine_db']:6.1f}"
                )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--realisations", type=int, default=16)
    parser.add_argument("--only", choices=["coherence", "shadow"], help="run one check")
    parser.add_argument(
        "--profile",
        action="store_true",
        help=f"eddies weakening with height from {PROFILE_HEIGHT_M:g} m up",
    )
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    start = time.time()
    results = {"profile": bool(args.profile)}
    if args.only in (None, "coherence"):
        results["coherence"] = check_coherence(args.realisations, profile=args.profile)
    if args.only in (None, "shadow"):
        results["shadow"] = check_shadow(args.realisations, profile=args.profile)
    print(f"({time.time() - start:.0f} s)")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
