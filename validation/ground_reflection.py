"""Check the reflection off a ground with an impedance against the exact solution.

    python validation/ground_reflection.py [--wave] [--parabolic] [--json results.json]

A point source over a locally reacting plane of normalised admittance beta has an exact
solution (Sommerfeld; Chien and Soroka, 1975; in e^(-iwt) and with the image source's
distance R2 and heights zs and zr over the plane): the reflected sound is

    e^(ik R2) / R2 - 2 k beta * integral from 0 to infinity of
        e^(-k beta q) e^(ik R(q)) / R(q) dq,   R(q) = sqrt(r^2 + (zs + zr + iq)^2),

which the script integrates numerically, far enough along q for the integrand to have
died away. The engine reflects with the Weyl-van der Pol factor Q, the asymptotic form
of that integral (see :mod:`acoustic_spaces.materials.ground`). The checks:

- Q against the exact solution: the level of the whole field (direct and reflected)
  over the free field's, at the third-octave centres from 50 Hz to 5 kHz, over snow, a
  layer of snow 10 cm thick on a hard backing, grass, gravel and asphalt (Miki's model,
  Nord2000's flow resistivities), at 5, 20 and 100 m;
- the engine's early part in an outdoor scene, a source and a listener over grass and
  nothing else, against the exact solution: the energy in each octave band, with the
  direct sound and its reflection added in pressure (the engine's default outdoors),
  over the free field's. A band's energy is its average over the band, as the engine
  reads it; the analysis filters' skirts, which reach into the neighbouring bands, fill
  a sharp dip in a band by up to about 1 dB more;
- with ``--wave``, the wave solver: its ground is fitted to the impedance itself (see
  :func:`acoustic_spaces.engine.impedance.fit_impedance`); the level over grass over the
  level over a rigid ground, from 40 to 315 Hz, against the exact solution. About a
  minute and a half;
- with ``--parabolic``, the parabolic equation that gives the sound over the open
  ground in wind and temperature gradients (:mod:`acoustic_spaces.engine.parabolic`),
  here in still air: the level over the same grounds at the third-octave centres from
  63 Hz to 2 kHz, from a source 1 m high at 25, 100 and 400 m and from one 119 m high
  (as a wind turbine's hub) at 178, 535 and 845 m, 1.5 and 1.7 m up.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np

from acoustic_spaces.materials.ground import PRESETS, spherical_reflection, surface_impedance

SPEED_OF_SOUND = 343.0
#: The grounds: name, flow resistivity (kPa s/m²) and the thickness of a layer (m).
GROUNDS = (
    ("snow", PRESETS["snow"][0], None),
    ("snow, 10 cm on ice", PRESETS["snow"][0], 0.1),
    ("grass", PRESETS["grass"][0], None),
    ("gravel", PRESETS["gravel"][0], None),
    ("asphalt", PRESETS["asphalt"][0], None),
)
#: Source to listener: distance along the ground, source height and listener height (m).
GEOMETRIES = ((5.0, 2.0, 1.2), (20.0, 1.0, 1.5), (100.0, 1.0, 1.5))
#: Frequencies the exact solution is averaged over in each octave band, in as many slices.
SLICES = 24
#: Third-octave centres from 50 Hz to 5 kHz.
FREQUENCIES = 1000.0 * 2.0 ** (np.arange(-13, 8) / 3.0)


def exact_reflection(beta: complex, k: float, r: float, zs: float, zr: float) -> complex:
    """The reflected pressure over the image source's, e^(-jkR2) / R2, in the engine's
    convention (e^(jwt)), off a plane of admittance ``beta`` (the engine's convention),
    numerically from the exact integral."""
    b = np.conj(beta)  # e^(-iwt), as the integral is written
    r2 = math.hypot(r, zs + zr)
    decay = k * (b.real + (zs + zr) / r2)
    q = np.linspace(0.0, 40.0 / decay, 2_000_001)
    rq = np.sqrt(r * r + (zs + zr + 1j * q) ** 2)
    rq = np.where(rq.real < 0.0, -rq, rq)
    integrand = np.exp(-k * b * q) * np.exp(1j * k * rq) / rq
    pressure = np.exp(1j * k * r2) / r2 - 2.0 * k * b * np.trapezoid(integrand, q)
    return complex(np.conj(pressure * r2 * np.exp(-1j * k * r2)))


def field(reflection, k, r1, r2):
    """The direct sound and its reflection, over the direct sound's free field."""
    return (np.exp(-1j * k * r1) / r1 + reflection * np.exp(-1j * k * r2) / r2) * r1


def check_q() -> dict:
    """Q against the exact solution: the largest difference in the field's level (dB),
    and where the field is more than 20 dB down (a deep dip), apart."""
    print("Weyl-van der Pol Q against the exact solution: largest difference in the")
    print("field's level over the free field's, 50 Hz to 5 kHz (dB)\n")
    header = "".join(f"{f'{r:g} m':>12}" for r, _, _ in GEOMETRIES)
    print(f"{'':22}{header}")
    out = {}
    for name, sigma, thickness in GROUNDS:
        row = []
        for r, zs, zr in GEOMETRIES:
            r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
            cosine = (zs + zr) / r2
            worst = deep = 0.0
            for f in FREQUENCIES:
                k = 2.0 * math.pi * f / SPEED_OF_SOUND
                beta = 1.0 / complex(surface_impedance(np.array(f), sigma, thickness=thickness))
                q = complex(spherical_reflection(beta, cosine, k, r2))
                exact = exact_reflection(beta, k, r, zs, zr)
                level = 20.0 * math.log10(abs(field(exact, k, r1, r2)))
                miss = abs(20.0 * math.log10(abs(field(q, k, r1, r2))) - level)
                if level < -20.0:
                    deep = max(deep, miss)
                else:
                    worst = max(worst, miss)
            row.append({"worst_db": worst, "in_deep_dips_db": deep})
        out[name] = row
        cells = "".join(
            f"{c['worst_db']:7.2f}"
            + (f" ({c['in_deep_dips_db']:.1f})" if c["in_deep_dips_db"] else "     ")
            for c in row
        )
        print(f"{name:22}{cells}")
    print("\n(in brackets: in dips more than 20 dB deep)\n")
    return out


def check_engine() -> dict:
    """The engine's early part over grass against the exact solution, per octave band."""
    from acoustic_spaces import Room, Scene, simulate
    from acoustic_spaces.engine.coherent import total_energy
    from acoustic_spaces.materials.bands import band_centres

    print("The engine over grass against the exact solution: early part over the free")
    print("field's, per octave band (dB)\n")
    bands = band_centres("octave")
    print(f"{'':28}" + "".join(f"{f:>7g}" for f in bands))
    out = {}
    for r, zs, zr in GEOMETRIES:
        scene = Scene.model_validate(
            {
                "geometry": {"ground": {"extent": [[-10, -10], [r + 10, 10]], "sky": 10}},
                "materials": {"field": {"impedance": "grass", "scattering": 0.0}},
                "default_material": "field",
                "sources": [{"name": "s", "position": [0, 0, zs]}],
                "listeners": [{"name": "l", "position": [r, 0, zr]}],
                "simulation": {
                    "ir_length_s": 0.1,
                    "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
                    "late": {"rays": 200},
                },
            }
        )
        room = Room.from_scene(scene)
        response = simulate(room)[("s", "l")]
        early = response.early
        direct = early.energy[early.order == 0][0]
        engine = 10.0 * np.log10(total_energy(early) / direct)
        # The exact solution's energy averaged over each band, as the engine reads the
        # energy of a band (its slices' average), at the middle of 24 slices of it.
        c = room.speed_of_sound()
        r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
        sigma = PRESETS["grass"][0]
        theory = []
        for centre in bands:
            low = centre / math.sqrt(2.0)
            f = low + low * (np.arange(SLICES) + 0.5) / SLICES
            k = 2.0 * math.pi * f / c
            beta = 1.0 / surface_impedance(f, sigma, speed_of_sound=c)
            exact = np.array(
                [exact_reflection(b, x, r, zs, zr) for b, x in zip(beta, k, strict=True)]
            )
            theory.append(10.0 * math.log10(np.mean(np.abs(field(exact, k, r1, r2)) ** 2)))
        theory = np.array(theory)
        name = f"{r:g} m, heights {zs:g} and {zr:g} m"
        out[name] = {"engine_db": engine.tolist(), "exact_db": theory.tolist()}
        print(f"{name:28}" + "".join(f"{v:7.2f}" for v in theory) + "  exact")
        print(f"{'':28}" + "".join(f"{v:+7.2f}" for v in engine - theory) + "  engine - exact")
    print()
    return out


def check_wave() -> dict:
    """The wave solver over grass over a rigid ground, against the exact solution."""
    from acoustic_spaces import Room, Scene
    from acoustic_spaces.engine.model import TraceModel
    from acoustic_spaces.engine.wave import wave_responses

    r, zs, zr = 10.0, 1.0, 1.5

    def run(material):
        scene = Scene.model_validate(
            {
                "geometry": {"ground": {"extent": [[-14, -10], [24, 10]], "sky": 7}},
                "materials": {"field": material},
                "default_material": "field",
                "sources": [{"name": "s", "position": [0, 0, zs]}],
                "listeners": [{"name": "l", "position": [r, 0, zr]}],
            }
        )
        room = Room.from_scene(scene)
        (response,), _ = wave_responses(
            TraceModel.from_room(room),
            np.array([0.0, 0.0, zs]),
            np.array([[r, 0.0, zr]]),
            ["l"],
            max_band_hz=250,
            length_s=0.12,
            sample_rate=48000.0,
            speed_of_sound=room.speed_of_sound(),
            points_per_wavelength=10.0,
        )
        return response, room.speed_of_sound()

    start = time.time()
    grass, c = run({"impedance": "grass"})
    rigid, _ = run({"absorption": [0.0] * 8, "scattering": 0.0})
    rate = grass.rate
    r1, r2 = math.hypot(r, zr - zs), math.hypot(r, zr + zs)
    # Up to 25 ms after the reflection, before anything off the grid's edges arrives.
    n = int(rate * (r2 / c + 0.025))
    window = np.ones(n)
    taper = int(rate * 0.008)
    window[-taper:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
    size = 1 << 16
    f = np.fft.rfftfreq(size, 1.0 / rate)
    over = np.fft.rfft(grass.signals[:n, 0] * window, size)
    under = np.fft.rfft(rigid.signals[:n, 0] * window, size)
    print(
        f"The wave solver over grass over a rigid ground, {r:g} m, "
        f"heights {zs:g} and {zr:g} m (dB)\n"
    )
    out = {}
    for centre in (40, 63, 80, 100, 125, 160, 200, 250, 315):
        i = int(np.argmin(np.abs(f - centre)))
        k = 2.0 * math.pi * f[i] / c
        beta = complex(
            1.0 / surface_impedance(np.array(f[i]), PRESETS["grass"][0], speed_of_sound=c)
        )
        exact = exact_reflection(beta, k, r, zs, zr)
        theory = 20.0 * math.log10(abs(field(exact, k, r1, r2) / field(1.0, k, r1, r2)))
        wave = 20.0 * math.log10(abs(over[i] / under[i]))
        out[centre] = {"wave_db": wave, "exact_db": theory}
        print(
            f"{centre:5d} Hz  wave {wave:6.2f}  exact {theory:6.2f}  "
            f"difference {wave - theory:+5.2f}"
        )
    print(f"\n({time.time() - start:.0f} s)\n")
    return out


#: The parabolic equation's checks: source and listener heights (m), and the distances
#: along the ground (m).
PE_GEOMETRIES = (((1.0, 1.5), (25.0, 100.0, 400.0)), ((119.0, 1.7), (178.0, 535.0, 845.0)))
#: Third-octave centres from 63 Hz to 2 kHz, the bands it gives outdoors.
PE_FREQUENCIES = 1000.0 * 2.0 ** (np.arange(-12, 4) / 3.0)


def check_parabolic() -> dict:
    """The parabolic equation outdoors, in still air, against the exact solution."""
    from acoustic_spaces.engine.parabolic import march

    print("The parabolic equation against the exact solution: largest difference in")
    print("the field's level over the free field's, 63 Hz to 2 kHz (dB)\n")
    start = time.time()
    still = lambda z: np.full(np.shape(z), SPEED_OF_SOUND)  # noqa: E731
    out = {}
    for (zs, zr), ranges in PE_GEOMETRIES:
        ranges = np.array(ranges)
        print(f"source {zs:g} m, listener {zr:g} m high" + "".join(f"{f'{r:g} m':>10}" for r in ranges))
        top = max(1.5 * zs, zs + 20.0, 0.15 * ranges.max())
        for name, sigma, thickness in GROUNDS:
            worst = np.zeros(len(ranges))
            for f in PE_FREQUENCIES:
                k = 2.0 * math.pi * f / SPEED_OF_SOUND
                z = complex(surface_impedance(np.array(f), sigma, thickness=thickness))
                # The parabolic equation's time goes as exp(-i omega t).
                pressure = march([f], still, lambda r, z=z: np.conj(z), zs, ranges, np.full(len(ranges), zr), top)[0]
                for i, r in enumerate(ranges):
                    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
                    exact = field(exact_reflection(1.0 / z, k, r, zs, zr), k, r1, r2)
                    miss = abs(20.0 * math.log10(abs(pressure[i]) * r1) - 20.0 * math.log10(abs(exact)))
                    worst[i] = max(worst[i], miss)
            out[f"{name}, heights {zs:g} and {zr:g} m"] = dict(zip((f"{r:g} m" for r in ranges), worst.tolist()))
            print(f"{name:34}" + "".join(f"{w:10.2f}" for w in worst))
        print()
    print(f"({time.time() - start:.0f} s)\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wave", action="store_true", help="also check the wave solver")
    parser.add_argument(
        "--parabolic", action="store_true", help="also check the parabolic equation"
    )
    parser.add_argument("--json", help="write the results to this file")
    args = parser.parse_args(argv)
    results = {"q": check_q(), "engine": check_engine()}
    if args.wave:
        results["wave"] = check_wave()
    if args.parabolic:
        results["parabolic"] = check_parabolic()
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
