"""Check the wave-based solver against theory.

    python validation/wave_solver.py [--json results.json] [--plot wave.png]

Five checks, printed as Markdown tables:

- the modes of a lightly damped 7 x 5 x 3 m box (every wall of specific admittance
  0.005, 10 cm cells), from its ringing: frequency and decay rate of each mode that
  stands apart from the others, against the rigid box's frequencies and the decay rates
  of perturbation theory, delta = c beta / 2 * sum over the three axes of (4 / L if the
  mode varies along that axis, else 2 / L);
- the reflection off the end of a duct at normal incidence, against (1 - beta) / (1 + beta)
  for the end's specific admittance beta;
- the attenuation of a plane wave grazing the walls of a duct, against
  exp(-beta * perimeter / (2 * area) per metre);
- the level of the direct sound in the 63 and 125 Hz bands, in a large and very
  absorbing room, against the free-field level the rays give;
- a 7 x 5 x 3 m box with 15 % absorption on every surface, simulated with the default
  settings (the 63 and 125 Hz bands from the wave solver), against the sum of the box's
  modes with those decay rates: G, EDT and T20 in both bands, also with twice as many
  cells per wavelength and from the rays alone; with ``--plot`` (needs matplotlib), the
  transfer functions from 15 to 250 Hz.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math

import numpy as np

from acoustic_spaces import Room, Scene, _native, simulate
from acoustic_spaces.engine.wave import COURANT, admittance, source_signal
from acoustic_spaces.geometry import box_mesh
from acoustic_spaces.parameters import from_impulse_response

C = 343.0
FS = 48_000
BOX = (7.0, 5.0, 3.0)


def _solver(size, admittances, source, probes, cell):
    """A wave solver in the box from (0, 0, 0) to ``size``, and its sample rate."""
    mesh = box_mesh(size)
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((np.array(size) - origin) / cell).astype(int) + 2
    solver = _native.WaveSolver(
        bvh,
        np.ascontiguousarray(admittances(mesh), dtype=float),
        origin,
        cell,
        *map(int, shape),
        np.asarray(source, dtype=float),
        np.asarray(probes, dtype=float).reshape(-1, 3),
        1.0,
        0,
    )
    return solver, C / (COURANT * cell)


def _run(solver, rate: float, cell: float, seconds: float) -> tuple[np.ndarray, int]:
    signal, delay = source_signal(rate, cell)
    steps = int(seconds * rate)
    drive = np.zeros(steps, np.float32)
    drive[: len(signal)] = signal[:steps]
    return solver.run(drive), delay


def box_modes(size, f_max: float, c: float = C) -> tuple[np.ndarray, np.ndarray]:
    """Mode numbers (modes, 3) and frequencies of a rigid box below ``f_max``."""
    top = [int(2 * f_max * s / c) + 1 for s in size]
    numbers = np.array(list(itertools.product(*(range(t + 1) for t in top))))[1:]
    f = c / 2 * np.sqrt(((numbers / np.array(size)) ** 2).sum(axis=1))
    keep = f < f_max
    return numbers[keep], f[keep]


def modal_decay(numbers: np.ndarray, beta: float, size, c: float = C) -> np.ndarray:
    """Pressure decay rate (1/s) of each mode of a box whose walls all have the small
    specific admittance ``beta`` (first-order perturbation theory)."""
    per_axis = np.where(numbers > 0, 4.0, 2.0) / np.array(size)
    return c * beta / 2 * per_axis.sum(axis=1)


def modes() -> list[dict]:
    beta, cell = 0.005, 0.1
    solver, rate = _solver(
        BOX, lambda m: np.full(len(m.faces), beta), (0.33, 0.27, 0.21), (6.61, 4.77, 2.83), cell
    )
    pressure = _run(solver, rate, cell, 8.0)[0][0]
    n_fft = 1 << int(math.ceil(math.log2(2 * len(pressure))))
    spectrum = np.fft.rfft(pressure, n_fft)
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    t = np.arange(len(pressure)) / rate
    numbers, frequencies = box_modes(BOX, 125.0)
    decay = modal_decay(numbers, beta, BOX)
    rows = []
    for n, f0, d0 in zip(numbers, frequencies, decay, strict=True):
        if np.sort(np.abs(frequencies - f0))[1] < 2.5:
            continue  # too close to another mode to tell them apart
        near = np.abs(f - f0) < 1.0
        peak = float(f[near][np.argmax(np.abs(spectrum[near]))])
        # The mode alone: a narrow band round it, and the slope of its envelope.
        gain = np.exp(-0.5 * ((f - peak) / 0.4) ** 2)
        analytic = np.fft.ifft(np.concatenate([2 * spectrum * gain, np.zeros(n_fft - len(f))]))
        level = 20 * np.log10(np.abs(analytic[: len(pressure)]) + 1e-30)
        fit = (t > 1.0) & (t < 6.0)
        slope = np.polyfit(t[fit], level[fit], 1)[0]
        rows.append(
            {
                "mode": [int(k) for k in n],
                "theory_hz": float(f0),
                "fdtd_hz": peak,
                "theory_decay": float(d0),
                "fdtd_decay": float(-slope * math.log(10) / 20),
            }
        )
    return sorted(rows, key=lambda row: row["theory_hz"])


def duct() -> list[dict]:
    size, cell = (80.0, 0.6, 0.6), 0.1
    rows = []
    for beta in (0.0, 0.1, 0.25, 0.5, 1.0):

        def admittances(mesh, beta=beta):
            x = mesh.vertices[mesh.faces][:, :, 0]
            far = np.all(np.isclose(x, size[0]), axis=1)
            near = np.all(np.isclose(x, 0.0), axis=1)
            return np.where(far, beta, np.where(near, 1.0, 0.0))

        solver, rate = _solver(size, admittances, (10.0, 0.3, 0.3), (30.0, 0.3, 0.3), cell)
        recorded, delay = _run(solver, rate, cell, 0.45)
        pressure = recorded[0]
        n_fft = 1 << 14
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
        pressure = np.fft.irfft(np.fft.rfft(pressure, n_fft) * gain, n_fft)[: len(pressure)]
        t = (np.arange(len(pressure)) - delay) / rate

        def gated(path: float, t=t, pressure=pressure) -> float:
            x = t - path / C
            window = np.where(np.abs(x) < 0.04, 0.5 + 0.5 * np.cos(np.pi * x / 0.04), 0.0)
            return float(np.sum((pressure * window) ** 2))

        rows.append(
            {
                "admittance": beta,
                "theory": (1 - beta) / (1 + beta),
                "fdtd": math.sqrt(gated(120.0) / gated(20.0)),
            }
        )
    return rows


def free_field() -> list[dict]:
    box = (30.0, 24.0, 24.0)
    source = [box[0] / 2 - 4.03, box[1] / 2 + 0.07, box[2] / 2 + 0.02]
    distances = (1.0, 2.0, 4.0, 8.0)
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": list(box)},
                "materials": {"a": {"absorption": [0.95] * 8, "scattering": 0.0}},
                "default_material": "a",
                "sources": [{"name": "s", "position": source}],
                "listeners": [
                    {"name": f"{d:g} m", "position": [source[0] + d, source[1], source[2]]}
                    for d in distances
                ],
                "simulation": {
                    "ir_length_s": 0.3,
                    "seed": 1,
                    "late": {"rays": 2000},
                    "early": {"max_order": 0},
                    "wave": {"enabled": True, "max_band_hz": 125},
                },
            }
        )
    )
    result = simulate(room)
    rows = []
    for (_, name), response in result.responses.items():
        d = float(name.split()[0])
        energy = response.energy()
        t = np.arange(energy.shape[1]) * response.time_step
        direct = energy[:2, t < (d + 6.5) / C].sum(axis=1) * 4 * math.pi * d * d
        rows.append(
            {
                "distance_m": d,
                "db_63": 10 * math.log10(direct[0]),
                "db_125": 10 * math.log10(direct[1]),
            }
        )
    return rows


def grazing() -> list[dict]:
    size, beta = (40.0, 0.6, 0.6), 0.05
    rows = []
    for cell in (0.1, 0.05):

        def admittances(mesh):
            x = mesh.vertices[mesh.faces][:, :, 0]
            ends = np.all(np.isclose(x, 0.0), axis=1) | np.all(np.isclose(x, size[0]), axis=1)
            return np.where(ends, 1.0, beta)

        probes = [(10.0, 0.3, 0.3), (20.0, 0.3, 0.3), (30.0, 0.3, 0.3)]
        solver, rate = _solver(size, admittances, (5.0, 0.3, 0.3), probes, cell)
        recorded, delay = _run(solver, rate, cell, 0.2)
        n_fft = 1 << 15
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
        t = (np.arange(recorded.shape[1]) - delay) / rate
        energies = []
        for k, (x, _, _) in enumerate(probes):
            band = np.fft.irfft(np.fft.rfft(recorded[k], n_fft) * gain, n_fft)[: len(t)]
            u = t - (x - 5.0) / C
            window = np.where(np.abs(u) < 0.03, 0.5 + 0.5 * np.cos(np.pi * u / 0.03), 0.0)
            energies.append(float(np.sum((band * window) ** 2)))
        rows.append(
            {
                "cell_m": cell,
                "theory": math.exp(-beta * 4 * size[1] / size[1] ** 2 / 2 * 10.0),
                "fdtd": [
                    math.sqrt(energies[1] / energies[0]),
                    math.sqrt(energies[2] / energies[1]),
                ],
            }
        )
    return rows


BOX_SOURCE = (1.3, 1.1, 1.4)
BOX_LISTENER = (5.8, 3.9, 1.7)
BOX_ABSORPTION = 0.15
BOX_LENGTH_S = 1.5


def _box_response(wave: bool, points_per_wavelength: float = 8.0):
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": list(BOX)},
                "materials": {"m": {"absorption": [BOX_ABSORPTION] * 8, "scattering": 0.1}},
                "default_material": "m",
                "sources": [{"name": "s", "position": list(BOX_SOURCE)}],
                "listeners": [{"name": "l", "position": list(BOX_LISTENER)}],
                "simulation": {
                    "ir_length_s": BOX_LENGTH_S,
                    "seed": 1,
                    "wave": {
                        "enabled": wave,
                        "points_per_wavelength": points_per_wavelength,
                    },
                },
            }
        )
    )
    result = simulate(room)
    seconds = result.details["s"].get("wave_s", 0.0)
    return result.responses[("s", "l")], seconds, room.speed_of_sound()


def modal_sum(frequencies: np.ndarray, c: float, f_max: float = 1500.0) -> np.ndarray:
    """Transfer function of the box from the sum of its modes up to ``f_max``, each with
    its decay rate for walls of the admittance of BOX_ABSORPTION; the direct sound at r
    metres is 1 / r, as in the simulation (air absorption, negligible here, left out)."""
    beta = float(admittance(np.array([BOX_ABSORPTION]))[0])
    numbers, f_n = box_modes(BOX, f_max, c)
    decay = modal_decay(numbers, beta, BOX, c)
    size = np.array(BOX)

    def shape(point) -> np.ndarray:
        return np.prod(np.cos(np.pi * numbers * np.array(point) / size), axis=1)

    weight = (
        shape(BOX_SOURCE) * shape(BOX_LISTENER) / np.prod(np.where(numbers > 0, 0.5, 1.0), axis=1)
    )
    weight *= 4 * math.pi * c**2 / np.prod(size)
    w = 2 * math.pi * frequencies
    out = np.zeros(len(frequencies), complex)
    w_n = 2 * math.pi * f_n
    for start in range(0, len(w_n), 2000):
        part = slice(start, start + 2000)
        denominator = w_n[part, None] ** 2 - w[None, :] ** 2 + 2j * decay[part, None] * w[None, :]
        out += (weight[part, None] / denominator).sum(axis=0)
    return out


def _band_figures(ir: np.ndarray) -> dict:
    """G, EDT and T20 in the 63 and 125 Hz octaves of an impulse response at FS, by the
    ISO 3382-1 analysis of measured responses."""
    p = from_impulse_response(ir, FS, 0.01, noise=False)
    return {
        f"{name}_{f}": float(getattr(p, name)[b])
        for name in ("g", "edt", "t20")
        for b, f in enumerate((63, 125))
    }


def box() -> tuple[list[dict], dict]:
    wave, seconds, c = _box_response(True)
    fine, fine_seconds, _ = _box_response(True, 16.0)
    rays, _, _ = _box_response(False)
    n = int(BOX_LENGTH_S * FS)
    n_fft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    reach = f <= 300
    theory = np.zeros(len(f), complex)
    theory[reach] = modal_sum(f[reach], c)
    theory *= np.clip((300 - f) / 50, 0, 1)  # no edge for the band filters to ring at
    spectra = {
        "modes": theory,
        "wave solver": np.fft.rfft(wave.impulse_response(FS, seed=1), n_fft),
        "wave solver, 16 per wavelength": np.fft.rfft(fine.impulse_response(FS, seed=1), n_fft),
        "rays only": np.fft.rfft(rays.impulse_response(FS, seed=1), n_fft),
    }
    rows = [
        {"model": name, **_band_figures(np.fft.irfft(x, n_fft)[:n])} for name, x in spectra.items()
    ]
    shown = (f >= 15) & (f <= 250)
    curves = {"f": f[shown], **{name: np.abs(x[shown]) for name, x in spectra.items()}}
    curves.update(seconds=seconds, fine_seconds=fine_seconds, cell=wave.wave.cell)
    return rows, curves


def plot(curves: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    f = curves["f"]
    styles = {
        "modes": {"color": "0.25", "lw": 2.4, "alpha": 0.45, "label": "sum of the modes"},
        "wave solver": {"color": "tab:blue", "lw": 1.1, "label": "wave solver (default grid)"},
        "rays only": {"color": "tab:orange", "lw": 1.0, "ls": ":", "label": "rays only"},
    }
    for name, style in styles.items():
        ax.plot(f, 20 * np.log10(np.maximum(curves[name], 1e-6)), **style)
    ax.axvspan(15, 22.3, color="0.93", zorder=0)
    ax.axvspan(176.8, 250, color="0.93", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([16, 22.3, 31.5, 44.2, 63, 88, 125, 177, 250])
    ax.set_xticklabels(["16", "22", "31.5", "44", "63", "88", "125", "177", "250"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(15, 250)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("level (dB re the direct sound at 1 m)")
    ax.set_title(
        f"{BOX[0]:g} x {BOX[1]:g} x {BOX[2]:g} m box, {BOX_ABSORPTION:.0%} absorption: the "
        "wave solver covers 22 to 177 Hz"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the numbers to this file")
    parser.add_argument("--plot", help="save the box's transfer functions to this image")
    args = parser.parse_args()

    results: dict = {}
    results["modes"] = modes()
    print("| mode | theory (Hz) | wave solver (Hz) | decay, theory (1/s) | wave solver |")
    print("|---|---:|---:|---:|---:|")
    for row in results["modes"]:
        print(
            f"| {tuple(row['mode'])} | {row['theory_hz']:.2f} | {row['fdtd_hz']:.2f} | "
            f"{row['theory_decay']:.3f} | {row['fdtd_decay']:.3f} |"
        )

    results["duct"] = duct()
    print("\n| end admittance | reflection, theory | wave solver |\n|---:|---:|---:|")
    for row in results["duct"]:
        print(f"| {row['admittance']:g} | {row['theory']:.3f} | {row['fdtd']:.3f} |")

    results["grazing"] = grazing()
    print("\n| cell | amplitude after 10 m, theory | wave solver |\n|---:|---:|---:|")
    for row in results["grazing"]:
        found = ", ".join(f"{x:.4f}" for x in row["fdtd"])
        print(f"| {100 * row['cell_m']:g} cm | {row['theory']:.4f} | {found} |")

    results["free_field"] = free_field()
    print("\n| distance | 63 Hz (dB re free field) | 125 Hz |\n|---:|---:|---:|")
    for row in results["free_field"]:
        print(f"| {row['distance_m']:g} m | {row['db_63']:+.2f} | {row['db_125']:+.2f} |")

    rows, curves = box()
    results["box"] = rows
    print("\n| model | G 63 Hz (dB) | 125 Hz | EDT 63 Hz (s) | 125 Hz | T20 63 Hz (s) | 125 Hz |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['model']} | {row['g_63']:.1f} | {row['g_125']:.1f} | {row['edt_63']:.2f} | "
            f"{row['edt_125']:.2f} | {row['t20_63']:.2f} | {row['t20_125']:.2f} |"
        )
    print(
        f"\nwave solver in the box: {curves['seconds']:.1f} s ({100 * curves['cell']:.0f} cm "
        f"cells), {curves['fine_seconds']:.1f} s with 16 per wavelength"
    )
    if args.plot:
        plot(curves, args.plot)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
