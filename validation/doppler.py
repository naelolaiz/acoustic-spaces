"""Check walks with Doppler against the exact field of a moving source and a moving
listener, and the pass-by levels against the sound they render.

    python validation/doppler.py [--json results.json]

A walk with ``doppler: true`` plays each early path through a delay that changes as the
walker goes (:mod:`acoustic_spaces.doppler`). Three checks:

- **Over a hard ground.** A source passes a listener 10 m from its line, 1 m up, the
  listener 1.5 m up, over a rigid ground, at 30 m/s (a car) and 80 m/s (a train), and a
  listener passes a source the same way at 30 m/s. The field is exact here: the direct
  sound and the image source's, each heard at its retarded time (for a moving source,
  the time t_e it left at, t = t_e + R(t_e) / c; for a moving listener, t - R(t) / c),
  over the distance at that time, with the air's attenuation, for a source that keeps
  its sound as it moves (quasi-static, as the engine's). Pure tones of 250 Hz,
  1 kHz and 4 kHz through the engine are compared with it sample by sample (the
  waveform's error relative to the field) and in 20 ms windows (the level's error).
  The same walks rendered without Doppler, crossfading from place to place, show what
  it changes.
- **The places.** The car at 1 kHz with places 0.25 to 4 m apart: how far apart they
  may be.
- **The pass-by levels.** White noise through the car's walk over grass (Miki's
  impedance, 200 kPa s/m²): the energy of the sound in each octave band against that of
  the noise through each place's own impulse response for as long as its sound is heard
  (Doppler moves the sound in time and pitch, but keeps its energy), and against the
  sound exposure level :func:`~acoustic_spaces.walkthrough.pass_by` gives from the
  places' band levels; the whole sound, and its early part alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acoustic_spaces import Room, Scene, synthesis  # noqa: E402
from acoustic_spaces.air import energy_attenuation_per_m, speed_of_sound  # noqa: E402
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ  # noqa: E402
from acoustic_spaces.spatial import Mono  # noqa: E402
from acoustic_spaces.walkthrough import pass_by, render_walk, simulate_walk  # noqa: E402

RATE = 48_000
#: The line the walker goes along (x, m), how far from it the other stands (m), and the
#: heights of the source and the listener (m).
X0, X1, D = -60.0, 60.0, 10.0
ZS, ZR = 1.0, 1.5
#: Distance between the places (m).
STEP_M = 1.0
TONES = (250.0, 1000.0, 4000.0)
#: The cases: who walks, at what speed (m/s), over what.
CASES = (
    ("car, 30 m/s", "source", 30.0, "hard"),
    ("train, 80 m/s", "source", 80.0, "hard"),
    ("passing listener, 30 m/s", "listener", 30.0, "hard"),
)
#: Windows of the level's error (s), and the part of the walk compared: from this much
#: after it starts to this much before it ends (s), clear of the sound's ends.
WINDOW_S = 0.02
MARGIN_S = (0.3, 0.1)
#: White noises through the walk for its energy, and where they stop (Hz).
NOISES = 16
NOISE_TOP_HZ = 18_000.0


def scene(walker: str, speed: float, ground: str, doppler: bool = True) -> Scene:
    walk = {"to": [[X1, 0.0, ZS if walker == "source" else ZR]], "speed_m_s": speed}
    walk |= {"doppler": doppler, "face_forward": False}
    source = {"name": "s", "position": [0.0, D, ZS]}
    listener = {"name": "l", "position": [0.0, D, ZR]}
    if walker == "source":
        source = {"name": "s", "position": [X0, 0.0, ZS], "walk": walk}
    else:
        listener = {"name": "l", "position": [X0, 0.0, ZR], "walk": walk}
    material = {"absorption": [0.0] * 8, "scattering": 0.0}
    if ground == "grass":
        material = {"impedance": {"preset": "grass"}}
    return Scene.model_validate(
        {
            "name": "pass-by",
            "geometry": {"ground": {"extent": [[-100, -40], [100, 40]], "sky": 30}},
            "materials": {"ground": material},
            "default_material": "ground",
            "sources": [source],
            "listeners": [listener],
            "simulation": {
                "seed": 1,
                "ir_length_s": 0.5,
                "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
                "late": {"rays": 2000},
            },
        }
    )


def exact(walker, speed, tone, n) -> np.ndarray:
    """The direct sound and the image source's of a tone ``tone`` (Hz) at each sample,
    each at its retarded time, as the walk has them (a walking source stands at the
    end once there, a walking listener too)."""
    c = speed_of_sound(20.0)
    band = int(np.argmin(np.abs(np.log2(np.array(OCTAVE_BANDS_HZ) / tone))))
    m = energy_attenuation_per_m()[band]
    t = np.arange(n) / RATE
    walked = (X1 - X0) / speed
    played = walked + 0.5
    out = np.zeros(n)
    for image in (False, True):
        zs = -ZS if image else ZS
        b2 = D**2 + (zs - ZR) ** 2
        if walker == "source":
            # c (t - t_e) = |x(t_e) - listener|: the earlier root of a quadratic.
            a2 = c**2 - speed**2
            b = c**2 * t + X0 * speed
            cc = c**2 * t**2 - X0**2 - b2
            te = (b - np.sqrt(b**2 - a2 * cc)) / a2
            done = te > walked
            te[done] = t[done] - math.sqrt(X1**2 + b2) / c
            x = X0 + speed * np.clip(te, 0.0, walked)
        else:
            x = X0 + speed * np.clip(t, 0.0, walked)
            te = t - np.sqrt(x**2 + b2) / c
        r = np.sqrt(x**2 + b2)
        heard = (te >= 0.0) & (te < played)
        level = np.exp(-m * r / 2.0) / r
        out += np.where(heard, level * np.sin(2.0 * np.pi * tone * te), 0.0)
    return out


def errors(wet: np.ndarray, field: np.ndarray, walked: float) -> dict:
    lo, hi = int(MARGIN_S[0] * RATE), int((walked - MARGIN_S[1]) * RATE)
    a, b = wet[lo:hi], field[lo:hi]
    window = int(WINDOW_S * RATE)
    levels = []
    for start in range(0, len(a) - window, window):
        pa, pb = np.sum(a[start : start + window] ** 2), np.sum(b[start : start + window] ** 2)
        levels.append(10.0 * math.log10(pa / pb))
    levels = np.abs(np.array(levels))
    return {
        "waveform_db": float(10.0 * np.log10(np.sum((a - b) ** 2) / np.sum(b**2))),
        "level_mean_db": float(levels.mean()),
        "level_max_db": float(levels.max()),
    }


def run(walker, speed, ground, doppler, signals, step_m=STEP_M) -> tuple:
    room = Room.from_scene(scene(walker, speed, ground, doppler))
    walk = simulate_walk(room, step_m=step_m, late_step_m=None)
    name = "s"
    return walk, [render_walk(walk, {name: s}, Mono())[:, 0] for s in signals]


def check_field(cases, ground: str) -> list[dict]:
    rows = []
    for label, walker, speed, _ in cases:
        walked = (X1 - X0) / speed
        n_dry = int((walked + 0.5) * RATE)
        t = np.arange(n_dry) / RATE
        signals = [np.sin(2.0 * np.pi * tone * t) for tone in TONES]
        _, with_doppler = run(walker, speed, ground, True, signals)
        _, without = run(walker, speed, ground, False, signals)
        for tone, wet, faded in zip(TONES, with_doppler, without, strict=True):
            field = exact(walker, speed, tone, len(wet))
            row = {"case": label, "ground": ground, "tone_hz": tone}
            row["doppler"] = errors(wet, field, walked)
            row["crossfades"] = errors(faded, field, walked)
            rows.append(row)
            d, x = row["doppler"], row["crossfades"]
            print(
                f"{label:26s} {ground:6s} {tone:5.0f} Hz   waveform {d['waveform_db']:6.1f} dB, "
                f"level {d['level_mean_db']:.3f} / {d['level_max_db']:.3f} dB   without "
                f"Doppler: {x['waveform_db']:5.1f} dB, {x['level_mean_db']:.2f} / "
                f"{x['level_max_db']:.2f} dB",
                flush=True,
            )
    return rows


def check_steps() -> list[dict]:
    """The car at 1 kHz with its places further and further apart."""
    speed, tone = 30.0, 1000.0
    walked = (X1 - X0) / speed
    t = np.arange(int((walked + 0.5) * RATE)) / RATE
    rows = []
    for step in (0.25, 1.0, 2.0, 4.0):
        _, (wet,) = run("source", speed, "hard", True, [np.sin(2.0 * np.pi * tone * t)], step)
        row = {"step_m": step, **errors(wet, exact("source", speed, tone, len(wet)), walked)}
        rows.append(row)
        print(
            f"places {step:4.2f} m apart   waveform {row['waveform_db']:6.1f} dB, level "
            f"{row['level_mean_db']:.3f} / {row['level_max_db']:.3f} dB",
            flush=True,
        )
    return rows


def _energies(walk, passing, walked: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """The energy per band, re the source's, of white noise through ``walk``, and of the
    noise through each place's own impulse response for its share of the time."""
    # The sound's energy in each band over the walk heard (from the first place's sound
    # to the last's), over the noise's power in the band while it walked: the energy re
    # the source's, averaged over NOISES noises. They stop at NOISE_TOP_HZ, so that
    # raised in pitch they stay below the cutoff of the delays' reading.
    lo, hi = (int(round(t * RATE)) for t in (passing.times[0], passing.times[-1]))
    n = int((walked + 0.5) * RATE)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    energy = np.zeros(len(OCTAVE_BANDS_HZ))
    power = np.zeros(len(OCTAVE_BANDS_HZ))
    for _ in range(NOISES):
        noise = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * (f < NOISE_TOP_HZ), n)
        wet = render_walk(walk, {"s": noise}, Mono())[:, 0]
        power += np.mean(synthesis.split_bands(noise, RATE)[:, : int(walked * RATE)] ** 2, axis=1)
        energy += np.sum(synthesis.split_bands(wet, RATE)[:, lo:hi] ** 2, axis=1) / RATE
    rendered = 10.0 * np.log10(energy / (synthesis.ENERGY_TO_IR * power))
    # Each place's impulse response, heard for its share of the time: in each band its
    # energy over a unit impulse's there. Their late parts are drawn from the walk's one
    # random sequence, as rendered.
    unit = np.zeros(1 << 16)
    unit[len(unit) // 2] = 1.0
    unit = np.fft.irfft(np.fft.rfft(unit) * (np.fft.rfftfreq(len(unit), 1.0 / RATE) < NOISE_TOP_HZ))
    unit_bands = np.sum(synthesis.split_bands(unit, RATE) ** 2, axis=1)
    shares = np.gradient(passing.times)
    shares[0] /= 2.0
    shares[-1] /= 2.0
    static = np.zeros(len(OCTAVE_BANDS_HZ))
    for name, share in zip(walk.names, shares, strict=True):
        ir = walk.result[name, "l"].impulse_response(RATE, walk.result.settings.seed)
        ir = np.fft.irfft(np.fft.rfft(ir) * (np.fft.rfftfreq(len(ir), 1.0 / RATE) < NOISE_TOP_HZ))
        static += share * np.sum(synthesis.split_bands(ir, RATE) ** 2, axis=1) / unit_bands
    return rendered, 10.0 * np.log10(static / synthesis.ENERGY_TO_IR)


def check_levels(speed: float = 30.0) -> list[dict]:
    """The energy of white noise through the car's walk over grass, per octave band,
    against the places' own impulse responses and the pass-by's sound exposure level:
    the whole sound, and its early part alone (the late part is crossfaded from place
    to place as without Doppler)."""
    walked = (X1 - X0) / speed
    room = Room.from_scene(scene("source", speed, "grass"))
    walk = simulate_walk(room, step_m=STEP_M, late_step_m=None)
    passing = pass_by(walk)
    rendered, static = _energies(walk, passing, walked, np.random.default_rng(1))
    # The early part alone: the late part's random sequence is not the same sound from
    # place to place, so its correlation with the early sound, which the impulse
    # responses each have their own of, is smeared as the delays move.
    responses = walk.result.responses
    for key, r in list(responses.items()):
        silent = np.zeros_like(r.late)
        responses[key] = dataclasses.replace(
            r, late=silent, late_directional=np.zeros_like(r.late_directional)
        )
    early, early_static = _energies(walk, passing, walked, np.random.default_rng(1))
    level = walk.result.room.scene.sources[0].power_db("octave")
    reported = passing.exposure - level
    rows = []
    print("          whole sound                     early part alone")
    print("band      rendered  responses  pass-by    rendered  responses")
    for f, a, b, c, d, e in zip(
        OCTAVE_BANDS_HZ, rendered, static, reported, early, early_static, strict=True
    ):
        rows.append(
            {
                "band_hz": f,
                "rendered_db": float(a),
                "responses_db": float(b),
                "pass_by_db": float(c),
                "early_rendered_db": float(d),
                "early_responses_db": float(e),
            }
        )
        print(
            f"{f:5d} Hz  {a:7.2f}  {b:7.2f} ({a - b:+.2f})  {c:7.2f} ({a - c:+.2f})"
            f"   {d:7.2f}  {e:7.2f} ({d - e:+.2f})"
        )
    print(
        f"L_Amax {passing.max_a:.1f} dB(A) at {passing.max_time:.2f} s, SEL "
        f"{passing.exposure_a:.1f} dB(A), L_Aeq {passing.equivalent_a:.1f} dB(A) over "
        f"{passing.duration_s:.1f} s (94 dB of sound power)"
    )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    print("The engine against the exact field: the waveform's error relative to the field,")
    print("and the level's error in 20 ms windows (mean / max)\n")
    rows = check_field(CASES, "hard")
    worst = max(r["doppler"]["waveform_db"] for r in rows)
    level = max(r["doppler"]["level_max_db"] for r in rows)
    print(f"\nwith Doppler: the waveform within {worst:.1f} dB, the level within {level:.2f} dB")
    print("\nThe car at 1 kHz, its places further apart\n")
    steps = check_steps()
    print("\nWhite noise through the car's walk over grass: its energy per band re the source's")
    print("(dB), as rendered, through the places' impulse responses and from the pass-by's")
    print("sound exposure level, and the rendered sound's difference from each\n")
    levels = check_levels()
    seconds = time.perf_counter() - started
    print(f"\n{seconds:.0f} s")
    if args.json:
        out = {"field": rows, "steps": steps, "levels": levels}
        args.json.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
