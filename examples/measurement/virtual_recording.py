"""Record a sweep in a simulated room: what a microphone would pick up.

Convolves the excitation with an impulse response, delays it as an audio interface
would, adds background noise and writes a two-channel recording: the microphone on
channel 1 and a loopback of the excitation on channel 2. Deconvolving it should give
the impulse response back:

    acoustic-spaces simulate examples/shoebox/scene.yaml --out sim
    acoustic-spaces sweep sweep.wav --duration 4 --silence 3 --repeats 2
    python examples/measurement/virtual_recording.py sweep.wav sim/talker__seat-a.wav rec.wav
    acoustic-spaces deconvolve rec.wav --sweep sweep.json --out ir.wav --loopback 2
    acoustic-spaces analyze ir.wav --reference-energy 0.01
"""

from __future__ import annotations

import argparse

import numpy as np

from acoustic_spaces.audio import read_wav, write_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", help="excitation WAV written by 'acoustic-spaces sweep'")
    parser.add_argument("ir", help="impulse response WAV (first channel used)")
    parser.add_argument("out", help="recording WAV to write")
    parser.add_argument("--latency-ms", type=float, default=5.0, help="interface latency")
    parser.add_argument("--noise-db", type=float, default=-80.0, help="noise level, dBFS rms")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sweep, rate = read_wav(args.sweep)
    ir, ir_rate = read_wav(args.ir)
    if rate != ir_rate:
        parser.error(f"the sweep is at {rate} Hz, the response at {ir_rate} Hz")
    x, h = sweep[:, 0], ir[:, 0]
    n_fft = 1 << (len(x) + len(h) - 1).bit_length()
    room = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)[: len(x)]

    lag = int(round(args.latency_ms * 1e-3 * rate))
    microphone = np.concatenate([np.zeros(lag), room])[: len(x)]
    loopback = np.concatenate([np.zeros(lag), x])[: len(x)]
    rng = np.random.default_rng(args.seed)
    microphone += 10.0 ** (args.noise_db / 20.0) * rng.standard_normal(len(x))
    write_wav(args.out, np.column_stack([microphone, loopback]), rate)
    print(f"wrote {args.out}: microphone on channel 1, loopback on channel 2")


if __name__ == "__main__":
    main()
