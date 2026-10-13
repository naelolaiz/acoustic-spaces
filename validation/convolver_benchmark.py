"""Time the real-time convolver block by block, as a sound card callback would run it.

    python validation/convolver_benchmark.py [--seconds 600] [--taps-s 3] [--outputs 16]
                                             [--block 128] [--rate 48000]

The defaults are the plan's target: a 3 s response with 16 output channels (third-order
Ambisonics) fed by one input, in blocks of 128 frames at 48 kHz, for 10 minutes of
audio. Every block goes through :meth:`acoustic_spaces.render.Convolver.process` from
Python, as in :func:`acoustic_spaces.render.play`, and is timed twice: wall-clock time,
which includes the operating system taking the core away, and the thread's CPU time,
which shows the convolver's own cost. A block that takes longer than it lasts would
be a dropout at the sound card. The first seconds of output are checked against
offline convolution.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

from acoustic_spaces.render import Convolver, convolve


def _response(taps: int, outputs: int, rate: int, rng: np.random.Generator) -> np.ndarray:
    """Exponentially decaying noise, 60 dB down after 1.5 s: the cost of a convolver does
    not depend on what the response sounds like, only on its size."""
    t = np.arange(taps) / rate
    return (rng.standard_normal((taps, outputs)) * 10 ** (-3.0 * t / 1.5)[:, None]).astype(
        np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=600.0, help="audio to process")
    parser.add_argument("--taps-s", type=float, default=3.0, help="response length (s)")
    parser.add_argument("--outputs", type=int, default=16)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--rate", type=int, default=48_000)
    parser.add_argument("--check-s", type=float, default=5.0, help="output checked offline")
    args = parser.parse_args()

    rng = np.random.default_rng(1)
    taps = int(round(args.taps_s * args.rate))
    ir = _response(taps, args.outputs, args.rate, rng)
    convolver = Convolver(ir, block=args.block)
    budget_ns = args.block / args.rate * 1e9
    blocks = int(args.seconds * args.rate / args.block)
    checked = int(args.check_s * args.rate / args.block)

    print(f"{platform.machine()}, {platform.system()}, Python {platform.python_version()}")
    print(
        f"response {args.taps_s:g} s x {args.outputs} outputs, block {args.block} at "
        f"{args.rate} Hz ({budget_ns / 1e6:.3f} ms per block), {blocks} blocks"
    )
    segments = ", ".join(f"{n} x {size}" for _, size, n in convolver.segments)
    print(f"partitions: {segments}")

    wall = np.empty(blocks, dtype=np.int64)
    cpu = np.empty(blocks, dtype=np.int64)
    kept_in = np.empty((checked * args.block,), np.float32)
    kept_out = np.empty((checked * args.block, args.outputs), np.float32)
    for k in range(blocks):
        x = rng.standard_normal(args.block).astype(np.float32) * 0.1
        w0, c0 = time.perf_counter_ns(), time.thread_time_ns()
        y = convolver.process(x)
        cpu[k], wall[k] = time.thread_time_ns() - c0, time.perf_counter_ns() - w0
        if k < checked:
            kept_in[k * args.block : (k + 1) * args.block] = x
            kept_out[k * args.block : (k + 1) * args.block] = y

    reference = convolve(kept_in.astype(np.float64), ir.astype(np.float64))[: len(kept_out)]
    error = np.max(np.abs(kept_out - reference)) / np.max(np.abs(reference))

    print()
    print("| per block | mean | median | 99 % | 99.9 % | max | over budget |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, times in (("wall clock", wall), ("thread CPU", cpu)):
        ms = times / 1e6
        share = [np.mean(ms), *np.percentile(ms, [50, 99, 99.9]), np.max(ms)]
        cells = " | ".join(f"{v:.3f} ms" for v in share)
        print(f"| {name} | {cells} | {int(np.sum(times > budget_ns))} of {blocks} |")
    print()
    print(f"load: mean {np.mean(wall) / budget_ns:.1%}, worst block {np.max(wall) / budget_ns:.1%}")
    print(f"largest error of the first {args.check_s:g} s against offline convolution: {error:.1e}")


if __name__ == "__main__":
    main()
