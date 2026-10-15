"""Measurements of the seminar room as built, made up by simulating it.

There is no real room behind this example, so its measured impulse responses are
simulated, of the room as built: the absorbing panels the design (``scene.yaml``) has on
the back wall were left out, so the wall is plain plasterboard. Each response has a
random sequence of its own, 5 ms of latency and background noise 80 dB under its peak,
as a measurement would have. Writes measured/dodecahedron__seat_a.wav and the others,
for ``compare`` and ``calibrate`` (see ``scene.yaml``):

    python make_measurements.py [--out measured]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, simulate
from acoustic_spaces.audio import write_wav
from acoustic_spaces.scene import Material

HERE = Path(__file__).resolve().parent
#: What was built instead of what was designed.
AS_BUILT = {"panels": Material(preset="plasterboard", scattering=0.1)}
SAMPLE_RATE = 48_000
LATENCY_S = 0.005
NOISE_DB = -80.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE / "measured"), help="folder to write to")
    args = parser.parse_args(argv)
    scene = Room.from_yaml(HERE / "scene.yaml").scene
    built = scene.model_copy(
        update={
            "materials": {**scene.materials, **AS_BUILT},
            # Another random sequence than the model's, as a real room would have.
            "simulation": scene.simulation.model_copy(update={"seed": 2026}),
        }
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    noise = np.random.default_rng(7)
    for response in simulate(Room.from_scene(built)):
        ir = response.impulse_response(SAMPLE_RATE, seed=2026)
        ir = np.concatenate([np.zeros(int(LATENCY_S * SAMPLE_RATE)), ir])
        peak = float(np.abs(ir).max())
        ir = ir + peak * 10.0 ** (NOISE_DB / 20.0) * noise.standard_normal(len(ir))
        path = out / f"{response.source}__{response.listener}.wav"
        write_wav(path, ir[:, None], SAMPLE_RATE)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
