"""Where the simulated IACC_E of one BRAS loudspeaker comes from: its response rendered
whole and in parts, against the measured one.

    python validation/bras_iacc_parts.py path/to/BRAS CR3 LS7 [--materials fitted]
                                         [--rays 50000] [--diffraction first_order]
                                         [--seed 0]

The loudspeaker is simulated at FABIAN as ``bras_binaural.py`` simulates it, and its
response rendered through FABIAN's head-related impulse responses with one seed of the
rendering: whole, and its direct sound, specular reflections, paths over edges and late
part each on their own. For each part the script prints its energy at the two ears in
the first 80 ms against the whole response's, per octave band, and its IACC over that
time. For the whole response, and for the measured one (BRAS's ``CRx_BRIRs.sofa``, the
head straight; needs the ``sofa`` extra, h5py), it prints the IACC of the direct sound
(the first 5 ms from its arrival at the nearer ear) and of the 75 ms after it, and the
energy of the second against the first. Each band is filtered from 0.1 s before the
response, so that its filter can ring in.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "bras"))

from bras_binaural import ROOMS, fabian, loudspeaker_scene  # noqa: E402
from prepare import read_binaural  # noqa: E402

from acoustic_spaces import Room, simulate  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402
from acoustic_spaces.parameters import _iacc, octave_bands, onset  # noqa: E402
from acoustic_spaces.spatial.render import Binaural  # noqa: E402

#: Silence before a response when it is filtered (s).
LEAD_S = 0.1
#: The direct sound's window, and the early part's (s from the onset).
DIRECT_S = 0.005
EARLY_S = 0.08
#: How far the ears' responses may be shifted against each other (s, ISO 3382-1).
LAG_S = 0.001


def windows(binaural: np.ndarray, rate: float, start: int) -> dict[str, tuple]:
    """Energy at both ears and IACC per octave band of ``binaural`` (samples, 2) over the
    direct sound, the 75 ms after it and the first 80 ms, all from ``start``."""
    lead = int(LEAD_S * rate)
    bands = np.stack(
        [
            octave_bands(np.concatenate([np.zeros(lead), binaural[:, ear]]), rate, "iec")
            for ear in (0, 1)
        ]
    )  # (ears, bands, samples)
    direct = lead + start + int(DIRECT_S * rate)
    early = lead + start + int(EARLY_S * rate)
    lag = int(LAG_S * rate)
    out = {}
    for name, part in (
        ("direct", slice(0, direct)),
        ("after", slice(direct, early)),
        ("early", slice(0, early)),
    ):
        energy = (bands[:, :, part] ** 2).sum(axis=(0, 2))
        iacc = np.array(
            [_iacc(bands[0, b, part], bands[1, b, part], lag) for b in range(bands.shape[1])]
        )
        out[name] = (energy, iacc)
    return out


def row(label: str, values, fmt: str) -> str:
    return f"{label:<31}" + "".join(f"{v:{fmt}}" for v in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bras", type=Path, help="folder of the unpacked BRAS archives")
    parser.add_argument("scene", choices=ROOMS)
    parser.add_argument("loudspeaker", help="LS3 to LS7")
    parser.add_argument("--materials", choices=["fitted", "initial"], default="fitted")
    parser.add_argument("--rays", type=int, help="late rays (default: the scene's)")
    parser.add_argument("--diffraction", choices=["none", "direct", "first_order"])
    parser.add_argument("--seed", type=int, default=0, help="seed of the rendering")
    args = parser.parse_args()

    scene, setup = loudspeaker_scene(
        args.scene, args.loudspeaker, args.materials, args.rays, args.diffraction
    )
    rate = int(setup["sample_rate"])
    response = simulate(Room.from_scene(scene))[args.loudspeaker, setup["listener"]["name"]]
    binaural = Binaural(fabian(args.bras))
    early = response.early
    quiet = dataclasses.replace(
        response,
        late=np.zeros_like(response.late),
        late_directional=np.zeros_like(response.late_directional),
    )
    kinds = {
        "direct sound": (early.order == 0) & (early.edge < 0),
        "specular reflections": (early.order > 0) & (early.edge < 0),
        "paths over edges": early.edge >= 0,
    }

    def render(part) -> np.ndarray:
        return part.spatial_impulse_response(binaural, rate, seed=args.seed)

    whole = render(response)
    start = min(onset(whole[:, 0]), onset(whole[:, 1]))
    found = windows(whole, rate, start)
    print(
        f"{scene.name}, {args.loudspeaker} at FABIAN: {args.materials} absorption, "
        f"diffraction {scene.simulation.early.diffraction}, {scene.simulation.late.rays} "
        f"rays per band, rendering seed {args.seed}; {len(early)} early paths "
        f"({', '.join(f'{int(m.sum())} {k}' for k, m in kinds.items())})\n"
    )
    print(row("", (f"{f} Hz" if f < 1000 else f"{f // 1000} kHz" for f in OCTAVE_BANDS_HZ), ">9"))
    print(row("IACC, first 80 ms", found["early"][1], "9.2f"))
    print("\nEach part on its own, first 80 ms: energy against the whole (dB), IACC")
    for label, part in [
        *((k, dataclasses.replace(quiet, early=early[m])) for k, m in kinds.items() if m.any()),
        ("late part", dataclasses.replace(response, early=early[np.zeros(len(early), bool)])),
    ]:
        energy, iacc = windows(render(part), rate, start)["early"]
        share = 10 * np.log10(np.maximum(energy, 1e-30) / found["early"][0])
        print(row(f"{label} (dB)", share, "9.1f"))
        print(row(f"{label} (IACC)", iacc, "9.2f"))

    measured = read_binaural(args.bras, args.scene)
    if measured["rate"] != rate:
        raise SystemExit("the measured responses are at another sample rate")
    emitter = measured["emitter_ids"].index(int(args.loudspeaker.removeprefix("LS")))
    ir = measured["ir"][:, emitter].T
    sides = {
        "simulated": found,
        "measured": windows(ir, rate, min(onset(ir[:, 0]), onset(ir[:, 1]))),
    }
    print("\nThe direct sound (first 5 ms) and the 75 ms after it")
    for side, w in sides.items():
        print(row(f"{side}: IACC, direct", w["direct"][1], "9.2f"))
        print(row(f"{side}: IACC, after", w["after"][1], "9.2f"))
        ratio = 10 * np.log10(w["after"][0] / w["direct"][0])
        print(row(f"{side}: after - direct (dB)", ratio, "9.1f"))


if __name__ == "__main__":
    main()
