"""Make the BRAS scenes and measured parameters in this folder from the BRAS archives.

    python validation/bras/prepare.py path/to/BRAS [--scenes CR2 CR3]

``path/to/BRAS`` is the folder the BRAS archives were unpacked into (the scene archives
``1_scene_descriptions-CRx.zip``, ``3_surface_descriptions.zip`` and
``4_additional_data.zip`` from https://depositonce.tu-berlin.de/handle/11303/7506.3).
Needs the ``sofa`` extra (h5py). For each scene this writes, next to the room model that
``convert_skp.py`` made:

- ``materials/fitted/*.csv`` and ``materials/initial/*.csv``: the BRAS surface data as
  published, absorption fitted to the measured reverberation time and the initial
  estimates;
- ``scene.yaml``: the room, its materials (the fitted ones), the air as measured, and
  the two loudspeaker and five microphone positions of the dodecahedron measurements;
- ``measured.json``: the ISO 3382-1 parameters of each of the ten measured responses,
  analysed by this package as it analyses simulated ones, and the spatial averages
  BRAS publishes (third octaves; D50 as a fraction).

The measured responses themselves (about 600 MB per scene) stay out of the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import from_impulse_response

HERE = Path(__file__).resolve().parent
CITATION = (
    "BRAS benchmark (Aspöck, Brinkmann, Ackermann, Weinzierl, Vorländer, 2020), "
    "DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0"
)
#: Name, and the air during the measurements (BRAS documentation, section 3).
SCENES = {
    "CR2": ("BRAS CR2 (seminar room)", 19.5, 41.7),
    "CR3": ("BRAS CR3 (chamber music hall)", 22.4, 40.9),
    "CR4": ("BRAS CR4 (auditorium)", 20.9, 37.5),
}
#: The bands of the published averages, which list 160 and 6300 Hz as 155 and 6350.
PUBLISHED_FIX = {155.0: 160.0, 6350.0: 6300.0}


def scene_folder(bras: Path, root: str, scene: str) -> Path:
    found = sorted((bras / root).glob(f"{scene} *"))
    if not found:
        raise SystemExit(f"no {scene} folder in {bras / root}")
    return found[0]


def read_sofa(path: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list, list]:
    """Impulse responses (receiver, emitter, samples), sample rate, emitter and receiver
    positions, and their BRAS numbers."""
    import h5py

    with h5py.File(path, "r") as f:
        ir = f["Data.IR"][()][0]
        rate = float(f["Data.SamplingRate"][()][0])
        emitters = np.squeeze(f["EmitterPosition"][()], axis=-1)
        receivers = np.squeeze(f["ReceiverPosition"][()], axis=-1)
        emitter_ids = [int(v) for v in np.ravel(f["EmitterID"][()])]
        receiver_ids = [int(v) for v in np.ravel(f["ReceiverID"][()])]
    return ir, rate, emitters, receivers, emitter_ids, receiver_ids


def published(bras: Path, scene: str) -> dict:
    data = bras / "4 Additional data" / "Room acoustic parameters" / "data"
    out: dict = {}
    for name, key in (("T20", "t20"), ("EDT", "edt"), ("C80", "c80"), ("D50", "d50")):
        rows = (data / f"Evaluation_{scene}_{name}.txt").read_text().split()
        bands = [PUBLISHED_FIX.get(float(v), float(v)) for v in rows[0].split(",")]
        values = [float(v) for v in rows[1].split(",")]
        if key == "d50":
            values = [round(v / 100.0, 6) for v in values]  # published in percent
        out.setdefault("bands_hz", bands)
        out[key] = values
    return out


def _round(values) -> list[float | None]:
    return [None if v is None else round(v, 4) for v in values]


def prepare(bras: Path, scene: str) -> None:
    name, temperature, humidity = SCENES[scene]
    folder = HERE / scene
    model = folder / f"{scene}.obj"
    if not model.exists():
        raise SystemExit(f"{model} is missing: make it with convert_skp.py first")

    surfaces = bras / "3 Surface descriptions" / "_csv"
    materials = sorted(p.stem for p in (surfaces / "fitted_estimates").glob(f"mat_{scene}_*.csv"))
    for kind in ("fitted", "initial"):
        (folder / "materials" / kind).mkdir(parents=True, exist_ok=True)
        for material in materials:
            shutil.copyfile(
                surfaces / f"{kind}_estimates" / f"{material}.csv",
                folder / "materials" / kind / f"{material}.csv",
            )

    sofa = scene_folder(bras, "1 Scene descriptions", scene) / "RIRs"
    ir, rate, emitters, receivers, emitter_ids, receiver_ids = read_sofa(
        next(sofa.glob(f"{scene}_RIRs_Dodecahedron.sofa"))
    )
    sources = {
        f"LS{n}": [round(float(v), 4) for v in p]
        for n, p in zip(emitter_ids, emitters, strict=True)
    }
    listeners = {
        f"MP{n}": [round(float(v), 4) for v in p]
        for n, p in zip(receiver_ids, receivers, strict=True)
    }

    pairs = {}
    for e, source in enumerate(sources):
        for r, listener in enumerate(listeners):
            params = from_impulse_response(ir[r, e], rate).as_dict()
            pairs[f"{source} {listener}"] = {k: _round(v) for k, v in params.items()}
    measured = {
        "scene": scene,
        "source": CITATION,
        "responses": f"{scene}_RIRs_Dodecahedron.sofa: the dodecahedron loudspeaker, in pascal",
        "analysis": "acoustic_spaces.parameters.from_impulse_response: IEC octave filters, "
        "noise cut and compensated (ISO 3382-1, 5.3.4); G is left out, as the responses have "
        "no free-field reference",
        "sample_rate": rate,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "sources": sources,
        "listeners": listeners,
        "pairs": pairs,
        "published": published(bras, scene),
    }
    text = json.dumps(measured, indent=1, ensure_ascii=False)
    # A list of numbers on one line each is long: put each list on one line.
    text = re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda m: "[" + " ".join(m[1].split()) + "]", text)
    (folder / "measured.json").write_text(text + "\n", encoding="utf-8")

    lines = [
        f"# {name}, from the BRAS benchmark (Aspöck et al., 2020),",
        "# DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0.",
        "# Made by validation/bras/prepare.py: the model converted from SketchUp, the surface",
        "# data as published (absorption fitted to the measured reverberation time; the",
        "# initial estimates are in materials/initial), and the dodecahedron loudspeaker and",
        "# microphone positions of the measured responses.",
        f"name: {name}",
        "geometry:",
        f"  file: {scene}.obj",
        "materials:",
    ]
    for material in materials:
        lines += [f"  {material}:", f"    file: materials/fitted/{material}.csv"]
    lines += [f"air: {{temperature_c: {temperature}, humidity_percent: {humidity}}}", "sources:"]
    lines += [f"  - {{name: {s}, position: {p}}}" for s, p in sources.items()]
    lines.append("listeners:")
    lines += [f"  - {{name: {s}, position: {p}}}" for s, p in listeners.items()]
    lines += [
        "simulation:",
        "  ir_length_s: 2.5",
        "  seed: 1",
        "outputs: [mono]",
    ]
    (folder / "scene.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{scene}: {len(materials)} materials, {len(pairs)} measured responses")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bras", type=Path, help="folder of the unpacked BRAS archives")
    parser.add_argument("--scenes", nargs="+", default=["CR2", "CR3"], choices=sorted(SCENES))
    args = parser.parse_args()
    for scene in args.scenes:
        prepare(args.bras, scene)


if __name__ == "__main__":
    main()
