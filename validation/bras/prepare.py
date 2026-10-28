"""Make the BRAS scenes and measured parameters in this folder from the BRAS archives.

    python validation/bras/prepare.py path/to/BRAS [--scenes CR1_DoorAngle1 CR2 ...]

``path/to/BRAS`` is the folder the BRAS archives were unpacked into (the scene archives
``1_scene_descriptions-CRx.zip``, ``3_surface_descriptions.zip``,
``4_additional_data.zip`` and, for the binaural measurements,
``2_source_and_receiver_descriptions-QSC_K8.zip`` from
https://depositonce.tu-berlin.de/handle/11303/7506.3).
CR1, the coupled rooms, was measured with the door between them at two angles, and each
is a scene of its own: ``CR1_DoorAngle1`` (4.1 degrees) and ``CR1_DoorAngle3`` (30.4).
Needs the ``sofa`` extra (h5py). For each scene this writes, next to the room model that
``convert_skp.py`` made:

- ``materials/fitted/*.csv`` and ``materials/initial/*.csv``: the BRAS surface data as
  published, absorption fitted to the measured reverberation time and the initial
  estimates;
- ``scene.yaml``: the room, its materials (the fitted ones), the air as measured, and
  the loudspeaker and microphone positions of the dodecahedron measurements (two
  loudspeakers; five microphones, two in CR1);
- ``measured.json``: the ISO 3382-1 parameters of each measured response, analysed by
  this package as it analyses simulated ones, but G: each band's level instead (10 log10
  of the sum of its squared samples), whose differences between the responses are those
  of G; and the spatial averages BRAS publishes (third octaves; D50 as a fraction; none
  for CR1);
- ``binaural.json`` (CR2, CR3 and CR4): the setup of the binaural measurements, FABIAN
  facing loudspeaker LS7 and the five QSC K8 loudspeakers LS3 to LS7, and IACC_E and
  IACC_L of each measured binaural response with the head straight (view 0).

and once, in this folder, ``QSC_K8.csv``: the QSC K8's directivity, from the magnitudes
of its third-octave spectra (``2 Source and receiver descriptions/QSC K8``) every 5
degrees, as this package reads a directivity CSV.

The measured responses themselves (up to about 800 MB per scene) stay out of the
repository.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import from_impulse_response, interaural_cross_correlation

HERE = Path(__file__).resolve().parent
CITATION = (
    "BRAS benchmark (Aspöck, Brinkmann, Ackermann, Weinzierl, Vorländer, 2020), "
    "DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0"
)
#: Name, the air during the measurements (BRAS documentation, appendix), and the length
#: of the simulated responses (s): at least the room's longest measured T30.
SCENES = {
    "CR1_DoorAngle1": ("BRAS CR1 (coupled rooms), door at 4.1 degrees", 18.2, 47.6, 8.0),
    "CR1_DoorAngle3": ("BRAS CR1 (coupled rooms), door at 30.4 degrees", 18.2, 47.6, 8.0),
    "CR2": ("BRAS CR2 (seminar room)", 19.5, 41.7, 2.5),
    "CR3": ("BRAS CR3 (chamber music hall)", 22.4, 40.9, 2.5),
    "CR4": ("BRAS CR4 (auditorium)", 20.9, 37.5, 3.0),
}
#: The bands of the published averages, which list 160 and 6300 Hz as 155 and 6350.
PUBLISHED_FIX = {155.0: 160.0, 6350.0: 6300.0}
#: The angles (yaw and pitch, degrees) of FABIAN and the loudspeakers in the binaural
#: measurements, from the labels of each room's ``CRx_BRIR.skp``, whose positions are
#: those of the SOFA file. The loudspeakers aim at FABIAN's head, LS7 horizontally;
#: FABIAN faces LS7 with its head level and straight on the torso.
BRIR_ANGLES = {
    "CR2": {
        "FABIAN": (270.0, 0.0),
        "LS3": (110.5, 2.7),
        "LS4": (98.3, 2.2),
        "LS5": (81.8, 2.2),
        "LS6": (69.4, 5.5),
        "LS7": (90.0, 0.0),
    },
    "CR3": {
        "FABIAN": (180.0, 0.0),
        "LS3": (8.7, -4.6),
        "LS4": (3.8, -4.1),
        "LS5": (-3.8, -4.1),
        "LS6": (-8.7, -3.3),
        "LS7": (0.0, 0.0),
    },
    "CR4": {
        "FABIAN": (180.0, 0.0),
        "LS3": (8.5, 0.3),
        "LS4": (3.8, 0.3),
        "LS5": (-3.8, 0.3),
        "LS6": (-8.5, 1.5),
        "LS7": (0.0, 0.0),
    },
}
#: The grid step of ``QSC_K8.csv`` (degrees): that onto which directivities are read.
QSC_STEP_DEG = 5


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


def bras_scene(scene: str) -> tuple[str, str]:
    """The BRAS scene (CR1 to CR4) and the variant of its files (a door angle, or "")."""
    room, _, variant = scene.partition("_")
    return room, variant


def published(bras: Path, scene: str) -> dict:
    data = bras / "4 Additional data" / "Room acoustic parameters" / "data"
    out: dict = {}
    if not (data / f"Evaluation_{scene}_T20.txt").exists():
        return out  # CR1: none published
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


def _json(data: dict) -> str:
    text = json.dumps(data, indent=1, ensure_ascii=False)
    # A list of numbers on one line each is long: put each list on one line.
    return re.sub(r"\[\s*([^\[\]{}]*?)\s*\]", lambda m: "[" + " ".join(m[1].split()) + "]", text)


def prepare_binaural(bras: Path, room: str) -> None:
    """``binaural.json``: the setup and the measured IACC of a room's binaural responses."""
    import h5py

    responses = f"{room}_BRIRs.sofa"
    sofa = scene_folder(bras, "1 Scene descriptions", room) / "BRIRs" / responses
    with h5py.File(sofa, "r") as f:
        views = np.asarray(f["ListenerView"][()], dtype=float)
        view = int(np.argmin(np.abs(views[:, 0]) + np.abs(views[:, 1])))
        ir = np.asarray(f["Data.IR"][view], dtype=float)  # (ears, emitters, samples)
        rate = float(f["Data.SamplingRate"][()][0])
        emitters = np.squeeze(f["EmitterPosition"][()], axis=-1)
        emitter_ids = [int(v) for v in np.ravel(f["EmitterID"][()])]
        listener = np.ravel(f["ListenerPosition"][()])
    angles = BRIR_ANGLES[room]
    sources = {}
    iacc = {}
    for e, (n, position) in enumerate(zip(emitter_ids, emitters, strict=True)):
        name = f"LS{n}"
        yaw, pitch = angles[name]
        sources[name] = {"position": _round(position), "yaw": yaw, "pitch": pitch}
        found = interaural_cross_correlation(ir[:, e].T, rate)
        iacc[name] = {k: _round(v) for k, v in found.items()}
    yaw, pitch = angles["FABIAN"]
    measured = {
        "scene": room,
        "source": CITATION,
        "responses": f"{responses}: FABIAN, a head and torso simulator, its head straight "
        f"on the torso (view {views[view, 0]:g}, {views[view, 1]:g} degrees), facing LS7; "
        "one QSC K8 loudspeaker at a time",
        "setup": f"positions from {responses} (the loudspeakers' tweeters), yaw and pitch "
        f"from the labels of {room}_BRIR.skp",
        "analysis": "acoustic_spaces.parameters.interaural_cross_correlation: IEC octave "
        "filters; IACC_E from the direct sound to 80 ms, IACC_L from there to the end "
        "(ISO 3382-1, annex B)",
        "sample_rate": rate,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "listener": {"name": "FABIAN", "position": _round(listener), "yaw": yaw, "pitch": pitch},
        "sources": sources,
        "iacc": iacc,
    }
    (HERE / room / "binaural.json").write_text(_json(measured) + "\n", encoding="utf-8")
    print(f"{room}: IACC of {len(iacc)} binaural responses")


def prepare_qsc(bras: Path) -> None:
    """``QSC_K8.csv``: the QSC K8's directivity, every ``QSC_STEP_DEG`` degrees."""
    spectra = (
        bras
        / "2 Source and receiver descriptions"
        / "QSC K8"
        / "QSCK8_1x1_64442_MPS_front_pole.csv"
    )
    step = QSC_STEP_DEG
    rows = []
    with spectra.open(encoding="utf-8") as f:
        frequencies = [h.strip() for h in f.readline().split(",")[1:] if h.strip()]
        for line in f:
            label, _, rest = line.partition(",")
            phi, theta = int(label[1:4]), int(label[5:8])  # P000T000
            if theta % step or (theta in (0, 180) and phi) or phi % step:
                continue  # each pole once
            values = [
                complex(v.replace(" ", "").replace("i", "j")) for v in rest.split(",") if v.strip()
            ]  # "a + bi", and a comma at the end
            rows.append((phi, theta, 20.0 * np.log10(np.maximum(np.abs(values), 1e-12))))
    # Front pole (BRAS documentation, figure 3a): theta from +x, phi round the x axis
    # from +z towards +y.
    phi = np.radians([r[0] for r in rows])
    theta = np.radians([r[1] for r in rows])
    x, y, z = np.cos(theta), np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi)
    azimuth = np.degrees(np.arctan2(y, x))
    elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lines = [
        "# QSC K8 loudspeaker, from the BRAS benchmark (Aspöck et al., 2020),",
        "# DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0. Made by validation/bras/prepare.py:",
        f"# the magnitudes of QSCK8_1x1_64442_MPS_front_pole.csv every {step} degrees, in dB",
        "# (the on-axis response included); azimuth from the front towards the left.",
        "azimuth, elevation, " + ", ".join(frequencies),
    ]
    for a, e, (_p, _t, level) in zip(azimuth, elevation, rows, strict=True):
        lines.append(f"{a:.2f}, {e:.2f}, " + ", ".join(f"{v:.2f}" for v in level))
    (HERE / "QSC_K8.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"QSC_K8.csv: {len(rows)} directions")


def prepare(bras: Path, scene: str) -> None:
    name, temperature, humidity, length = SCENES[scene]
    room, variant = bras_scene(scene)
    folder = HERE / scene
    model = folder / f"{scene}.obj"
    if not model.exists():
        raise SystemExit(f"{model} is missing: make it with convert_skp.py first")

    surfaces = bras / "3 Surface descriptions" / "_csv"
    materials = sorted(p.stem for p in (surfaces / "fitted_estimates").glob(f"mat_{room}_*.csv"))
    for kind in ("fitted", "initial"):
        (folder / "materials" / kind).mkdir(parents=True, exist_ok=True)
        for material in materials:
            shutil.copyfile(
                surfaces / f"{kind}_estimates" / f"{material}.csv",
                folder / "materials" / kind / f"{material}.csv",
            )

    responses = f"{room}_RIRs_{variant + '_' if variant else ''}Dodecahedron.sofa"
    sofa = scene_folder(bras, "1 Scene descriptions", room) / "RIRs" / responses
    ir, rate, emitters, receivers, emitter_ids, receiver_ids = read_sofa(sofa)
    sources = {
        f"LS{n}": [round(float(v), 4) for v in p]
        for n, p in zip(emitter_ids, emitters, strict=True)
    }
    listeners = {
        f"MP{n}": [round(float(v), 4) for v in p]
        for n, p in zip(receiver_ids, receivers, strict=True)
    }

    pairs = {}
    unit = np.ones(len(OCTAVE_BANDS_HZ))  # G is then the band's level
    for e, source in enumerate(sources):
        for r, listener in enumerate(listeners):
            params = from_impulse_response(ir[r, e], rate, unit).as_dict()
            level, params["g"] = params["g"], [None] * len(unit)
            pairs[f"{source} {listener}"] = {k: _round(v) for k, v in params.items()}
            pairs[f"{source} {listener}"]["level_db"] = _round(level)
    measured = {
        "scene": scene,
        "source": CITATION,
        "responses": f"{responses}: the dodecahedron loudspeaker, in pascal",
        "analysis": "acoustic_spaces.parameters.from_impulse_response: IEC octave filters, "
        "noise cut and compensated (ISO 3382-1, 5.3.4); G is left out, as the responses have "
        "no free-field reference of their own; level_db is each band's level, 10 log10 of "
        "the sum of its squared samples, whose differences between the responses are those "
        "of G",
        "sample_rate": rate,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "sources": sources,
        "listeners": listeners,
        "pairs": pairs,
        "published": published(bras, room),
    }
    (folder / "measured.json").write_text(_json(measured) + "\n", encoding="utf-8")

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
        f"  ir_length_s: {length}",
        "  seed: 1",
        "outputs: [mono]",
    ]
    (folder / "scene.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{scene}: {len(materials)} materials, {len(pairs)} measured responses")
    if room in BRIR_ANGLES:
        prepare_binaural(bras, room)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bras", type=Path, help="folder of the unpacked BRAS archives")
    parser.add_argument("--scenes", nargs="+", default=sorted(SCENES), choices=sorted(SCENES))
    args = parser.parse_args()
    for scene in args.scenes:
        prepare(args.bras, scene)
    if any(bras_scene(scene)[0] in BRIR_ANGLES for scene in args.scenes):
        prepare_qsc(args.bras)


if __name__ == "__main__":
    main()
