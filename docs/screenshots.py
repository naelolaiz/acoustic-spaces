"""Make most of the pictures the documentation shows.

    xvfb-run -a -s "-screen 0 1920x1080x24" python docs/screenshots.py [--only NAME ...]
                                                                        [--bras DIR]

Needs the viz and gui extras. The 3D pictures and the editor need a display with
OpenGL, which ``xvfb-run`` gives on a machine without one. The simulated pictures are
drawn from the bundled examples with fixed seeds, so running the script again after a
change shows what the change did; only ``editor_map.png`` takes a new seed every time,
as the lecture hall sets none. ``coverage.png``, ``plan.png``, ``editor.png``,
``diffraction.png`` and ``orchestra_video.png`` are made by hand, ``wave.png`` by
``validation/wave_solver.py --plot``, and ``church_demo.png`` and ``stadium_demo.png`` are
frames of the videos the church and stadium demos make (docs/demos.md), made smaller.
``measured.png`` analyses a measured response of the BRAS benchmark (DOI
10.14279/depositonce-6726.3, CC BY-SA 4.0): ``--bras`` is the folder the BRAS archives
were unpacked into; without it, that picture is skipped.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early_paths
from acoustic_spaces.examples import example_path

IMAGES = Path(__file__).resolve().parent / "images"
#: The response the plots of one simulation show: a seat in the middle of the stalls.
SOURCE, LISTENER = "soloist", "stalls-middle"


def _room(example: str, **simulation) -> Room:
    scene = Scene.from_yaml(example_path(example))
    for key, value in simulation.items():
        setattr(scene.simulation, key, value)
    return Room.from_scene(scene)


def church_paths(out: Path) -> None:
    """The church in 3D with its early reflection paths up to the third order."""
    from acoustic_spaces.viz import show_room

    room = _room("church")
    show_room(room, out, paths=early_paths(room, max_order=3))


def particles(out: Path) -> None:
    """Sound particles leaving the soloist of the concert hall, as a GIF."""
    from acoustic_spaces.engine import trace_particles
    from acoustic_spaces.viz import animate_particles

    room = _room("concert_hall")
    tracked = trace_particles(room, SOURCE, duration=0.09, frames=40, seed=1)
    animate_particles(room, tracked, out, fps=12)


def response(out: Path) -> None:
    """Echogram, decays and arrival directions of one seat in the concert hall."""
    from acoustic_spaces.viz import plot_directions, plot_response

    room = _room("concert_hall", seed=1)
    shown = simulate(room)[SOURCE, LISTENER]
    plot_response(shown, out)
    plot_directions(shown, out.with_name("directions.png"))


def directivity(out: Path) -> None:
    """Polar patterns of a source that narrows with frequency, as in the documentation."""
    from acoustic_spaces.directivity import load_directivity
    from acoustic_spaces.viz import plot_directivity

    pattern = ["omni", "omni", "omni", "subcardioid", "subcardioid", "cardioid", "cardioid", 0.3]
    plot_directivity(load_directivity(SimpleNamespace(pattern=pattern, file=None)), out)


def measured(out: Path, bras: Path | None) -> None:
    """``analyze`` on a measured response: BRAS CR2, loudspeaker 1, microphone 1."""
    from acoustic_spaces.audio import read_wav
    from acoustic_spaces.viz import plot_impulse_response

    if bras is None:
        print(f"skipped {out.name}: no --bras folder")
        return
    wav = next(bras.glob("1 Scene descriptions/CR2*/RIRs/wav/CR2_RIR_LS1_MP1_Dodecahedron.wav"))
    ir, rate = read_wav(wav)
    plot_impulse_response(ir[:, 0], rate, out, title="BRAS CR2 seminar room, LS1 to MP1")


def editor_map(out: Path) -> None:
    """The editor with the coverage map of the lecture hall."""
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    from PySide6.QtWidgets import QApplication

    from acoustic_spaces.gui.app import MainWindow

    app = QApplication.instance() or QApplication(["acoustic-spaces"])
    window = MainWindow()
    window.resize(1600, 900)
    window.show()
    window.open_example(example_path("lecture_hall"))
    window.run_map()
    while window.job is not None:
        app.processEvents()
        time.sleep(0.05)
    window.map_panel.in_3d.setChecked(True)
    # The map from above is in coverage.png: the bottom panels give the 3D view their room.
    for name in ("results", "map", "messages"):
        window.docks[name].hide()
    for _ in range(20):  # let the 3D view draw the map
        app.processEvents()
        time.sleep(0.05)
    window.grab().save(str(out))
    window.close()


PICTURES = {
    "church_paths.png": church_paths,
    "particles.gif": particles,
    "response.png": response,
    "directivity.png": directivity,
    "measured.png": measured,
    "editor_map.png": editor_map,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(PICTURES), help="pictures to make")
    parser.add_argument("--bras", type=Path, help="folder of the unpacked BRAS archives")
    args = parser.parse_args()
    IMAGES.mkdir(exist_ok=True)
    for name in args.only or PICTURES:
        out = IMAGES / name
        start = time.perf_counter()
        if name == "measured.png":
            measured(out, args.bras)
        else:
            PICTURES[name](out)
        print(f"{name}: {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
