"""The editor. Runs on Qt's off-screen platform unless QT_QPA_PLATFORM says otherwise;
with a display (or Xvfb) the 3D view tests run too."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyvistaqt")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from acoustic_spaces.audio import write_wav  # noqa: E402
from acoustic_spaces.examples import example_path, list_examples  # noqa: E402
from acoustic_spaces.gui.document import (  # noqa: E402
    Document,
    EditError,
    box_scene,
    model_scene,
    unique_name,
)
from acoustic_spaces.gui.viewport import placement_position  # noqa: E402
from acoustic_spaces.gui.worker import SimulationJob  # noqa: E402
from acoustic_spaces.room import Room  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication(["acoustic-spaces-tests"])


@pytest.fixture
def window(qapp, monkeypatch):
    from acoustic_spaces.gui.app import MainWindow

    win = MainWindow()
    monkeypatch.setattr(win, "ask_save", lambda: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: QMessageBox.StandardButton.Ok)
    win.show()
    yield win
    win.close()
    win.deleteLater()
    qapp.processEvents()


def _quick(document: Document) -> None:
    """Make the document's simulation fast."""

    def change(data: dict) -> None:
        data["simulation"]["ir_length_s"] = 0.3
        data["simulation"]["seed"] = 1
        data["simulation"].setdefault("late", {})["rays"] = 500
        data["simulation"].setdefault("early", {})["max_order"] = 2

    document.edit("quick", change)


def _wait(qapp, done, timeout: float = 120.0) -> None:
    started = time.monotonic()
    while not done():
        assert time.monotonic() - started < timeout, "timed out"
        qapp.processEvents()
        time.sleep(0.01)


# -- the document -----------------------------------------------------------------------


def test_edits_undo_and_redo(qapp) -> None:
    document = Document(box_scene())
    changes = []
    document.changed.connect(lambda: changes.append(1))
    assert not document.modified and document.undo_text() is None
    index = document.add_point("listeners", (2.0, 2.0, 1.2))
    assert index == 1 and document.scene.listeners[1].name == "listener-2"
    document.update_point("listeners", 1, name="seat")
    assert document.modified and document.undo_text() == "change listener"
    document.undo()
    assert document.scene.listeners[1].name == "listener-2"
    assert document.redo_text() == "change listener"
    document.undo()
    assert len(document.scene.listeners) == 1 and not document.modified
    document.redo()
    document.redo()
    assert document.scene.listeners[1].name == "seat" and document.redo_text() is None
    assert len(changes) == 6


def test_refused_edits_leave_the_scene_alone(qapp) -> None:
    document = Document(box_scene())
    scene = document.scene
    with pytest.raises(EditError, match="unique"):
        document.update_point("sources", 0, name="listener")
    with pytest.raises(EditError, match="at least one source"):
        document.remove_point("sources", 0)
    with pytest.raises(EditError, match="geometry"):
        document.edit("break", lambda data: data["geometry"].update(box=[-1, 2, 3]))
    assert document.scene is scene and document.undo_text() is None


def test_materials(qapp) -> None:
    document = Document(box_scene())
    name = document.add_material("carpet_cotton")
    assert name == "carpet_cotton" and document.add_material("carpet_cotton") == "carpet_cotton-2"
    document.rename_material("floor", "ground")
    assert document.scene.assignments["floor"] == "ground"
    document.rename_material("plaster", "walls")
    assert document.scene.default_material == "walls"
    with pytest.raises(EditError, match="already"):
        document.rename_material("walls", "ground")
    document.assign("ceiling", "carpet_cotton")
    document.remove_material("carpet_cotton")
    assert "ceiling" not in document.scene.assignments
    document.remove_material("walls")
    assert document.scene.default_material is None
    assert document.room.problems()  # the walls have no material now
    assert unique_name("a", {"a", "a-2"}) == "a-3"


def test_save_and_open(qapp, tmp_path: Path) -> None:
    document = Document(box_scene())
    with pytest.raises(ValueError, match="no file"):
        document.save()
    document.update_point("sources", 0, position=[1.0, 1.0, 1.5])
    document.save(tmp_path / "room.yaml")
    assert not document.modified and document.path == tmp_path / "room.yaml"
    again = Document.open(tmp_path / "room.yaml")
    assert again.scene.sources[0].position == (1.0, 1.0, 1.5)
    (tmp_path / "bad.yaml").write_text("name: x\n", encoding="utf-8")
    with pytest.raises(EditError, match="geometry"):
        Document.open(tmp_path / "bad.yaml")


def test_saving_an_example_copies_its_files(qapp, tmp_path: Path) -> None:
    document = Document.open(example_path("concert_hall"))
    document.save(tmp_path / "hall" / "scene.yaml", copy_files=True)
    assert (tmp_path / "hall" / "concert_hall.obj").is_file()
    assert "file: concert_hall.obj" in (tmp_path / "hall" / "scene.yaml").read_text()
    assert document.scene.geometry.file == (tmp_path / "hall" / "concert_hall.obj").resolve()
    assert not document.modified


def test_model_scene_puts_points_inside(qapp) -> None:
    scene = model_scene(EXAMPLES / "concert_hall" / "concert_hall.obj")
    room = Room.from_scene(scene)
    assert scene.name == "concert_hall"
    assert room.problems() == []
    assert scene.sources[0].position != scene.listeners[0].position


def test_placement_position(qapp) -> None:
    room = Room.from_scene(box_scene((8.0, 6.0, 3.0)))
    on_floor = placement_position(room, np.array([4.0, 3.0, 0.0]), np.array([0, 0, 1]), "sources")
    assert on_floor == pytest.approx((4.0, 3.0, 1.5))
    on_wall = placement_position(room, np.array([0.0, 3.0, 1.0]), np.array([-1, 0, 0]), "listeners")
    assert on_wall == pytest.approx((0.5, 3.0, 1.0))


def test_simulation_job(qapp) -> None:
    document = Document(box_scene())
    _quick(document)
    job = SimulationJob(document.room)
    results, messages = [], []
    job.progress.connect(messages.append)
    job.done.connect(results.append)
    job.start()
    _wait(qapp, lambda: job.isFinished() and results)
    job.wait()
    assert len(results[0]) == 1 and messages

    cancelled = []
    job = SimulationJob(document.room)
    job.cancelled.connect(lambda: cancelled.append(1))
    job.cancel()
    job.run()  # in this thread: stops at the first stage
    assert cancelled == [1]


# -- the window -------------------------------------------------------------------------


def test_window_opens_every_example(window) -> None:
    titles = [a.text() for a in window.examples_menu.actions()]
    assert titles == [e.title for e in list_examples()]
    for example in list_examples():
        assert window.open_example(example.path)
        assert window.from_example
        assert "problem" not in window.summary.text()
        assert example.path.name in window.windowTitle()


def test_selection_and_forms(window) -> None:
    window.open_example(example_path("studio"))
    window.select("sources", 1)
    page = window.inspector.pages["sources"]
    assert window.inspector.currentWidget() is page
    assert page.name.text() == "right-speaker"
    page.position[2].setValue(1.4)
    assert window.document.scene.sources[1].position[2] == pytest.approx(1.4)
    assert window.selection == ("sources", 1)
    assert window.undo_action.text() == "&Undo change source"

    page.name.setText("left-speaker")  # taken: refused, and the form shows the old name
    page.name.editingFinished.emit()
    assert window.document.scene.sources[1].name == "right-speaker"
    assert page.name.text() == "right-speaker"
    assert "Not changed" in window.messages.toPlainText()

    window.select("material", "side_panels")
    page = window.inspector.pages["material"]
    page.name.setText("panels")
    page.name.editingFinished.emit()
    assert "panels" in window.document.scene.materials
    assert window.selection == ("material", "panels")

    labels = window.document.room.mesh.labels
    ceiling = next(k for k, label in enumerate(labels) if "ceiling" in label)
    window.select("surface", ceiling)
    window.inspector.pages["surface"].material.setCurrentText("panels")
    assert window.document.scene.assignments["ceiling"] == "panels"

    window.select("simulation")
    window.inspector.pages["simulation"].late_rays.setValue(1234)
    assert window.document.scene.simulation.late.rays == 1234
    window.select("outputs")
    window.inspector.pages["outputs"].binaural.setChecked(False)
    assert [o.format for o in window.document.scene.outputs] == ["mono"]

    window.undo()
    assert [o.format for o in window.document.scene.outputs] == ["mono", "binaural"]
    window.select("sources", 7)  # gone: the scene instead
    assert window.selection == ("scene", None)


def test_tree_selects(window) -> None:
    tree = window.tree
    for item in tree._items():
        if tree._keys[item.data(0, 256)] == ("listeners", 0):
            tree.setCurrentItem(item)
            break
    assert window.selection == ("listeners", 0)
    assert window.inspector.kind == "listeners"


def test_place_remove_and_add_material(window, monkeypatch) -> None:
    window.start_placing("sources")
    if window.viewport.available:  # with a 3D view the next click places it
        window.viewport.place.emit("sources", (2.0, 3.0, 1.5))
    assert [s.name for s in window.document.scene.sources] == ["source", "source-2"]
    assert window.selection == ("sources", 1)
    window.remove_selected()
    assert len(window.document.scene.sources) == 1
    window.select("sources", 0)
    window.remove_selected()  # the last one stays
    assert len(window.document.scene.sources) == 1

    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("carpet_cotton", True))
    window.add_material()
    assert window.selection == ("material", "carpet_cotton")


def test_run_export_and_listen(window, qapp, tmp_path: Path, fake_sounddevice, monkeypatch) -> None:
    window.open_example(example_path("studio"))
    _quick(window.document)
    window.run_simulation()
    assert not window.run_action.isEnabled()
    _wait(qapp, lambda: window.job is None)
    assert window.result is not None and window.run_action.isEnabled()
    assert "Done in" in window.messages.toPlainText()
    results = window.results
    assert results.pair() == ("left-speaker", "mix-position")
    assert results.table.item(1, 4).text() not in ("", "–")  # T20 at 1 kHz
    for tab in range(results.tabs.count()):
        results.tabs.setCurrentIndex(tab)
    results.paths.setChecked(True)
    assert window.paths_action.isChecked()

    report = window.write_results(tmp_path / "out")
    assert report is not None and (tmp_path / "out" / "parameters.json").is_file()
    from PySide6.QtWidgets import QFileDialog

    glb = tmp_path / "studio.glb"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(glb), ""))
    window.blender_action.trigger()
    assert glb.read_bytes()[:4] == b"glTF"
    assert "reflection paths from left-speaker to mix-position" in window.messages.toPlainText()

    fake_sounddevice.play = lambda data, rate: fake_sounddevice.played.append((data, rate))
    fake_sounddevice.stop = lambda: None
    write_wav(tmp_path / "dry.wav", np.r_[1.0, np.zeros(999)], 16_000)
    assert window.play_file(tmp_path / "dry.wav")
    data, rate = fake_sounddevice.played[-1]
    assert data.shape[1] == 2 and rate == window.result.settings.sample_rate  # binaural
    assert np.abs(data).max() == pytest.approx(10 ** (-1 / 20), rel=1e-5)
    window.stop_playing()
    assert not results.stop.isEnabled()

    window.document.update_point("sources", 0, level_db=90.0)
    assert window._stale() and "changed" in results.stale.text()
    window.undo()
    assert not window._stale()


def test_export_to_blender(window, monkeypatch, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QFileDialog

    window.open_example(example_path("studio"))
    target = tmp_path / "particles.glb"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    window.select("sources", 1)
    window.particles_action.trigger()
    assert target.read_bytes()[:4] == b"glTF"
    assert "particles from right-speaker, 100 ms of sound over 4 s" in window.messages.toPlainText()
    target.unlink()
    assert window.export_blender()  # no run yet: the room alone
    assert window.messages.toPlainText().splitlines()[-1].startswith(f"Wrote {target}: the room.")
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert not window.export_blender(particles=True)


def test_run_refuses_a_scene_with_problems(window) -> None:
    window.document.edit("move", lambda data: data["sources"][0].update(position=[50, 0, 0]))
    window.run_simulation()
    assert window.job is None and "Cannot run" in window.messages.toPlainText()


def test_unsaved_changes_are_asked_about(window, monkeypatch, tmp_path: Path) -> None:
    window.document.update_point("sources", 0, name="talker")
    assert window.isWindowModified()
    monkeypatch.setattr(window, "ask_save", lambda: QMessageBox.StandardButton.Cancel)
    assert not window.open_file(example_path("shoebox"))
    assert window.close() is False and window.isVisible()
    from PySide6.QtWidgets import QFileDialog

    target = tmp_path / "room.yaml"
    monkeypatch.setattr(window, "ask_save", lambda: QMessageBox.StandardButton.Save)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    assert window.open_file(example_path("shoebox"))
    assert target.is_file() and window.document.path == example_path("shoebox")


def test_entry_points_explain_the_missing_extra(monkeypatch, capsys) -> None:
    from acoustic_spaces.cli import main
    from acoustic_spaces.gui import main as editor

    monkeypatch.setitem(sys.modules, "pyvistaqt", None)
    with pytest.raises(ImportError, match=r"acoustic-spaces\[gui\]"):
        editor([])
    assert main(["edit"]) == 1
    assert "gui extra" in capsys.readouterr().err


# -- the 3D view (needs OpenGL: a display or Xvfb) --------------------------------------


@pytest.mark.render
def test_clicks_select_and_place(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    qapp.processEvents()
    viewport.plotter.render()
    renderer = viewport.plotter.renderer

    def display(point):
        renderer.SetWorldPoint(*point, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        return round(x), round(y)

    picked = []
    viewport.picked.connect(lambda kind, key: picked.append((kind, key)))
    viewport.pick(*display(window.document.scene.listeners[0].position))
    assert picked[-1] == ("listeners", 0)
    assert window.selection == ("listeners", 0)

    placed = []
    viewport.place.connect(lambda kind, position: placed.append((kind, position)))
    viewport.start_placing("sources")
    viewport.pick(*display((4.0, 1.0, 0.0)))
    assert placed and placed[0][0] == "sources"
    assert Room.from_scene(window.document.scene).problems() == []
    assert len(window.document.scene.sources) == 2 and viewport.placing is None
