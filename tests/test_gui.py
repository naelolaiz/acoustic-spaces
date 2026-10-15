"""The editor. Runs on Qt's off-screen platform unless QT_QPA_PLATFORM says otherwise;
with a display (or Xvfb) the 3D view tests run too."""

import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")
pytest.importorskip("pyvistaqt")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from acoustic_spaces.audio import write_wav  # noqa: E402
from acoustic_spaces.examples import example_path, list_examples  # noqa: E402
from acoustic_spaces.geometry import RoomMesh  # noqa: E402
from acoustic_spaces.gui.document import (  # noqa: E402
    Document,
    EditError,
    box_scene,
    floor_height,
    follow_floor,
    model_scene,
    plan_corners,
    unique_name,
)
from acoustic_spaces.gui.plan_view import grid_step, outline_segments, snap  # noqa: E402
from acoustic_spaces.gui.viewport import placement_position  # noqa: E402
from acoustic_spaces.gui.worker import MapJob, SimulationJob  # noqa: E402
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


def test_floor_plans_in_the_document(qapp) -> None:
    document = Document(box_scene())
    np.testing.assert_allclose(plan_corners(document.scene), [[0, 0], [8, 0], [8, 6], [0, 6]])
    # Moving a corner of a box makes it a floor plan with the same walls.
    document.set_floor_plan([[0, 0], [8, 0], [8, 6], [3, 6], [3, 4], [0, 4]], "move corner")
    geometry = document.scene.geometry
    assert geometry.box is None and geometry.height == 3.0 and len(geometry.floor_plan) == 6
    assert document.undo_text() == "move corner" and document.room.problems() == []
    with pytest.raises(EditError, match="cross"):
        document.set_floor_plan([[0, 0], [8, 0], [0, 6], [8, 6]])
    assert len(document.scene.geometry.floor_plan) == 6
    document.undo()
    assert document.scene.geometry.box == (8.0, 6.0, 3.0)

    # Plans in other units keep them.
    document.edit(
        "cm",
        lambda d: d.update(
            geometry={"floor_plan": [[0, 0], [800, 0], [0, 600]], "height": 300, "units": "cm"}
        ),
    )
    np.testing.assert_allclose(plan_corners(document.scene), [[0, 0], [8, 0], [0, 6]])
    document.set_floor_plan([[0, 0], [9, 0], [0, 6]])
    assert document.scene.geometry.floor_plan[1] == (900.0, 0.0)

    # A drawn room keeps the height; points that would be outside move inside, in the
    # same arrangement.
    document = Document(box_scene())
    moved = document.draw_floor_plan([[10, 10], [20, 10], [20, 15], [10, 15]])
    assert moved == ["source", "listener"] and document.room.problems() == []
    source, listener = document.scene.sources[0], document.scene.listeners[0]
    assert source.position[0] < listener.position[0] and source.position[2] == 1.5
    assert document.undo_text() == "draw floor plan"
    assert document.draw_floor_plan([[9, 9], [21, 9], [21, 16], [9, 16]], 4.0) == []
    assert document.scene.geometry.height == 4.0
    with pytest.raises(EditError, match="cross"):
        document.draw_floor_plan([[0, 0], [1, 1], [1, 0], [0, 1]])
    document.move_point("listeners", 0, (1.0000000001, 2.0, 1.2))
    assert document.scene.listeners[0].position == (1.0, 2.0, 1.2)
    assert document.undo_text() == "move listener"

    model = Document(model_scene(EXAMPLES / "concert_hall" / "concert_hall.obj"))
    assert plan_corners(model.scene) is None
    with pytest.raises(EditError, match="model file"):
        model.set_floor_plan([[0, 0], [1, 0], [0, 1]])
    segments, labels = outline_segments(model.room)
    assert len(segments) == len(labels) > 4


def _ramp_mesh() -> RoomMesh:
    """A floor from x = 0 to 10 rising 0.2 m per metre, then a platform 3 m higher."""
    vertices = np.array(
        [
            [0, 0, 0],
            [10, 0, 2],
            [10, 4, 2],
            [0, 4, 0],
            [10, 0, 5],
            [20, 0, 5],
            [20, 4, 5],
            [10, 4, 5],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    return RoomMesh(vertices, faces, np.zeros(4, dtype=np.int64), [("floor",)])


def test_dragged_points_follow_the_floor() -> None:
    mesh = _ramp_mesh()
    assert floor_height(mesh, 5.0, 2.0) == pytest.approx(1.0)
    assert floor_height(mesh, 25.0, 2.0) is None
    x, y, z = follow_floor(mesh, (2.0, 2.0, 1.6), (7.0, 1.0))
    assert (x, y) == (7.0, 1.0) and z == pytest.approx(2.6)  # 1.2 above the rake
    # Onto the platform, 3 m higher: too big a step to follow.
    assert follow_floor(mesh, (9.0, 2.0, 3.0), (15.0, 2.0))[2] == pytest.approx(3.0)
    assert follow_floor(mesh, (5.0, 2.0, 2.2), (25.0, 2.0))[2] == pytest.approx(2.2)


def test_plan_helpers() -> None:
    np.testing.assert_allclose(snap([1.04, -2.96], 0.1), [1.0, -3.0])
    np.testing.assert_allclose(snap([1.04, -2.96], 0.0), [1.04, -2.96])
    assert grid_step(40.0) == 2.0  # 60 pixels at 40 pixels per metre
    assert grid_step(600.0) == 0.1 and grid_step(1.0) == 100.0


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
        if example.missing_extra:  # greyed out in the menu
            action = window.examples_menu.actions()[titles.index(example.title)]
            assert not action.isEnabled() and example.missing_extra in action.statusTip()
            continue
        assert window.open_example(example.path)
        assert window.from_example
        assert "problem" not in window.summary.text()
        assert example.path.name in window.windowTitle()


def test_leaks_are_found_and_shown(window, monkeypatch) -> None:
    import acoustic_spaces.engine.leaks as leaks_module

    found = []
    find = leaks_module.find_leaks
    monkeypatch.setattr(
        leaks_module, "find_leaks", lambda room: found.append(room) or find(room, rays=300)
    )
    window.open_example(example_path("leaky_classroom"))
    window.leaks_action.trigger()
    log = window.messages.toPlainText()
    assert "Leaks: 300 rays" in log and "by east_wall" in log
    assert len(found) == 1
    # Found again only when the room changes where it matters.
    window.select("listeners", 0)
    window.document.update_point("listeners", 0, position=[3.0, 2.0, 1.2])
    assert len(found) == 1
    window.document.update_point("sources", 0, position=[1.5, 3.0, 1.6])
    assert len(found) == 2
    window.leaks_action.trigger()  # hidden
    window.document.update_point("sources", 0, position=[1.4, 3.0, 1.6])
    assert len(found) == 2


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
    assert window.inspector.pages["simulation"].diffraction.currentText() == "first_order"
    window.inspector.pages["simulation"].diffraction.setCurrentText("direct")
    assert window.document.scene.simulation.early.diffraction == "direct"
    assert window.document.scene.simulation.late.rays == 1234
    page = window.inspector.pages["simulation"]
    assert page.coherent.currentText() == "125"
    page.coherent.setCurrentText("none")
    assert window.document.scene.simulation.early.coherent_max_band_hz is None
    page.coherent.setCurrentText("250")
    assert window.document.scene.simulation.early.coherent_max_band_hz == 250
    assert not page.wave.isChecked() and not page.wave_band.isEnabled()
    assert not page.wave_device.isEnabled()
    page.wave.setChecked(True)
    assert page.wave_band.isEnabled() and page.wave_device.isEnabled()
    page.wave_band.setCurrentText("63")
    page.wave_device.setCurrentText("auto")
    wave = window.document.scene.simulation.wave
    assert wave.enabled and wave.max_band_hz == 63 and wave.device == "auto"
    assert page.threads.value() == 0 and page.threads.specialValueText().startswith("All cores")
    assert page.ray_tree.currentText() == "sah"
    page.threads.setValue(2)
    page.ray_tree.setCurrentText("median")
    performance = window.document.scene.simulation.performance
    assert performance.threads == 2 and performance.ray_tree == "median"
    assert window.document.scene.simulation.wave.device == "auto"
    assert window.document.scene.simulation.early.diffraction == "direct"
    assert page.scattering_method.currentText() == "vector" and page.from_size.isChecked()
    assert page.by_band.isChecked() and page.by_band.isEnabled()
    page.by_band.setChecked(False)
    assert not window.document.scene.simulation.scattering.by_band
    page.scattering_method.setCurrentText("random")
    page.from_size.setChecked(False)
    scattering = window.document.scene.simulation.scattering
    assert scattering.method == "random" and not scattering.from_size
    assert not page.by_band.isEnabled()  # random scattering weights every band anyway
    assert window.document.scene.simulation.wave.enabled
    window.select("outputs")
    window.inspector.pages["outputs"].binaural.setChecked(False)
    assert [o.format for o in window.document.scene.outputs] == ["mono"]

    window.undo()
    assert [o.format for o in window.document.scene.outputs] == ["mono", "binaural"]
    window.select("sources", 7)  # gone: the scene instead
    assert window.selection == ("scene", None)


def test_third_octave_materials_and_bands(window) -> None:
    window.open_example(example_path("studio"))
    third = [round(0.01 + 0.03 * k, 2) for k in range(24)]
    window.document.set_material("side_panels", {"absorption": third, "scattering": 0.3})
    window.select("material", "side_panels")
    page = window.inspector.pages["material"]
    assert page.table.columnCount() == 24
    header = [page.table.horizontalHeaderItem(k).text() for k in (0, 13, 23)]
    assert header == ["50", "1k", "10k"]
    assert page.table.item(0, 23).text() == "0.70" and page.table.item(1, 5).text() == "0.30"
    page.table.item(0, 0).setText("0.50")
    material = window.document.scene.materials["side_panels"]
    assert material.absorption == [0.5, *third[1:]] and material.source.startswith("edited")
    window.select("material", "wooden_floor")  # octave data: 8 columns again
    assert page.table.columnCount() == 8
    assert page.table.horizontalHeaderItem(4).text() == "1 kHz"

    window.select("simulation")
    simulation = window.inspector.pages["simulation"]
    assert simulation.bands.currentText() == "octave"
    simulation.bands.setCurrentText("third_octave")
    assert window.document.scene.simulation.bands == "third_octave"
    window.undo()
    assert window.document.scene.simulation.bands == "octave"


def test_materials_let_sound_through(window) -> None:
    window.open_example(example_path("studio"))
    window.select("material", "wooden_floor")
    page = window.inspector.pages["material"]
    floor = window.document.scene.materials["wooden_floor"]
    absorption = list(floor.absorption)
    assert page.passing.currentText() == "nothing" and page.table.item(2, 0).text() == ""
    assert not page.table.item(2, 0).flags() & Qt.ItemFlag.ItemIsEditable

    def choose(kind: int):
        page.passing.setCurrentIndex(kind)
        page.passing.activated.emit(kind)
        return window.document.scene.materials["wooden_floor"]

    floor = choose(1)  # part of the sound: 40 dB to start from, the preset kept
    assert floor.transmission_loss == [40.0] * 8 and floor.preset == "stage_floor"
    assert page.table.item(2, 4).text() == "40.0"
    page.table.item(2, 3).setText("25")
    floor = window.document.scene.materials["wooden_floor"]
    assert floor.transmission_loss == [40.0] * 3 + [25.0] + [40.0] * 4
    page.table.item(2, 2).setText("")  # a blank band takes its neighbour's value
    assert window.document.scene.materials["wooden_floor"].transmission_loss[2] == 40.0
    page.table.item(2, 0).setText("-3")  # R is not negative: refused, shown as it was
    assert page.table.item(2, 0).text() == "40.0"
    floor = choose(2)  # an opening absorbs all, as all passes
    assert floor.transmission == [1.0] * 8 and floor.coefficients()[0] == [1.0] * 8
    assert page.table.item(2, 0).text() == "0.0" and page.table.item(0, 0).text() == "1.00"
    window.undo()
    window.undo()
    window.undo()
    window.undo()
    floor = window.document.scene.materials["wooden_floor"]
    assert floor.transmission_loss is None and floor.absorption == absorption

    # Audience seating, kept when the coefficients or the database entry change.
    window.select("material", "side_panels")
    page.audience.setChecked(True)
    assert window.document.scene.materials["side_panels"].audience
    page.table.item(0, 0).setText("0.55")
    panels = window.document.scene.materials["side_panels"]
    assert panels.audience and panels.absorption[0] == 0.55
    page.preset.setCurrentText("stage_floor")
    page.preset.activated.emit(0)
    panels = window.document.scene.materials["side_panels"]
    assert panels.audience and panels.preset == "stage_floor"
    window.select("material", "wooden_floor")
    assert not page.audience.isChecked()


def test_source_spectrum_gain_equalizer_and_delay(window) -> None:
    window.open_example(example_path("studio"))
    window.select("sources", 0)
    page = window.inspector.pages["sources"]
    assert page.spectrum.currentText() == "flat" and page.eq.text() == ""
    page.spectrum.setCurrentText("speech_female")
    page.spectrum.activated.emit(page.spectrum.currentIndex())
    page.gain.setValue(-3.0)
    page.delay.setValue(12.5)
    page.eq.setText("0, 0, 0, -3, 0, 0, 0, 0")
    page.eq.editingFinished.emit()
    source = window.document.scene.sources[0]
    assert source.spectrum == "speech_female" and source.talker == "female"
    assert source.gain_db == -3.0 and source.delay_ms == 12.5
    assert source.eq_db == [0, 0, 0, -3, 0, 0, 0, 0]
    page.eq.setText("loud")
    page.eq.editingFinished.emit()
    assert "numbers" in window.messages.toPlainText()
    assert window.document.scene.sources[0].eq_db == [0, 0, 0, -3, 0, 0, 0, 0]
    assert page.eq.text() == "0, 0, 0, -3, 0, 0, 0, 0"
    page.eq.setText("")
    page.eq.editingFinished.emit()
    assert window.document.scene.sources[0].eq_db is None
    # A level per band shows as its total and is edited in the scene file.
    window.document.update_point("sources", 0, level_db=[80.0] * 8, spectrum="flat")
    window.select("sources", 0)
    assert not page.level.isEnabled() and not page.spectrum.isEnabled()
    assert page.level.value() == pytest.approx(80.0 + 10 * np.log10(8), abs=0.05)


def test_line_and_surface_sources(window, qapp) -> None:
    window.open_example(example_path("studio"))
    window.select("sources", 0)
    page = window.inspector.pages["sources"]
    form = page._form
    assert page.shape.currentText() == "point" and not form.isRowVisible(page.extent[0])
    page.extent[0].setValue(3.0)  # no line yet: nothing changes
    assert not window.document.scene.sources[0].extended
    page.shape.setCurrentText("line")
    page.shape.activated.emit(page.shape.currentIndex())
    source = window.document.scene.sources[0]
    assert source.line_m == 3.0 and len(source.elements()) == 6
    assert form.isRowVisible(page.extent[0]) and not form.isRowVisible(page.extent[1])
    assert page.elements.text() == "6 uncorrelated point sources"
    page.spacing.setValue(1.0)
    assert len(window.document.scene.sources[0].elements()) == 3
    page.extent[1].setValue(2.0)
    page.shape.setCurrentText("surface")
    page.shape.activated.emit(page.shape.currentIndex())
    source = window.document.scene.sources[0]
    assert source.surface_m == (3.0, 2.0) and source.line_m is None
    assert form.labelForField(page.extent[0]).text() == "Width"
    assert form.isRowVisible(page.extent[1])
    page.directivity.setCurrentText("lambert")
    page.directivity.activated.emit(page.directivity.currentIndex())
    assert window.document.scene.sources[0].directivity == "lambert"
    assert "6.0" in page.directivity_index.text()
    # Too many points: refused, and the form shows the source as it was.
    page.spacing.setValue(0.05)
    assert "more than 400 point sources" in window.messages.toPlainText()
    assert window.document.scene.sources[0].spacing_m == 1.0 and page.spacing.value() == 1.0
    _plan(window, qapp).grab()  # drawn from above as a rectangle
    page.shape.setCurrentText("point")
    page.shape.activated.emit(page.shape.currentIndex())
    assert not window.document.scene.sources[0].extended


def test_a_surface_takes_a_material_of_its_own(window) -> None:
    from acoustic_spaces.gui.viewport import surface_materials

    window.open_example(example_path("lecture_hall"))  # walls: concrete, wall_1: wood
    labels = window.document.room.mesh.labels
    wall = next(k for k, label in enumerate(labels) if label[0] == "wall_2")
    window.select("surface", wall)
    page = window.inspector.pages["surface"]
    assert page.material.currentText() == "concrete"  # from the name all walls share
    page.material.setCurrentText("wood")
    assignments = window.document.scene.assignments
    assert assignments["wall_2"] == "wood" and assignments["walls"] == "concrete"
    room = window.document.room
    used = {
        label[0]: room.material_names[m]
        for label, m in zip(room.mesh.labels, surface_materials(room), strict=True)
    }
    assert used["wall_2"] == "wood" and used["wall_3"] == used["wall_4"] == "concrete"
    page.material.setCurrentText("(default material)")  # back to what the walls have
    assert "wall_2" not in window.document.scene.assignments
    assert window.document.scene.assignments["walls"] == "concrete"


def test_outputs_form_keeps_a_custom_loudspeaker_layout(window) -> None:
    from acoustic_spaces.scene import Scene

    data = box_scene().model_dump(mode="json", exclude_none=True)
    custom = {"format": "loudspeakers", "speakers": [{"name": "L", "azimuth": 30}]}
    custom["speakers"].append({"name": "R", "azimuth": -30})
    data["outputs"] = [custom]
    window._attach(Document(Scene.model_validate(data)))
    window.select("outputs")
    page = window.inspector.pages["outputs"]
    assert not page.loudspeakers.isChecked()  # the box is for a named layout
    assert "1 custom loudspeaker layout" in page.custom.text()
    page.mono.setChecked(True)
    outputs = window.document.scene.outputs
    assert [(o.format, o.layout) for o in outputs] == [("mono", None), ("loudspeakers", None)]
    assert [s.name for s in outputs[1].speakers] == ["L", "R"]
    page.loudspeakers.setChecked(True)
    assert [o.layout for o in window.document.scene.outputs] == [None, "5.0", None]
    assert page.loudspeakers.isChecked() and page.layout_name.currentText() == "5.0"


def test_forms_write_back_only_what_changed(window) -> None:
    window._attach(Document(box_scene((8.125, 6.0, 3.0))))
    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    page.box[2].setValue(3.5)  # the length shows rounded, to 8.12 or 8.13
    assert window.document.scene.geometry.box == (8.125, 6.0, 3.5)
    window.document.move_point("sources", 0, (1.234567, 2.345678, 1.5))  # as dragged
    window.document.update_point("sources", 0, orientation={"yaw": 12.345})
    window.select("sources", 0)
    page = window.inspector.pages["sources"]
    page.position[2].setValue(1.6)
    page.angles[1].setValue(10.0)
    source = window.document.scene.sources[0]
    assert source.position == (1.234567, 2.345678, 1.6)
    assert (source.orientation.yaw, source.orientation.pitch) == (12.345, 10.0)


def test_scene_form_keeps_the_humidity_to_a_tenth(window) -> None:
    window.document.edit("humid", lambda data: data["air"].update(humidity_percent=52.5))
    window.select("scene", None)
    page = window.inspector.pages["scene"]
    assert page.humidity.value() == 52.5
    page.humidity.stepUp()
    assert window.document.scene.air.humidity_percent == 57.5


def test_geometry_form_shows_the_room_units(window) -> None:
    from acoustic_spaces.scene import Scene

    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    assert page.box_group.title() == "Box (m)"  # no units given: metres
    data = box_scene().model_dump(mode="json", exclude_none=True)
    data["geometry"] = {"box": [8000, 6000, 3000], "units": "mm"}
    window._attach(Document(Scene.model_validate(data)))
    window.select("geometry", None)
    assert page.box_group.title() == "Box (mm)"
    assert [box.value() for box in page.box] == [8000, 6000, 3000]
    window.document.set_floor_plan([[0, 0], [8, 0], [8, 6]])  # corners in metres
    assert page.plan_group.title() == "Floor plan (mm)"
    assert page.plan.item(1, 0).text() == "8000" and page.height.value() == 3000


def test_an_ifc_model_picks_its_room(window) -> None:
    pytest.importorskip("ifcopenshell")
    window.open_example(example_path("bim_office"))
    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    assert page.file_form.isRowVisible(page.space)
    rooms = [page.space.itemText(k) for k in range(page.space.count())]
    assert rooms == ["every element", "Office"] and page.space.currentText() == "Office"
    page.space.setCurrentIndex(0)
    assert window.document.scene.geometry.space is None
    assert not window.document.room.report.closed  # walls as solids
    page.space.setCurrentIndex(1)
    assert window.document.scene.geometry.space == "Office"
    window.open_example(example_path("shoebox"))
    window.select("geometry", None)
    assert not page.file_form.isRowVisible(page.space)


def test_model_units_default_to_the_file(window) -> None:
    window.open_example(example_path("shoebox"))  # its scene says units: m
    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    assert page.units.currentText() == "m"
    page.units.setCurrentText("file default")
    assert window.document.scene.geometry.units is None
    assert window.document.room.volume_m3 == pytest.approx(144.0)  # an OBJ is in metres
    page.units.setCurrentText("cm")
    assert window.document.scene.geometry.units == "cm"
    assert window.document.room.volume_m3 == pytest.approx(144e-6)
    window.undo()
    page.reload()
    assert page.units.currentText() == "file default"


def test_source_directivity(window, monkeypatch, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QFileDialog

    from acoustic_spaces.gui.inspector import DIRECTIVITIES, MEASURED

    window.open_example(example_path("studio"))
    window.select("sources", 0)
    page = window.inspector.pages["sources"]
    box = page.directivity
    assert box.currentText().startswith("per band: omni, omni, omni, subcardioid")
    assert page.directivity_index.text().startswith("Directivity index: 0.0, 0.0, 0.0, 2.8")
    box.activated.emit(DIRECTIVITIES.index("cardioid"))
    assert window.document.scene.sources[0].directivity == "cardioid"
    assert box.currentText() == "cardioid"
    assert "4.8, 4.8" in page.directivity_index.text()

    table = tmp_path / "speaker.csv"
    rows = ["0, 0, 0", "180, 0, -20", "90, 0, -6", "270, 0, -6", "0, 90, -6", "0, -90, -6"]
    table.write_text("azimuth, elevation, 1000 Hz\n" + "\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(table), ""))
    box.activated.emit(box.findText(MEASURED))
    assert window.document.scene.sources[0].directivity.file == table
    assert box.currentText() == "file: speaker.csv"
    assert "Hz copied from the nearest octave" in page.directivity_index.text()

    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    box.activated.emit(box.findText(MEASURED))  # cancelled: nothing changes
    assert window.document.scene.sources[0].directivity.file == table
    assert box.currentText() == "file: speaker.csv"

    # An array, from its YAML description, and given in the scene.
    column = tmp_path / "column.yaml"
    column.write_text("element: cardioid\ncount: 4\nspacing_m: 0.15\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(column), ""))
    box.activated.emit(box.findText(MEASURED))
    assert box.currentText() == "file: column.yaml"
    assert window.document.room.directivities[0].name == "array of 4 elements"
    window.document.update_point(
        "sources", 0, directivity={"array": {"count": 3, "spacing_m": 0.2}}
    )
    window.select("sources", 0)
    assert box.currentText() == "array of 3 elements"
    window.undo()
    window.undo()
    assert window.document.scene.sources[0].directivity.file == table

    page.show_pattern()
    assert page.pattern_window.isVisible()
    assert page.pattern_window.windowTitle() == "Directivity of left-speaker"
    window.undo()
    assert window.document.scene.sources[0].directivity == "cardioid"
    page.pattern_window.close()


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

    # From a CATT-Acoustic or ODEON library: the entry's name, without its number.
    libraries = example_path("material_libraries").parent
    shown = []
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *a, **k: shown.extend(a[3]) or ("1002 Wooden door, solid", True),
    )
    window.add_library_material(str(libraries / "odeon_materials.li8"))
    assert shown[0].startswith("1001 Acoustic ceiling tiles")
    assert window.selection == ("material", "wooden_door_solid")
    door = window.document.scene.materials["wooden_door_solid"]
    assert door.entry == 1002 and door.source.startswith("'1002 Wooden door, solid' of")
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("walls", True))
    window.add_library_material(str(libraries / "catt_materials.geo"))
    assert window.document.scene.materials["walls"].scattering[5] == 0.1
    window.add_library_material(str(libraries / "scene.yaml"))
    assert "no materials found" in window.messages.toPlainText()


def test_another_scene_cancels_placing(window) -> None:
    window.viewport.start_placing("sources")  # waiting for a click in the 3D view
    window.plan.start_placing("listeners")
    window.open_example(example_path("studio"))
    assert window.viewport.placing is None and window.plan.canvas.placing is None


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
    from acoustic_spaces.gui.results_view import ROWS

    shown = [name for row, (name, _, _) in enumerate(ROWS) if not results.table.isRowHidden(row)]
    assert "lf" in shown and "st_early" not in shown  # the speakers are 2 m away
    assert "spl" in shown and results.table.item(ROWS.index(("spl", "SPL (dB)", "{:.1f}")), 4)
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

    # Head-tracked listening: a head tracker or the sliders turn the head.
    head = results.head
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as taken:
        taken.bind(("127.0.0.1", 0))
        head.port.setValue(taken.getsockname()[1])
        results.tracking.setChecked(True)
        assert head.isVisible() and head.receiver is None
        assert "Cannot listen for a head tracker on UDP port" in window.messages.toPlainText()
    head.listen()
    assert (
        head.receiver is not None
        and "Listening for a head tracker" in window.messages.toPlainText()
    )
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tracker:
        ypr = b"/ypr\0\0\0\0,fff\0\0\0\0" + struct.pack(">fff", 30.0, 10.0, 0.0)
        tracker.sendto(ypr, ("127.0.0.1", head.receiver.port))
        _wait(qapp, lambda: head.receiver.received == 1, timeout=10)
    head.refresh()
    assert (head.yaw.value(), head.pitch.value()) == (30, 10)
    assert "facing yaw 30°, pitch 10°, roll 0°; 1 tracker messages" in head.facing.text()
    head.centre.click()
    assert head.yaw.value() == 0
    head.yaw.setValue(-45)
    assert head.head.angles()[0] == pytest.approx(-45)
    assert window.play_file(tmp_path / "dry.wav")
    assert results.stop.isEnabled()
    _wait(qapp, lambda: window._player is None, timeout=30)
    assert not results.stop.isEnabled()
    assert fake_sounddevice.played[-1].shape == (512, 2)
    assert "head-tracked (order 3 Ambisonics to binaural)" in window.messages.toPlainText()
    results.tracking.setChecked(False)
    assert head.receiver is None and not head.isVisible()

    window.document.update_point("sources", 0, level_db=90.0)
    assert window._stale() and "changed" in results.stale.text()
    window.undo()
    assert not window._stale()


def test_one_playback_at_a_time(window, qapp, tmp_path: Path, fake_sounddevice) -> None:
    _quick(window.document)
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    stops = []
    fake_sounddevice.play = lambda data, rate: fake_sounddevice.played.append((data, rate))
    fake_sounddevice.stop = lambda: stops.append(1)
    write_wav(tmp_path / "dry.wav", np.r_[1.0, np.zeros(5 * 16_000)], 16_000)  # 5 s
    results = window.results
    assert window.play_file(tmp_path / "dry.wav")
    assert results.stop.isEnabled() and not results.listen.isEnabled()
    window.run_simulation()  # ends while the sound plays: Listen waits for Stop
    _wait(qapp, lambda: window.job is None)
    assert results.stop.isEnabled() and not results.listen.isEnabled()
    # Playing again (here head-tracked) stops what plays first.
    results.head.port.setValue(0)  # no head tracker
    results.tracking.setChecked(True)
    assert window.play_file(tmp_path / "dry.wav")
    assert stops and not window._play_timer.isActive()
    _wait(qapp, lambda: window._player is None, timeout=30)
    assert not results.stop.isEnabled() and results.listen.isEnabled()
    results.tracking.setChecked(False)


def test_results_follow_the_selected_listener(window, qapp) -> None:
    window.open_example(example_path("church"))
    _quick(window.document)
    window.select("listeners", 2)
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    results = window.results
    assert results.pair() == ("altar", "nave-back")  # the selected listener, not the first
    window.select("sources", 1)
    assert results.pair() == ("organ", "nave-back")
    window.select("material", "stone")
    assert results.pair() == ("organ", "nave-back")

    # Turning the listener and running again keeps its results on show.
    window.select("listeners", 2)
    window.inspector.pages["listeners"].angles[0].setValue(90.0)
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    assert results.pair() == ("organ", "nave-back")
    turned = np.array(window.document.scene.listeners[2].orientation.rotation())
    np.testing.assert_allclose(results.response().orientation, turned)
    results.listener.setCurrentText("nave-middle")
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    assert results.pair() == ("organ", "nave-middle")


def _quick_map(document: Document) -> None:
    """Make the document's coverage map fast."""
    _quick(document)
    document.edit("coarse map", lambda data: data.update(coverage={"spacing": 2.0, "rays": 500}))


def test_changing_absorption_re_weights_the_run(window, qapp) -> None:
    _quick(window.document)
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    traced = window.result
    assert traced is not None and window._traced is not None
    before = traced[("source", "listener")].energy().sum(axis=-1)
    fields = window.document.scene.materials["plaster"].model_dump(mode="json", exclude_none=True)
    fields.pop("preset", None)
    fields["absorption"] = [0.3] * 8
    window.document.set_material("plaster", fields)
    assert window.result is not traced and not window._stale()
    after = window.result[("source", "listener")].energy().sum(axis=-1)
    assert np.all(after < 0.8 * before)
    assert "re-weight" in window.results.stale.text()
    window.document.undo()
    assert window.result is traced and not window.results.stale.isVisible()
    window.document.edit("move", lambda data: data["sources"][0].update(position=[3.0, 3.0, 1.5]))
    assert window._stale() and "run again" in window.results.stale.text()


def test_map_job(qapp) -> None:
    document = Document(box_scene())
    _quick_map(document)
    job = MapJob(document.room)
    results = []
    job.done.connect(results.append)
    job.start()
    _wait(qapp, lambda: job.isFinished() and results)
    job.wait()
    assert len(results[0].grid) > 0 and results[0].combined is None


def test_coverage_map_in_the_editor(window, qapp, tmp_path: Path) -> None:
    import json

    window.open_example(example_path("church"))
    _quick_map(window.document)
    panel = window.map_panel
    assert panel.current() is None and not panel.save.isEnabled()
    window.map_action.trigger()
    assert not window.map_action.isEnabled() and not window.run_action.isEnabled()
    _wait(qapp, lambda: window.job is None)
    assert window.coverage is not None and window.map_action.isEnabled()
    assert "Map done in" in window.messages.toPlainText()
    # Two sources: the map of both together comes first.
    assert [panel.source.itemText(k) for k in range(panel.source.count())] == [
        "All sources",
        "altar",
        "organ",
    ]
    assert panel.name() == "sti" and not panel.band.isEnabled()
    assert panel.summary.text().startswith("STI: ")
    panel.parameter.setCurrentIndex(panel.parameter.findData("c80"))
    assert panel.band.isEnabled() and "C80 (dB), 1 kHz" in panel.summary.text()
    panel.source.setCurrentText("organ")
    assert panel.current().source == "organ"
    panel.in_3d.setChecked(True)
    assert panel.shown_in_3d()[1:] == ("c80", 4)

    assert window.write_map(tmp_path / "map.png")
    assert (tmp_path / "map.png").stat().st_size > 10_000
    assert window.write_map(tmp_path / "map.json")
    report = json.loads((tmp_path / "map.json").read_text(encoding="utf-8"))
    assert set(report["maps"]) == {"altar", "organ", "all sources"}

    # A map ignores listeners, even outside the room; a change makes it stale.
    window.document.update_point("listeners", 0, position=(100.0, 0.0, 1.2))
    assert "changed" in panel.note.text()
    window.run_map()
    _wait(qapp, lambda: window.job is None)
    assert panel.source.currentText() == "organ"  # kept across runs
    assert not panel.note.isVisible()


def test_coverage_settings_form(window) -> None:
    window.open_example(example_path("church"))
    window.select("coverage")
    page = window.inspector.pages["coverage"]
    assert window.inspector.currentWidget() is page
    assert page.automatic.isChecked() and not page.surfaces.isEnabled()
    assert page.spacing.value() == 1.5  # set in the example
    page.spacing.setValue(2.0)
    assert window.document.scene.coverage.spacing == 2.0
    page.automatic.setChecked(False)  # starts from the surfaces the map would take
    assert window.document.scene.coverage.surfaces == ["floor"]
    assert page.surfaces.isEnabled()
    assert page.surfaces.findItems("floor", Qt.MatchFlag.MatchExactly)[0].checkState() == (
        Qt.CheckState.Checked
    )
    page.noisy.setChecked(True)
    assert window.document.scene.coverage.noise_db == [35.0] * 8
    page.noise.item(0, 4).setText("30")
    assert window.document.scene.coverage.noise_db[4] == 30.0
    page.same_rays.setChecked(False)
    page.rays.setValue(12_345)
    assert window.document.scene.coverage.rays == 12_345
    assert window.undo_action.text() == "&Undo change coverage map settings"
    window.undo()
    assert window.document.scene.coverage.rays == 25_000  # the late rays, when unticked
    window.undo()
    assert window.document.scene.coverage.rays is None


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


def test_runs_of_a_scene_no_longer_open_are_dropped(window, qapp) -> None:
    for run in (window.run_simulation, window.run_map):
        window.open_example(example_path("shoebox"))
        _quick_map(window.document)
        run()
        assert window.job is not None
        window.job.wait()  # done, but the window has not heard yet
        window.new_box()
        _wait(qapp, lambda: window.job is None)
        assert window.result is None and window.results.result is None
        assert window.coverage is None and window.map_panel.current() is None
        assert "no longer open" in window.messages.toPlainText()
        assert window.run_action.isEnabled() and window.map_action.isEnabled()
    # Opening another scene while a run goes on stops it.
    _quick(window.document)
    window.run_simulation()
    job = window.job
    window.new_box()
    assert job._stop
    _wait(qapp, lambda: window.job is None)
    assert window.result is None


def test_saving_an_example_keeps_its_results(window, qapp, monkeypatch, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QFileDialog

    window.open_example(example_path("shoebox"))  # its model file goes along
    _quick_map(window.document)
    for run in (window.run_simulation, window.run_map):
        run()
        _wait(qapp, lambda: window.job is None)
    window.results.paths.setChecked(True)
    assert window._shown_response() is not None
    target = tmp_path / "box" / "scene.yaml"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    assert window.save()
    assert window.document.scene.geometry.file == (tmp_path / "box" / "shoebox.obj").resolve()
    # The same scene with a copy of its model: the run and the map still hold.
    assert not window._stale() and window.results.stale.text() == ""
    assert not window._map_stale() and window.map_panel.note.text() == ""
    assert window._shown_response() is not None


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


def test_save_as_adds_the_suffix(window, monkeypatch, tmp_path: Path) -> None:
    from PySide6.QtWidgets import QFileDialog

    for typed, saved in (("room", "room.yaml"), ("hall.yml", "hall.yml")):
        answer = (str(tmp_path / typed), "")
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, answer=answer, **k: answer)
        assert window.save_as()
        assert window.document.path == tmp_path / saved and (tmp_path / saved).is_file()
    assert not (tmp_path / "room").exists()
    # The added suffix names a file the dialog did not ask about: ask before replacing it.
    asked = []
    for reply, saved in (
        (QMessageBox.StandardButton.No, False),
        (QMessageBox.StandardButton.Yes, True),
    ):
        (tmp_path / "room.yaml").write_text("kept\n", encoding="utf-8")
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "room"), "")
        )
        monkeypatch.setattr(
            QMessageBox, "question", lambda *a, reply=reply, **k: asked.append(a[2]) or reply
        )
        assert window.save_as() is saved
        assert ((tmp_path / "room.yaml").read_text(encoding="utf-8") == "kept\n") is not saved
    assert asked == ["room.yaml already exists. Replace it?"] * 2


def test_entry_points_explain_the_missing_extra(monkeypatch, capsys) -> None:
    from acoustic_spaces.cli import main
    from acoustic_spaces.gui import main as editor

    monkeypatch.setitem(sys.modules, "pyvistaqt", None)
    with pytest.raises(ImportError, match=r'pip install "PySide6-Essentials'):
        editor([])
    assert main(["edit"]) == 1
    assert "gui extra" in capsys.readouterr().err


# -- the plan ---------------------------------------------------------------------------


def _plan(window, qapp):
    window.views.setCurrentWidget(window.plan)
    qapp.processEvents()
    window.plan.fit()
    return window.plan.canvas


def _at(canvas, point) -> QPoint:
    p = canvas.to_screen(point)
    return QPoint(round(p.x()), round(p.y()))


def _drag(qapp, widget, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    for t in (0.2, 0.6, 1.0):
        step = QPoint(
            round(start.x() + t * (end.x() - start.x())),
            round(start.y() + t * (end.y() - start.y())),
        )
        QTest.mouseMove(widget, step)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
    qapp.processEvents()


def test_plan_drags_corners_and_points(window, qapp) -> None:
    canvas = _plan(window, qapp)
    document = window.document
    assert canvas.mode == "box" and canvas.grab().width() > 100
    # A box corner dragged (and snapped to the 10 cm grid): now a floor plan.
    _drag(qapp, canvas, _at(canvas, (8, 6)), _at(canvas, (9.02, 7.03)))
    assert document.scene.geometry.floor_plan[2] == (9.0, 7.0)
    assert document.scene.geometry.height == 3.0 and canvas.mode == "plan"
    assert "now a floor plan" in window.messages.toPlainText()
    # A drag that would make walls cross is refused.
    _drag(qapp, canvas, _at(canvas, (9, 7)), _at(canvas, (4, -2)))
    assert document.scene.geometry.floor_plan[2] == (9.0, 7.0)
    assert "Not moved: walls" in window.messages.toPlainText()
    # Points move by dragging, keeping their height.
    _drag(qapp, canvas, _at(canvas, (5.6, 3.0)), _at(canvas, (6.01, 4.02)))
    assert document.scene.listeners[0].position == (6.0, 4.0, 1.2)
    assert window.selection == ("listeners", 0) and document.undo_text() == "move listener"
    # A double-click on a wall adds a corner; a click selects the wall.
    QTest.mouseDClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (4, 0))
    )
    qapp.processEvents()
    assert document.scene.geometry.floor_plan[:3] == [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)]
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (6, 0))
    )
    kind, label = window.selection
    assert kind == "surface" and document.room.mesh.labels[label][0] == "wall_2"
    # The right-click menu removes corners and places points.
    _menu(canvas, _at(canvas, (4, 0)), "Remove corner")
    assert len(document.scene.geometry.floor_plan) == 4
    _menu(canvas, _at(canvas, (2, 2)), "Place listener here")
    assert document.scene.listeners[-1].position == (2.0, 2.0, 1.2)
    # Empty space: the scene; the wheel zooms about the mouse.
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (4, 3))
    )
    assert window.selection == ("scene", None)
    pixel = QPointF(_at(canvas, (4, 3)))
    before = canvas.to_world(pixel)
    wheel = QWheelEvent(
        pixel,
        canvas.mapToGlobal(pixel),
        QPoint(0, 0),
        QPoint(0, 240),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    scale = canvas._scale
    QApplication.sendEvent(canvas, wheel)
    assert canvas._scale > scale
    np.testing.assert_allclose(canvas.to_world(pixel), before, atol=1e-9)
    canvas.grab()


def _menu(canvas, at: QPoint, choice: str) -> None:
    """Pick ``choice`` from the right-click menu at ``at``."""
    menu = canvas.context_menu(QPointF(at))
    texts = [action.text() for action in menu.actions()]
    assert choice in texts, texts
    menu.actions()[texts.index(choice)].trigger()


def test_plan_draws_rooms_and_places_points(window, qapp, monkeypatch) -> None:
    canvas = _plan(window, qapp)
    window.draw_room()
    assert canvas.mode == "drawing" and window.plan.draw.isChecked()
    corners = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 9), (0, 9)]
    for corner in corners:
        QTest.mouseClick(
            canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, corner)
        )
        QTest.mouseMove(canvas, _at(canvas, (5, 5)))
    canvas.grab()  # the drawing so far, with the length of the next wall
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (0, 0))
    )
    qapp.processEvents()
    geometry = window.document.scene.geometry
    assert geometry.floor_plan == [tuple(map(float, c)) for c in corners] and geometry.height == 3.0
    assert window.document.room.problems() == [] and window.selection == ("geometry", None)
    assert canvas.mode == "plan" and not window.plan.draw.isChecked()

    # A drawing that crosses itself is not a room; Esc stops drawing.
    window.plan.draw.click()
    for corner in [(0, 0), (4, 4), (4, 0), (0, 4)]:
        QTest.mouseClick(
            canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, corner)
        )
    QTest.keyClick(canvas, Qt.Key.Key_Backspace)
    assert len(canvas.drawing) == 3
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (0, 4))
    )
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    assert "Not a room yet: walls" in window.messages.toPlainText() and canvas.mode == "drawing"
    window._escape()
    assert canvas.mode == "plan" and len(window.document.scene.geometry.floor_plan) == 6

    # Placing from the plan: at the usual height over the floor.
    window.start_placing("sources")
    assert canvas.mode == "placing"
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (2, 7))
    )
    assert window.document.scene.sources[-1].position == (2.0, 7.0, 1.5)
    assert window.selection == ("sources", 1)

    # Drawing over a model file asks first.
    window.open_example(example_path("concert_hall"))
    canvas = _plan(window, qapp)
    assert canvas.mode == "file"
    canvas.grab()
    asked = []
    monkeypatch.setattr(window, "confirm_replace_model", lambda: asked.append(1) or False)
    window.plan.canvas.drawn.emit([[0, 0], [30, 0], [30, 20], [0, 20]])
    assert asked and window.document.scene.geometry.file is not None
    monkeypatch.setattr(window, "confirm_replace_model", lambda: True)
    window.plan.canvas.drawn.emit([[0, 0], [30, 0], [30, 20], [0, 20]])
    assert window.document.scene.geometry.floor_plan is not None
    assert window.document.room.problems() == []


def test_new_room_from_a_floor_plan(window, qapp) -> None:
    window.new_floor_plan()
    assert window.views.currentWidget() is window.plan and window.plan.canvas.mode == "drawing"
    window.plan.canvas.drawn.emit([[0, 0], [5, 0], [5, 5]])
    assert len(window.document.scene.geometry.floor_plan) == 3
    assert window.document.room.problems() == []


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


@pytest.mark.render
def test_a_line_source_is_drawn_and_picked_in_the_3d_view(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    window.document.update_point("sources", 0, line_m=3.0)
    window.select("scene", None)
    qapp.processEvents()
    viewport.plotter.render()
    renderer = viewport.plotter.renderer
    assert len(viewport._point_actors["sources", 0]) == 2  # the sphere and the line
    end = window.document.scene.sources[0].outline()[1]
    renderer.SetWorldPoint(*end, 1.0)
    renderer.WorldToDisplay()
    x, y, _ = renderer.GetDisplayPoint()
    assert viewport.marker_at(round(x), round(y)) == ("sources", 0)


@pytest.mark.render
def test_the_3d_view_keeps_its_lights(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    import pyvista as pv

    reference = pv.Plotter(off_screen=True)  # the light kit, as the view starts with
    lights = len(reference.renderer.lights)
    reference.close()
    assert lights > 0
    # The room is drawn when the window opens, and again for every change.
    assert len(viewport.plotter.renderer.lights) == lights
    window.select("sources", 0)
    window.open_example(example_path("studio"))
    assert len(viewport.plotter.renderer.lights) == lights


@pytest.mark.render
def test_the_camera_stays_while_the_room_is_edited(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    plotter = viewport.plotter

    def camera() -> np.ndarray:  # position, focal point and view up
        return np.array(plotter.camera_position.to_list(), dtype=float)

    plotter.camera.Azimuth(30)
    turned = camera()
    window.select("geometry", None)
    window.inspector.pages["geometry"].box[2].setValue(3.5)  # a new mesh
    assert window.document.room.report.bounds_max[2] == pytest.approx(3.5)
    np.testing.assert_allclose(camera(), turned)
    window.undo()
    np.testing.assert_allclose(camera(), turned)
    # A room drawn anew, or another scene, gets the usual view of the whole room.
    window.plan.canvas.drawn.emit([[20, 20], [30, 20], [30, 30], [20, 30]])
    assert window.document.scene.geometry.floor_plan is not None
    np.testing.assert_allclose(camera()[1, :2], [25, 25], atol=1e-6)
    plotter.camera.Azimuth(30)
    turned = camera()
    window.open_example(example_path("studio"))
    assert not np.allclose(camera(), turned)


@pytest.mark.render
def test_arrows_show_the_orientation(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    import pyvista as pv

    window.select("listeners", 0)
    page = window.inspector.pages["listeners"]
    for box, value in zip(page.angles, (90.0, 0.0, 90.0), strict=True):
        box.setValue(value)
    listener = window.document.scene.listeners[0]
    assert (listener.orientation.yaw, listener.orientation.roll) == (90.0, 90.0)
    qapp.processEvents()
    arrows = []
    for actor in viewport.plotter.actors.values():
        mapper = actor.GetMapper() if hasattr(actor, "GetMapper") else None
        mesh = pv.wrap(mapper.GetInput()) if hasattr(mapper, "GetInput") else None
        if mesh is not None and mesh.n_points == pv.Arrow().n_points:
            offset = mesh.points.mean(axis=0) - listener.position
            arrows.append(offset / np.linalg.norm(offset))
    # Facing +y and rolled to the right, the top of the head points to +x.
    assert any(np.allclose(a, [0, 1, 0], atol=1e-6) for a in arrows)
    assert any(np.allclose(a, [1, 0, 0], atol=1e-6) for a in arrows)


@pytest.mark.render
def test_dragging_points_in_3d(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    qapp.processEvents()
    viewport.plotter.render()
    renderer = viewport.plotter.renderer
    widget = viewport.plotter.interactor

    def widget_point(point) -> QPoint:
        renderer.SetWorldPoint(*point, 1.0)
        renderer.WorldToDisplay()
        x, y, _ = renderer.GetDisplayPoint()
        ratio = widget.devicePixelRatioF()
        return QPoint(round(x / ratio), round(widget.height() - 1 - y / ratio))

    camera = viewport.plotter.camera_position
    start = window.document.scene.listeners[0].position
    _drag(qapp, widget, widget_point(start), widget_point((6.0, 4.0, 1.2)))
    x, y, z = window.document.scene.listeners[0].position
    assert abs(x - 6.0) < 0.1 and abs(y - 4.0) < 0.1 and z == pytest.approx(1.2)
    assert window.document.undo_text() == "move listener"
    assert viewport.plotter.camera_position == camera  # the camera did not turn
    # A click without a drag selects.
    QTest.mouseClick(
        widget,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
        widget_point(window.document.scene.sources[0].position),
    )
    assert window.selection == ("sources", 0)
    # Up and down, with Shift.
    position = window.document.scene.sources[0].position
    target = viewport.drag_position(
        position, *_display(viewport, widget_point((position[0], position[1], 2.5))), vertical=True
    )
    assert target[:2] == position[:2] and abs(target[2] - 2.5) < 0.1


def _display(viewport, point: QPoint) -> tuple[int, int]:
    widget = viewport.plotter.interactor
    ratio = widget.devicePixelRatioF()
    return round(point.x() * ratio), round((widget.height() - point.y() - 1) * ratio)
