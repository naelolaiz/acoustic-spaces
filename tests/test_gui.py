"""The editor. Runs on Qt's off-screen platform unless QT_QPA_PLATFORM says otherwise;
with a display (or Xvfb) the 3D view tests run too."""

import json
import os
import re
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
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QToolBar,
)

from acoustic_spaces.audio import write_wav  # noqa: E402
from acoustic_spaces.examples import example_groups, example_path, list_examples  # noqa: E402
from acoustic_spaces.geometry import RoomMesh  # noqa: E402
from acoustic_spaces.gui.document import (  # noqa: E402
    NEW_PART_SIZES,
    Document,
    EditError,
    _in_units,
    box_scene,
    floor_height,
    follow_floor,
    model_scene,
    outdoor_scene,
    plan_corners,
    referenced_files,
    unique_name,
)
from acoustic_spaces.gui.levels import level_colour  # noqa: E402
from acoustic_spaces.gui.plan_view import grid_step, outline_segments, snap  # noqa: E402
from acoustic_spaces.gui.tree import EYE  # noqa: E402
from acoustic_spaces.gui.tune_dialog import FOLLOW, MAKE_DEFAULT, TuneDialog  # noqa: E402
from acoustic_spaces.gui.viewport import placement_position  # noqa: E402
from acoustic_spaces.gui.worker import MapJob, SimulationJob  # noqa: E402
from acoustic_spaces.room import Room  # noqa: E402
from acoustic_spaces.scene import Scene  # noqa: E402

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
    (tmp_path / "broken.yaml").write_text("name: [x\n", encoding="utf-8")
    with pytest.raises(EditError, match="broken.yaml: not valid YAML"):
        Document.open(tmp_path / "broken.yaml")


def test_saving_an_example_copies_its_files(qapp, tmp_path: Path) -> None:
    document = Document.open(example_path("concert_hall"))
    document.save(tmp_path / "hall" / "scene.yaml", copy_files=True)
    assert (tmp_path / "hall" / "concert_hall.obj").is_file()
    assert "file: concert_hall.obj" in (tmp_path / "hall" / "scene.yaml").read_text()
    assert document.scene.geometry.file == (tmp_path / "hall" / "concert_hall.obj").resolve()
    assert not document.modified


def test_saving_copies_the_element_file_of_an_inline_array(qapp, tmp_path: Path) -> None:
    folder = tmp_path / "old"
    folder.mkdir()
    bands = ", ".join(f"{f} Hz" for f in (125, 250, 500, 1000, 2000, 4000, 8000))
    rows = [f"{az}, {el}" + ", 0" * 7 for el in range(-90, 91, 10) for az in range(0, 360, 10)]
    text = "\n".join([f"azimuth, elevation, {bands}", *rows])
    (folder / "speaker.csv").write_text(text, encoding="utf-8")
    document = Document(box_scene(), folder / "scene.yaml")
    array = {"element": {"file": "speaker.csv"}, "count": 2, "spacing_m": 0.1}
    document.edit("array", lambda data: data["sources"][0].update(directivity={"array": array}))
    document.save(tmp_path / "new" / "scene.yaml", copy_files=True)
    assert (tmp_path / "new" / "speaker.csv").is_file()
    assert "file: speaker.csv" in (tmp_path / "new" / "scene.yaml").read_text()
    element = document.scene.sources[0].directivity.array.element
    assert element.file == (tmp_path / "new" / "speaker.csv").resolve()


def _speaker_csv(path: Path) -> None:
    bands = ", ".join(f"{f} Hz" for f in (125, 250, 500, 1000, 2000, 4000, 8000))
    rows = [f"{az}, {el}" + ", 0" * 7 for el in range(-90, 91, 10) for az in range(0, 360, 10)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([f"azimuth, elevation, {bands}", *rows]), encoding="utf-8")


def test_saving_copies_the_files_an_array_file_names(qapp, tmp_path: Path) -> None:
    """An array's YAML file names its element's file relative to itself: both move, or
    neither does."""
    old = tmp_path / "old"
    _speaker_csv(old / "arrays" / "box.csv")
    column = "element: {file: box.csv}\ncount: 2\nspacing_m: 0.4\n"
    (old / "arrays" / "column.yaml").write_text(column, encoding="utf-8")
    _speaker_csv(tmp_path / "shared" / "box.csv")
    outside = "element: {file: ../../shared/box.csv}\ncount: 2\nspacing_m: 0.4\n"
    (old / "arrays" / "outside.yaml").write_text(outside, encoding="utf-8")
    document = Document(box_scene(), old / "scene.yaml")
    document.edit(
        "arrays",
        lambda data: data["sources"][0].update(directivity={"file": "arrays/column.yaml"}),
    )
    loaded = document.room
    document.save(tmp_path / "new" / "scene.yaml", copy_files=True)
    assert (tmp_path / "new" / "arrays" / "box.csv").is_file()
    # The copies hold the same bytes: what was loaded from them is kept, not read again.
    assert document.room.directivities[0] is loaded.directivities[0]
    assert document.room.mesh is loaded.mesh
    assert "file: arrays/column.yaml" in (tmp_path / "new" / "scene.yaml").read_text()
    assert (
        document.scene.sources[0].directivity.file
        == (tmp_path / "new" / "arrays" / "column.yaml").resolve()
    )
    # An element's file outside the scene's folder stays where it is, and so does the
    # array file that names it.
    document = Document(box_scene(), old / "scene.yaml")
    document.edit(
        "arrays",
        lambda data: data["sources"][0].update(directivity={"file": "arrays/outside.yaml"}),
    )
    (tmp_path / "other").mkdir()  # as the editor makes it before saving
    document.save(tmp_path / "other" / "scene.yaml", copy_files=True)
    assert not (tmp_path / "other" / "arrays").exists()
    assert document.scene.sources[0].directivity.file == old / "arrays" / "outside.yaml"
    assert Room.from_yaml(tmp_path / "other" / "scene.yaml").problems() == []


def test_a_failed_save_of_an_example_leaves_no_copies(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = Document.open(example_path("concert_hall"))
    scene = document.scene

    def broken(*args, **kwargs):
        raise ValueError("cannot read it")

    monkeypatch.setattr(Scene, "model_validate", broken)
    with pytest.raises(ValueError, match="cannot read it"):
        document.save(tmp_path / "hall" / "deeper" / "scene.yaml", copy_files=True)
    assert not (tmp_path / "hall").exists()
    assert document.scene is scene


def test_saving_an_example_copies_its_terrain_and_map(qapp, tmp_path: Path) -> None:
    document = Document.open(example_path("hillside"))
    loaded = document.room
    document.save(tmp_path / "hill" / "scene.yaml", copy_files=True)
    assert document.room.mesh is loaded.mesh  # the terrain is not read again
    geometry = document.scene.geometry
    assert geometry.ground.terrain.file == (tmp_path / "hill" / "hill.asc").resolve()
    assert geometry.buildings_file.file == (tmp_path / "hill" / "houses.geojson").resolve()
    saved = (tmp_path / "hill" / "scene.yaml").read_text(encoding="utf-8")
    assert "file: hill.asc" in saved and "file: houses.geojson" in saved


def test_using_and_hiding_sources_and_listeners(qapp) -> None:
    """Ticks leave sources and listeners out in one step of the history; eyes hide them
    in the views without changing the scene."""
    document = Document(box_scene())
    document.add_point("sources", (2.0, 2.0, 1.5))
    document.add_point("sources", (3.0, 2.0, 1.5))
    document.set_enabled("sources", [1, 2], False)
    assert [s.in_use for s in document.scene.sources] == [True, False, False]
    assert document.undo_text() == "leave out 2 sources"
    document.set_enabled("sources", [0, 1], False)  # only the one in use changes
    assert document.undo_text() == "leave out source"
    document.undo()
    document.set_enabled("sources", [1, 2], True)
    assert document.undo_text() == "use 2 sources"
    assert all("enabled" not in s for s in document.scene.to_data()["sources"])
    document.set_enabled("listeners", [0], False)
    assert document.undo_text() == "leave out listener"
    assert document.scene.to_data()["listeners"][0]["enabled"] is False
    saved = document.scene
    document.set_hidden("sources", [1], True)
    assert document.hidden == {("sources", "source-2")} and document.scene is saved
    document.update_point("sources", 1, name="spare")
    assert document.hidden == {("sources", "spare")}
    document.set_hidden("sources", [0, 1], False)
    assert not document.hidden


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
    groups = [action.menu() for action in window.examples_menu.actions()]
    assert [menu.title() for menu in groups] == [group for group, _ in example_groups()]
    actions = {action.text(): action for menu in groups for action in menu.actions()}
    assert sorted(actions) == sorted(e.title for e in list_examples())
    for example in list_examples():
        if example.missing_extra:  # greyed out in the menu
            action = actions[example.title]
            assert not action.isEnabled() and example.missing_extra in action.statusTip()
            continue
        assert window.open_example(example.path)
        assert window.from_example
        assert "problem" not in window.summary.text()
        assert example.path.name in window.windowTitle()


def test_warnings_and_errors_stand_out(window) -> None:
    window.log("all fine")
    window.log("careful", "warning")
    window.log("broken\nreally", "error")
    brushes = {}
    block = window.messages.document().begin()
    while block.isValid():
        brushes[block.text()] = block.begin().fragment().charFormat().foreground()
        block = block.next()
    assert brushes["all fine"].style() == Qt.BrushStyle.NoBrush  # the theme's own
    assert brushes["careful"].color() == level_colour(window.messages, "warning")
    error = level_colour(window.messages, "error")
    assert brushes["broken"].color() == brushes["really"].color() == error
    assert window.messages.toPlainText().endswith("all fine\ncareful\nbroken\nreally")
    with pytest.raises(ValueError, match="info, warning, error"):
        window.log("loud", "shout")
    # Problems in the status bar, and the scene's warnings in its form.
    window.document.edit("move", lambda data: data["sources"][0].update(position=[50, 0, 0]))
    summary = window.summary.text()
    assert f'<span style="color:{error.name()}">, ' in summary and "problem" in summary
    window.open_example(example_path("leaky_classroom"))
    window.select("scene")
    stats = window.inspector.pages["scene"].stats.text()
    warning = level_colour(window.messages, "warning").name()
    assert f'<span style="color:{warning}">Warning: the model is not closed' in stats


def test_messages_show_hide_filter_and_clear(window) -> None:
    """Ticks show or hide each level, the item list and the filter keep only some
    messages, every other message is shaded and Clear removes those on show."""
    from acoustic_spaces.gui.messages import LIMIT, row_shade

    panel = window.message_panel
    window.open_example(example_path("studio"))
    panel.clear_shown()
    assert window.messages.toPlainText() == "" and panel.entries == []
    window.log("left-speaker is 2 m from mix-position")
    window.log("right-speaker clips", "warning")
    window.log("left-speakers is no name", "warning")
    window.log("broken\nreally", "error")
    assert [box.text() for box in panel.levels.values()] == [
        "Info (1)",
        "Warnings (2)",
        "Errors (1)",
    ]
    # Every other message on a shaded row, all the lines of one alike.
    shade = row_shade(window.messages)
    backgrounds = []
    block = window.messages.document().begin()
    while block.isValid():
        backgrounds.append(block.blockFormat().background())
        block = block.next()
    assert [b.style() == Qt.BrushStyle.NoBrush for b in backgrounds] == [
        True,
        False,
        True,
        False,
        False,
    ]
    assert backgrounds[1].color() == backgrounds[3].color() == backgrounds[4].color() == shade
    # Warnings hidden: the rest are shaded afresh from the top.
    panel.levels["warning"].setChecked(False)
    assert window.messages.toPlainText() == (
        "left-speaker is 2 m from mix-position\nbroken\nreally"
    )
    assert window.messages.document().begin().next().blockFormat().background().color() == shade
    panel.levels["warning"].setChecked(True)
    # The items that messages name, as whole names, with how many do.
    items = [panel.item.itemText(i) for i in range(panel.item.count())]
    assert items == ["All items", "left-speaker (1)", "mix-position (1)", "right-speaker (1)"]
    panel.item.setCurrentIndex(panel.item.findData("left-speaker"))
    assert window.messages.toPlainText() == "left-speaker is 2 m from mix-position"
    panel.item.setCurrentIndex(0)
    panel.filter.setText("CLIPS")
    assert window.messages.toPlainText() == "right-speaker clips"
    # Clear removes the messages on show; the hidden ones stay.
    panel.clear_button.click()
    panel.filter.clear()
    shown = window.messages.toPlainText()
    assert "clips" not in shown and "broken" in shown and "left-speakers" in shown
    assert panel.levels["warning"].text() == "Warnings (1)"
    assert panel.item.findData("right-speaker") == -1
    # A hidden message still shows in the status bar.
    panel.levels["info"].setChecked(False)
    window.log("hidden but said")
    assert "hidden but said" not in window.messages.toPlainText()
    assert window.statusBar().currentMessage() == "hidden but said"
    # Past the limit, the oldest go.
    for index in range(LIMIT + 201):
        panel.add(f"message {index}", "error")
    assert len(panel.entries) <= LIMIT + 200 and panel.entries[-1][0] == f"message {LIMIT + 200}"
    assert window.messages.toPlainText().endswith(f"message {LIMIT + 200}")


def test_background_task_says_what_runs(window, qapp) -> None:
    """While a run goes on, the status bar says what runs, its step and its time, with
    Cancel beside it; resting on the moving bar says so at length."""
    window.open_example(example_path("studio"))
    _quick(window.document)
    activity = window.activity
    assert activity.isHidden()
    window.run_simulation()
    assert activity.isVisible() and activity.cancel.defaultAction() is window.cancel_action
    assert activity.text().startswith("Simulating Control room")
    tip = activity.bar.toolTip()
    assert tip.startswith("Simulating Control room") and "Shift+F5" in tip
    assert activity.label.toolTip() == tip
    _wait(qapp, lambda: window.job is None)
    assert activity.isHidden()
    # The steps reached the bar, and the messages, not the status bar's message.
    assert "early reflections" in window.messages.toPlainText()
    assert "Now: " in activity.bar.toolTip() and " s: " in activity.text()
    assert "early reflections" not in window.statusBar().currentMessage()
    activity.start("Mapping a hall…")
    activity.step("map: early reflections at 40 points")
    assert activity.text() == "Mapping a hall… 0 s: map: early reflections at 40 points"
    assert activity.bar.toolTip().splitlines()[:3] == [
        "Mapping a hall…",
        "Now: map: early reflections at 40 points",
        "Running for 0 s",
    ]
    activity.stop()


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
    last = window.messages.document().lastBlock().begin().fragment().charFormat()
    assert last.foreground().color() == level_colour(window.messages, "warning")
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
    page = window.inspector.pages["simulation"]
    assert page.diffraction_reflections.currentText() == "one side"
    assert page.diffraction_reflections.isEnabled()
    page.diffraction_reflections.setCurrentText("both sides")
    assert window.document.scene.simulation.early.diffraction_reflections == "both_sides"
    window.inspector.pages["simulation"].diffraction.setCurrentText("direct")
    assert window.document.scene.simulation.early.diffraction == "direct"
    assert window.document.scene.simulation.late.rays == 1234
    # Only diffraction with reflections has paths that reflect on both sides of an edge.
    assert not page.diffraction_reflections.isEnabled()
    page.diffraction.setCurrentText("second_order")
    assert page.diffraction_reflections.isEnabled()
    page.diffraction_reflections.setCurrentText("one side")
    page.diffraction.setCurrentText("direct")
    assert window.document.scene.simulation.early.diffraction_reflections == "one_side"
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
    # Not set in the scene: this computer's default, the CPU until tune makes it auto.
    assert page.late_device.currentText() == "default (cpu)" and page.late_device.isEnabled()
    page.late_device.setCurrentText("both")
    assert window.document.scene.simulation.late.device == "both"
    assert window.document.scene.simulation.wave.device == "auto"
    page.late_device.setCurrentIndex(0)
    assert window.document.scene.simulation.late.device is None
    page.late_device.setCurrentText("both")
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


def test_every_control_explains_itself(window) -> None:
    """Resting the mouse on a label, field, button or column header shows a tip."""
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QCheckBox,
        QComboBox,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSlider,
        QTableWidget,
        QWidget,
    )

    from acoustic_spaces.gui.inspector import DIRECTIVITY_TIP

    window.open_example(example_path("studio"))
    scene = window.document.scene
    for kind, key in (
        ("geometry", None),
        ("surface", 0),
        ("material", next(iter(scene.materials))),
        ("sources", 0),
        ("listeners", 0),
        ("coverage", None),
    ):
        window.select(kind, key)
    roots = [
        *window.inspector.pages.values(),
        window.results,
        window.map_panel,
        window.plan,
        window.message_panel,
    ]
    fields = (QAbstractSpinBox, QCheckBox, QComboBox, QListWidget, QPushButton, QSlider)
    missing = []
    for root in roots:
        for widget in root.findChildren(QWidget):
            if isinstance(widget, QLabel):
                if widget.buddy() is not None and widget.text() and not widget.toolTip():
                    missing.append(f"label {widget.text()}")
            elif isinstance(widget, QLineEdit):
                inner = isinstance(widget.parentWidget(), (QComboBox, QAbstractSpinBox))
                if not inner and not widget.toolTip():
                    missing.append(f"line edit under {type(root).__name__}")
            elif isinstance(widget, fields) and not widget.toolTip():
                text = widget.text() if hasattr(widget, "text") else ""
                missing.append(f"{type(widget).__name__} {text} under {type(root).__name__}")
            elif isinstance(widget, QTableWidget):
                items = [widget.horizontalHeaderItem(c) for c in range(widget.columnCount())]
                items += [widget.verticalHeaderItem(r) for r in range(widget.rowCount())]
                for item in items:
                    if item is not None and not item.toolTip():
                        missing.append(f"header {item.text()} under {type(root).__name__}")
    assert not missing

    # The Results panel's band: the label explains it as the list does, and each band
    # column says which frequencies it holds.
    results = window.results
    band = next(
        label
        for label in results.findChildren(QLabel)
        if label.text() == "Band" and label.buddy() is not None
    )
    assert band.toolTip() == results.band.toolTip() and "octave band" in band.toolTip()
    header = results.table.horizontalHeaderItem(4).toolTip()
    assert "1 kHz" in header and "707 Hz to 1.41 kHz" in header
    assert results.table.verticalHeaderItem(2).toolTip().startswith("Reverberation time (s)")

    # A tip that follows the scene: the source's directivity names the choice in use.
    window.select("sources", 0)
    page = window.inspector.pages["sources"]
    tip = page.directivity.toolTip()
    assert tip.startswith(DIRECTIVITY_TIP) and page.directivity.currentText() in tip
    assert page._form.labelForField(page.directivity).toolTip() == tip

    # Every menu entry says what it does, in the status bar and as a tip in the menu.
    menus = [menu for menu in window.menuBar().findChildren(QMenu) if menu.title()]
    titles = {menu.title() for menu in menus}
    assert len(menus) >= 6
    for menu in menus:
        assert menu.toolTipsVisible(), menu.title()
        for action in menu.actions():
            if not action.isSeparator() and action.text() not in titles:
                assert action.statusTip() and action.toolTip(), action.text()
    assert window.run_action.toolTip().endswith("(F5)")


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

    def fine(data: dict) -> None:
        corners = [[0, 0], [8.1234567, 0], [8, 6.7654321], [0, 6]]
        data["geometry"] = {"floor_plan": corners, "height": 3.0}
        data["simulation"].update(ir_length_s=0.105, transition_ms=12.5)

    window.document.edit("fine", fine)
    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    page.plan.item(0, 1).setText("0.5")  # the others show rounded, to 8.12346 and 6.76543
    page._add_corner()  # half-way from the last corner to the first
    plan = [(0.0, 0.5), (8.1234567, 0.0), (8.0, 6.7654321), (0.0, 6.0), (0.0, 3.25)]
    assert window.document.scene.geometry.floor_plan == plan
    window.select("simulation")
    page = window.inspector.pages["simulation"]
    page.coherent.setCurrentText("none")
    page.late_rays.setValue(1234)  # the length shows as 0.10 or 0.11, the transition as 12 or 13
    simulation = window.document.scene.simulation
    assert (simulation.ir_length_s, simulation.transition_ms) == (0.105, 12.5)
    assert simulation.late.rays == 1234
    assert simulation.early.coherent_max_band_hz is None


def test_scene_form_keeps_the_humidity_to_a_tenth(window) -> None:
    window.document.edit("humid", lambda data: data["air"].update(humidity_percent=52.5))
    window.select("scene", None)
    page = window.inspector.pages["scene"]
    assert page.humidity.value() == 52.5
    page.humidity.stepUp()
    assert window.document.scene.air.humidity_percent == 57.5


def test_geometry_form_shows_the_room_units(window) -> None:

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


def test_outdoor_geometry_form(window, monkeypatch) -> None:
    window.open_example(example_path("street"))
    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    assert not page.ground_group.isHidden() and page.box_group.isHidden()
    assert page.ground_group.title() == "Outdoors (m)"
    assert page.sky.text() == "auto" and page.margin.value() == 30
    buildings = page.parts["buildings"]
    assert buildings.rowCount() == 2
    assert [buildings.item(0, c).text() for c in range(3)] == ["houses", "9", "4"]
    buildings.item(0, 1).setText("12")
    assert window.document.scene.geometry.buildings[0].height == 12.0
    page.sky.setValue(60.0)
    assert window.document.scene.geometry.ground.sky == 60.0
    assert window.document.room.outdoor.sky == 60.0
    page.sky.setValue(0.0)  # auto again
    assert window.document.scene.geometry.ground.sky is None
    page.parts["barriers"].setCurrentCell(0, 0)
    page._remove_part("barriers")
    assert window.document.scene.geometry.barriers is None
    assert page.parts["barriers"].rowCount() == 0
    page._remove_part("barriers")  # nothing left to remove
    # Drawing a room over the scene asks first, and the sky is no default material.
    asked = []
    monkeypatch.setattr(window, "confirm_replace_outdoors", lambda: asked.append(1) or False)
    window.plan.canvas.drawn.emit([[0, 0], [10, 0], [10, 8], [0, 8]])
    assert asked and window.document.scene.geometry.ground is not None
    monkeypatch.setattr(window, "confirm_replace_outdoors", lambda: True)
    window.plan.canvas.drawn.emit([[0, 0], [10, 0], [10, 8], [0, 8]])
    assert window.document.scene.geometry.floor_plan is not None
    assert window.document.scene.default_material == "asphalt"
    assert page.ground_group.isHidden()
    # Outdoors the early paths were in pressure up to 8 kHz; a room takes its default.
    assert window.document.scene.simulation.early.coherent_max_band_hz == 125


def test_vegetation_table(window) -> None:
    window.open_example(example_path("tree_belt"))
    window.select("geometry", None)
    page = window.inspector.pages["geometry"]
    plants = page.parts["vegetation"]
    assert plants.rowCount() == 2 and plants.columnCount() == 4
    assert [plants.item(0, c).text() for c in range(4)] == [
        "wood",
        "15",
        "belt, 4 corners",
        "dense foliage, 0.08 stems/m², 0.3 m across",
    ]
    assert plants.item(1, 2).text() == "tree, crown 10 across"
    plants.item(0, 1).setText("18")
    assert window.document.scene.geometry.vegetation[0].height == 18.0
    assert window.document.room.outdoor.vegetation[0].high == 18.0
    plants.setCurrentCell(1, 0)
    page._remove_part("vegetation")
    assert [v.name for v in window.document.scene.geometry.vegetation] == ["wood"]
    page._remove_part("vegetation")
    assert window.document.scene.geometry.vegetation is None and plants.rowCount() == 0
    window.open_example(example_path("street"))
    assert page.parts["vegetation"].rowCount() == 0


def test_outdoor_parts_in_the_document(qapp, tmp_path) -> None:
    from acoustic_spaces.geometry.terrain import HeightGrid, write_asc

    document = Document(outdoor_scene())
    scene = document.scene
    assert scene.geometry.ground.extent == ((-100.0, -100.0), (100.0, 100.0))
    assert scene.sources[0].position == (-25.0, 0.0, 1.5)
    assert scene.listeners[0].position[0] == 25.0 and scene.listeners[0].orientation.yaw == 180.0
    assert scene.materials["meadow"].impedance.preset == "grass"
    assert document.room.problems() == [] and document.room.outdoor is not None

    # Parts and zones are added with their corners in metres; the names follow.
    assert (
        document.add_part("buildings", {"footprint": [[0, 0], [20, 0], [20, 15]], "height": 10})
        == 0
    )
    assert document.undo_text() == "add building"
    building = document.scene.geometry.buildings[0]
    assert building.footprint == [(0.0, 0.0), (20.0, 0.0), (20.0, 15.0)] and building.height == 10
    assert document.add_part("barriers", {"path": [[-30, 20], [30, 20]], "height": 3}) == 0
    assert document.undo_text() == "add barrier" and document.room.problems() == []
    assert any("building_1_walls" in label for label in document.room.mesh.labels)

    # The extent, in the geometry's units; None leaves the margin to decide.
    document.set_extent([[-50, -60], [70, 80]])
    assert document.scene.geometry.ground.extent == ((-50.0, -60.0), (70.0, 80.0))
    assert document.room.outdoor.extent == (-50.0, -60.0, 70.0, 80.0)
    with pytest.raises(EditError, match="min < max"):
        document.set_extent([[50, 0], [-50, 10]])
    document.set_extent(None)
    assert document.scene.geometry.ground.extent is None
    assert document.room.outdoor.extent == (-60.0, -30.0, 60.0, 50.0)  # the parts, 30 m round
    assert document.undo_text() == "change ground"

    assert document.add_part("vegetation", {"tree": [40, -30], **NEW_PART_SIZES["tree"]}) == 0
    assert (
        document.add_part("vegetation", {"path": [[0, 40], [50, 40]], **NEW_PART_SIZES["hedge"]})
        == 1
    )
    plants = document.scene.geometry.vegetation
    assert (
        plants[0].tree == (40.0, -30.0)
        and plants[0].crown_diameter == 8.0
        and plants[0].base == 2.5
    )
    assert plants[1].width == 1.5 and plants[1].height == 2.0
    assert document.add_zone([[-80, -80], [-40, -80], [-40, -40], [-80, -40]]) == 0
    assert document.scene.geometry.ground.zone_names() == ["zone_1"]
    assert document.undo_text() == "add zone" and document.room.problems() == []
    with pytest.raises(EditError, match="cross"):
        document.add_part("buildings", {"footprint": [[0, 0], [4, 4], [4, 0], [0, 4]], "height": 5})
    assert len(document.scene.geometry.buildings) == 1
    with pytest.raises(EditError):
        document.add_part("vegetation", {"tree": [0, 0], "height": 1.0, "base": 2.0})
    document.undo()
    assert document.scene.geometry.ground.zones is None

    # A terrain from a file sets the ground's size; an extent goes with it.
    document.set_extent([[-60, -60], [60, 60]])
    assert document.room.outdoor.extent == (-60.0, -60.0, 60.0, 60.0)
    hill = tmp_path / "hill.asc"
    rows = [[0.5 * i for i in range(21)] for _ in range(21)]  # a slope up along x, 10 m high
    write_asc(hill, HeightGrid((-100.0, -100.0), 10.0, rows))
    document.set_terrain(hill)
    assert document.scene.geometry.ground.terrain.file == hill.resolve()
    assert document.scene.geometry.ground.extent is None
    assert document.room.outdoor.extent == (-100.0, -100.0, 100.0, 100.0)
    assert document.room.outdoor.ground_at([[0.0, 0.0]]).tolist() == [5.0]
    assert document.undo_text() == "set terrain"
    # The source and the listener stood on the flat ground; the slope buries them (the
    # ground is 3.75 and 6.25 m high there), and the editor says which.
    problems = document.room.problems()
    assert [p.split(" at ")[0] for p in problems] == ["source 'source'", "listener 'listener'"]
    assert all("outside the open air: below the terrain" in p for p in problems)
    with pytest.raises(EditError):
        document.set_terrain(tmp_path / "nowhere.asc")
    assert document.scene.geometry.ground.terrain.file == hill.resolve()

    # Buildings from a map join the drawn ones, and leave with the map.
    town = tmp_path / "town.geojson"
    town.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "hall", "height": 9},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-60, -60], [-50, -60], [-50, -50], [-60, -50]]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "shed"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[60, 60], [66, 60], [66, 66], [60, 66]]],
                        },
                    },
                ],
            }
        )
    )
    document.set_buildings_file(town, default_height=4.0)
    assert document.scene.geometry.building_names() == ["building_1", "hall", "shed"]
    assert [b.height for b in document.scene.geometry.all_buildings()] == [10.0, 9.0, 4.0]
    assert referenced_files(document.scene) == [hill.resolve(), town.resolve()]
    document.edit(
        "loss", lambda d: d["simulation"].update(insertion_loss_of=["hall", "building_1"])
    )
    document.set_buildings_file(None)
    assert document.scene.geometry.buildings_file is None
    assert document.scene.simulation.insertion_loss_of == ["building_1"]
    assert document.undo_text() == "remove the map's buildings"
    document.set_buildings_file(town, height=6.0, name="name")
    assert [b.height for b in document.scene.geometry.all_buildings()] == [10.0, 6.0, 6.0]
    document.set_terrain(None)
    assert document.scene.geometry.ground.terrain is None and document.room.problems() == []

    # Drawn in metres, kept in the geometry's units.
    assert _in_units({"footprint": [[1, 2], [3, 4]], "height": 5, "name": "a"}, 0.01) == {
        "footprint": [[100.0, 200.0], [300.0, 400.0]],
        "height": 500.0,
        "name": "a",
    }
    assert _in_units({"tree": [1.5, -2], "crown_diameter": 8, "base": 2.5, "height": 10}, 0.5) == {
        "tree": [3.0, -4.0],
        "crown_diameter": 16.0,
        "base": 5.0,
        "height": 20.0,
    }
    assert _in_units({"path": [[0, 0], [1, 0]], "width": 1.5, "height": 2}, 1.0) == {
        "path": [[0.0, 0.0], [1.0, 0.0]],
        "width": 1.5,
        "height": 2.0,
    }


def test_new_outdoor_scene_and_drawn_parts(window, qapp) -> None:
    window.new_outdoor()
    assert window.document.scene.geometry.ground is not None and not window.document.modified
    assert window.selection == ("geometry", None)
    assert "New outdoor scene: a meadow 200 m across" in window.messages.toPlainText()
    page = window.inspector.pages["geometry"]
    assert not page.ground_group.isHidden() and page.ground_group.title() == "Outdoors (m)"
    assert page.extent.text().startswith("Ground from (-100, -100) to (100, 100) m")
    canvas = _plan(window, qapp)

    # The Geometry form's buttons start a drawing of their kind in the plan view.
    window.views.setCurrentWidget(window.viewport)
    page._add("buildings")
    assert window.views.currentWidget() is window.plan
    assert canvas.mode == "drawing" and canvas.purpose == "building"
    assert not window.plan.draw.isChecked() and "footprint" in window.plan.hint.text()
    window.plan.canvas.part_drawn.emit("building", [[0, 0], [20, 0], [20, 15], [0, 15]])
    buildings = window.document.scene.geometry.buildings
    assert len(buildings) == 1 and buildings[0].height == 10.0
    assert buildings[0].footprint == [(0.0, 0.0), (20.0, 0.0), (20.0, 15.0), (0.0, 15.0)]
    assert "New building with 4 corners, 10 m high" in window.messages.toPlainText()
    assert page.parts["buildings"].rowCount() == 1 and window.selection == ("geometry", None)
    assert [page.parts["buildings"].item(0, c).text() for c in range(3)] == [
        "building_1",
        "10",
        "4",
    ]

    # A barrier is clicked point by point and finished with Enter.
    page._add("barriers")
    assert canvas.purpose == "barrier" and "end to end" in window.plan.hint.text()
    for point in [(-30, 20), (30, 20)]:
        QTest.mouseClick(
            canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, point)
        )
        QTest.mouseMove(canvas, _at(canvas, (0, 30)))
    canvas.grab()  # the path so far, with the length of the next stretch
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    qapp.processEvents()
    barriers = window.document.scene.geometry.barriers
    assert len(barriers) == 1 and barriers[0].height == 3.0
    assert np.allclose(barriers[0].path, [(-30, 20), (30, 20)], atol=0.5)
    assert canvas.mode == "outdoor" and canvas.purpose == "room"
    assert "Geometry form's buttons" in window.plan.hint.text()
    assert "New barrier along 2 points, 3 m high" in window.messages.toPlainText()

    # Vegetation takes the shape chosen beside the button: a tree is one click.
    page.plant_shape.setCurrentIndex(page.plant_shape.findData("tree"))
    page._add("vegetation")
    assert canvas.purpose == "tree"
    QTest.mouseMove(canvas, _at(canvas, (40, -30)))
    canvas.grab()  # the crown follows the cursor
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (40, -30))
    )
    qapp.processEvents()
    plants = window.document.scene.geometry.vegetation
    assert len(plants) == 1 and plants[0].kind == "tree" and plants[0].crown_diameter == 8.0
    assert np.allclose(plants[0].tree, (40, -30), atol=0.5) and canvas.mode == "outdoor"
    assert "New tree 10 m high, its crown 8 m across from 2.5 m up" in window.messages.toPlainText()
    page.plant_shape.setCurrentIndex(page.plant_shape.findData("hedge"))
    page._add("vegetation")
    assert canvas.purpose == "hedge"
    window.plan.canvas.part_drawn.emit("hedge", [[0, 40], [50, 40], [50, 60]])
    plants = window.document.scene.geometry.vegetation
    assert plants[1].kind == "hedge" and plants[1].width == 1.5 and plants[1].height == 2.0
    page.plant_shape.setCurrentIndex(page.plant_shape.findData("belt"))
    page._add("vegetation")
    assert canvas.purpose == "belt"
    window.plan.canvas.part_drawn.emit("belt", [[-90, 50], [-60, 50], [-60, 90], [-90, 90]])
    plants = window.document.scene.geometry.vegetation
    assert plants[2].kind == "belt" and plants[2].height == 12.0
    assert page.parts["vegetation"].rowCount() == 3
    assert page.parts["vegetation"].item(2, 2).text() == "belt, 4 corners"

    # A zone, drawn and closed on its first corner, takes the ground's material.
    page._add("zone")
    assert canvas.purpose == "zone" and "zone" in window.plan.hint.text()
    for point in [(-80, -80), (-40, -80), (-40, -40), (-80, -40), (-80, -80)]:
        QTest.mouseClick(
            canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, point)
        )
    qapp.processEvents()
    zones = window.document.scene.geometry.ground.zones
    assert len(zones) == 1 and len(zones[0].polygon) == 4 and canvas.mode == "outdoor"
    assert window.document.scene.geometry.ground.zone_names() == ["zone_1"]
    assert "New ground zone with 4 corners" in window.messages.toPlainText()
    assert page.zones.rowCount() == 1 and page.zones.item(0, 0).text() == "zone_1"
    assert window.document.room.problems() == []
    canvas.grab()  # zones and vegetation drawn in the plan
    window.document.set_extent(None)
    canvas.grab()

    # What cannot be a building yet says so and goes on; Esc stops it.
    page._add("buildings")
    for point in [(0, -50), (20, -30), (20, -50), (0, -30)]:
        QTest.mouseClick(
            canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, point)
        )
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    assert "Not a building yet:" in window.messages.toPlainText() and canvas.mode == "drawing"
    window._escape()
    assert canvas.mode == "outdoor" and canvas.purpose == "room"
    assert len(window.document.scene.geometry.buildings) == 1
    page._add("barriers")
    QTest.mouseClick(
        canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _at(canvas, (0, -50))
    )
    QTest.keyClick(canvas, Qt.Key.Key_Return)
    assert "Not a barrier yet: it needs two points" in window.messages.toPlainText()
    window._escape()
    # The Draw room button draws a room, whatever was being drawn before.
    page._add("zone")
    window.plan.draw.click()
    assert canvas.mode == "drawing" and canvas.purpose == "room"
    window.plan.draw.click()
    assert canvas.mode == "outdoor"

    # A refused part is reported, not added.
    window.plan.canvas.part_drawn.emit("zone", [[-80, -80], [-40, -80], [-40, -40], [-80, -40]])
    assert "Not changed:" in window.messages.toPlainText()  # overlaps the first zone
    assert len(window.document.scene.geometry.ground.zones) == 1

    # The removals, and undo all the way back to the meadow.
    page._remove_part("vegetation")
    assert len(window.document.scene.geometry.vegetation) == 2
    page._remove_zone()
    assert window.document.scene.geometry.ground.zones is None
    while window.document.undo_text() is not None:
        window.document.undo()
    geometry = window.document.scene.geometry
    assert geometry.buildings is None and geometry.barriers is None and geometry.vegetation is None


def test_outdoor_extent_terrain_and_map_controls(window, qapp, tmp_path, monkeypatch) -> None:
    import acoustic_spaces.gui.inspector as inspector
    from acoustic_spaces.geometry.terrain import HeightGrid, write_asc
    from acoustic_spaces.gui.inspector import MapDialog

    window.new_outdoor()
    page = window.inspector.pages["geometry"]
    document = window.document
    assert page.extent_set.isChecked()  # a new scene sets its extent
    assert [box.value() for box in page.extent_spins] == [-100.0, -100.0, 100.0, 100.0]
    page.extent_spins[2].setValue(150.0)
    assert document.scene.geometry.ground.extent == ((-100.0, -100.0), (150.0, 100.0))
    assert document.room.outdoor.extent == (-100.0, -100.0, 150.0, 100.0)
    page.extent_set.setChecked(False)  # nothing stands on the ground to go round
    assert "Not changed: " in window.messages.toPlainText() and "needs an extent" in (
        window.messages.toPlainText()
    )
    assert page.extent_set.isChecked() and document.scene.geometry.ground.extent is not None
    document.add_part(
        "buildings", {"footprint": [[-20, -10], [20, -10], [20, 10], [-20, 10]], "height": 10}
    )
    page.extent_set.setChecked(False)
    assert document.scene.geometry.ground.extent is None
    assert not page.extent_spins[0].isEnabled()
    assert page.extent.text().startswith("Ground from (-50, -40) to (50, 40) m")  # 30 m round
    assert [box.value() for box in page.extent_spins] == [-50.0, -40.0, 50.0, 40.0]
    page.extent_set.setChecked(True)  # takes the extent as it stands
    assert document.scene.geometry.ground.extent == ((-50.0, -40.0), (50.0, 40.0))
    assert page.extent_spins[0].isEnabled()
    page.extent_spins[0].setValue(60.0)  # west of east: refused, and the form reloads
    assert "min < max" in window.messages.toPlainText()
    assert page.extent_spins[0].value() == -50.0
    assert document.scene.geometry.ground.extent == ((-50.0, -40.0), (50.0, 40.0))

    # A terrain from a file, through the file dialog.
    hill = tmp_path / "hill.asc"
    rows = [[0.5 * i for i in range(21)] for _ in range(21)]
    write_asc(hill, HeightGrid((-100.0, -100.0), 10.0, rows))
    chosen = [str(hill)]
    monkeypatch.setattr(
        inspector.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (chosen[0], ""))
    )
    assert page.remove_terrain.isHidden() and page.terrain.isHidden()
    page.terrain_button.click()
    assert document.scene.geometry.ground.terrain.file == hill.resolve()
    assert document.scene.geometry.ground.extent is None
    assert not page.remove_terrain.isHidden() and not page.terrain.isHidden()
    assert page.terrain.text() == "Terrain: 21 × 21 points, z from 0 to 10 m, grid 10 m"
    assert document.room.outdoor.extent == (-100.0, -100.0, 100.0, 100.0)
    assert not page.extent_set.isChecked()
    assert [box.value() for box in page.extent_spins] == [-100.0, -100.0, 100.0, 100.0]
    chosen[0] = ""  # cancelled
    page.terrain_button.click()
    assert document.scene.geometry.ground.terrain is not None
    page.remove_terrain.click()
    assert document.scene.geometry.ground.terrain is None and page.remove_terrain.isHidden()

    # Buildings from a map: the file dialog, then how to read its heights and names.
    town = tmp_path / "town.geojson"
    town.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"name": "hall", "height": 9, "floors": 3},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[-60, -60], [-50, -60], [-50, -50], [-60, -50]]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"name": "shed", "floors": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[60, 60], [66, 60], [66, 66], [60, 66]]],
                        },
                    },
                ],
            }
        )
    )
    dialogs = []

    def accept(dialog) -> int:
        dialogs.append(dialog)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(MapDialog, "exec", accept)
    chosen[0] = str(town)
    assert page.remove_map.isHidden() and page.map_label.isHidden()
    page.map_button.click()
    dialog = dialogs[-1]
    assert dialog.height.currentData() == "height" and dialog.name.currentData() == "name"
    assert [dialog.height.itemText(k) for k in range(dialog.height.count())] == [
        "height",
        "floors",
        "the same for all",
    ]
    assert dialog.metres_label.text() == "Height when missing" and dialog.metres.value() == 10.0
    spec = document.scene.geometry.buildings_file
    assert spec.file == town.resolve() and spec.height == "height" and spec.default_height == 10.0
    assert document.scene.geometry.building_names() == ["building_1", "hall", "shed"]
    assert [b.height for b in document.scene.geometry.all_buildings()] == [10.0, 9.0, 10.0]
    assert page.map_label.text() == "Map town.geojson: 2 buildings, height from height"
    assert not page.remove_map.isHidden() and page.parts["buildings"].rowCount() == 3
    assert page.parts["buildings"].item(0, 1).flags() & Qt.ItemFlag.ItemIsEditable
    assert page.parts["buildings"].item(1, 1).toolTip().startswith("From the buildings file")
    assert not page.parts["buildings"].item(1, 1).flags() & Qt.ItemFlag.ItemIsEditable
    # One of its buildings left out, by name; the map stays.
    page.parts["buildings"].setCurrentCell(2, 0)
    page._remove_part("buildings")
    assert document.scene.geometry.buildings_file.without == ["shed"]
    assert document.scene.geometry.building_names() == ["building_1", "hall"]
    page.remove_map.click()
    assert document.scene.geometry.buildings_file is None
    assert page.parts["buildings"].rowCount() == 1 and page.remove_map.isHidden()

    # The same height for every building, and numbered names.
    def same_for_all(dialog) -> int:
        dialog.height.setCurrentIndex(dialog.height.count() - 1)
        assert dialog.metres_label.text() == "Height of all"
        dialog.metres.setValue(6.5)
        dialog.name.setCurrentIndex(dialog.name.count() - 1)
        return int(QDialog.DialogCode.Accepted)

    monkeypatch.setattr(MapDialog, "exec", same_for_all)
    page.map_button.click()
    spec = document.scene.geometry.buildings_file
    assert spec.height == 6.5 and spec.default_height is None and spec.name == "name"
    assert page.map_label.text() == "Map town.geojson: 2 buildings, 6.5 high"
    monkeypatch.setattr(MapDialog, "exec", lambda dialog: int(QDialog.DialogCode.Rejected))
    page.map_button.click()
    assert document.scene.geometry.buildings_file.height == 6.5  # cancelled: as it was

    # A file without footprints, or no GeoJSON at all, is refused with a message.
    empty = tmp_path / "empty.geojson"
    empty.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
    chosen[0] = str(empty)
    page.map_button.click()
    assert (
        "Not changed: empty.geojson holds no building footprints" in window.messages.toPlainText()
    )
    (tmp_path / "point.geojson").write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    chosen[0] = str(tmp_path / "point.geojson")
    page.map_button.click()
    assert "not GeoJSON buildings" in window.messages.toPlainText()
    assert document.scene.geometry.buildings_file.height == 6.5

    # The dialog on its own: a map whose properties hold no numbers offers one height.
    dialog = MapDialog(window, "plain.geojson", 1, {"name": "text", "use": "text"})
    assert dialog.height.currentData() is None and dialog.height.count() == 1
    assert dialog.choice() == {"name": "name", "height": 10.0}
    dialog.name.setCurrentIndex(dialog.name.findData("use"))
    assert dialog.choice() == {"name": "use", "height": 10.0}
    dialog.deleteLater()

    # Saved next to its files, the scene refers to them by name.
    document.set_terrain(hill)
    document.save(tmp_path / "scene.yaml")
    text = (tmp_path / "scene.yaml").read_text(encoding="utf-8")
    assert "file: hill.asc" in text and "file: town.geojson" in text and str(tmp_path) not in text


def test_ground_impedance_and_zones(window) -> None:
    from PySide6.QtCore import Qt

    window.open_example(example_path("roadside"))
    window.select("material", "meadow")
    page = window.inspector.pages["material"]
    assert page.ground.isChecked() and page.ground_kind.currentData() == "grass"
    assert not page.preset.isEnabled() and not page.passing.isEnabled()
    assert not page.flow.isEnabled() and page.flow.value() == 200.0
    assert not page.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable
    assert page.source.text() == "Source: the ground's impedance"

    def choose(combo, value) -> None:
        combo.setCurrentIndex(combo.findData(value))
        combo.activated.emit(combo.currentIndex())

    def impedance():
        return window.document.scene.materials["meadow"].impedance

    choose(page.ground_kind, "snow")
    assert impedance().preset == "snow"
    choose(page.ground_kind, "")  # a flow resistivity of its own, the preset's to start
    assert impedance().preset is None and impedance().flow_resistivity == 12.5
    assert page.flow.isEnabled()
    page.flow.setValue(300.0)
    assert impedance().flow_resistivity == 300.0
    page.thickness.setValue(0.1)
    assert impedance().thickness == 0.1
    assert page.reaction.currentData() == "local" and impedance().reaction is None
    choose(page.reaction, "extended")
    assert impedance().reaction == "extended" and impedance().thickness == 0.1
    choose(page.ground_model, "two_parameter")  # reacts locally: no reaction, no thickness
    assert impedance().model == "two_parameter" and impedance().thickness is None
    assert impedance().reaction is None
    assert page.porosity.isEnabled() and not page.thickness.isEnabled()
    assert not page.reaction.isEnabled()
    choose(page.ground_model, "miki")  # back to a model with an index: local again
    assert page.reaction.isEnabled() and impedance().reaction is None
    page.table.item(1, 0).setText("0.40")  # the scattering is the material's own
    assert window.document.scene.materials["meadow"].scattering[0] == 0.4
    assert impedance().model == "miki"
    page.ground.setChecked(False)  # the absorption stays, as values of its own
    material = window.document.scene.materials["meadow"]
    assert material.impedance is None and len(material.absorption) == 24
    assert page.preset.isEnabled() and material.source == "edited from a ground impedance"
    page.ground.setChecked(True)
    assert impedance() is not None and impedance().flow_resistivity == 300.0

    window.select("geometry", None)
    geometry = window.inspector.pages["geometry"]
    assert geometry.zones.rowCount() == 2
    assert [geometry.zones.item(0, c).text() for c in range(2)] == ["road", "4"]
    geometry.zones.setCurrentCell(1, 0)
    geometry._remove_zone()
    assert [z.name for z in window.document.scene.geometry.ground.zones] == ["road"]
    geometry._remove_zone()
    assert window.document.scene.geometry.ground.zones is None
    assert geometry.zones.rowCount() == 0

    # In pressure up to 8 kHz outdoors; up to 500 Hz in a room.
    window.select("simulation")
    simulation = window.inspector.pages["simulation"]
    assert simulation.coherent.currentText() == "8000"
    top = simulation.coherent.findData(8000)
    assert simulation.coherent.model().item(top).isEnabled()
    # The wave solver takes the wind outdoors, as an effective speed of sound or as air
    # that moves.
    assert not simulation.wave_wind.isEnabled()
    simulation.wave.setChecked(True)
    assert simulation.wave_wind.isEnabled()
    assert simulation.wave_wind.currentData() == "effective"
    simulation.wave_wind.setCurrentIndex(simulation.wave_wind.findData("moving"))
    assert window.document.scene.simulation.wave.wind == "moving"
    window.open_example(example_path("shoebox"))
    window.select("simulation")
    assert not simulation.coherent.model().item(top).isEnabled()
    simulation.wave.setChecked(True)
    assert not simulation.wave_wind.isEnabled()
    assert simulation.coherent.model().item(simulation.coherent.findData(500)).isEnabled()


def test_parabolic_equation_choice(window) -> None:
    """Outdoors in wind the simulation page picks the narrow-angle or the wide-angle
    parabolic equation; in still air and in a room there is none to pick."""
    window.open_example(example_path("open_air_stage"))
    window.select("simulation")
    page = window.inspector.pages["simulation"]
    assert page.parabolic_equation.isEnabled()
    assert page.parabolic_equation.currentText() == "narrow angle"
    page.parabolic_equation.setCurrentIndex(page.parabolic_equation.findData("wide_angle"))
    assert window.document.scene.simulation.early.parabolic_equation == "wide_angle"
    window.document.edit("still air", lambda data: data.pop("atmosphere"))
    window.select("simulation")
    assert not page.parabolic_equation.isEnabled()
    window.open_example(example_path("shoebox"))
    window.select("simulation")
    assert not page.parabolic_equation.isEnabled()
    assert page.parabolic_equation.currentData() == "narrow_angle"


def test_wind_and_temperature_form(window) -> None:
    window.open_example(example_path("open_air_stage"))
    window.select("scene", None)
    page = window.inspector.pages["scene"]
    assert not page.weather_group.isHidden()
    assert page.weather.currentData() == "windy" and page.wind_speed.value() == 5.0

    def atmosphere():
        return window.document.scene.atmosphere

    page.wind_speed.setValue(8.0)  # a value of its own makes the weather custom
    assert atmosphere().preset is None and atmosphere().wind.speed_m_s == 8.0
    assert page.weather.currentData() == "custom"
    page.wind_direction.setValue(90.0)
    assert atmosphere().wind.direction_deg == 90.0 and atmosphere().wind.speed_m_s == 8.0
    # Another weather keeps the way the wind blows.
    page.weather.setCurrentIndex(page.weather.findData("sunny_day"))
    assert atmosphere().preset == "sunny_day" and atmosphere().wind.direction_deg == 90.0
    page.gradient.setValue(-0.02)
    assert atmosphere().preset is None and atmosphere().temperature_log_k == -0.5
    assert atmosphere().temperature_gradient_k_m == -0.02
    # The eddies: a preset, then a value of their own; they stay as the weather changes.
    page.eddies.setCurrentIndex(page.eddies.findData("moderate"))
    assert atmosphere().turbulence.preset == "moderate" and page.wind_eddies.value() == 0.12
    page.wind_eddies.setValue(0.3)
    eddies = atmosphere().turbulence
    assert eddies.preset is None and eddies.wind_structure == 0.3
    assert eddies.temperature_structure == 0.008 and page.eddies.currentData() == "custom"
    page.weather.setCurrentIndex(page.weather.findData("windy"))
    assert atmosphere().preset == "windy" and atmosphere().turbulence.wind_structure == 0.3
    page.weather.setCurrentIndex(0)  # (none): still air, still with its eddies
    assert atmosphere().still and atmosphere().turbulence.wind_structure == 0.3
    assert page.weather.currentData() is None and page.wind_speed.value() == 0.0
    # Weaker with height: a height (0 is none) and a fall-off, kept through a preset.
    assert atmosphere().turbulence.height_m is None and page.eddy_height.value() == 0.0
    page.eddy_height.setValue(2.0)
    assert atmosphere().turbulence.height_m == 2.0
    page.eddy_falloff.setValue(1.0)
    assert atmosphere().turbulence.falloff == 1.0
    page.eddies.setCurrentIndex(page.eddies.findData("strong"))
    eddies = atmosphere().turbulence
    assert eddies.preset == "strong" and eddies.height_m == 2.0 and eddies.falloff == 1.0
    page.eddy_height.setValue(0.0)
    assert atmosphere().turbulence.height_m is None and page.eddies.currentData() == "custom"
    page.eddies.setCurrentIndex(0)
    assert atmosphere() is None and page.eddies.currentData() is None
    # A measured profile comes from the scene file; the form can drop it.
    window.document.edit(
        "measured", lambda data: data.update(atmosphere={"table": [[0, 15, 0], [10, 14, 3]]})
    )
    form = page._weather_form
    assert form.isRowVisible(page.drop_table) and not page.wind_speed.isEnabled()
    assert page.table.text() == "Measured profile at 2 heights, from the scene file"
    page.wind_direction.setValue(180.0)  # the table's wind blows that way
    assert atmosphere().table is not None and atmosphere().wind.direction_deg == 180.0
    page.drop_table.click()
    assert atmosphere().table is None and not form.isRowVisible(page.drop_table)
    page.show_profile()
    assert page.profile_window.isVisible()
    page.profile_window.close()
    window.open_example(example_path("shoebox"))
    window.select("scene", None)
    assert page.weather_group.isHidden()


def test_weather_zones_form(window, qapp) -> None:
    window.open_example(example_path("open_air_stage"))
    window.select("scene", None)
    page = window.inspector.pages["scene"]
    form = page._weather_form
    canvas = _plan(window, qapp)

    def atmosphere():
        return window.document.scene.atmosphere

    # The form shows the scene's weather; Add weather zone draws a zone in the plan.
    assert page.weather_zone.count() == 1 and page.weather_zone.currentData() is None
    assert not page.remove_weather_zone.isEnabled()
    assert not form.isRowVisible(page.zone_temperature)
    window.views.setCurrentWidget(window.viewport)
    page.add_weather_zone.click()
    assert window.views.currentWidget() is window.plan
    assert canvas.mode == "drawing" and canvas.purpose == "weather"
    assert "weather zone" in window.plan.hint.text()
    window.plan.canvas.part_drawn.emit("weather", [[20, -40], [80, -40], [80, 40], [20, 40]])
    zones = atmosphere().zones
    assert len(zones) == 1 and zones[0].polygon == [(20, -40), (80, -40), (80, 40), (20, 40)]
    assert zones[0].preset is None and zones[0].wind is None and zones[0].temperature_c is None
    assert atmosphere().preset == "windy" and atmosphere().wind.speed_m_s == 5.0
    assert "New weather zone with 4 corners" in window.messages.toPlainText()
    assert window.selection == ("scene", None)
    window.plan.cancel()  # the signal alone does not end the drawing
    assert canvas.mode == "outdoor"
    assert [page.weather_zone.itemText(k) for k in range(2)] == ["the whole scene", "weather_1"]
    canvas.grab()  # the plan draws the zone, dotted, with its name

    # Choosing the zone shows its weather, the scene's unchanged; the eddies are the
    # scene's everywhere, so their fields are greyed out.
    page.weather_zone.setCurrentIndex(1)
    assert page.weather.currentData() is None and page.weather.itemText(0) == "as the whole scene"
    assert page.wind_speed.value() == 5.0 and page.remove_weather_zone.isEnabled()
    assert form.isRowVisible(page.zone_temperature)
    assert page.zone_temperature.value() == window.document.scene.air.temperature_c
    assert not page.eddies.isEnabled() and not page.wind_eddies.isEnabled()

    # A value of its own changes the zone only; what it leaves out stays the scene's.
    page.wind_speed.setValue(1.0)
    zone = atmosphere().zones[0]
    assert zone.wind.speed_m_s == 1.0 and zone.wind.direction_deg is None
    assert zone.preset is None and atmosphere().wind.speed_m_s == 5.0
    assert page.weather.currentData() == "custom" and page.wind_speed.value() == 1.0
    assert atmosphere().weathers()[1].wind.speed_m_s == 1.0
    page.zone_temperature.setValue(12.0)
    assert atmosphere().zones[0].temperature_c == 12.0
    assert window.document.scene.air.temperature_c != 12.0
    # A preset of its own starts the zone from that preset: an inversion in still air.
    page.weather.setCurrentIndex(page.weather.findData("clear_night"))
    zone = atmosphere().zones[0]
    assert zone.preset == "clear_night" and zone.wind is None and zone.temperature_c == 12.0
    assert page.wind_speed.value() == 0.0 and page.log_term.value() == 0.5
    assert atmosphere().weathers()[1].temperature_log_k == 0.5
    page.show_profile()  # the zone's weather, as the engine resolves it
    assert page.profile_window.isVisible()
    page.profile_window.close()
    # Back to the scene's weather in the zone, then back to the scene's temperature.
    page.weather.setCurrentIndex(0)
    zone = atmosphere().zones[0]
    assert zone.preset is None and zone.wind is None and zone.temperature_c == 12.0
    page.zone_temperature.setValue(window.document.scene.air.temperature_c)
    assert atmosphere().zones[0].temperature_c is None
    # The whole scene's fields are its own, with the zone kept.
    page.weather_zone.setCurrentIndex(0)
    assert page.weather.currentData() == "windy" and page.eddies.isEnabled()
    assert page.weather.itemText(0) == "(none)" and not form.isRowVisible(page.zone_temperature)
    page.weather.setCurrentIndex(page.weather.findData("sunny_day"))
    assert atmosphere().preset == "sunny_day" and len(atmosphere().zones) == 1

    # Remove weather zone drops the chosen one; the scene's weather holds there again.
    page.weather_zone.setCurrentIndex(1)
    page.remove_weather_zone.click()
    assert atmosphere().zones is None and atmosphere().preset == "sunny_day"
    assert page.weather_zone.count() == 1 and page.weather_zone.currentData() is None
    assert not page.remove_weather_zone.isEnabled() and page.eddies.isEnabled()
    window.document.undo()
    assert len(atmosphere().zones) == 1 and page.weather_zone.count() == 2


def _differ(a, b) -> int:
    """How many pixels of two pictures of a widget differ."""

    def pixels(image):
        image = image.convertToFormat(image.Format.Format_RGB32)
        rows = np.array(image.constBits(), dtype=np.uint8)  # a copy: the image goes
        return rows.reshape(image.height(), -1)[:, : 4 * image.width()].reshape(
            image.height(), image.width(), 4
        )

    return int(np.count_nonzero(np.any(pixels(a.toImage()) != pixels(b.toImage()), axis=-1)))


def test_the_wind_is_drawn_outdoors(window, qapp) -> None:
    window.open_example(example_path("open_air_stage"))
    canvas = _plan(window, qapp)
    wind, profile = window.wind_action, window.wind_profile_action
    assert wind.isEnabled() and wind.isChecked() and profile.isEnabled() and profile.isChecked()
    assert wind.toolTip() and profile.toolTip()
    assert canvas.show_wind and canvas._wind is not None and canvas._wind.windy
    shown = canvas.grab()
    wind.setChecked(False)
    hidden = canvas.grab()
    assert not canvas.show_wind and _differ(shown, hidden) > 500  # arrows and the key
    wind.setChecked(True)
    assert _differ(shown, canvas.grab()) == 0
    # A weather zone carries its name and its wind.
    window.plan.canvas.part_drawn.emit("weather", [[20, -40], [80, -40], [80, 40], [20, 40]])
    window.plan.cancel()
    window.inspector.pages["scene"].weather_zone.setCurrentIndex(1)
    window.inspector.pages["scene"].wind_speed.setValue(1.0)
    assert canvas._wind.label(1) == "weather_1: 1.0 m/s"
    canvas.grab()
    # Indoors there is no wind to show.
    window.open_example(example_path("studio"))
    assert not wind.isEnabled() and not profile.isEnabled() and canvas._wind is None


def test_the_wind_in_the_scene_tree(window, qapp) -> None:
    """Outdoors the scene tree has the wind, with eyes for its arrows and its masts that
    View > Show wind and Show wind profile follow, and its weather zones, which show
    their weather in the scene's form."""
    window.open_example(example_path("lakeside_concert"))
    tree = window.tree

    def entry(key):
        return next(item for item in tree._items() if tree._keys[item.data(0, 256)] == key)

    def shown(key) -> bool:
        return tree._points[entry(key).data(0, 256)][2]

    wind, masts = entry(("scene", "wind")), entry(("scene", "profile"))
    assert wind.text(1) == "2.5 m/s at 10 m" and masts.parent() is wind
    zones = [entry(("scene", f"zone:{k}")) for k in range(2)]
    assert [(zone.text(0), zone.text(1)) for zone in zones] == [
        ("lake", "3.5 m/s at 10 m"),
        ("village", "1.0 m/s at 10 m"),
    ]
    assert all(item.toolTip(0) and item.toolTip(EYE) for item in (wind, masts))
    assert all(zone.toolTip(0) and zone.parent() is wind for zone in zones)
    tree.toggle_shown(wind)
    qapp.processEvents()
    assert not window.wind_action.isChecked() and window.wind_profile_action.isChecked()
    assert not window.plan.canvas.show_wind and not shown(("scene", "wind"))
    tree.toggle_shown(entry(("scene", "profile")))
    qapp.processEvents()
    assert not window.wind_profile_action.isChecked() and not shown(("scene", "profile"))
    window.wind_action.setChecked(True)
    window.wind_profile_action.setChecked(True)
    assert shown(("scene", "wind")) and shown(("scene", "profile"))
    # A zone's entry shows its weather in the scene's form; the wind's, the scene's.
    page = window.inspector.pages["scene"]
    tree.setCurrentItem(entry(("scene", "zone:1")))
    qapp.processEvents()
    assert window.selection == ("scene", "zone:1")
    assert page.weather_zone.currentData() == "zone:1"
    assert page.wind_speed.value() == pytest.approx(1.0)
    tree.setCurrentItem(entry(("scene", "wind")))
    qapp.processEvents()
    assert page.weather_zone.currentIndex() == 0
    assert page.wind_speed.value() == pytest.approx(2.5)
    # Without an atmosphere there is no wind to list.
    window.open_example(example_path("street"))
    assert ("scene", "wind") not in tree._keys


@pytest.mark.render
def test_the_wind_in_the_3d_view(window, qapp) -> None:
    viewport = window.viewport
    if not viewport.available:
        pytest.skip(viewport.note)
    window.open_example(example_path("open_air_stage"))

    def actors() -> int:
        qapp.processEvents()
        return len(viewport.plotter.renderer.actors)

    both = actors()
    window.wind_profile_action.setChecked(False)
    field = actors()
    window.wind_action.setChecked(False)
    none = actors()
    window.wind_profile_action.setChecked(True)
    mast = actors()
    # The arrows and the key in the corner; a mast's pole, curve, arrows and labels.
    assert field == none + 2 and mast == none + 5 and both == none + 6
    texts = [
        actor.GetText(k)
        for actor in viewport.plotter.renderer.actors.values()
        if hasattr(actor, "GetText")
        for k in range(4)
    ]
    assert "wind 5.0 m/s at 10 m" in texts


def test_weather_classes_form(window, qapp) -> None:
    window.open_example(example_path("open_air_stage"))
    window.select("scene", None)
    page = window.inspector.pages["scene"]
    form = page._weather_form
    document = window.document

    def long_term():
        return document.scene.long_term

    # Without classes the form shows no shares or periods.
    assert long_term() is None and not page.remove_weather_class.isEnabled()
    assert not form.isRowVisible(page.class_shares) and not form.isRowVisible(page.periods)

    # The first class holds all the time in the scene's weather, and the form shows it.
    page.add_weather_class.click()
    assert [c.name for c in long_term().classes] == ["class_1"]
    assert long_term().classes[0].share == 1.0 and document.undo_text() == "add weather class"
    assert page.weather_zone.currentData() == "class:0"
    assert page.weather_zone.currentText() == "weather class class_1"
    assert page.remove_weather_class.isEnabled() and form.isRowVisible(page.class_shares)
    assert page.periods.text() == (
        "Levels over a long term: 1 weather class over day 12 h, evening 4 h +5 dB, "
        "night 8 h +10 dB (L_den)"
    )
    shares = page.class_shares
    assert [shares.horizontalHeaderItem(k).text() for k in range(3)] == ["day", "evening", "night"]
    assert [shares.item(0, k).text() for k in range(3)] == ["100", "100", "100"]
    assert not shares.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable  # the only class
    assert page.weather.currentData() is None and page.weather.itemText(0) == "as the whole scene"
    assert page.wind_speed.value() == 5.0  # the scene's weather
    assert page.eddies.isEnabled() and page.eddies.itemText(0) == "as the whole scene"
    assert page.eddies.currentData() is None and form.isRowVisible(page.zone_temperature)

    # A second class: a clear night's inversion, for 60 % of the night.
    page.add_weather_class.click()
    assert page.weather_zone.currentData() == "class:1" and long_term().classes[1].share == 0.0
    page.weather.setCurrentIndex(page.weather.findData("clear_night"))
    assert long_term().classes[1].preset == "clear_night"
    assert page.log_term.value() == 0.5 and page.wind_speed.value() == 0.0
    assert document.scene.atmosphere.preset == "windy"  # the scene's weather stays
    page.class_shares.item(0, 2).setText("60")
    assert long_term().shares().tolist() == [[1.0, 1.0, 0.4], [0.0, 0.0, 0.6]]
    assert long_term().classes[1].share == {"day": 0.0, "evening": 0.0, "night": 0.6}
    assert page.class_shares.item(0, 2).text() == "60"
    # Its own eddies, and the ground's temperature on such a night.
    page.eddies.setCurrentIndex(page.eddies.findData("weak"))
    assert long_term().classes[1].turbulence.preset == "weak"
    assert document.scene.atmosphere.turbulence.preset == "moderate"
    page.zone_temperature.setValue(8.0)
    assert long_term().classes[1].temperature_c == 8.0
    assert document.undo_text() == "change the weather class"
    page.show_profile()  # the class's weather, as the engine resolves it
    assert page.profile_window.isVisible()
    page.profile_window.close()

    # The whole scene's form keeps its own eddies and shows the periods.
    page.weather_zone.setCurrentIndex(0)
    assert page.eddies.itemText(0) == "(none)" and page.eddies.currentData() == "moderate"
    assert not form.isRowVisible(page.class_shares) and form.isRowVisible(page.periods)
    assert "2 weather classes" in page.periods.text()

    # Removing a class gives its share to the others.
    page.weather_zone.setCurrentIndex(page.weather_zone.findData("class:0"))
    page.remove_weather_class.click()
    assert [c.name for c in long_term().classes] == ["class_2"]
    assert long_term().classes[0].share == 1.0 and page.weather_zone.currentData() is None
    document.undo()
    assert len(long_term().classes) == 2
    # A floor plan drawn over the scene leaves the outdoors and its classes behind.
    document.draw_floor_plan([[-10, -10], [10, -10], [10, 10], [-10, 10]], height=5.0)
    assert document.scene.long_term is None and document.scene.atmosphere is None


def test_a_run_over_a_long_term(window, qapp) -> None:
    from acoustic_spaces.gui.results_view import ROWS

    scene = Scene.model_validate(
        {
            "geometry": {"ground": {"extent": [[-40, -40], [40, 40]], "sky": 20}},
            "materials": {"field": {"impedance": "grass", "scattering": 0.0}},
            "default_material": "field",
            "sources": [{"name": "s", "position": [0, 0, 1.0]}],
            "listeners": [{"name": "l", "position": [20, 0, 1.5]}],
            "atmosphere": {"preset": "windy"},
            "simulation": {
                "ir_length_s": 0.3,
                "seed": 2,
                "early": {"method": "image_sources", "max_order": 1},
                "late": {"rays": 500},
            },
            "long_term": {
                "classes": [
                    {"name": "downwind", "share": 0.6},
                    {"name": "upwind", "share": 0.4, "wind": {"direction_deg": 180}},
                ]
            },
        }
    )
    window._attach(Document(scene))
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    result = window.result
    assert result is not None and result.long_term is not None
    results = window.results
    row = [name for name, _label, _template in ROWS].index("long_term")
    assert not results.table.isRowHidden(row) and results.table.item(row, 4).text() != "–"
    text = results.details.toPlainText()
    assert "Levels over a long term in 2 weather classes (downwind, upwind), rated as L_den" in text
    assert "s -> l: L_den" in text and "in the weather classes downwind" in text
    # The map gains the long-term level and the rating level.
    document = window.document
    document.edit("coarse map", lambda d: d.update(coverage={"spacing": 20.0, "rays": 100}))
    window.run_map()
    _wait(qapp, lambda: window.job is None)
    assert {"long_term", "rating_level"} <= set(window.coverage.parameters())


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
    box.activated.emit([value for value, _label in DIRECTIVITIES].index("cardioid"))
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
    assert not page.near_field.isEnabled()  # for arrays only

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
    # Its elements added up one by one near it.
    assert page.near_field.isEnabled() and not page.near_field.isChecked()
    page.near_field.setChecked(True)
    assert window.document.scene.sources[0].directivity.near_field
    assert window.document.room.directivities[0].near is not None
    window.undo()
    assert not window.document.scene.sources[0].directivity.near_field
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

    from PySide6.QtWidgets import QFileDialog, QInputDialog

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
    # From the menu: the file is picked in a dialog that opens in the scene's folder.
    folders = []
    picked = (str(libraries / "catt_materials.geo"), "")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: folders.append(a[2]) or picked
    )
    window.add_library_material_action.trigger()
    assert folders == [str(window.document.base_dir)]
    assert window.selection == ("material", "walls-2")


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

    # The export runs in the background, so the window keeps answering.
    window.write_results(tmp_path / "out")
    assert window.job is not None and not window.run_action.isEnabled()
    _wait(qapp, lambda: window.job is None)
    assert (tmp_path / "out" / "parameters.json").is_file()
    assert "impulse response files and parameters.json" in window.messages.toPlainText()
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
    assert 0 < np.abs(data).max() <= 10 ** (-1 / 20) * (1 + 1e-6)
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
    assert "facing yaw 30°, pitch 10°, roll 0°; 1 tracker message" in head.facing.text()
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


def test_stop_playback(window, qapp, tmp_path: Path, fake_sounddevice) -> None:
    _quick(window.document)
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    stops = []
    fake_sounddevice.play = lambda data, rate: fake_sounddevice.played.append((data, rate))
    fake_sounddevice.stop = lambda: stops.append(1)
    write_wav(tmp_path / "dry.wav", np.r_[1.0, np.zeros(5 * 16_000)], 16_000)  # 5 s
    results, stop = window.results, window.stop_action
    # Stop playback sits in the toolbar and the Simulation menu, apart from Cancel.
    assert stop in window.findChild(QToolBar, "main-toolbar").actions()
    menus = {menu.title(): menu for menu in window.findChildren(QMenu)}
    assert stop in menus["&Simulation"].actions()
    assert stop is not window.cancel_action and not stop.isEnabled()
    assert not window.now_playing.isVisible() and not window.stop_button.isVisible()
    assert window.play_file(tmp_path / "dry.wav")
    assert stop.isEnabled() and results.stop.isEnabled()
    assert fake_sounddevice.played[-1][0].shape[1] == 2  # binaural, though the scene is mono
    # The status bar says what plays, whichever panel is on show.
    window.docks["map"].raise_()
    assert window.now_playing.isVisible() and window.stop_button.isVisible()
    assert window.now_playing.text().startswith("Playing dry.wav from ")
    # What plays does not follow edits, and the status bar says so until they are undone.
    window.document.update_point("sources", 0, level_db=90.0)
    assert window.now_playing.text().endswith("(before the latest changes)")
    assert "does not follow them" in window.messages.toPlainText()
    window.undo()
    assert "before" not in window.now_playing.text()
    window.stop_button.click()
    assert stops and not stop.isEnabled() and not results.stop.isEnabled()
    assert not window.now_playing.isVisible() and not window.stop_button.isVisible()
    assert results.listen.isEnabled() and not window._play_timer.isActive()

    # Esc first stops placing a source, then the playback.
    assert window.play_file(tmp_path / "dry.wav")
    window.views.setCurrentWidget(window.plan)
    window.start_placing("sources")
    assert window.plan.canvas.placing is not None
    window._escape()
    assert window.plan.canvas.placing is None and stop.isEnabled() and len(stops) == 1
    window._escape()
    assert len(stops) == 2 and not stop.isEnabled() and not results.stop.isEnabled()


def test_playback_follows_the_pair_on_show(window, qapp, tmp_path: Path, fake_sounddevice) -> None:
    window.open_example(example_path("church"))
    _quick(window.document)
    window.run_simulation()
    _wait(qapp, lambda: window.job is None)
    fake_sounddevice.play = lambda data, rate: fake_sounddevice.played.append((data, rate))
    fake_sounddevice.stop = lambda: None
    write_wav(tmp_path / "dry.wav", np.r_[1.0, np.zeros(5 * 16_000)], 16_000)  # 5 s
    results = window.results
    assert window.play_file(tmp_path / "dry.wav")
    played = window._played
    # Another listener, picked in the scene: the sound goes on from where it was, no
    # louder than before (the difference between listeners stays audible).
    window.select("listeners", 2)
    assert window._played["pair"] == results.pair() == (played["pair"][0], "nave-back")
    assert window._played["start"] > 0 and window._played["gain"] <= played["gain"]
    assert len(fake_sounddevice.played) == 2 and "at nave-back" in window.now_playing.text()
    # Or picked in the results panel.
    results.listener.setCurrentIndex(0)
    assert window._played["pair"] == results.pair() == played["pair"]
    # Ticking head tracking turns the playback into head-tracked playback, as loud.
    gain = window._played["gain"]
    results.head.port.setValue(0)  # no head tracker
    results.tracking.setChecked(True)
    assert window._played["tracked"] and window._player is not None
    assert window._played["gain"] == gain
    window.stop_playing()
    results.tracking.setChecked(False)
    assert window._played is None and not results.stop.isEnabled()


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


def test_tune_in_the_editor(window, qapp, monkeypatch) -> None:
    from acoustic_spaces.engine import tuning

    timed = []

    def tune(room, *, threads=None, say=None, keep=True):
        assert not keep  # the window saves on OK only
        say("late stage: starting the GPU")
        timed.append(room.scene.name)
        late = [tuning.LateTiming(1.0, {"cpu": 2.0, "gpu": 1.0, "both": 0.8}, "both")]
        return tuning.Tuning(room.scene.name, "a GPU | 4 CPU threads", 500, late=late)

    monkeypatch.setattr(tuning, "tune", tune)
    shown = []
    monkeypatch.setattr(TuneDialog, "open", lambda dialog: shown.append(dialog))
    alerts = []
    monkeypatch.setattr(QApplication, "alert", lambda widget, msec=0: alerts.append(widget))

    def run_tune() -> TuneDialog:
        window.tune_action.trigger()
        _wait(qapp, lambda: window.job is None)
        return shown[-1]

    # Once in the Simulation menu, not in the toolbar.
    tune_action = window.tune_action
    menus = {menu.title(): menu for menu in window.findChildren(QMenu)}
    assert tune_action in menus["&Simulation"].actions()
    assert tune_action not in window.findChild(QToolBar, "main-toolbar").actions()
    tune_action.trigger()
    assert not tune_action.isEnabled() and not window.run_action.isEnabled()
    _wait(qapp, lambda: window.job is None)
    assert timed == [window.document.scene.name] and tune_action.isEnabled()
    assert alerts == [window]  # flashes in the taskbar if another window is in front
    assert "Tuned in" in window.messages.toPlainText()
    assert "late stage 1 s both." in window.messages.toPlainText()
    # The table, auto as this computer's default (ticked); a scene naming no device
    # follows the default already, so nothing asks about it.
    dialog = shown[0]
    texts = "".join(label.text() for label in dialog.findChildren(QLabel))
    assert "a GPU | 4 CPU threads" in texts
    assert dialog.make_default.text() == MAKE_DEFAULT and dialog.make_default.isChecked()
    assert dialog.follow is None
    # Cancel saves nothing, not even the timings.
    dialog.reject()
    assert not tuning.cache_file().exists() and tuning.default_device() == "cpu"
    assert "Tuning not saved" in window.messages.toPlainText()
    # OK saves the timings and the default.
    run_tune().accept()
    assert tuning.default_device() == "auto"
    assert tuning.late_choices()[1.0]["device"] == "both"
    assert tuning.late_choices()[1.0]["from"] == "tune"
    window.select("simulation")
    assert window.inspector.pages["simulation"].late_device.currentText() == "default (auto)"
    assert window.document.scene.simulation.late.device is None
    # A scene naming its own devices: OK can let it follow the default, and unticked
    # the default goes back to the CPU.
    window.document.edit("cpu", lambda data: data["simulation"]["late"].update(device="cpu"))
    dialog = run_tune()
    assert dialog.follow.text() == FOLLOW and dialog.follow.isChecked()
    dialog.make_default.setChecked(False)
    dialog.accept()
    assert tuning.default_device() == "cpu"
    assert window.document.scene.simulation.late.device is None
    assert window.document.undo_text() == "run on the default device"
    assert window.inspector.pages["simulation"].late_device.currentText() == "default (cpu)"

    # A scene that cannot run times the concert hall example; failures are said.
    def refuse(room, **_):
        timed.append(room.scene.name)
        raise ValueError("no GPU here")

    monkeypatch.setattr(tuning, "tune", refuse)
    window.document.edit("move", lambda data: data["sources"][0].update(position=[50, 0, 0]))
    window.run_tune()
    _wait(qapp, lambda: window.job is None)
    assert timed[-1].startswith("Concert hall")
    assert "The tuning failed: no GPU here" in window.messages.toPlainText()


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
    with pytest.raises(ImportError, match=r'pip install "acoustic-spaces\[gui\]"'):
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


@pytest.mark.parametrize("language", ["es", "ca", "it", "de", "nl"])
def test_choosing_the_language(window, english, monkeypatch, language) -> None:
    """View > Language keeps the choice for the next start; the editor stays as it is."""
    from acoustic_spaces import i18n

    menus = {menu.title(): menu for menu in window.menuBar().findChildren(QMenu)}
    assert window.language_menu.menuAction() in menus["&View"].actions()
    actions = window.language_actions
    assert set(actions) == {None, *i18n.LANGUAGES}
    assert actions[None].isChecked() and actions[None].text().startswith("System (")
    assert actions["es"].text() == "Español"  # each language in its own words
    assert actions["ca"].text() == "Català"
    assert actions["it"].text() == "Italiano"
    assert actions["de"].text() == "Deutsch"
    assert actions["nl"].text() == "Nederlands"
    assert all(action.statusTip() and action.toolTip() for action in actions.values())

    actions[language].trigger()
    assert i18n.saved_language() == language and actions[language].isChecked()
    assert i18n.language() == "en"  # from the next start
    # The variable the tests set wins over the choice, and the editor says so.
    assert "ACOUSTIC_SPACES_LANG is set to en" in window.messages.toPlainText()

    monkeypatch.delenv(i18n.ENV)
    assert i18n.chosen_language() == (language, "saved")
    actions[None].trigger()
    assert i18n.saved_language() is None and actions[None].isChecked()
    name = i18n.LANGUAGES[i18n.system_language()]
    assert window.messages.toPlainText().endswith(
        f"Language from the next start of the editor and the command line: {name}"
    )


@pytest.mark.parametrize("language", ["es", "ca", "it", "de", "nl"])
def test_qt_speaks_the_language_in_use(qapp, language) -> None:
    """Qt's own dialogs and buttons follow the language, from the translations Qt ships."""
    from PySide6.QtCore import QLibraryInfo

    from acoustic_spaces import i18n
    from acoustic_spaces.gui.app import install_qt_translation

    assert install_qt_translation(qapp) is None  # English: nothing to load
    i18n.set_language(language)
    translator = install_qt_translation(qapp)
    folder = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
    if (folder / f"qtbase_{language}.qm").exists():
        assert translator is not None
        assert qapp.removeTranslator(translator)
    else:
        assert translator is None


def _editor_in(qapp, monkeypatch, language: str):
    """A new editor window showing its text in ``language`` (en, es, ca, it, de, nl or pseudo)."""
    from acoustic_spaces import i18n
    from acoustic_spaces.gui.app import MainWindow

    monkeypatch.setenv(i18n.ENV, language)
    i18n.set_language(None)
    win = MainWindow()
    monkeypatch.setattr(win, "ask_save", lambda: QMessageBox.StandardButton.Discard)
    return win


def _close(qapp, win) -> None:
    win.close()
    win.deleteLater()
    qapp.processEvents()


def test_every_text_of_the_editor_is_translatable(qapp, english, monkeypatch) -> None:
    """In the pseudo language every text marked for translation shows as ⟦text⟧: menus,
    toolbar, panel titles, tabs, buttons, form labels, group titles, list entries and tips
    without brackets are text not yet marked."""

    from PySide6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QGroupBox,
        QTableWidget,
        QTabWidget,
        QWidget,
    )

    win = _editor_in(qapp, monkeypatch, "pseudo")
    try:
        win.open_example(example_path("studio"))
        scene = win.document.scene
        for kind, key in (
            ("scene", None),
            ("geometry", None),
            ("surface", 0),
            ("material", next(iter(scene.materials))),
            ("sources", 0),
            ("listeners", 0),
            ("simulation", None),
            ("outputs", None),
            ("coverage", None),
        ):
            win.select(kind, key)
        texts = []  # (what, text)
        for menu in win.menuBar().findChildren(QMenu):
            if menu is win.examples_menu:  # its entries are the groups' menus, below
                continue
            texts.append(("menu", menu.title()))
            if menu.parent() is win.examples_menu:  # the examples' own titles and descriptions
                continue
            for action in menu.actions():
                if not action.isSeparator() and menu is not win.language_menu:
                    texts.append(("menu entry", action.text()))
                texts += [("tip", action.toolTip()), ("status tip", action.statusTip())]
        for dock in win.docks.values():
            texts.append(("panel", dock.windowTitle()))
        roots = [win, *win.inspector.pages.values()]
        presets = win.inspector.pages["material"].preset  # the database's ids, as written
        for root in roots:
            for widget in root.findChildren(QWidget):
                if widget.objectName().startswith("qt_"):  # Qt's own, which Qt translates
                    continue
                texts.append((f"tip of {type(widget).__name__}", widget.toolTip()))
                if isinstance(widget, QLabel) and widget.buddy() is not None:
                    texts.append(("form label", widget.text()))
                elif isinstance(widget, QAbstractButton) and widget.parent() is not None:
                    if not isinstance(widget.parentWidget(), QToolBar):
                        texts.append(("button", widget.text()))
                elif isinstance(widget, QGroupBox):
                    texts.append(("group", widget.title()))
                elif isinstance(widget, QTabWidget):
                    texts += [("tab", widget.tabText(k)) for k in range(widget.count())]
                elif isinstance(widget, QComboBox) and widget is not presets:
                    for k in range(widget.count()):
                        # Names from the scene (materials, sources) show as they are.
                        if widget.itemText(k) != widget.itemData(k):
                            texts.append(("list entry", widget.itemText(k)))
                elif isinstance(widget, QTableWidget):
                    items = [widget.horizontalHeaderItem(c) for c in range(widget.columnCount())]
                    items += [widget.verticalHeaderItem(r) for r in range(widget.rowCount())]
                    texts += [("header tip", item.toolTip()) for item in items if item]
        # Words need marking; numbers, units, symbols and single letters (x, y, z) do not.
        unmarked = {
            (what, text)
            for what, text in texts
            if re.search(r"[A-Za-z]{3,}", text.replace("kHz", "")) and "⟦" not in text
        }
        assert not unmarked
    finally:
        _close(qapp, win)


@pytest.mark.parametrize(
    ("language", "words"),
    [
        (
            "es",
            {
                "menus": ["&Archivo", "&Editar", "&Ver", "&Simulación", "A&yuda"],
                "run": "&Calcular",
                "properties": "Propiedades",
                "new_room": "Recinto rectangular nuevo",
                "undo": "&Deshacer",
                "add_source": "añadir fuente",
                "late_methods": ["trazado de rayos estocástico", "estadístico"],
                "example": "Piso con una puerta",
                "rooms": "Recintos",
            },
        ),
        (
            "ca",
            {
                "menus": ["&Fitxer", "&Editar", "&Veure", "&Simulació", "&Ajuda"],
                "run": "&Calcular",
                "properties": "Propietats",
                "new_room": "Recinte rectangular nou",
                "undo": "&Desfer",
                "add_source": "afegir font",
                "late_methods": ["traçat de raigs estocàstic", "estadístic"],
                "example": "Pis amb una porta",
                "rooms": "Recintes",
            },
        ),
        (
            "it",
            {
                "menus": ["&File", "&Modifica", "&Visualizza", "&Simulazione", "&Aiuto"],
                "run": "&Calcola",
                "properties": "Proprietà",
                "new_room": "Nuovo ambiente rettangolare",
                "undo": "&Annulla",
                "add_source": "aggiungi sorgente",
                "late_methods": ["tracciamento di raggi stocastico", "statistico"],
                "example": "Appartamento con una porta",
                "rooms": "Ambienti",
            },
        ),
        (
            "de",
            {
                "menus": ["&Datei", "&Bearbeiten", "&Ansicht", "&Simulation", "&Hilfe"],
                "run": "&Berechnen",
                "properties": "Eigenschaften",
                "new_room": "Neuer Quaderraum",
                "undo": "&Rückgängig",
                "add_source": "Quelle hinzufügen",
                "late_methods": ["stochastische Strahlenverfolgung", "statistisch"],
                "example": "Wohnung mit einer Tür",
                "rooms": "Räume",
            },
        ),
        (
            "nl",
            {
                "menus": ["&Bestand", "Be&werken", "B&eeld", "&Simulatie", "&Help"],
                "run": "&Berekenen",
                "properties": "Eigenschappen",
                "new_room": "Nieuwe rechthoekige ruimte",
                "undo": "&Ongedaan maken",
                "add_source": "bron toevoegen",
                "late_methods": ["stochastische straaltracering", "statistisch"],
                "example": "Appartement met een deur",
                "rooms": "Ruimtes",
            },
        ),
    ],
)
def test_the_editor_in_translation(qapp, english, monkeypatch, language, words) -> None:
    """The editor shows its menus, panels and messages in the chosen language."""
    win = _editor_in(qapp, monkeypatch, language)
    try:
        titles = [action.text() for action in win.menuBar().actions()]
        assert titles == words["menus"]
        assert win.run_action.text() == words["run"]
        assert win.docks["inspector"].windowTitle() == words["properties"]
        win.new_box()
        assert words["new_room"] in win.messages.toPlainText()
        assert win.undo_action.text() == words["undo"]
        win.document.add_point("sources", (2.0, 2.0, 1.5))
        assert win.undo_action.text() == f"{words['undo']} {words['add_source']}"
        # Choices show in words; the scene keeps its values.
        page = win.inspector.pages["simulation"]
        assert [page.late_method.itemText(k) for k in range(2)] == words["late_methods"]
        assert page.late_method.itemData(0) == "stochastic_ray_tracing"
        rooms = win.examples_menu.actions()[0].menu()
        assert rooms.title() == words["rooms"]
        assert any(action.text().startswith(words["example"]) for action in rooms.actions())
        # One shortcut letter per entry in each menu.
        for menu in win.menuBar().findChildren(QMenu):
            letters = [
                re.search(r"&(\w)", action.text()).group(1).lower()
                for action in menu.actions()
                if "&" in action.text()
            ]
            assert len(letters) == len(set(letters)), menu.title()
    finally:
        _close(qapp, win)


def test_numbers_follow_the_regional_settings(qapp) -> None:
    """The editor writes and reads numbers as the computer's regional settings do."""
    from PySide6.QtCore import QLocale

    from acoustic_spaces.gui.numbers import (
        fmt,
        numbers_text,
        parse_number,
        parse_numbers,
        point_text,
    )

    before = QLocale()
    try:
        QLocale.setDefault(QLocale("es_ES"))
        assert fmt("{t:.2f} s, {n:,} rays, {name}", t=1.5, n=12000, name="hall_1.2") == (
            "1,50 s, 12.000 rays, hall_1.2"
        )
        assert point_text((1.0, 2.5, 3.25)) == "1,00; 2,50; 3,25"
        assert numbers_text([1.5, -2]) == "1,5; -2"
        assert parse_number("0,85") == parse_number("0.85") == 0.85
        assert parse_numbers("1,5; -2 3") == [1.5, -2.0, 3.0]
        QLocale.setDefault(QLocale("en_US"))
        assert fmt("{t:.2f} s, {n:,} rays", t=1.5, n=12000) == "1.50 s, 12,000 rays"
        assert point_text((1.0, 2.5)) == "1.00, 2.50"
        assert parse_numbers("1.5, -2 3") == [1.5, -2.0, 3.0]
    finally:
        QLocale.setDefault(before)


def test_insertion_loss_list(window) -> None:
    """The simulation page of an outdoor scene lists its buildings, barriers and
    vegetation to tick for their insertion loss; a part removed from the geometry page
    leaves the choice; a room has nothing to tick."""
    window.open_example(example_path("street"))
    window.select("simulation")
    page = window.inspector.pages["simulation"]
    assert not page.insertion_group.isHidden()

    def listed():
        return [page.insertion.item(r).text() for r in range(page.insertion.count())]

    def ticked():
        return [
            page.insertion.item(r).text()
            for r in range(page.insertion.count())
            if page.insertion.item(r).checkState() == Qt.CheckState.Checked
        ]

    names = listed()
    assert names == window.document.scene.geometry.part_names()
    assert names == ["houses", "offices", "noise_barrier"]
    assert ticked() == [] and window.document.scene.simulation.insertion_loss_of is None
    page.insertion.item(names.index("noise_barrier")).setCheckState(Qt.CheckState.Checked)
    assert window.document.scene.simulation.insertion_loss_of == ["noise_barrier"]
    page.insertion.item(0).setCheckState(Qt.CheckState.Checked)
    assert window.document.scene.simulation.insertion_loss_of == ["houses", "noise_barrier"]
    window.select("simulation")  # loaded again, the ticks are the scene's
    assert ticked() == ["houses", "noise_barrier"]
    # Removing the barrier on the geometry page drops it from the choice.
    window.select("geometry", None)
    geometry = window.inspector.pages["geometry"]
    geometry.parts["barriers"].setCurrentCell(0, 0)
    geometry._remove_part("barriers")
    assert window.document.scene.simulation.insertion_loss_of == ["houses"]
    window.select("simulation")
    assert listed() == ["houses", "offices"] and ticked() == ["houses"]
    page.insertion.item(0).setCheckState(Qt.CheckState.Unchecked)
    assert window.document.scene.simulation.insertion_loss_of is None
    window.open_example(example_path("shoebox"))
    window.select("simulation")
    assert page.insertion_group.isHidden()


def test_scene_tree_groups_ticks_and_eyes(window, qapp) -> None:
    """Sources and listeners sit in their groups; a tick leaves them out, an eye hides
    them in the 3D view and the plan, and the form sets both."""
    window.document.update_point("sources", 0, group="PA/Mains")
    tree = window.tree

    def entry(key):
        return next(item for item in tree._items() if tree._keys[item.data(0, 256)] == key)

    pa = entry(("group", ("sources", ("PA",))))
    assert entry(("group", ("sources", ("PA", "Mains")))).parent() is pa
    assert entry(("sources", 0)).parent().parent() is pa
    assert pa.checkState(0) == Qt.CheckState.Checked
    pa.setCheckState(0, Qt.CheckState.Unchecked)
    qapp.processEvents()
    assert not window.document.scene.sources[0].in_use
    assert entry(("sources", 0)).checkState(0) == Qt.CheckState.Unchecked
    window.undo()
    assert window.document.scene.sources[0].in_use
    name = window.document.scene.sources[0].name
    tree.toggle_shown(entry(("sources", 0)))
    qapp.processEvents()
    assert window.document.hidden == {("sources", name)}
    assert [kind for kind, _k, _item in window.plan.canvas._points()] == ["listeners"]
    tree.toggle_shown(entry(("", "sources")))  # none shown: shows them all
    qapp.processEvents()
    assert not window.document.hidden
    window.select("sources", 0)
    page = window.inspector.pages["sources"]
    assert page.group.currentText() == "PA/Mains" and page.in_use.isChecked()
    page.group.setCurrentText("Monitors")
    page._group()
    assert window.document.scene.sources[0].group == "Monitors"
    page.in_use.setChecked(False)
    assert not window.document.scene.sources[0].in_use
    assert window.document.undo_text() == "leave out source"
