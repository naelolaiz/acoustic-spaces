"""The scene as a tree: the room, its geometry and surfaces, materials, sources,
listeners, the wind outdoors, simulation settings, outputs and the coverage map.
Selecting an entry shows its form. Sources and listeners sit in their groups; each, and
each group, has a tick that uses it in simulations or leaves it out, and an eye that
shows or hides it in the 3D view and the plan. The wind and the masts of its profile
have eyes too, and the weather zones sit under the wind."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from acoustic_spaces.gui.inspector import label_text
from acoustic_spaces.gui.numbers import fmt, point_text
from acoustic_spaces.gui.viewport import material_colour, surface_materials
from acoustic_spaces.i18n import N_, _
from acoustic_spaces.room import UNASSIGNED, Room
from acoustic_spaces.viz import LISTENER_COLOUR, PROFILE_SPEED, SOURCE_COLOUR, WindMap

ROLE = Qt.ItemDataRole.UserRole
#: The column of the eyes that show or hide sources, listeners and the wind.
EYE = 2
#: The tip of each kind of entry, by (kind, key); a key of None stands for any.
TIPS = {
    ("scene", None): N_("The scene: its name, default material and air"),
    ("geometry", None): N_("The room's shape: a box, a floor plan or a model file"),
    ("geometry", "surfaces"): N_("The groups of faces that share a name in the model"),
    ("surface", None): N_("Select it to give it a material; its material and area follow"),
    ("", "materials"): N_("The materials of the scene, with the area each covers"),
    ("material", None): N_(
        "Select it to change its absorption, scattering and what it lets through"
    ),
    ("", "sources"): N_(
        "The sound sources; place more with Place source. Untick them to leave them all out "
        "of simulations"
    ),
    ("sources", None): N_(
        "Select it to change it; drag it in the 3D view or the plan to move it. Untick it to "
        "leave it out of simulations and maps"
    ),
    ("", "listeners"): N_(
        "The listening positions; place more with Place listener. Untick them to leave them "
        "all out of simulations"
    ),
    ("listeners", None): N_(
        "Select it to change it; drag it in the 3D view or the plan to move it. Untick it to "
        "leave it out of simulations and maps"
    ),
    ("group", None): N_(
        "A group of the scene's sources or listeners (their Group field). Untick it to leave "
        "them all out of simulations and maps"
    ),
    ("scene", "wind"): N_(
        "Outdoors, the wind: its eye shows or hides its arrows and the weather zones in the "
        "3D view and the plan; select it to change the wind under Properties"
    ),
    ("scene", "profile"): N_(
        "The masts of the wind's profile in the 3D view, arrows up them and a curve: how "
        "the wind grows with height. Their eye shows or hides them"
    ),
    ("simulation", None): N_(
        "How the early reflections and the late reverberation are "
        "computed, and the responses' length, sample rate and bands"
    ),
    ("outputs", None): N_(
        "The formats every response is written in: mono, binaural, Ambisonics or loudspeakers"
    ),
    ("coverage", None): N_(
        "Where the coverage map lies, how fine it is, and the background noise for the STI"
    ),
}
#: The tip of a weather zone's entry.
ZONE_TIP = N_(
    "A weather zone, a part of the scene with wind and temperature of its own, and its "
    "wind's speed at the scene's wind height; select it to change its weather under "
    "Properties"
)


def swatch(colour: str, faded: bool = False) -> QIcon:
    pixmap = QPixmap(12, 12)
    fill = QColor(colour)
    if faded:
        fill.setAlpha(70)
    pixmap.fill(fill)
    return QIcon(pixmap)


def eye(colour: QColor, open_: bool) -> QIcon:
    """An eye, open (shown) or struck through (hidden)."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QBrush(colour), 1.3))
    outline = QPainterPath()
    outline.moveTo(1.5, 8.0)
    outline.quadTo(8.0, 1.5, 14.5, 8.0)
    outline.quadTo(8.0, 14.5, 1.5, 8.0)
    painter.drawPath(outline)
    painter.setBrush(QBrush(colour))
    painter.drawEllipse(QPointF(8.0, 8.0), 2.2, 2.2)
    if not open_:
        painter.drawLine(QPointF(2.5, 13.5), QPointF(13.5, 2.5))
    painter.end()
    return QIcon(pixmap)


class SceneTree(QTreeWidget):
    """Emits ``selected(kind, key)`` when the user picks an entry,
    ``enabled_changed(kind, indices, enabled)`` when they tick or untick sources or
    listeners (``kind``), and ``shown_changed(kind, indices, shown)`` when they click an
    eye: ``kind`` is then ``wind`` or ``wind_profile`` (with no indices) for the wind's
    eyes. The wind's entries select the scene, keyed ``wind``, ``profile`` and
    ``zone:k`` for weather zone k."""

    selected = Signal(str, object)
    enabled_changed = Signal(str, object, bool)
    shown_changed = Signal(str, object, bool)

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels([_("Scene"), "", ""])
        self.setColumnCount(3)
        self.headerItem().setToolTip(
            EYE,
            _(
                "Shows or hides a source, a listener, a group or the wind in the 3D view and "
                "the plan"
            ),
        )
        self.setUniformRowHeights(True)
        self.itemSelectionChanged.connect(self._on_selection)
        self.itemChanged.connect(self._on_changed)
        self._quiet = False
        #: (kind, key) of every entry; items store their index here.
        self._keys: list[tuple[str, object]] = []
        #: The sources or listeners under an entry, by the entry's index: (kind, their
        #: indices, whether any is shown); the wind's entries have no indices.
        self._points: dict[int, tuple[str, list[int], bool]] = {}

    def _item(self, parent, text: str, kind: str, key=None, detail: str = "", icon=None):
        item = QTreeWidgetItem(parent, [text, detail])
        item.setData(0, ROLE, len(self._keys))
        tip = TIPS.get((kind, key)) or TIPS.get((kind, None))
        if tip:
            tip = _(tip)
            for column in range(2):
                item.setToolTip(column, tip)
        self._keys.append((kind, key))
        if icon is not None:
            item.setIcon(0, icon)
        return item

    def rebuild(
        self,
        room: Room,
        selection: tuple[str, object] | None,
        hidden: set[tuple[str, str]] = frozenset(),
        wind: tuple[bool, bool] = (True, True),
    ) -> None:
        """Fill the tree from ``room`` (and its scene) and select ``selection``;
        ``hidden`` holds the (kind, name) of the sources and listeners hidden in the
        views, and ``wind`` whether the views show the wind and its profile's masts."""
        scene = room.scene
        self._quiet = True
        self.clear()
        self._keys.clear()
        self._points.clear()
        areas = room.area_by_material()
        face_areas = room.mesh.face_areas()

        top = self._item(self, scene.name, "scene")
        geometry = self._item(top, _("Geometry"), "geometry", detail=scene.geometry.describe())
        materials = surface_materials(room)
        surfaces = self._item(
            geometry, _("Surfaces"), "geometry", "surfaces", detail=str(len(materials))
        )
        for label, material in enumerate(materials):
            area = float(face_areas[room.mesh.face_labels == label].sum())
            name = room.material_names[material] if material != UNASSIGNED else _("no material")
            self._item(
                surfaces,
                label_text(room.mesh.labels[label]),
                "surface",
                label,
                fmt(_("{material}, {area:,.1f} m²"), material=name, area=area),
                swatch(material_colour(room, material)),
            )
        group = self._item(top, _("Materials"), "", "materials", str(len(scene.materials)))
        for k, name in enumerate(room.material_names):
            self._item(
                group,
                name,
                "material",
                name,
                fmt(_("{area:,.1f} m²"), area=areas.get(name, 0.0)),
                swatch(material_colour(room, k)),
            )
        for kind, title, colour in (
            ("sources", _("Sources"), SOURCE_COLOUR),
            ("listeners", _("Listeners"), LISTENER_COLOUR),
        ):
            self._points_of(kind, title, colour, top, getattr(scene, kind), hidden)
        weathers = WindMap.of(room)
        if weathers is not None:
            self._wind_of(top, weathers, *wind)
        simulation = scene.simulation
        self._item(
            top,
            _("Simulation"),
            "simulation",
            detail=f"{simulation.early.method}, {simulation.late.method}",
        )
        self._item(
            top,
            _("Outputs"),
            "outputs",
            detail=", ".join(o.format for o in scene.outputs),
        )
        coverage = scene.coverage
        spacing = coverage.spacing if coverage else 1.0
        if coverage and coverage.surfaces:
            detail = fmt(
                # Translators: {surfaces} lists the surfaces the map lies on.
                _("{surfaces}, {spacing:g} m apart"),
                surfaces=", ".join(coverage.surfaces),
                spacing=spacing,
            )
        else:
            detail = fmt(_("audience, {spacing:g} m apart"), spacing=spacing)
        self._item(top, _("Coverage map"), "coverage", detail=detail)
        self.expandAll()
        surfaces.setExpanded(len(materials) <= 12)
        self.resizeColumnToContents(0)
        self.resizeColumnToContents(EYE)
        self.select(selection)
        self._quiet = False

    def _points_of(self, kind: str, title: str, colour: str, top, items, hidden) -> None:
        """The sources or listeners (``kind``) under ``top``, in their groups, with their
        ticks and eyes."""
        header = self._item(top, title, "", kind, str(len(items)))
        folders = {(): header}
        members: dict[tuple[str, ...], list[int]] = {(): []}
        faded = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        for index, item in enumerate(items):
            path = tuple(item.group.split("/")) if item.group else ()
            for depth in range(1, len(path) + 1):
                if path[:depth] not in folders:
                    folders[path[:depth]] = self._item(
                        folders[path[: depth - 1]], path[depth - 1], "group", (kind, path[:depth])
                    )
                    members[path[:depth]] = []
            for depth in range(len(path) + 1):
                members[path[:depth]].append(index)
            where = point_text(np.asarray(item.position))
            entry = self._item(
                folders[path], item.name, kind, index, f"({where})", swatch(colour, not item.in_use)
            )
            if not item.in_use:
                for column in range(2):
                    entry.setForeground(column, QBrush(faded))
        text = self.palette().color(QPalette.ColorRole.Text)
        icons = {True: eye(text, True), False: eye(faded, False)}
        for path, folder in folders.items():
            if path:
                folder.setText(1, str(len(members[path])))
            self._mark(folder, kind, members[path], items, hidden, icons)
        for entry in self._items_under(header):
            key = self._keys[entry.data(0, ROLE)]
            if key[0] == kind:
                self._mark(entry, kind, [key[1]], items, hidden, icons)

    def _wind_of(self, top, wind: WindMap, shown: bool, profile: bool) -> None:
        """Outdoors, the wind under ``top``: its entry, with its speed and the eye of its
        arrows and zones; the masts of its profile, with their eye; and each weather
        zone, with its wind's speed."""

        def speed(k: int) -> str:
            return fmt(_(PROFILE_SPEED), speed=wind.speeds[k], height=wind.height)

        entry = self._item(top, _("Wind"), "scene", "wind", speed(0))
        masts = self._item(entry, _("Wind profile"), "scene", "profile")
        text = self.palette().color(QPalette.ColorRole.Text)
        faded = self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
        entry.setIcon(EYE, eye(text, True) if shown else eye(faded, False))
        entry.setToolTip(
            EYE,
            _("Hide it in the 3D view and the plan")
            if shown
            else _("Show it in the 3D view and the plan"),
        )
        masts.setIcon(EYE, eye(text, True) if profile else eye(faded, False))
        masts.setToolTip(
            EYE, _("Hide them in the 3D view") if profile else _("Show them in the 3D view")
        )
        self._points[entry.data(0, ROLE)] = ("wind", [], shown)
        self._points[masts.data(0, ROLE)] = ("wind_profile", [], profile)
        for k in range(1, len(wind.names)):
            zone = self._item(entry, wind.names[k], "scene", f"zone:{k - 1}", speed(k))
            for column in range(2):
                zone.setToolTip(column, _(ZONE_TIP))

    def _mark(self, entry, kind: str, indices: list[int], items, hidden, icons) -> None:
        """Give ``entry`` the tick and the eye of the points at ``indices``."""
        if not indices:
            return
        used = sum(items[k].in_use for k in indices)
        shown = any((kind, items[k].name) not in hidden for k in indices)
        entry.setFlags(entry.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        state = (
            Qt.CheckState.Checked
            if used == len(indices)
            else Qt.CheckState.Unchecked
            if used == 0
            else Qt.CheckState.PartiallyChecked
        )
        entry.setCheckState(0, state)
        entry.setIcon(EYE, icons[shown])
        entry.setToolTip(
            EYE,
            _("Hide it in the 3D view and the plan")
            if shown
            else _("Show it in the 3D view and the plan"),
        )
        self._points[entry.data(0, ROLE)] = (kind, list(indices), shown)

    def _items_under(self, top):
        stack = [top]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _on_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """A tick changed: use the points under it, or leave them out."""
        if self._quiet or column != 0:
            return
        found = self._points.get(item.data(0, ROLE))
        if found is not None and found[1]:  # the wind's entries have no tick
            kind, indices, _shown = found
            self.enabled_changed.emit(kind, indices, item.checkState(0) != Qt.CheckState.Unchecked)

    def toggle_shown(self, item: QTreeWidgetItem) -> None:
        """Hide the points under ``item`` when any is shown, or show them all."""
        found = self._points.get(item.data(0, ROLE))
        if found is not None:
            kind, indices, shown = found
            self.shown_changed.emit(kind, indices, not shown)

    def mousePressEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        on_eye = self.columnAt(int(event.position().x())) == EYE
        if item is not None and on_eye and event.button() == Qt.MouseButton.LeftButton:
            self.toggle_shown(item)  # without selecting it
            event.accept()
            return
        super().mousePressEvent(event)

    def select(self, selection: tuple[str, object] | None) -> None:
        """Select the entry for ``selection`` without emitting ``selected``."""
        quiet, self._quiet = self._quiet, True
        self.clearSelection()
        if selection is not None:
            wanted = tuple(selection)
            for item in self._items():
                if self._keys[item.data(0, ROLE)] == wanted:
                    self.setCurrentItem(item)
                    self.scrollToItem(item)
                    break
        self._quiet = quiet

    def _items(self):
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _on_selection(self) -> None:
        if self._quiet:
            return
        item = self.currentItem()
        if item is None or not item.isSelected():
            items = self.selectedItems()
            if not items:
                return
            item = items[0]
        kind, key = self._keys[item.data(0, ROLE)]
        self.selected.emit(kind, key)


__all__ = ["EYE", "SceneTree", "eye", "swatch"]
