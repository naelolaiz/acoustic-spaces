"""The scene as a tree: the room, its geometry and surfaces, materials, sources,
listeners, simulation settings, outputs and the coverage map. Selecting an entry shows
its form."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from acoustic_spaces.gui.inspector import label_text
from acoustic_spaces.gui.viewport import material_colour, surface_materials
from acoustic_spaces.room import UNASSIGNED, Room
from acoustic_spaces.viz import LISTENER_COLOUR, SOURCE_COLOUR

ROLE = Qt.ItemDataRole.UserRole


def swatch(colour: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(colour))
    return QIcon(pixmap)


class SceneTree(QTreeWidget):
    """Emits ``selected(kind, key)`` when the user picks an entry."""

    selected = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels(["Scene", ""])
        self.setColumnCount(2)
        self.setUniformRowHeights(True)
        self.itemSelectionChanged.connect(self._on_selection)
        self._quiet = False
        #: (kind, key) of every entry; items store their index here.
        self._keys: list[tuple[str, object]] = []

    def _item(self, parent, text: str, kind: str, key=None, detail: str = "", icon=None):
        item = QTreeWidgetItem(parent, [text, detail])
        item.setData(0, ROLE, len(self._keys))
        self._keys.append((kind, key))
        if icon is not None:
            item.setIcon(0, icon)
        return item

    def rebuild(self, room: Room, selection: tuple[str, object] | None) -> None:
        """Fill the tree from ``room`` (and its scene) and select ``selection``."""
        scene = room.scene
        self._quiet = True
        self.clear()
        self._keys.clear()
        areas = room.area_by_material()
        face_areas = room.mesh.face_areas()

        top = self._item(self, scene.name, "scene")
        geometry = self._item(top, "Geometry", "geometry", detail=scene.geometry.describe())
        materials = surface_materials(room)
        surfaces = self._item(
            geometry, "Surfaces", "geometry", "surfaces", detail=str(len(materials))
        )
        for label, material in enumerate(materials):
            area = float(face_areas[room.mesh.face_labels == label].sum())
            name = room.material_names[material] if material != UNASSIGNED else "no material"
            self._item(
                surfaces,
                label_text(room.mesh.labels[label]),
                "surface",
                label,
                f"{name}, {area:.1f} m²",
                swatch(material_colour(room, material)),
            )
        group = self._item(top, "Materials", "", "materials", str(len(scene.materials)))
        for k, name in enumerate(room.material_names):
            self._item(
                group,
                name,
                "material",
                name,
                f"{areas.get(name, 0.0):.1f} m²",
                swatch(material_colour(room, k)),
            )
        for kind, title, colour in (
            ("sources", "Sources", SOURCE_COLOUR),
            ("listeners", "Listeners", LISTENER_COLOUR),
        ):
            items = getattr(scene, kind)
            group = self._item(top, title, "", kind, str(len(items)))
            for index, item in enumerate(items):
                where = ", ".join(f"{v:.2f}" for v in np.asarray(item.position))
                self._item(group, item.name, kind, index, f"({where})", swatch(colour))
        simulation = scene.simulation
        self._item(
            top,
            "Simulation",
            "simulation",
            detail=f"{simulation.early.method}, {simulation.late.method}",
        )
        self._item(
            top,
            "Outputs",
            "outputs",
            detail=", ".join(o.format for o in scene.outputs),
        )
        coverage = scene.coverage
        where = ", ".join(coverage.surfaces) if coverage and coverage.surfaces else "audience"
        spacing = coverage.spacing if coverage else 1.0
        self._item(top, "Coverage map", "coverage", detail=f"{where}, {spacing:g} m apart")
        self.expandAll()
        surfaces.setExpanded(len(materials) <= 12)
        self.resizeColumnToContents(0)
        self.select(selection)
        self._quiet = False

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


__all__ = ["SceneTree", "swatch"]
