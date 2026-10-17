"""The scene as a tree: the room, its geometry and surfaces, materials, sources,
listeners, simulation settings, outputs and the coverage map. Selecting an entry shows
its form."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from acoustic_spaces.gui.inspector import label_text
from acoustic_spaces.gui.numbers import fmt, point_text
from acoustic_spaces.gui.viewport import material_colour, surface_materials
from acoustic_spaces.i18n import N_, _
from acoustic_spaces.room import UNASSIGNED, Room
from acoustic_spaces.viz import LISTENER_COLOUR, SOURCE_COLOUR

ROLE = Qt.ItemDataRole.UserRole
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
    ("", "sources"): N_("The sound sources; place more with Place source"),
    ("sources", None): N_("Select it to change it; drag it in the 3D view or the plan to move it"),
    ("", "listeners"): N_("The listening positions; place more with Place listener"),
    ("listeners", None): N_(
        "Select it to change it; drag it in the 3D view or the plan to move it"
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


def swatch(colour: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(colour))
    return QIcon(pixmap)


class SceneTree(QTreeWidget):
    """Emits ``selected(kind, key)`` when the user picks an entry."""

    selected = Signal(str, object)

    def __init__(self) -> None:
        super().__init__()
        self.setHeaderLabels([_("Scene"), ""])
        self.setColumnCount(2)
        self.setUniformRowHeights(True)
        self.itemSelectionChanged.connect(self._on_selection)
        self._quiet = False
        #: (kind, key) of every entry; items store their index here.
        self._keys: list[tuple[str, object]] = []

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

    def rebuild(self, room: Room, selection: tuple[str, object] | None) -> None:
        """Fill the tree from ``room`` (and its scene) and select ``selection``."""
        scene = room.scene
        self._quiet = True
        self.clear()
        self._keys.clear()
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
            items = getattr(scene, kind)
            group = self._item(top, title, "", kind, str(len(items)))
            for index, item in enumerate(items):
                where = point_text(np.asarray(item.position))
                self._item(group, item.name, kind, index, f"({where})", swatch(colour))
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
