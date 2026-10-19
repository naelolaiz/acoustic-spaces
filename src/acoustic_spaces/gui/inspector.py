"""Forms for the selected item: the scene, its geometry, a surface, a material, a source
or listener, the simulation settings, the output formats and the coverage map.

Every form shows the current scene and turns each change into one
:meth:`Document.edit`; a change the scene schema refuses is reported and the form shows
the scene as it still is.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.engine import tuning
from acoustic_spaces.geometry.io import SUPPORTED_SUFFIXES, ifc_spaces
from acoustic_spaces.gui.document import Document, EditError
from acoustic_spaces.gui.levels import rich_lines
from acoustic_spaces.gui.numbers import fmt, number, numbers_text, parse_number, parse_numbers
from acoustic_spaces.gui.tips import band_headers, row_headers, share_tips
from acoustic_spaces.i18n import N_, _, ngettext
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.materials.bands import convert
from acoustic_spaces.room import UNASSIGNED, matching_name
from acoustic_spaces.scene.model import LAYOUT_NAMES

BAND_LABELS = [f"{f} Hz" if f < 1000 else f"{f // 1000} kHz" for f in OCTAVE_BANDS_HZ]
#: Column labels of the third-octave bands, short to fit 24 columns.
THIRD_OCTAVE_LABELS = [f"{f}" if f < 1000 else f"{f / 1000:g}k" for f in THIRD_OCTAVE_BANDS_HZ]
# The lists below pair a value as the scene file writes it with the text shown for it;
# the English text is the value itself, so that it reads as in the scene file.
#: The spectra a source's sound power can have.
SPECTRA = [
    # Translators: this and the entries below are choices as the scene file names them;
    # translate them as words, and keep the tips that name them in step.
    ("flat", N_("flat")),
    ("speech_male", N_("speech_male")),
    ("speech_female", N_("speech_female")),
]
DIRECTIVITIES = [
    ("omni", N_("omni")),
    ("subcardioid", N_("subcardioid")),
    ("cardioid", N_("cardioid")),
    ("supercardioid", N_("supercardioid")),
    ("hypercardioid", N_("hypercardioid")),
    ("figure8", N_("figure8")),
    ("lambert", N_("lambert")),
]
#: A source is a point, a line or a surface.
SHAPES = [("point", N_("point")), ("line", N_("line")), ("surface", N_("surface"))]
EARLY_METHODS = [("ray_tracing", N_("ray_tracing")), ("image_sources", N_("image_sources"))]
DIFFRACTION = [
    ("none", N_("none")),
    ("direct", N_("direct")),
    ("first_order", N_("first_order")),
    ("second_order", N_("second_order")),
]
LATE_METHODS = [
    ("stochastic_ray_tracing", N_("stochastic_ray_tracing")),
    ("statistical", N_("statistical")),
]
RECEPTIONS = [
    ("diffuse_rain", N_("diffuse_rain")),
    ("secondary_sources", N_("secondary_sources")),
    ("detector", N_("detector")),
]
# The first entry of a Run on list stands for this computer's default (see _set_device).
LATE_DEVICES = [
    (None, N_("default")),
    ("cpu", "cpu"),
    ("gpu", "gpu"),
    ("both", N_("both")),
    ("auto", N_("auto")),
]
WAVE_DEVICES = [(None, N_("default")), ("cpu", "cpu"), ("gpu", "gpu"), ("auto", N_("auto"))]
SCATTERING_METHODS = [("vector", N_("vector")), ("random", N_("random"))]
RAY_TREES = [("sah", "sah"), ("median", N_("median"))]
BANDS = [("octave", N_("octave")), ("third_octave", N_("third_octave"))]
#: The directivity list entry that asks for a SOFA or CSV file.
MEASURED = N_("Measured data or an array from a file…")
#: The tip of a source's directivity list, before the choice it shows.
DIRECTIVITY_TIP = N_(
    "How the source radiates round it: omni evenly; subcardioid to hypercardioid ever more "
    "to the front; figure8 to the front and the back; lambert as a vibrating surface, to "
    "the front only; or measured data (SOFA or CSV) or an array of sources from a file"
)
#: The tip of a source's sound power level.
LEVEL_TIP = N_(
    "The source's sound power level in all bands together (dB re 1 pW); Spectrum spreads "
    "it over the bands"
)
SAMPLE_RATES = [16_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_000]
#: Shown in the material list of a surface for "no assignment of its own".
FALLBACK = N_("(default material)")
#: Largest side or height of a box or floor plan room the forms take, in the room's
#: units: a kilometre in millimetres.
LARGEST = 1e6


def spin(low: float, high: float, decimals: int = 2, step: float = 0.1, suffix: str = ""):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setKeyboardTracking(False)
    if suffix:
        box.setSuffix(f" {suffix}")
    return box


def int_spin(low: int, high: int, step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setKeyboardTracking(False)
    box.setGroupSeparatorShown(True)
    return box


def label_text(label: tuple[str, ...]) -> str:
    return " / ".join(label) if label else _("(unnamed)")


class Page(QWidget):
    """One form. ``load`` fills it from the document; edits go through ``apply``."""

    #: An edit was refused; the message says why.
    failed = Signal(str)
    #: The page asks to select another item (after a rename or removal).
    select = Signal(str, object)

    def __init__(self, document: Callable[[], Document]) -> None:
        super().__init__()
        self._document = document
        self._loading = False
        self.key = None

    @property
    def document(self) -> Document:
        return self._document()

    def show_item(self, key) -> None:
        self.key = key
        self.reload()

    def reload(self) -> None:
        self._loading = True
        try:
            self.load()
        finally:
            self._loading = False

    def load(self) -> None:  # pragma: no cover - every page overrides it
        raise NotImplementedError

    def apply(self, description: str, change: Callable[[dict], None]) -> bool:
        if self._loading:
            return False
        try:
            self.document.edit(description, change)
            return True
        except EditError as exc:
            self.failed.emit(str(exc))
            self.reload()
            return False

    def run(self, action: Callable[[], object]) -> object:
        """Run a document method, reporting a refused edit."""
        if self._loading:
            return None
        try:
            return action()
        except EditError as exc:
            self.failed.emit(str(exc))
            self.reload()
            return None


class EmptyPage(Page):
    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        text = QLabel(
            _(
                "Select something in the scene tree or click it in the 3D view.\n\n"
                "Place a new source or listener with the toolbar buttons, then click a floor "
                "or wall in the 3D view."
            )
        )
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addStretch()

    def load(self) -> None:
        pass


class ScenePage(Page):
    """Name, default material, air, and the room's statistics and checks."""

    def __init__(self, document) -> None:
        super().__init__(document)
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setToolTip(_("The scene's name"))
        self.name.editingFinished.connect(self._name)
        form.addRow(_("Name"), self.name)
        self.default = QComboBox()
        self.default.setToolTip(
            _(
                "The material of every surface that has no material of its own; with none, "
                "such surfaces are a problem and the run refuses to start"
            )
        )
        self.default.currentIndexChanged.connect(self._default)
        form.addRow(_("Default material"), self.default)
        self.temperature = spin(-20, 50, 1, 1, "°C")
        self.temperature.setToolTip(
            _(
                "Temperature of the air (°C): it sets the speed of sound and, with the "
                "humidity, how much the air absorbs at high frequencies (ISO 9613-1)"
            )
        )
        self.humidity = spin(0, 100, 1, 5, "%")
        self.humidity.setToolTip(
            _(
                "Relative humidity of the air (%): with the temperature, it sets how much the "
                "air absorbs at high frequencies"
            )
        )
        self.pressure = spin(50, 110, 3, 0.5, "kPa")
        self.pressure.setToolTip(
            _("Air pressure (kPa), 101.325 at sea level: it changes the air absorption a little")
        )
        for box, field in (
            (self.temperature, "temperature_c"),
            (self.humidity, "humidity_percent"),
            (self.pressure, "pressure_kpa"),
        ):
            box.valueChanged.connect(lambda value, f=field: self._air(f, value))
        form.addRow(_("Air temperature"), self.temperature)
        form.addRow(_("Relative humidity"), self.humidity)
        form.addRow(_("Air pressure"), self.pressure)
        self.stats = QLabel()
        self.stats.setTextFormat(Qt.TextFormat.RichText)
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stats.setToolTip(
            _(
                "The room's size, and its reverberation time estimated from the volume and the "
                "absorption alone (the Eyring and Sabine formulas, with air absorption). A run "
                "gives the full answer"
            )
        )
        form.addRow(self.stats)

    def load(self) -> None:
        scene, room = self.document.scene, self.document.room
        self.name.setText(scene.name)
        self.default.clear()
        self.default.addItem(_("(none)"), None)
        for name in scene.materials:
            self.default.addItem(name, name)
        found = self.default.findData(scene.default_material)
        self.default.setCurrentIndex(max(found, 0))
        self.temperature.setValue(scene.air.temperature_c)
        self.humidity.setValue(scene.air.humidity_percent)
        self.pressure.setValue(scene.air.pressure_kpa)
        lines = [fmt(_("Surface area: {area:,.1f} m²"), area=room.report.surface_area_m2)]
        count = len(room.room_volumes)
        if count > 1:
            each = "; ".join(fmt("{v:,.1f}", v=v) for v in room.room_volumes)
            lines.append(
                fmt(
                    ngettext(
                        "Volume: {volume:,.1f} m³ in {count} room ({each} m³)",
                        "Volume: {volume:,.1f} m³ in {count} rooms ({each} m³)",
                        count,
                    ),
                    volume=room.volume_m3,
                    count=count,
                    each=each,
                )
            )
        elif room.volume_m3:
            lines.append(fmt(_("Volume: {volume:,.1f} m³"), volume=room.volume_m3))
        else:
            lines.append(_("Volume: model not closed"))
        # Outdoors there is no diffuse field for a reverberation time.
        if room.volume_m3 and room.outdoor is None:
            sabine = room.reverberation_time("sabine", air_absorption=True)
            eyring = room.reverberation_time("eyring", air_absorption=True)
            for band in (500, 1000):
                k = OCTAVE_BANDS_HZ.index(band)
                lines.append(
                    fmt(
                        _(
                            "Reverberation time at {band}: {eyring:.2f} s (Eyring), "
                            "{sabine:.2f} s (Sabine)"
                        ),
                        band=BAND_LABELS[k],
                        eyring=eyring[k],
                        sabine=sabine[k],
                    )
                )
        shown = [(line, "info") for line in lines]
        shown += [
            (_("Problem: {problem}").format(problem=problem), "error")
            for problem in room.problems()
        ]
        shown += [
            (_("Warning: {warning}").format(warning=warning), "warning")
            for warning in room.warnings()
        ]
        self.stats.setText(rich_lines(self.stats, shown))

    def _name(self) -> None:
        name = self.name.text().strip() or "untitled"  # a scene-file value
        if name != self.document.scene.name:
            self.apply(_("rename scene"), lambda data: data.update(name=name))

    def _default(self) -> None:
        if self.default.currentIndex() < 0:
            return
        value = self.default.currentData()

        def change(data: dict) -> None:
            if value is None:
                data.pop("default_material", None)
            else:
                data["default_material"] = value

        if value != self.document.scene.default_material:
            self.apply(_("change default material"), change)

    def _air(self, field: str, value: float) -> None:
        self.apply(_("change air"), lambda data: data.setdefault("air", {}).update({field: value}))


class GeometryPage(Page):
    """A box's dimensions, a floor plan's corners and height, a model file, or the
    ground, buildings and barriers of an outdoor scene."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        self.kind = QLabel()
        self.kind.setWordWrap(True)
        layout.addWidget(self.kind)

        self.box_group = QGroupBox(_("Box"))
        box_form = QFormLayout(self.box_group)
        self.box = [spin(0.1, LARGEST, 2, 0.5) for _ in range(3)]
        for k, (label, box, tip) in enumerate(
            zip(
                (_("Length (x)"), _("Width (y)"), _("Height (z)")),
                self.box,
                (
                    _("How long the room is along x, in the units the title shows"),
                    _("How wide the room is along y, in the units the title shows"),
                    _("How high the room is, floor to ceiling, in the units the title shows"),
                ),
                strict=True,
            )
        ):
            box.setToolTip(tip)
            box.valueChanged.connect(lambda value, k=k: self._box(k, value))
            box_form.addRow(label, box)
        layout.addWidget(self.box_group)

        self.plan_group = QGroupBox(_("Floor plan"))
        plan_layout = QVBoxLayout(self.plan_group)
        self.plan = QTableWidget(0, 2)
        self.plan.setHorizontalHeaderLabels(["x", "y"])
        for col, axis in enumerate("xy"):
            self.plan.horizontalHeaderItem(col).setToolTip(
                # Translators: {axis} is x or y.
                _(
                    "The {axis} of each corner, in the units the title shows; the corners go "
                    "round the room in turn"
                ).format(axis=axis)
            )
        self.plan.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.plan.itemChanged.connect(self._plan)
        plan_layout.addWidget(self.plan)
        buttons = QHBoxLayout()
        add, remove = QPushButton(_("Add corner")), QPushButton(_("Remove corner"))
        add.setToolTip(_("Add a corner halfway between the last corner and the first"))
        remove.setToolTip(_("Remove the selected corner, or the last one when none is selected"))
        add.clicked.connect(self._add_corner)
        remove.clicked.connect(self._remove_corner)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        plan_layout.addLayout(buttons)
        height_form = QFormLayout()
        self.height = spin(0.1, LARGEST, 2, 0.5)
        self.height.setToolTip(
            _("How high the room is, floor to ceiling, in the units the title shows")
        )
        self.height.valueChanged.connect(self._height)
        height_form.addRow(_("Height"), self.height)
        plan_layout.addLayout(height_form)
        layout.addWidget(self.plan_group)

        self.file_group = QGroupBox(_("Model file"))
        file_form = QFormLayout(self.file_group)
        self.file = QLabel()
        self.file.setWordWrap(True)
        self.file.setToolTip(_("The model file the room comes from"))
        choose = QPushButton(_("Choose another file…"))
        choose.setToolTip(
            _("Take the room from another model file; the materials and their assignments stay")
        )
        choose.clicked.connect(self._choose)
        self.units = QComboBox()
        # Item 0 keeps to the file's own units; the others are scene-file values.
        self.units.addItem(_("file default"))
        self.units.addItems(["m", "cm", "mm", "in", "ft"])
        self.units.setToolTip(
            _(
                "The unit the model's lengths are in; file default: the units a DXF, IFC or "
                "Collada file records, else metres"
            )
        )
        self.units.currentIndexChanged.connect(self._units)
        self.up_axis = QComboBox()
        self.up_axis.addItem(_("format default"))
        self.up_axis.addItems(["y", "z"])
        self.up_axis.setToolTip(
            _(
                "The axis that points up in the model; format default: y for glTF and SketchUp, "
                "the one a Collada file records, z for the others (Blender's OBJ export writes "
                "y up)"
            )
        )
        self.up_axis.currentIndexChanged.connect(self._up_axis)
        self.space = QComboBox()
        self.space.setToolTip(
            _(
                "The room of an IFC model to take on its own, from its space boundaries; "
                "every element reads walls as solids, with both sides"
            )
        )
        self.space.currentIndexChanged.connect(self._space)
        file_form.addRow(self.file)
        file_form.addRow(choose)
        file_form.addRow(_("Units"), self.units)
        file_form.addRow(_("Up axis"), self.up_axis)
        file_form.addRow(_("IFC room"), self.space)
        self.file_form = file_form
        layout.addWidget(self.file_group)

        self.ground_group = QGroupBox(_("Outdoors"))
        ground_layout = QVBoxLayout(self.ground_group)
        ground_form = QFormLayout()
        self.ground_height = spin(-LARGEST, LARGEST, 2, 0.5)
        self.ground_height.setToolTip(
            _("The height (z) of the flat ground, in the units the title shows")
        )
        self.ground_height.valueChanged.connect(lambda value: self._ground("height", value))
        self.margin = spin(0.1, LARGEST, 1, 5.0)
        self.margin.setToolTip(
            _(
                "How far the ground reaches past the buildings, barriers and model, unless "
                "its extent is set, and how high the sky is over the highest of them, "
                "unless the sky is set; in the units the title shows"
            )
        )
        self.margin.valueChanged.connect(lambda value: self._ground("margin", value))
        self.sky = spin(0.0, LARGEST, 1, 5.0)
        # Translators: the sky's height when it is not set: worked out from the scene.
        self.sky.setSpecialValueText(_("auto"))
        self.sky.setToolTip(
            _(
                "How high the open sky lies over the ground, in the units the title shows; "
                "auto: the margin over the highest building, barrier or model part. Sound "
                "that reaches the sky leaves the scene"
            )
        )
        self.sky.valueChanged.connect(lambda value: self._ground("sky", value or None))
        ground_form.addRow(_("Ground height"), self.ground_height)
        ground_form.addRow(_("Margin"), self.margin)
        ground_form.addRow(_("Sky"), self.sky)
        ground_layout.addLayout(ground_form)
        self.extent = QLabel()
        self.extent.setWordWrap(True)
        self.extent.setToolTip(
            _("Where the ground ends: past it, sound leaves the scene as if the ground went on")
        )
        ground_layout.addWidget(self.extent)
        self.parts = {}
        for kind, title, count, tips in (
            (
                "buildings",
                _("Buildings"),
                _("Corners"),
                (
                    _("The building's name; its surfaces are named after it"),
                    _("How high the building is, ground to roof, in the units the title shows"),
                    _("How many corners the building's footprint has (set in the scene file)"),
                    _("Remove the selected building, or the last one when none is selected"),
                ),
            ),
            (
                "barriers",
                _("Barriers"),
                _("Points"),
                (
                    _("The barrier's name, which its surfaces carry"),
                    _("How high the barrier is over the ground, in the units the title shows"),
                    _("How many points the barrier's path has (set in the scene file)"),
                    _("Remove the selected barrier, or the last one when none is selected"),
                ),
            ),
        ):
            ground_layout.addWidget(QLabel(title))
            table = QTableWidget(0, 3)
            table.setHorizontalHeaderLabels([_("Name"), _("Height"), count])
            for col in range(3):
                table.horizontalHeaderItem(col).setToolTip(tips[col])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.itemChanged.connect(lambda item, kind=kind: self._part_height(kind, item))
            ground_layout.addWidget(table)
            remove = QPushButton(
                _("Remove building") if kind == "buildings" else _("Remove barrier")
            )
            remove.setToolTip(tips[3])
            remove.clicked.connect(lambda _checked=False, kind=kind: self._remove_part(kind))
            ground_layout.addWidget(remove)
            self.parts[kind] = table
        layout.addWidget(self.ground_group)

        self.report = QLabel()
        self.report.setWordWrap(True)
        self.report.setToolTip(
            _(
                "A closed model has no gaps: each edge joins two faces. The volume and the "
                "reverberation estimates need one; View > Show leaks finds the gaps"
            )
        )
        layout.addWidget(self.report)
        layout.addStretch()

    def load(self) -> None:
        geometry, room = self.document.scene.geometry, self.document.room
        self.kind.setText(_("Geometry: {kind}").format(kind=geometry.describe()))
        self.box_group.setVisible(geometry.box is not None)
        self.plan_group.setVisible(geometry.floor_plan is not None)
        self.file_group.setVisible(geometry.file is not None)
        self.ground_group.setVisible(geometry.ground is not None)
        units = geometry.units or "m"  # a box or floor plan without units is in metres
        self.box_group.setTitle(_("Box ({units})").format(units=units))
        self.plan_group.setTitle(_("Floor plan ({units})").format(units=units))
        self.ground_group.setTitle(_("Outdoors ({units})").format(units=units))
        if geometry.ground is not None:
            self._load_outdoors(geometry, room)
        if geometry.box is not None:
            for box, value in zip(self.box, geometry.box, strict=True):
                box.setValue(value)
        if geometry.floor_plan is not None:
            self.plan.blockSignals(True)  # a half-filled table is no floor plan
            self.plan.setRowCount(len(geometry.floor_plan))
            for row, corner in enumerate(geometry.floor_plan):
                for col, value in enumerate(corner):
                    item = QTableWidgetItem(number(value))
                    # Translators: {axis} is x or y.
                    item.setToolTip(
                        _("The {axis} of corner {number}").format(axis="xy"[col], number=row + 1)
                    )
                    self.plan.setItem(row, col, item)
            self.plan.blockSignals(False)
            self.height.setValue(geometry.height)
        if geometry.file is not None:
            self.file.setText(str(geometry.file))
            self.units.setCurrentIndex(max(self.units.findText(geometry.units or ""), 0))
            self.up_axis.setCurrentIndex(max(self.up_axis.findText(geometry.up_axis or ""), 0))
            is_ifc = geometry.file.suffix.lower() == ".ifc"
            self.file_form.setRowVisible(self.space, is_ifc)
            self.space.clear()
            if is_ifc:
                self.space.addItem(_("every element"), None)
                try:
                    rooms = ifc_spaces(geometry.file)
                except (ImportError, ValueError, OSError):
                    rooms = []
                for key, shown in rooms:
                    self.space.addItem(shown, key)
                if geometry.space is not None and self.space.findData(geometry.space) < 0:
                    self.space.addItem(geometry.space, geometry.space)  # as the scene names it
                self.space.setCurrentIndex(max(self.space.findData(geometry.space), 0))
        r = room.report
        text = (
            _("{count:,} triangles, {area:,.1f} m², closed")
            if r.closed
            else _("{count:,} triangles, {area:,.1f} m², not closed")
        )
        self.report.setText(fmt(text, count=len(room.mesh.faces), area=r.surface_area_m2))

    def _box(self, axis: int, value: float) -> None:
        # Only the side that changed: the others show rounded to the spin boxes' decimals.
        def change(data: dict) -> None:
            data["geometry"]["box"][axis] = value

        self.apply(_("resize box"), change)

    def _plan(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        try:
            value = parse_number(item.text())
        except ValueError:
            self.failed.emit(_("floor plan corners need two numbers each"))
            self.reload()
            return
        row, col = item.row(), item.column()

        # Only the number that changed: the others show rounded.
        def change(data: dict) -> None:
            data["geometry"]["floor_plan"][row][col] = value

        self.apply(_("change floor plan"), change)

    def _add_corner(self) -> None:
        def change(data: dict) -> None:
            corners = data["geometry"]["floor_plan"]
            corners.append(list(np.mean([corners[-1], corners[0]], axis=0)))

        self.apply(_("add corner"), change)

    def _remove_corner(self) -> None:
        row = self.plan.currentRow()

        def change(data: dict) -> None:
            corners = data["geometry"]["floor_plan"]
            corners.pop(row if 0 <= row < len(corners) else -1)

        self.apply(_("remove corner"), change)

    def _height(self, value: float) -> None:
        self.apply(_("change height"), lambda data: data["geometry"].update(height=value))

    def _choose(self) -> None:
        if self._loading:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("Room model"),
            str(self.document.base_dir),
            # Translators: two file types of an open dialog, kept apart by ;; (keep it).
            _("Room models ({patterns});;All files (*)").format(
                patterns=" ".join("*" + s for s in SUPPORTED_SUFFIXES)
            ),
        )
        if path:

            def change(data: dict) -> None:
                data["geometry"]["file"] = path
                data["geometry"].pop("space", None)  # a room of the file it replaces

            self.apply(_("replace model"), change)

    def _space(self, index: int) -> None:
        if self._loading or index < 0:
            return
        key = self.space.itemData(index)

        def change(data: dict) -> None:
            if key is None:
                data["geometry"].pop("space", None)
            else:
                data["geometry"]["space"] = key

        self.apply(_("choose IFC room"), change)

    def _units(self, index: int) -> None:
        if index < 0:
            return
        units = self.units.itemText(index)

        def change(data: dict) -> None:
            if index == 0:
                data["geometry"].pop("units", None)
            else:
                data["geometry"]["units"] = units

        self.apply(_("change units"), change)

    def _up_axis(self, index: int) -> None:
        if index < 0:
            return
        axis = self.up_axis.itemText(index)

        def change(data: dict) -> None:
            if index == 0:
                data["geometry"].pop("up_axis", None)
            else:
                data["geometry"]["up_axis"] = axis

        self.apply(_("change up axis"), change)

    def _load_outdoors(self, geometry, room) -> None:
        ground = geometry.ground
        self.ground_height.setValue(ground.height)
        self.margin.setValue(ground.margin)
        self.sky.setValue(ground.sky or 0.0)
        if room.outdoor is not None:
            x0, y0, x1, y1 = room.outdoor.extent
            self.extent.setText(
                fmt(
                    _("Ground from ({x0:g}, {y0:g}) to ({x1:g}, {y1:g}) m, sky {sky:g} m over it"),
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    sky=room.outdoor.sky - room.outdoor.ground,
                )
            )
        for kind, names, items in (
            ("buildings", geometry.building_names(), geometry.buildings or []),
            ("barriers", geometry.barrier_names(), geometry.barriers or []),
        ):
            table = self.parts[kind]
            table.blockSignals(True)
            table.setRowCount(len(items))
            for row, (name, item) in enumerate(zip(names, items, strict=True)):
                points = item.footprint if kind == "buildings" else item.path
                for col, text in enumerate((name, number(item.height), str(len(points)))):
                    cell = QTableWidgetItem(text)
                    cell.setToolTip(table.horizontalHeaderItem(col).toolTip())
                    if col != 1:
                        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, col, cell)
            table.blockSignals(False)

    def _ground(self, field: str, value: float | None) -> None:
        def change(data: dict) -> None:
            ground = data["geometry"]["ground"]
            if value is None:
                ground.pop(field, None)
            else:
                ground[field] = value

        self.apply(_("change ground"), change)

    def _part_height(self, kind: str, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 1:
            return
        try:
            value = parse_number(item.text())
        except ValueError:
            self.failed.emit(_("a height is one number"))
            self.reload()
            return
        row = item.row()

        def change(data: dict) -> None:
            data["geometry"][kind][row]["height"] = value

        self.apply(_("change height"), change)

    def _remove_part(self, kind: str) -> None:
        if not self.parts[kind].rowCount():
            return
        row = self.parts[kind].currentRow()

        def change(data: dict) -> None:
            items = data["geometry"][kind]
            items.pop(row if 0 <= row < len(items) else -1)
            if not items:
                del data["geometry"][kind]

        self.apply(_("remove building") if kind == "buildings" else _("remove barrier"), change)


class SurfacePage(Page):
    """A group of faces that share a name: its area and its material."""

    def __init__(self, document) -> None:
        super().__init__(document)
        form = QFormLayout(self)
        self.name = QLabel()
        self.name.setWordWrap(True)
        self.name.setToolTip(
            _(
                "The names this group of faces has in the model (group, layer or object), "
                "which material assignments match"
            )
        )
        form.addRow(_("Surface"), self.name)
        self.area = QLabel()
        self.area.setToolTip(_("The surface's area (m²)"))
        form.addRow(_("Area"), self.area)
        self.material = QComboBox()
        self.material.setToolTip(
            _(
                "The material of the surface's first name; (default material) leaves it to the "
                "scene's default material"
            )
        )
        self.material.currentIndexChanged.connect(self._material)
        form.addRow(_("Material"), self.material)
        self.note = QLabel()
        self.note.setWordWrap(True)
        form.addRow(self.note)

    def _label(self) -> tuple[str, ...]:
        return self.document.room.mesh.labels[self.key]

    def load(self) -> None:
        room, scene = self.document.room, self.document.scene
        label = self._label()
        faces = room.mesh.face_labels == self.key
        self.name.setText(label_text(label))
        self.area.setText(fmt(_("{area:,.1f} m²"), area=float(room.mesh.face_areas()[faces].sum())))
        self.material.clear()
        self.material.addItem(_(FALLBACK), None)
        for name in scene.materials:
            self.material.addItem(name, name)
        key = matching_name(label, scene.assignments)
        own = scene.assignments[key] if key is not None else None
        self.material.setCurrentIndex(max(self.material.findData(own), 0))
        self.material.setEnabled(bool(label))
        index = int(room.face_materials[np.flatnonzero(faces)[0]]) if faces.any() else UNASSIGNED
        used = room.material_names[index] if index != UNASSIGNED else _("none")
        if label and len(label) > 1:
            text = _(
                "Now: {material}. Choosing a material assigns it to the name “{name}”, which "
                "other surfaces may share."
            )
        elif label:
            text = _("Now: {material}. Choosing a material assigns it to the name “{name}”.")
        else:
            text = _("Now: {material}. This surface has no name; it takes the default material.")
        self.note.setText(text.format(material=used, name=label[0] if label else ""))

    def _material(self) -> None:
        label = self._label()
        if not label or self.material.currentIndex() < 0:
            return
        choice = self.material.currentData()
        # The first name, as the note says, even when a later one (``walls``, which every
        # wall of a floor plan has) carries the assignment in use: changing that one
        # would change every surface that has it.
        name = label[0]
        self.run(lambda: self.document.assign(name, choice))


#: The material table's rows: absorption, scattering, and the sound reduction index.
_MATERIAL_ROWS = [N_("Absorption"), N_("Scattering"), N_("Sound reduction (dB)")]
#: What the rows of the material table mean, for their tips.
_MATERIAL_TIPS = [
    N_(
        "Absorption coefficient (0 to 1): the share of the sound energy the surface takes in "
        "at each reflection; 0 reflects everything, 1 takes in everything"
    ),
    N_(
        "Scattering coefficient (0 to 1): the share of the reflected energy that leaves in "
        "random directions rather than as from a mirror; 0 for a smooth surface"
    ),
    N_(
        "Sound reduction index R (dB): how much quieter the sound is on the other side of "
        "the surface, 30 to 60 dB for walls. Typed only when it lets part of the sound "
        "through; a blank band takes its neighbour's value"
    ),
]
#: What a surface of the material lets through: nothing, part of the sound (with a
#: sound reduction index per band), or everything (an opening).
PASSING = [N_("nothing"), N_("part of the sound"), N_("everything (an opening)")]
#: The sound reduction index (dB) a material starts from when it begins to let sound through.
DEFAULT_REDUCTION_DB = 40.0


def _reduction_text(passing: float) -> str:
    """How the material table shows a transmission: as R in dB, blank for none."""
    return "" if passing <= 0.0 else number(max(0.0, -10.0 * math.log10(passing)), ".1f")


def _nearest_filled(values: list[float | None]) -> list[float]:
    """``values`` with each None taking the value of the nearest band that has one."""
    have = [k for k, v in enumerate(values) if v is not None]
    return [
        v if v is not None else values[min(have, key=lambda j: abs(j - k))]
        for k, v in enumerate(values)
    ]


#: Fields a material keeps when its coefficients or database entry change.
_KEPT = ("transmission", "transmission_loss", "audience")


def _without_transmission(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if k not in ("transmission", "transmission_loss")}


class MaterialPage(Page):
    """A material: its name, the database entry it comes from, its coefficients, and
    what it lets through: nothing, part of the sound (the sound reduction index R per
    band) or everything (an opening)."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setToolTip(
            _(
                "The material's name; renaming it renames it in the assignments and the default "
                "material too"
            )
        )
        self.name.editingFinished.connect(self._rename)
        form.addRow(_("Name"), self.name)
        self.preset = QComboBox()
        self.preset.setToolTip(
            _(
                "The entry of the built-in material database the absorption comes from; type to "
                "search. Picking one replaces the absorption and keeps the scattering"
            )
        )
        self.preset.setEditable(True)
        self.preset.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        entries = builtin_database().entries("absorption")
        self.preset.addItems([""] + [e.id for e in entries])
        completer = QCompleter([e.id for e in entries])
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.preset.setCompleter(completer)
        self.preset.activated.connect(self._preset)
        form.addRow(_("Database entry"), self.preset)
        self.passing = QComboBox()
        self.passing.addItems([_(text) for text in PASSING])
        self.passing.setToolTip(
            _(
                "Sound that passes through a surface goes on into the room behind it, a "
                "partition's at its sound reduction index R (30 to 60 dB for walls), an "
                "opening's all of it. What passes counts as absorbed, so the absorption is "
                "at least 10^(-R/10), and an opening's is 1."
            )
        )
        self.passing.activated.connect(self._passing)
        form.addRow(_("Lets through"), self.passing)
        self.audience = QCheckBox(_("Audience seating"))
        self.audience.setToolTip(
            _(
                "Sound that reaches a listener low over these seats loses more in the 500 Hz "
                "to 2 kHz bands, as it passes heads and seat backs (grazing incidence; the "
                "fit of Jurkiewicz et al., 2025)"
            )
        )
        self.audience.toggled.connect(self._audience)
        form.addRow(self.audience)
        layout.addLayout(form)
        self.table = QTableWidget(len(_MATERIAL_ROWS), len(OCTAVE_BANDS_HZ))
        self.table.setHorizontalHeaderLabels(BAND_LABELS)
        self.table.setVerticalHeaderLabels([_(text) for text in _MATERIAL_ROWS])
        row_headers(self.table, [_(text) for text in _MATERIAL_TIPS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.setMaximumHeight(165)
        self.table.itemChanged.connect(self._coefficients)
        layout.addWidget(self.table)
        self.source = QLabel()
        self.source.setWordWrap(True)
        self.source.setToolTip(_("Where the coefficients come from"))
        layout.addWidget(self.source)
        self.usage = QLabel()
        self.usage.setWordWrap(True)
        self.usage.setToolTip(_("How much of the room's surface has this material"))
        layout.addWidget(self.usage)
        remove = QPushButton(_("Remove material"))
        remove.setToolTip(
            _(
                "Remove this material and the assignments that use it; its surfaces take the "
                "default material"
            )
        )
        remove.clicked.connect(self._remove)
        layout.addWidget(remove)
        layout.addStretch()

    def load(self) -> None:
        if self.key not in self.document.scene.materials:
            return
        material = self.document.scene.materials[self.key]
        self.name.setText(self.key)
        self.preset.setCurrentText(material.preset or "")
        # Third-octave columns when any list has third-octave values.
        bands = self._bands(material)
        third = bands == "third_octave"
        labels = THIRD_OCTAVE_LABELS if third else BAND_LABELS
        self.table.setColumnCount(len(labels))
        self.table.setHorizontalHeaderLabels(labels)
        band_headers(self.table, bands)
        mode = QHeaderView.ResizeMode.ResizeToContents if third else QHeaderView.ResizeMode.Stretch
        self.table.horizontalHeader().setSectionResizeMode(mode)
        for row, values in enumerate(material.coefficients(bands)):
            for col, value in enumerate(values):
                item = QTableWidgetItem(number(value, ".2f"))
                item.setToolTip(_(_MATERIAL_TIPS[row]))
                self.table.setItem(row, col, item)
        passing = material.transmission_coefficients(bands)
        kind = self._kind(passing)
        self.passing.setCurrentIndex(kind)
        self.audience.setChecked(material.audience)
        for col, value in enumerate(passing or [0.0] * len(labels)):
            item = QTableWidgetItem(_reduction_text(value))
            item.setToolTip(_(_MATERIAL_TIPS[2]))
            if kind != 1:  # set by the choice above, not typed
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(2, col, item)
        # Translators: {source} is where the coefficients come from, as the material
        # database (in English) or the scene file says it.
        self.source.setText(
            _("Source: {source}").format(source=material.source)
            if material.source
            else _("Source: values given in the scene")
        )
        area = self.document.room.area_by_material().get(self.key, 0.0)
        self.usage.setText(fmt(_("Covers {area:,.1f} m² of the room."), area=area))

    @staticmethod
    def _kind(passing: list[float] | None) -> int:
        """The index in :data:`PASSING` of what a transmission lets through."""
        if passing is None or max(passing) <= 0.0:
            return 0
        return 2 if min(passing) >= 1.0 else 1

    def _fields(self) -> dict:
        material = self.document.scene.materials[self.key]
        return material.model_dump(mode="json", exclude_none=True)

    @staticmethod
    def _bands(material) -> str:
        """The bands the table shows ``material`` in: third octaves when it has any."""
        third = len(THIRD_OCTAVE_BANDS_HZ)
        passing = material.transmission_loss or material.transmission or []
        lengths = (len(material.absorption), len(material.scattering), len(passing))
        return "third_octave" if third in lengths else "octave"

    def _rename(self) -> None:
        new = self.name.text().strip()
        if new and new != self.key:
            old = self.key
            if self.run(lambda: self.document.rename_material(old, new) or True):
                self.select.emit("material", new)

    def _remove(self) -> None:
        if self.run(lambda: self.document.remove_material(self.key) or True):
            self.select.emit("scene", None)

    def _preset(self) -> None:
        entry = self.preset.currentText().strip()
        if not entry:
            return
        fields = {"preset": entry, "scattering": self.document.scene.materials[self.key].scattering}
        fields |= {k: v for k, v in self._fields().items() if k in _KEPT}
        self.run(lambda: self.document.set_material(self.key, fields))

    def _audience(self, checked: bool) -> None:
        if self._loading:
            return
        fields = {**self._fields(), "audience": checked}
        self.run(lambda: self.document.set_material(self.key, fields))

    def _passing(self, kind: int) -> None:
        material = self.document.scene.materials[self.key]
        if kind == self._kind(material.transmission_coefficients()):
            return
        fields = _without_transmission(self._fields())
        if kind == 1:
            fields["transmission_loss"] = DEFAULT_REDUCTION_DB
        elif kind == 2:  # all passes, so all counts as absorbed
            fields = {"transmission": 1.0, "scattering": fields["scattering"]}
        self.run(lambda: self.document.set_material(self.key, fields))

    def _coefficients(self) -> None:
        if self._loading:
            return
        columns = range(self.table.columnCount())
        try:
            rows = [[parse_number(self.table.item(r, c).text()) for c in columns] for r in range(2)]
        except (AttributeError, ValueError):
            self.failed.emit(_("coefficients are numbers from 0 to 1"))
            self.reload()
            return
        material = self.document.scene.materials[self.key]
        bands = self._bands(material)
        passing = material.transmission_coefficients(bands)
        reduction = None
        if self._kind(passing) == 1:
            texts = [self.table.item(2, c).text().strip() for c in columns]
            try:
                typed = [parse_number(t) if t else None for t in texts]
            except ValueError:
                self.failed.emit(_("the sound reduction is in dB"))
                self.reload()
                return
            # A blank band takes its neighbour's value; all blank, nothing passes.
            reduction = _nearest_filled(typed) if any(v is not None for v in typed) else None
        fields = self._fields()
        shown = [convert(fields[k], bands) for k in ("absorption", "scattering")]
        same = rows == [[round(v, 2) for v in row] for row in shown]
        if reduction is not None:
            now = list(map(_reduction_text, passing))
            if same and [_reduction_text(10.0 ** (-r / 10.0)) for r in reduction] == now:
                return
        elif same and self._kind(passing) != 1:
            return
        if same:  # only what passes changed: the rest stays as it is given
            fields = _without_transmission(fields)
        else:
            # Scene-file text, in English as the scene file is.
            origin = fields.get("source") or "values given in the scene"
            kept = {k: fields[k] for k in _KEPT if k in fields}
            fields = {
                "absorption": rows[0],
                "scattering": rows[1],
                "source": origin if origin.startswith("edited") else f"edited from {origin}",
                **kept,
            }
        if reduction is not None:
            fields.pop("transmission", None)
            fields["transmission_loss"] = reduction
        self.run(lambda: self.document.set_material(self.key, fields))


def _fill(combo: QComboBox, choices) -> None:
    """Fill a list with (value, text) pairs: the text shows, translated, and the value
    is the item's data."""
    for value, text in choices:
        combo.addItem(_(text), value)


def _choose(combo: QComboBox, value) -> None:
    """Show the entry whose value is ``value``."""
    combo.setCurrentIndex(max(combo.findData(value), 0))


def _directivity_label(spec) -> str:
    """How the directivity list shows a source's setting that is not a pattern name."""
    if spec.file is not None:
        return _("file: {name}").format(name=spec.file.name)
    if spec.array is not None:
        count = len(spec.array.layout())
        return ngettext("array of {count} element", "array of {count} elements", count).format(
            count=count
        )
    # Translators: {patterns} lists a pattern name per band, such as omni, cardioid.
    return _("per band: {patterns}").format(patterns=", ".join(str(v) for v in spec.pattern))


class PointPage(Page):
    """A source or listener: name, position and orientation; sources also have a
    directivity and a sound power level."""

    def __init__(self, document, kind: str) -> None:
        super().__init__(document)
        self.kind = kind
        source = kind == "sources"
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.name.setToolTip(
            _("The source's name, as the results and the exported files show it")
            if source
            else _("The listener's name, as the results and the exported files show it")
        )
        self.name.editingFinished.connect(self._name)
        form.addRow(_("Name"), self.name)
        self.position = [spin(-1e4, 1e4, 2, 0.1, "m") for _ in range(3)]
        tips = (
            (
                _(
                    "Where the source is along x (m). Dragging it in the 3D view or the plan "
                    "moves it too"
                ),
                _(
                    "Where the source is along y (m). Dragging it in the 3D view or the plan "
                    "moves it too"
                ),
                _(
                    "Where the source is along z (m); z is up. Dragging it in the 3D view or "
                    "the plan moves it too"
                ),
            )
            if source
            else (
                _(
                    "Where the listener is along x (m). Dragging it in the 3D view or the plan "
                    "moves it too"
                ),
                _(
                    "Where the listener is along y (m). Dragging it in the 3D view or the plan "
                    "moves it too"
                ),
                _(
                    "Where the listener is along z (m); z is up. Dragging it in the 3D view or "
                    "the plan moves it too"
                ),
            )
        )
        for k, (axis, box, tip) in enumerate(zip("xyz", self.position, tips, strict=True)):
            box.setToolTip(tip)
            box.valueChanged.connect(lambda value, k=k: self._position(k, value))
            form.addRow(axis, box)
        self.angles = [spin(-360, 360, 1, 5, "°") for _ in range(3)]
        tips = (
            (
                _(
                    "Turns the source left, from facing +x towards +y (degrees). The way it "
                    "faces sets where the source radiates most with a directional pattern"
                ),
                _(
                    "Tilts the source to look up (degrees). The way it faces sets where the "
                    "source radiates most with a directional pattern"
                ),
                _(
                    "Tilts the source to the right, round the way it faces (degrees). The way "
                    "it faces sets where the source radiates most with a directional pattern"
                ),
            )
            if source
            else (
                _(
                    "Turns the listener left, from facing +x towards +y (degrees). The way it "
                    "faces sets what the binaural, Ambisonics and loudspeaker outputs hear as "
                    "the front"
                ),
                _(
                    "Tilts the listener to look up (degrees). The way it faces sets what the "
                    "binaural, Ambisonics and loudspeaker outputs hear as the front"
                ),
                _(
                    "Tilts the listener to the right, round the way it faces (degrees). The way "
                    "it faces sets what the binaural, Ambisonics and loudspeaker outputs hear "
                    "as the front"
                ),
            )
        )
        for angle, label, box, tip in zip(
            ("yaw", "pitch", "roll"),
            (_("Yaw"), _("Pitch"), _("Roll")),
            self.angles,
            tips,
            strict=True,
        ):
            box.setToolTip(tip)
            box.valueChanged.connect(lambda value, a=angle: self._orientation(a, value))
            form.addRow(label, box)
        self.directivity = QComboBox()
        self.directivity.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.directivity.setMinimumContentsLength(12)
        self.directivity.setToolTip(_(DIRECTIVITY_TIP))
        self.directivity.activated.connect(self._directivity)
        self.directivity_index = QLabel()
        self.directivity_index.setWordWrap(True)
        self.directivity_index.setToolTip(
            _(
                "How much louder the source is straight ahead than on average over all "
                "directions, per octave band (dB): 0 dB for omni, about 5 dB for a cardioid"
            )
        )
        self.pattern = QPushButton(_("Show the pattern"))
        self.pattern.setToolTip(
            _(
                "Open the source's polar patterns in every band; the selected source also shows "
                "its balloon in the 3D view"
            )
        )
        self.pattern.clicked.connect(self.show_pattern)
        self.pattern_window = None
        self.level = spin(0, 200, 1, 1, "dB")
        self.level.setToolTip(_(LEVEL_TIP))
        self.level.valueChanged.connect(self._level)
        self.spectrum = QComboBox()
        _fill(self.spectrum, SPECTRA)
        self.spectrum.setToolTip(
            _(
                "How the sound power spreads over the bands: evenly over the octave bands, or "
                "as male or female speech (IEC 60268-16), which the STI then weights for"
            )
        )
        self.spectrum.activated.connect(self._spectrum)
        self.gain = spin(-100, 60, 1, 1, "dB")
        self.gain.setToolTip(
            _("Gain of the source's feed (dB): on its level and its impulse responses")
        )
        self.gain.valueChanged.connect(lambda value: self._update(gain_db=value))
        self.eq = QLineEdit()
        self.eq.setPlaceholderText(_("none"))
        self.eq.setToolTip(
            _(
                "Equalizer gain per band (dB): 8 octave values from 63 Hz to 8 kHz, or 24 "
                "third-octave values from 50 Hz to 10 kHz, separated by commas"
            )
            if QLocale().decimalPoint() != ","
            else _(
                "Equalizer gain per band (dB): 8 octave values from 63 Hz to 8 kHz, or 24 "
                "third-octave values from 50 Hz to 10 kHz, separated by semicolons"
            )
        )
        self.eq.editingFinished.connect(self._eq)
        self.delay = spin(0, 10_000, 2, 1, "ms")
        self.delay.setToolTip(
            _(
                "The source starts this long after the others: its arrivals and responses "
                "come that much later"
            )
        )
        self.delay.valueChanged.connect(lambda value: self._update(delay_ms=value))
        self.shape = QComboBox()
        _fill(self.shape, SHAPES)
        self.shape.setToolTip(
            _(
                "A line runs along the source's left axis and a surface spans its left and up "
                "axes, both centred on its position: uncorrelated point sources share the power"
            )
        )
        self.shape.activated.connect(self._shape)
        self.extent = [spin(0.01, 1000, 2, 0.5, "m") for _ in range(2)]
        self.extent[0].setToolTip(
            _(
                "How long a line source is, or how wide a surface source is, along the source's "
                "left axis (m)"
            )
        )
        self.extent[1].setToolTip(_("How high a surface source is, along the source's up axis (m)"))
        for box in self.extent:
            box.valueChanged.connect(lambda _: self._extent())
        self.spacing = spin(0.05, 100, 2, 0.1, "m")
        self.spacing.setToolTip(
            _(
                "The most the point sources of a line or surface are apart (m); closer is finer "
                "and slower"
            )
        )
        self.spacing.valueChanged.connect(lambda value: self._update(spacing_m=value))
        self.elements = QLabel()
        self.elements.setWordWrap(True)
        self.elements.setToolTip(
            _("The line or surface is made of these point sources; they share its sound power")
        )
        self._form = form
        if kind == "sources":
            form.addRow(_("Shape"), self.shape)
            form.addRow(_("Length"), self.extent[0])
            form.addRow(_("Height"), self.extent[1])
            form.addRow(_("Spacing"), self.spacing)
            form.addRow(self.elements)
            form.addRow(_("Directivity"), self.directivity)
            form.addRow(self.directivity_index)
            form.addRow(self.pattern)
            form.addRow(_("Sound power"), self.level)
            form.addRow(_("Spectrum"), self.spectrum)
            form.addRow(_("Gain"), self.gain)
            form.addRow(_("Equalizer"), self.eq)
            form.addRow(_("Delay"), self.delay)
        remove = QPushButton(_("Remove source") if source else _("Remove listener"))
        remove.setToolTip(
            _("Remove this source from the scene (Delete in the tree or a view)")
            if source
            else _("Remove this listener from the scene (Delete in the tree or a view)")
        )
        remove.clicked.connect(self._remove)
        form.addRow(remove)
        hint = QLabel(
            _(
                "Yaw turns left from +x towards +y, pitch looks up and roll tilts to the right. "
                "In the 3D view the long arrow points forward and the short one up."
            )
        )
        hint.setWordWrap(True)
        form.addRow(hint)

    def _item(self):
        return getattr(self.document.scene, self.kind)[self.key]

    def load(self) -> None:
        items = getattr(self.document.scene, self.kind)
        if not 0 <= self.key < len(items):
            return
        item = self._item()
        self.name.setText(item.name)
        for box, value in zip(self.position, item.position, strict=True):
            box.setValue(value)
        o = item.orientation
        for box, value in zip(self.angles, (o.yaw, o.pitch, o.roll), strict=True):
            box.setValue(value)
        if self.kind == "sources":
            self.directivity.clear()
            _fill(self.directivity, DIRECTIVITIES)
            if isinstance(item.directivity, str):
                _choose(self.directivity, item.directivity)
            else:
                self.directivity.addItem(_directivity_label(item.directivity), "")
                self.directivity.setCurrentIndex(len(DIRECTIVITIES))
            self.directivity.addItem(_(MEASURED), MEASURED)
            # Translators: the directivity list's tip, then the choice it shows.
            tip = _("{tip}. Now: {choice}").format(
                tip=_(DIRECTIVITY_TIP), choice=self.directivity.currentText()
            )
            self.directivity.setToolTip(tip)
            self._form.labelForField(self.directivity).setToolTip(tip)
            directivity = self.document.room.directivities[self.key]
            index = directivity.in_bands("octave").index()
            values = numbers_text(index, ".1f")
            if directivity.extrapolated:
                bands = numbers_text(directivity.extrapolated)
                text = _(
                    "Directivity index: {values} dB (63 Hz to 8 kHz); {bands} Hz copied from "
                    "the nearest octave"
                ).format(values=values, bands=bands)
            else:
                text = _("Directivity index: {values} dB (63 Hz to 8 kHz)").format(values=values)
            self.directivity_index.setText(text)
            per_band = isinstance(item.level_db, list)
            total = 10.0 * np.log10(np.sum(10.0 ** (np.array(item.level_db) / 10.0)))
            self.level.setValue(float(total))
            self.level.setEnabled(not per_band)
            tip = _(LEVEL_TIP)
            if per_band:
                # Translators: {tip} is the sound power's tip.
                tip = _("{tip}. Given per band in the scene file; this is their sum").format(
                    tip=tip
                )
            self.level.setToolTip(tip)
            self._form.labelForField(self.level).setToolTip(tip)
            _choose(self.spectrum, item.spectrum)
            self.spectrum.setEnabled(not per_band)
            self.gain.setValue(item.gain_db)
            self.eq.setText("" if item.eq_db is None else numbers_text(item.eq_db))
            self.delay.setValue(item.delay_ms)
            shape = "line" if item.line_m else "surface" if item.surface_m else "point"
            _choose(self.shape, shape)
            if item.line_m is not None:
                self.extent[0].setValue(item.line_m)
            elif item.surface_m is not None:
                for box, value in zip(self.extent, item.surface_m, strict=True):
                    box.setValue(value)
            self.spacing.setValue(item.spacing_m)
            self._form.labelForField(self.extent[0]).setText(
                _("Width") if shape == "surface" else _("Length")
            )
            self._form.setRowVisible(self.extent[0], shape != "point")
            self._form.setRowVisible(self.extent[1], shape == "surface")
            self._form.setRowVisible(self.spacing, shape != "point")
            count = len(item.elements())
            self.elements.setText(
                ngettext(
                    "{count} uncorrelated point source", "{count} uncorrelated point sources", count
                ).format(count=count)
                if count > 1
                else ""
            )
            self._form.setRowVisible(self.elements, count > 1)
            if self.pattern_window is not None and self.pattern_window.isVisible():
                self.show_pattern()

    def _remove(self) -> None:
        if self.run(lambda: self.document.remove_point(self.kind, self.key) or True):
            self.select.emit("scene", None)

    def _update(self, **fields) -> None:
        if not self._loading:
            self.run(lambda: self.document.update_point(self.kind, self.key, **fields))

    def _name(self) -> None:
        name = self.name.text().strip()
        if name and name != self._item().name:
            self._update(name=name)

    def _position(self, axis: int, value: float) -> None:
        # One value at a time (the orientation too): the other boxes show theirs rounded,
        # and a dragged point is placed to the micrometre.
        if not self._loading:
            position = list(self._item().position)
            position[axis] = value
            self._update(position=position)

    def _orientation(self, angle: str, value: float) -> None:
        if not self._loading:
            orientation = self._item().orientation.model_dump()
            orientation[angle] = value
            self._update(orientation=orientation)

    def _directivity(self, index: int) -> None:
        value = self.directivity.itemData(index)
        if value in dict(DIRECTIVITIES):
            self._update(directivity=value)
        elif value == MEASURED:
            path, _filter = QFileDialog.getOpenFileName(
                self,
                _("Source directivity"),
                str(self.document.base_dir),
                _("Directivity data or arrays (*.sofa *.csv *.yaml *.yml);;All files (*)"),
            )
            if path:
                self._update(directivity={"file": path})
            else:
                self.reload()

    def show_pattern(self) -> None:
        """Open (or refresh) a window with the polar patterns of the source."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from acoustic_spaces.viz import plot_directivity

        directivity = self.document.room.directivities[self.key]
        if self.pattern_window is None:
            self.pattern_window = QWidget(self, Qt.WindowType.Window)
            QVBoxLayout(self.pattern_window)
            self.pattern_window.resize(1000, 520)
        layout = self.pattern_window.layout()
        while layout.count():
            layout.takeAt(0).widget().deleteLater()
        layout.addWidget(FigureCanvasQTAgg(plot_directivity(directivity)))
        self.pattern_window.setWindowTitle(
            _("Directivity of {name}").format(name=self._item().name)
        )
        self.pattern_window.show()
        self.pattern_window.raise_()

    def _shape(self, index: int) -> None:
        shape = SHAPES[index][0]
        width, height = (box.value() for box in self.extent)
        if shape == "line":
            self._update(line_m=width, surface_m=None)
        elif shape == "surface":
            self._update(line_m=None, surface_m=[width, height])
        else:
            self._update(line_m=None, surface_m=None)

    def _extent(self) -> None:
        if not self._loading:
            self._shape(self.shape.currentIndex())

    def _level(self, value: float) -> None:
        self._update(level_db=value)

    def _spectrum(self, index: int) -> None:
        self._update(spectrum=self.spectrum.itemData(index))

    def _eq(self) -> None:
        try:
            values = parse_numbers(self.eq.text()) or None
        except ValueError:
            self.failed.emit(_("the equalizer takes numbers (dB), one per band"))
            self.reload()
            return
        if values != self._item().eq_db:
            self._update(eq_db=values)


def _set_device(combo: QComboBox, device: str | None) -> None:
    """Show a scene's device in a Run on box; its first item, this computer's default,
    stands for none."""
    # Translators: {device} is a device such as cpu or gpu, as the scene file writes it.
    combo.setItemText(0, _("default ({device})").format(device=tuning.default_device()))
    _choose(combo, device)


def _device(combo: QComboBox) -> str | None:
    """The device a Run on box names, None for this computer's default."""
    return combo.currentData()


class SimulationPage(Page):
    """The early and late stages, scattering, the wave solver, the response length,
    sample rate, bands and seed."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        early = QGroupBox(_("Early reflections"))
        early_form = QFormLayout(early)
        self.early_method = QComboBox()
        _fill(self.early_method, EARLY_METHODS)
        self.early_method.setToolTip(
            _(
                "How the early reflection paths are found: ray_tracing (rays find them; the "
                "cost grows with the rays, not the surfaces) or image_sources (every sequence "
                "of surfaces; complete, but it grows as the surfaces to the power of the "
                "order). The paths found are exact either way"
            )
        )
        self.max_order = int_spin(0, 20)
        self.max_order.setToolTip(
            _(
                "The most reflections an early path has; the late reverberation takes the "
                "higher orders"
            )
        )
        self.early_rays = int_spin(1, 10_000_000, 1000)
        self.early_rays.setToolTip(
            _(
                "Rays that look for early reflection paths (ray tracing only); more find more "
                "of the higher-order paths"
            )
        )
        self.diffraction = QComboBox()
        _fill(self.diffraction, DIFFRACTION)
        self.diffraction.setToolTip(
            _(
                "Sound bending round edges (pillars, balcony fronts, screens): none, of the "
                "direct sound only, of the direct sound and first-order reflections, or also "
                "of the direct sound over two edges in turn"
            )
        )
        self.coherent = QComboBox()
        _fill(self.coherent, [(None, N_("none"))] + [(f, str(f)) for f in (63, 125, 250, 500)])
        self.coherent.setToolTip(
            _(
                "The highest octave band (Hz) in which the early paths add in pressure, with "
                "the phase of every reflection, rather than in energy; the wave solver's bands "
                "take their place when it runs up to this band or higher"
            )
        )
        early_form.addRow(_("Method"), self.early_method)
        early_form.addRow(_("Highest order"), self.max_order)
        early_form.addRow(_("Rays"), self.early_rays)
        early_form.addRow(_("Diffraction"), self.diffraction)
        early_form.addRow(_("In pressure up to"), self.coherent)
        layout.addWidget(early)

        late = QGroupBox(_("Late reverberation"))
        late_form = QFormLayout(late)
        self.late_method = QComboBox()
        _fill(self.late_method, LATE_METHODS)
        self.late_method.setToolTip(
            _(
                "How the late reverberation is found: stochastic_ray_tracing (rays per band "
                "reflect and scatter off the surfaces) or statistical (a smooth decay at the "
                "Eyring reverberation time, by Barron's revised theory; fast, for simple rooms)"
            )
        )
        self.late_rays = int_spin(1, 100_000_000, 10_000)
        self.late_rays.setToolTip(
            _("Rays per band for the late reverberation; more give smoother decays and take longer")
        )
        self.reception = QComboBox()
        _fill(self.reception, RECEPTIONS)
        self.reception.setToolTip(
            _(
                "How reflected energy reaches the listeners: diffuse_rain (the scattered part "
                "of each reflection radiates to every listener it can see, the rest counts when "
                "a ray crosses a detector sphere), secondary_sources (all of it radiates, as in "
                "ODEON: smoother, but it treats specular sound as diffuse) or detector (the "
                "sphere only, classic ray tracing)"
            )
        )
        self.directions = int_spin(0, 1024, 16)
        self.directions.setToolTip(
            _(
                "How many directions the late energy is sorted into, every 10 ms, so that the "
                "binaural, Ambisonics and loudspeaker outputs hear where the reverberation "
                "comes from"
            )
        )
        self.splitting = QCheckBox(_("Split strong rays, thin out weak ones"))
        self.splitting.setToolTip(
            _(
                "Split rays that carry much more energy than usual and thin out those that "
                "carry much less; the expected result stays the same, and late tails stay "
                "smooth where few rays survive (about 1.5 times the run time)"
            )
        )
        self.late_device = QComboBox()
        _fill(self.late_device, LATE_DEVICES)
        self.late_device.setToolTip(
            _(
                "Where it traces the rays: this computer's default (the CPU, or auto once "
                "Simulation > Tune CPU and GPU… made it so), the CPU, a GPU through WebGPU "
                "(needs the gpu extra), both sharing the rays (with a GPU, results can vary "
                "slightly from run to run), or auto: the fastest of those on this computer, as "
                "Tune CPU and GPU… or a short trial found"
            )
        )
        late_form.addRow(_("Method"), self.late_method)
        late_form.addRow(_("Rays per band"), self.late_rays)
        late_form.addRow(_("Reception"), self.reception)
        late_form.addRow(_("Arrival directions"), self.directions)
        late_form.addRow(self.splitting)
        late_form.addRow(_("Run on"), self.late_device)
        layout.addWidget(late)

        scattering = QGroupBox(_("Scattering"))
        scattering_form = QFormLayout(scattering)
        self.scattering_method = QComboBox()
        _fill(self.scattering_method, SCATTERING_METHODS)
        self.scattering_method.setToolTip(
            _(
                "Where a reflected ray goes: vector (between the specular direction and a "
                "random Lambert one, weighted by the scattering coefficient, as in ODEON) or "
                "random (specular, or with the scattering coefficient as probability Lambert)"
            )
        )
        self.from_size = QCheckBox(_("Add the scattering of each surface's size"))
        self.from_size.setToolTip(
            _(
                "Small surfaces, distant ones and grazing reflections reflect less "
                "specularly at low frequencies (Christensen and Rindel, 2005)"
            )
        )
        self.by_band = QCheckBox(_("Trace each band with rays of its own"))
        self.by_band.setToolTip(
            _(
                "Vector-based scattering: where the bands scatter differently, each band "
                "follows its own scattering (exact). Off: one ray for all bands, turned by "
                "their mean scattering (faster)"
            )
        )
        scattering_form.addRow(_("Direction"), self.scattering_method)
        scattering_form.addRow(self.from_size)
        scattering_form.addRow(self.by_band)
        layout.addWidget(scattering)

        wave = QGroupBox(_("Low frequencies"))
        wave_form = QFormLayout(wave)
        self.wave = QCheckBox(_("Wave solver"))
        self.wave.setToolTip(
            _(
                "Compute the lowest octave bands from the sound field itself (finite differences "
                "on a grid) instead of rays: room modes, and sound bending round everything"
            )
        )
        self.wave_band = QComboBox()
        self.wave_band.addItems(["63", "125", "250", "500"])
        self.wave_band.setToolTip(
            _("The highest band it computes; each costs about 16 times the one below")
        )
        self.wave_device = QComboBox()
        _fill(self.wave_device, WAVE_DEVICES)
        self.wave_device.setToolTip(
            _(
                "Where it steps its grid: this computer's default (the CPU, or auto once "
                "Simulation > Tune CPU and GPU… made it so), the CPU, a GPU through WebGPU "
                "(needs the gpu extra), or auto: a hardware GPU when there is one, unless "
                "Tune CPU and GPU… found the CPU faster for the run's length"
            )
        )
        wave_form.addRow(self.wave)
        wave_form.addRow(_("Up to band (Hz)"), self.wave_band)
        wave_form.addRow(_("Run on"), self.wave_device)
        layout.addWidget(wave)

        performance = QGroupBox(_("Performance"))
        performance_form = QFormLayout(performance)
        cores = os.cpu_count() or 1
        self.threads = int_spin(0, 1024)
        self.threads.setSpecialValueText(_("All cores ({count})").format(count=cores))
        self.threads.setToolTip(
            _(
                "Worker threads for ray tracing and the wave solver on the CPU. Fewer leave the "
                "computer free for other work; the results stay the same"
            )
        )
        self.ray_tree = QComboBox()
        _fill(self.ray_tree, RAY_TREES)
        self.ray_tree.setToolTip(
            _(
                "How the tree that finds the surface a ray meets is split: by the surface area "
                "heuristic (sah, several times faster in detailed models) or at the middle face "
                "(median, as before; for comparison). The results stay the same"
            )
        )
        performance_form.addRow(_("CPU threads"), self.threads)
        performance_form.addRow(_("Ray search tree"), self.ray_tree)
        layout.addWidget(performance)

        response = QGroupBox(_("Response"))
        response_form = QFormLayout(response)
        self.length = spin(0.05, 60, 2, 0.5, "s")
        self.length.setToolTip(
            _(
                "How long each impulse response lasts (s); longer than the reverberation time "
                "keeps the whole decay"
            )
        )
        self.sample_rate = QComboBox()
        self.sample_rate.addItems([str(r) for r in SAMPLE_RATES])
        self.sample_rate.setToolTip(
            _(
                "Samples per second of the impulse responses (Hz); match the recordings you "
                "will play through them"
            )
        )
        self.bands = QComboBox()
        _fill(self.bands, BANDS)
        self.bands.setToolTip(
            _(
                "The bands it runs in: 8 octave bands, or 24 third-octave bands for finer "
                "detail in the response (about three times the work). Room parameters are "
                "given in octave bands either way"
            )
        )
        self.auto_transition = QCheckBox(_("Automatic"))
        self.auto_transition.setToolTip(
            _(
                "Ticked, the early paths hand over to the late reverberation by reflection "
                "order (with the statistical method, at the mixing time: the square root of "
                "the volume, in ms). Unticked, at the time below"
            )
        )
        self.transition = spin(1, 10_000, 0, 10, "ms")
        self.transition.setToolTip(
            _(
                "The early paths end this long after the direct sound (ms); the late "
                "reverberation takes the rest"
            )
        )
        self.random_seed = QCheckBox(_("New random seed every run"))
        self.random_seed.setToolTip(
            _(
                "Ticked, every run draws new random rays, so results vary a little from run to "
                "run. Unticked, the seed below makes runs repeat, exactly with the late rays "
                "on the CPU (on a GPU they can vary slightly)"
            )
        )
        self.seed = int_spin(0, 2**31 - 1)
        self.seed.setToolTip(
            _(
                "The random seed: the same seed and settings give the same results (on a GPU "
                "they can vary slightly)"
            )
        )
        response_form.addRow(_("Length"), self.length)
        response_form.addRow(_("Sample rate (Hz)"), self.sample_rate)
        response_form.addRow(_("Bands"), self.bands)
        response_form.addRow(_("Early to late"), self.auto_transition)
        response_form.addRow("", self.transition)
        response_form.addRow(_("Seed"), self.random_seed)
        response_form.addRow("", self.seed)
        layout.addWidget(response)
        layout.addStretch()

        for combo, key in (
            (self.early_method, "early.method"),
            (self.diffraction, "early.diffraction"),
            (self.late_method, "late.method"),
            (self.reception, "late.reception"),
            (self.scattering_method, "scattering.method"),
            (self.ray_tree, "performance.ray_tree"),
            (self.bands, "bands"),
        ):
            combo.currentIndexChanged.connect(
                lambda index, combo=combo, key=key: self._set(key, combo.itemData(index))
            )
        for combo, key in ((self.wave_band, "wave.max_band_hz"), (self.sample_rate, "sample_rate")):
            combo.currentTextChanged.connect(lambda text, key=key: self._set(key, int(text)))
        self.coherent.currentIndexChanged.connect(
            lambda index: self._set("early.coherent_max_band_hz", self.coherent.itemData(index))
        )
        for combo, key in ((self.late_device, "late.device"), (self.wave_device, "wave.device")):
            combo.currentIndexChanged.connect(
                lambda _index, combo=combo, key=key: self._set(key, _device(combo))
            )
        for box, key in (
            (self.max_order, "early.max_order"),
            (self.early_rays, "early.rays"),
            (self.late_rays, "late.rays"),
            (self.directions, "late.directions"),
            (self.threads, "performance.threads"),
            (self.length, "ir_length_s"),
            (self.transition, "transition_ms"),
            (self.seed, "seed"),
        ):
            box.valueChanged.connect(lambda value, key=key: self._set(key, value))
        for check, key in (
            (self.splitting, "late.splitting"),
            (self.from_size, "scattering.from_size"),
            (self.by_band, "scattering.by_band"),
            (self.wave, "wave.enabled"),
        ):
            check.toggled.connect(lambda on, key=key: self._set(key, on))
        self.auto_transition.toggled.connect(
            lambda on: self._set("transition_ms", None if on else self.transition.value())
        )
        self.random_seed.toggled.connect(
            lambda on: self._set("seed", None if on else self.seed.value())
        )

    def load(self) -> None:
        s = self.document.scene.simulation
        _choose(self.early_method, s.early.method)
        self.max_order.setValue(s.early.max_order)
        self.early_rays.setValue(s.early.rays)
        _choose(self.diffraction, s.early.diffraction)
        coherent = s.early.coherent_max_band_hz
        _choose(self.coherent, coherent)
        _choose(self.late_method, s.late.method)
        self.late_rays.setValue(s.late.rays)
        _choose(self.reception, s.late.reception)
        self.directions.setValue(s.late.directions)
        self.splitting.setChecked(s.late.splitting)
        _set_device(self.late_device, s.late.device)
        _choose(self.scattering_method, s.scattering.method)
        self.from_size.setChecked(s.scattering.from_size)
        self.by_band.setChecked(s.scattering.by_band)
        self.by_band.setEnabled(s.scattering.method == "vector")
        self.wave.setChecked(s.wave.enabled)
        self.wave_band.setCurrentText(str(s.wave.max_band_hz))
        self.wave_band.setEnabled(s.wave.enabled)
        _set_device(self.wave_device, s.wave.device)
        self.wave_device.setEnabled(s.wave.enabled)
        self.threads.setValue(s.performance.threads)
        _choose(self.ray_tree, s.performance.ray_tree)
        self.length.setValue(s.ir_length_s)
        if self.sample_rate.findText(str(s.sample_rate)) < 0:
            self.sample_rate.addItem(str(s.sample_rate))
        self.sample_rate.setCurrentText(str(s.sample_rate))
        _choose(self.bands, s.bands)
        self.auto_transition.setChecked(s.transition_ms is None)
        self.transition.setEnabled(s.transition_ms is not None)
        self.transition.setValue(s.transition_ms or 80.0)
        self.random_seed.setChecked(s.seed is None)
        self.seed.setEnabled(s.seed is not None)
        self.seed.setValue(s.seed or 0)
        stochastic = s.late.method == "stochastic_ray_tracing"
        for box in (
            self.late_rays,
            self.reception,
            self.directions,
            self.splitting,
            self.late_device,
        ):
            box.setEnabled(stochastic)
        self.early_rays.setEnabled(s.early.method == "ray_tracing")

    def _set(self, key: str, value) -> None:
        """Write the one setting that changed (``late.rays``): the others show rounded to
        their boxes' decimals."""
        *groups, name = key.split(".")

        def change(data: dict) -> None:
            settings = data["simulation"]
            for group in groups:
                settings = settings[group]
            settings[name] = value

        self.apply(_("change simulation settings"), change)


class OutputsPage(Page):
    """The formats responses are rendered in."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.mono = QCheckBox(_("Mono"))
        self.mono.setToolTip(
            _(
                "Write one impulse response per source and listener, as an omnidirectional "
                "microphone hears it"
            )
        )
        self.binaural = QCheckBox(_("Binaural"))
        self.binaural.setToolTip(
            _("Write a two-channel response for headphones, through the HRTF below")
        )
        self.hrtf = QLabel()
        self.hrtf.setWordWrap(True)
        self.hrtf.setToolTip(
            _(
                "The head-related transfer functions (HRTF) the binaural response goes through: "
                "a SOFA file, or a spherical head model"
            )
        )
        hrtf_buttons = QHBoxLayout()
        choose = QPushButton(_("HRTF file…"))
        choose.setToolTip(_("Pick a SOFA file of head-related transfer functions"))
        choose.clicked.connect(self._choose_hrtf)
        head = QPushButton(_("Spherical head"))
        head.setToolTip(_("Use a spherical head model instead of an HRTF file"))
        head.clicked.connect(lambda: self._set_hrtf(None))
        hrtf_buttons.addWidget(choose)
        hrtf_buttons.addWidget(head)
        self.ambisonics = QCheckBox(_("Ambisonics (AmbiX)"))
        self.ambisonics.setToolTip(
            _(
                "Write an Ambisonics response (AmbiX: ACN channel order, SN3D normalisation), "
                "to decode for any loudspeakers or turn with a head tracker"
            )
        )
        self.order = int_spin(0, 7)
        self.order.setToolTip(
            _(
                "The Ambisonics order: (order + 1)² channels; higher orders place sounds more "
                "sharply"
            )
        )
        self.loudspeakers = QCheckBox(_("Loudspeakers"))
        self.loudspeakers.setToolTip(
            _("Write one channel per loudspeaker of a standard layout, panned between them (VBAP)")
        )
        self.layout_name = QComboBox()
        _fill(
            self.layout_name,
            [(name, N_("stereo") if name == "stereo" else name) for name in LAYOUT_NAMES],
        )
        self.layout_name.setToolTip(
            _(
                "The loudspeaker layout: stereo, surround (5.0, 7.0), with a low-frequency "
                "channel (5.1, 7.1), and with four height speakers (.4)"
            )
        )
        form.addRow(self.mono)
        form.addRow(self.binaural)
        form.addRow("", self.hrtf)
        form.addRow("", hrtf_buttons)
        form.addRow(self.ambisonics)
        form.addRow(_("Order"), self.order)
        form.addRow(self.loudspeakers)
        form.addRow(_("Layout"), self.layout_name)
        layout.addLayout(form)
        self.custom = QLabel()
        self.custom.setWordWrap(True)
        self.custom.setToolTip(_("Layouts of your own, given speaker by speaker in the scene file"))
        layout.addWidget(self.custom)
        layout.addStretch()
        for check in (self.mono, self.binaural, self.ambisonics, self.loudspeakers):
            check.toggled.connect(self._changed)
        self.order.valueChanged.connect(self._changed)
        self.layout_name.currentIndexChanged.connect(self._changed)
        self._hrtf: str | None = None

    def load(self) -> None:
        outputs = self.document.scene.outputs
        by_format = {}
        for output in outputs:
            # Loudspeakers here means a named layout; custom ones are kept as they are.
            if output.format != "loudspeakers" or not output.speakers:
                by_format.setdefault(output.format, output)
        self.mono.setChecked("mono" in by_format)
        self.binaural.setChecked("binaural" in by_format)
        binaural = by_format.get("binaural")
        self._hrtf = str(binaural.hrtf) if binaural and binaural.hrtf else None
        self.hrtf.setText(
            _("HRTF: {file}").format(file=self._hrtf)
            if self._hrtf
            else _("HRTF: spherical head model")
        )
        self.ambisonics.setChecked("ambisonics" in by_format)
        self.order.setValue(by_format["ambisonics"].order if "ambisonics" in by_format else 1)
        speakers = by_format.get("loudspeakers")
        self.loudspeakers.setChecked(speakers is not None)
        _choose(self.layout_name, (speakers.layout if speakers else None) or "5.0")
        custom = [o for o in outputs if o.format == "loudspeakers" and o.speakers]
        self.custom.setText(
            ngettext(
                "The scene also has {count} custom loudspeaker layout; it is kept.",
                "The scene also has {count} custom loudspeaker layouts; they are kept.",
                len(custom),
            ).format(count=len(custom))
            if custom
            else ""
        )

    def _changed(self) -> None:
        if self._loading:
            return
        outputs: list[dict] = []
        if self.mono.isChecked():
            outputs.append({"format": "mono"})
        if self.binaural.isChecked():
            entry = {"format": "binaural"}
            if self._hrtf:
                entry["hrtf"] = self._hrtf
            outputs.append(entry)
        if self.ambisonics.isChecked():
            outputs.append({"format": "ambisonics", "order": self.order.value()})
        if self.loudspeakers.isChecked():
            outputs.append({"format": "loudspeakers", "layout": self.layout_name.currentData()})
        for output in self.document.scene.outputs:
            if output.format == "loudspeakers" and output.speakers:
                outputs.append(output.model_dump(mode="json", exclude_none=True))
        if not outputs:
            self.failed.emit(_("keep at least one output format"))
            self.reload()
            return
        self.apply(_("change outputs"), lambda data: data.update(outputs=outputs))

    def _choose_hrtf(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("HRTF set"),
            str(self.document.base_dir),
            _("SOFA files (*.sofa);;All files (*)"),
        )
        if path:
            self._set_hrtf(path)

    def _set_hrtf(self, path: str | None) -> None:
        self._hrtf = path
        if not self.binaural.isChecked():
            self.binaural.setChecked(True)  # emits _changed
        else:
            self._changed()


class CoveragePage(Page):
    """Where the coverage map lies, how fine it is, its rays and the background noise
    for the speech transmission index."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        area = QGroupBox(_("Audience area"))
        area_layout = QVBoxLayout(area)
        self.automatic = QCheckBox(_("Automatic: surfaces named audience, else the floor"))
        self.automatic.setToolTip(
            _(
                "Ticked, the map lies over the surfaces whose names hold audience, else over "
                "the floor. Unticked, over the surfaces ticked below"
            )
        )
        self.surfaces = QListWidget()
        self.surfaces.setToolTip(_("The surfaces the map lies over, when Automatic is unticked"))
        self.surfaces.setMaximumHeight(140)
        area_layout.addWidget(self.automatic)
        area_layout.addWidget(self.surfaces)
        form = QFormLayout()
        self.height = spin(0.1, 20, 2, 0.1, "m")
        self.height.setToolTip(
            _("How high above the surfaces the map's points are (m): about 1.2 m for seated ears")
        )
        self.spacing = spin(0.1, 20, 2, 0.25, "m")
        self.spacing.setToolTip(_("How far apart the map's points are (m); finer maps take longer"))
        form.addRow(_("Listening height"), self.height)
        form.addRow(_("Point spacing"), self.spacing)
        area_layout.addLayout(form)
        layout.addWidget(area)

        rays = QGroupBox(_("Rays"))
        rays_form = QFormLayout(rays)
        self.same_rays = QCheckBox(_("As for the late reverberation"))
        self.same_rays.setToolTip(
            _("Ticked, the map traces as many rays per band as the simulation's late reverberation")
        )
        self.rays = int_spin(1, 100_000_000, 10_000)
        self.rays.setToolTip(_("Rays per source and band for the map; more give smoother maps"))
        rays_form.addRow(self.same_rays)
        rays_form.addRow(_("Rays per source and band"), self.rays)
        layout.addWidget(rays)

        noise = QGroupBox(_("Speech transmission index"))
        noise_layout = QVBoxLayout(noise)
        self.noisy = QCheckBox(_("Background noise (dB SPL per octave band)"))
        self.noisy.setToolTip(
            _(
                "The background noise in each octave band (dB SPL), which lowers the speech "
                "transmission index (STI). Unticked, there is none"
            )
        )
        self.noise = QTableWidget(1, len(OCTAVE_BANDS_HZ))
        self.noise.setHorizontalHeaderLabels(BAND_LABELS)
        band_headers(self.noise)
        self.noise.verticalHeader().hide()
        self.noise.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.noise.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.noise.setMaximumHeight(60)
        noise_layout.addWidget(self.noisy)
        noise_layout.addWidget(self.noise)
        hint = QLabel(
            _(
                "Each source radiates its sound power over the bands as its spectrum, gain and "
                "equalizer set, and the STI takes that as male speech, or female for the female "
                "speech spectrum. Run the map with Simulation > Run coverage map (F6)."
            )
        )
        hint.setWordWrap(True)
        noise_layout.addWidget(hint)
        layout.addWidget(noise)
        layout.addStretch()

        for check in (self.automatic, self.same_rays, self.noisy):
            check.toggled.connect(self._changed)
        for box in (self.height, self.spacing, self.rays):
            box.valueChanged.connect(self._changed)
        self.surfaces.itemChanged.connect(self._changed)
        self.noise.itemChanged.connect(self._changed)

    def load(self) -> None:
        settings = self.document.scene.coverage
        room = self.document.room
        chosen = set(settings.surfaces or []) if settings else set()
        self.automatic.setChecked(settings is None or not settings.surfaces)
        self.surfaces.clear()
        for name in room.mesh.label_names():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in chosen else Qt.CheckState.Unchecked)
            self.surfaces.addItem(item)
        self.surfaces.setEnabled(not self.automatic.isChecked())
        defaults = settings or self._defaults()
        self.height.setValue(defaults.height)
        self.spacing.setValue(defaults.spacing)
        self.same_rays.setChecked(defaults.rays is None)
        self.rays.setEnabled(defaults.rays is not None)
        self.rays.setValue(defaults.rays or self.document.scene.simulation.late.rays)
        noise = defaults.noise_db
        self.noisy.setChecked(noise is not None)
        self.noise.setEnabled(noise is not None)
        values = [35.0] * len(OCTAVE_BANDS_HZ) if noise is None else noise
        if not isinstance(values, list):
            values = [float(values)] * len(OCTAVE_BANDS_HZ)
        for col, value in enumerate(values):
            item = QTableWidgetItem(number(value))
            item.setToolTip(_("Background noise at {band} (dB SPL)").format(band=BAND_LABELS[col]))
            self.noise.setItem(0, col, item)

    @staticmethod
    def _defaults():
        from acoustic_spaces.scene import Coverage

        return Coverage()

    def _changed(self) -> None:
        if self._loading:
            return
        settings: dict = {"height": self.height.value(), "spacing": self.spacing.value()}
        if not self.automatic.isChecked():
            surfaces = [
                self.surfaces.item(k).text()
                for k in range(self.surfaces.count())
                if self.surfaces.item(k).checkState() == Qt.CheckState.Checked
            ]
            if surfaces:
                settings["surfaces"] = surfaces
            elif self.sender() is self.automatic:
                # Unticked just now: start from the surfaces the map would take.
                from acoustic_spaces.engine.coverage import audience_surfaces

                try:
                    settings["surfaces"] = audience_surfaces(self.document.room)
                except ValueError:
                    pass
        if not self.same_rays.isChecked():
            settings["rays"] = self.rays.value()
        if self.noisy.isChecked():
            try:
                settings["noise_db"] = [
                    parse_number(self.noise.item(0, c).text()) for c in range(len(OCTAVE_BANDS_HZ))
                ]
            except (AttributeError, ValueError):
                self.failed.emit(_("noise levels are numbers in dB"))
                self.reload()
                return
        self.apply(_("change coverage map settings"), lambda data: data.update(coverage=settings))


class Inspector(QStackedWidget):
    """The form for the selected item."""

    failed = Signal(str)
    select = Signal(str, object)

    def __init__(self, document: Callable[[], Document]) -> None:
        super().__init__()
        self.pages: dict[str, Page] = {
            "": EmptyPage(document),
            "scene": ScenePage(document),
            "geometry": GeometryPage(document),
            "surface": SurfacePage(document),
            "material": MaterialPage(document),
            "sources": PointPage(document, "sources"),
            "listeners": PointPage(document, "listeners"),
            "simulation": SimulationPage(document),
            "outputs": OutputsPage(document),
            "coverage": CoveragePage(document),
        }
        for page in self.pages.values():
            share_tips(page)
            self.addWidget(page)
            page.failed.connect(self.failed)
            page.select.connect(self.select)
        self.kind = ""

    def show_item(self, kind: str, key=None) -> None:
        page = self.pages.get(kind, self.pages[""])
        self.kind = kind if kind in self.pages else ""
        self.setCurrentWidget(page)
        page.show_item(key)

    def refresh(self) -> None:
        self.currentWidget().reload()


__all__ = ["Inspector", "Page", "label_text"]
