"""Forms for the selected item: the scene, its geometry, a surface, a material, a source
or listener, the simulation settings, the output formats and the coverage map.

Every form shows the current scene and turns each change into one
:meth:`Document.edit`; a change the scene schema refuses is reported and the form shows
the scene as it still is.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt, Signal
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

from acoustic_spaces.gui.document import Document, EditError
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.materials.bands import convert
from acoustic_spaces.room import UNASSIGNED
from acoustic_spaces.scene.model import LAYOUT_NAMES

BAND_LABELS = [f"{f} Hz" if f < 1000 else f"{f // 1000} kHz" for f in OCTAVE_BANDS_HZ]
#: Column labels of the third-octave bands, short to fit 24 columns.
THIRD_OCTAVE_LABELS = [f"{f}" if f < 1000 else f"{f / 1000:g}k" for f in THIRD_OCTAVE_BANDS_HZ]
#: The spectra a source's sound power can have.
SPECTRA = ["flat", "speech_male", "speech_female"]
DIRECTIVITIES = [
    "omni",
    "subcardioid",
    "cardioid",
    "supercardioid",
    "hypercardioid",
    "figure8",
    "lambert",
]
#: A source is a point, a line or a surface.
SHAPES = ["point", "line", "surface"]
#: The directivity list entry that asks for a SOFA or CSV file.
MEASURED = "Measured data or an array from a file…"
SAMPLE_RATES = [16_000, 22_050, 32_000, 44_100, 48_000, 88_200, 96_000]
#: Shown in the material list of a surface for "no assignment of its own".
FALLBACK = "(default material)"
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
    return " / ".join(label) if label else "(unnamed)"


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
            "Select something in the scene tree or click it in the 3D view.\n\n"
            "Place a new source or listener with the toolbar buttons, then click a floor "
            "or wall in the 3D view."
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
        self.name.editingFinished.connect(self._name)
        form.addRow("Name", self.name)
        self.default = QComboBox()
        self.default.currentIndexChanged.connect(self._default)
        form.addRow("Default material", self.default)
        self.temperature = spin(-20, 50, 1, 1, "°C")
        self.humidity = spin(0, 100, 1, 5, "%")
        self.pressure = spin(50, 110, 3, 0.5, "kPa")
        for box, field in (
            (self.temperature, "temperature_c"),
            (self.humidity, "humidity_percent"),
            (self.pressure, "pressure_kpa"),
        ):
            box.valueChanged.connect(lambda value, f=field: self._air(f, value))
        form.addRow("Air temperature", self.temperature)
        form.addRow("Relative humidity", self.humidity)
        form.addRow("Air pressure", self.pressure)
        self.stats = QLabel()
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.stats)

    def load(self) -> None:
        scene, room = self.document.scene, self.document.room
        self.name.setText(scene.name)
        self.default.clear()
        self.default.addItems(["(none)", *scene.materials])
        self.default.setCurrentText(scene.default_material or "(none)")
        self.temperature.setValue(scene.air.temperature_c)
        self.humidity.setValue(scene.air.humidity_percent)
        self.pressure.setValue(scene.air.pressure_kpa)
        lines = [
            f"Surface area: {room.report.surface_area_m2:.1f} m²",
            f"Volume: {room.volume_m3:.1f} m³" if room.volume_m3 else "Volume: model not closed",
        ]
        if len(room.room_volumes) > 1:
            each = ", ".join(f"{v:.1f}" for v in room.room_volumes)
            lines[-1] += f" in {len(room.room_volumes)} rooms ({each} m³)"
        if room.volume_m3:
            sabine = room.reverberation_time("sabine", air_absorption=True)
            eyring = room.reverberation_time("eyring", air_absorption=True)
            for band in (500, 1000):
                k = OCTAVE_BANDS_HZ.index(band)
                lines.append(
                    f"Reverberation time at {BAND_LABELS[k]}: {eyring[k]:.2f} s (Eyring), "
                    f"{sabine[k]:.2f} s (Sabine)"
                )
        for problem in room.problems():
            lines.append(f"Problem: {problem}")
        for warning in room.warnings():
            lines.append(f"Warning: {warning}")
        self.stats.setText("\n".join(lines))

    def _name(self) -> None:
        name = self.name.text().strip() or "untitled"
        if name != self.document.scene.name:
            self.apply("rename scene", lambda data: data.update(name=name))

    def _default(self) -> None:
        value = self.default.currentText()

        def change(data: dict) -> None:
            if value == "(none)":
                data.pop("default_material", None)
            else:
                data["default_material"] = value

        if value and value != (self.document.scene.default_material or "(none)"):
            self.apply("change default material", change)

    def _air(self, field: str, value: float) -> None:
        self.apply("change air", lambda data: data.setdefault("air", {}).update({field: value}))


class GeometryPage(Page):
    """A box's dimensions, a floor plan's corners and height, or a model file."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        self.kind = QLabel()
        self.kind.setWordWrap(True)
        layout.addWidget(self.kind)

        self.box_group = QGroupBox("Box")
        box_form = QFormLayout(self.box_group)
        self.box = [spin(0.1, LARGEST, 2, 0.5) for _ in range(3)]
        for k, (label, box) in enumerate(
            zip(("Length (x)", "Width (y)", "Height (z)"), self.box, strict=True)
        ):
            box.valueChanged.connect(lambda value, k=k: self._box(k, value))
            box_form.addRow(label, box)
        layout.addWidget(self.box_group)

        self.plan_group = QGroupBox("Floor plan")
        plan_layout = QVBoxLayout(self.plan_group)
        self.plan = QTableWidget(0, 2)
        self.plan.setHorizontalHeaderLabels(["x", "y"])
        self.plan.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.plan.itemChanged.connect(self._plan)
        plan_layout.addWidget(self.plan)
        buttons = QHBoxLayout()
        add, remove = QPushButton("Add corner"), QPushButton("Remove corner")
        add.clicked.connect(self._add_corner)
        remove.clicked.connect(self._remove_corner)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        plan_layout.addLayout(buttons)
        height_form = QFormLayout()
        self.height = spin(0.1, LARGEST, 2, 0.5)
        self.height.valueChanged.connect(self._height)
        height_form.addRow("Height", self.height)
        plan_layout.addLayout(height_form)
        layout.addWidget(self.plan_group)

        self.file_group = QGroupBox("Model file")
        file_form = QFormLayout(self.file_group)
        self.file = QLabel()
        self.file.setWordWrap(True)
        choose = QPushButton("Choose another file…")
        choose.clicked.connect(self._choose)
        self.units = QComboBox()
        self.units.addItems(["file default", "m", "cm", "mm", "in", "ft"])
        self.units.setToolTip("file default: the units a DXF file records, else metres")
        self.units.currentTextChanged.connect(self._units)
        self.up_axis = QComboBox()
        self.up_axis.addItems(["format default", "y", "z"])
        self.up_axis.currentTextChanged.connect(self._up_axis)
        file_form.addRow(self.file)
        file_form.addRow(choose)
        file_form.addRow("Units", self.units)
        file_form.addRow("Up axis", self.up_axis)
        layout.addWidget(self.file_group)

        self.report = QLabel()
        self.report.setWordWrap(True)
        layout.addWidget(self.report)
        layout.addStretch()

    def load(self) -> None:
        geometry, room = self.document.scene.geometry, self.document.room
        self.kind.setText(f"Geometry: {geometry.describe()}")
        self.box_group.setVisible(geometry.box is not None)
        self.plan_group.setVisible(geometry.floor_plan is not None)
        self.file_group.setVisible(geometry.file is not None)
        units = geometry.units or "m"  # a box or floor plan without units is in metres
        self.box_group.setTitle(f"Box ({units})")
        self.plan_group.setTitle(f"Floor plan ({units})")
        if geometry.box is not None:
            for box, value in zip(self.box, geometry.box, strict=True):
                box.setValue(value)
        if geometry.floor_plan is not None:
            self.plan.blockSignals(True)  # a half-filled table is no floor plan
            self.plan.setRowCount(len(geometry.floor_plan))
            for row, corner in enumerate(geometry.floor_plan):
                for col, value in enumerate(corner):
                    self.plan.setItem(row, col, QTableWidgetItem(f"{value:g}"))
            self.plan.blockSignals(False)
            self.height.setValue(geometry.height)
        if geometry.file is not None:
            self.file.setText(str(geometry.file))
            self.units.setCurrentText(geometry.units or "file default")
            self.up_axis.setCurrentText(geometry.up_axis or "format default")
        r = room.report
        closed = "closed" if r.closed else "not closed"
        self.report.setText(
            f"{len(room.mesh.faces)} triangles, {r.surface_area_m2:.1f} m², {closed}"
        )

    def _box(self, axis: int, value: float) -> None:
        # Only the side that changed: the others show rounded to the spin boxes' decimals.
        def change(data: dict) -> None:
            data["geometry"]["box"][axis] = value

        self.apply("resize box", change)

    def _corners(self) -> list[list[float]] | None:
        corners = []
        for row in range(self.plan.rowCount()):
            try:
                corners.append([float(self.plan.item(row, c).text()) for c in range(2)])
            except (AttributeError, ValueError):
                return None
        return corners

    def _plan(self) -> None:
        if self._loading:
            return
        corners = self._corners()
        if corners is None:
            self.failed.emit("floor plan corners need two numbers each")
            self.reload()
            return
        self.apply("change floor plan", lambda data: data["geometry"].update(floor_plan=corners))

    def _add_corner(self) -> None:
        corners = self._corners() or []
        if len(corners) >= 2:
            new = list(np.mean([corners[-1], corners[0]], axis=0))
        else:
            new = [0.0, 0.0]
        corners.append(new)
        self.apply("add corner", lambda data: data["geometry"].update(floor_plan=corners))

    def _remove_corner(self) -> None:
        corners = self._corners() or []
        row = self.plan.currentRow()
        if not corners:
            return
        corners.pop(row if 0 <= row < len(corners) else -1)
        self.apply("remove corner", lambda data: data["geometry"].update(floor_plan=corners))

    def _height(self, value: float) -> None:
        self.apply("change height", lambda data: data["geometry"].update(height=value))

    def _choose(self) -> None:
        if self._loading:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Room model",
            str(self.document.base_dir),
            "Room models (*.obj *.stl *.ply *.off *.gltf *.glb *.dxf *.skp);;All files (*)",
        )
        if path:
            self.apply("replace model", lambda data: data["geometry"].update(file=path))

    def _units(self, units: str) -> None:
        def change(data: dict) -> None:
            if units == "file default":
                data["geometry"].pop("units", None)
            else:
                data["geometry"]["units"] = units

        self.apply("change units", change)

    def _up_axis(self, axis: str) -> None:
        def change(data: dict) -> None:
            if axis == "format default":
                data["geometry"].pop("up_axis", None)
            else:
                data["geometry"]["up_axis"] = axis

        self.apply("change up axis", change)


class SurfacePage(Page):
    """A group of faces that share a name: its area and its material."""

    def __init__(self, document) -> None:
        super().__init__(document)
        form = QFormLayout(self)
        self.name = QLabel()
        self.name.setWordWrap(True)
        form.addRow("Surface", self.name)
        self.area = QLabel()
        form.addRow("Area", self.area)
        self.material = QComboBox()
        self.material.currentIndexChanged.connect(self._material)
        form.addRow("Material", self.material)
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
        self.area.setText(f"{float(room.mesh.face_areas()[faces].sum()):.1f} m²")
        self.material.clear()
        self.material.addItems([FALLBACK, *scene.materials])
        own = next((scene.assignments[n] for n in label if n in scene.assignments), None)
        self.material.setCurrentText(own or FALLBACK)
        self.material.setEnabled(bool(label))
        index = int(room.face_materials[np.flatnonzero(faces)[0]]) if faces.any() else UNASSIGNED
        used = room.material_names[index] if index != UNASSIGNED else "none"
        if label:
            self.note.setText(
                f"Now: {used}. Choosing a material assigns it to the name “{label[0]}”"
                + (", which other surfaces may share." if len(label) > 1 else ".")
            )
        else:
            self.note.setText(
                f"Now: {used}. This surface has no name; it takes the default material."
            )

    def _material(self) -> None:
        label = self._label()
        choice = self.material.currentText()
        if not label or not choice:
            return
        # The first name, as the note says, even when a later one (``walls``, which every
        # wall of a floor plan has) carries the assignment in use: changing that one
        # would change every surface that has it.
        name = label[0]
        self.run(lambda: self.document.assign(name, None if choice == FALLBACK else choice))


#: The material table's rows: absorption, scattering, and the sound reduction index.
_MATERIAL_ROWS = ["Absorption", "Scattering", "Sound reduction (dB)"]
#: What a surface of the material lets through: nothing, part of the sound (with a
#: sound reduction index per band), or everything (an opening).
PASSING = ["nothing", "part of the sound", "everything (an opening)"]
#: The sound reduction index (dB) a material starts from when it begins to let sound through.
DEFAULT_REDUCTION_DB = 40.0


def _reduction_text(passing: float) -> str:
    """How the material table shows a transmission: as R in dB, blank for none."""
    return "" if passing <= 0.0 else f"{max(0.0, -10.0 * math.log10(passing)):.1f}"


def _nearest_filled(values: list[float | None]) -> list[float]:
    """``values`` with each None taking the value of the nearest band that has one."""
    have = [k for k, v in enumerate(values) if v is not None]
    return [
        v if v is not None else values[min(have, key=lambda j: abs(j - k))]
        for k, v in enumerate(values)
    ]


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
        self.name.editingFinished.connect(self._rename)
        form.addRow("Name", self.name)
        self.preset = QComboBox()
        self.preset.setEditable(True)
        self.preset.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        entries = builtin_database().entries("absorption")
        self.preset.addItems([""] + [e.id for e in entries])
        completer = QCompleter([e.id for e in entries])
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.preset.setCompleter(completer)
        self.preset.activated.connect(self._preset)
        form.addRow("Database entry", self.preset)
        self.passing = QComboBox()
        self.passing.addItems(PASSING)
        self.passing.setToolTip(
            "Sound that passes through a surface goes on into the room behind it, a "
            "partition's at its sound reduction index R (30 to 60 dB for walls), an "
            "opening's all of it. What passes counts as absorbed, so the absorption is "
            "at least 10^(-R/10), and an opening's is 1."
        )
        self.passing.activated.connect(self._passing)
        form.addRow("Lets through", self.passing)
        layout.addLayout(form)
        self.table = QTableWidget(len(_MATERIAL_ROWS), len(OCTAVE_BANDS_HZ))
        self.table.setHorizontalHeaderLabels(BAND_LABELS)
        self.table.setVerticalHeaderLabels(_MATERIAL_ROWS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.table.setMaximumHeight(165)
        self.table.itemChanged.connect(self._coefficients)
        layout.addWidget(self.table)
        self.source = QLabel()
        self.source.setWordWrap(True)
        layout.addWidget(self.source)
        self.usage = QLabel()
        self.usage.setWordWrap(True)
        layout.addWidget(self.usage)
        remove = QPushButton("Remove material")
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
        mode = QHeaderView.ResizeMode.ResizeToContents if third else QHeaderView.ResizeMode.Stretch
        self.table.horizontalHeader().setSectionResizeMode(mode)
        for row, values in enumerate(material.coefficients(bands)):
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(f"{value:.2f}"))
        passing = material.transmission_coefficients(bands)
        kind = self._kind(passing)
        self.passing.setCurrentIndex(kind)
        for col, value in enumerate(passing or [0.0] * len(labels)):
            item = QTableWidgetItem(_reduction_text(value))
            if kind != 1:  # set by the choice above, not typed
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(2, col, item)
        self.source.setText(f"Source: {material.source or 'values given in the scene'}")
        area = self.document.room.area_by_material().get(self.key, 0.0)
        self.usage.setText(f"Covers {area:.1f} m² of the room.")

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
        material = self.document.scene.materials[self.key]
        fields = {"preset": entry, "scattering": material.scattering}
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
            rows = [[float(self.table.item(r, c).text()) for c in columns] for r in range(2)]
        except (AttributeError, ValueError):
            self.failed.emit("coefficients are numbers from 0 to 1")
            self.reload()
            return
        material = self.document.scene.materials[self.key]
        bands = self._bands(material)
        passing = material.transmission_coefficients(bands)
        reduction = None
        if self._kind(passing) == 1:
            texts = [self.table.item(2, c).text().strip() for c in columns]
            try:
                typed = [float(t) if t else None for t in texts]
            except ValueError:
                self.failed.emit("the sound reduction is in dB")
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
            origin = fields.get("source") or "values given in the scene"
            kept = {k: fields[k] for k in ("transmission", "transmission_loss") if k in fields}
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


def _directivity_label(spec) -> str:
    """How the directivity list shows a source's setting that is not a pattern name."""
    if spec.file is not None:
        return f"file: {spec.file.name}"
    if spec.array is not None:
        return f"array of {len(spec.array.layout())} elements"
    return "per band: " + ", ".join(str(v) for v in spec.pattern)


class PointPage(Page):
    """A source or listener: name, position and orientation; sources also have a
    directivity and a sound power level."""

    def __init__(self, document, kind: str) -> None:
        super().__init__(document)
        self.kind = kind
        form = QFormLayout(self)
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._name)
        form.addRow("Name", self.name)
        self.position = [spin(-1e4, 1e4, 2, 0.1, "m") for _ in range(3)]
        for k, (axis, box) in enumerate(zip("xyz", self.position, strict=True)):
            box.valueChanged.connect(lambda value, k=k: self._position(k, value))
            form.addRow(axis, box)
        self.angles = [spin(-360, 360, 1, 5, "°") for _ in range(3)]
        for angle, box in zip(("yaw", "pitch", "roll"), self.angles, strict=True):
            box.valueChanged.connect(lambda value, a=angle: self._orientation(a, value))
            form.addRow(angle.capitalize(), box)
        self.directivity = QComboBox()
        self.directivity.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.directivity.setMinimumContentsLength(12)
        self.directivity.activated.connect(self._directivity)
        self.directivity_index = QLabel()
        self.directivity_index.setWordWrap(True)
        self.pattern = QPushButton("Show the pattern")
        self.pattern.clicked.connect(self.show_pattern)
        self.pattern_window = None
        self.level = spin(0, 200, 1, 1, "dB")
        self.level.valueChanged.connect(self._level)
        self.spectrum = QComboBox()
        self.spectrum.addItems(SPECTRA)
        self.spectrum.setToolTip(
            "How the sound power spreads over the bands: evenly over the octave bands, or "
            "as male or female speech (IEC 60268-16), which the STI then weights for"
        )
        self.spectrum.activated.connect(self._spectrum)
        self.gain = spin(-100, 60, 1, 1, "dB")
        self.gain.setToolTip("Gain of the source's feed: on its level and its responses")
        self.gain.valueChanged.connect(lambda value: self._update(gain_db=value))
        self.eq = QLineEdit()
        self.eq.setPlaceholderText("none")
        self.eq.setToolTip(
            "Equalizer gain per band (dB): 8 octave values from 63 Hz to 8 kHz, or 24 "
            "third-octave values from 50 Hz to 10 kHz, separated by commas"
        )
        self.eq.editingFinished.connect(self._eq)
        self.delay = spin(0, 10_000, 2, 1, "ms")
        self.delay.setToolTip(
            "The source starts this long after the others: its arrivals and responses "
            "come that much later"
        )
        self.delay.valueChanged.connect(lambda value: self._update(delay_ms=value))
        self.shape = QComboBox()
        self.shape.addItems(SHAPES)
        self.shape.setToolTip(
            "A line runs along the source's left axis and a surface spans its left and up "
            "axes, both centred on its position: uncorrelated point sources share the power"
        )
        self.shape.activated.connect(self._shape)
        self.extent = [spin(0.01, 1000, 2, 0.5, "m") for _ in range(2)]
        for box in self.extent:
            box.valueChanged.connect(lambda _: self._extent())
        self.spacing = spin(0.05, 100, 2, 0.1, "m")
        self.spacing.setToolTip("The most the point sources of a line or surface are apart")
        self.spacing.valueChanged.connect(lambda value: self._update(spacing_m=value))
        self.elements = QLabel()
        self.elements.setWordWrap(True)
        self._form = form
        if kind == "sources":
            form.addRow("Shape", self.shape)
            form.addRow("Length", self.extent[0])
            form.addRow("Height", self.extent[1])
            form.addRow("Spacing", self.spacing)
            form.addRow(self.elements)
            form.addRow("Directivity", self.directivity)
            form.addRow(self.directivity_index)
            form.addRow(self.pattern)
            form.addRow("Sound power", self.level)
            form.addRow("Spectrum", self.spectrum)
            form.addRow("Gain", self.gain)
            form.addRow("Equalizer", self.eq)
            form.addRow("Delay", self.delay)
        remove = QPushButton("Remove " + ("source" if kind == "sources" else "listener"))
        remove.clicked.connect(self._remove)
        form.addRow(remove)
        hint = QLabel(
            "Yaw turns left from +x towards +y, pitch looks up and roll tilts to the right. "
            "In the 3D view the long arrow points forward and the short one up."
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
            self.directivity.addItems(DIRECTIVITIES)
            if isinstance(item.directivity, str):
                self.directivity.setCurrentText(item.directivity)
            else:
                self.directivity.addItem(_directivity_label(item.directivity))
                self.directivity.setCurrentIndex(len(DIRECTIVITIES))
            self.directivity.addItem(MEASURED)
            self.directivity.setToolTip(self.directivity.currentText())
            directivity = self.document.room.directivities[self.key]
            index = directivity.index()
            text = (
                "Directivity index: "
                + ", ".join(f"{v:.1f}" for v in index)
                + " dB (63 Hz to 8 kHz)"
            )
            if directivity.extrapolated:
                bands = ", ".join(str(f) for f in directivity.extrapolated)
                text += f"; {bands} Hz copied from the nearest octave"
            self.directivity_index.setText(text)
            per_band = isinstance(item.level_db, list)
            total = 10.0 * np.log10(np.sum(10.0 ** (np.array(item.level_db) / 10.0)))
            self.level.setValue(float(total))
            self.level.setEnabled(not per_band)
            self.level.setToolTip(
                "Given per band in the scene file; this is their sum" if per_band else ""
            )
            self.spectrum.setCurrentText(item.spectrum)
            self.spectrum.setEnabled(not per_band)
            self.gain.setValue(item.gain_db)
            self.eq.setText("" if item.eq_db is None else ", ".join(f"{v:g}" for v in item.eq_db))
            self.delay.setValue(item.delay_ms)
            shape = "line" if item.line_m else "surface" if item.surface_m else "point"
            self.shape.setCurrentText(shape)
            if item.line_m is not None:
                self.extent[0].setValue(item.line_m)
            elif item.surface_m is not None:
                for box, value in zip(self.extent, item.surface_m, strict=True):
                    box.setValue(value)
            self.spacing.setValue(item.spacing_m)
            self._form.labelForField(self.extent[0]).setText(
                "Width" if shape == "surface" else "Length"
            )
            self._form.setRowVisible(self.extent[0], shape != "point")
            self._form.setRowVisible(self.extent[1], shape == "surface")
            self._form.setRowVisible(self.spacing, shape != "point")
            count = len(item.elements())
            self.elements.setText(f"{count} uncorrelated point sources" if count > 1 else "")
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
        value = self.directivity.itemText(index)
        if value in DIRECTIVITIES:
            self._update(directivity=value)
        elif value == MEASURED:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Source directivity",
                str(self.document.base_dir),
                "Directivity data or arrays (*.sofa *.csv *.yaml *.yml);;All files (*)",
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
        self.pattern_window.setWindowTitle(f"Directivity of {self._item().name}")
        self.pattern_window.show()
        self.pattern_window.raise_()

    def _shape(self, index: int) -> None:
        shape = SHAPES[index]
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
        self._update(spectrum=self.spectrum.itemText(index))

    def _eq(self) -> None:
        text = self.eq.text().replace(",", " ").split()
        try:
            values = [float(v) for v in text] or None
        except ValueError:
            self.failed.emit("the equalizer takes numbers (dB), one per band")
            self.reload()
            return
        if values != self._item().eq_db:
            self._update(eq_db=values)


class SimulationPage(Page):
    """The early and late stages, scattering, the wave solver, the response length,
    sample rate, bands and seed."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        early = QGroupBox("Early reflections")
        early_form = QFormLayout(early)
        self.early_method = QComboBox()
        self.early_method.addItems(["ray_tracing", "image_sources"])
        self.max_order = int_spin(0, 20)
        self.early_rays = int_spin(1, 10_000_000, 1000)
        self.diffraction = QComboBox()
        self.diffraction.addItems(["none", "direct", "first_order"])
        self.diffraction.setToolTip(
            "Sound bending round edges (pillars, balcony fronts, screens): none, of the "
            "direct sound only, or of the direct sound and first-order reflections"
        )
        early_form.addRow("Method", self.early_method)
        early_form.addRow("Highest order", self.max_order)
        early_form.addRow("Rays", self.early_rays)
        early_form.addRow("Diffraction", self.diffraction)
        layout.addWidget(early)

        late = QGroupBox("Late reverberation")
        late_form = QFormLayout(late)
        self.late_method = QComboBox()
        self.late_method.addItems(["stochastic_ray_tracing", "statistical"])
        self.late_rays = int_spin(1, 100_000_000, 10_000)
        self.reception = QComboBox()
        self.reception.addItems(["diffuse_rain", "secondary_sources", "detector"])
        self.directions = int_spin(0, 1024, 16)
        self.splitting = QCheckBox("Split strong rays, thin out weak ones")
        late_form.addRow("Method", self.late_method)
        late_form.addRow("Rays per band", self.late_rays)
        late_form.addRow("Reception", self.reception)
        late_form.addRow("Arrival directions", self.directions)
        late_form.addRow(self.splitting)
        layout.addWidget(late)

        scattering = QGroupBox("Scattering")
        scattering_form = QFormLayout(scattering)
        self.scattering_method = QComboBox()
        self.scattering_method.addItems(["vector", "random"])
        self.scattering_method.setToolTip(
            "Where a reflected ray goes: vector (between the specular direction and a "
            "random Lambert one, weighted by the scattering coefficient, as in ODEON) or "
            "random (specular, or with the scattering coefficient as probability Lambert)"
        )
        self.from_size = QCheckBox("Add the scattering of each surface's size")
        self.from_size.setToolTip(
            "Small surfaces, distant ones and grazing reflections reflect less "
            "specularly at low frequencies (Christensen and Rindel, 2005)"
        )
        self.by_band = QCheckBox("Trace each band with rays of its own")
        self.by_band.setToolTip(
            "Vector-based scattering: where the bands scatter differently, each band "
            "follows its own scattering (exact). Off: one ray for all bands, turned by "
            "their mean scattering (faster)"
        )
        scattering_form.addRow("Direction", self.scattering_method)
        scattering_form.addRow(self.from_size)
        scattering_form.addRow(self.by_band)
        layout.addWidget(scattering)

        wave = QGroupBox("Low frequencies")
        wave_form = QFormLayout(wave)
        self.wave = QCheckBox("Wave solver")
        self.wave.setToolTip(
            "Compute the lowest octave bands from the sound field itself (finite differences "
            "on a grid) instead of rays: room modes, and sound bending round everything"
        )
        self.wave_band = QComboBox()
        self.wave_band.addItems(["63", "125", "250", "500"])
        self.wave_band.setToolTip(
            "The highest band it computes; each costs about 16 times the one below"
        )
        wave_form.addRow(self.wave)
        wave_form.addRow("Up to band (Hz)", self.wave_band)
        layout.addWidget(wave)

        response = QGroupBox("Response")
        response_form = QFormLayout(response)
        self.length = spin(0.05, 60, 2, 0.5, "s")
        self.sample_rate = QComboBox()
        self.sample_rate.addItems([str(r) for r in SAMPLE_RATES])
        self.bands = QComboBox()
        self.bands.addItems(["octave", "third_octave"])
        self.bands.setToolTip(
            "The bands it runs in: 8 octave bands, or 24 third-octave bands for finer "
            "detail in the response (about three times the work). Room parameters are "
            "given in octave bands either way"
        )
        self.auto_transition = QCheckBox("Automatic")
        self.transition = spin(1, 10_000, 0, 10, "ms")
        self.random_seed = QCheckBox("New random seed every run")
        self.seed = int_spin(0, 2**31 - 1)
        response_form.addRow("Length", self.length)
        response_form.addRow("Sample rate (Hz)", self.sample_rate)
        response_form.addRow("Bands", self.bands)
        response_form.addRow("Early to late", self.auto_transition)
        response_form.addRow("", self.transition)
        response_form.addRow("Seed", self.random_seed)
        response_form.addRow("", self.seed)
        layout.addWidget(response)
        layout.addStretch()

        for combo in (
            self.early_method,
            self.diffraction,
            self.late_method,
            self.reception,
            self.scattering_method,
            self.wave_band,
            self.sample_rate,
            self.bands,
        ):
            combo.currentTextChanged.connect(self._changed)
        for box in (self.max_order, self.early_rays, self.late_rays, self.directions, self.seed):
            box.valueChanged.connect(self._changed)
        for box in (self.length, self.transition):
            box.valueChanged.connect(self._changed)
        for check in (
            self.splitting,
            self.from_size,
            self.by_band,
            self.wave,
            self.auto_transition,
            self.random_seed,
        ):
            check.toggled.connect(self._changed)

    def load(self) -> None:
        s = self.document.scene.simulation
        self.early_method.setCurrentText(s.early.method)
        self.max_order.setValue(s.early.max_order)
        self.early_rays.setValue(s.early.rays)
        self.diffraction.setCurrentText(s.early.diffraction)
        self.late_method.setCurrentText(s.late.method)
        self.late_rays.setValue(s.late.rays)
        self.reception.setCurrentText(s.late.reception)
        self.directions.setValue(s.late.directions)
        self.splitting.setChecked(s.late.splitting)
        self.scattering_method.setCurrentText(s.scattering.method)
        self.from_size.setChecked(s.scattering.from_size)
        self.by_band.setChecked(s.scattering.by_band)
        self.by_band.setEnabled(s.scattering.method == "vector")
        self.wave.setChecked(s.wave.enabled)
        self.wave_band.setCurrentText(str(s.wave.max_band_hz))
        self.wave_band.setEnabled(s.wave.enabled)
        self.length.setValue(s.ir_length_s)
        if self.sample_rate.findText(str(s.sample_rate)) < 0:
            self.sample_rate.addItem(str(s.sample_rate))
        self.sample_rate.setCurrentText(str(s.sample_rate))
        self.bands.setCurrentText(s.bands)
        self.auto_transition.setChecked(s.transition_ms is None)
        self.transition.setEnabled(s.transition_ms is not None)
        self.transition.setValue(s.transition_ms or 80.0)
        self.random_seed.setChecked(s.seed is None)
        self.seed.setEnabled(s.seed is not None)
        self.seed.setValue(s.seed or 0)
        stochastic = s.late.method == "stochastic_ray_tracing"
        for box in (self.late_rays, self.reception, self.directions, self.splitting):
            box.setEnabled(stochastic)
        self.early_rays.setEnabled(s.early.method == "ray_tracing")

    def _changed(self) -> None:
        if self._loading:
            return
        settings = {
            "early": {
                "method": self.early_method.currentText(),
                "max_order": self.max_order.value(),
                "rays": self.early_rays.value(),
                "diffraction": self.diffraction.currentText(),
            },
            "late": {
                **self.document.scene.simulation.late.model_dump(mode="json", exclude_none=True),
                "method": self.late_method.currentText(),
                "rays": self.late_rays.value(),
                "reception": self.reception.currentText(),
                "directions": self.directions.value(),
                "splitting": self.splitting.isChecked(),
            },
            "scattering": {
                "method": self.scattering_method.currentText(),
                "from_size": self.from_size.isChecked(),
                "by_band": self.by_band.isChecked(),
            },
            "wave": {
                **self.document.scene.simulation.wave.model_dump(mode="json"),
                "enabled": self.wave.isChecked(),
                "max_band_hz": int(self.wave_band.currentText()),
            },
            "ir_length_s": self.length.value(),
            "sample_rate": int(self.sample_rate.currentText()),
            "bands": self.bands.currentText(),
        }
        if not self.auto_transition.isChecked():
            settings["transition_ms"] = self.transition.value()
        if not self.random_seed.isChecked():
            settings["seed"] = self.seed.value()
        self.apply("change simulation settings", lambda data: data.update(simulation=settings))


class OutputsPage(Page):
    """The formats responses are rendered in."""

    def __init__(self, document) -> None:
        super().__init__(document)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.mono = QCheckBox("Mono")
        self.binaural = QCheckBox("Binaural")
        self.hrtf = QLabel()
        self.hrtf.setWordWrap(True)
        hrtf_buttons = QHBoxLayout()
        choose = QPushButton("HRTF file…")
        choose.clicked.connect(self._choose_hrtf)
        head = QPushButton("Spherical head")
        head.clicked.connect(lambda: self._set_hrtf(None))
        hrtf_buttons.addWidget(choose)
        hrtf_buttons.addWidget(head)
        self.ambisonics = QCheckBox("Ambisonics (AmbiX)")
        self.order = int_spin(0, 7)
        self.loudspeakers = QCheckBox("Loudspeakers")
        self.layout_name = QComboBox()
        self.layout_name.addItems(LAYOUT_NAMES)
        form.addRow(self.mono)
        form.addRow(self.binaural)
        form.addRow("", self.hrtf)
        form.addRow("", hrtf_buttons)
        form.addRow(self.ambisonics)
        form.addRow("Order", self.order)
        form.addRow(self.loudspeakers)
        form.addRow("Layout", self.layout_name)
        layout.addLayout(form)
        self.custom = QLabel()
        self.custom.setWordWrap(True)
        layout.addWidget(self.custom)
        layout.addStretch()
        for check in (self.mono, self.binaural, self.ambisonics, self.loudspeakers):
            check.toggled.connect(self._changed)
        self.order.valueChanged.connect(self._changed)
        self.layout_name.currentTextChanged.connect(self._changed)
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
        self.hrtf.setText(f"HRTF: {self._hrtf or 'spherical head model'}")
        self.ambisonics.setChecked("ambisonics" in by_format)
        self.order.setValue(by_format["ambisonics"].order if "ambisonics" in by_format else 1)
        speakers = by_format.get("loudspeakers")
        self.loudspeakers.setChecked(speakers is not None)
        self.layout_name.setCurrentText((speakers.layout if speakers else None) or "5.0")
        custom = [o for o in outputs if o.format == "loudspeakers" and o.speakers]
        self.custom.setText(
            f"The scene also has {len(custom)} custom loudspeaker layout(s); they are kept."
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
            outputs.append({"format": "loudspeakers", "layout": self.layout_name.currentText()})
        for output in self.document.scene.outputs:
            if output.format == "loudspeakers" and output.speakers:
                outputs.append(output.model_dump(mode="json", exclude_none=True))
        if not outputs:
            self.failed.emit("keep at least one output format")
            self.reload()
            return
        self.apply("change outputs", lambda data: data.update(outputs=outputs))

    def _choose_hrtf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "HRTF set", str(self.document.base_dir), "SOFA files (*.sofa);;All files (*)"
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
        area = QGroupBox("Audience area")
        area_layout = QVBoxLayout(area)
        self.automatic = QCheckBox("Automatic: surfaces named audience, else the floor")
        self.surfaces = QListWidget()
        self.surfaces.setMaximumHeight(140)
        area_layout.addWidget(self.automatic)
        area_layout.addWidget(self.surfaces)
        form = QFormLayout()
        self.height = spin(0.1, 20, 2, 0.1, "m")
        self.spacing = spin(0.1, 20, 2, 0.25, "m")
        form.addRow("Listening height", self.height)
        form.addRow("Point spacing", self.spacing)
        area_layout.addLayout(form)
        layout.addWidget(area)

        rays = QGroupBox("Rays")
        rays_form = QFormLayout(rays)
        self.same_rays = QCheckBox("As for the late reverberation")
        self.rays = int_spin(1, 100_000_000, 10_000)
        rays_form.addRow(self.same_rays)
        rays_form.addRow("Rays per source and band", self.rays)
        layout.addWidget(rays)

        noise = QGroupBox("Speech transmission index")
        noise_layout = QVBoxLayout(noise)
        self.noisy = QCheckBox("Background noise (dB SPL per octave band)")
        self.noise = QTableWidget(1, len(OCTAVE_BANDS_HZ))
        self.noise.setHorizontalHeaderLabels(BAND_LABELS)
        self.noise.verticalHeader().hide()
        self.noise.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.noise.setEditTriggers(QAbstractItemView.EditTrigger.AllEditTriggers)
        self.noise.setMaximumHeight(60)
        noise_layout.addWidget(self.noisy)
        noise_layout.addWidget(self.noise)
        hint = QLabel(
            "Each source talks at its sound power level, spread evenly over the bands. "
            "Run the map with Simulation > Run coverage map (F6)."
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
            self.noise.setItem(0, col, QTableWidgetItem(f"{value:g}"))

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
                    float(self.noise.item(0, c).text()) for c in range(len(OCTAVE_BANDS_HZ))
                ]
            except (AttributeError, ValueError):
                self.failed.emit("noise levels are numbers in dB")
                self.reload()
                return
        self.apply("change coverage map settings", lambda data: data.update(coverage=settings))


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
