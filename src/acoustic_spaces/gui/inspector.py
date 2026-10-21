"""Forms for the selected item: the scene, its geometry, a surface, a material, a source
or listener, the simulation settings, the output formats and the coverage map.

Every form shows the current scene and turns each change into one
:meth:`Document.edit`; a change the scene schema refuses is reported and the form shows
the scene as it still is.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLocale, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
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
from acoustic_spaces.engine.long_term import rating_name
from acoustic_spaces.geometry import UNIT_SCALE
from acoustic_spaces.geometry.io import SUPPORTED_SUFFIXES, ifc_spaces
from acoustic_spaces.geometry.terrain import footprint_properties
from acoustic_spaces.gui.document import NEW_PART_SIZES, Document, EditError
from acoustic_spaces.gui.levels import rich_lines
from acoustic_spaces.gui.numbers import fmt, number, numbers_text, parse_number, parse_numbers
from acoustic_spaces.gui.tips import band_headers, row_headers, share_tips
from acoustic_spaces.i18n import N_, _, ngettext
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.materials.bands import convert
from acoustic_spaces.materials.ground import PRESETS
from acoustic_spaces.room import UNASSIGNED, matching_name
from acoustic_spaces.scene.model import (
    ATMOSPHERE_PRESETS,
    INDOOR_COHERENT_MAX_BAND_HZ,
    LAYOUT_NAMES,
    TURBULENCE_PRESETS,
    Atmosphere,
    LongTerm,
    Turbulence,
    Wind,
)

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
DIFFRACTION_REFLECTIONS = [("one_side", N_("one side")), ("both_sides", N_("both sides"))]
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
WAVE_WINDS = [("effective", N_("effective speed of sound")), ("moving", N_("moving air"))]
SCATTERING_METHODS = [("vector", N_("vector")), ("random", N_("random"))]
RAY_TREES = [("sah", "sah"), ("median", N_("median"))]
BANDS = [("octave", N_("octave")), ("third_octave", N_("third_octave"))]
#: The weathers an outdoor scene starts from: none, a preset, or values of its own.
WEATHERS = [
    (None, N_("(none)")),
    ("neutral", N_("neutral")),
    ("windy", N_("windy")),
    ("sunny_day", N_("sunny_day")),
    ("clear_night", N_("clear_night")),
    ("custom", N_("custom")),
]
WIND_PROFILES = [("log", N_("log")), ("linear", N_("linear")), ("constant", N_("constant"))]
EDDIES = [
    (None, N_("(none)")),
    ("weak", N_("weak")),
    ("moderate", N_("moderate")),
    ("strong", N_("strong")),
    ("custom", N_("custom")),
]
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
    #: The page asks for something to be drawn in the plan view: one of
    #: :data:`~acoustic_spaces.gui.plan_view.PURPOSES` other than a room.
    draw = Signal(str)

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
        form.addRow(self._weather_group())
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

    def _weather_group(self) -> QGroupBox:
        """Outdoors: the wind and the temperature over the ground, which bend sound."""
        self.weather_group = QGroupBox(_("Wind and temperature"))
        form = QFormLayout(self.weather_group)
        self.weather_zone = QComboBox()
        self.weather_zone.setToolTip(
            _(
                "Whose weather the fields below show and change: the whole scene's; a "
                "weather zone's, a part of the scene with wind and temperature of its own "
                "(the lee of buildings, where the wind drops; a lake, over which the air "
                "lies cooler), drawn in the plan; or a weather class's, a weather that holds "
                "for a share of the time, for the levels over a long term (L_den). Outside "
                "the zones the scene's weather holds, and the eddies are the scene's "
                "everywhere but in a weather class, which may have its own"
            )
        )
        self.weather_zone.currentIndexChanged.connect(self._weather_zone_chosen)
        form.addRow(_("Weather of"), self.weather_zone)
        self.weather = QComboBox()
        _fill(self.weather, WEATHERS)
        self.weather.setToolTip(
            _(
                "The weather, which bends sound outdoors: (none) or neutral, still air and "
                "sound in straight lines; windy, 5 m/s 10 m up, bending sound down downwind "
                "and up upwind; sunny_day, air cooling with height, bending sound up into a "
                "shadow a few hundred metres away; clear_night, air warming with height, "
                "bending sound down; custom, the values below. Changing one of them makes "
                "the weather custom. The air temperature above is the ground's. For a "
                "weather zone, (none) is the scene's weather unchanged"
            )
        )
        self.weather.currentIndexChanged.connect(self._weather)
        form.addRow(_("Weather"), self.weather)
        self.wind_speed = spin(0, 40, 1, 0.5, "m/s")
        self.wind_speed.setToolTip(
            _(
                "The wind's speed at the height below (m/s): sound goes faster with the wind "
                "and slower against it"
            )
        )
        self.wind_direction = spin(0, 360, 1, 15, "°")
        self.wind_direction.setWrapping(True)
        self.wind_direction.setToolTip(
            _(
                "The way the wind blows to, in degrees counterclockwise from +x seen from "
                "above (0: along +x, 90: along +y)"
            )
        )
        self.wind_height = spin(0.1, 1000, 1, 1, "m")
        self.wind_height.setToolTip(_("The height above the ground the wind's speed is at (m)"))
        self.wind_profile = QComboBox()
        _fill(self.wind_profile, WIND_PROFILES)
        self.wind_profile.setToolTip(
            _(
                "How the wind's speed grows with height: log, with the logarithm of the "
                "height over the roughness, the usual profile; linear, in proportion to the "
                "height; constant, the same at every height"
            )
        )
        self.gradient = spin(-0.2, 0.2, 4, 0.001, "K/m")
        self.gradient.setToolTip(
            _(
                "How the temperature changes per metre of height (K/m): -0.0098 in air "
                "stirred by the sun, positive in an inversion. Sound goes faster in warmer "
                "air and bends away from it"
            )
        )
        self.log_term = spin(-5, 5, 2, 0.1, "K")
        self.log_term.setToolTip(
            _(
                "A change of temperature fastest near the ground (K, times ln(1 + height / "
                "roughness)): negative on a sunny day, positive on a clear night"
            )
        )
        self.roughness = spin(0.001, 2, 3, 0.01, "m")
        self.roughness.setToolTip(
            _(
                "The ground's roughness length (m), for the log profiles: 0.01 to 0.1 over "
                "grass, up to 1 among buildings and trees"
            )
        )
        for box, field in (
            (self.wind_speed, "speed_m_s"),
            (self.wind_direction, "direction_deg"),
            (self.wind_height, "height_m"),
        ):
            box.valueChanged.connect(lambda value, f=field: self._wind(f, value))
        self.wind_profile.currentIndexChanged.connect(
            lambda: self._wind("profile", self.wind_profile.currentData())
        )
        for box, field in (
            (self.gradient, "temperature_gradient_k_m"),
            (self.log_term, "temperature_log_k"),
            (self.roughness, "roughness_m"),
        ):
            box.valueChanged.connect(lambda value, f=field: self._atmosphere(f, value))
        form.addRow(_("Wind speed"), self.wind_speed)
        form.addRow(_("Wind direction"), self.wind_direction)
        form.addRow(_("Wind height"), self.wind_height)
        form.addRow(_("Wind profile"), self.wind_profile)
        form.addRow(_("Temperature gradient"), self.gradient)
        form.addRow(_("Temperature near the ground"), self.log_term)
        form.addRow(_("Roughness"), self.roughness)
        self.zone_temperature = spin(-20, 50, 1, 0.5, "°C")
        self.zone_temperature.setToolTip(
            _(
                "The temperature at the ground in the weather zone or class (°C), where it "
                "differs from the scene's air temperature: cooler over water, warmer over a "
                "car park, colder on a winter night"
            )
        )
        self.zone_temperature.valueChanged.connect(self._zone_temperature)
        form.addRow(_("Ground temperature"), self.zone_temperature)
        zone_buttons = QHBoxLayout()
        self.add_weather_zone = QPushButton(_("Add weather zone"))
        self.add_weather_zone.setToolTip(
            _(
                "Draw a part of the scene with weather of its own in the plan, corner by "
                "corner, and close it on the first corner; then choose it under Weather of "
                "and give it its wind and temperature"
            )
        )
        self.add_weather_zone.clicked.connect(lambda _checked=False: self.draw.emit("weather"))
        self.remove_weather_zone = QPushButton(_("Remove weather zone"))
        self.remove_weather_zone.setToolTip(
            _("Remove the weather zone chosen under Weather of: the scene's weather holds there")
        )
        self.remove_weather_zone.clicked.connect(self._remove_weather_zone)
        zone_buttons.addWidget(self.add_weather_zone)
        zone_buttons.addWidget(self.remove_weather_zone)
        zone_buttons.addStretch(1)
        form.addRow(zone_buttons)
        self.class_shares = QTableWidget(1, 3)
        self.class_shares.verticalHeader().hide()
        self.class_shares.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.class_shares.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.class_shares.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.class_shares.setToolTip(
            _(
                "The share of each period of the day this weather class holds (%), from "
                "weather statistics of the place: the inversions of a clear night come at "
                "night, a strong sun by day. In each period the classes' shares add up to "
                "100 %: changing one makes room in the others, in proportion"
            )
        )
        self.class_shares.itemChanged.connect(self._class_share)
        form.addRow(_("Share of the time"), self.class_shares)
        self.periods = QLabel()
        self.periods.setWordWrap(True)
        self.periods.setToolTip(
            _(
                "The periods of the day the levels over a long term are rated over, with "
                "their hours and the penalty each adds to its level: by default the day, the "
                "evening and the night of L_den. Other periods are set in the scene file "
                "(long_term.periods)"
            )
        )
        form.addRow(self.periods)
        class_buttons = QHBoxLayout()
        self.add_weather_class = QPushButton(_("Add weather class"))
        self.add_weather_class.setToolTip(
            _(
                "Add a weather that holds for a share of the time, for the levels over a long "
                "term (L_den): the first holds all the time in the scene's weather, the next "
                "ones none of it until given their share. A run then runs the scene in each "
                "class and combines them"
            )
        )
        self.add_weather_class.clicked.connect(self._add_weather_class)
        self.remove_weather_class = QPushButton(_("Remove weather class"))
        self.remove_weather_class.setToolTip(
            _(
                "Remove the weather class chosen under Weather of: the others take its share "
                "of the time, in proportion"
            )
        )
        self.remove_weather_class.clicked.connect(self._remove_weather_class)
        class_buttons.addWidget(self.add_weather_class)
        class_buttons.addWidget(self.remove_weather_class)
        class_buttons.addStretch(1)
        form.addRow(class_buttons)
        self.eddies = QComboBox()
        _fill(self.eddies, EDDIES)
        self.eddies.setToolTip(
            _(
                "The eddies of the wind and of the temperature, which make sound outdoors "
                "flicker: the direct sound and its reflections stop adding in step (the dip "
                "of the ground fills in), sound is scattered into shadow zones, and renders "
                "and walkthroughs change as the eddies drift past. (none): steady air; weak, "
                "overcast with a light wind; moderate, a breeze and some sun; strong, a strong "
                "wind or a hot sun; custom, the values below"
            )
        )
        self.eddies.currentIndexChanged.connect(self._eddies)
        self.temperature_eddies = spin(0, 10, 4, 0.001)
        self.temperature_eddies.setToolTip(
            _(
                "How strong the temperature's eddies are: its structure parameter C_T² "
                "(K² m^-2/3), about 0.001 to 1 near the ground, the most in sunshine"
            )
        )
        self.wind_eddies = spin(0, 10, 3, 0.01)
        self.wind_eddies.setToolTip(
            _(
                "How strong the wind's eddies are: its structure parameter C_v² (m^4/3 s^-2), "
                "about 0.01 to 1 near the ground, the more the stronger the wind"
            )
        )
        self.eddy_size = spin(0.1, 1000, 1, 1, "m")
        self.eddy_size.setToolTip(
            _(
                "The size of the largest eddies (m), some metres near the ground: they take "
                "this long to drift past at the wind's speed, which sets how fast the sound "
                "flickers"
            )
        )
        self.eddy_height = spin(0, 100, 1, 0.5, "m")
        self.eddy_height.setSpecialValueText(_("the same at every height"))
        self.eddy_height.setToolTip(
            _(
                "The height above the ground the eddies' strength holds at (where it was "
                "measured or estimated, usually 2 m), from which they weaken with height; "
                "nearer the ground than the eddies' size they are taken as at that height. "
                "0: the same strength at every height"
            )
        )
        self.eddy_falloff = spin(0, 2, 2, 0.1)
        self.eddy_falloff.setToolTip(
            _(
                "How fast the eddies weaken with height: the power of the height over the "
                "eddies' height that their strength falls by. 0.67 (two thirds) in a neutral "
                "or stable surface layer, the usual; up to 1.33 in strong sunshine"
            )
        )
        for box, field in (
            (self.temperature_eddies, "temperature_structure"),
            (self.wind_eddies, "wind_structure"),
            (self.eddy_size, "outer_scale_m"),
            (self.eddy_falloff, "falloff"),
        ):
            box.valueChanged.connect(lambda value, f=field: self._eddy(f, value))
        self.eddy_height.valueChanged.connect(lambda value: self._eddy("height_m", value or None))
        form.addRow(_("Eddies"), self.eddies)
        form.addRow(_("Temperature eddies"), self.temperature_eddies)
        form.addRow(_("Wind eddies"), self.wind_eddies)
        form.addRow(_("Eddy size"), self.eddy_size)
        form.addRow(_("Eddies' height"), self.eddy_height)
        form.addRow(_("Weaker with height"), self.eddy_falloff)
        self.table = QLabel()
        self.table.setWordWrap(True)
        self.table.setToolTip(
            _(
                "A measured profile from the scene file: the temperature and the wind's "
                "speed come from it, in straight lines between its heights"
            )
        )
        form.addRow(self.table)
        self.drop_table = QPushButton(_("Remove the measured profile"))
        self.drop_table.setToolTip(
            _("Use the values above in place of the measured profile of the scene file")
        )
        self.drop_table.clicked.connect(self._drop_table)
        form.addRow(self.drop_table)
        self.profile = QPushButton(_("Show the profile"))
        self.profile.setToolTip(
            _(
                "Open a plot of the temperature, the wind and the speed of sound over the "
                "ground: across the wind, downwind and upwind"
            )
        )
        self.profile.clicked.connect(self.show_profile)
        self.profile_window = None
        form.addRow(self.profile)
        self._weather_form = form
        return self.weather_group

    def _weather_choice(self) -> tuple[str, int] | None:
        """Whose weather the form shows: None for the whole scene's, ("zone", k) for a
        weather zone's and ("class", k) for a weather class's."""
        value = self.weather_zone.currentData()
        if value is None:
            return None
        kind, k = str(value).split(":")
        return kind, int(k)

    def _weather_zone_index(self) -> int | None:
        """The weather zone the form shows, if it shows one."""
        choice = self._weather_choice()
        return choice[1] if choice is not None and choice[0] == "zone" else None

    def _weather_class_index(self) -> int | None:
        """The weather class the form shows, if it shows one."""
        choice = self._weather_choice()
        return choice[1] if choice is not None and choice[0] == "class" else None

    def _load_weather(self, scene) -> None:
        atmosphere, spec = scene.atmosphere, scene.long_term
        # The zones and classes to choose from, keeping the choice while it is still there.
        zones = [] if atmosphere is None else atmosphere.zone_names()
        classes = [] if spec is None else spec.class_names()
        choice = self._weather_choice()
        self.weather_zone.blockSignals(True)
        self.weather_zone.clear()
        self.weather_zone.addItem(_("the whole scene"), None)
        for k, name in enumerate(zones):
            self.weather_zone.addItem(name, f"zone:{k}")
        for k, name in enumerate(classes):
            # Translators: {name} is a weather class's name from the scene file.
            self.weather_zone.addItem(_("weather class {name}").format(name=name), f"class:{k}")
        if choice is not None:
            found = self.weather_zone.findData(f"{choice[0]}:{choice[1]}")
            self.weather_zone.setCurrentIndex(max(found, 0))
        self.weather_zone.blockSignals(False)
        zone, given = self._weather_zone_index(), self._weather_class_index()
        whole = zone is None and given is None
        self.remove_weather_zone.setEnabled(zone is not None)
        self.remove_weather_class.setEnabled(given is not None)
        self.weather.setItemText(0, _("(none)") if whole else _("as the whole scene"))
        self.eddies.setItemText(0, _("(none)") if given is None else _("as the whole scene"))
        shown = atmosphere or Atmosphere()
        own_eddies = shown.turbulence
        eddies = own_eddies
        if zone is not None:
            entry = atmosphere.zones[zone]
            shown = atmosphere.weathers()[zone + 1]
        elif given is not None:
            entry = spec.classes[given]
            shown = shown.under(entry)
            eddies = entry.turbulence  # None: the scene's
        if whole:
            _choose(self.weather, _weather_of(atmosphere))
        else:
            _choose(self.weather, _zone_weather_of(entry))
            self.zone_temperature.setValue(
                scene.air.temperature_c if entry.temperature_c is None else entry.temperature_c
            )
        self._weather_form.setRowVisible(self.zone_temperature, not whole)
        _choose(self.eddies, None if eddies is None else eddies.preset or "custom")
        # The eddies are the scene's everywhere, and a weather class's while it holds.
        for box in (
            self.eddies,
            self.temperature_eddies,
            self.wind_eddies,
            self.eddy_size,
            self.eddy_height,
            self.eddy_falloff,
        ):
            box.setEnabled(zone is None)
        eddies = eddies or own_eddies or Turbulence()
        self.temperature_eddies.setValue(eddies.temperature_structure)
        self.wind_eddies.setValue(eddies.wind_structure)
        self.eddy_size.setValue(eddies.outer_scale_m)
        self.eddy_height.setValue(eddies.height_m or 0.0)
        self.eddy_falloff.setValue(eddies.falloff)
        wind = shown.wind or Wind()
        self.wind_speed.setValue(wind.speed_m_s)
        self.wind_direction.setValue(wind.direction_deg % 360.0)
        self.wind_height.setValue(wind.height_m)
        _choose(self.wind_profile, wind.profile)
        self.gradient.setValue(shown.temperature_gradient_k_m)
        self.log_term.setValue(shown.temperature_log_k)
        self.roughness.setValue(shown.roughness_m)
        # A table gives the temperature and the wind's speed; the direction still counts.
        measured = shown.table is not None
        for box in (
            self.wind_speed,
            self.wind_height,
            self.wind_profile,
            self.gradient,
            self.log_term,
            self.roughness,
        ):
            box.setEnabled(not measured)
        if measured:
            count = len(shown.table)
            self.table.setText(
                ngettext(
                    "Measured profile at {count} height, from the scene file",
                    "Measured profile at {count} heights, from the scene file",
                    count,
                ).format(count=count)
            )
        self._weather_form.setRowVisible(self.table, measured)
        self._weather_form.setRowVisible(self.drop_table, measured)
        self._load_long_term(spec, given)
        if self.profile_window is not None and self.profile_window.isVisible():
            self.show_profile()

    def _load_long_term(self, spec: LongTerm | None, given: int | None) -> None:
        """The periods of the day, and the chosen weather class's share of each."""
        self._weather_form.setRowVisible(self.periods, spec is not None)
        self._weather_form.setRowVisible(self.class_shares, given is not None)
        if spec is None:
            return
        periods = spec.period_list()
        parts = [
            # Translators: {name} is a period of the day from the scene file (day, evening,
            # night), {hours} how long it lasts and {penalty} what its level gets added.
            fmt(
                _("{name} {hours:g} h +{penalty:g} dB"),
                name=p.name,
                hours=p.hours,
                penalty=p.penalty_db,
            )
            if p.penalty_db
            else fmt(_("{name} {hours:g} h"), name=p.name, hours=p.hours)
            for p in periods
        ]
        count = len(spec.classes)
        rating = rating_name(spec.period_names(), spec.hours(), spec.penalties())
        self.periods.setText(
            fmt(
                ngettext(
                    "Levels over a long term: {count} weather class over {periods} ({rating})",
                    "Levels over a long term: {count} weather classes over {periods} ({rating})",
                    count,
                ),
                count=count,
                periods=", ".join(parts),
                rating=rating,
            )
        )
        if given is None:
            return
        table = self.class_shares
        table.blockSignals(True)
        table.setColumnCount(len(periods))
        table.setHorizontalHeaderLabels([p.name for p in periods])
        shares = spec.shares()
        for col, period in enumerate(periods):
            table.horizontalHeaderItem(col).setToolTip(
                # Translators: {period} is a period of the day from the scene file (day,
                # evening, night).
                _("The share of the {period} this weather class holds (%)").format(
                    period=period.name
                )
            )
            item = QTableWidgetItem(number(100.0 * shares[given, col], ".4g"))
            if count == 1:  # the only class holds all the time
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(0, col, item)
        table.blockSignals(False)
        header = table.horizontalHeader().sizeHint().height()
        table.setFixedHeight(header + table.rowHeight(0) + 2 * table.frameWidth())

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
        outdoors = scene.geometry.ground is not None
        self.weather_group.setVisible(outdoors)
        if outdoors:
            self._load_weather(scene)
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

    def _weather_zone_chosen(self) -> None:
        # Showing another zone's or class's weather fills the form: that edits nothing.
        if not self._loading:
            self._loading = True
            try:
                self._load_weather(self.document.scene)
            finally:
                self._loading = False

    def _weather(self) -> None:
        if self.weather.currentIndex() < 0:
            return
        value = self.weather.currentData()
        chosen = self._weather_zone_index()
        given = self._weather_class_index()
        scene = self.document.scene
        atmosphere = scene.atmosphere
        if chosen is not None:
            if value == _zone_weather_of(atmosphere.zones[chosen]):
                return
            self._zone_weather(chosen, value)
            return
        if given is not None:
            if value == _zone_weather_of(scene.long_term.classes[given]):
                return
            self._zone_weather(given, value, kind="class")
            return
        if value == _weather_of(atmosphere):
            return

        def change(data: dict) -> None:
            old = data.get("atmosphere") or {}
            if value is None:
                data.pop("atmosphere", None)
            elif value == "custom":
                data["atmosphere"] = _written_out(old)
            else:
                # The way the wind blows is the scene's, not the weather's: it stays.
                data["atmosphere"] = {"preset": value}
                wind = old.get("wind")
                if isinstance(wind, dict) and "direction_deg" in wind:
                    data["atmosphere"]["wind"] = {"direction_deg": wind["direction_deg"]}
            # So do the eddies, and the weather zones.
            for kept in ("turbulence", "zones"):
                if old.get(kept) is not None:
                    data.setdefault("atmosphere", {})[kept] = old[kept]

        self.apply(_("change the weather"), change)

    def _zone_weather(self, chosen: int, value, kind: str = "zone") -> None:
        """The weather of zone (or ``kind`` "class": weather class) ``chosen``: None, the
        scene's unchanged; a preset; or custom, the values it shows written out."""
        held = ("name", "polygon", "share", "temperature_c", "turbulence")

        def change(data: dict) -> None:
            zones = _entries(data, kind)
            old = zones[chosen]
            kept = {k: old[k] for k in held if k in old}
            if value is None:
                zones[chosen] = kept
            elif value == "custom":
                zones[chosen] = {**kept, **_zone_written_out(old)}
            else:
                zones[chosen] = {**kept, "preset": value}
                wind = old.get("wind")
                if isinstance(wind, dict) and "direction_deg" in wind:
                    zones[chosen]["wind"] = {"direction_deg": wind["direction_deg"]}

        self.apply(_("change the weather"), change)

    def _zone_temperature(self, value: float) -> None:
        choice = self._weather_choice()
        if choice is None:
            return
        kind, chosen = choice
        scene = self.document.scene
        scene_c = scene.air.temperature_c
        entry = (scene.atmosphere.zones if kind == "zone" else scene.long_term.classes)[chosen]
        if (scene_c if entry.temperature_c is None else entry.temperature_c) == value:
            return

        def change(data: dict) -> None:
            zone = _entries(data, kind)[chosen]
            if value == scene_c:
                zone.pop("temperature_c", None)
            else:
                zone["temperature_c"] = value

        if kind == "zone":
            self.apply(_("change the weather zone"), change)
        else:
            self.apply(_("change the weather class"), change)

    def _add_weather_class(self) -> None:
        """A weather class in the scene's weather: the first for all the time, the next
        ones for none of it until given a share. The form then shows it."""
        spec = self.document.scene.long_term
        names = [] if spec is None else spec.class_names()
        name = unique_class_name(names)

        def change(data: dict) -> None:
            if spec is None:
                data["long_term"] = {"classes": [{"name": name, "share": 1.0}]}
            else:
                data["long_term"]["classes"].append({"name": name, "share": 0.0})

        if self.apply(_("add weather class"), change):
            self.weather_zone.setCurrentIndex(self.weather_zone.findData(f"class:{len(names)}"))

    def _remove_weather_class(self) -> None:
        given = self._weather_class_index()
        spec = self.document.scene.long_term
        if given is None or spec is None:
            return
        shares = np.delete(spec.shares(), given, axis=0)
        periods = spec.period_names()

        def change(data: dict) -> None:
            classes = data["long_term"]["classes"]
            classes.pop(given)
            if not classes:
                del data["long_term"]
                return
            _write_shares(classes, _balanced(shares), periods)

        self.weather_zone.setCurrentIndex(0)
        self.apply(_("remove weather class"), change)

    def _class_share(self, item: QTableWidgetItem) -> None:
        """A weather class's share of one period: the other classes make room for it,
        each in proportion to its own share (equally if none has any)."""
        given = self._weather_class_index()
        spec = self.document.scene.long_term
        if given is None or spec is None or self._loading or len(spec.classes) < 2:
            return
        try:
            value = parse_number(item.text()) / 100.0
        except ValueError:
            self.failed.emit(_("a share of the time is a number from 0 to 100 %"))
            self.reload()
            return
        shares = spec.shares()
        period = item.column()
        shares[given, period] = min(max(value, 0.0), 1.0)
        shares[:, period] = _balanced(shares, keep=given)[:, period]
        periods = spec.period_names()

        def change(data: dict) -> None:
            _write_shares(data["long_term"]["classes"], shares, periods)

        self.apply(_("change the weather class"), change)

    def _remove_weather_zone(self) -> None:
        chosen = self._weather_zone_index()
        if chosen is None:
            return

        def change(data: dict) -> None:
            atmosphere = data["atmosphere"]
            atmosphere["zones"].pop(chosen)
            if not atmosphere["zones"]:
                del atmosphere["zones"]
            if atmosphere == Atmosphere().model_dump(mode="json", exclude_none=True):
                del data["atmosphere"]

        self.weather_zone.setCurrentIndex(0)
        self.apply(_("remove weather zone"), change)

    def _eddies(self) -> None:
        if self.eddies.currentIndex() < 0:
            return
        value = self.eddies.currentData()
        scene = self.document.scene
        given = self._weather_class_index()
        if given is not None:
            eddies = scene.long_term.classes[given].turbulence
        else:
            eddies = None if scene.atmosphere is None else scene.atmosphere.turbulence
        if value == (None if eddies is None else eddies.preset or "custom"):
            return
        if given is not None:
            self._class_eddies(given, value)
            return

        def change(data: dict) -> None:
            atmosphere = data.get("atmosphere") or {}
            old = atmosphere.get("turbulence")
            if value is None:
                atmosphere.pop("turbulence", None)
            elif value == "custom":
                atmosphere["turbulence"] = _eddies_written_out(old)
            else:
                atmosphere["turbulence"] = {"preset": value}
                # Their size is the place's, not the weather's, and their height the
                # measurement's: they stay.
                for kept in ("outer_scale_m", "height_m", "falloff"):
                    if isinstance(old, dict) and kept in old:
                        atmosphere["turbulence"][kept] = old[kept]
            # Still air without eddies is no atmosphere (the scene writes the defaults out).
            if atmosphere and atmosphere != Atmosphere().model_dump(mode="json", exclude_none=True):
                data["atmosphere"] = atmosphere
            else:
                data.pop("atmosphere", None)

        self.apply(_("change the eddies"), change)

    def _class_eddies(self, given: int, value) -> None:
        """The eddies of weather class ``given``: None, the scene's; a preset, keeping
        their size and height; or custom, the values the form shows written out."""

        def change(data: dict) -> None:
            entry = data["long_term"]["classes"][given]
            old = entry.get("turbulence")
            if value is None:
                entry.pop("turbulence", None)
            elif value == "custom":
                if old is None:  # the scene's eddies, written out as the class's own
                    atmosphere = self.document.scene.atmosphere
                    eddies = None if atmosphere is None else atmosphere.turbulence
                    if eddies is not None:
                        old = eddies.model_dump(mode="json", exclude_none=True)
                entry["turbulence"] = _eddies_written_out(old)
            else:
                entry["turbulence"] = {"preset": value}
                for kept in ("outer_scale_m", "height_m", "falloff"):
                    if isinstance(old, dict) and kept in old:
                        entry["turbulence"][kept] = old[kept]

        self.apply(_("change the eddies"), change)

    def _eddy(self, field: str, value) -> None:
        given = self._weather_class_index()

        def change(data: dict) -> None:
            if given is not None:
                # The class's own eddies, starting from the scene's when it has none.
                entry = data["long_term"]["classes"][given]
                old = entry.get("turbulence")
                if old is None:
                    old = (data.get("atmosphere") or {}).get("turbulence")
                eddies = _eddies_written_out(old)
                eddies[field] = value
                entry["turbulence"] = eddies
                return
            atmosphere = data.get("atmosphere") or {}
            eddies = _eddies_written_out(atmosphere.get("turbulence"))
            eddies[field] = value
            atmosphere["turbulence"] = eddies
            data["atmosphere"] = atmosphere

        self.apply(_("change the eddies"), change)

    def _weather_data(self, data: dict) -> dict:
        """The scene-file entry the wind and temperature fields change, its preset
        written out: the atmosphere's, or the chosen weather zone's or class's."""
        choice = self._weather_choice()
        if choice is None:
            data["atmosphere"] = _written_out(data.get("atmosphere") or {})
            return data["atmosphere"]
        kind, chosen = choice
        entries = _entries(data, kind)
        entries[chosen] = _zone_written_out(entries[chosen])
        return entries[chosen]

    def _atmosphere(self, field: str, value) -> None:
        def change(data: dict) -> None:
            self._weather_data(data)[field] = value

        self.apply(_("change the wind and temperature"), change)

    def _wind(self, field: str, value) -> None:
        def change(data: dict) -> None:
            weather = self._weather_data(data)
            if not isinstance(weather.get("wind"), dict):
                weather["wind"] = {}
            weather["wind"][field] = value

        self.apply(_("change the wind and temperature"), change)

    def _drop_table(self) -> None:
        def change(data: dict) -> None:
            self._weather_data(data).pop("table", None)

        self.apply(_("remove the measured profile"), change)

    def show_profile(self) -> None:
        """Open (or refresh) a window with the temperature, wind and speed of sound over
        the ground."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        from acoustic_spaces.viz import plot_atmosphere

        scene, outdoor = self.document.scene, self.document.room.outdoor
        top = 100.0 if outdoor is None else max(outdoor.sky - outdoor.ground, 1.0)
        atmosphere, ground_c = scene.atmosphere or Atmosphere(), scene.air.temperature_c
        chosen, given = self._weather_zone_index(), self._weather_class_index()
        if chosen is not None:
            # The zone's weather, as the engine resolves it.
            atmosphere = scene.atmosphere.weathers()[chosen + 1]
            ground_c = scene.atmosphere.ground_temperatures(ground_c)[chosen + 1]
        elif given is not None:
            entry = scene.long_term.classes[given]
            atmosphere = atmosphere.under(entry)
            if entry.temperature_c is not None:
                ground_c = entry.temperature_c
        figure = plot_atmosphere(atmosphere, ground_c, top=top)
        if self.profile_window is None:
            self.profile_window = QWidget(self, Qt.WindowType.Window)
            QVBoxLayout(self.profile_window)
            self.profile_window.resize(1000, 520)
        layout = self.profile_window.layout()
        while layout.count():
            layout.takeAt(0).widget().deleteLater()
        layout.addWidget(FigureCanvasQTAgg(figure))
        self.profile_window.setWindowTitle(
            _("Wind and temperature of {name}").format(name=scene.name)
        )
        self.profile_window.show()
        self.profile_window.raise_()


def _weather_of(atmosphere: Atmosphere | None) -> str | None:
    """The weather the form shows: none, a preset or custom (the eddies aside)."""
    if atmosphere is None:
        return None
    if atmosphere.preset is not None:
        return atmosphere.preset
    plain = atmosphere.model_copy(update={"turbulence": None})
    return None if plain == Atmosphere() else "custom"


def _eddies_written_out(eddies) -> dict:
    """A copy of the eddies' scene-file data with their preset's values written out in
    place of its name, so that changing one of them keeps the others."""
    if eddies is None:
        return {}
    eddies = {"preset": eddies} if isinstance(eddies, str) else copy.deepcopy(eddies)
    preset = eddies.pop("preset", None)
    return {**TURBULENCE_PRESETS.get(preset, {}), **eddies}


def _entries(data: dict, kind: str) -> list[dict]:
    """The scene file's weather zones ("zone") or weather classes ("class")."""
    return data["atmosphere"]["zones"] if kind == "zone" else data["long_term"]["classes"]


def unique_class_name(names) -> str:
    """A name for a new weather class: ``class_<n>``, n from one more than the count of
    ``names`` up, the first not among them."""
    taken = set(names)
    n = len(taken) + 1
    while f"class_{n}" in taken:
        n += 1
    return f"class_{n}"


def _balanced(shares, keep: int | None = None) -> np.ndarray:
    """``shares`` (classes, periods) made to add up to 1 in each period: the classes but
    ``keep`` scaled in proportion to their shares (equally where none has any)."""
    shares = np.array(shares, dtype=float)
    others = [k for k in range(len(shares)) if k != keep]
    for period in range(shares.shape[1]):
        room = 1.0 if keep is None else 1.0 - shares[keep, period]
        total = shares[others, period].sum()
        if total > 0.0:
            shares[others, period] *= room / total
        elif others:
            shares[others, period] = room / len(others)
    return shares


def _write_shares(classes: list[dict], shares, periods: list[str]) -> None:
    """Each class's share in the scene file: one number when it is the same in every
    period, else one per period."""
    for entry, row in zip(classes, np.round(shares, 6), strict=True):
        values = [float(v) for v in row]
        if max(values) - min(values) < 1e-9:
            entry["share"] = values[0]
        else:
            entry["share"] = dict(zip(periods, values, strict=True))


def _zone_weather_of(zone) -> str | None:
    """The weather the form shows for a weather zone: none (the scene's unchanged), a
    preset or custom."""
    if zone.preset is not None:
        return zone.preset
    fields = ("wind", "temperature_gradient_k_m", "temperature_log_k", "roughness_m", "table")
    return "custom" if any(getattr(zone, f) is not None for f in fields) else None


def _zone_written_out(zone: dict) -> dict:
    """A copy of a weather zone's scene-file data with its preset's values written out in
    place of its name (no wind, one temperature, the preset's fields over them), so that
    changing one of them keeps the others."""
    zone = copy.deepcopy(zone)
    preset = zone.pop("preset", None)
    if preset not in ATMOSPHERE_PRESETS:
        return zone
    values = {
        "wind": {"speed_m_s": 0.0},
        "temperature_gradient_k_m": 0.0,
        "temperature_log_k": 0.0,
        **copy.deepcopy(ATMOSPHERE_PRESETS[preset]),
    }
    if zone.get("table") is not None:
        values.pop("temperature_gradient_k_m", None)
        values.pop("temperature_log_k", None)
    if isinstance(zone.get("wind"), dict):
        values["wind"] = {**values["wind"], **zone["wind"]}
    return {**values, **{k: v for k, v in zone.items() if k != "wind"}}


def _written_out(atmosphere: dict) -> dict:
    """A copy of an atmosphere's scene-file data with its weather's values written out in
    place of the weather's name, so that changing one of them keeps the others."""
    atmosphere = copy.deepcopy(atmosphere)
    preset = atmosphere.pop("preset", None)
    if preset not in ATMOSPHERE_PRESETS:
        return atmosphere
    values = copy.deepcopy(ATMOSPHERE_PRESETS[preset])
    if atmosphere.get("table") is not None:
        # A table gives the temperature; the formulas would contradict it.
        values.pop("temperature_gradient_k_m", None)
        values.pop("temperature_log_k", None)
    if isinstance(atmosphere.get("wind"), dict) and "wind" in values:
        atmosphere["wind"] = {**values.pop("wind"), **atmosphere["wind"]}
    return {**values, **atmosphere}


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
            _(
                "The height (z) of the flat ground, or added to the terrain's heights, in "
                "the units the title shows"
            )
        )
        self.ground_height.valueChanged.connect(lambda value: self._ground("height", value))
        self.margin = spin(0.1, LARGEST, 1, 5.0)
        self.margin.setToolTip(
            _(
                "How far the ground reaches past the buildings, barriers and model, unless "
                "its extent is set or a terrain sets its size, and how high the sky is over "
                "the highest of them, unless the sky is set; in the units the title shows"
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
        extent_row = QHBoxLayout()
        self.extent_set = QCheckBox(_("Extent"))
        self.extent_set.setToolTip(
            _(
                "Where the ground ends, from its south-west corner (x, y) to its north-east "
                "one, in the units the title shows; off: the margin round the buildings, "
                "barriers and model, or the whole terrain. Past the ground's edge, sound "
                "leaves the scene as if the ground went on"
            )
        )
        self.extent_set.toggled.connect(self._extent_toggled)
        extent_row.addWidget(self.extent_set)
        self.extent_spins = []
        for tip in (
            _("x where the ground starts (its west edge)"),
            _("y where the ground starts (its south edge)"),
            _("x where the ground ends (its east edge)"),
            _("y where the ground ends (its north edge)"),
        ):
            box = spin(-LARGEST, LARGEST, 2, 5.0)
            box.setToolTip(tip)
            box.valueChanged.connect(self._extent_changed)
            extent_row.addWidget(box)
            self.extent_spins.append(box)
        ground_layout.addLayout(extent_row)
        self.extent = QLabel()
        self.extent.setWordWrap(True)
        self.extent.setToolTip(
            _("Where the ground ends: past it, sound leaves the scene as if the ground went on")
        )
        ground_layout.addWidget(self.extent)
        self.terrain = QLabel()
        self.terrain.setWordWrap(True)
        self.terrain.setToolTip(
            _(
                "The terrain the ground follows: its heights on a grid, from a file or given "
                "in the scene file, sampled over the ground's extent"
            )
        )
        ground_layout.addWidget(self.terrain)
        terrain_row = QHBoxLayout()
        self.terrain_button = QPushButton(_("Terrain from a file…"))
        self.terrain_button.setToolTip(
            _(
                "Make the ground follow the heights in a file: an Esri ASCII grid (.asc) or "
                "x y z points on a regular grid (.xyz, .txt, .csv), in the units the title "
                "shows. The terrain sets the ground's size, unless an extent inside it is set"
            )
        )
        self.terrain_button.clicked.connect(self._choose_terrain)
        self.remove_terrain = QPushButton(_("Remove terrain"))
        self.remove_terrain.setToolTip(
            _("Flatten the ground again, at the ground height, with the margin round the parts")
        )
        self.remove_terrain.clicked.connect(self._remove_terrain)
        terrain_row.addWidget(self.terrain_button)
        terrain_row.addWidget(self.remove_terrain)
        terrain_row.addStretch(1)
        ground_layout.addLayout(terrain_row)
        self.parts = {}
        for kind, title, columns, tips, add_text, add_tip, remove_text in (
            (
                "buildings",
                _("Buildings"),
                (_("Name"), _("Height"), _("Corners")),
                (
                    _("The building's name; its surfaces are named after it"),
                    _("How high the building is, ground to roof, in the units the title shows"),
                    _("How many corners the building's footprint has (as drawn in the plan)"),
                    _("Remove the selected building, or the last one when none is selected"),
                ),
                _("Add building"),
                _(
                    "Draw a building's footprint in the plan, corner by corner, and close it "
                    "on the first corner; it rises {height:g} m from the ground, to change in "
                    "the table"
                ).format(height=NEW_PART_SIZES["building"]["height"]),
                _("Remove building"),
            ),
            (
                "barriers",
                _("Barriers"),
                (_("Name"), _("Height"), _("Points")),
                (
                    _("The barrier's name, which its surfaces carry"),
                    _("How high the barrier is over the ground, in the units the title shows"),
                    _("How many points the barrier's path has (as drawn in the plan)"),
                    _("Remove the selected barrier, or the last one when none is selected"),
                ),
                _("Add barrier"),
                _(
                    "Draw a barrier's path in the plan, point by point, and finish it with "
                    "Enter or a double click; it stands {height:g} m high, to change in the "
                    "table"
                ).format(height=NEW_PART_SIZES["barrier"]["height"]),
                _("Remove barrier"),
            ),
            (
                "vegetation",
                _("Vegetation"),
                (_("Name"), _("Height"), _("Shape"), _("Plants")),
                (
                    _(
                        "The name of the tree, hedge or belt of trees: no surface, a volume "
                        "that sound passes through, losing some of itself"
                    ),
                    _("How high its top is over the ground, in the units the title shows"),
                    _(
                        "A belt of trees (a footprint raised from the ground), a hedge (a path "
                        "of a width) or a lone tree's crown, as drawn in the plan"
                    ),
                    _(
                        "What fills it: foliage, which takes so many dB per metre from sound "
                        "passing through, or leaves by their area and width, which work that "
                        "out for themselves; and stems, which scatter it sideways"
                    ),
                    _("Remove the selected vegetation, or the last one when none is selected"),
                ),
                _("Add vegetation"),
                _(
                    "Draw the shape chosen beside in the plan: a belt's footprint closed on its "
                    "first corner, a hedge's path finished with Enter, or a tree's place with "
                    "one click; dense foliage, its height to change in the table and the rest "
                    "in the scene file"
                ),
                _("Remove vegetation"),
            ),
        ):
            ground_layout.addWidget(QLabel(title))
            table = QTableWidget(0, len(columns))
            table.setHorizontalHeaderLabels(list(columns))
            for col in range(len(columns)):
                table.horizontalHeaderItem(col).setToolTip(tips[col])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.itemChanged.connect(lambda item, kind=kind: self._part_height(kind, item))
            ground_layout.addWidget(table)
            buttons = QHBoxLayout()
            if kind == "vegetation":
                self.plant_shape = QComboBox()
                for purpose, label in PLANT_SHAPES:
                    self.plant_shape.addItem(_(label), purpose)
                self.plant_shape.setToolTip(
                    _(
                        "What to draw next: a belt of trees (its footprint, corner by corner), "
                        "a hedge (its path, point by point, {width:g} m wide) or a lone tree "
                        "(one click, its crown {crown:g} m across)"
                    ).format(
                        width=NEW_PART_SIZES["hedge"]["width"],
                        crown=NEW_PART_SIZES["tree"]["crown_diameter"],
                    )
                )
                buttons.addWidget(self.plant_shape)
            add = QPushButton(add_text)
            add.setToolTip(add_tip)
            add.clicked.connect(lambda _checked=False, kind=kind: self._add(kind))
            buttons.addWidget(add)
            remove = QPushButton(remove_text)
            remove.setToolTip(tips[-1])
            remove.clicked.connect(lambda _checked=False, kind=kind: self._remove_part(kind))
            buttons.addWidget(remove)
            buttons.addStretch(1)
            ground_layout.addLayout(buttons)
            self.parts[kind] = table
            if kind == "buildings":
                self.map_label = QLabel()
                self.map_label.setWordWrap(True)
                self.map_label.setToolTip(
                    _(
                        "The map the buildings marked 'from the buildings file' come from: a "
                        "GeoJSON file of footprints, each raised to the height a property of it "
                        "gives (or one height for all) and named by another"
                    )
                )
                ground_layout.addWidget(self.map_label)
                map_row = QHBoxLayout()
                self.map_button = QPushButton(_("Buildings from a map…"))
                self.map_button.setToolTip(
                    _(
                        "Add the buildings of a GeoJSON file of footprints, on a plane in the "
                        "units the title shows (not longitude and latitude), each as high as a "
                        "property of it says; they join the ones drawn here"
                    )
                )
                self.map_button.clicked.connect(self._choose_buildings_file)
                self.remove_map = QPushButton(_("Remove the map's buildings"))
                self.remove_map.setToolTip(
                    _(
                        "Drop the map and every building that came from it; the buildings "
                        "drawn here stay"
                    )
                )
                self.remove_map.clicked.connect(self._remove_buildings_file)
                map_row.addWidget(self.map_button)
                map_row.addWidget(self.remove_map)
                map_row.addStretch(1)
                ground_layout.addLayout(map_row)
        ground_layout.addWidget(QLabel(_("Ground zones")))
        self.zones = QTableWidget(0, 2)
        self.zones.setHorizontalHeaderLabels([_("Name"), _("Corners")])
        for col, tip in enumerate(
            (
                _(
                    "The zone's name: a part of the ground (a road, a lawn) whose surfaces "
                    "carry it, to give it a material of its own"
                ),
                _("How many corners the zone's outline has (as drawn in the plan)"),
            )
        ):
            self.zones.horizontalHeaderItem(col).setToolTip(tip)
        self.zones.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.zones.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        ground_layout.addWidget(self.zones)
        remove = QPushButton(_("Remove zone"))
        remove.setToolTip(
            _(
                "Remove the selected ground zone, or the last one when none is selected: its "
                "part of the ground takes the ground's material"
            )
        )
        remove.clicked.connect(self._remove_zone)
        buttons = QHBoxLayout()
        add = QPushButton(_("Add zone"))
        add.setToolTip(
            _(
                "Draw a part of the ground in the plan, corner by corner, and close it on the "
                "first corner; then give its surface a material of its own in the scene tree"
            )
        )
        add.clicked.connect(lambda _checked=False: self._add("zone"))
        buttons.addWidget(add)
        buttons.addWidget(remove)
        buttons.addStretch(1)
        ground_layout.addLayout(buttons)
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
        scale = UNIT_SCALE[geometry.units or "m"]
        given = ground.extent is not None
        if given:
            (x0, y0), (x1, y1) = ground.extent
        elif room.outdoor is not None:
            x0, y0, x1, y1 = (value / scale for value in room.outdoor.extent)
        else:
            x0 = y0 = x1 = y1 = 0.0
        self.extent_set.setChecked(given)
        for box, value in zip(self.extent_spins, (x0, y0, x1, y1), strict=True):
            box.setValue(value)
            box.setEnabled(given)
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
        terrain = room.outdoor.terrain if room.outdoor is not None else None
        self.terrain.setVisible(terrain is not None)
        self.remove_terrain.setVisible(terrain is not None)
        spec = geometry.buildings_file
        self.map_label.setVisible(spec is not None)
        self.remove_map.setVisible(spec is not None)
        if spec is not None:
            count = len(geometry.all_buildings()) - len(geometry.buildings or [])
            if isinstance(spec.height, str):
                height = _("height from {property}").format(property=spec.height)
            else:
                height = _("{height} high").format(height=number(spec.height))
            self.map_label.setText(
                ngettext(
                    "Map {name}: {count} building, {height}",
                    "Map {name}: {count} buildings, {height}",
                    count,
                ).format(name=spec.file.name, count=count, height=height)
            )
        if terrain is not None:
            self.terrain.setText(
                fmt(
                    _("Terrain: {points} points, z from {low} to {high} m, grid {spacing} m"),
                    points=f"{terrain.nx + 1} × {terrain.ny + 1}",
                    low=number(terrain.lowest),
                    high=number(terrain.highest),
                    spacing=number(terrain.grid.spacing),
                )
            )
        own = len(geometry.buildings or [])
        for kind, names, items in (
            ("buildings", geometry.building_names(), geometry.all_buildings()),
            ("barriers", geometry.barrier_names(), geometry.barriers or []),
            ("vegetation", geometry.vegetation_names(), geometry.vegetation or []),
        ):
            table = self.parts[kind]
            table.blockSignals(True)
            table.setRowCount(len(items))
            for row, (name, item) in enumerate(zip(names, items, strict=True)):
                if kind == "vegetation":
                    rest = (_plant_shape(item), _plant_fill(item))
                else:
                    rest = (str(len(item.footprint if kind == "buildings" else item.path)),)
                from_file = kind == "buildings" and row >= own
                for col, text in enumerate((name, number(item.height), *rest)):
                    cell = QTableWidgetItem(text)
                    cell.setToolTip(table.horizontalHeaderItem(col).toolTip())
                    if col != 1 or from_file:
                        cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    if from_file:
                        cell.setToolTip(
                            _("From the buildings file, which sets its footprint and height")
                        )
                    table.setItem(row, col, cell)
            table.blockSignals(False)
        zones = ground.zones or []
        self.zones.setRowCount(len(zones))
        for row, (name, zone) in enumerate(zip(ground.zone_names(), zones, strict=True)):
            for col, text in enumerate((name, str(len(zone.polygon)))):
                cell = QTableWidgetItem(text)
                cell.setToolTip(self.zones.horizontalHeaderItem(col).toolTip())
                self.zones.setItem(row, col, cell)

    def _add(self, kind: str) -> None:
        """Ask for a part of ``kind`` ("buildings", "barriers", "vegetation" or "zone")
        to be drawn in the plan view."""
        if self._loading:
            return
        if kind == "vegetation":
            self.draw.emit(self.plant_shape.currentData())
        else:
            self.draw.emit({"buildings": "building", "barriers": "barrier"}.get(kind, kind))

    def _extent_toggled(self, on: bool) -> None:
        if self._loading:
            return
        if on:
            self._extent_changed()
        else:
            self.run(lambda: self.document.set_extent(None))

    def _extent_changed(self) -> None:
        if self._loading or not self.extent_set.isChecked():
            return
        x0, y0, x1, y1 = (box.value() for box in self.extent_spins)
        self.run(lambda: self.document.set_extent([[x0, y0], [x1, y1]]))

    def _choose_terrain(self) -> None:
        if self._loading:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("Terrain"),
            str(self.document.base_dir),
            _("Height grids (*.asc *.xyz *.txt *.csv);;All files (*)"),
        )
        if path:
            self.run(lambda: self.document.set_terrain(path))

    def _remove_terrain(self) -> None:
        self.run(lambda: self.document.set_terrain(None))

    def _choose_buildings_file(self) -> None:
        if self._loading:
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            _("Buildings from a map"),
            str(self.document.base_dir),
            _("GeoJSON (*.geojson *.json);;All files (*)"),
        )
        if not path:
            return
        try:
            count, properties = footprint_properties(path)
        except (OSError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        if not count:
            self.failed.emit(_("{name} holds no building footprints").format(name=Path(path).name))
            return
        dialog = MapDialog(self, Path(path).name, count, properties)
        if dialog.exec():
            choice = dialog.choice()
            self.run(lambda: self.document.set_buildings_file(path, **choice))

    def _remove_buildings_file(self) -> None:
        self.run(lambda: self.document.set_buildings_file(None))

    def _remove_zone(self) -> None:
        if not self.zones.rowCount():
            return
        row = self.zones.currentRow()

        def change(data: dict) -> None:
            ground = data["geometry"]["ground"]
            zones = ground["zones"]
            zones.pop(row if 0 <= row < len(zones) else -1)
            if not zones:
                del ground["zones"]

        self.apply(_("remove zone"), change)

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
        geometry = self.document.scene.geometry
        if kind == "buildings" and geometry.buildings_file is not None:
            names = geometry.building_names()
            index = row if 0 <= row < len(names) else len(names) - 1
            if index >= len(geometry.buildings or []):
                # A building from the file: left out there, by name.
                name = names[index]

                def leave_out(data: dict) -> None:
                    spec = data["geometry"]["buildings_file"]
                    spec["without"] = [*(spec.get("without") or []), name]
                    settings = data["simulation"]
                    kept = [n for n in settings.get("insertion_loss_of") or [] if n != name]
                    if kept:
                        settings["insertion_loss_of"] = kept
                    else:
                        settings.pop("insertion_loss_of", None)

                self.apply(_("leave out building"), leave_out)
                return

        def change(data: dict) -> None:
            items = data["geometry"][kind]
            index = row if 0 <= row < len(items) else len(items) - 1
            before = [_part_name(kind, k, item) for k, item in enumerate(items)]
            items.pop(index)
            # The insertion loss keeps to the parts that stay, under their new names (an
            # unnamed part's name is its place in the list).
            renamed = dict(
                zip(
                    before[:index] + before[index + 1 :],
                    [_part_name(kind, k, item) for k, item in enumerate(items)],
                    strict=True,
                )
            )
            settings = data["simulation"]
            kept = [
                renamed.get(name, name)
                for name in settings.get("insertion_loss_of") or []
                if name != before[index]
            ]
            if kept:
                settings["insertion_loss_of"] = kept
            else:
                settings.pop("insertion_loss_of", None)
            if not items:
                del data["geometry"][kind]

        what = {
            "buildings": _("remove building"),
            "barriers": _("remove barrier"),
            "vegetation": _("remove vegetation"),
        }
        self.apply(what[kind], change)


#: The vegetation the Geometry form draws, as the plan view names it, with its label.
PLANT_SHAPES = [
    ("belt", N_("belt of trees")),
    ("hedge", N_("hedge")),
    ("tree", N_("lone tree")),
]


class MapDialog(QDialog):
    """How to read the buildings of a map: the property that holds their heights (or
    one height for all of them), the height of those without it, and the property
    that names them. :meth:`choice` gives :meth:`Document.set_buildings_file` its
    arguments."""

    def __init__(self, parent, name: str, count: int, properties: dict[str, str]) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Buildings from a map"))
        layout = QVBoxLayout(self)
        intro = QLabel(
            ngettext(
                "{name} holds {count} building. How high is it, and what is it called?",
                "{name} holds {count} buildings. How high are they, and what are they called?",
                count,
            ).format(name=name, count=count)
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.height = QComboBox()
        self.height.setToolTip(
            _(
                "The property of each building that holds its height (one that holds a "
                "number in every building that has it), or one height for all of them"
            )
        )
        numbers = [key for key, kind in properties.items() if kind == "number"]
        for key in numbers:
            self.height.addItem(key, key)
        # Translators: the choice of one height for every building of the map.
        self.height.addItem(_("the same for all"), None)
        self.height.setCurrentIndex(
            numbers.index("height") if "height" in numbers else len(numbers)
        )
        self.height.currentIndexChanged.connect(self._relabel)
        self.metres = spin(0.1, LARGEST, 1, 1.0)
        self.metres.setValue(NEW_PART_SIZES["building"]["height"])
        self.metres.setToolTip(
            _(
                "The height of every building, or of those whose property is missing, in "
                "the geometry's units"
            )
        )
        self.metres_label = QLabel()
        self.metres_label.setToolTip(self.metres.toolTip())
        self.name = QComboBox()
        self.name.setToolTip(
            _("The property that names each building, or numbered names for all of them")
        )
        for key in properties:
            self.name.addItem(key, key)
        self.name.addItem(_("building_1, building_2, …"), None)
        keys = list(properties)
        self.name.setCurrentIndex(keys.index("name") if "name" in keys else len(keys))
        height_label = QLabel(_("Height"))
        height_label.setToolTip(self.height.toolTip())
        form.addRow(height_label, self.height)
        form.addRow(self.metres_label, self.metres)
        name_label = QLabel(_("Name"))
        name_label.setToolTip(self.name.toolTip())
        form.addRow(name_label, self.name)
        layout.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._relabel()

    def _relabel(self) -> None:
        same = self.height.currentData() is None
        self.metres_label.setText(_("Height of all") if same else _("Height when missing"))

    def choice(self) -> dict:
        """The arguments of :meth:`Document.set_buildings_file` the user chose."""
        key = self.height.currentData()
        out = {"name": self.name.currentData() or "name"}
        if key is None:
            out["height"] = float(self.metres.value())
        else:
            out["height"] = key
            out["default_height"] = float(self.metres.value())
        return out


def _part_name(kind: str, index: int, item: dict) -> str:
    """The name of a building, barrier or vegetation in the scene data (as
    :meth:`~acoustic_spaces.scene.Geometry.part_names` gives it)."""
    prefix = {"buildings": "building", "barriers": "barrier", "vegetation": "vegetation"}[kind]
    return item.get("name") or f"{prefix}_{index + 1}"


def _plant_shape(plant) -> str:
    """A vegetation volume's shape, in words."""
    if plant.kind == "belt":
        return fmt(_("belt, {count} corners"), count=len(plant.footprint))
    if plant.kind == "hedge":
        return fmt(
            _("hedge {width} wide, {count} points"),
            width=number(plant.width),
            count=len(plant.path),
        )
    return fmt(_("tree, crown {diameter} across"), diameter=number(plant.crown_diameter))


def _plant_fill(plant) -> str:
    """What fills a vegetation volume, in words."""
    parts = []
    foliage = plant.foliage
    if foliage.preset == "dense":
        parts.append(_("dense foliage"))
    elif foliage.leaf_area_per_m3 is not None:
        parts.append(
            fmt(
                _("leaves {area} m²/m³, {width} m wide"),
                area=number(foliage.leaf_area_per_m3),
                width=number(foliage.leaf_width),
            )
        )
    elif not foliage.bare:
        values = "/".join(number(v) for v in foliage.attenuation)
        parts.append(fmt(_("foliage {values} dB/m"), values=values))
    if plant.stems is not None:
        parts.append(
            fmt(
                _("{count} stems/m², {diameter} m across"),
                count=number(plant.stems.per_m2),
                diameter=number(plant.stems.diameter),
            )
        )
    return ", ".join(parts)


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


#: The octave bands the early paths may add in pressure up to; those above
#: INDOOR_COHERENT_MAX_BAND_HZ outdoors only.
COHERENT_BANDS_HZ = (63, 125, 250, 500, 1000, 2000, 4000, 8000)

#: Fields a material keeps when its coefficients or database entry change.
_KEPT = ("transmission", "transmission_loss", "audience")
#: The ground presets (materials.ground.PRESETS) as the material page lists them, and
#: a flow resistivity of one's own.
_GROUNDS = (
    ("snow", N_("snow")),
    ("forest_floor", N_("forest floor")),
    ("loose_soil", N_("loose soil")),
    ("grass", N_("grass")),
    ("compacted_soil", N_("compacted soil")),
    ("gravel", N_("gravel")),
    ("asphalt", N_("asphalt")),
    ("water", N_("water")),
    ("", N_("flow resistivity below")),
)
#: The impedance models (materials.ground.MODELS).
_GROUND_MODELS = (
    ("miki", N_("Miki")),
    ("delany_bazley", N_("Delany-Bazley")),
    ("two_parameter", N_("two-parameter")),
)
_REACTIONS = (
    ("local", N_("local: the same at every angle")),
    ("extended", N_("extended: changes with the angle")),
)


def _without_transmission(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if k not in ("transmission", "transmission_loss")}


class MaterialPage(Page):
    """A material: its name, the database entry it comes from, its coefficients, and
    what it lets through: nothing, part of the sound (the sound reduction index R per
    band) or everything (an opening)."""

    def __init__(self, document) -> None:
        super().__init__(document)
        #: The material the ground controls last showed an impedance of.
        self._ground_of = None
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
        self.ground = QGroupBox(_("Ground impedance"))
        self.ground.setCheckable(True)
        self.ground.setToolTip(
            _(
                "Ticked, the material is a porous ground (grass, soil, snow, gravel, "
                "asphalt): its absorption and its reflection come from its impedance, with "
                "the phase and the ground wave that make the dip over the ground outdoors. "
                "Unticked, the absorption is as given"
            )
        )
        self.ground.toggled.connect(self._ground_toggled)
        ground_form = QFormLayout(self.ground)
        self.ground_kind = QComboBox()
        _fill(self.ground_kind, _GROUNDS)
        self.ground_kind.setToolTip(
            _(
                "The kind of ground: the ground classes of Nord2000, from snow (softest) to "
                "water and dense concrete (hardest), each with its flow resistivity; or a "
                "flow resistivity of your own"
            )
        )
        self.ground_kind.activated.connect(lambda _index: self._set_impedance())
        self.ground_model = QComboBox()
        _fill(self.ground_model, _GROUND_MODELS)
        self.ground_model.setToolTip(
            _(
                "How the impedance follows from the flow resistivity: Miki (passive at every "
                "frequency, the default), Delany-Bazley (as most of the literature quotes "
                "it), or two-parameter (a porosity that falls with depth, which fits "
                "measured grass and soil better)"
            )
        )
        self.ground_model.activated.connect(lambda _index: self._set_impedance())
        self.flow = spin(0.1, 1e7, 1, 10.0, "kPa s/m²")
        self.flow.setToolTip(
            _(
                "How hard it is to push air through the ground: about 10 to 30 kPa s/m² for "
                "snow, 200 for grass, 2000 for gravel, 20000 and more for asphalt"
            )
        )
        self.flow.valueChanged.connect(lambda _value: self._set_impedance())
        self.thickness = spin(0.0, 100.0, 3, 0.01, "m")
        # Translators: a ground layer with no thickness set: as deep as it likes.
        self.thickness.setSpecialValueText(_("deep"))
        self.thickness.setToolTip(
            _(
                "A layer of the ground this thick on a hard backing (snow on frozen ground, "
                "a thin soil over rock); deep: as deep as it likes"
            )
        )
        self.thickness.valueChanged.connect(lambda _value: self._set_impedance())
        self.porosity = spin(0.0, 1e4, 1, 10.0, "1/m")
        self.porosity.setToolTip(
            _("Two-parameter model: how fast the ground's porosity falls with depth")
        )
        self.porosity.valueChanged.connect(lambda _value: self._set_impedance())
        self.reaction = QComboBox()
        _fill(self.reaction, _REACTIONS)
        self.reaction.setToolTip(
            _(
                "How the ground reacts to sound meeting it at an angle: locally, with the "
                "same impedance at every angle (the usual model, right for grass and "
                "harder grounds); or extended, sound also travelling sideways inside it, so "
                "the impedance a reflection meets changes with its angle, which tells for "
                "soft snow, moss, forest floors and thin layers. Not with the two-parameter "
                "model"
            )
        )
        self.reaction.activated.connect(lambda _index: self._set_impedance())
        ground_form.addRow(_("Ground"), self.ground_kind)
        ground_form.addRow(_("Model"), self.ground_model)
        ground_form.addRow(_("Flow resistivity"), self.flow)
        ground_form.addRow(_("Layer thickness"), self.thickness)
        ground_form.addRow(_("Porosity rate"), self.porosity)
        ground_form.addRow(_("Reaction"), self.reaction)
        layout.addWidget(self.ground)
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
        impedance = material.impedance
        self.ground.setChecked(impedance is not None)
        self.preset.setEnabled(impedance is None)
        self.passing.setEnabled(impedance is None)
        if impedance is not None:
            _choose(self.ground_kind, impedance.preset or "")
            _choose(self.ground_model, impedance.model)
            self.flow.setValue(impedance.resistivity)
            self.thickness.setValue(impedance.thickness or 0.0)
            self.porosity.setValue(impedance.porosity_rate or 0.0)
            _choose(self.reaction, impedance.reaction or "local")
            for col in range(len(labels)):  # the impedance gives the absorption
                item = self.table.item(0, col)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._ground_of = self.key
        elif self._ground_of != self.key:  # ticked again, the same as before; else grass
            self._ground_of = None
            _choose(self.ground_kind, "grass")
            _choose(self.ground_model, "miki")
            self.flow.setValue(PRESETS["grass"][0])
            self.thickness.setValue(0.0)
            self.porosity.setValue(0.0)
            _choose(self.reaction, "local")
        # Set after the group's tick, which enables its controls again when ticked.
        self.flow.setEnabled(impedance is not None and impedance.preset is None)
        two = impedance is not None and impedance.model == "two_parameter"
        self.thickness.setEnabled(impedance is not None and not two)
        self.porosity.setEnabled(two)
        self.reaction.setEnabled(impedance is not None and not two)
        # Translators: {source} is where the coefficients come from, as the material
        # database (in English) or the scene file says it.
        if impedance is not None:
            self.source.setText(_("Source: the ground's impedance"))
        elif material.source:
            self.source.setText(_("Source: {source}").format(source=material.source))
        else:
            self.source.setText(_("Source: values given in the scene"))
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

    def _impedance(self) -> dict:
        """The impedance the ground controls show, as the scene file writes it."""
        out: dict = {}
        kind = self.ground_kind.currentData()
        if not kind:
            out["flow_resistivity"] = self.flow.value()
        else:
            out["preset"] = kind
        model = self.ground_model.currentData()
        if model != "miki":
            out["model"] = model
        if model == "two_parameter":
            out["porosity_rate"] = self.porosity.value()
        else:
            if self.thickness.value() > 0.0:
                out["thickness"] = self.thickness.value()
            if self.reaction.currentData() == "extended":
                out["reaction"] = "extended"
        return out

    def _set_impedance(self) -> None:
        if self._loading or not self.ground.isChecked():
            return
        material = self.document.scene.materials[self.key]
        fields = {"impedance": self._impedance(), "scattering": material.scattering}
        if material.audience:
            fields["audience"] = True
        self.run(lambda: self.document.set_material(self.key, fields))

    def _ground_toggled(self, checked: bool) -> None:
        if self._loading:
            return
        if checked:
            self._set_impedance()
            return
        # The absorption the impedance gave stays, as values of the material's own.
        fields = self._fields()
        fields.pop("impedance", None)
        fields["source"] = "edited from a ground impedance"
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
        if material.impedance is not None:  # only the scattering is typed
            if rows[1] == [round(v, 2) for v in convert(material.scattering, bands)]:
                return
            fields = {"impedance": self._impedance(), "scattering": rows[1]}
            if material.audience:
                fields["audience"] = True
            self.run(lambda: self.document.set_material(self.key, fields))
            return
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
        self.diffraction_reflections = QComboBox()
        _fill(self.diffraction_reflections, DIFFRACTION_REFLECTIONS)
        self.diffraction_reflections.setToolTip(
            _(
                "Where a path over one edge may reflect: on one side of the edge, before or "
                "after it (off the ground on the way to a barrier, or off a façade behind "
                "it), or on both sides, before and after (off the ground on the way to a "
                "barrier and again behind it: all four paths over a barrier on the ground)"
            )
        )
        self.coherent = QComboBox()
        _fill(self.coherent, [(None, N_("none"))] + [(f, str(f)) for f in COHERENT_BANDS_HZ])
        self.coherent.setToolTip(
            _(
                "The highest octave band (Hz) in which the early paths add in pressure, with "
                "the phase of every reflection, rather than in energy; the wave solver's bands "
                "take their place when it runs up to this band or higher. Up to 500 Hz in a "
                "room; outdoors, where the paths are few, up to 8000 Hz, the default there"
            )
        )
        early_form.addRow(_("Method"), self.early_method)
        early_form.addRow(_("Highest order"), self.max_order)
        early_form.addRow(_("Rays"), self.early_rays)
        early_form.addRow(_("Diffraction"), self.diffraction)
        early_form.addRow(_("Reflections over an edge"), self.diffraction_reflections)
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
        self.wave_wind = QComboBox()
        _fill(self.wave_wind, WAVE_WINDS)
        self.wave_wind.setToolTip(
            _(
                "Outdoors, how it takes the wind: as a speed of sound that the wind along the "
                "way from the source changes, as the rays do, or as air that moves, so that "
                "sound going against the wind is slower whichever way it turned, off a "
                "building too (three to four times the work, on the CPU)"
            )
        )
        wave_form.addRow(self.wave)
        wave_form.addRow(_("Up to band (Hz)"), self.wave_band)
        wave_form.addRow(_("Wind"), self.wave_wind)
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

        self.insertion_group = QGroupBox(_("Insertion loss"))
        insertion_layout = QVBoxLayout(self.insertion_group)
        self.insertion = QListWidget()
        self.insertion.setToolTip(
            _(
                "Outdoors: tick buildings, barriers or vegetation to run the scene, and the "
                "map, again without them. The results and the map then give their insertion "
                "loss, the level without them less the level with them; each run takes twice "
                "as long"
            )
        )
        self.insertion.setMaximumHeight(140)
        insertion_layout.addWidget(self.insertion)
        layout.addWidget(self.insertion_group)
        layout.addStretch()
        self.insertion.itemChanged.connect(self._insertion)

        for combo, key in (
            (self.early_method, "early.method"),
            (self.diffraction, "early.diffraction"),
            (self.diffraction_reflections, "early.diffraction_reflections"),
            (self.late_method, "late.method"),
            (self.reception, "late.reception"),
            (self.scattering_method, "scattering.method"),
            (self.ray_tree, "performance.ray_tree"),
            (self.bands, "bands"),
            (self.wave_wind, "wave.wind"),
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
        _choose(self.diffraction_reflections, s.early.diffraction_reflections)
        # Only diffraction with reflections has paths that reflect on both sides.
        self.diffraction_reflections.setEnabled(
            s.early.diffraction in ("first_order", "second_order")
        )
        coherent = s.early.coherent_max_band_hz
        outdoor = self.document.scene.geometry.ground is not None
        for row in range(self.coherent.count()):  # the bands above 500 Hz are outdoors only
            value = self.coherent.itemData(row)
            if value is not None and value > INDOOR_COHERENT_MAX_BAND_HZ:
                self.coherent.model().item(row).setEnabled(outdoor)
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
        _choose(self.wave_wind, s.wave.wind)
        self.wave_wind.setEnabled(s.wave.enabled and outdoor)  # wind blows outdoors only
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
        parts = self.document.scene.geometry.part_names() if outdoor else []
        chosen = set(s.insertion_loss_of or [])
        self.insertion_group.setVisible(bool(parts))
        self.insertion.clear()
        for name in parts:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name in chosen else Qt.CheckState.Unchecked)
            item.setToolTip(
                _("Ticked, the run is also made without {name}, for its insertion loss").format(
                    name=name
                )
            )
            self.insertion.addItem(item)
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

    def _insertion(self) -> None:
        names = [
            self.insertion.item(row).text()
            for row in range(self.insertion.count())
            if self.insertion.item(row).checkState() == Qt.CheckState.Checked
        ]

        def change(data: dict) -> None:
            if names:
                data["simulation"]["insertion_loss_of"] = names
            else:
                data["simulation"].pop("insertion_loss_of", None)

        self.apply(_("change the insertion loss"), change)

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
    draw = Signal(str)

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
            page.draw.connect(self.draw)
        self.kind = ""

    def show_item(self, kind: str, key=None) -> None:
        page = self.pages.get(kind, self.pages[""])
        self.kind = kind if kind in self.pages else ""
        self.setCurrentWidget(page)
        page.show_item(key)

    def refresh(self) -> None:
        self.currentWidget().reload()


__all__ = ["Inspector", "MapDialog", "Page", "label_text"]
