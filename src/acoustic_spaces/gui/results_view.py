"""The results of a run: ISO 3382-1 parameters, echogram and decay curves, arrival
directions and run details for one source and listener; export and listening, also
head-tracked."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.inspector import BAND_LABELS
from acoustic_spaces.gui.numbers import fmt, number_text
from acoustic_spaces.gui.tips import (
    band_headers,
    band_tip,
    label_for,
    row_headers,
    share_tips,
)
from acoustic_spaces.headtracking import DEFAULT_OSC_PORT, HeadOrientation, OscReceiver
from acoustic_spaces.i18n import N_, _, ngettext
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import MORE_PARAMETERS, PARAMETERS

#: Smallest size of a plot (pixels); a smaller panel scrolls instead of squashing it.
PLOT_SIZE = (720, 480)
#: Rows of the parameter table: name, label, number format. After the first eight comes
#: the sound pressure level at the source's sound power; the rest show only where the
#: response gives them: outdoors the level relative to free field, the insertion loss of
#: what the run was also made without and the level over a long term (weather classes),
#: then those of ISO 3382-1 that need directions (stage support needs a listener near
#: the source).
ROWS = [
    ("edt", N_("EDT (s)"), "{:.2f}"),
    ("t20", N_("T20 (s)"), "{:.2f}"),
    ("t30", N_("T30 (s)"), "{:.2f}"),
    ("c50", N_("C50 (dB)"), "{:.1f}"),
    ("c80", N_("C80 (dB)"), "{:.1f}"),
    ("d50", N_("D50"), "{:.2f}"),
    ("ts", N_("Ts (ms)"), "{:.0f}"),
    ("g", N_("G (dB)"), "{:.1f}"),
    ("spl", N_("SPL (dB)"), "{:.1f}"),
    ("re_free_field", N_("Re free field (dB)"), "{:.1f}"),
    ("insertion_loss", N_("Insertion loss (dB)"), "{:.1f}"),
    ("long_term", N_("Long-term level (dB)"), "{:.1f}"),
    ("lf", N_("LF"), "{:.2f}"),
    ("lfc", N_("LFC"), "{:.2f}"),
    ("lj", N_("LJ (dB)"), "{:.1f}"),
    ("st_early", N_("ST early (dB)"), "{:.1f}"),
    ("st_late", N_("ST late (dB)"), "{:.1f}"),
    ("t30_x", N_("T30 X (s)"), "{:.2f}"),
    ("t30_y", N_("T30 Y (s)"), "{:.2f}"),
    ("t30_z", N_("T30 Z (s)"), "{:.2f}"),
]
assert {row[0] for row in ROWS[:8]} == set(PARAMETERS)
#: The outdoor rows, right after the sound pressure level.
OUTDOOR_ROWS = ("re_free_field", "insertion_loss", "long_term")
assert {row[0] for row in ROWS[9:12]} == set(OUTDOOR_ROWS)
assert {row[0] for row in ROWS[12:]} <= set(MORE_PARAMETERS)
#: What each row of the parameter table means, for its tip.
ROW_TIPS = {
    "edt": N_(
        "Early decay time (s): how long the sound would take to die away by 60 dB at "
        "the rate of its first 10 dB; the closest to how reverberant the room sounds"
    ),
    "t20": N_(
        "Reverberation time (s): how long the sound takes to die away by 60 dB, from "
        "the decay between -5 and -25 dB"
    ),
    "t30": N_(
        "Reverberation time (s): how long the sound takes to die away by 60 dB, from "
        "the decay between -5 and -35 dB"
    ),
    "c50": N_(
        "Clarity for speech (dB): the energy of the first 50 ms over the energy after. "
        "Higher is clearer"
    ),
    "c80": N_(
        "Clarity for music (dB): the energy of the first 80 ms over the energy after. "
        "Higher is clearer"
    ),
    "d50": N_("Definition (0 to 1): the share of all the energy that arrives in the first 50 ms"),
    "ts": N_("Centre time (ms): the mean arrival time of the energy. Lower is clearer"),
    "g": N_(
        "Strength (dB): how loud the source is here, over the same source 10 m away in free field"
    ),
    "spl": N_(
        "Sound pressure level (dB re 20 µPa) here, with the source at the sound power "
        "level its form sets"
    ),
    "re_free_field": N_(
        "Level relative to free field (dB), outdoors: how much louder the source is here "
        "than the same source as far away in free field, with the same air. The ground, "
        "the weather, buildings, barriers and vegetation make the difference; negative, "
        "it is their excess attenuation"
    ),
    "insertion_loss": N_(
        "Insertion loss (dB) of the buildings, barriers or vegetation the simulation page "
        "names: the level here without them less the level with them. Positive where they "
        "shield the listener"
    ),
    "long_term": N_(
        "Level over a long term (dB), outdoors: the energy mean of the levels in the "
        "scene's weather classes, each by its share of the time, over the periods of the "
        "day. The Run details tab gives the rating level (L_den) and the level of each "
        "period and each class"
    ),
    "lf": N_(
        "Early lateral energy fraction (0 to 1): the share of the early sound (5 to "
        "80 ms) that comes from the sides, which makes the source sound wider"
    ),
    "lfc": N_(
        "Early lateral energy fraction as LF, with each arrival weighted by the cosine "
        "of its angle to the side rather than its square"
    ),
    "lj": N_(
        "Late lateral sound level (dB): the level of the sound from the sides after "
        "80 ms, which surrounds the listener"
    ),
    "st_early": N_(
        "Early stage support (dB), for a listener within 1.5 m of the source: "
        "the energy from 20 to 100 ms over that of the first 10 ms, how well musicians hear "
        "each other"
    ),
    "st_late": N_(
        "Late stage support (dB), for a listener within 1.5 m of the source: the "
        "energy from 100 to 1000 ms over that of the first 10 ms, how much of the room "
        "musicians hear"
    ),
    "t30_x": N_(
        "T30 (s) of a figure-of-eight microphone facing forward from the listener. "
        "Longer than T30 where sound keeps running between two hard walls along that line"
    ),
    "t30_y": N_(
        "T30 (s) of a figure-of-eight microphone facing left from the listener. "
        "Longer than T30 where sound keeps running between two hard walls along that line"
    ),
    "t30_z": N_(
        "T30 (s) of a figure-of-eight microphone facing up from the listener. Longer "
        "than T30 where sound keeps running between floor and ceiling"
    ),
}
assert set(ROW_TIPS) == {row[0] for row in ROWS}
#: How often the head controls show the latest orientation (ms).
HEAD_REFRESH_MS = 100


def _long_term_line(long_term, key) -> str:
    """One pair's levels over a long term, A-weighted: the rating level, the long-term
    level, and the level in each period and in each weather class."""
    periods = ", ".join(
        fmt("{name} {level:.1f}", name=name, level=level)
        for name, level in zip(long_term.periods, long_term.period_a[key], strict=True)
    )
    classes = ", ".join(
        fmt("{name} {level:.1f}", name=name, level=level)
        for name, level in zip(long_term.classes, long_term.class_a[key], strict=True)
    )
    return fmt(
        # Translators: {rating} is the rating level's symbol (L_den); {periods} and
        # {classes} are names from the scene file, each with its level.
        _(
            "{source} -> {listener}: {rating} {rated:.1f} dB(A), long-term level "
            "{level:.1f} dB(A); {periods} dB(A); in the weather classes {classes} dB(A)"
        ),
        source=key[0],
        listener=key[1],
        rating=long_term.rating,
        rated=long_term.rating_level[key],
        level=long_term.levels_a[key],
        periods=periods,
        classes=classes,
    )


class HeadControls(QWidget):
    """The listener's head for head-tracked listening: yaw and pitch by hand, or from a
    head tracker sending OSC to a UDP port of this computer (off at port 0). Centre takes
    the tracker's current orientation as straight ahead."""

    #: A line for the messages panel.
    message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.head = HeadOrientation()
        self.receiver: OscReceiver | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.yaw = QSlider(Qt.Orientation.Horizontal)
        self.yaw.setRange(-180, 180)
        self.yaw.setToolTip(_("Turn the head (degrees): left is positive"))
        self.pitch = QSlider(Qt.Orientation.Horizontal)
        self.pitch.setRange(-90, 90)
        self.pitch.setToolTip(_("Tilt the head (degrees): up is positive"))
        self.centre = QPushButton(_("Centre"))
        self.centre.setToolTip(_("Take the head tracker's current orientation as straight ahead"))
        self.port = QSpinBox()
        self.port.setRange(0, 65535)
        self.port.setSpecialValueText(_("off"))
        self.port.setValue(DEFAULT_OSC_PORT)
        self.port.setToolTip(
            _(
                "UDP port of this computer a head tracker sends OSC to "
                "(/ypr, /quaternion, /yaw, /pitch, /roll)"
            )
        )
        self.facing = QLabel()
        self.facing.setToolTip(
            _("Where the head faces now (degrees), and how many messages the head tracker sent")
        )
        for label, widget in ((_("Yaw"), self.yaw), (_("Pitch"), self.pitch)):
            layout.addWidget(label_for(label, widget))
            layout.addWidget(widget, 1)
        layout.addWidget(self.centre)
        layout.addWidget(label_for(_("OSC port"), self.port))
        layout.addWidget(self.port)
        layout.addWidget(self.facing, 1)
        share_tips(self)
        self._timer = QTimer(self)
        self._timer.setInterval(HEAD_REFRESH_MS)
        self._timer.timeout.connect(self.refresh)
        self.yaw.valueChanged.connect(self._look)
        self.pitch.valueChanged.connect(self._look)
        self.centre.clicked.connect(self._centre)
        self.port.editingFinished.connect(self.listen)
        self.refresh()

    def start(self) -> None:
        """Follow the head, and listen for a tracker."""
        self._timer.start()
        self.listen()

    def stop(self) -> None:
        """Stop following the head and free the port."""
        self._timer.stop()
        if self.receiver is not None:
            self.receiver.stop()
            self.receiver = None
        self.refresh()

    def listen(self) -> None:
        """(Re)open the OSC port the spin box shows."""
        port = self.port.value()
        if self.receiver is not None:
            if self.receiver.port == port:
                return
            self.receiver.stop()
            self.receiver = None
        if port:
            try:
                self.receiver = OscReceiver(self.head, port).start()
            except OSError as exc:
                self.message.emit(
                    _("Cannot listen for a head tracker on UDP port {port}: {reason}").format(
                        port=port, reason=exc
                    )
                )
            else:
                self.message.emit(
                    _("Listening for a head tracker on UDP port {port} (OSC).").format(port=port)
                )
        self.refresh()

    def refresh(self) -> None:
        """Show the latest orientation."""
        yaw, pitch, roll = self.head.angles()
        for slider, value in ((self.yaw, yaw), (self.pitch, pitch)):
            if not slider.isSliderDown():
                slider.blockSignals(True)
                slider.setValue(round(value))
                slider.blockSignals(False)
        angles = {"yaw": yaw, "pitch": pitch, "roll": roll}
        if self.receiver is None:
            text = fmt(_("facing yaw {yaw:.0f}°, pitch {pitch:.0f}°, roll {roll:.0f}°"), **angles)
        else:
            count = self.receiver.received
            text = fmt(
                ngettext(
                    "facing yaw {yaw:.0f}°, pitch {pitch:.0f}°, roll {roll:.0f}°; {count:,} "
                    "tracker message",
                    "facing yaw {yaw:.0f}°, pitch {pitch:.0f}°, roll {roll:.0f}°; {count:,} "
                    "tracker messages",
                    count,
                ),
                **angles,
                count=count,
            )
        self.facing.setText(text)

    def _look(self) -> None:
        self.head.look(self.yaw.value(), self.pitch.value(), self.head.angles()[2])
        self.refresh()

    def _centre(self) -> None:
        self.head.centre()
        self.refresh()


class ResultsPanel(QWidget):
    """Shows one response of a :class:`SimulationResult` at a time."""

    #: The source and listener on show changed.
    pair_changed = Signal(str, str)
    #: "Show paths in 3D" was switched.
    paths_toggled = Signal(bool)
    export_requested = Signal()
    listen_requested = Signal()
    stop_requested = Signal()
    #: Playback started (True) or stopped (False).
    playing_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.source = QComboBox()
        self.source.setToolTip(
            _(
                "The source whose results the panel shows; selecting a source in the scene "
                "shows its results too"
            )
        )
        self.listener = QComboBox()
        self.listener.setToolTip(
            _(
                "The listener whose results the panel shows; selecting a listener in the scene "
                "shows its results too"
            )
        )
        self.band = QComboBox()
        self.band.addItems(BAND_LABELS)
        self.band.setCurrentIndex(OCTAVE_BANDS_HZ.index(1000))
        for index, centre in enumerate(OCTAVE_BANDS_HZ):
            self.band.setItemData(index, band_tip(centre), Qt.ItemDataRole.ToolTipRole)
        self.band.setToolTip(
            _(
                "The octave band (a range of frequencies, named by its centre) the echogram and "
                "the arrival directions show. The parameter table and the decay curves show "
                "every band"
            )
        )
        self.paths = QCheckBox(_("Paths in 3D"))
        self.paths.setToolTip(
            _(
                "Draw the early reflection paths from this source to this listener in the 3D "
                "view, coloured by order"
            )
        )
        self.tracking = QCheckBox(_("Head tracking"))
        self.tracking.setToolTip(
            _(
                "Listen binaurally through Ambisonics that turns with the head: by hand, or as "
                "a head tracker reports it"
            )
        )
        self.export = QPushButton(_("Export…"))
        self.export.setToolTip(
            _(
                "Write every response of the run to a folder: a WAV file per output format and "
                "parameters.json, as acoustic-spaces simulate does"
            )
        )
        self.listen = QPushButton(_("Listen…"))
        self.listen.setToolTip(
            _(
                "Play a dry WAV recording through this response on the sound card, binaurally "
                "(needs the measure extra)"
            )
        )
        self.stop = QPushButton(_("Stop"))
        self.stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop.setToolTip(_("Stop playing (Esc)"))
        self.stop.setEnabled(False)
        for label, widget in ((_("Source"), self.source), (_("Listener"), self.listener)):
            bar.addWidget(label_for(label, widget))
            bar.addWidget(widget, 1)
        bar.addWidget(label_for(_("Band"), self.band))
        bar.addWidget(self.band)
        bar.addWidget(self.paths)
        bar.addStretch()
        for button in (self.tracking, self.export, self.listen, self.stop):
            bar.addWidget(button)
        layout.addLayout(bar)
        self.head = HeadControls()
        self.head.hide()
        layout.addWidget(self.head)
        self.stale = QLabel()
        self.stale.setWordWrap(True)
        layout.addWidget(self.stale)

        self.tabs = QTabWidget()
        self.table = QTableWidget(len(ROWS), len(OCTAVE_BANDS_HZ))
        self.table.setHorizontalHeaderLabels(BAND_LABELS)
        self.table.setVerticalHeaderLabels([_(row[1]) for row in ROWS])
        band_headers(self.table)
        row_headers(self.table, [_(ROW_TIPS[row[0]]) for row in ROWS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table, _("Parameters"))
        self.tabs.setTabToolTip(
            0, _("Room parameters (ISO 3382-1) in every octave band; rest on a row for its meaning")
        )
        self.echogram = QScrollArea()
        self.tabs.addTab(self.echogram, _("Echogram and decay"))
        self.tabs.setTabToolTip(
            1,
            _(
                "The energy arriving over time in the chosen band, with the early reflections "
                "coloured by order, and the decay curves of every band"
            ),
        )
        self.directions = QScrollArea()
        self.tabs.addTab(self.directions, _("Directions"))
        self.tabs.setTabToolTip(
            2, _("Where the sound comes from, as the listener faces, in the chosen band")
        )
        for holder in (self.echogram, self.directions):
            holder.setWidgetResizable(True)
            holder.setWidget(QWidget())
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details, _("Run details"))
        self.tabs.setTabToolTip(
            3, _("What the run did: its settings per source, and the paths it found per pair")
        )
        layout.addWidget(self.tabs, 1)
        share_tips(self)

        self.source.currentTextChanged.connect(self._pair)
        self.listener.currentTextChanged.connect(self._pair)
        self.band.currentIndexChanged.connect(self._draw)
        self.tabs.currentChanged.connect(self._draw)
        self.paths.toggled.connect(self.paths_toggled)
        self.tracking.toggled.connect(self._tracking)
        self.export.clicked.connect(self.export_requested)
        self.listen.clicked.connect(self.listen_requested)
        self.stop.clicked.connect(self.stop_requested)
        self.set_result(None)

    # -- content --------------------------------------------------------------------

    def set_result(self, result) -> None:
        """Show ``result`` (a SimulationResult), or nothing. The source and listener on
        show stay when the new result has them."""
        self.result = result
        for widget in (self.export, self.source, self.listener):
            widget.setEnabled(result is not None)
        # While something plays, Listen waits for Stop (see set_playing).
        self.listen.setEnabled(result is not None and not self.stop.isEnabled())
        self._note("" if result is not None else _("Run the simulation (F5) to see results here."))
        shown = [combo.currentText() for combo in (self.source, self.listener)]
        for combo in (self.source, self.listener):
            combo.blockSignals(True)
            combo.clear()
        if result is not None:
            sources = list(dict.fromkeys(pair[0] for pair in result.responses))
            listeners = list(dict.fromkeys(pair[1] for pair in result.responses))
            self.source.addItems(sources)
            self.listener.addItems(listeners)
            for combo, name in zip((self.source, self.listener), shown, strict=True):
                if combo.findText(name) >= 0:
                    combo.setCurrentText(name)
        for combo in (self.source, self.listener):
            combo.blockSignals(False)
        self._details()
        self._pair()

    def show_pair(self, source: str | None = None, listener: str | None = None) -> None:
        """Show the results of ``source`` and ``listener`` (None keeps the one on show),
        if the result has them. Does not emit ``pair_changed``."""
        before = self.pair()
        for combo, name in ((self.source, source), (self.listener, listener)):
            if name is not None and combo.findText(name) >= 0:
                combo.blockSignals(True)
                combo.setCurrentText(name)
                combo.blockSignals(False)
        if self.pair() != before:
            self._table()
            self._draw()

    def mark_stale(self, stale: bool, reweighted: bool = False) -> None:
        """Say whether the scene changed since the results were computed, or whether
        they re-weight the run for other absorption."""
        if self.result is None:
            return
        if stale:
            self._note(_("The scene changed since this run; run again to update the results."))
        elif reweighted:
            self._note(
                _(
                    "The absorption changed since this run: these results re-weight its rays. "
                    "Run again to trace them afresh."
                )
            )
        else:
            self._note("")

    def _note(self, text: str) -> None:
        self.stale.setText(text)
        self.stale.setVisible(bool(text))

    def pair(self) -> tuple[str, str] | None:
        if self.result is None:
            return None
        key = (self.source.currentText(), self.listener.currentText())
        return key if key in self.result.responses else None

    def response(self):
        key = self.pair()
        return None if key is None else self.result.responses[key]

    def _pair(self) -> None:
        self._table()
        self._draw()
        key = self.pair()
        if key is not None:
            self.pair_changed.emit(*key)

    def _table(self) -> None:
        self.table.clearContents()
        response = self.response()
        if response is None:
            return
        parameters = response.parameters()
        sources = {s.name: s for s in self.result.room.scene.sources}
        source = sources.get(response.source)
        power = 94.0 if source is None else source.power_db("octave")
        spl = response.levels(power, parameters)
        nan = np.full(len(OCTAVE_BANDS_HZ), np.nan)
        for row, (name, _label, template) in enumerate(ROWS):
            if name == "spl":
                values = spl
            elif name == "re_free_field":
                values = response.relative_to_free_field()
            elif name == "insertion_loss":
                values = self.result.insertion_loss.get(self.pair(), nan)
            elif name == "long_term":
                long_term = self.result.long_term
                values = nan if long_term is None else long_term.levels.get(self.pair(), nan)
            else:
                values = getattr(parameters, name) * (1e3 if name == "ts" else 1.0)
            tip = _(ROW_TIPS[name])
            # Translators: {tip} is what a parameter means.
            undefined = _("{tip}. Not defined for this response").format(tip=tip)
            for col, value in enumerate(values):
                defined = np.isfinite(value)
                item = QTableWidgetItem(number_text(template.format(value)) if defined else "–")
                item.setToolTip(tip if defined else undefined)
                self.table.setItem(row, col, item)
            self.table.setRowHidden(row, row > 8 and not np.isfinite(values).any())

    def _details(self) -> None:
        if self.result is None:
            self.details.setPlainText("")
            return
        lines = []
        for source, info in self.result.details.items():
            parts = [fmt("{key} = {value:g}", key=key, value=value) for key, value in info.items()]
            lines.append(f"{source}: " + "; ".join(parts))
        removed = self.result.removed
        if removed:
            lines.append(
                fmt(
                    ngettext(
                        "Insertion loss of {names}: the scene was also run without it",
                        "Insertion loss of {names}: the scene was also run without them",
                        len(removed),
                    ),
                    names=", ".join(removed),
                )
            )
        long_term = self.result.long_term
        if long_term is not None:
            count = len(long_term.classes)
            lines.append(
                fmt(
                    ngettext(
                        "Levels over a long term in {count} weather class ({names}), "
                        "rated as {rating}",
                        "Levels over a long term in {count} weather classes ({names}), "
                        "rated as {rating}",
                        count,
                    ),
                    count=count,
                    names=", ".join(long_term.classes),
                    rating=long_term.rating,
                )
            )
        for (source, listener), response in self.result.responses.items():
            diffracted = int(response.early.diffracted.sum())
            count = len(response.early)
            values = {
                "source": source,
                "listener": listener,
                "delay": response.direct_delay_s * 1e3,
                "count": count,
                "diffracted": diffracted,
            }
            if diffracted:
                text = ngettext(
                    "{source} -> {listener}: direct sound at {delay:.1f} ms, {count} early "
                    "path ({diffracted} diffracted)",
                    "{source} -> {listener}: direct sound at {delay:.1f} ms, {count} early "
                    "paths ({diffracted} diffracted)",
                    count,
                )
            else:
                text = ngettext(
                    "{source} -> {listener}: direct sound at {delay:.1f} ms, {count} early path",
                    "{source} -> {listener}: direct sound at {delay:.1f} ms, {count} early paths",
                    count,
                )
            lines.append(fmt(text, **values))
            if long_term is not None and (source, listener) in long_term.levels:
                lines.append(_long_term_line(long_term, (source, listener)))
        self.details.setPlainText("\n".join(lines))

    def _replace_canvas(self, holder: QScrollArea, figure) -> None:
        """Show ``figure`` in ``holder``; the scroll area deletes the previous one."""
        if figure is None:
            holder.setWidget(QWidget())
            return
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumSize(*PLOT_SIZE)
        holder.setWidget(canvas)

    def _draw(self) -> None:
        """Draw the plot of the visible tab (plots are drawn when first looked at)."""
        from acoustic_spaces.viz import plot_directions, plot_response

        response = self.response()
        band = self.band.currentIndex()
        tab = self.tabs.currentWidget()
        if tab is self.echogram:
            figure = None if response is None else plot_response(response, band=band)
            self._replace_canvas(self.echogram, figure)
        elif tab is self.directions:
            figure = None if response is None else plot_directions(response, band=band)
            self._replace_canvas(self.directions, figure)

    def _tracking(self, on: bool) -> None:
        self.head.setVisible(on)
        if on:
            self.head.start()
        else:
            self.head.stop()

    def set_playing(self, playing: bool) -> None:
        self.stop.setEnabled(playing)
        self.listen.setEnabled(not playing and self.result is not None)
        self.playing_changed.emit(playing)


__all__ = ["HEAD_REFRESH_MS", "PLOT_SIZE", "ROWS", "ROW_TIPS", "HeadControls", "ResultsPanel"]
