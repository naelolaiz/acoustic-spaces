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
from acoustic_spaces.gui.tips import (
    band_headers,
    band_tip,
    label_for,
    row_headers,
    share_tips,
)
from acoustic_spaces.headtracking import DEFAULT_OSC_PORT, HeadOrientation, OscReceiver
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import MORE_PARAMETERS, PARAMETERS

#: Smallest size of a plot (pixels); a smaller panel scrolls instead of squashing it.
PLOT_SIZE = (720, 480)
#: Rows of the parameter table: name, label, number format. After the first eight comes
#: the sound pressure level at the source's sound power; the rest show only where the
#: response gives them (stage support needs a listener near the source).
ROWS = [
    ("edt", "EDT (s)", "{:.2f}"),
    ("t20", "T20 (s)", "{:.2f}"),
    ("t30", "T30 (s)", "{:.2f}"),
    ("c50", "C50 (dB)", "{:.1f}"),
    ("c80", "C80 (dB)", "{:.1f}"),
    ("d50", "D50", "{:.2f}"),
    ("ts", "Ts (ms)", "{:.0f}"),
    ("g", "G (dB)", "{:.1f}"),
    ("spl", "SPL (dB)", "{:.1f}"),
    ("lf", "LF", "{:.2f}"),
    ("lfc", "LFC", "{:.2f}"),
    ("lj", "LJ (dB)", "{:.1f}"),
    ("st_early", "ST early (dB)", "{:.1f}"),
    ("st_late", "ST late (dB)", "{:.1f}"),
    ("t30_x", "T30 X (s)", "{:.2f}"),
    ("t30_y", "T30 Y (s)", "{:.2f}"),
    ("t30_z", "T30 Z (s)", "{:.2f}"),
]
assert {name for name, _, _ in ROWS[:8]} == set(PARAMETERS)
assert {name for name, _, _ in ROWS[9:]} <= set(MORE_PARAMETERS)
#: What each row of the parameter table means, for its tip.
ROW_TIPS = {
    "edt": "Early decay time (s): how long the sound would take to die away by 60 dB at "
    "the rate of its first 10 dB; the closest to how reverberant the room sounds",
    "t20": "Reverberation time (s): how long the sound takes to die away by 60 dB, from "
    "the decay between -5 and -25 dB",
    "t30": "Reverberation time (s): how long the sound takes to die away by 60 dB, from "
    "the decay between -5 and -35 dB",
    "c50": "Clarity for speech (dB): the energy of the first 50 ms over the energy after. "
    "Higher is clearer",
    "c80": "Clarity for music (dB): the energy of the first 80 ms over the energy after. "
    "Higher is clearer",
    "d50": "Definition (0 to 1): the share of all the energy that arrives in the first 50 ms",
    "ts": "Centre time (ms): the mean arrival time of the energy. Lower is clearer",
    "g": "Strength (dB): how loud the source is here, over the same source 10 m away in free field",
    "spl": "Sound pressure level (dB re 20 µPa) here, with the source at the sound power "
    "level its form sets",
    "lf": "Early lateral energy fraction (0 to 1): the share of the early sound (5 to "
    "80 ms) that comes from the sides, which makes the source sound wider",
    "lfc": "Early lateral energy fraction as LF, with each arrival weighted by the cosine "
    "of its angle to the side rather than its square",
    "lj": "Late lateral sound level (dB): the level of the sound from the sides after "
    "80 ms, which surrounds the listener",
    "st_early": "Early stage support (dB), for a listener within 1.5 m of the source: "
    "the energy from 20 to 100 ms over that of the first 10 ms, how well musicians hear "
    "each other",
    "st_late": "Late stage support (dB), for a listener within 1.5 m of the source: the "
    "energy from 100 to 1000 ms over that of the first 10 ms, how much of the room "
    "musicians hear",
    "t30_x": "T30 (s) of a figure-of-eight microphone facing forward from the listener. "
    "Longer than T30 where sound keeps running between two hard walls along that line",
    "t30_y": "T30 (s) of a figure-of-eight microphone facing left from the listener. "
    "Longer than T30 where sound keeps running between two hard walls along that line",
    "t30_z": "T30 (s) of a figure-of-eight microphone facing up from the listener. Longer "
    "than T30 where sound keeps running between floor and ceiling",
}
assert set(ROW_TIPS) == {name for name, _, _ in ROWS}
#: How often the head controls show the latest orientation (ms).
HEAD_REFRESH_MS = 100


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
        self.yaw.setToolTip("Turn the head (degrees): left is positive")
        self.pitch = QSlider(Qt.Orientation.Horizontal)
        self.pitch.setRange(-90, 90)
        self.pitch.setToolTip("Tilt the head (degrees): up is positive")
        self.centre = QPushButton("Centre")
        self.centre.setToolTip("Take the head tracker's current orientation as straight ahead")
        self.port = QSpinBox()
        self.port.setRange(0, 65535)
        self.port.setSpecialValueText("off")
        self.port.setValue(DEFAULT_OSC_PORT)
        self.port.setToolTip(
            "UDP port of this computer a head tracker sends OSC to "
            "(/ypr, /quaternion, /yaw, /pitch, /roll)"
        )
        self.facing = QLabel()
        self.facing.setToolTip(
            "Where the head faces now (degrees), and how many messages the head tracker sent"
        )
        for label, widget in (("Yaw", self.yaw), ("Pitch", self.pitch)):
            layout.addWidget(label_for(label, widget))
            layout.addWidget(widget, 1)
        layout.addWidget(self.centre)
        layout.addWidget(label_for("OSC port", self.port))
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
                self.message.emit(f"Cannot listen for a head tracker on UDP port {port}: {exc}")
            else:
                self.message.emit(f"Listening for a head tracker on UDP port {port} (OSC).")
        self.refresh()

    def refresh(self) -> None:
        """Show the latest orientation."""
        yaw, pitch, roll = self.head.angles()
        for slider, value in ((self.yaw, yaw), (self.pitch, pitch)):
            if not slider.isSliderDown():
                slider.blockSignals(True)
                slider.setValue(round(value))
                slider.blockSignals(False)
        text = f"facing yaw {yaw:.0f}°, pitch {pitch:.0f}°, roll {roll:.0f}°"
        if self.receiver is not None:
            text += f"; {self.receiver.received} tracker messages"
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
            "The source whose results the panel shows; selecting a source in the scene "
            "shows its results too"
        )
        self.listener = QComboBox()
        self.listener.setToolTip(
            "The listener whose results the panel shows; selecting a listener in the scene "
            "shows its results too"
        )
        self.band = QComboBox()
        self.band.addItems(BAND_LABELS)
        self.band.setCurrentIndex(OCTAVE_BANDS_HZ.index(1000))
        for index, centre in enumerate(OCTAVE_BANDS_HZ):
            self.band.setItemData(index, band_tip(centre), Qt.ItemDataRole.ToolTipRole)
        self.band.setToolTip(
            "The octave band (a range of frequencies, named by its centre) the echogram and "
            "the arrival directions show. The parameter table and the decay curves show "
            "every band"
        )
        self.paths = QCheckBox("Paths in 3D")
        self.paths.setToolTip(
            "Draw the early reflection paths from this source to this listener in the 3D "
            "view, coloured by order"
        )
        self.tracking = QCheckBox("Head tracking")
        self.tracking.setToolTip(
            "Listen binaurally through Ambisonics that turns with the head: by hand, or as "
            "a head tracker reports it"
        )
        self.export = QPushButton("Export…")
        self.export.setToolTip(
            "Write every response of the run to a folder: a WAV file per output format and "
            "parameters.json, as acoustic-spaces simulate does"
        )
        self.listen = QPushButton("Listen…")
        self.listen.setToolTip(
            "Play a dry WAV recording through this response on the sound card, binaurally "
            "(needs the measure extra)"
        )
        self.stop = QPushButton("Stop")
        self.stop.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.stop.setToolTip("Stop playing (Esc)")
        self.stop.setEnabled(False)
        for label, widget in (("Source", self.source), ("Listener", self.listener)):
            bar.addWidget(label_for(label, widget))
            bar.addWidget(widget, 1)
        bar.addWidget(label_for("Band", self.band))
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
        self.table.setVerticalHeaderLabels([label for _, label, _ in ROWS])
        band_headers(self.table)
        row_headers(self.table, [ROW_TIPS[name] for name, _, _ in ROWS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table, "Parameters")
        self.tabs.setTabToolTip(
            0, "Room parameters (ISO 3382-1) in every octave band; rest on a row for its meaning"
        )
        self.echogram = QScrollArea()
        self.tabs.addTab(self.echogram, "Echogram and decay")
        self.tabs.setTabToolTip(
            1,
            "The energy arriving over time in the chosen band, with the early reflections "
            "coloured by order, and the decay curves of every band",
        )
        self.directions = QScrollArea()
        self.tabs.addTab(self.directions, "Directions")
        self.tabs.setTabToolTip(
            2, "Where the sound comes from, as the listener faces, in the chosen band"
        )
        for holder in (self.echogram, self.directions):
            holder.setWidgetResizable(True)
            holder.setWidget(QWidget())
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details, "Run details")
        self.tabs.setTabToolTip(
            3, "What the run did: its settings per source, and the paths it found per pair"
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
        self._note("" if result is not None else "Run the simulation (F5) to see results here.")
        shown = [combo.currentText() for combo in (self.source, self.listener)]
        for combo in (self.source, self.listener):
            combo.blockSignals(True)
            combo.clear()
        if result is not None:
            sources = list(dict.fromkeys(s for s, _ in result.responses))
            listeners = list(dict.fromkeys(lis for _, lis in result.responses))
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
            self._note("The scene changed since this run; run again to update the results.")
        elif reweighted:
            self._note(
                "The absorption changed since this run: these results re-weight its rays. "
                "Run again to trace them afresh."
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
        for row, (name, _, fmt) in enumerate(ROWS):
            if name == "spl":
                values = spl
            else:
                values = getattr(parameters, name) * (1e3 if name == "ts" else 1.0)
            tip = ROW_TIPS[name]
            for col, value in enumerate(values):
                defined = np.isfinite(value)
                item = QTableWidgetItem(fmt.format(value) if defined else "–")
                item.setToolTip(tip if defined else f"{tip}. Not defined for this response")
                self.table.setItem(row, col, item)
            self.table.setRowHidden(row, row > 8 and not np.isfinite(values).any())

    def _details(self) -> None:
        if self.result is None:
            self.details.setPlainText("")
            return
        lines = []
        for source, info in self.result.details.items():
            parts = [f"{key} = {value:g}" for key, value in info.items()]
            lines.append(f"{source}: " + ", ".join(parts))
        for (source, listener), response in self.result.responses.items():
            diffracted = int(response.early.diffracted.sum())
            lines.append(
                f"{source} -> {listener}: direct sound at {response.direct_delay_s * 1e3:.1f} ms, "
                f"{len(response.early)} early paths"
                + (f" ({diffracted} diffracted)" if diffracted else "")
            )
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
