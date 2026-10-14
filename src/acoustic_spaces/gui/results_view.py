"""The results of a run: ISO 3382-1 parameters, echogram and decay curves, arrival
directions and run details for one source and listener; export and listening."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.inspector import BAND_LABELS
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import PARAMETERS

#: Smallest size of a plot (pixels); a smaller panel scrolls instead of squashing it.
PLOT_SIZE = (720, 480)
#: Rows of the parameter table: name, label, number format.
ROWS = [
    ("edt", "EDT (s)", "{:.2f}"),
    ("t20", "T20 (s)", "{:.2f}"),
    ("t30", "T30 (s)", "{:.2f}"),
    ("c50", "C50 (dB)", "{:.1f}"),
    ("c80", "C80 (dB)", "{:.1f}"),
    ("d50", "D50", "{:.2f}"),
    ("ts", "Ts (ms)", "{:.0f}"),
    ("g", "G (dB)", "{:.1f}"),
]
assert {name for name, _, _ in ROWS} == set(PARAMETERS)


class ResultsPanel(QWidget):
    """Shows one response of a :class:`SimulationResult` at a time."""

    #: The source and listener on show changed.
    pair_changed = Signal(str, str)
    #: "Show paths in 3D" was switched.
    paths_toggled = Signal(bool)
    export_requested = Signal()
    listen_requested = Signal()
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.source = QComboBox()
        self.listener = QComboBox()
        self.band = QComboBox()
        self.band.addItems(BAND_LABELS)
        self.band.setCurrentIndex(OCTAVE_BANDS_HZ.index(1000))
        self.paths = QCheckBox("Paths in 3D")
        self.export = QPushButton("Export…")
        self.listen = QPushButton("Listen…")
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        for label, widget in (("Source", self.source), ("Listener", self.listener)):
            bar.addWidget(QLabel(label))
            bar.addWidget(widget, 1)
        bar.addWidget(QLabel("Band"))
        bar.addWidget(self.band)
        bar.addWidget(self.paths)
        bar.addStretch()
        for button in (self.export, self.listen, self.stop):
            bar.addWidget(button)
        layout.addLayout(bar)
        self.stale = QLabel()
        self.stale.setWordWrap(True)
        layout.addWidget(self.stale)

        self.tabs = QTabWidget()
        self.table = QTableWidget(len(ROWS), len(OCTAVE_BANDS_HZ))
        self.table.setHorizontalHeaderLabels(BAND_LABELS)
        self.table.setVerticalHeaderLabels([label for _, label, _ in ROWS])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabs.addTab(self.table, "Parameters")
        self.echogram = QScrollArea()
        self.tabs.addTab(self.echogram, "Echogram and decay")
        self.directions = QScrollArea()
        self.tabs.addTab(self.directions, "Directions")
        for holder in (self.echogram, self.directions):
            holder.setWidgetResizable(True)
            holder.setWidget(QWidget())
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details, "Run details")
        layout.addWidget(self.tabs, 1)

        self.source.currentTextChanged.connect(self._pair)
        self.listener.currentTextChanged.connect(self._pair)
        self.band.currentIndexChanged.connect(self._draw)
        self.tabs.currentChanged.connect(self._draw)
        self.paths.toggled.connect(self.paths_toggled)
        self.export.clicked.connect(self.export_requested)
        self.listen.clicked.connect(self.listen_requested)
        self.stop.clicked.connect(self.stop_requested)
        self.set_result(None)

    # -- content --------------------------------------------------------------------

    def set_result(self, result) -> None:
        """Show ``result`` (a SimulationResult), or nothing. The source and listener on
        show stay when the new result has them."""
        self.result = result
        for widget in (self.export, self.listen, self.source, self.listener):
            widget.setEnabled(result is not None)
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

    def mark_stale(self, stale: bool) -> None:
        """Say whether the scene changed since the results were computed."""
        if self.result is None:
            return
        self._note(
            "The scene changed since this run; run again to update the results." if stale else ""
        )

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
        for row, (name, _, fmt) in enumerate(ROWS):
            values = getattr(parameters, name) * (1e3 if name == "ts" else 1.0)
            for col, value in enumerate(values):
                text = fmt.format(value) if np.isfinite(value) else "–"
                self.table.setItem(row, col, QTableWidgetItem(text))

    def _details(self) -> None:
        if self.result is None:
            self.details.setPlainText("")
            return
        lines = []
        for source, info in self.result.details.items():
            parts = [f"{key} = {value:g}" for key, value in info.items()]
            lines.append(f"{source}: " + ", ".join(parts))
        for (source, listener), response in self.result.responses.items():
            lines.append(
                f"{source} -> {listener}: direct sound at {response.direct_delay_s * 1e3:.1f} ms, "
                f"{len(response.early)} early paths"
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

    def set_playing(self, playing: bool) -> None:
        self.stop.setEnabled(playing)
        self.listen.setEnabled(not playing and self.result is not None)


__all__ = ["PLOT_SIZE", "ROWS", "ResultsPanel"]
