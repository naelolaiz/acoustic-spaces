"""The coverage map of a run: one parameter over the audience area seen from above, for
one source or all of them together, and the choice of what the 3D view shows."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.engine.coverage import MAP_PARAMETERS
from acoustic_spaces.gui.inspector import BAND_LABELS
from acoustic_spaces.gui.results_view import PLOT_SIZE
from acoustic_spaces.gui.tips import band_tip, label_for, share_tips
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

#: The source list entry for all sources playing together.
ALL_SOURCES = "All sources"


class MapPanel(QWidget):
    """Shows one parameter of a :class:`~acoustic_spaces.engine.CoverageResult`."""

    #: What the map shows changed (source, parameter, band, or "Map in 3D").
    changed = Signal()
    run_requested = Signal()
    save_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.result = None
        self.room = None
        layout = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.source = QComboBox()
        self.source.setToolTip("The source whose map is on show, or all sources playing together")
        self.parameter = QComboBox()
        self.parameter.setToolTip(
            "The room parameter the map shows; rest on an entry in the list for its meaning"
        )
        for name, info in MAP_PARAMETERS.items():
            self.parameter.addItem(info.label, name)
            self.parameter.setItemData(
                self.parameter.count() - 1, info.description, Qt.ItemDataRole.ToolTipRole
            )
        self.parameter.setCurrentIndex(list(MAP_PARAMETERS).index("sti"))
        self.band = QComboBox()
        self.band.addItems(BAND_LABELS)
        self.band.setCurrentIndex(OCTAVE_BANDS_HZ.index(1000))
        for index, centre in enumerate(OCTAVE_BANDS_HZ):
            self.band.setItemData(index, band_tip(centre), Qt.ItemDataRole.ToolTipRole)
        self.band.setToolTip(
            "The octave band (a range of frequencies, named by its centre) the map shows. "
            "STI and SPL (dB(A)) take every band together, so they have no band"
        )
        self.in_3d = QCheckBox("Map in 3D")
        self.in_3d.setToolTip("Lay the map over the room in the 3D view")
        self.run = QPushButton("Run map")
        self.run.setToolTip("Simulate the coverage map of every source (F6)")
        self.save = QPushButton("Save…")
        self.save.setToolTip("Save the map as a PNG picture, or every map as JSON")
        for label, widget in (("Source", self.source), ("Parameter", self.parameter)):
            bar.addWidget(label_for(label, widget))
            bar.addWidget(widget, 1)
        bar.addWidget(label_for("Band", self.band))
        bar.addWidget(self.band)
        bar.addWidget(self.in_3d)
        bar.addStretch()
        bar.addWidget(self.run)
        bar.addWidget(self.save)
        layout.addLayout(bar)
        self.note = QLabel()
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.summary = QLabel()
        self.summary.setToolTip("The lowest, highest and median values of the map on show")
        layout.addWidget(self.summary)
        self.plot = QScrollArea()
        self.plot.setWidgetResizable(True)
        self.plot.setWidget(QWidget())
        layout.addWidget(self.plot, 1)

        for combo in (self.source, self.parameter, self.band):
            combo.currentIndexChanged.connect(self._shown)
        self.in_3d.toggled.connect(lambda *_: self.changed.emit())
        self.run.clicked.connect(self.run_requested)
        self.save.clicked.connect(self.save_requested)
        share_tips(self)
        self.set_result(None)

    # -- content --------------------------------------------------------------------

    def set_result(self, result, room=None) -> None:
        """Show ``result`` (a CoverageResult) of ``room``, or nothing. The source on
        show stays when the new result has it."""
        self.result, self.room = result, room
        self.save.setEnabled(result is not None)
        self.source.setEnabled(result is not None)
        shown = self.source.currentText()
        self.source.blockSignals(True)
        self.source.clear()
        if result is not None:
            if result.combined is not None:
                self.source.addItem(ALL_SOURCES)
            self.source.addItems(list(result.maps))
            if self.source.findText(shown) >= 0:
                self.source.setCurrentText(shown)
        self.source.blockSignals(False)
        self._note("" if result is not None else "Run the coverage map (F6) to see it here.")
        self._shown()

    def mark_stale(self, stale: bool) -> None:
        """Say whether the scene changed since the map was computed."""
        if self.result is None:
            return
        self._note("The scene changed since this map; run it again to update it." if stale else "")

    def _note(self, text: str) -> None:
        self.note.setText(text)
        self.note.setVisible(bool(text))

    def set_running(self, running: bool) -> None:
        self.run.setEnabled(not running)

    def name(self) -> str:
        """The parameter on show (a key of ``MAP_PARAMETERS``)."""
        return self.parameter.currentData()

    def band_index(self) -> int:
        return self.band.currentIndex()

    def current(self):
        """The CoverageMap on show, or None."""
        if self.result is None:
            return None
        source = self.source.currentText()
        return self.result.get(None if source == ALL_SOURCES else source)

    def shown_in_3d(self):
        """(map, parameter, band) for the 3D view, or None."""
        current = self.current()
        if current is None or not self.in_3d.isChecked():
            return None
        return current, self.name(), self.band_index()

    def _shown(self) -> None:
        self.band.setEnabled(MAP_PARAMETERS[self.name()].per_band)
        self._draw()
        self.changed.emit()

    def _draw(self) -> None:
        from acoustic_spaces.viz import map_values, plot_map

        current = self.current()
        if current is None:
            self.summary.setText("")
            self.plot.setWidget(QWidget())
            return
        values, title = map_values(current, self.name(), self.band_index())
        finite = values[np.isfinite(values)]
        info = MAP_PARAMETERS[self.name()]
        if len(finite):
            low, mid, high = (
                info.format.format(v) for v in (finite.min(), np.median(finite), finite.max())
            )
            self.summary.setText(f"{title}: {low} to {high}, median {mid}")
        else:
            self.summary.setText(f"{title}: no values")
        figure = plot_map(current, parameter=self.name(), band=self.band_index(), room=self.room)
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumSize(*PLOT_SIZE)
        self.plot.setWidget(canvas)


__all__ = ["ALL_SOURCES", "MapPanel"]
