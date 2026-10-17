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
from acoustic_spaces.gui.numbers import number_text
from acoustic_spaces.gui.results_view import PLOT_SIZE
from acoustic_spaces.gui.tips import band_tip, label_for, share_tips
from acoustic_spaces.i18n import N_, _
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

#: The source list entry for all sources playing together.
ALL_SOURCES = N_("All sources")


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
        self.source.setToolTip(
            _("The source whose map is on show, or all sources playing together")
        )
        self.parameter = QComboBox()
        self.parameter.setToolTip(
            _("The room parameter the map shows; rest on an entry in the list for its meaning")
        )
        for name, info in MAP_PARAMETERS.items():
            self.parameter.addItem(_(info.label), name)
            self.parameter.setItemData(
                self.parameter.count() - 1, _(info.description), Qt.ItemDataRole.ToolTipRole
            )
        self.parameter.setCurrentIndex(list(MAP_PARAMETERS).index("sti"))
        self.band = QComboBox()
        self.band.addItems(BAND_LABELS)
        self.band.setCurrentIndex(OCTAVE_BANDS_HZ.index(1000))
        for index, centre in enumerate(OCTAVE_BANDS_HZ):
            self.band.setItemData(index, band_tip(centre), Qt.ItemDataRole.ToolTipRole)
        self.band.setToolTip(
            _(
                "The octave band (a range of frequencies, named by its centre) the map shows. "
                "STI and SPL (dB(A)) take every band together, so they have no band"
            )
        )
        self.in_3d = QCheckBox(_("Map in 3D"))
        self.in_3d.setToolTip(_("Lay the map over the room in the 3D view"))
        self.run = QPushButton(_("Run map"))
        self.run.setToolTip(_("Simulate the coverage map of every source (F6)"))
        self.save = QPushButton(_("Save…"))
        self.save.setToolTip(_("Save the map as a PNG picture, or every map as JSON"))
        for label, widget in ((_("Source"), self.source), (_("Parameter"), self.parameter)):
            bar.addWidget(label_for(label, widget))
            bar.addWidget(widget, 1)
        bar.addWidget(label_for(_("Band"), self.band))
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
        self.summary.setToolTip(_("The lowest, highest and median values of the map on show"))
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
        # The list's entries hold the source's name, or "" for all sources together.
        shown = self.source.currentData() if self.source.currentIndex() >= 0 else None
        self.source.blockSignals(True)
        self.source.clear()
        if result is not None:
            if result.combined is not None:
                self.source.addItem(_(ALL_SOURCES), "")
            for name in result.maps:
                self.source.addItem(name, name)
            if shown is not None and self.source.findData(shown) >= 0:
                self.source.setCurrentIndex(self.source.findData(shown))
        self.source.blockSignals(False)
        self._note("" if result is not None else _("Run the coverage map (F6) to see it here."))
        self._shown()

    def mark_stale(self, stale: bool) -> None:
        """Say whether the scene changed since the map was computed."""
        if self.result is None:
            return
        self._note(
            _("The scene changed since this map; run it again to update it.") if stale else ""
        )

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
        if self.source.currentIndex() < 0:
            return None
        return self.result.get(self.source.currentData() or None)

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
                number_text(info.format.format(v))
                for v in (finite.min(), np.median(finite), finite.max())
            )
            self.summary.setText(
                # Translators: {title} names the map on show, such as "STI".
                _("{title}: {low} to {high}, median {median}").format(
                    title=title, low=low, high=high, median=mid
                )
            )
        else:
            self.summary.setText(_("{title}: no values").format(title=title))
        figure = plot_map(current, parameter=self.name(), band=self.band_index(), room=self.room)
        canvas = FigureCanvasQTAgg(figure)
        canvas.setMinimumSize(*PLOT_SIZE)
        self.plot.setWidget(canvas)


__all__ = ["ALL_SOURCES", "MapPanel"]
