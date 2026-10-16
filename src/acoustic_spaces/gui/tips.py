"""Hover tips: what a label, field, button or column header explains when the mouse
rests on it. A label that names a field (its buddy) explains the field."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PySide6.QtWidgets import QLabel, QTableWidget, QWidget

from acoustic_spaces.materials import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ


def _hz(frequency: float) -> str:
    return f"{frequency:.3g} Hz" if frequency < 1000 else f"{frequency / 1000:.3g} kHz"


def band_tip(centre_hz: float, bands: str = "octave") -> str:
    """What a band holds: the frequencies around its (nominal) centre, from the exact
    base-2 centre so that neighbouring bands meet."""
    steps = 1 if bands == "octave" else 3
    exact = 1000.0 * 2.0 ** (round(math.log2(centre_hz / 1000.0) * steps) / steps)
    half = 2.0 ** (0.5 / steps)
    kind = "Octave" if bands == "octave" else "Third-octave"
    return (
        f"{kind} band centred on {_hz(centre_hz)}: the frequencies from "
        f"{_hz(exact / half)} to {_hz(exact * half)}"
    )


def band_headers(table: QTableWidget, bands: str = "octave") -> None:
    """Give each band column of ``table`` (its labels set) the tip of its band."""
    centres = OCTAVE_BANDS_HZ if bands == "octave" else THIRD_OCTAVE_BANDS_HZ
    for col, centre in enumerate(centres[: table.columnCount()]):
        item = table.horizontalHeaderItem(col)
        if item is not None:
            item.setToolTip(band_tip(centre, bands))


def row_headers(table: QTableWidget, tips: Sequence[str]) -> None:
    """Give the rows of ``table`` (their labels set) these tips."""
    for row, text in enumerate(tips):
        item = table.verticalHeaderItem(row)
        if item is not None:
            item.setToolTip(text)


def label_for(text: str, field: QWidget) -> QLabel:
    """A label that names ``field``, for a row of controls outside a form."""
    label = QLabel(text)
    label.setBuddy(field)
    return label


def share_tips(root: QWidget) -> None:
    """Give every label under ``root`` that names a field the field's tip, unless it has
    one of its own, so that resting on the label explains the field too."""
    for label in root.findChildren(QLabel):
        buddy = label.buddy()
        if buddy is not None and not label.toolTip() and buddy.toolTip():
            label.setToolTip(buddy.toolTip())


__all__ = ["band_headers", "band_tip", "label_for", "row_headers", "share_tips"]
