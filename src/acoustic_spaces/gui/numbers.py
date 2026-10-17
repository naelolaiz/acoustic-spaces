"""Numbers in the editor's text written as the computer's regional settings write them
(0,85 in Spain, 0.85 in Mexico), as Qt's own number fields already do."""

from __future__ import annotations

import string
from numbers import Number

from PySide6.QtCore import QLocale


class _Formatter(string.Formatter):
    def format_field(self, value, format_spec: str) -> str:
        text = super().format_field(value, format_spec)
        if isinstance(value, Number) and not isinstance(value, bool):
            text = number_text(text)
        return text


_FORMATTER = _Formatter()


def number_text(text: str) -> str:
    """``text``, a number written with Python's ``.`` and ``,``, with the decimal point
    and the group separator of the current locale."""
    locale = QLocale()
    point, group = locale.decimalPoint(), locale.groupSeparator()
    if (point, group) == (".", ","):
        return text
    return "".join(point if c == "." else group if c == "," else c for c in text)


def fmt(template: str, /, **values) -> str:
    """``template.format(**values)`` with every number in the current locale's way:
    ``fmt(_("{t:.2f} s"), t=1.5)`` is "1,50 s" in Spain."""
    return _FORMATTER.vformat(template, (), values)


def _comma() -> bool:
    return QLocale().decimalPoint() == ","


def number(value: float, spec: str = "g") -> str:
    """``value`` written as ``format(value, spec)`` with the current locale's marks."""
    return number_text(format(value, spec))


def numbers_text(values, spec: str = "g") -> str:
    """Several numbers, apart by "; " where the decimal point is a comma, else ", "."""
    return ("; " if _comma() else ", ").join(number(v, spec) for v in values)


def point_text(position, digits: int = 2) -> str:
    """A point's coordinates, written as :func:`numbers_text` does."""
    return numbers_text([float(v) for v in position], f".{digits}f")


def parse_number(text: str) -> float:
    """A number typed in the current locale's way or with a decimal point; ValueError
    when ``text`` is none."""
    text = text.strip()
    if _comma():
        text = text.replace(",", ".")
    return float(text)


def parse_numbers(text: str) -> list[float]:
    """Numbers typed apart by spaces, or by commas (by semicolons where the decimal
    point is a comma); ValueError when one is not a number."""
    text = text.replace(";", " ") if _comma() else text.replace(",", " ")
    return [parse_number(t) for t in text.split()]


__all__ = [
    "fmt",
    "number",
    "number_text",
    "numbers_text",
    "parse_number",
    "parse_numbers",
    "point_text",
]
