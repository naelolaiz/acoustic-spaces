"""Acoustic material data: the built-in database and coefficient files."""

from acoustic_spaces.materials.bands import (
    OCTAVE_BANDS_HZ,
    THIRD_OCTAVE_BANDS_HZ,
    band_centres,
    to_bands,
    to_octave_bands,
)
from acoustic_spaces.materials.database import MaterialDatabase, MaterialEntry, builtin_database
from acoustic_spaces.materials.files import read_coefficients_csv

__all__ = [
    "OCTAVE_BANDS_HZ",
    "THIRD_OCTAVE_BANDS_HZ",
    "MaterialDatabase",
    "MaterialEntry",
    "band_centres",
    "builtin_database",
    "read_coefficients_csv",
    "to_bands",
    "to_octave_bands",
]
