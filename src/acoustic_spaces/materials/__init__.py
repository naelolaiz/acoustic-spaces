"""Acoustic material data: the built-in database, coefficient files and the material
libraries of CATT-Acoustic and ODEON."""

from acoustic_spaces.materials.bands import (
    OCTAVE_BANDS_HZ,
    THIRD_OCTAVE_BANDS_HZ,
    band_centres,
    to_bands,
    to_octave_bands,
)
from acoustic_spaces.materials.database import MaterialDatabase, MaterialEntry, builtin_database
from acoustic_spaces.materials.files import read_coefficients_csv
from acoustic_spaces.materials.libraries import LibraryMaterial, find_material, read_library

__all__ = [
    "OCTAVE_BANDS_HZ",
    "THIRD_OCTAVE_BANDS_HZ",
    "LibraryMaterial",
    "MaterialDatabase",
    "MaterialEntry",
    "band_centres",
    "builtin_database",
    "find_material",
    "read_coefficients_csv",
    "read_library",
    "to_bands",
    "to_octave_bands",
]
