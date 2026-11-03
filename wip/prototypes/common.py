"""Shared helpers for the design-minimal probes (not engine code)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parabolic as P  # noqa: E402  the prototype with obstacles
import wide_angle as W  # noqa: E402
from acoustic_spaces.scene import Atmosphere  # noqa: E402
from acoustic_spaces.engine.diffraction import band_frequencies  # noqa: E402
from acoustic_spaces.materials.ground import surface_impedance  # noqa: E402

F = band_frequencies()[0]  # (8 bands, 8 frequencies)
CENTRES = [63, 125, 250, 500, 1000, 2000, 4000, 8000]


def atmosphere(preset="windy", cosine=1.0):
    atm = Atmosphere.model_validate({"preset": preset, "wind": {"direction_deg": 0}})

    def speed(h):
        h = np.atleast_1d(np.asarray(h, dtype=float))
        return np.asarray(atm.sound_speed(h, 20.0, np.array([cosine, math.sqrt(max(1 - cosine * cosine, 0.0))])), dtype=float)

    return speed


def still(c0):
    def speed(h):
        return np.full(np.shape(np.atleast_1d(h)), float(c0))

    return speed


def ground(kind, f):
    """impedance(r) for the march: 'rigid' or flow resistivity in kPa s/m^2."""
    if kind == "rigid":
        return lambda r: None
    zg = np.conj(surface_impedance(f, float(kind)))
    return lambda r: zg


def band_energy(p, distance):
    """Band energy re free field (linear), p (F, n) -> (n,)."""
    return np.mean(np.abs(p) ** 2, axis=0) * distance**2
