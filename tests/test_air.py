import numpy as np
import pytest

from acoustic_spaces.air import (
    EXACT_OCTAVE_HZ,
    absorption_db_per_m,
    energy_attenuation_per_m,
    speed_of_sound,
)


def test_iso_9613_1_table_at_20c_50_percent() -> None:
    # ISO 9613-1:1993 Table 1, dB/km, 101.325 kPa, 125 Hz to 8 kHz exact midbands.
    expected = [0.445, 1.32, 2.73, 4.66, 9.86, 29.4, 104.0]
    got = absorption_db_per_m(EXACT_OCTAVE_HZ[1:]) * 1000.0
    np.testing.assert_allclose(got, expected, rtol=0.01)


def test_iso_9613_1_other_conditions() -> None:
    assert absorption_db_per_m([1000.0], 10.0, 70.0)[0] * 1000.0 == pytest.approx(3.66, rel=0.01)


def test_energy_attenuation_matches_db() -> None:
    m = energy_attenuation_per_m()
    db = absorption_db_per_m(EXACT_OCTAVE_HZ)
    np.testing.assert_allclose(10.0 * np.log10(np.exp(m)), db, rtol=1e-9)


def test_speed_of_sound() -> None:
    assert speed_of_sound(0.0) == pytest.approx(331.3)
    assert speed_of_sound(20.0) == pytest.approx(343.2, abs=0.1)
