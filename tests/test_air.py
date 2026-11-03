import numpy as np
import pytest

from acoustic_spaces.air import (
    EXACT_OCTAVE_HZ,
    absorption_db_per_m,
    energy_attenuation_per_m,
    speed_of_sound,
    stated_pressures_kpa,
    viscosity,
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


def test_viscosity() -> None:
    # The ICAO standard atmosphere at sea level, 15 degrees: 1.4607e-5 m²/s.
    assert viscosity() == pytest.approx(1.4607e-5, rel=1e-3)
    # Thinner air, the same dynamic viscosity: half the pressure, twice the kinematic.
    assert viscosity(15.0, 101.325 / 2) == pytest.approx(2 * viscosity())


@pytest.mark.parametrize(
    ("bands", "inside", "outside"),
    [
        # 63.096 Hz / 4e-4 Hz/Pa = 157.7 kPa; 7943 Hz / 10 Hz/Pa = 0.794 kPa.
        ("octave", (157.0, 0.80, 101.325), (159.0, 0.79, 200.0)),
        # 50.119 Hz / 4e-4 Hz/Pa = 125.3 kPa; 10 kHz / 10 Hz/Pa = 1.0 kPa.
        ("third_octave", (125.0, 1.0, 101.325), (126.0, 0.99)),
    ],
)
def test_a_pressure_outside_iso_9613_1_s_stated_accuracy_is_warned_about(
    bands, inside, outside
) -> None:
    """ISO 9613-1:1993, clause 7, states the accuracy of its attenuation (±10 % to
    ±50 %) for frequency over pressure from 4e-4 to 10 Hz/Pa, below 200 kPa: a scene
    whose bands leave that range at its pressure is warned about. Any humidity is in
    the range (7.3 takes in a molar concentration of water vapour below 0.005 %, so a
    dry air is no warning)."""
    from acoustic_spaces import Room, Scene

    low, high = stated_pressures_kpa(bands)
    assert (low, high) == pytest.approx(
        {"octave": (0.7943, 157.74), "third_octave": (1.0, 125.30)}[bands], rel=1e-4
    )

    def warned(**air) -> list[str]:
        scene = Scene.model_validate(
            {
                "geometry": {"box": [6.0, 5.0, 3.0]},
                "materials": {"m": {"absorption": [0.2] * 8}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [1.5, 1.5, 1.5]}],
                "listeners": [{"name": "l", "position": [4.5, 3.5, 1.2]}],
                "air": air,
                "simulation": {"bands": bands},
            }
        )
        return [w for w in Room.from_scene(scene).warnings() if "ISO 9613-1" in w]

    for pressure in inside:
        assert warned(pressure_kpa=pressure) == []
    for pressure in outside:
        (warning,) = warned(pressure_kpa=pressure)
        assert f"air.pressure_kpa is {pressure:g} kPa" in warning
        assert "0.0004 to 10 Hz/Pa" in warning
    assert warned(humidity_percent=0.0) == []
