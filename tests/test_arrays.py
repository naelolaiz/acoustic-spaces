"""Coherent arrays: their far-field patterns against the analytic sums of line arrays,
steering, polarity, element patterns, and arrays given in a scene or a YAML file."""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.air import speed_of_sound
from acoustic_spaces.directivity import (
    SPEED_OF_SOUND,
    FirstOrder,
    array_directivity,
    array_length,
    far_field_distance,
    read_directivity,
)
from acoustic_spaces.materials import THIRD_OCTAVE_BANDS_HZ
from acoustic_spaces.materials.bands import crossover_frequencies
from acoustic_spaces.scene import ArraySpec, DirectivitySpec

C = SPEED_OF_SOUND
SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"


def _edges() -> np.ndarray:
    inner = crossover_frequencies("third_octave")
    return np.concatenate([[inner[0] / 2 ** (1 / 3)], inner, [inner[-1] * 2 ** (1 / 3)]])


def _band_mean(function, band: int, count: int = 4000) -> np.ndarray:
    """The mean of function(f) over a third-octave band, by brute force."""
    low, high = _edges()[band : band + 2]
    frequencies = low + (high - low) * (np.arange(count) + 0.5) / count
    return np.mean([function(f) for f in frequencies], axis=0)


def _elevation_cut(balloon, elevations_deg) -> np.ndarray:
    el = np.radians(elevations_deg)
    return balloon.energy(np.column_stack([np.cos(el), 0 * el, np.sin(el)]))


def test_a_column_has_the_line_array_directivity_index() -> None:
    n, d = 8, 0.2
    balloon = array_directivity(ArraySpec(count=n, spacing_m=d))
    assert balloon.table.shape[-1] == 24 and balloon.name == "array of 8 elements"
    # On axis the elements are in phase (n^2); over the sphere, pairs r apart add
    # sin(kr) / kr: the directivity factor of points, averaged over each band.
    r = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) * d

    def spread(f):
        return np.sum(np.sinc(2 * f * r / C))  # numpy's sinc is sin(pi x) / (pi x)

    expected = [10 * np.log10(n**2 / _band_mean(spread, b, 400)) for b in range(24)]
    np.testing.assert_allclose(balloon.index(), expected, atol=0.05)
    # 0 dB in the bass, 10 log10(n) = 9 dB once the spacing is over a wavelength.
    assert balloon.index()[0] < 0.5 and abs(balloon.index()[-1] - 9.03) < 0.2


def test_a_column_has_the_line_array_pattern() -> None:
    n, d = 8, 0.2
    balloon = array_directivity(ArraySpec(count=n, spacing_m=d))
    elevations = np.linspace(-80, 80, 161)
    for band in (9, 12, 15):  # 400 Hz, 800 Hz and 1.6 kHz
        f_c = THIRD_OCTAVE_BANDS_HZ[band]

        def sum_of(f, elevations=elevations):
            psi = 2 * math.pi * f * d * np.sin(np.radians(elevations)) / C
            return np.abs(np.exp(1j * np.outer(psi, np.arange(n))).sum(axis=1)) ** 2

        analytic = _band_mean(sum_of, band)
        ours = _elevation_cut(balloon, elevations)[:, band]
        level = 10 * np.log10(analytic / analytic.max())
        loud = level > -15
        np.testing.assert_allclose(
            10 * np.log10(ours / ours.max())[loud], level[loud], atol=0.3, err_msg=f"{f_c} Hz"
        )
        # Seen from the side the column is a point: the same level all round.
        side = balloon.energy([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]])[:, band]
        np.testing.assert_allclose(side, side[0], rtol=1e-6)


def test_steering_tilts_the_beam() -> None:
    balloon = array_directivity(ArraySpec(count=16, spacing_m=0.1, steer_deg=-12.0))
    elevations = np.arange(-40.0, 40.5, 0.5)
    for band in (15, 18):  # 1.6 and 3.15 kHz
        cut = _elevation_cut(balloon, elevations)[:, band]
        assert elevations[np.argmax(cut)] == pytest.approx(-12.0, abs=1.0)
    # The same as delays given element by element: 0.1 m sin(12 deg) / c apart.
    step = 0.1 * math.sin(math.radians(12.0)) / C * 1e3
    column = ArraySpec(count=16, spacing_m=0.1).layout()
    elements = [e.model_copy(update={"delay_ms": k * step}) for k, e in enumerate(column)]
    np.testing.assert_allclose(
        array_directivity(ArraySpec(elements=elements)).table, balloon.table, rtol=1e-6
    )


def test_element_patterns_and_polarity() -> None:
    one = array_directivity(ArraySpec(element="cardioid", elements=[{"position": [0, 0, 0]}]))
    np.testing.assert_allclose(one.index(), FirstOrder.pattern("cardioid").index()[0], atol=0.05)
    # Two close elements in opposite polarity: a dipole, a figure-8 in the bass.
    pair = ArraySpec(
        elements=[{"position": [0.05, 0, 0]}, {"position": [-0.05, 0, 0], "invert": True}]
    )
    dipole = array_directivity(pair)
    assert dipole.index()[0] == pytest.approx(10 * math.log10(3), abs=0.05)
    assert dipole.energy([[0, 1.0, 0]])[0, 0] < 1e-3  # nothing to the side
    # Two cardioids aimed 30 degrees up and down add up, close together, to
    # 1 + cos(30) cos(theta): a wider first-order pattern than one cardioid's.
    splayed = ArraySpec(
        element="cardioid",
        elements=[
            {"position": [0, 0, 0.01], "orientation": {"pitch": 30}},
            {"position": [0, 0, -0.01], "orientation": {"pitch": -30}},
        ],
    )
    a = 1 / (1 + math.cos(math.radians(30)))
    expected = 10 * math.log10(3 / (3 * a**2 + (1 - a) ** 2))
    assert array_directivity(splayed).index()[0] == pytest.approx(expected, abs=0.05)
    assert expected < FirstOrder.pattern("cardioid").index()[0] - 0.3


def test_arrays_check_their_layout() -> None:
    with pytest.raises(ValidationError, match="not both"):
        ArraySpec(count=4, spacing_m=0.1, elements=[{"position": [0, 0, 0]}])
    with pytest.raises(ValidationError, match="needs both count and spacing_m"):
        ArraySpec(count=4)
    with pytest.raises(ValidationError, match="cannot be arrays"):
        ArraySpec(count=2, spacing_m=0.1, element={"array": {"count": 2, "spacing_m": 0.1}})
    with pytest.raises(ValidationError, match="a file or an array"):
        DirectivitySpec(pattern=["omni"] * 8, array={"count": 2, "spacing_m": 0.1})


def test_arrays_in_scenes_and_files(tmp_path: Path) -> None:
    (tmp_path / "column.yaml").write_text(
        "# a column loudspeaker: 6 cardioid drivers 12 cm apart, aimed 5 degrees down\n"
        "element: cardioid\ncount: 6\nspacing_m: 0.12\nsteer_deg: -5\n",
        encoding="utf-8",
    )
    from_file = read_directivity(tmp_path / "column.yaml")
    inline = array_directivity(ArraySpec(element="cardioid", count=6, spacing_m=0.12, steer_deg=-5))
    np.testing.assert_allclose(from_file.table, inline.table)
    (tmp_path / "bad.yaml").write_text("count: 6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.yaml: not an array description"):
        read_directivity(tmp_path / "bad.yaml")

    scene = Scene.from_yaml(SHOEBOX)
    data = scene.model_dump(mode="json")
    data["geometry"]["file"] = str(SHOEBOX.parent / "shoebox.obj")
    data["sources"][0]["directivity"] = {"file": "column.yaml"}
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    room = Room.from_yaml(tmp_path / "scene.yaml")
    np.testing.assert_allclose(room.directivities[0].table, from_file.table, rtol=1e-9)
    # The phases follow the air's temperature: a warmer room, a slightly wider beam
    # (below 2 kHz, where the 12 cm spacing is under a wavelength).
    data["sources"][0]["directivity"] = {"array": {"count": 6, "spacing_m": 0.12}}
    data["air"]["temperature_c"] = 35.0
    warm = Room.from_scene(Scene.model_validate(data, context={"base_dir": tmp_path}))
    cold = array_directivity(ArraySpec(count=6, spacing_m=0.12))
    assert np.all(warm.directivities[0].index()[10:17] < cold.index()[10:17] - 0.05)


def test_an_array_aims_its_sound(tmp_path: Path) -> None:
    # Free field (one absorbing triangle 100 m away), a listener ahead of a column and
    # one 60 degrees above, both 4 m away: their levels differ as its pattern does.
    (tmp_path / "far.obj").write_text("v 100 0 0\nv 100 0.1 0\nv 100 0 0.1\nf 1 2 3\n")
    up = (4 * math.cos(math.radians(60)), 0.0, 4 * math.sin(math.radians(60)))

    def run(directivity):
        scene = Scene.model_validate(
            {
                "geometry": {"file": str(tmp_path / "far.obj")},
                "materials": {"absorber": {"absorption": [1.0] * 8, "scattering": 0.0}},
                "default_material": "absorber",
                "sources": [{"name": "s", "position": [0, 0, 0], "directivity": directivity}],
                "listeners": [
                    {"name": "ahead", "position": [4, 0, 0]},
                    {"name": "above", "position": list(up)},
                ],
                "simulation": {"ir_length_s": 0.1, "late": {"rays": 500}, "seed": 1},
            }
        )
        room = Room.from_scene(scene)
        result = simulate(room)
        return room, [r.parameters().g for r in result]

    room, (ahead, above) = run({"array": {"count": 8, "spacing_m": 0.2}})
    _, (omni_ahead, omni_above) = run("omni")
    pattern = room.directivity("s")
    expected = 10 * np.log10(pattern.energy([[1.0, 0, 0], np.divide(up, 4)]))
    np.testing.assert_allclose(ahead - omni_ahead, expected[0], atol=1e-6)
    np.testing.assert_allclose(above - omni_above, expected[1], atol=1e-6)
    # A beam from 500 Hz up; above 1.7 kHz, where the 20 cm spacing is more than a
    # wavelength, grating lobes send some sound upwards too.
    assert np.all((ahead - above)[3:5] > 15) and np.all((ahead - above)[5:] > 8)


def test_listeners_near_an_array_are_warned_about(tmp_path: Path) -> None:
    # The far field starts about the length squared over the wavelength away: 16 m for
    # a 2 m array at 1.4 kHz, 3.2 m for a column of 8 drivers 15 cm apart at 1 kHz.
    assert far_field_distance(2.0, 1400.0, 343.0) == pytest.approx(16.3, abs=0.05)
    (tmp_path / "column.yaml").write_text("count: 8\nspacing_m: 0.15\n", encoding="utf-8")
    from_file = DirectivitySpec.model_validate(
        {"file": "column.yaml"}, context={"base_dir": tmp_path}
    )
    assert array_length(from_file) == pytest.approx(1.05)
    assert array_length(DirectivitySpec(array=ArraySpec(count=3, spacing_m=0.5))) == 1.0
    assert array_length("omni") is None
    assert array_length(DirectivitySpec(pattern=["cardioid"] * 8)) is None

    scene = Scene.from_yaml(SHOEBOX)
    data = scene.model_dump(mode="json")
    data["geometry"]["file"] = str(SHOEBOX.parent / "shoebox.obj")
    source = data["sources"][0]
    near = {**data["listeners"][0], "name": "near", "position": [4.0, 3.0, 1.5]}
    data["listeners"].append(near)

    def warned(directivity) -> list[str]:
        source["directivity"] = directivity
        room = Room.from_scene(Scene.model_validate(data, context={"base_dir": tmp_path}))
        return [w for w in room.warnings() if "far-field" in w]

    reach = far_field_distance(1.05, 1000.0, speed_of_sound(scene.air.temperature_c))
    assert 3.0 < reach < 3.4  # the shoebox's own listener, 4 m away, is beyond it
    for directivity in ({"file": "column.yaml"}, {"array": {"count": 8, "spacing_m": 0.15}}):
        (message,) = warned(directivity)
        assert message.startswith(f"listener 'near' is 2.0 m from array {source['name']!r}")
        assert "(1.05 m long)" in message and f"about {reach:.1f} m at 1000 Hz" in message
    assert warned("omni") == [] and warned({"array": {"count": 4, "spacing_m": 0.15}}) == []
