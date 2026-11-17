from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from acoustic_spaces.cli import main
from acoustic_spaces.directivity import (
    PATTERNS,
    Balloon,
    FirstOrder,
    _sphere,
    array_files,
    load_directivity,
    read_directivity,
    read_directivity_csv,
    write_directivity_csv,
)
from acoustic_spaces.engine import early_paths, trace_particles
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene
from acoustic_spaces.scene.model import Orientation

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
#: A loudspeaker-like source: omni in the bass, narrowing towards the treble.
SPEAKER = (1.0, 0.95, 0.85, 0.7, 0.5, 0.37, 0.3, 0.25)


def _energy_db(directivity, direction) -> np.ndarray:
    return 10.0 * np.log10(directivity.energy([direction])[0])


# -- patterns and balloons -------------------------------------------------------------


@pytest.mark.parametrize("name", list(PATTERNS))
def test_patterns_keep_the_sound_power(name: str) -> None:
    pattern = FirstOrder.pattern(name)
    assert pattern.energy(_sphere(20_000)).mean(axis=0) == pytest.approx(np.ones(8), rel=1e-3)
    assert pattern.omni == (name == "omni")


def test_directivity_index_of_the_patterns() -> None:
    expected = {"omni": 0.0, "subcardioid": 2.84, "cardioid": 4.77, "hypercardioid": 6.02}
    for name, index in expected.items():
        assert FirstOrder.pattern(name).index() == pytest.approx([index] * 8, abs=0.01)
    hyper = FirstOrder.pattern("hypercardioid")
    # The rear lobe of a hypercardioid: half the pressure, 6 dB down.
    front, back = _energy_db(hyper, [1, 0, 0]), _energy_db(hyper, [-1, 0, 0])
    assert front - back == pytest.approx([6.02] * 8, abs=0.01)


def test_first_order_checks_its_values() -> None:
    with pytest.raises(ValueError, match="8 values"):
        FirstOrder((0.5,) * 7)
    with pytest.raises(ValueError, match="between 0 and 1"):
        FirstOrder((1.5,) * 8)


def test_a_balloon_interpolates_measured_directions() -> None:
    pattern = FirstOrder(SPEAKER)
    measured = _sphere(2000)
    balloon = Balloon.from_measurements(measured, 7.0 * pattern.energy(measured), "speaker")
    probe = _sphere(5000)
    assert balloon.energy(probe).mean(axis=0) == pytest.approx(np.ones(8), rel=1e-3)
    np.testing.assert_allclose(balloon.energy(probe), pattern.energy(probe), atol=0.05)
    assert balloon.index() == pytest.approx(pattern.index(), abs=0.05)
    a, table = balloon.native()
    assert a.tolist() == [1.0] * 8 and table.shape == (37, 72, 8)


def test_a_balloon_fills_silent_bands_and_refuses_bad_data() -> None:
    directions = _sphere(200)
    energy = np.ones((200, 8))
    energy[:, 0] = 0.0
    balloon = Balloon.from_measurements(directions, energy)
    assert balloon.extrapolated == (63,)
    assert balloon.energy(directions)[:, 0] == pytest.approx(np.ones(200))
    with pytest.raises(ValueError, match="non-negative"):
        Balloon.from_measurements(directions, -energy)
    with pytest.raises(ValueError, match="8 bands"):
        Balloon.from_measurements(directions, energy[:, :4])


def _lattice(azimuths, elevations) -> np.ndarray:
    """Unit vectors (elevations, azimuths, 3) towards every pair of the angles (degrees)."""
    az, el = np.meshgrid(np.radians(azimuths), np.radians(elevations))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def test_a_balloon_takes_the_directions_equally_near_a_point_alike() -> None:
    """A table measured every 10 degrees of azimuth and 2.5 of elevation, from -180 to
    180: on the equator, a grid point halfway between two meridians has two directions
    equally near and four more equally near after them. It is the mean of the two; the
    table of a pattern that is the same on the left and the right, above and below, is
    so too, and the order of the rows changes nothing."""
    azimuths, elevations = np.arange(-180.0, 181.0, 10.0), np.arange(-90.0, 90.1, 2.5)
    measured = _lattice(azimuths, elevations)
    off = np.arccos(np.clip(measured[..., 0], -1.0, 1.0))
    tall = 1.0 + 3.0 * measured[..., 2] ** 2
    energy = np.stack([np.exp(-(off**2) * (1.0 + band) / tall) + 0.01 for band in range(8)], -1)
    balloon = Balloon.from_measurements(measured.reshape(-1, 3), energy.reshape(-1, 8))
    table = balloon.table  # elevations -90 to 90, azimuths 0 to 355, every 5 degrees
    np.testing.assert_allclose(table[:, 1:], table[:, :0:-1], rtol=1e-12)  # left and right
    np.testing.assert_allclose(table, table[::-1], rtol=1e-12)  # above and below
    # The equator's row 36 of the measured elevations, its azimuths 0 and 10 at 18 and 19;
    # the grid's row 18, its azimuths 0, 5 and 10 at 0, 1 and 2.
    mean = 0.5 * (energy[36, 18] + energy[36, 19])
    np.testing.assert_allclose(table[18, 1] / table[18, 0], mean / energy[36, 18], rtol=1e-9)
    np.testing.assert_allclose(table[18, 2] / table[18, 0], energy[36, 19] / energy[36, 18])
    order = np.random.default_rng(1).permutation(measured.shape[0] * measured.shape[1])
    shuffled = Balloon.from_measurements(
        measured.reshape(-1, 3)[order], energy.reshape(-1, 8)[order]
    )
    np.testing.assert_allclose(shuffled.table, table, rtol=1e-12)


def test_a_direction_measured_more_than_once_counts_once() -> None:
    """The rows at -180 and 180 degrees of azimuth and those at a pole are one direction
    each, with the mean of their energies; a ring of directions equally near a point,
    with none nearer, is taken whole."""
    azimuths, elevations = np.arange(-180.0, 181.0, 10.0), np.arange(-90.0, 91.0, 10.0)
    measured = _lattice(azimuths, elevations)
    energy = np.random.default_rng(2).uniform(0.2, 3.0, (*measured.shape[:2], 8))
    table = Balloon.from_measurements(measured.reshape(-1, 3), energy.reshape(-1, 8)).table
    # On the equator (row 9 of the measured, 18 of the grid) the direction behind is the
    # mean of its two rows, and the pole above that of its 37.
    behind = 0.5 * (energy[9, 0] + energy[9, -1])
    np.testing.assert_allclose(table[18, 36] / table[18, 0], behind / energy[9, 18], rtol=1e-9)
    above = energy[-1].mean(axis=0)
    np.testing.assert_allclose(table[-1] / table[18, 0], above / energy[9, 18] * np.ones((72, 1)))
    # Halfway between 170 degrees (35 of the measured, 34 of the grid) and behind, the
    # direction behind counts as much as the one at 170 degrees, not twice.
    beside = 0.5 * (energy[9, 35] + behind)
    np.testing.assert_allclose(table[18, 35] / table[18, 0], beside / energy[9, 18], rtol=1e-9)
    # Without the rows given twice, with the means in their place, it is the same table.
    once, levels = measured[1:-1, 1:], energy[1:-1, 1:].copy()
    levels[:, -1] = 0.5 * (energy[1:-1, 0] + energy[1:-1, -1])
    poles = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    same = Balloon.from_measurements(
        np.concatenate([once.reshape(-1, 3), poles]),
        np.concatenate([levels.reshape(-1, 8), [energy[0].mean(axis=0), above]]),
    )
    np.testing.assert_allclose(same.table, table, rtol=1e-12)
    # Two rings 10 degrees from the poles and nothing nearer: each pole is its ring's mean.
    rings = _lattice(np.arange(0.0, 360.0, 10.0), [-80.0, 80.0])
    levels = np.random.default_rng(4).uniform(0.2, 3.0, (2, 36, 8))
    table = Balloon.from_measurements(rings.reshape(-1, 3), levels.reshape(-1, 8)).table
    np.testing.assert_allclose(
        table[-1, 0] / table[0, 0], levels[1].mean(axis=0) / levels[0].mean(axis=0)
    )
    for none in (np.nan, np.inf, 0.0):
        with pytest.raises(ValueError, match="finite directions"):
            Balloon.from_measurements([[none, 0.0, 0.0], [0.0, 1.0, 0.0]], np.ones((2, 8)))
    # A direction is its vector's, whatever the vector's length.
    long = measured.reshape(-1, 3) * np.random.default_rng(3).choice([1e-200, 0.3, 1e200], (703, 1))
    scaled = Balloon.from_measurements(long, energy.reshape(-1, 8)).table
    np.testing.assert_allclose(
        scaled,
        Balloon.from_measurements(measured.reshape(-1, 3), energy.reshape(-1, 8)).table,
        rtol=1e-12,
    )


def test_directions_joined_by_others_are_one_in_any_order() -> None:
    """Three directions in a row, each within a millionth of a radian of the next and
    the ends further apart, are one direction, the mean of them, whichever comes first."""
    measured = _lattice(np.arange(0.0, 360.0, 10.0), np.arange(-80.0, 81.0, 10.0)).reshape(-1, 3)
    energy = np.random.default_rng(5).uniform(0.2, 3.0, (len(measured), 8))
    # Beside the row at 10 degrees of azimuth and of elevation (9 * 36 + 1 of the measured).
    more = _lattice(10.0 + np.degrees(0.9e-6) * np.arange(1.0, 3.0), [10.0]).reshape(-1, 3)
    extra = np.random.default_rng(6).uniform(0.2, 3.0, (2, 8))
    last = Balloon.from_measurements(
        np.concatenate([measured, more]), np.concatenate([energy, extra])
    ).table
    first = Balloon.from_measurements(
        np.concatenate([more[::-1], measured]), np.concatenate([extra[::-1], energy])
    ).table
    np.testing.assert_allclose(first, last, rtol=1e-9)
    # The grid's point there (row 20, azimuth 2) against the one straight ahead (the
    # measured row 8 * 36), both on a direction.
    mean = (energy[325] + extra.sum(axis=0)) / 3.0
    np.testing.assert_allclose(last[20, 2] / last[18, 0], mean / energy[288], rtol=1e-6)


def test_a_balloon_weighs_by_the_inverse_squares_of_the_angles() -> None:
    """Two directions 20 degrees apart: the grid's point 5 degrees from one and 15 from
    the other takes them 9 to 1."""
    rad = np.radians(20.0)
    measured = [[1.0, 0.0, 0.0], [np.cos(rad), np.sin(rad), 0.0]]
    energy = np.array([np.full(8, 1.0), np.full(8, 3.0)])
    table = Balloon.from_measurements(measured, energy).table
    np.testing.assert_allclose(table[18, 1] / table[18, 0], (9 * 1.0 + 3.0) / 10, rtol=1e-9)


def test_a_balloon_takes_four_directions_and_groups_whole() -> None:
    """Round the grid's point straight ahead, one direction 2 degrees away and others 6
    degrees away. With three of those the four are taken; with four they do not all fit
    among the four, and none of them is taken; and a direction a ten-thousandth of a
    radian further than the three is not as near as they are."""

    def around(degrees: float, turns: list[float]) -> np.ndarray:
        """Unit vectors ``degrees`` from straight ahead, each a turn (degrees) round it."""
        off, turn = np.radians(degrees), np.radians(turns)
        ahead = np.full(turn.shape, np.cos(off))
        return np.stack([ahead, np.sin(off) * np.cos(turn), np.sin(off) * np.sin(turn)], axis=-1)

    def ahead(others: np.ndarray, levels: list[float]) -> float:
        """The point straight ahead against the one behind, which is on a direction of
        energy 1, as is the direction 2 degrees from straight ahead."""
        measured = np.concatenate([[[-1.0, 0.0, 0.0]], around(2.0, [50.0]), others])
        energy = np.outer([1.0, 1.0, *levels], np.ones(8))
        table = Balloon.from_measurements(measured, energy).table
        return float(table[18, 0, 0] / table[18, 36, 0])

    three = around(6.0, [10.0, 130.0, 250.0])
    four = (1.0 / 2**2 + (2.0 + 3.0 + 5.0) / 6**2) / (1.0 / 2**2 + 3.0 / 6**2)
    assert ahead(three, [2.0, 3.0, 5.0]) == pytest.approx(four, rel=1e-9)
    ring = around(6.0, [10.0, 100.0, 190.0, 280.0])
    assert ahead(ring, [2.0, 3.0, 5.0, 7.0]) == pytest.approx(1.0, rel=1e-9)
    further = around(6.0 + np.degrees(1e-4), [300.0])
    assert ahead(np.concatenate([three, further]), [2.0, 3.0, 5.0, 7.0]) == pytest.approx(
        four, rel=1e-9
    )


def test_a_balloon_of_fewer_directions_than_neighbours_once_merged() -> None:
    """Four rows, two of them one direction: three directions, the merged one with the
    mean of its two energies."""
    measured = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    energy = np.array([np.full(8, 1.0), np.full(8, 3.0), np.full(8, 1.0), np.full(8, 1.0)])
    table = Balloon.from_measurements(measured, energy).table
    np.testing.assert_allclose(table[18, 0] / table[-1, 0], 2.0)  # ahead against the pole


# -- files ---------------------------------------------------------------------------


def _write_csv(path: Path, pattern: FirstOrder, frequencies=OCTAVE_BANDS_HZ[1:]) -> Path:
    """Levels of ``pattern`` on a 10 degree grid, one column per frequency."""
    lines = [
        "# made-up data for a test",
        "azimuth, elevation, " + ", ".join(f"{f} Hz" for f in frequencies),
    ]
    for elevation in range(-90, 91, 10):
        for azimuth in range(0, 360, 10):
            el, az = np.radians(elevation), np.radians(azimuth)
            v = [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
            bands = pattern.energy([v])[0]
            levels = [
                10 * np.log10(max(bands[OCTAVE_BANDS_HZ.index(f)], 1e-12)) for f in frequencies
            ]
            lines.append(f"{azimuth}, {elevation}, " + ", ".join(f"{x:.3f}" for x in levels))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_csv_tables(tmp_path: Path) -> None:
    pattern = FirstOrder(SPEAKER)
    balloon = read_directivity(_write_csv(tmp_path / "speaker.csv", pattern))
    assert balloon.name == "file speaker.csv"
    # No 63 Hz column: that octave copies 125 Hz.
    assert balloon.extrapolated == (63,)
    expected = pattern.index()
    expected[0] = expected[1]
    assert balloon.index() == pytest.approx(expected, abs=0.1)


def test_a_balloon_written_and_read_back(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    balloon = Balloon.from_measurements(_sphere(300), rng.uniform(0.1, 2.0, (300, 8)))
    path = write_directivity_csv(tmp_path / "made.csv", balloon, "made up\nfor a test")
    assert path.read_text(encoding="utf-8").startswith("# made up\n# for a test\nazimuth")
    back = read_directivity_csv(path)
    assert back.extrapolated == ()
    np.testing.assert_allclose(back.table, balloon.table, rtol=2e-3)


def test_csv_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("az, el, 1000\n0, 0, 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        read_directivity_csv(bad)
    bad.write_text("azimuth, elevation, 1000\n0, 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2 values for 3 columns"):
        read_directivity_csv(bad)
    bad.write_text("azimuth, elevation, 1000\n0, 0, loud\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected numbers"):
        read_directivity_csv(bad)
    bad.write_text("azimuth, elevation, 20000\n0, 0, 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no directivity data"):
        read_directivity_csv(bad)
    with pytest.raises(ValueError, match="not found"):
        read_directivity(tmp_path / "missing.sofa")


def _write_sofa(path: Path, amplitude, *, impulse: bool = False, view=(0.0, 1.0, 0.0)) -> Path:
    """A FreeFieldDirectivityTF-like file: microphones 2 m around a loudspeaker that
    looks along ``view``; ``amplitude(m, directions)`` gives each measurement's pressure
    (one value per microphone, the same at every frequency)."""
    h5py = pytest.importorskip("h5py")
    grid = [(az, el) for el in range(-80, 81, 10) for az in range(0, 360, 10)]
    grid += [(0, -90), (0, 90)]
    spherical = np.array([[az, el, 2.0] for az, el in grid], dtype=float)
    az, el = np.radians(spherical[:, 0]), np.radians(spherical[:, 1])
    world = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pressure = np.stack([amplitude(m, world) for m in range(2)])  # (M, R)
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "SOFA"
        f.attrs["SOFAConventions"] = "FreeFieldDirectivityTF"
        f.attrs["Title"] = "Test speaker"
        receiver = f.create_dataset("ReceiverPosition", data=spherical[:, :, None])
        receiver.attrs["Type"] = "spherical"
        receiver.attrs["Units"] = "degree, degree, metre"
        f.create_dataset("ListenerPosition", data=np.zeros((1, 3)))
        f.create_dataset("SourcePosition", data=np.zeros((1, 3)))
        f.create_dataset("SourceView", data=np.array([view], dtype=float))
        f.create_dataset("SourceUp", data=np.array([[0.0, 0.0, 1.0]]))
        if impulse:
            ir = np.zeros((2, len(grid), 64))
            ir[:, :, 3] = pressure
            f.create_dataset("Data.IR", data=ir)
            f.create_dataset("Data.SamplingRate", data=np.array([48_000.0]))
        else:
            frequencies = np.array([100.0, 1000.0, 4000.0])
            f.create_dataset("N", data=frequencies)
            f.create_dataset("Data.Real", data=np.repeat(pressure[:, :, None], 3, axis=2))
            f.create_dataset("Data.Imag", data=np.zeros((2, len(grid), 3)))
    return path


def _cardioid_towards_y(m: int, world: np.ndarray) -> np.ndarray:
    return 0.5 + 0.5 * world[:, 1] if m == 1 else np.ones(len(world))


def test_sofa_transfer_functions_in_the_source_frame(tmp_path: Path) -> None:
    path = _write_sofa(tmp_path / "speaker.sofa", lambda m, w: 0.5 + 0.5 * w[:, 1])
    balloon = read_directivity(path)
    assert balloon.name == "SOFA Test speaker"
    # The speaker looks along +y, so +y in the file is the front of the balloon.
    assert balloon.index() == pytest.approx([4.77] * 8, abs=0.1)
    # Its left is -x in the file, where a cardioid has half its on-axis pressure.
    assert _energy_db(balloon, [0, 1, 0]) == pytest.approx([-1.25] * 8, abs=0.1)
    assert balloon.energy([[-1, 0, 0]])[0] == pytest.approx(np.zeros(8), abs=0.02)
    # Data at 100 Hz, 1 kHz and 4 kHz: the other octaves copy the nearest.
    assert balloon.extrapolated == (63, 250, 500, 2000, 8000)


def test_sofa_impulse_responses_and_measurements(tmp_path: Path) -> None:
    path = _write_sofa(tmp_path / "ir.sofa", _cardioid_towards_y, impulse=True, view=(1, 0, 0))
    omni, cardioid, both = (read_directivity(path, m) for m in (0, 1, None))
    assert omni.index() == pytest.approx([0.0] * 8, abs=0.05)
    assert _energy_db(cardioid, [0, 1, 0]) == pytest.approx([4.77] * 8, abs=0.3)
    # All measurements: their energies averaged (cardioid has a third of the power of
    # omni here), then normalised.
    probe = _sphere(500)
    mixed = (1.0 + (0.5 + 0.5 * probe[:, 1]) ** 2) / 2 / (2 / 3)
    np.testing.assert_allclose(both.energy(probe), mixed[:, None] * np.ones(8), atol=0.05)
    with pytest.raises(ValueError, match="measurement 5 of 2"):
        read_directivity(path, 5)


# -- scenes --------------------------------------------------------------------------


def _shoebox(**source) -> dict:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(SHOEBOX.parent / "shoebox.obj")
    data["sources"][0].update(source)
    return data


def test_scene_directivity_entries(tmp_path: Path) -> None:
    per_band = ["omni", "omni", 0.85, "subcardioid", "cardioid", 0.37, 0.3, "hypercardioid"]
    scene = Scene.model_validate(_shoebox(directivity={"pattern": per_band}))
    directivity = load_directivity(scene.sources[0].directivity)
    assert directivity.native()[0] == pytest.approx([1, 1, 0.85, 0.7, 0.5, 0.37, 0.3, 0.25])
    assert directivity.name.startswith("per band: omni, omni, a=0.85")

    for bad, message in (
        ({"pattern": ["cardioid"] * 7}, "expected 8 pattern values"),
        ({}, "give one of"),
        ({"pattern": per_band, "file": "x.csv"}, "give one of"),
        ({"file": "x.csv", "measurement": 1}, "measurement is for SOFA"),
        ("wide", "omni"),
    ):
        with pytest.raises(ValidationError, match=message):
            Scene.model_validate(_shoebox(directivity=bad))


def test_scene_directivity_files_are_resolved_and_saved_relative(tmp_path: Path) -> None:
    _write_csv(tmp_path / "speaker.csv", FirstOrder.pattern("cardioid"), OCTAVE_BANDS_HZ)
    data = _shoebox(directivity={"file": "speaker.csv"})
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    room = Room.from_yaml(tmp_path / "scene.yaml")
    assert room.scene.sources[0].directivity.file == tmp_path / "speaker.csv"
    assert room.directivity("talker").index() == pytest.approx([4.77] * 8, abs=0.1)
    with pytest.raises(KeyError):
        room.directivity("nobody")
    # An unchanged entry is not read again.
    assert Room.from_scene(room.scene, reuse=room).directivities[0] is room.directivities[0]

    room.scene.to_yaml(tmp_path / "copy.yaml")
    saved = yaml.safe_load((tmp_path / "copy.yaml").read_text(encoding="utf-8"))
    assert saved["sources"][0]["directivity"] == {"file": "speaker.csv"}

    data["sources"][0]["directivity"] = {"file": "missing.csv"}
    scene = Scene.model_validate(data, context={"base_dir": tmp_path})
    with pytest.raises(ValueError, match="source 'talker': directivity file not found"):
        Room.from_scene(scene)


def test_an_inline_array_saves_its_element_file_relative(tmp_path: Path) -> None:
    folder = tmp_path / "scene"
    folder.mkdir()
    _write_csv(folder / "speaker.csv", FirstOrder.pattern("cardioid"), OCTAVE_BANDS_HZ)
    array = {"element": {"file": "speaker.csv"}, "count": 2, "spacing_m": 0.1}
    data = _shoebox(directivity={"array": array})
    (folder / "scene.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    scene = Scene.from_yaml(folder / "scene.yaml")
    assert scene.sources[0].directivity.array.element.file == folder / "speaker.csv"
    scene.to_yaml(folder / "copy.yaml")
    saved = yaml.safe_load((folder / "copy.yaml").read_text(encoding="utf-8"))
    assert saved["sources"][0]["directivity"]["array"]["element"] == {"file": "speaker.csv"}
    # The folder can move.
    folder.rename(tmp_path / "moved")
    room = Room.from_yaml(tmp_path / "moved" / "copy.yaml")
    assert room.scene.sources[0].directivity.array.element.file == (
        tmp_path / "moved" / "speaker.csv"
    )


def test_a_scene_lists_the_files_its_arrays_name(tmp_path: Path) -> None:
    """An array's element can be an array in a YAML file, which names its element's file
    relative to itself. The scene lists every file once, and saves those in its own
    entries relative to its folder."""
    folder = tmp_path / "scene"
    (folder / "arrays").mkdir(parents=True)
    _write_csv(folder / "arrays" / "box.csv", FirstOrder.pattern("cardioid"), OCTAVE_BANDS_HZ)
    column = {"element": {"file": "box.csv"}, "count": 3, "spacing_m": 0.4}
    (folder / "arrays" / "column.yaml").write_text(yaml.safe_dump(column), encoding="utf-8")
    data = _shoebox(directivity={"file": "arrays/column.yaml"})
    second = {**data["sources"][0], "name": "second", "position": [2.0, 2.0, 1.5]}
    pair = {"element": {"file": "arrays/column.yaml"}, "count": 2, "spacing_m": 1.0}
    second["directivity"] = {"array": pair}
    data["sources"].append(second)
    (folder / "scene.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    scene = Scene.from_yaml(folder / "scene.yaml")
    model, *arrays = scene.files()
    assert model.name == "shoebox.obj"
    assert arrays == [folder / "arrays" / "column.yaml", folder / "arrays" / "box.csv"]
    scene.to_yaml(folder / "copy.yaml")
    saved = yaml.safe_load((folder / "copy.yaml").read_text(encoding="utf-8"))
    assert saved["sources"][1]["directivity"]["array"]["element"] == {"file": "arrays/column.yaml"}
    # An array file that names itself is listed once.
    (folder / "loop.yaml").write_text(
        "element: {file: loop.yaml}\ncount: 2\nspacing_m: 1\n", encoding="utf-8"
    )
    assert array_files(folder / "loop.yaml") == [folder / "loop.yaml"]


# -- simulation ----------------------------------------------------------------------


def _direct(directivity, orientation: dict) -> np.ndarray:
    room = Room.from_scene(
        Scene.model_validate(_shoebox(directivity=directivity, orientation=orientation))
    )
    paths = early_paths(room, max_order=0)
    assert len(paths) == 1
    return paths.energy[0]


@pytest.mark.parametrize("orientation", [{"yaw": 0}, {"yaw": 40, "pitch": -25, "roll": 30}])
def test_direct_sound_follows_the_directivity(orientation: dict) -> None:
    omni = _direct("omni", orientation)
    per_band = {"pattern": [float(a) for a in SPEAKER]}
    # The listener in the shoebox example is in front of the talker, along +x.
    room = Room.from_yaml(SHOEBOX)
    toward = np.subtract(room.scene.listeners[0].position, room.scene.sources[0].position)
    local = toward / np.linalg.norm(toward) @ np.array(Orientation(**orientation).rotation())
    expected = FirstOrder(SPEAKER).energy([local])[0]
    assert _direct(per_band, orientation) / omni == pytest.approx(expected, rel=1e-9)


def test_direct_sound_behind_a_cardioid_is_silent() -> None:
    omni = _direct("omni", {})
    # The listener sits 0.3 m lower than the talker, so not quite straight behind.
    assert np.all(_direct("cardioid", {"yaw": 180}) / omni < 1e-5)
    assert _direct("cardioid", {"yaw": 0}) / omni == pytest.approx([3.0] * 8, rel=0.01)


def _box(size: float):
    vertices = np.array(
        [[x, y, z] for x in (0.0, size) for y in (0.0, size) for z in (0.0, size)], dtype=float
    )
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    faces = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))], dtype=np.int64)
    return vertices, faces


def _late_direct(directivity, orientation: np.ndarray, receivers: np.ndarray) -> np.ndarray:
    """Energy (receivers, bands) that reaches detector spheres straight from the source in
    a box whose walls absorb everything."""
    from acoustic_spaces import _native

    vertices, faces = _box(20.0)
    a, table = directivity.native()
    energy, *_ = _native.trace_stochastic(
        bvh=_native.Bvh(vertices, faces),
        source=np.array([10.0, 10.0, 10.0]),
        directivity_a=a,
        orientation=orientation,
        directivity_table=table,
        absorption=np.ones((12, 8)),
        scattering=np.zeros((12, 8)),
        air=np.zeros(8),
        surface_size=np.zeros((0, 2)),
        band_hz=np.zeros(0),
        vector_scattering=True,
        size_from_order=1,
        transmission=np.zeros((0, 0)),
        face_room=np.zeros((0, 2), dtype=np.int64),
        source_room=-1,
        receivers=receivers,
        directions=np.zeros((0, 3)),
        rays=400_000,
        max_time=0.05,
        time_step=1e-3,
        dir_time_step=1e-2,
        speed_of_sound=343.0,
        receiver_radius=np.full(len(receivers), 0.5),
        diffuse_rain=False,
        specular_rain=False,
        covered_order=-1,
        covered_time=np.zeros(0),
        seed=3,
        threads=0,
        window_time_step=0.01,
        first_ray=0,
        split_target=np.zeros(0),
        split_ratio=4.0,
        max_split=8,
        max_family=256,
        room_level=np.zeros(0),
        face_group=np.zeros(0, dtype=np.int64),
        groups=0,
        moment_time_step=0.01,
    )
    return energy.sum(axis=2)


def test_late_rays_leave_the_source_with_its_directivity() -> None:
    turn = np.array(Orientation(yaw=90, pitch=20).rotation())
    local = np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    receivers = 10.0 + 3.0 * local @ turn.T
    pattern = FirstOrder(SPEAKER)
    omni = _late_direct(FirstOrder.pattern("omni"), turn, receivers)
    first_order = _late_direct(pattern, turn, receivers)
    measured = _sphere(3000)
    balloon = Balloon.from_measurements(measured, pattern.energy(measured))
    tabled = _late_direct(balloon, turn, receivers)
    expected = pattern.energy(local)
    # The same rays: only the weights differ, so the ratios are exact up to the averaging
    # over each detector sphere.
    ratio = first_order / omni
    np.testing.assert_allclose(ratio, expected, rtol=0.03, atol=0.02)
    np.testing.assert_allclose(tabled / omni, ratio, rtol=0.03, atol=0.03)
    # The compiled tracer skips rays a figure-8 sends nowhere, without losing energy.
    figure8 = _late_direct(FirstOrder.pattern("figure8"), turn, receivers)
    assert figure8[0] / omni[0] == pytest.approx([3.0] * 8, rel=0.05)


def test_particles_leave_the_source_with_its_directivity() -> None:
    room = Room.from_scene(Scene.model_validate(_shoebox(directivity="cardioid")))
    particles = trace_particles(room, count=2000, duration=0.001, frames=2)
    moved = particles.positions[1] - particles.positions[0]
    forward = moved[:, 0] / np.linalg.norm(moved, axis=1)
    expected = (0.5 + 0.5 * forward) ** 2 * 3.0
    np.testing.assert_allclose(particles.energy[0], expected, atol=1e-9)


# -- command line --------------------------------------------------------------------


def test_directivity_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("matplotlib")
    path = _write_csv(tmp_path / "speaker.csv", FirstOrder(SPEAKER))
    plot = tmp_path / "speaker.png"
    assert main(["directivity", str(path), "--plot", str(plot)]) == 0
    out = capsys.readouterr().out
    assert "file speaker.csv (63 Hz copied from the nearest octave)" in out
    assert "directivity index dB" in out and plot.stat().st_size > 10_000
    assert main(["directivity", "cardioid"]) == 0
    assert "4.8   4.8" in capsys.readouterr().out
    assert main(["directivity", str(tmp_path / "missing.csv")]) == 1
    assert "not found" in capsys.readouterr().err


def test_inspect_lists_source_directivities(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _shoebox(directivity={"pattern": ["omni"] * 4 + ["cardioid"] * 4})
    (tmp_path / "scene.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["inspect", str(tmp_path / "scene.yaml")]) == 0
    out = capsys.readouterr().out
    assert "sources (directivity index per octave band, dB)" in out
    assert "talker" in out and "0.0   0.0   0.0   0.0   4.8   4.8   4.8   4.8" in out
