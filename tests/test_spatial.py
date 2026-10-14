import math
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Scene, simulate
from acoustic_spaces.scene import Output
from acoustic_spaces.scene.model import Orientation
from acoustic_spaces.spatial import (
    HRTF,
    LAYOUTS,
    Ambisonics,
    Binaural,
    Loudspeakers,
    Mono,
    Speaker,
    fibonacci_sphere,
    from_angles,
    from_output,
    read_sofa,
    real_spherical_harmonics,
    spherical_head,
    to_angles,
    to_head,
)

SHOEBOX_OBJ = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "shoebox.obj"
FS = 48_000


def test_spherical_harmonics_refuse_directions_without_length() -> None:
    with pytest.raises(ValueError, match="length above zero"):
        real_spherical_harmonics(1, [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="length above zero"):
        Ambisonics(2).gains([[math.nan, 0.0, 1.0]])


def test_spherical_harmonics_are_sn3d() -> None:
    grid = fibonacci_sphere(8000)
    y = real_spherical_harmonics(4, grid)
    gram = y.T @ y * 4 * math.pi / len(grid)
    n = np.floor(np.sqrt(np.arange(25)))
    np.testing.assert_allclose(gram, np.diag(4 * math.pi / (2 * n + 1)), atol=1e-3)
    # First order is the direction itself (ACN: W, Y, Z, X), and order 2 and 3 match
    # the closed forms.
    v = from_angles(37.0, 21.0)
    az, el = math.radians(37.0), math.radians(21.0)
    y = real_spherical_harmonics(3, v[None])[0]
    np.testing.assert_allclose(y[:4], [1.0, v[1], v[2], v[0]], atol=1e-12)
    s3 = math.sqrt(3) / 2
    order2 = [
        s3 * math.cos(el) ** 2 * math.sin(2 * az),
        s3 * math.sin(2 * el) * math.sin(az),
        0.5 * (3 * math.sin(el) ** 2 - 1),
        s3 * math.sin(2 * el) * math.cos(az),
        s3 * math.cos(el) ** 2 * math.cos(2 * az),
    ]
    np.testing.assert_allclose(y[4:9], order2, atol=1e-12)
    assert y[12] == pytest.approx(0.5 * math.sin(el) * (5 * math.sin(el) ** 2 - 3))
    assert y[15] == pytest.approx(math.sqrt(5 / 8) * math.cos(el) ** 3 * math.cos(3 * az))
    with pytest.raises(ValueError):
        Ambisonics(8)


def test_orientation_turns_room_directions_into_head_directions() -> None:
    # Facing +y (yaw 90): +y is ahead, -x is to the left.
    rotation = np.array(Orientation(yaw=90).rotation())
    np.testing.assert_allclose(to_head([0.0, 1.0, 0.0], rotation), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(to_head([-1.0, 0.0, 0.0], rotation), [0, 1, 0], atol=1e-12)
    # Rolled 90 degrees to the right, the left ear points up.
    rotation = np.array(Orientation(roll=90).rotation())
    np.testing.assert_allclose(to_head([0.0, 0.0, 1.0], rotation), [0, 1, 0], atol=1e-12)
    rotation = np.array(Orientation(yaw=30, pitch=-20, roll=40).rotation())
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation[:, 0], Orientation(yaw=30, pitch=-20).forward())
    azimuth, elevation = to_angles(from_angles([120.0, -45.0], [10.0, -80.0]))
    np.testing.assert_allclose(azimuth, [120, -45])
    np.testing.assert_allclose(elevation, [10, -80])


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_vbap_pans_with_unit_power(name: str) -> None:
    layout = Loudspeakers.layout(name)
    mains = [i for i, s in enumerate(layout.speakers) if not s.lfe]
    np.testing.assert_allclose(
        layout.gains(layout.directions()[mains])[:, mains], np.eye(len(mains)), atol=1e-12
    )
    grid = np.vstack([fibonacci_sphere(600), [[0, 0, 1], [0, 0, -1]]])
    gains = layout.gains(grid)
    np.testing.assert_allclose((gains**2).sum(axis=1), 1.0)
    assert gains.min() >= 0.0
    np.testing.assert_array_equal(np.delete(gains, mains, axis=1), 0.0)  # LFE stays silent
    if name == "7.0.4":
        # Above the horizon the speakers surround every direction: the gains point there.
        up = grid[:, 2] >= 0
        aim = gains[up] @ layout.directions()
        aim /= np.linalg.norm(aim, axis=1, keepdims=True)
        assert np.min(np.sum(aim * grid[up], axis=1)) > math.cos(math.radians(0.5))
    assert layout.channel_mask() > 0


def test_vbap_between_and_outside_speakers() -> None:
    stereo = Loudspeakers.layout("stereo")
    gains = stereo.gains(from_angles([0.0, 180.0, 100.0], 0.0))
    np.testing.assert_allclose(gains[0], [math.sqrt(0.5)] * 2)
    np.testing.assert_allclose(gains[1:], [[1, 0], [1, 0]])  # behind: the nearer speaker
    surround = Loudspeakers.layout("5.0")
    np.testing.assert_allclose(surround.gains(from_angles(180.0, 0.0))[0, 3:], [math.sqrt(0.5)] * 2)
    # Straight down in a layout without low speakers: the horizon plays alike.
    down = Loudspeakers.layout("7.0.4").gains([[0.0, 0.0, -1.0]])[0]
    np.testing.assert_allclose(down[:7], 1 / math.sqrt(7))
    # 5.x.4 has 140 degrees between its surrounds: arrivals on the horizon there stay on
    # it, panned as in 5.0, instead of going up to the rear height speakers.
    horizon = from_angles(np.arange(0.0, 360.0, 2.5), 0.0)
    for name in ("5.0.4", "5.1.4"):
        layout = Loudspeakers.layout(name)
        flat = Loudspeakers.layout(name[:3])
        gains = layout.gains(horizon)
        np.testing.assert_allclose(gains[:, : flat.channels], flat.gains(horizon), atol=1e-12)
        np.testing.assert_array_equal(gains[:, flat.channels :], 0.0)
    custom = Loudspeakers((Speaker("a", 45), Speaker("b", -45), Speaker("c", 180)))
    assert custom.channel_mask() == 0 and custom.label == "speakers"
    with pytest.raises(ValueError, match="unknown layout"):
        Loudspeakers.layout("22.2")
    with pytest.raises(ValueError, match="at least two"):
        Loudspeakers((Speaker("a", 0),))
    with pytest.raises(ValueError, match="at least two"):
        Loudspeakers((Speaker("a", 0), Speaker("LFE", 0, lfe=True)))
    surround = Loudspeakers.layout("7.1.4")
    assert surround.channel_names[:4] == ["L", "R", "C", "LFE"]
    assert surround.channel_mask() == 0x2D63F
    assert Loudspeakers.layout("5.1").channel_mask() == 0x3F


def test_spherical_head_delays_and_shadows_the_far_ear() -> None:
    head = spherical_head(FS)
    left = head.nearest(from_angles(90.0, 0.0)[None])[0]
    onset = np.argmax(np.abs(head.hrir[left]), axis=1)
    itd = (onset[1] - onset[0]) / FS
    assert itd == pytest.approx(0.0006, abs=0.0001)
    spectrum = np.abs(np.fft.rfft(head.hrir[left], 1024, axis=1))
    k = int(8000 / FS * 1024)
    assert 20 * np.log10(spectrum[0, k] / spectrum[1, k]) > 10.0  # 8 kHz: near ear louder
    assert spectrum[0, 1] == pytest.approx(spectrum[1, 1], rel=0.01)  # low frequencies
    resampled = head.resampled(44_100)
    onset = np.argmax(np.abs(resampled.hrir[left]), axis=1)
    assert (onset[1] - onset[0]) / 44_100 == pytest.approx(itd, abs=0.00005)


@pytest.mark.parametrize("rate", [24_000, 44_100, 96_000])
def test_resampling_keeps_the_gain_of_the_responses(rate: float) -> None:
    # A flat response (an impulse) stays flat at unit gain at any rate.
    hrir = np.zeros((1, 2, 256))
    hrir[0, :, 10] = 1.0
    directions = np.array([[2, 0, 0]])
    hrtf = HRTF(directions, hrir, 48_000)
    np.testing.assert_array_equal(directions, [[2, 0, 0]])  # the caller's array stays
    spectrum = np.abs(np.fft.rfft(hrtf.resampled(rate).hrir, axis=-1))
    band = slice(1, int(0.4 * spectrum.shape[-1]))
    np.testing.assert_allclose(spectrum[..., band], 1.0, atol=1e-9)


def _write_sofa(path: Path, right_first: bool = False) -> np.ndarray:
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(3)
    ir = rng.standard_normal((4, 2, 32))
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = "SOFA"
        f.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        f.attrs["Title"] = "test set"
        f["Data.IR"] = ir
        f["Data.SamplingRate"] = np.array([44_100.0])
        f["Data.Delay"] = np.zeros((1, 2))
        positions = f.create_dataset(
            "SourcePosition", data=[[0.0, 0, 1.2], [90, 0, 1.2], [180, 0, 1.2], [0, 90, 1.2]]
        )
        positions.attrs["Type"] = "spherical"
        positions.attrs["Units"] = "degree, degree, metre"
        ears = f.create_dataset(
            "ReceiverPosition",
            data=[[0, -0.09, 0], [0, 0.09, 0]] if right_first else [[0, 0.09, 0], [0, -0.09, 0]],
        )
        ears.attrs["Type"] = "cartesian"
        f.create_dataset("ListenerView", data=[[1.0, 0.0, 0.0]]).attrs["Type"] = "cartesian"
    return ir


@pytest.mark.parametrize("right_first", [False, True])
def test_read_sofa(tmp_path: Path, right_first: bool) -> None:
    ir = _write_sofa(tmp_path / "set.sofa", right_first)
    hrtf = read_sofa(tmp_path / "set.sofa")
    assert hrtf.sample_rate == 44_100 and hrtf.name == "test set"
    np.testing.assert_allclose(
        hrtf.directions, [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12
    )
    np.testing.assert_allclose(hrtf.hrir, ir[:, ::-1] if right_first else ir)
    assert hrtf.nearest(from_angles([80.0, 0.0], [0.0, 70.0])).tolist() == [1, 3]


def test_read_sofa_with_a_turning_head_and_spherical_ears(tmp_path: Path) -> None:
    """The head turns for each measurement ([M C] ListenerView) while the source stays in
    front of the room; the ears are given in spherical coordinates, right ear first."""
    h5py = pytest.importorskip("h5py")
    ir = np.random.default_rng(4).standard_normal((3, 2, 16))
    yaw = np.radians([0.0, 90.0, -90.0])
    with h5py.File(tmp_path / "turn.sofa", "w") as f:
        f["Data.IR"] = ir
        f["Data.SamplingRate"] = np.array([48_000.0])
        f["SourcePosition"] = [[1.5, 0.0, 0.0]]
        f["ListenerView"] = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(3)])
        f["ListenerUp"] = [[0.0, 0.0, 1.0]]
        ears = f.create_dataset("ReceiverPosition", data=[[-90.0, 0, 0.09], [90.0, 0, 0.09]])
        ears.attrs["Type"] = "spherical"
        ears.attrs["Units"] = "degree, degree, metre"
    hrtf = read_sofa(tmp_path / "turn.sofa")
    # Turned left, the source is on the right of the head; turned right, on its left.
    np.testing.assert_allclose(hrtf.directions, [[1, 0, 0], [0, -1, 0], [0, 1, 0]], atol=1e-12)
    np.testing.assert_allclose(hrtf.hrir, ir[:, ::-1])


def test_read_sofa_without_h5py_says_how_to_install(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "h5py", None)
    with pytest.raises(ImportError, match=r"acoustic-spaces\[sofa\]"):
        read_sofa(tmp_path / "missing.sofa")


def _scene(late: dict, yaw: float = 0.0) -> Scene:
    return Scene.model_validate(
        {
            "geometry": {"file": str(SHOEBOX_OBJ)},
            "materials": {"m": {"absorption": [0.2] * 8, "scattering": 0.5}},
            "default_material": "m",
            "sources": [{"name": "s", "position": [2.1, 2.3, 1.7]}],
            "listeners": [{"name": "l", "position": [5.9, 3.7, 1.2], "orientation": {"yaw": yaw}}],
            "simulation": {"seed": 5, "ir_length_s": 0.6, "late": late},
        }
    )


@pytest.fixture(scope="module")
def response():
    # The listener faces 30 degrees to the right of the source.
    to_source = np.array([2.1 - 5.9, 2.3 - 3.7])
    yaw = math.degrees(math.atan2(to_source[1], to_source[0])) - 30.0
    return simulate(_scene({"rays": 4000}, yaw))["s", "l"]


def _direct(response) -> int:
    return int(round(response.direct_delay_s * FS))


def test_late_directions_hold_the_late_energy(response) -> None:
    assert response.late_directional.shape[:2] == (64, 8)
    per_slot = response.late.reshape(8, -1, 10).sum(axis=2)
    np.testing.assert_allclose(response.late_directional.sum(axis=0), per_slot, rtol=1e-9)


def test_ambisonics_keeps_mono_in_w_and_points_at_the_source(response) -> None:
    mono = response.impulse_response(FS)
    ir = response.spatial_impulse_response(Ambisonics(1), FS)
    assert ir.shape == (len(mono), 4)
    np.testing.assert_array_equal(ir[:, 0], mono)
    k = _direct(response)
    w, y, z, x = ir[k - 2 : k + 3].T
    peak = np.argmax(np.abs(w))
    head = to_head(
        response.early.direction[np.argmin(response.early.delay_s)], response.orientation
    )
    assert to_angles(head)[0] == pytest.approx(30.0, abs=1.0)  # 30 degrees to the left
    np.testing.assert_allclose([x[peak], y[peak], z[peak]] / w[peak], head, atol=0.01)


def test_loudspeakers_keep_the_energy(response) -> None:
    mono = response.impulse_response(FS)
    ir = response.spatial_impulse_response(Loudspeakers.layout("5.0"), FS)
    k = _direct(response)
    # The direct sound, 30 degrees to the left, plays from L, with the mono energy.
    direct = slice(k - FS // 1000, k + FS // 1000)
    assert np.argmax(np.abs(ir[direct]).max(axis=0)) == 0
    assert np.sum(ir[direct] ** 2) == pytest.approx(np.sum(mono[direct] ** 2), rel=0.01)
    # So does the late part. (Early reflections that overlap add up in the mono response
    # but not across speakers.)
    late = slice(k + int(0.05 * FS), None)
    assert np.sum(ir[late] ** 2) == pytest.approx(np.sum(mono[late] ** 2), rel=0.05)


def test_binaural_with_flat_hrtfs_is_the_mono_response(response) -> None:
    grid = fibonacci_sphere(100)
    flat = np.zeros((100, 2, 64))
    flat[:, :, 0] = 1.0
    ir = response.spatial_impulse_response(Binaural(HRTF(grid, flat, FS)), FS)
    mono = response.impulse_response(FS)
    np.testing.assert_allclose(ir[:, 0], ir[:, 1], atol=1e-12)
    np.testing.assert_allclose(ir[:, 0], mono, atol=1e-3 * np.abs(mono).max())


def test_binaural_hears_the_source_on_the_left(response) -> None:
    ir = response.spatial_impulse_response(Binaural(), FS)
    k = _direct(response)
    window = slice(k, k + int(0.002 * FS))
    onset = np.argmax(np.abs(ir[window]), axis=0)
    assert onset[0] < onset[1]  # the left ear first
    level = np.sum(ir[window] ** 2, axis=0)
    assert level[0] > level[1]
    # Late energy at both ears is about the same.
    tail = ir[k + int(0.1 * FS) :]
    assert np.sum(tail[:, 0] ** 2) == pytest.approx(np.sum(tail[:, 1] ** 2), rel=0.3)


def test_statistical_tail_arrives_from_all_around() -> None:
    response = simulate(_scene({"method": "statistical"}))["s", "l"]
    assert response.late_directional.size == 0
    ir = response.spatial_impulse_response(Ambisonics(2), FS)
    tail = ir[_direct(response) + int(0.1 * FS) :]
    energy = np.sum(tail**2, axis=0)
    order = np.floor(np.sqrt(np.arange(9))).astype(int)
    # In a diffuse field each SN3D channel of order n carries 1 / (2n + 1) of W.
    np.testing.assert_allclose(energy[1:] * (2 * order[1:] + 1) / energy[0], 1.0, atol=0.25)


def test_outputs_in_scene_files() -> None:
    assert Output.model_validate("binaural").format == "binaural"
    assert Output.model_validate({"format": "loudspeakers", "layout": 5.0}).layout == "5.0"
    custom = Output.model_validate(
        {
            "format": "loudspeakers",
            "speakers": [{"name": "a", "azimuth": 30}, {"name": "b", "azimuth": -30}],
        }
    )
    assert from_output(custom).channel_names == ["a", "b"]
    assert from_output(Output(format="ambisonics", order=3)).channels == 16
    assert isinstance(from_output(Output(format="mono")), Mono)
    for bad in (
        {"format": "loudspeakers"},
        {"format": "loudspeakers", "layout": "9.1"},
        {"format": "ambisonics", "layout": "5.0"},
        {"format": "mono", "hrtf": "x.sofa"},
    ):
        with pytest.raises(ValueError):
            Output.model_validate(bad)
    assert [o.format for o in _scene({}).outputs] == ["mono"]
