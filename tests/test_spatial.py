import contextlib
import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Scene, simulate
from acoustic_spaces.directivity import PATTERNS
from acoustic_spaces.scene import Listener, Output
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
    for_listener,
    from_angles,
    from_output,
    read_sofa,
    real_spherical_harmonics,
    spherical_head,
    to_angles,
    to_head,
)
from acoustic_spaces.spatial.microphone import blumlein, first_order, ms, xy

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


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_vbap_pans_mirrored_directions_alike(name: str) -> None:
    # The height ring of the .4 layouts and the rear corners of 7.x.4 lie on one plane;
    # split along one diagonal, it panned to one side.
    layout = Loudspeakers.layout(name)
    speakers = [(s.azimuth, s.elevation, s.lfe) for s in layout.speakers]
    mirror = [speakers.index((-azimuth, elevation, lfe)) for azimuth, elevation, lfe in speakers]
    grid = np.vstack([fibonacci_sphere(600), [[0, 0, 1]]])
    np.testing.assert_allclose(
        layout.gains(grid * [1, -1, 1]), layout.gains(grid)[:, mirror], atol=1e-12
    )
    if name.endswith(".4"):  # straight up: the four height speakers alike
        np.testing.assert_allclose(layout.gains([[0.0, 0.0, 1.0]])[0, -4:], 0.5)


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


def _itd(hrir: np.ndarray) -> np.ndarray:
    """Interaural time difference (s, left ear first) of (dirs, 2, taps) responses, from
    the phase of the ears' cross-spectrum between 200 and 600 Hz."""
    f = np.fft.rfftfreq(4096, 1.0 / FS)
    band = (f > 200) & (f < 600)
    left, right = np.fft.rfft(hrir, 4096, axis=-1)[:, :, band].transpose(1, 0, 2)
    phase = np.unwrap(np.angle(right * np.conj(left)), axis=1)
    return -(phase / (2 * np.pi * f[band])).mean(axis=1)


def test_interpolated_hrtfs_give_the_measured_responses_and_a_smooth_itd() -> None:
    coarse = spherical_head(FS, directions=128)
    np.testing.assert_allclose(coarse.interpolate(coarse.directions), coarse.hrir, atol=1e-12)
    # Round the horizontal plane, a degree at a time: the nearest measured response
    # jumps in steps of about 20 degrees, the interpolated one follows.
    azimuth = np.arange(0.0, 91.0)
    heads = from_angles(azimuth, np.zeros_like(azimuth))
    dense = spherical_head(FS, directions=20_000)
    truth = _itd(dense.hrir[dense.nearest(heads)])
    smooth = _itd(coarse.interpolate(heads))
    steps = _itd(coarse.hrir[coarse.nearest(heads)])
    assert np.abs(np.diff(smooth)).max() < 50e-6 < 150e-6 < np.abs(np.diff(steps)).max()
    assert np.abs(smooth - truth).mean() < 0.3 * np.abs(steps - truth).mean()
    # Fewer than four directions: the nearest one.
    few = HRTF(coarse.directions[:3], coarse.hrir[:3], FS)
    np.testing.assert_array_equal(few.interpolate(coarse.directions[1] + 0.01), few.hrir[[1]])
    # The onsets alone are the ones that come with the responses.
    for hrtf in (coarse, few):
        _, onsets = hrtf.interpolate(heads, onsets=True)
        np.testing.assert_array_equal(hrtf.onsets_at(heads), onsets)


def test_interpolated_hrtfs_keep_the_spectrum() -> None:
    """Responses delayed by fractions of a sample ring before their peak, more the
    nearer the fraction is to a half: aligned at their onsets, mixed ones still add in
    phase, so the mix keeps their spectrum (the spherical head lost up to 13 dB at
    8 kHz when they were aligned at the first sample near the peak)."""
    grid = fibonacci_sphere(256)
    taps = np.arange(64)
    delays = 20.0 + 6.0 * grid[:, 0] + 2.0 * grid[:, 1]
    # Flat pulses (band-limited impulses) at fractional delays, the same at both ears.
    pulses = np.sinc(taps[None, :] - delays[:, None])
    hrtf = HRTF(grid, np.stack([pulses, pulses], axis=1), FS)
    heads = fibonacci_sphere(500)
    spectrum = np.abs(np.fft.rfft(hrtf.interpolate(heads), 1024, axis=-1))
    f = np.fft.rfftfreq(1024, 1.0 / FS)
    band = f < 16_000
    assert 20 * np.log10(spectrum[..., band].min()) > -0.5
    head = spherical_head(FS)
    level = lambda h: np.abs(np.fft.rfft(h, 4096, axis=-1))[..., 683]  # noqa: E731  8 kHz
    between = level(head.interpolate(heads)) / level(head.hrir[head.nearest(heads)])
    assert np.abs(20 * np.log10(between)).max() < 1.0


def test_interpolating_a_ring_of_directions() -> None:
    """Measured round the horizon only: between two directions the pulse arrives
    between their times, and above the horizon a direction takes the ring's response
    under it (straight up, the nearest)."""
    azimuth = np.arange(0.0, 360.0, 10.0)
    arrival = 40.0 + 20.0 * np.sin(np.radians(azimuth))
    pulse = lambda centre: np.exp(-0.5 * ((np.arange(128) - centre) / 2.0) ** 2)  # noqa: E731
    hrir = np.stack([np.stack([pulse(t), pulse(t + 3.0)]) for t in arrival])
    ring = HRTF(from_angles(azimuth, np.zeros_like(azimuth)), hrir, FS)
    heads = from_angles([5.0, 30.0, 0.0], [0.0, 60.0, 90.0])
    between, above, top = ring.interpolate(heads)
    halfway = (arrival[0] + arrival[1]) / 2
    np.testing.assert_allclose(between, [pulse(halfway), pulse(halfway + 3.0)], atol=0.01)
    np.testing.assert_allclose(above, hrir[3], atol=1e-12)
    np.testing.assert_allclose(top, hrir[ring.nearest(heads[2])[0]], atol=1e-12)


def test_interpolating_half_a_sphere() -> None:
    """Measured above the horizon only: below it, nothing is interpolated, and each
    direction takes the nearest response."""
    grid = fibonacci_sphere(512)
    rim = from_angles(np.arange(0.0, 360.0, 15.0), np.zeros(24))
    directions = np.vstack([grid[grid[:, 2] > 0.05], rim])
    hrir = np.random.default_rng(6).standard_normal((len(directions), 2, 64))
    half = HRTF(directions, hrir, FS)
    heads = fibonacci_sphere(300)
    below = heads[heads[:, 2] < -0.01]
    np.testing.assert_allclose(half.interpolate(below), hrir[half.nearest(below)], atol=1e-12)
    above = heads[heads[:, 2] > 0.1]
    assert np.abs(half.interpolate(above) - hrir[half.nearest(above)]).max() > 0.1


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


def test_read_sofa_keeps_fractional_delays(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    _write_sofa(tmp_path / "set.sofa")
    # Smooth pulses (as good as band-limited) at sample 14, delayed by 0.5 and 2.25.
    pulse = lambda centre: np.exp(-0.5 * ((np.arange(35) - centre) / 2.0) ** 2)  # noqa: E731
    with h5py.File(tmp_path / "set.sofa", "r+") as f:
        f["Data.IR"][...] = np.broadcast_to(pulse(14.0)[:32], (4, 2, 32))
        f["Data.Delay"][...] = [[0.5, 2.25]]
    hrtf = read_sofa(tmp_path / "set.sofa")
    assert hrtf.taps == 35
    np.testing.assert_allclose(hrtf.hrir[:, 0], np.broadcast_to(pulse(14.5), (4, 35)), atol=1e-6)
    np.testing.assert_allclose(hrtf.hrir[:, 1], np.broadcast_to(pulse(16.25), (4, 35)), atol=1e-6)
    np.testing.assert_allclose(hrtf.onsets()[:, 1] - hrtf.onsets()[:, 0], 1.75, atol=0.05)


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
    with pytest.raises(ImportError, match=r'pip install "acoustic-spaces\[sofa\]"'):
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


def _first_order(ambisonics: np.ndarray, w: float, d) -> np.ndarray:
    """What a capsule ``w + d . v`` picks up, from first-order AmbiX (W, Y, Z, X)."""
    x, y, z = d
    return ambisonics @ np.array([w, y, z, x])


def test_a_listeners_microphone_is_its_first_order_field(response) -> None:
    ambisonics = response.spatial_impulse_response(Ambisonics(1), FS)
    scale = np.abs(ambisonics[:, 0]).max()
    for pattern in ("subcardioid", "cardioid", "supercardioid", "hypercardioid", "figure8"):
        a = PATTERNS[pattern]
        ir = response.spatial_impulse_response(first_order(pattern), FS)
        assert ir.shape == (len(ambisonics), 1)
        expected = _first_order(ambisonics, a, (1.0 - a, 0.0, 0.0))
        np.testing.assert_allclose(ir[:, 0], expected, atol=1e-9 * scale)
    # A pair's channels too, the left capsule aimed to the left of the front.
    half = math.radians(110.0) / 2.0
    ir = response.spatial_impulse_response(xy(110.0, "supercardioid"), FS)
    a = PATTERNS["supercardioid"]
    for channel, sign in zip(ir.T, (1.0, -1.0), strict=True):
        d = (1.0 - a) * np.array([math.cos(half), sign * math.sin(half), 0.0])
        np.testing.assert_allclose(channel, _first_order(ambisonics, a, d), atol=1e-9 * scale)
    ir = response.spatial_impulse_response(ms("cardioid", 6.0), FS)
    g = 10.0 ** (6.0 / 20.0)
    for channel, sign in zip(ir.T, (1.0, -1.0), strict=True):
        expected = _first_order(ambisonics, 0.5, (0.5, sign * g, 0.0))
        np.testing.assert_allclose(channel, expected, atol=1e-9 * scale)


def test_a_microphone_whose_pattern_changes_with_frequency(response) -> None:
    """A pattern per band picks up as that band's pattern does wherever its neighbours
    agree: omni in the bass and a figure-eight in the treble here, crossing over in
    between; the same pattern in every band is the named one, rendered once."""
    from acoustic_spaces.synthesis import split_bands

    per_band = ["omni", "omni", "omni", 0.5, 0.5, "figure8", "figure8", "figure8"]
    microphone = first_order(per_band)
    assert microphone.a == (1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0)
    assert microphone.label == "per band: omni, omni, omni, 0.5, 0.5, figure8, figure8, figure8"
    assert microphone.weights == (0.5,)  # its balloon is drawn at 1 kHz

    def bands(fmt) -> np.ndarray:
        ir = response.spatial_impulse_response(fmt, FS)[:, 0]
        return 10.0 * np.log10(np.sum(split_bands(ir, FS) ** 2, axis=1))

    heard, omni, eight = bands(microphone), bands(Mono()), bands(first_order("figure8"))
    np.testing.assert_allclose(heard[:2], omni[:2], atol=0.01)  # 63 and 125 Hz
    np.testing.assert_allclose(heard[-2:], eight[-2:], atol=0.01)  # 4 and 8 kHz
    assert all(abs(heard[k] - omni[k]) > 3.0 for k in (5, 6, 7))
    # Rendered in two parts, which add up exactly where the bands agree.
    steady, scaled = first_order("cardioid"), microphone.parts()[1]
    assert scaled.weights == (1.0,) and scaled.axes == ((-1.0, 0.0, 0.0),)
    forced = dataclasses.replace(
        steady, a=(0.5,) * 8, aims=((1.0, 0.0, 0.0),), sides=((0.0, 0.0, 0.0),)
    )
    whole = response.spatial_impulse_response(steady, FS)
    np.testing.assert_allclose(
        response.spatial_impulse_response(forced, FS), whole, atol=1e-12 * abs(whole).max()
    )
    # The same in every band, as names or values: the named pattern itself.
    assert first_order(["cardioid"] * 8) == first_order(["cardioid"] * 24) == steady
    assert first_order(0.5).a is None and first_order(0.5).axes == steady.axes
    # Third octaves given, octaves simulated: each octave the mean of its three.
    thirds = first_order([1.0] * 12 + [0.0] * 12)
    assert thirds.a_in(8).tolist() == pytest.approx([1, 1, 1, 1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="parts"):
        microphone.gains([[1.0, 0.0, 0.0]])
    # In each band, from straight ahead and from behind.
    gains = microphone.band_gains(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]), 8)
    np.testing.assert_allclose(gains[0, 0], 1.0)
    np.testing.assert_allclose(gains[1, 0], [1, 1, 1, 0, 0, -1, -1, -1])


def test_a_microphone_hears_the_direct_sound_through_its_pattern(response) -> None:
    mono = response.impulse_response(FS)
    k = _direct(response)
    direct = slice(k - 2, k + 3)
    # The source is 30 degrees to the listener's left.
    for pattern in ("cardioid", "figure8"):
        a = PATTERNS[pattern]
        ir = response.spatial_impulse_response(first_order(pattern), FS)[:, 0]
        gain = a + (1.0 - a) * math.cos(math.radians(30.0))
        np.testing.assert_allclose(ir[direct], gain * mono[direct], atol=0.02 * abs(mono).max())
    left, right = response.spatial_impulse_response(blumlein(), FS)[direct].T
    peak = np.argmax(np.abs(mono[direct]))
    # Figure-eights aimed 45 degrees to each side: cos 15 and cos 75 degrees.
    assert left[peak] / mono[direct][peak] == pytest.approx(math.cos(math.radians(15.0)), abs=0.02)
    assert right[peak] / mono[direct][peak] == pytest.approx(math.cos(math.radians(75.0)), abs=0.02)


def test_a_microphone_in_a_diffuse_field_picks_up_its_random_energy_efficiency() -> None:
    response = simulate(_scene({"method": "statistical"}))["s", "l"]
    tail = slice(_direct(response) + int(0.1 * FS), None)
    omni = np.sum(response.impulse_response(FS)[tail] ** 2)
    for pattern in ("cardioid", "supercardioid", "figure8"):
        a = PATTERNS[pattern]
        ir = response.spatial_impulse_response(first_order(pattern), FS)
        # Its energy over the sphere: a^2 + (1 - a)^2 / 3, 1/3 for a cardioid.
        efficiency = a**2 + (1.0 - a) ** 2 / 3.0
        assert np.sum(ir[tail] ** 2) / omni == pytest.approx(efficiency, rel=0.25)


def test_a_listeners_pattern_picks_its_mono_output() -> None:
    scene = _scene({})
    plain = scene.listeners[0]
    assert for_listener(Mono(), plain) == Mono()
    omni = plain.model_copy(update={"pattern": "omni"})
    assert for_listener(Mono(), omni) == Mono()
    listener = Listener.model_validate({"name": "m", "position": [0, 0, 0], "pattern": "cardioid"})
    assert for_listener(Mono(), listener) == first_order("cardioid")
    # Only the mono output: the others hear as they do.
    assert for_listener(Binaural(), listener) == Binaural()
    pairs = {
        "xy": (
            {"pair": "xy", "angle_deg": 120, "capsule": "hypercardioid"},
            xy(120, "hypercardioid"),
        ),
        "blumlein": ({"pair": "blumlein"}, blumlein()),
        "ms": ({"pair": "ms", "mid": "figure8", "side_db": -3}, ms("figure8", -3)),
    }
    for spec, expected in pairs.values():
        listener = Listener.model_validate({"name": "m", "position": [0, 0, 0], "pattern": spec})
        heard = for_listener(Mono(), listener)
        assert heard == expected
        assert heard.channel_names == ["L", "R"]
    # A pattern per band, for one capsule or a pair's; all omni is none.
    wide = ["omni", "omni", "subcardioid", "cardioid", "cardioid", 0.4, 0.37, 0.37]
    for spec, expected in (
        (wide, first_order(wide)),
        (["omni"] * 8, Mono()),
        ({"pair": "xy", "capsule": wide}, xy(90.0, wide)),
        ({"pair": "ms", "mid": wide, "side_db": 2}, ms(wide, 2)),
    ):
        listener = Listener.model_validate({"name": "m", "position": [0, 0, 0], "pattern": spec})
        assert for_listener(Mono(), listener) == expected
    for bad in (
        "lambert",
        {"pair": "ab"},
        {"pair": "blumlein", "angle_deg": 90},
        {"pair": "xy", "angle_deg": 0},
        {"pair": "xy", "mid": "cardioid"},
        ["cardioid"] * 5,
        ["cardioid"] * 7 + [1.5],
        {"pair": "ms", "mid": ["lambert"] * 8},
    ):
        with pytest.raises(ValueError):
            Listener.model_validate({"name": "m", "position": [0, 0, 0], "pattern": bad})


def test_a_microphones_capsules_are_its_channels() -> None:
    directions = fibonacci_sphere(200)
    for microphone in (first_order("supercardioid"), xy(70.0, "cardioid"), blumlein(), ms()):
        gains = microphone.gains(directions)
        for k, (pattern, turn) in enumerate(microphone.capsules()):
            # The pattern aimed by the turn, scaled to the channel's gain straight ahead.
            on_axis = microphone.gains(turn[:, 0])[0, k]
            local = directions @ turn
            a = pattern.a[0]
            np.testing.assert_allclose(
                on_axis * (a + (1.0 - a) * local[:, 0]), gains[:, k], atol=1e-12
            )


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


@contextlib.contextmanager
def _serving(folder: Path):
    """A local server that answers range requests, as the repositories the sets come
    from do: the address it listens on."""
    import functools
    import http.server
    import socketserver
    import threading

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:  # quiet
            pass

        def do_GET(self) -> None:  # noqa: N802  Qt-free, this is http.server's name
            data = Path(self.translate_path(self.path)).read_bytes()
            asked = self.headers.get("Range")
            if asked:
                first, _, last = asked.removeprefix("bytes=").partition("-")
                start, end = int(first), int(last) if last else len(data) - 1
                part = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            else:
                part = data
                self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)

    handler = functools.partial(Handler, directory=str(folder))
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _test_set(tmp_path: Path, name: str = "test", members: tuple[str, ...] = ()):
    """A set that comes from a file beside the tests, so nothing is downloaded."""
    from acoustic_spaces.spatial import hrtf_sets

    source = tmp_path / "source" / ("ears.zip" if members else "ears.sofa")
    source.parent.mkdir(parents=True, exist_ok=True)
    if members:
        import zipfile

        with zipfile.ZipFile(source, "w") as archive:
            for member in members:
                archive.writestr(member, b"SOFA " + member.encode())
    else:
        source.write_bytes(b"SOFA" + bytes(32))
    return hrtf_sets.HrtfSet(
        name=name,
        title="Test ears",
        ears="a test",
        licence="CC0 1.0",
        credit="the tests",
        page="https://example.invalid/ears",
        url=source.as_uri(),
        size_mb=0.0001,
        note="nothing much in it",
        members=members,
    )


def test_hrtf_sets_say_what_they_are() -> None:
    """Every set on offer names whose ears it is, its licence, where it comes from and
    what the download costs."""
    from acoustic_spaces.spatial import hrtf_sets

    names = [item.name for item in hrtf_sets.sets()]
    assert names == sorted(set(names), key=names.index) and len(names) >= 3
    free = ("CC BY", "CC0", "Apache", "ODbL")
    for item in hrtf_sets.sets():
        assert item.licence.startswith(free), item.name
        assert item.page.startswith("https://") and item.url.startswith("https://")
        assert item.size_mb > 0 and item.ears and item.note and item.credit
        assert item.file() == f"{item.name}.sofa"
        # A set of several heads names them, and the first is the default.
        if item.members:
            assert item.member == item.members[0]
            assert all(member.endswith(".sofa") for member in item.members)
    assert hrtf_sets.named("ku100").title.startswith("Neumann KU100")
    with pytest.raises(KeyError, match="no HRTF set 'nope'"):
        hrtf_sets.named("nope")


def test_hrtf_sets_download_once(tmp_path: Path, monkeypatch) -> None:
    """A set is fetched into the folder, left alone when it is there, and --again
    fetches it over; a set of several heads takes the one asked for."""
    from acoustic_spaces.spatial import hrtf_sets

    folder = tmp_path / "sets"
    monkeypatch.setattr(hrtf_sets, "SETS", (_test_set(tmp_path),))
    assert hrtf_sets.installed(folder) == {}
    path = hrtf_sets.get("test", folder=folder)
    assert path == folder / "test.sofa" and path.read_bytes().startswith(b"SOFA")
    assert hrtf_sets.installed(folder) == {"test": path}

    path.write_bytes(b"changed")
    assert hrtf_sets.get("test", folder=folder).read_bytes() == b"changed"  # left alone
    assert hrtf_sets.get("test", folder=folder, again=True).read_bytes().startswith(b"SOFA")

    seen: list[tuple[int, int | None]] = []
    hrtf_sets.get("test", folder=folder, progress=lambda done, total: seen.append((done, total)))
    assert seen == []  # there already: nothing is fetched
    with pytest.raises(KeyError, match="a single SOFA file"):
        hrtf_sets.get("test", member="subj_A.sofa", folder=folder)

    # One head of a set that holds several: the archive is read where it is, by range
    # requests, and only that head comes down.
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    heads = _test_set(tmp_path, "many", ("subj_A.sofa", "subj_B.sofa"))
    with _serving(tmp_path / "source") as address:
        heads = dataclasses.replace(heads, url=f"{address}/ears.zip")
        monkeypatch.setattr(hrtf_sets, "SETS", (heads,))
        first = hrtf_sets.get(heads, folder=folder)
        other = hrtf_sets.get(heads, member="subj_B.sofa", folder=folder)
        assert first.name == "many.sofa" and other.name == "many-subj_B.sofa"
        assert other.read_bytes() == b"SOFA subj_B.sofa"
        with pytest.raises(KeyError, match="no 'subj_Z.sofa'"):
            hrtf_sets.get(heads, member="subj_Z.sofa", folder=folder)

    # Any other SOFA file, by its address.
    mine = tmp_path / "mine.sofa"
    mine.write_bytes(b"SOFA mine")
    assert hrtf_sets.other(mine.as_uri(), "me", folder).read_bytes() == b"SOFA mine"
    assert (folder / "me.sofa").exists()


def test_hrtf_folder_follows_the_system(monkeypatch, tmp_path: Path) -> None:
    """The sets are kept in the user's data folder, or where the environment says."""
    from acoustic_spaces.spatial import hrtf_sets

    monkeypatch.delenv("ACOUSTIC_SPACES_HRTF", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    if sys.platform not in ("win32", "darwin"):
        assert hrtf_sets.data_folder() == tmp_path / "data" / "acoustic-spaces" / "hrtf"
    assert hrtf_sets.data_folder().name == hrtf_sets.FOLDER
    monkeypatch.setenv("ACOUSTIC_SPACES_HRTF", str(tmp_path / "mine"))
    assert hrtf_sets.data_folder() == tmp_path / "mine"
