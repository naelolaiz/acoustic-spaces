"""Walks with Doppler (``walk.doppler``, :mod:`acoustic_spaces.doppler`): the delays,
the paths followed from place to place, the sound against the exact field of a passing
source and listener, and the pass-by levels (:func:`acoustic_spaces.walkthrough.pass_by`)."""

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces import doppler as dp
from acoustic_spaces.cli import main
from acoustic_spaces.directivity import _grid
from acoustic_spaces.emission import Mixture
from acoustic_spaces.render import convolve, gusts_for
from acoustic_spaces.scene.model import Walk
from acoustic_spaces.spatial import Binaural, Mono
from acoustic_spaces.spatial.directions import fibonacci_sphere
from acoustic_spaces.spatial.hrtf import HRTF
from acoustic_spaces.walkthrough import pass_by, render_walk, simulate_walk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import doppler as reference  # noqa: E402

RATE = 48_000


@pytest.fixture
def short(monkeypatch: pytest.MonkeyPatch):
    """The validation's pass-by, 30 m long."""
    monkeypatch.setattr(reference, "X0", -15.0)
    monkeypatch.setattr(reference, "X1", 15.0)
    monkeypatch.setattr(reference, "MARGIN_S", (0.1, 0.05))
    return reference


def _standing(ground: str, doppler: bool = True) -> Scene:
    """A source that walks a hair's breadth: Doppler should change nothing."""
    scene = reference.scene("source", 30.0, ground, doppler)
    source = scene.sources[0]
    walk = source.walk.model_copy(update={"to": [(-20.0 + 1e-7, 0.0, 1.0)], "speed_m_s": 1.0})
    source = source.model_copy(update={"position": (-20.0, 0.0, 1.0), "walk": walk})
    return scene.model_copy(update={"sources": [source]})


def _noise(seconds: float, seed: int = 2) -> np.ndarray:
    n = int(seconds * RATE)
    f = np.fft.rfftfreq(n, 1.0 / RATE)
    return np.fft.irfft(np.fft.rfft(np.random.default_rng(seed).standard_normal(n)) * (f < 20e3), n)


def _error_db(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return 10.0 * math.log10(np.sum((a[:n] - b[:n]) ** 2) / np.sum(b[:n] ** 2))


# -- the scene -----------------------------------------------------------------------


def test_doppler_is_off_unless_asked_and_round_trips(tmp_path: Path) -> None:
    assert Walk.model_validate({"to": [[1.0, 0.0, 1.0]]}).doppler is False
    scene = reference.scene("source", 30.0, "hard")
    assert scene.sources[0].walk.doppler is True
    path = tmp_path / "scene.yaml"
    scene.to_yaml(path)
    assert "doppler: true" in path.read_text()
    assert Scene.from_yaml(path).sources[0].walk.doppler is True


def test_doppler_needs_a_walk_slower_than_sound_and_no_wave_solver() -> None:
    scene = reference.scene("source", 400.0, "hard")
    with pytest.raises(ValueError, match="as fast as sound or faster"):
        simulate_walk(Room.from_scene(scene), step_m=10.0, late_step_m=None)
    scene = reference.scene("source", 30.0, "hard")
    data = scene.model_dump(mode="json", exclude_none=True)
    data["simulation"]["wave"] = {"enabled": True}
    with pytest.raises(ValueError, match="wave solver"):
        simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=10.0, late_step_m=None)


# -- the delays ----------------------------------------------------------------------


def test_hermite_goes_through_its_points_and_holds_at_the_ends() -> None:
    x = np.array([0.0, 1.0, 2.5, 3.0, 5.0])
    y = 0.3 * x**2 - x + 2.0
    at = np.linspace(-1.0, 6.0, 141)
    curve = dp.hermite(x, y, at)
    assert np.allclose(dp.hermite(x, y, x), y)
    # The slopes are a parabola's own inside: between inner points it is exact.
    inner = (at >= 1.0) & (at <= 3.0)
    assert np.allclose(curve[inner], 0.3 * at[inner] ** 2 - at[inner] + 2.0)
    assert np.all(curve[at < 0.0] == y[0]) and np.all(curve[at > 5.0] == y[-1])
    # Not held, it goes on along the end chords.
    carried = dp.hermite(x, y, np.array([-1.0, 6.0]), hold=False)
    assert np.allclose(carried, [y[0] - (y[1] - y[0]), y[-1] + (y[-1] - y[-2]) / 2.0])
    line = dp.hermite(x, 2.0 * x + 1.0, at, hold=False)
    assert np.allclose(line, 2.0 * at + 1.0)
    assert np.all(dp.hermite([1.0], [4.0], at) == 4.0)


def test_read_delays_a_signal_by_a_fraction_of_a_sample() -> None:
    n = 4000
    t = np.arange(n) / RATE
    tone = np.sin(2 * np.pi * 3000.0 * t)
    shift = 10.37
    out = dp.read(tone, np.arange(n) - shift, 1.0)
    inner = slice(200, n - 200)
    expected = np.sin(2 * np.pi * 3000.0 * (t - shift / RATE))
    assert np.max(np.abs(out[inner] - expected[inner])) < 1e-4
    # Nothing before the signal starts or after it ends.
    assert np.all(dp.read(tone, np.array([-100.0, n + 100.0]), 1.0) == 0.0)
    # Read faster than it was sampled, the cutoff keeps a tone near Nyquist from
    # folding down.
    high = np.sin(2 * np.pi * 22_000.0 * t)
    sped = dp.read(high, np.arange(n // 2) * 1.5, 1.0 / 1.5)
    assert np.sqrt(np.mean(sped[200:-200] ** 2)) < 1e-3


def test_paths_are_followed_from_place_to_place(short) -> None:
    room = Room.from_scene(short.scene("source", 30.0, "hard"))
    walk = simulate_walk(room, step_m=1.0, late_step_m=None)
    places = [walk.result[name, "l"].early for name in walk.names]
    found = dp.tracks(
        places,
        walk.points.times,
        source_walks=True,
        planes=dp.plane_ids(room.mesh),
        step_s=1.5 / 343.0 + 1e-4,
    )
    # The direct sound and the ground's reflection, all the way.
    assert len(found) == 2
    for track in found:
        assert track.span == (0, len(places) - 1) and track.present.all()
        assert np.all(np.abs(np.diff(track.delays)) < 1.0 / 343.0)
    direct, ground = sorted(found, key=lambda track: track.delays[0])
    assert np.all(direct.delays < ground.delays)
    assert np.allclose(direct.received, walk.points.times + direct.delays)


# -- the sound -----------------------------------------------------------------------


def _standing_walk(ground: str):
    """The standing walk, with the first place's late part at both places, so that only
    the early sound's way differs."""
    walk = simulate_walk(Room.from_scene(_standing(ground)), step_m=1.0, late_step_m=None)
    responses = walk.result.responses
    first = responses[walk.names[0], "l"]
    for name in walk.names[1:]:
        responses[name, "l"] = dataclasses.replace(
            responses[name, "l"],
            late=first.late,
            seed=first.seed,
            late_directions=first.late_directions,
            late_directional=first.late_directional,
        )
    return walk, first


@pytest.mark.parametrize("ground", ["hard", "grass"])
def test_a_walker_that_stands_still_is_a_convolution(ground: str) -> None:
    walk, first = _standing_walk(ground)
    assert walk.doppler
    dry = _noise(1.0)
    wet = render_walk(walk, {"s": dry}, Mono())[:, 0]
    ir = first.spatial_impulse_response(Mono(), RATE, walk.result.settings.seed)
    assert _error_db(wet, convolve(dry, ir)[:, 0]) < -50.0


def test_a_walker_that_stands_still_is_a_convolution_at_the_ears() -> None:
    walk, first = _standing_walk("grass")
    dry = _noise(1.0)
    fmt = Binaural()
    wet = render_walk(walk, {"s": dry}, fmt)
    ref = convolve(dry, first.spatial_impulse_response(fmt, RATE, walk.result.settings.seed))
    for ear in range(2):
        assert _error_db(wet[:, ear], ref[:, ear]) < -45.0


def _delays_hrtf(onsets: np.ndarray, taps: int = 64) -> HRTF:
    """Ears that only delay: each measured direction of a Fibonacci lattice a unit pulse
    at its ``onsets`` (directions, 2) samples."""
    directions = fibonacci_sphere(len(onsets))
    hrir = np.zeros((len(onsets), 2, taps))
    for m, ear in np.ndindex(len(onsets), 2):
        hrir[m, ear, int(onsets[m, ear])] = 1.0
    return HRTF(directions, hrir, RATE)


def test_ears_whose_onsets_jump_keep_the_treble_between_places(short) -> None:
    """Measured ears can start a few samples apart from one direction to the next
    (KEMAR's far ear does, by up to 11 samples between places 1 m apart along a road
    25 m away). Each ear's onset is read in its delay, not mixed in its filter: the
    treble does not cancel between places whose ears start apart. Through ears that only
    delay, over a ground that takes all the sound (the direct sound alone), it is the
    treble of ears that all start together; mixing the filters as they were lost up to
    5.4 dB of it."""
    data = short.scene("source", 30.0, "hard").model_dump(mode="json", exclude_none=True)
    data["materials"]["ground"] = {"absorption": [1.0] * 8, "scattering": 0.0}
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=1.0, late_step_m=None)
    dry = _noise(1.5)
    count = 300
    jumping = 20.0 + 4.0 * (np.arange(count) % 2)[:, None] * np.ones(2)
    together = np.full((count, 2), 20.0)
    wet, flat = (
        render_walk(walk, {"s": dry}, Binaural(hrtf=_delays_hrtf(onsets)))
        for onsets in (jumping, together)
    )
    frame = 960
    f = np.fft.rfftfreq(frame, 1.0 / RATE)
    band = (f >= 6000.0) & (f <= 16000.0)

    def treble_db(x: np.ndarray) -> np.ndarray:
        frames = x[: len(x) // frame * frame].reshape(-1, frame, 2)
        return 10.0 * np.log10(np.sum(np.abs(np.fft.rfft(frames, axis=1)[:, band]) ** 2, axis=1))

    heard = treble_db(flat) > treble_db(flat).max() - 30.0
    assert np.abs(treble_db(wet) - treble_db(flat))[heard].max() < 0.3


def test_a_passing_source_and_listener_sound_as_the_exact_field(short) -> None:
    speed, tone = 30.0, 1000.0
    walked = (short.X1 - short.X0) / speed
    t = np.arange(int((walked + 0.5) * RATE)) / RATE
    dry = np.sin(2 * np.pi * tone * t)
    for walker in ("source", "listener"):
        _, (wet,) = short.run(walker, speed, "hard", True, [dry])
        _, (faded,) = short.run(walker, speed, "hard", False, [dry])
        field = short.exact(walker, speed, tone, len(wet))
        with_doppler = short.errors(wet, field, walked)
        assert with_doppler["waveform_db"] < -45.0, walker
        assert with_doppler["level_max_db"] < 0.1, walker
        # Crossfading from place to place, the pitch stays put and the phase jumps.
        assert short.errors(faded, field, walked)["waveform_db"] > 0.0, walker


@pytest.mark.parametrize("directivity", [None, "cardioid"])
def test_the_sound_sent_each_way_as_the_exact_field(short, directivity) -> None:
    """Each path carries what the source sends the way it leaves (a tone louder ahead,
    turned in phase with the height), and a directivity of the source's own is taken
    out of the early paths."""
    speed, tone = 30.0, 1000.0
    walked = (short.X1 - short.X0) / speed
    sent = short.emission(tone, int((walked + 0.5) * RATE))
    for walker in ("source", "listener"):
        room = Room.from_scene(short.scene(walker, speed, "hard", True, directivity))
        walk = simulate_walk(room, step_m=short.STEP_M, late_step_m=None)
        wet = render_walk(walk, {"s": sent}, Mono())[:, 0]
        field = short.exact(walker, speed, tone, len(wet), sent=True)
        errors = short.errors(wet, field, walked)
        assert errors["waveform_db"] < -45.0, walker
        assert errors["level_max_db"] < 0.1, walker


def test_the_same_sound_sent_every_way_is_the_signal_itself(short) -> None:
    walked = (short.X1 - short.X0) / 30.0
    tone = np.sin(2 * np.pi * 1000.0 * np.arange(int((walked + 0.5) * RATE)) / RATE)
    elevation, azimuth = _grid(10.0)
    every_way = Mixture(tone[None, :], np.ones((len(elevation), len(azimuth), 1)), RATE)
    walk = simulate_walk(Room.from_scene(short.scene("source", 30.0, "hard")), late_step_m=None)
    plain = render_walk(walk, {"s": tone}, Mono())
    sent = render_walk(walk, {"s": every_way}, Mono())
    np.testing.assert_allclose(sent, plain, rtol=0.0, atol=1e-12 * np.abs(plain).max())


def test_the_sound_sent_each_way_needs_a_walk_with_doppler(short) -> None:
    elevation, azimuth = _grid(30.0)
    sent = Mixture(np.ones((1, 100)), np.ones((len(elevation), len(azimuth), 1)), RATE)
    walk = simulate_walk(Room.from_scene(short.scene("source", 30.0, "hard", False)))
    with pytest.raises(ValueError, match="plays only along a walk with Doppler"):
        render_walk(walk, {"s": sent}, Mono())
    walk = simulate_walk(Room.from_scene(short.scene("source", 30.0, "hard")), late_step_m=None)
    with pytest.raises(ValueError, match="at 48000 Hz, the rendering at 44100 Hz"):
        render_walk(walk, {"s": sent}, Mono(), sample_rate=44_100)


def test_the_late_part_and_the_eddies_go_along(short) -> None:
    data = short.scene("source", 30.0, "grass").model_dump(mode="json", exclude_none=True)
    data["atmosphere"] = {"turbulence": "strong"}
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=2.0, late_step_m=4.0)
    dry = _noise(1.5)
    wet = render_walk(walk, {"s": dry}, Mono())
    assert np.isfinite(wet).all() and np.max(np.abs(wet)) > 0.0
    calm = simulate_walk(
        Room.from_scene(short.scene("source", 30.0, "grass")), step_m=2.0, late_step_m=4.0
    )
    still = render_walk(calm, {"s": dry}, Mono())
    assert not np.allclose(wet, still)
    # Rendered again, the same: the eddies and the late part are drawn the same way.
    assert np.array_equal(wet, render_walk(walk, {"s": dry}, Mono()))


def test_a_walker_standing_in_the_eddies_hears_each_path_half_way_between_moments() -> None:
    """With Doppler too, each path's delay goes from one moment of the eddies to the
    next's, its top band's in the delay and the rest in the filter: half-way, a tone
    at 8 kHz has the response whose paths are delayed half-way. The filter alone held
    only how the other bands lag behind the top one, so the eddies did not move the
    top band at all: the largest difference here was 43 % of the tone's rms amplitude
    (now 0.009 %)."""
    from test_turbulence import half_way_errors

    data = _standing("hard").model_dump(mode="json", exclude_none=True)
    data["atmosphere"] = {"turbulence": {"preset": "strong", "outer_scale_m": 2.0}}
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), late_step_m=None)
    response = next(iter(walk.result.responses.values()))
    rate, f = RATE, 8000.0
    dry = np.sin(2 * np.pi * f * np.arange(3 * rate) / rate)
    wet = render_walk(walk, {"s": dry}, Mono())[:, 0]
    seed = walk.result.settings.seed
    gusts = gusts_for(response, len(wet) / rate, seed)
    moments = range(4, int(2.5 / gusts.step_s))
    assert half_way_errors(response, wet, f, rate, gusts, seed, moments) < -40.0


def test_a_walking_listener_hears_a_moving_and_a_standing_source(short) -> None:
    data = short.scene("listener", 30.0, "hard").model_dump(mode="json", exclude_none=True)
    data["sources"].append({"name": "t", "position": [0.0, -10.0, 1.0]})
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=1.0, late_step_m=None)
    dry = _noise(1.5)
    both = render_walk(walk, {"s": dry, "t": dry}, Mono())
    alone = [render_walk(walk, {name: dry}, Mono()) for name in ("s", "t")]
    assert np.allclose(both, alone[0] + alone[1], atol=1e-12)


# -- the levels ----------------------------------------------------------------------


def test_pass_by_levels(short) -> None:
    room = Room.from_scene(short.scene("source", 30.0, "hard"))
    walk = simulate_walk(room, step_m=1.0, late_step_m=None)
    passing = pass_by(walk)
    assert (passing.source, passing.listener) == ("s", "l")
    responses = [walk.result[name, "l"] for name in walk.names]
    delays = np.array([r.direct_delay_s for r in responses])
    assert np.allclose(passing.times, walk.points.times + delays)
    assert passing.levels.shape == (len(walk.names), 8)
    # Loudest nearest, at the middle of the walk.
    assert passing.max_time == pytest.approx(passing.times[len(walk.names) // 2], abs=0.04)
    assert passing.max_a == pytest.approx(np.max(passing.levels_a))
    # SEL over the time heard: L_Aeq + 10 lg T.
    assert passing.equivalent_a == pytest.approx(
        passing.exposure_a - 10.0 * math.log10(passing.duration_s)
    )
    assert passing.duration_s == pytest.approx(passing.times[-1] - passing.times[0])
    assert passing.equivalent_a < passing.max_a < passing.exposure_a + 10.0
    power = 10.0 ** (passing.levels_a / 10.0)
    energy = np.sum((power[1:] + power[:-1]) / 2.0 * np.diff(passing.times))
    assert passing.exposure_a == pytest.approx(10.0 * math.log10(energy))
    with pytest.raises(ValueError, match="does not walk"):
        pass_by(walk, "l")


def test_pass_by_of_a_walking_listener(short) -> None:
    data = short.scene("listener", 30.0, "hard").model_dump(mode="json", exclude_none=True)
    data["sources"].append({"name": "t", "position": [0.0, -10.0, 1.0]})
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=2.0, late_step_m=None)
    with pytest.raises(ValueError, match="pick a source"):
        pass_by(walk)
    with pytest.raises(ValueError, match="no source 'x'"):
        pass_by(walk, "x")
    passing = pass_by(walk, "t")
    assert passing.listener == "l" and np.allclose(passing.times, walk.points.times)


def test_cli_prints_and_saves_the_pass_by(
    short, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from acoustic_spaces.audio import write_wav

    path = tmp_path / "scene.yaml"
    short.scene("source", 30.0, "hard").to_yaml(path)
    dry = tmp_path / "dry.wav"
    write_wav(dry, _noise(0.5)[:, None] * 0.1, RATE)
    out, csv = tmp_path / "walk.wav", tmp_path / "levels.csv"
    args = ["walkthrough", str(path), f"s={dry}", "--out", str(out), "--step", "2"]
    assert main([*args, "--levels", str(csv)]) == 0
    printed = capsys.readouterr().out
    assert "s at l: L_Amax " in printed and "SEL " in printed and "L_Aeq " in printed
    lines = csv.read_text().splitlines()
    assert lines[0].startswith("source,listener,time_s,L_A_dB,L_63Hz_dB")
    assert len(lines) == 1 + 16 and lines[1].startswith("s,l,")
    assert main(["inspect", str(path)]) == 0
    assert "with Doppler" in capsys.readouterr().out
    assert main([*args, "--levels", str(tmp_path / "missing" / "levels.csv")]) == 1
    assert "error: " in capsys.readouterr().err
