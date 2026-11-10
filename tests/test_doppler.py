"""Walks with Doppler (``walk.doppler``, :mod:`acoustic_spaces.doppler`): the delays,
the paths followed from place to place, the sound against the exact field of a passing
source and listener, and the pass-by levels (:func:`acoustic_spaces.walkthrough.pass_by`)."""

import dataclasses
import functools
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces import doppler as dp
from acoustic_spaces.cli import main
from acoustic_spaces.directivity import _grid
from acoustic_spaces.emission import Mixture
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.render import convolve, gusts_for
from acoustic_spaces.scene.model import Walk
from acoustic_spaces.spatial import Ambisonics, Binaural, Loudspeakers, Mono
from acoustic_spaces.spatial.directions import fibonacci_sphere
from acoustic_spaces.spatial.hrtf import HRTF
from acoustic_spaces.spatial.microphone import first_order, xy
from acoustic_spaces.walkthrough import (
    crossfade_weights,
    pass_by,
    render_walk,
    simulate_movement,
    simulate_walk,
)

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


@pytest.mark.parametrize("doppler", [True, False])
@pytest.mark.parametrize("walker", ["source", "listener"])
def test_every_walk_goes_slower_than_sound(walker: str, doppler: bool) -> None:
    """With Doppler or without, a walk as fast as sound or faster is refused before
    anything is simulated: shock waves are not simulated."""
    room = Room.from_scene(reference.scene(walker, 400.0, "hard", doppler))
    with pytest.raises(ValueError, match="as fast as sound or faster"):
        simulate_walk(room, step_m=10.0, late_step_m=None)
    with pytest.raises(ValueError, match="as fast as sound or faster"):
        simulate_movement(room, step_m=10.0)
    # Just slower than sound is a walk.
    room = Room.from_scene(reference.scene(walker, 330.0, "hard", doppler))
    assert len(simulate_walk(room, step_m=40.0, late_step_m=None).names) > 1


def test_doppler_needs_no_wave_solver() -> None:
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


def test_the_slope_of_the_curve_through_the_points() -> None:
    x = np.array([0.0, 1.0, 2.5, 3.0, 5.0])
    y = np.sin(x)
    at = np.linspace(0.01, 4.99, 499)
    step = 1e-6
    chords = (dp.hermite(x, y, at + step) - dp.hermite(x, y, at - step)) / (2.0 * step)
    assert np.allclose(dp._slope(x, y, at), chords, rtol=0.0, atol=1e-5)
    # Held beyond its ends, the curve has no slope there; nor has one point's.
    assert np.all(dp._slope(x, y, np.array([-1.0, 6.0])) == 0.0)
    assert np.all(dp._slope(np.array([1.0]), np.array([4.0]), at) == 0.0)


def test_an_ear_hears_what_passed_the_head_an_onset_earlier() -> None:
    """A walking source's sound passes the head's centre at a time s and reaches an ear
    at s + onset(s). From the times heard, ``passed`` gives s, and how much of s goes
    by in each moment heard: one over (1 + the onset's slope), here up to 0.2 % either
    way."""
    received = np.linspace(0.0, 1.0, 21)
    turn = 3e-4 * np.sin(2.0 * np.pi * received)
    onsets = 1e-3 + np.column_stack([turn, -turn])
    track = dp.Track(
        np.ones(21, dtype=bool), np.zeros(21, dtype=np.int64), received, np.full(21, 0.01)
    )
    reading = dp._Reading(track, np.arange(21), 0, 0, 1.0, None, onsets)
    t = np.linspace(0.1, 0.9, 2001)
    for ear in (0, 1):
        s, share = reading.passed(t, ear)
        assert np.allclose(s + reading.onset(s, ear), t, rtol=0.0, atol=1e-10)
        assert np.allclose(share[1:-1], np.gradient(s, t)[1:-1], rtol=0.0, atol=1e-6)
        assert 1.5e-3 < np.abs(share - 1.0).max() < 2e-3


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


def test_a_filter_goes_linearly_from_one_place_to_the_next() -> None:
    """A path twice as loud at the next place, 0.2 s later (ten windows of the filter),
    and no further: a tone grows linearly in between, as the triangles that join the
    windows give it. Squared sines there rippled once a window about the line, by 0.7 %
    of the tone's peak."""
    walk, first = _standing_walk("hard")
    louder = dataclasses.replace(first.early, energy=4.0 * first.early.energy)
    responses = [first, dataclasses.replace(first, early=louder)]
    t = np.arange(int(0.4 * RATE)) / RATE
    dry = np.sin(2 * np.pi * 1000.0 * t)

    def heard(places: list, times: list[float]) -> np.ndarray:
        planes = dp.plane_ids(walk.result.room.mesh)
        return dp.render(
            dry,
            places,
            np.array(times),
            Mono(),
            RATE,
            len(dry),
            source_walks=False,
            planes=planes,
            step_s=1e-4,
        )[:, 0]

    growing, steady = heard(responses, [0.1, 0.3]), heard(responses[:1], [0.0])
    assert _error_db(growing, np.interp(t, [0.1, 0.3], [1.0, 2.0]) * steady) < -70.0
    # Places heard 6 ms apart, more than three within one window of the filter (HOP_S),
    # the path twice as loud at every other one: the filter is taken at each place's
    # time too, so the tone still goes linearly from each place to the next. Taken
    # every HOP_S alone it skipped places, and the difference was 17 % of the tone (RMS).
    times = 0.1 + 0.006 * np.arange(34)
    gains = 1.0 + np.arange(34) % 2
    waving = heard([responses[k % 2] for k in range(34)], list(times))
    assert _error_db(waving, np.interp(t, times, gains) * steady) < -70.0


def test_the_windows_of_a_filter_add_up_to_one_and_turn_at_the_places() -> None:
    """The filter is taken at every block's centre and at the places' times between
    them; each time's window is one there and falls linearly to nothing at the times on
    either side, so together they add up to one at every sample."""
    hop, rate, blocks = 960, RATE, 8
    received = np.array([0.0, 0.0213, 0.0251, 0.04, 0.0537, 0.2])
    when, at, block = dp._turns(received, rate, hop, blocks)
    # A place heard at a block's centre (0 and 0.04 s) adds no time of its own, nor
    # does one heard past the last block's centre (0.2 s).
    assert when.tolist() == [0.0213, 0.0251, 0.0537]
    assert np.allclose(at, when * rate) and block.tolist() == [1, 1, 2]
    assert dp._turns(received[:1], rate, hop, blocks)[0].size == 0
    reading = dp._Reading(dp.Track(*[np.zeros(0)] * 4), np.zeros(0), 0, 7, 1.0, None, None)
    plain = reading.taken(3, hop, rate)
    assert plain[0] == [3 * hop / rate] and plain[1].tolist() == [2 * hop, 3 * hop, 4 * hop]
    reading = dataclasses.replace(reading, turn_s=when, turn_at=at, turn_block=block)
    samples = np.arange(0, 5 * hop)
    total = np.zeros(len(samples))
    for j in range(0, 6):
        whens, turns = reading.taken(j, hop, rate)
        assert whens[0] == j * hop / rate and whens[1:] == when[block == j].tolist()
        windows = dp._windows(turns, samples)
        assert windows.shape == (len(whens), len(samples))
        # One at its own time, nothing at the others' nor from the times on either side on.
        assert np.array_equal(dp._windows(turns, turns[1:-1]), np.eye(len(whens)))
        assert not np.any(windows[:, (samples <= turns[0]) | (samples >= turns[-1])])
        total += windows.sum(axis=0)
    np.testing.assert_allclose(total, 1.0, rtol=0.0, atol=1e-12)
    # The block after the place at 0.0537 s rises from it, not from the centre before.
    assert reading.taken(3, hop, rate)[1].tolist() == [0.0537 * rate, 3 * hop, 4 * hop]


def _late_only(walker: str, monkeypatch: pytest.MonkeyPatch):
    """The short pass-by of a walking ``walker``, 1 m between places, each with its own
    late part, and the early paths left out of its renders: the walk and each place's
    late part as it plays it (the response without its early paths)."""
    walk = simulate_walk(
        Room.from_scene(reference.scene(walker, 30.0, "grass")), step_m=1.0, late_step_m=None
    )
    monkeypatch.setattr(dp, "render", lambda *args, **kwargs: 0.0)
    seed = walk.result.settings.seed
    irs = []
    for name in walk.names:
        r = walk.result[(name, "l") if walker == "source" else ("s", name)]
        r = dataclasses.replace(r, early=EarlyReflections.empty(r.bands))
        irs.append(r.spatial_impulse_response(Mono(), RATE, seed)[:, 0])
    return walk, irs


def test_a_walking_source_sends_its_late_part_from_where_it_is(short, monkeypatch) -> None:
    """A walking source's late part is that of the place it sends its sound from, as its
    early paths are (heard at the retarded time): a click sent while the source is at a
    place gives that place's late part, from the moment it was sent. Faded from place to
    place as it was heard instead, it was the late part of the places the source had
    gone on to: here off by 38 % of the place's late part's rms amplitude."""
    walk, irs = _late_only("source", monkeypatch)
    times = walk.points.times
    k = len(times) // 2
    sent = round(times[k] * RATE)
    assert crossfade_weights(times, [sent / RATE])[k, 0] == 1.0
    # Its late part arrives once the source has gone more than half-way to the next.
    assert walk.result[walk.names[k], "l"].direct_delay_s > (times[k + 1] - times[k]) / 2
    click = np.zeros(RATE)
    click[sent] = 1.0
    wet = render_walk(walk, {"s": click}, Mono())[:, 0]
    expected = np.zeros_like(wet)
    expected[sent : sent + len(irs[k])] = irs[k][: len(wet) - sent]
    assert _error_db(wet, expected) < -100.0


def test_a_walking_source_with_one_late_part_everywhere_plays_it(short, monkeypatch) -> None:
    """The places' weights, read as the sound is sent, add up to one: with the same
    response at every place, a walking source's late part is the signal through it."""
    walk, irs = _late_only("source", monkeypatch)
    responses = walk.result.responses
    for name in walk.names[1:]:
        responses[name, "l"] = responses[walk.names[0], "l"]
    dry = _noise(1.5)
    wet = render_walk(walk, {"s": dry}, Mono())[:, 0]
    expected = convolve(dry, irs[0])
    assert np.max(np.abs(wet[len(expected) :]), initial=0.0) == 0.0
    np.testing.assert_allclose(
        wet[: len(expected)], expected, rtol=0.0, atol=1e-12 * np.abs(expected).max()
    )


def test_a_walking_listener_hears_the_late_part_of_where_it_is(short, monkeypatch) -> None:
    """A walking listener hears, at each moment, the late part of the place it is at:
    the sound through each place's late part, faded from place to place as it is
    heard."""
    walk, irs = _late_only("listener", monkeypatch)
    dry = _noise(1.0)
    wet = render_walk(walk, {"s": dry}, Mono())[:, 0]
    weights = crossfade_weights(walk.points.times, np.arange(len(wet)) / RATE)
    expected = np.zeros_like(wet)
    for weight, ir in zip(weights, irs, strict=True):
        heard = convolve(dry, ir)[: len(wet)]
        expected[: len(heard)] += weight[: len(heard)] * heard
    np.testing.assert_allclose(wet, expected, rtol=0.0, atol=1e-12 * np.abs(expected).max())


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


@pytest.mark.parametrize("walker", ["source", "listener"])
@pytest.mark.parametrize("heard_by", ["mono", "cardioid", "ears"])
def test_a_fast_walker_passing_close_keeps_the_level_of_the_exact_field(
    short, walker, heard_by
) -> None:
    """At 70 m/s, 2 m from the other's line, the distance changes by more than half of
    itself within ``HOP_S``, and the walker passes more than five places: the level
    follows one over the distance sample by sample, and through a cardioid microphone
    aimed at the walker's line the pattern from place to place (the filter is taken at
    each place's time), here in windows a quarter of ``HOP_S`` long. Between two places
    0.25 m apart a straight line strays from the capsule's 0.5 + 0.5 cos by 0.01 dB at
    most (a chord's sag, an eighth of the step squared times the curvature, 0.5 / 2.06²
    a metre squared at the pass).

    At the ears of a listener the walker passes on one side, a walking source's sound
    is what a standing head makes of the pressure at its centre, each sample of it
    through the ears' responses to the direction it arrives from, and a walking
    listener's ears are each a point their onset further along the path
    (``reference._at_ears``). The level stays within 0.02 dB there. With a source's
    onset added to its delay instead, it was 2.7 dB off and the difference from the
    field 61 % of the field; with the filter taken every ``HOP_S`` alone, a walking
    listener's level was 0.27 dB off."""
    speed, tone = 70.0, 1000.0
    line = dataclasses.replace(short.road(), x0=-8.0, x1=8.0, d=2.0)
    walked = (line.x1 - line.x0) / speed
    t = np.arange(int((walked + 0.5) * RATE)) / RATE
    dry = np.sin(2 * np.pi * tone * t)
    step = short.CLOSE_STEP_M
    _, (wet,) = short.run(walker, speed, "hard", True, [dry], step, line, heard_by)
    field = short.exact(walker, speed, tone, len(wet), line=line, heard_by=heard_by)
    found = short.errors(wet, field, walked, window_s=dp.HOP_S / 4)
    assert found["level_max_db"] < 0.05
    assert found["waveform_db"] < -45.0


@pytest.mark.parametrize("way", ["goes", "comes"])
def test_a_path_that_goes_or_comes_is_nowhere_louder_than_at_the_places_it_is_heard_from(
    way: str,
) -> None:
    """A listener walks 1 m at a time at a source (``goes``: 1.9 m, 0.9 m and 0.1 m
    from it) or away from it (``comes``: 0.1 m, 0.9 m, 1.9 m), and the direct sound is
    not heard at the place 0.1 m away. Between that place and the next it fades, its
    delay carried on along its slope towards zero; the fade is the filter's, from one
    over the distance at the place it is heard from (here 0.9 m), never above it."""
    x0, x1 = (-1.9, 0.1) if way == "goes" else (-0.1, 1.9)
    line = reference.Line(x0, x1, 0.05, 1.5, 1.5)
    room = Room.from_scene(reference.scene("listener", 10.0, "hard", line=line))
    dry = np.sin(2 * np.pi * 1000.0 * np.arange(int(0.5 * RATE)) / RATE)

    def without_direct(at: list[int]) -> np.ndarray:
        walk = simulate_walk(room, step_m=1.0, late_step_m=None)
        assert len(walk.names) == 3
        responses = walk.result.responses
        for k in at:
            place = responses["s", walk.names[k]]
            responses["s", walk.names[k]] = dataclasses.replace(
                place, early=place.early[place.early.order > 0]
            )
        return render_walk(walk, {"s": dry}, Mono())[:, 0]

    gone = 2 if way == "goes" else 0
    # The paths add: what the direct sound alone gives, the ground's reflection taken out.
    direct = without_direct([gone]) - without_direct([0, 1, 2])
    nearest_heard = 1.0 / math.hypot(0.9, 0.05)
    assert 0.99 * nearest_heard < np.abs(direct).max() < 1.001 * nearest_heard


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


class _Forgetful(OrderedDict):
    """A cache that keeps nothing: every filter, path and angle is worked out anew."""

    def __setitem__(self, key, value) -> None:
        pass


@pytest.mark.parametrize(
    "fmt",
    [
        Mono(),
        Binaural(),
        Ambisonics(order=1),
        Loudspeakers.layout("5.0"),
        xy(),
        first_order("hypercardioid"),
    ],
    ids=["mono", "binaural", "foa", "5.0", "xy", "hypercardioid"],
)
def test_the_caches_change_nothing(short, monkeypatch: pytest.MonkeyPatch, fmt) -> None:
    """The filters, the paths at each moment of the eddies and their phases' angles that
    a walk with Doppler keeps for the places in use give, sample for sample, what working
    them out anew each time gives, in every format."""
    data = short.scene("source", 30.0, "grass").model_dump(mode="json", exclude_none=True)
    data["atmosphere"] = {"turbulence": "strong"}
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=2.0, late_step_m=4.0)
    dry = _noise(0.6)
    kept = render_walk(walk, {"s": dry}, fmt)
    monkeypatch.setattr(dp, "OrderedDict", _Forgetful)
    anew = render_walk(walk, {"s": dry}, fmt)
    assert np.max(np.abs(kept)) > 0.0
    assert np.array_equal(kept, anew)


def test_each_filter_is_worked_out_once_and_kept_while_in_use(
    short, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A walk with Doppler works out the filters of each place (at each moment of the
    eddies) once, and keeps them only from the first block that takes them to the
    last: a walk whose first two knots are 1 ms apart kept every place's (17 here),
    and keeps 3 at most."""
    made: list[tuple[int, int]] = []
    most = [0]
    spectra = dp.Filters._spectra

    def counted(self, k: int, moment: int):
        made.append((k, moment))
        most[0] = max(most[0], len(self.cache) + 1)
        return spectra(self, k, moment)

    monkeypatch.setattr(dp.Filters, "_spectra", counted)
    data = short.scene("source", 30.0, "hard").model_dump(mode="json", exclude_none=True)
    walk = data["sources"][0]["walk"]
    del walk["to"], walk["speed_m_s"]
    walk["keyframes"] = [
        {"time_s": 0.001, "position": [short.X0 + 0.03, 0.0, short.ZS]},
        {"time_s": 1.0, "position": [short.X1, 0.0, short.ZS]},
    ]
    keyed = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=2.0, late_step_m=None)
    assert np.diff(keyed.points.times)[0] == pytest.approx(0.001)
    render_walk(keyed, {"s": _noise(1.0)}, Mono())
    assert sorted(made) == [(k, 0) for k in range(17)] and len(keyed.names) == 17
    assert most[0] == 3
    # In turbulence, at two moments at once.
    data = short.scene("source", 30.0, "grass").model_dump(mode="json", exclude_none=True)
    data["atmosphere"] = {"turbulence": "strong"}
    walk = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=2.0, late_step_m=4.0)
    made.clear()
    most[0] = 0
    render_walk(walk, {"s": _noise(0.6)}, Mono())
    assert len(made) == len(set(made)) and len({moment for _, moment in made}) > 1
    assert most[0] <= 6


@pytest.mark.parametrize(
    "fmt", [Ambisonics(order=3), Loudspeakers.layout("7.0.4")], ids=["ambix3", "7.0.4"]
)
def test_filters_kept_as_shaping_and_gains_are_the_channels_spectra(
    short, monkeypatch: pytest.MonkeyPatch, fmt
) -> None:
    """From three channels on, a place's filters are kept as the paths' shaping
    (complex128, one row per path) and the format's gains (one per path and channel):
    an eighth of the bytes of every channel's spectra in third-order Ambisonics. Each
    path's filter formed from them is, bit for bit, its row of every channel's
    spectra."""
    walk = simulate_walk(
        Room.from_scene(short.scene("source", 30.0, "hard")), step_m=4.0, late_step_m=None
    )
    responses = [walk.result[name, "l"] for name in walk.names]
    n_fft = 1 << 15
    kept = dp.Filters(responses, fmt, RATE, n_fft, n_fft // 4, None, None)
    monkeypatch.setattr(dp, "_SHAPING_FROM", fmt.channels + 1)
    whole = dp.Filters(responses, fmt, RATE, n_fft, n_fft // 4, None, None)
    for k in range(len(responses)):
        spectra = whole._spectra(k, 0)
        shaping, gains = kept._spectra(k, 0)
        paths = len(responses[k].early)
        assert paths > 1 and spectra.shape == (paths, fmt.channels, n_fft // 2 + 1)
        assert shaping.shape == (paths, n_fft // 2 + 1) and shaping.dtype == np.complex128
        assert gains.shape == (paths, fmt.channels)
        for row in range(paths):
            assert np.array_equal(kept.row(k, 0, row), spectra[row])
    assert np.array_equal(kept.row(0, 0, 1), whole.row(0, 0, 1))


def test_the_paths_share_a_reading_table_for_each_reach(
    short, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each path is read with a sinc cut off at 0.97 of Nyquist over the most its pitch
    rises, rounded down to the lowest cutoff with as many taps, so no lower than a share
    1 / n of it, n the taps on each side: the paths share one table of taps for each n
    (the car's direct sound and the ground's reflection had one each)."""
    readings: list = []
    reading = dp._Reading

    def kept(*args):
        readings.append(reading(*args))
        return readings[-1]

    built: list[float] = []
    build = dp._table.__wrapped__

    def table(cutoff: float):
        built.append(cutoff)
        return build(cutoff)

    monkeypatch.setattr(dp, "_Reading", kept)
    monkeypatch.setattr(dp, "_table", functools.lru_cache(maxsize=16)(table))
    walk = simulate_walk(
        Room.from_scene(short.scene("source", 30.0, "hard")), step_m=1.0, late_step_m=None
    )
    render_walk(walk, {"s": _noise(1.2)}, Mono())
    assert len(readings) >= 2
    for each in readings:
        heard = each.heard
        rises = np.max(
            1.0 - np.diff(each.track.delays[heard]) / np.diff(each.track.received[heard])
        )
        cutoff = 0.97 / max(rises, 1.0)
        assert cutoff * (1.0 - 1.0 / dp._reach(cutoff)) < each.cutoff <= cutoff
    assert len(built) == len({dp._reach(cutoff) for cutoff in built}) >= 1
    # 32 / (32 / n) rounds to just over n for some n: their cutoff still takes n taps.
    for n in (33, 46, 49, 98, 103, 107, 196, 392):
        assert dp._reach(dp._ZEROS / dp._reach(dp._ZEROS / n)) == n
        assert dp._reach(dp._ZEROS / n * (1.0 - 1e-6)) == n + 1


def test_sounds_that_follow_the_walker_are_heard_as_its_own_later(short) -> None:
    """A sound following the walker (a train's wheels behind its front) is heard as the
    walker's own sound would be, its lag later, through the early paths and the late
    part: exactly a render of its own, delayed, but for the filters' faint lead-in that
    such a render cuts at its start."""
    walk = simulate_walk(
        Room.from_scene(short.scene("source", 30.0, "grass")), step_m=2.0, late_step_m=4.0
    )
    lead, behind = _noise(1.0, seed=3), _noise(1.0, seed=4)
    first = render_walk(walk, {"s": lead}, Mono())
    second = render_walk(walk, {"s": behind}, Mono())
    for lag_s in (0.25, 0.0):
        lag = round(lag_s * RATE)
        together = render_walk(walk, {"s": lead}, Mono(), followers=[(lag_s, behind)])
        assert len(together) == lag + len(second)
        expected = np.zeros_like(together)
        expected[: len(first)] += first
        expected[lag:] += second
        peak = np.abs(expected).max()
        np.testing.assert_allclose(together[lag:], expected[lag:], rtol=0.0, atol=1e-12 * peak)
        assert np.abs(together[:lag] - expected[:lag]).max(initial=0.0) < 1e-3 * peak
    with pytest.raises(ValueError, match="0 s or more behind"):
        render_walk(walk, {"s": lead}, Mono(), followers=[(-0.1, behind)])
    elevation, azimuth = _grid(30.0)
    sent = Mixture(lead[None, :], np.ones((len(elevation), len(azimuth), 1)), RATE)
    with pytest.raises(ValueError, match="sends one sound every way"):
        render_walk(walk, {"s": sent}, Mono(), followers=[(0.1, behind)])
    for still in (
        simulate_walk(Room.from_scene(short.scene("source", 30.0, "hard", False))),
        simulate_walk(Room.from_scene(short.scene("listener", 30.0, "hard")), late_step_m=None),
    ):
        with pytest.raises(ValueError, match="a walking source along a walk with Doppler"):
            render_walk(still, {"s": lead}, Mono(), followers=[(0.1, behind)])


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
