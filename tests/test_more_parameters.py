"""The parameters beyond those of an omnidirectional response: lateral fractions, IACC,
stage support, directional decay, STIPA, U50 and the echo criteria, against responses
whose values are known."""

import math

import numpy as np
import pytest

from acoustic_spaces import parameters as iso3382
from acoustic_spaces.compare import compare_parameters
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.response import Response
from acoustic_spaces.parameters import (
    ALL_PARAMETERS,
    PARAMETERS,
    RoomParameters,
    echo_criterion,
    from_energy,
    from_impulse_response,
    interaural_cross_correlation,
    speech_transmission_index,
    useful_to_detrimental,
)

FS = 48_000
DT = 1e-3
C = 343.0
REFERENCE = 1.0 / (4.0 * math.pi * 100.0)
#: The bands the octave filters of an impulse response resolve a 20 ms gap in.
MIDDLE = slice(3, 8)  # 500 Hz to 8 kHz


def _paths(delays, directions, energies) -> EarlyReflections:
    directions = np.array(directions, dtype=float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    delays = np.array(delays, dtype=float)
    return EarlyReflections(
        order=np.arange(len(delays)),
        delay_s=delays,
        distance_m=delays * C,
        direction=directions,
        energy=np.repeat(np.array(energies, dtype=float)[:, None], 8, axis=1),
    )


def _response(early: EarlyReflections, late=None, length=1.0, **more) -> Response:
    """The listener faces +x with the source straight ahead, 10 ms away."""
    slots = int(round(length / DT))
    return Response(
        source="s",
        listener="l",
        early=early,
        late=np.zeros((8, slots)) if late is None else late,
        time_step=DT,
        length_s=length,
        direct_delay_s=0.010,
        transition_s=1.0,
        volume_m3=1000.0,
        speed_of_sound=C,
        **more,
    )


def _decaying(t60: float, start: float, energy: float, length: float = 1.0) -> np.ndarray:
    """(8, slots) late energy decaying from ``start`` with ``t60``, ``energy`` in all."""
    t = np.arange(int(round(length / DT))) * DT
    k = 6.0 * math.log(10.0) / t60
    e = np.where(t >= start, np.exp(-k * (t - start)), 0.0)
    return np.repeat((energy * e / e.sum())[None, :], 8, axis=0)


# -- lateral energy ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "lf", "lfc"),
    [
        ((0.0, 1.0, 0.0), 0.5 / 1.5, 0.5 / 1.5),  # from the side: all lateral
        ((1.0, 1.0, 0.0), 0.5 * 0.5 / 1.5, 0.5 * math.sqrt(0.5) / 1.5),  # 45 degrees
        ((-1.0, 0.0, 0.0), 0.0, 0.0),  # from behind: in the figure of eight's null
        ((0.0, 0.0, 1.0), 0.0, 0.0),  # from above
    ],
)
def test_lateral_fractions_of_one_reflection(direction, lf, lfc) -> None:
    # The direct sound from the front with energy 1, a reflection 20 ms later with 0.5.
    early = _paths([0.010, 0.030], [(1, 0, 0), direction], [1.0, 0.5])
    p = _response(early).parameters()
    assert p.lf == pytest.approx(np.full(8, lf), abs=1e-9)
    assert p.lfc == pytest.approx(np.full(8, lfc), abs=1e-9)
    assert np.isnan(p.lj).all()  # nothing lateral after 80 ms


def test_late_lateral_level_is_g_of_the_figure_of_eight() -> None:
    early = _paths([0.010, 0.050, 0.150], [(1, 0, 0), (0, 1, 0), (0, -1, 0)], [1.0, 0.3, 0.2])
    p = _response(early).parameters()
    assert p.lj == pytest.approx(np.full(8, 10.0 * math.log10(0.2 / REFERENCE)))
    assert p.lf == pytest.approx(np.full(8, 0.3 / 1.3))


def test_the_figure_of_eight_turns_with_the_source() -> None:
    # Source to the listener's left: the lateral axis is then the listener's front.
    early = _paths([0.010, 0.030], [(0, 1, 0), (1, 0, 0)], [1.0, 0.5])
    response = _response(early, source_direction=np.array([0.0, 1.0, 0.0]))
    assert abs(response.lateral_axis() @ [1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert response.parameters().lf == pytest.approx(np.full(8, 1 / 3))


def test_late_energy_from_all_around_is_a_third_lateral() -> None:
    late = _decaying(1.0, 0.015, 0.5)
    early = _paths([0.010], [(1, 0, 0)], [1.0])
    p = _response(early, late).parameters()
    e = np.cumsum(late[0])
    expected = (e[89] - e[14]) / 3.0 / (1.0 + e[89])  # 5 to 80 ms over 0 to 80 ms
    assert p.lf == pytest.approx(np.full(8, expected), rel=0.02)
    assert p.lfc == pytest.approx(np.full(8, 1.5 * expected), rel=0.02)


def test_late_energy_keeps_its_directions() -> None:
    # Late energy all from the sides: the figure of eight picks up all of it.
    directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)], dtype=float)
    late = _decaying(1.0, 0.015, 0.5)
    split = np.zeros((4, 8, 100))
    coarse = late.reshape(8, 100, 10).sum(axis=2)
    split[2] = split[3] = 0.5 * coarse
    response = _response(
        _paths([0.010], [(1, 0, 0)], [1.0]),
        late,
        late_directions=directions,
        late_directional=split,
        direction_step=0.01,
    )
    e = np.cumsum(late[0])
    expected = (e[89] - e[14]) / (1.0 + e[89])
    assert response.parameters().lf == pytest.approx(np.full(8, expected), rel=0.02)


def test_lateral_fractions_from_impulse_responses_match_the_energies() -> None:
    # W and the Y of first-order Ambisonics: the direct sound from the front, then
    # reflections from the left and from 45 degrees to the right.
    n = FS // 2
    w, y = np.zeros(n), np.zeros(n)
    arrivals = [(0.010, 1.0, 0.0), (0.030, 0.6, 0.6), (0.055, 0.5, -0.5 * math.sqrt(0.5))]
    for t, amplitude, lateral in arrivals:
        w[int(t * FS)] += amplitude
        y[int(t * FS)] += lateral
    measured = from_impulse_response(w, FS, 0.01, lateral=y, noise=False)
    energy = sum(a * a for _, a, _ in arrivals)
    expected_lf = (0.36 + 0.125) / energy
    expected_lfc = (0.36 + 0.25 * math.sqrt(0.5)) / energy
    assert measured.lf[MIDDLE] == pytest.approx(np.full(5, expected_lf), abs=0.01)
    assert measured.lfc[MIDDLE] == pytest.approx(np.full(5, expected_lfc), abs=0.01)
    early = _paths(
        [t for t, _, _ in arrivals],
        [(1, 0, 0), (0, 1, 0), (1, -1, 0)],
        [a * a for _, a, _ in arrivals],
    )
    assert _response(early).parameters().lf == pytest.approx(np.full(8, expected_lf))


def test_lateral_response_must_be_as_long() -> None:
    with pytest.raises(ValueError, match="as long"):
        from_impulse_response(np.eye(1, 1000)[0], FS, lateral=np.zeros(500))


# -- directional decay ------------------------------------------------------------------


def test_directional_decay_follows_each_axis() -> None:
    # Front-back energy dies away in 2 s, the rest in 1 s.
    directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    length = 3.0
    slow, fast = _decaying(2.0, 0.015, 0.5, length), _decaying(1.0, 0.015, 0.5, length)
    split = np.zeros((6, 8, 300))
    for k, part in enumerate([slow, slow, fast, fast, fast, fast]):
        split[k] = 0.5 * part.reshape(8, 300, 10).sum(axis=2) * (1.0 if k < 2 else 0.5)
    late = slow + fast
    response = _response(
        _paths([0.010], [(1, 0, 0)], [1.0]),
        late,
        length=length,
        late_directions=directions.astype(float),
        late_directional=split,
        direction_step=0.01,
    )
    p = response.parameters()
    assert p.t30_x == pytest.approx(np.full(8, 2.0), rel=0.01)
    assert p.t30_y == pytest.approx(np.full(8, 1.0), rel=0.01)
    assert p.t30_z == pytest.approx(np.full(8, 1.0), rel=0.01)
    assert (p.t30 > 1.0).all() and (p.t30 < 2.0).all()


def test_directional_decay_from_first_order_ambisonics() -> None:
    rng = np.random.default_rng(3)
    n = 3 * FS
    t = np.arange(n) / FS
    noise = rng.standard_normal((n, 3))
    # X decays in 2 s, Y and Z in 1 s; W holds them all.
    envelopes = [np.exp(-3.0 * math.log(10.0) * t / t60) for t60 in (2.0, 1.0, 1.0)]
    xyz = noise * np.column_stack(envelopes)
    xyz[: FS // 100] = 0.0
    w = xyz.sum(axis=1)
    w[FS // 100] = 3.0
    p = from_impulse_response(w, FS, directional=xyz, noise=False)
    assert p.t30_x[2:] == pytest.approx(np.full(6, 2.0), rel=0.05)
    assert p.t30_y[2:] == pytest.approx(np.full(6, 1.0), rel=0.05)


# -- IACC -------------------------------------------------------------------------------


def _noise_decay(rng, n: int, t60: float = 1.0) -> np.ndarray:
    t = np.arange(n) / FS
    return rng.standard_normal(n) * np.exp(-3.0 * math.log(10.0) * t / t60)


def test_iacc_of_equal_and_delayed_ears_is_one() -> None:
    ear = _noise_decay(np.random.default_rng(1), FS)
    ear[0] = 5.0
    same = interaural_cross_correlation(np.column_stack([ear, ear]), FS)
    late = np.concatenate([np.zeros(int(0.0005 * FS)), ear])[:FS]  # 0.5 ms later
    delayed = interaural_cross_correlation(np.column_stack([ear, late]), FS)
    assert same["iacc_e"] == pytest.approx(np.ones(8)) == same["iacc_l"]
    # A little of the delayed ear's sound moves across the 80 ms mark.
    assert delayed["iacc_e"] == pytest.approx(np.ones(8), abs=0.02)
    assert delayed["iacc_l"] == pytest.approx(np.ones(8), abs=0.02)


def test_iacc_of_unrelated_ears_is_small() -> None:
    rng = np.random.default_rng(2)
    ears = np.column_stack([_noise_decay(rng, 2 * FS), _noise_decay(rng, 2 * FS)])
    result = interaural_cross_correlation(ears, FS)
    # What is left is chance, the more of it the narrower the band.
    assert (result["iacc_l"][2:] < 0.3).all() and (result["iacc_l"][5:] < 0.15).all()


def test_iacc_needs_two_ears() -> None:
    with pytest.raises(ValueError, match="two channels"):
        interaural_cross_correlation(np.zeros((100, 1)), FS)


# -- stage support ----------------------------------------------------------------------


def test_stage_support_of_known_energies() -> None:
    # 1 in the first 10 ms, 0.01 at 50 ms and 0.02 at 500 ms after the direct sound.
    energy = np.zeros((8, 1200))
    energy[:, 10], energy[:, 60], energy[:, 510] = 1.0, 0.01, 0.02
    p = from_energy(energy, DT, 0.010, stage=True)
    assert p.st_early == pytest.approx(np.full(8, -20.0))
    assert p.st_late == pytest.approx(np.full(8, 10.0 * math.log10(0.02)))
    assert np.isnan(from_energy(energy, DT, 0.010).st_early).all()


def test_stage_support_only_near_the_source() -> None:
    early = _paths([0.003, 0.050], [(1, 0, 0), (0, 0, -1)], [1.0, 0.1])
    near = _response(early)
    near.direct_delay_s = 1.0 / C
    assert near.parameters().st_early == pytest.approx(np.full(8, -10.0))
    far = _response(early)
    far.direct_delay_s = 3.0 / C
    assert np.isnan(far.parameters().st_early).all()
    # A source fed 10 ms late is still 1 m away.
    delayed = near.fed(np.zeros(8), 0.010)
    assert delayed.parameters().st_early == pytest.approx(np.full(8, -10.0))


# -- speech -----------------------------------------------------------------------------


def _transmission(m: np.ndarray) -> np.ndarray:
    snr = np.clip(10.0 * np.log10(m / (1.0 - m)), -15.0, 15.0)
    return (snr + 15.0) / 30.0


@pytest.mark.parametrize("t60", [0.3, 1.0, 2.5])
def test_stipa_of_an_exponential_decay(t60: float) -> None:
    # An exponential decay has m(F) = 1 / sqrt(1 + (2 pi F T / 13.8)^2) (Houtgast).
    t = np.arange(int(5 * t60 / DT)) * DT
    energy = np.repeat(np.exp(-6.0 * math.log(10.0) * t / t60)[None, :], 8, axis=0)
    alpha = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
    beta = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
    f = np.array(iso3382.STIPA_MODULATION_HZ)
    mti = _transmission(1.0 / np.sqrt(1.0 + (2.0 * np.pi * f * t60 / 13.8) ** 2)).mean(axis=1)
    expected = alpha @ mti - beta @ np.sqrt(mti[:-1] * mti[1:])
    stipa = speech_transmission_index(energy, DT, stipa=True)
    assert stipa == pytest.approx(expected, abs=0.002)
    assert stipa == pytest.approx(speech_transmission_index(energy, DT), abs=0.01)


def test_u50_is_c50_without_noise_and_drops_with_it() -> None:
    d50 = np.array([0.5, 0.8])
    c50 = 10.0 * np.log10(d50 / (1.0 - d50))
    assert useful_to_detrimental(d50, 60.0) == pytest.approx(c50)
    # Noise 10 dB below the speech adds a tenth of its energy to the detrimental part.
    expected = 10.0 * np.log10(d50 / (1.0 - d50 + 0.1))
    assert useful_to_detrimental(d50, [60.0, 60.0], 50.0) == pytest.approx(expected)


def test_speech_of_a_response() -> None:
    late = _decaying(1.0, 0.015, 0.5)
    response = _response(_paths([0.010], [(1, 0, 0)], [1.0]), late)
    p = response.parameters()
    quiet, noisy = response.speech(), response.speech(noise_db=60.0)
    assert quiet.u50 == pytest.approx(p.c50)
    assert (noisy.u50 < quiet.u50).all() and noisy.sti < quiet.sti
    assert quiet.stipa == pytest.approx(quiet.sti, abs=0.02)
    assert quiet.signal_db == pytest.approx(
        94.0 - 10.0 * math.log10(8) + p.g + 10 * math.log10(REFERENCE)
    )
    assert set(quiet.as_dict()) == {"sti", "stipa", "u50", "signal_db"}


# -- echo criteria ----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["speech", "music"])
def test_echo_criterion_of_one_echo(kind: str) -> None:
    # The direct sound and an echo as loud 100 ms later: the centre time jumps by half
    # of that within the window.
    exponent, window, _, (_, audible) = iso3382.ECHO_CRITERIA[kind]
    ir = np.zeros(FS // 2)
    ir[FS // 100] = 1.0
    ir[FS // 100 + FS // 10] = 1.0
    value, at = echo_criterion(ir, FS, kind)
    assert value == pytest.approx(0.05 / window, rel=0.1)
    assert value > audible
    assert 0.100 <= at <= 0.100 + window
    # Ten times quieter (-20 dB): the jump shrinks with 0.1^n.
    ir[FS // 100 + FS // 10] = 0.1
    quiet = 0.1 * 0.1**exponent / (1.0 + 0.1**exponent) / window
    assert echo_criterion(ir, FS, kind)[0] == pytest.approx(quiet, rel=0.15)


def test_no_echo_in_a_smooth_decay() -> None:
    ir = _noise_decay(np.random.default_rng(4), FS)
    ir[: FS // 100] = 0.0
    ir[FS // 100] = 3.0
    for kind, (_, _, _, (hardly, _)) in iso3382.ECHO_CRITERIA.items():
        assert echo_criterion(ir, FS, kind)[0] < hardly


def test_the_music_criterion_hears_an_echo_alike_across_its_two_octaves() -> None:
    # Dietsch and Kraak filter the response to two octaves about 1.4 kHz for music: an
    # echo 100 ms after the direct sound of a smooth decay, a 16 ms burst at 1, 1.4 or
    # 2 kHz, reaches the 50 % mark (1.8) at the same level within 0.25 dB. (The 1 and
    # 2 kHz octave filters added rise 3 dB where they meet: there it was heard 2.2 dB
    # quieter.)
    ir = _noise_decay(np.random.default_rng(4), FS)
    ir[: FS // 100] = 0.0
    ir[FS // 100] = 3.0
    t = np.arange(-0.008, 0.008, 1.0 / FS)
    audible = iso3382.ECHO_CRITERIA["music"][3][1]
    levels = []
    for f in (1000.0, 1414.0, 2000.0):
        burst = np.cos(2 * np.pi * f * t) * np.hanning(len(t))
        low, high = 0.0, 16.0
        for _ in range(24):
            middle = 0.5 * (low + high)
            echoed = ir.copy()
            echoed[FS // 100 + FS // 10 :][: len(burst)] += middle * burst
            heard = echo_criterion(echoed, FS, "music")[0] > audible
            low, high = (low, middle) if heard else (middle, high)
        levels.append(20 * math.log10(low + high))
    assert max(levels) - min(levels) < 0.25, levels


def test_echo_criterion_kinds() -> None:
    with pytest.raises(ValueError, match="unknown echo criterion"):
        echo_criterion(np.zeros(100), FS, "song")
    assert np.isnan(echo_criterion(np.eye(1, 100)[0], FS)[0])  # shorter than the window


# -- the parameters themselves ----------------------------------------------------------


def test_round_trip_and_table() -> None:
    early = _paths([0.010, 0.030], [(1, 0, 0), (0, 1, 0)], [1.0, 0.5])
    p = _response(early, _decaying(1.0, 0.015, 0.5)).parameters()
    back = RoomParameters.from_dict(p.as_dict())
    for name in ALL_PARAMETERS:
        assert np.array_equal(getattr(back, name), getattr(p, name), equal_nan=True)
    assert set(p.given()) == set(ALL_PARAMETERS) - {"iacc_e", "iacc_l", "st_early", "st_late"}
    assert "LF " in p.table() and "IACC" not in p.table()
    # Old reports, with the eight parameters alone, read back with the rest NaN.
    core = {name: p.as_dict()[name] for name in PARAMETERS}
    plain = RoomParameters.from_dict(core)
    assert plain.given() == list(PARAMETERS)
    assert plain.table().splitlines()[1].startswith("EDT (s)      ")


def test_compare_takes_lateral_fractions_both_sides_give() -> None:
    early = _paths([0.010, 0.030], [(1, 0, 0), (0, 1, 0)], [1.0, 0.5])
    simulated = _response(early, _decaying(1.0, 0.015, 0.5)).parameters()
    c = compare_parameters(simulated, simulated)
    assert "lf" in c.parameters and "lj" not in c.parameters
    assert c["lf"].within_jnd == 8
    omni = RoomParameters.from_dict({name: simulated.as_dict()[name] for name in PARAMETERS})
    assert "lf" not in compare_parameters(simulated, omni).parameters
