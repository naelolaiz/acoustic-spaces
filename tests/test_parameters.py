import math

import numpy as np
import pytest

from acoustic_spaces.parameters import (
    CURVATURE_LIMIT,
    NON_LINEARITY_LIMIT,
    PARAMETERS,
    STI_MODULATION_HZ,
    decay_curve,
    from_energy,
    from_impulse_response,
    modulation_transfer,
    speech_transmission_index,
)
from acoustic_spaces.synthesis import REFERENCE_ENERGY_10M

DT = 1e-3


def _exponential(t60: float, length: float, onset: float = 0.0) -> np.ndarray:
    """Energy per slot of a decay e^(-k t) starting at ``onset``, integrated per slot."""
    k = 6.0 * math.log(10.0) / t60
    edges = np.maximum(np.arange(int(round(length / DT)) + 1) * DT - onset, 0.0)
    return (np.exp(-k * edges[:-1]) - np.exp(-k * edges[1:])) / k


@pytest.mark.parametrize("t60", [0.4, 1.0, 2.5])
def test_decay_times_of_an_exponential(t60: float) -> None:
    # Truncated at 2 s: the long decay needs the tail estimate to come out right.
    energy = np.tile(_exponential(t60, 2.0), (8, 1))
    p = from_energy(energy, DT, onset_s=0.0)
    for value in (p.edt, p.t20, p.t30):
        np.testing.assert_allclose(value, t60, rtol=0.02)


def test_energy_ratios_of_an_exponential() -> None:
    t60, onset = 1.0, 0.0105
    k = 6.0 * math.log(10.0) / t60
    energy = np.tile(_exponential(t60, 3.0, onset), (8, 1))
    p = from_energy(energy, DT, onset_s=onset)
    early = 1.0 - math.exp(-k * 0.08)
    np.testing.assert_allclose(p.c80, 10 * math.log10(early / (1 - early)), atol=0.05)
    np.testing.assert_allclose(p.d50, 1.0 - math.exp(-k * 0.05), atol=0.005)
    np.testing.assert_allclose(p.ts, 1.0 / k, rtol=0.01)


def test_strength_against_free_field() -> None:
    reference = 1.0 / (4 * math.pi * 100)
    energy = np.zeros((8, 100))
    energy[:, 10] = reference * 10  # ten times the free field energy at 10 m
    p = from_energy(energy, DT, onset_s=0.0101, reference_energy=reference)
    np.testing.assert_allclose(p.g, 10.0, atol=1e-9)
    assert np.all(np.isnan(p.t30))  # a single impulse has no decay to fit


def test_decay_curve_starts_at_zero_and_falls() -> None:
    curve = decay_curve(_exponential(1.0, 0.5), DT)
    assert curve[0] == pytest.approx(0.0)
    assert np.all(np.diff(curve) < 0)
    assert curve[300] == pytest.approx(-18.0, abs=0.2)


def test_from_impulse_response_of_decaying_noise() -> None:
    fs, t60 = 48_000, 0.8
    rng = np.random.default_rng(3)
    t = np.arange(int(1.5 * fs)) / fs
    ir = rng.standard_normal(len(t)) * np.exp(-3.0 * math.log(10.0) * t / t60)
    ir = np.concatenate([np.zeros(480), ir])
    p = from_impulse_response(ir, fs, REFERENCE_ENERGY_10M)
    np.testing.assert_allclose(p.t30[2:], t60, rtol=0.1)
    assert np.all(np.isfinite(p.g))


def test_tail_estimate_ignores_a_late_spike() -> None:
    # With few rays, one strong ray can land in the last few milliseconds of a response
    # that stops long before it has decayed. That must not read as a decay that stopped.
    # It falls by 26 dB here (one that stops above -20 dB has no parameters at all).
    t60, length = 2.5, 1.1
    k = 6.0 * math.log(10.0) / t60
    energy = _exponential(t60, length)
    energy[1090] += 0.03 * energy.sum()
    p = from_energy(energy[None, :], DT, onset_s=0.0, reference_energy=1.0 / k)
    assert abs(p.g[0]) < 0.3  # total energy, against the untruncated decay's 1 / k
    # Centre time, against that of the same energies followed by the decay's exact tail.
    tail = math.exp(-k * length) / k
    moment = (np.arange(len(energy)) + 0.5) * DT @ energy + tail * (length + 1.0 / k)
    assert p.ts[0] == pytest.approx(moment / (energy.sum() + tail), rel=0.1)


def test_a_response_that_stops_early_has_no_parameters() -> None:
    # A 2 s decay cut off after 6 dB used to read EDT 1.1 s and G 1.1 dB low; cut off
    # after 25 dB, everything but T30 can be read.
    k = 6.0 * math.log(10.0) / 2.0
    short = from_energy(_exponential(2.0, 0.2)[None, :], DT, onset_s=0.0, reference_energy=1 / k)
    for name in PARAMETERS:
        assert np.isnan(getattr(short, name)[0]), name
    longer = from_energy(_exponential(2.0, 0.833)[None, :], DT, 0.0, reference_energy=1 / k)
    assert np.isnan(longer.t30[0])
    assert longer.edt[0] == pytest.approx(2.0, rel=0.01) and abs(longer.g[0]) < 0.01
    for name in ("t20", "c50", "c80", "d50", "ts"):
        assert np.isfinite(getattr(longer, name)[0]), name


def test_the_silent_end_of_a_response_is_no_decay() -> None:
    """A response that ends in silence: its decay curve drops to nothing after the last
    sound, the end of the data and not a decay, so no decay time is fitted through that
    drop. Outdoors one late reflection and then silence used to read EDT 7.5 s, and with
    a few more steps T20 and T30 as long."""
    energy = np.zeros((1, 3500))
    energy[0, 10] = 1.0  # the direct sound
    energy[0, 1510] = 0.5  # one reflection 1.5 s later, then nothing
    p = from_energy(energy, DT, onset_s=0.0101, reference_energy=1.0, open_air=True)
    assert np.isnan(p.edt[0]) and np.isnan(p.t20[0]) and np.isnan(p.t30[0])
    # Nothing is left to a tail: the energies are those the response holds.
    assert p.c80[0] == pytest.approx(10 * math.log10(1.0 / 0.5))
    assert p.g[0] == pytest.approx(10 * math.log10(1.5))
    # Steps down to -30 dB, then silence: T20 is read from the steps (with the
    # non-linearity that shows they are steps), T30 needs -35 dB and is not there.
    energy = np.zeros((1, 3500))
    for slot, share in ((10, 1.0), (400, 0.3), (900, 0.05), (1500, 0.002)):
        energy[0, slot] = share
    p = from_energy(energy, DT, onset_s=0.0101, reference_energy=1.0)
    assert np.isfinite(p.t20[0]) and np.isnan(p.t30[0])
    assert p.xi_t20[0] > NON_LINEARITY_LIMIT and np.isnan(p.xi_t30[0])
    assert np.isnan(p.curvature[0])
    # A decay that ends in silence after falling far enough is read as before.
    decay = _exponential(1.0, 1.2)
    decay[1100:] = 0.0  # -66 dB on, nothing
    p = from_energy(decay[None, :], DT, onset_s=0.0)
    for value in (p.edt, p.t20, p.t30):
        np.testing.assert_allclose(value, 1.0, rtol=0.02)


def test_the_decay_lines_are_checked_as_iso_3382_2_says() -> None:
    """ξ = 1000 (1 - r²) of the line each of T20 and T30 is read from, and the curvature
    100 (T30 / T20 - 1): about nothing for a straight decay, past ISO 3382-2's 10 ‰
    and 10 % for one of two slopes (15 dB at 1 s, then 3 s)."""
    straight = from_energy(np.tile(_exponential(1.0, 2.0), (8, 1)), DT, onset_s=0.0)
    assert np.all(straight.xi_t20 < 0.1) and np.all(straight.xi_t30 < 0.1)
    assert np.all(np.abs(straight.curvature) < 2.0)
    t = np.arange(6001) * DT
    db = np.where(t < 0.25, -60.0 * t, -15.0 - 20.0 * (t - 0.25))
    remaining = 10.0 ** (db / 10.0)
    two = from_energy((remaining[:-1] - remaining[1:])[None, :], DT, onset_s=0.0)
    assert two.edt[0] == pytest.approx(1.0, rel=0.05)
    assert two.xi_t20[0] > NON_LINEARITY_LIMIT and two.curvature[0] > CURVATURE_LIMIT
    assert two.t30[0] > two.t20[0] > 1.0
    assert "ξ_T20 (‰)" in two.table() and "Curvature (%)" in two.table()
    assert set(two.as_dict()) >= {"xi_t20", "xi_t30", "curvature"}
    # Measured: a check goes with its decay time where the noise leaves no room for it.
    noisy = from_impulse_response(_noisy_decay(1.0, -40.0, 2.0), 48_000)
    assert np.all(np.isnan(noisy.xi_t30)) and np.all(np.isnan(noisy.curvature))
    assert np.all(np.isfinite(noisy.xi_t20[2:]))


def test_octave_filters_meet_their_band_edges() -> None:
    from acoustic_spaces.parameters import OCTAVE_RATIO, octave_filter_gains, octave_midbands

    fs, n_fft = 48_000, 1 << 18
    gains = octave_filter_gains(n_fft, fs)
    f = np.fft.rfftfreq(n_fft, 1 / fs)

    def gain_db(band: int, frequency: float) -> float:
        return 20 * math.log10(np.interp(frequency, f, gains[band]))

    for band, fm in enumerate(octave_midbands()[:-1]):
        assert gain_db(band, fm) == pytest.approx(0.0, abs=0.01)
        assert gain_db(band, fm * OCTAVE_RATIO**0.5) == pytest.approx(-3.01, abs=0.05)
        assert gain_db(band, fm / OCTAVE_RATIO**0.5) == pytest.approx(-3.01, abs=0.05)
        assert gain_db(band, fm * OCTAVE_RATIO) < -18.0
        assert gain_db(band, fm / OCTAVE_RATIO) < -18.0


def _noisy_decay(t60: float, noise_db: float, length: float, seed: int = 4) -> np.ndarray:
    fs = 48_000
    rng = np.random.default_rng(seed)
    t = np.arange(int(length * fs)) / fs
    decay = rng.standard_normal(len(t)) * np.exp(-3.0 * math.log(10.0) * t / t60)
    ir = np.concatenate([np.zeros(480), decay])
    return ir + 10 ** (noise_db / 20) * rng.standard_normal(len(ir))


def test_noise_floor_is_found_and_removed() -> None:
    from acoustic_spaces.parameters import band_decay

    t60, noise_db = 0.8, -55.0
    ir = _noisy_decay(t60, noise_db, 2.0)
    decay = band_decay(ir, 48_000)
    # The decay meets the noise where it has fallen by noise_db (the same in every band).
    expected = -noise_db / 60.0 * t60
    np.testing.assert_allclose(decay.crosspoint / 48_000, expected, rtol=0.15)
    np.testing.assert_allclose(decay.decay_range_db, -noise_db, atol=6.0)
    p = from_impulse_response(ir, 48_000)
    np.testing.assert_allclose(p.t30[2:], t60, rtol=0.05)
    raw = from_impulse_response(ir, 48_000, noise=False)
    assert np.all(raw.t30[2:] > p.t30[2:])  # the noise floor stretches the decay


def test_decay_times_need_range_above_the_noise() -> None:
    p = from_impulse_response(_noisy_decay(1.0, -40.0, 2.0), 48_000)
    assert np.all(np.isnan(p.t30))  # needs 45 dB
    assert np.all(np.isfinite(p.t20[2:]))  # needs 35 dB
    np.testing.assert_allclose(p.t20[2:], 1.0, rtol=0.1)


def test_a_clean_decay_has_no_noise_floor() -> None:
    from acoustic_spaces.parameters import noise_crosspoint

    e = _exponential(1.0, 1.0)
    assert noise_crosspoint(e, 1 / DT) == (len(e), 0.0, math.inf)
    assert noise_crosspoint(np.zeros(100), 1 / DT) == (100, 0.0, math.inf)


def test_trailing_silence_is_no_noise_floor() -> None:
    from acoustic_spaces.parameters import band_decay

    # Zero padding, or the zeros that removing latency leaves at the end, must not hide
    # the noise floor before them: with half a second of them, T30 read 1.3 to 1.7 s.
    ir = _noisy_decay(1.0, -45.0, 1.5)
    plain = from_impulse_response(ir, 48_000)
    padded = from_impulse_response(np.concatenate([ir, np.zeros(24_000)]), 48_000)
    for name in PARAMETERS:
        np.testing.assert_allclose(getattr(padded, name), getattr(plain, name), rtol=1e-9)
    # A lone impulse rings in the band filters 190 dB and more below its peak, where a
    # noise floor used to be found: that is no noise.
    impulse = np.zeros(48_000)
    impulse[480] = 0.1
    decay = band_decay(impulse, 48_000)
    assert np.all(decay.decay_range_db == math.inf) and np.all(decay.crosspoint == 47_520)


def test_impulse_responses_of_any_length_and_one_column() -> None:
    from acoustic_spaces.parameters import onset

    # An empty response has no parameters, and one read from a mono WAV file is
    # (samples, 1).
    assert onset(np.zeros(0)) == 0
    empty = from_impulse_response(np.zeros(0), 48_000, REFERENCE_ENERGY_10M)
    for name in PARAMETERS:
        assert np.all(np.isnan(getattr(empty, name)))
    ir = _noisy_decay(0.8, -55.0, 1.0)
    column = from_impulse_response(ir[:, None], 48_000, REFERENCE_ENERGY_10M)
    flat = from_impulse_response(ir, 48_000, REFERENCE_ENERGY_10M)
    for name in PARAMETERS:
        np.testing.assert_array_equal(getattr(column, name), getattr(flat, name))
    with pytest.raises(ValueError, match="one channel"):
        from_impulse_response(np.zeros((100, 2)), 48_000)


def test_free_field_reference_moves_to_ten_metres() -> None:
    from acoustic_spaces.parameters import band_energy_share, free_field_reference

    fs, distance = 48_000, 2.0
    ir = np.zeros(fs)
    ir[4800] = 1.0 / distance  # the direct sound at 2 m
    reference = free_field_reference(ir, fs, distance)
    np.testing.assert_allclose(reference, REFERENCE_ENERGY_10M * band_energy_share(fs), rtol=0.02)
    # A response of that source at 10 m reads G = 0 in every band.
    ir10 = np.zeros(fs)
    ir10[4800] = 0.1
    p = from_impulse_response(ir10, fs, reference)
    np.testing.assert_allclose(p.g, 0.0, atol=0.1)


@pytest.mark.parametrize("filters", ["iec", "crossover"])
@pytest.mark.parametrize("at_s", [0.0, 0.0005, 0.0029, 0.1])
def test_free_field_reference_of_a_response_that_starts_with_its_sound(
    filters: str, at_s: float
) -> None:
    # The filters' ringing before the first sample counts: an impulse at sample 0 kept
    # 0.50 of the energy at 63 Hz to 0.71 at 8 kHz (G 3 dB too high).
    from acoustic_spaces.parameters import band_energy_share, free_field_reference

    fs = 48_000
    ir = np.zeros(fs // 4)
    ir[round(at_s * fs)] = 0.1  # the direct sound at 10 m
    reference = free_field_reference(ir, fs, 10.0, filters=filters)
    expected = REFERENCE_ENERGY_10M * band_energy_share(fs, filters)
    np.testing.assert_allclose(reference, expected, rtol=1e-3)


@pytest.mark.parametrize("filters", ["iec", "crossover"])
def test_strength_from_a_broadband_reference(filters: str) -> None:
    ir = np.zeros(48_000)
    ir[480] = 0.1  # the free-field direct sound at 10 m
    p = from_impulse_response(ir, 48_000, REFERENCE_ENERGY_10M, filters=filters)
    np.testing.assert_allclose(p.g, 0.0, atol=0.1)


@pytest.mark.parametrize("bands", [1, 24])
def test_a_reference_for_other_bands_is_refused(bands: int) -> None:
    # 24 third-octave references were cut to their first 8 and taken for octaves; one
    # in a list raised an IndexError at the second band.
    ir = np.zeros(48_000)
    ir[480] = 0.1
    with pytest.raises(ValueError, match=f"one per octave band \\(8\\), not {bands}"):
        from_impulse_response(ir, 48_000, [REFERENCE_ENERGY_10M] * bands)


def test_modulation_transfer_of_an_exponential() -> None:
    t60 = 1.2
    m = modulation_transfer(_exponential(t60, 6.0), DT)
    f = np.array(STI_MODULATION_HZ)
    expected = 1.0 / np.sqrt(1.0 + (2.0 * math.pi * f * t60 / (6.0 * math.log(10.0))) ** 2)
    np.testing.assert_allclose(m[0], expected, atol=2e-3)


@pytest.mark.parametrize("t60, sti", [(0.5, 0.74), (1.0, 0.59), (2.0, 0.44)])
def test_sti_of_reverberation_alone(t60: float, sti: float) -> None:
    # Published values for an exponential decay without noise (Houtgast and Steeneken).
    energy = np.tile(_exponential(t60, 8.0), (8, 1))
    assert speech_transmission_index(energy, DT) == pytest.approx(sti, abs=0.01)


def test_sti_falls_with_background_noise() -> None:
    energy = np.tile(_exponential(0.1, 1.0), (8, 1))
    speech = np.full(8, 60.0)
    quiet = speech_transmission_index(energy, DT, signal_db=speech)
    # Speech at 60 dB is well above the reception threshold except at 125 Hz.
    assert quiet == pytest.approx(speech_transmission_index(energy, DT), abs=0.03)
    # Noise as loud as the speech halves the modulation: 0 dB apparent SNR, 0.5 MTI.
    noisy = speech_transmission_index(energy, DT, signal_db=speech, noise_db=60.0)
    assert noisy == pytest.approx(0.5, abs=0.04)
    assert speech_transmission_index(energy, DT, signal_db=speech, noise_db=90.0) < 0.05


def test_sti_checks_its_arguments() -> None:
    energy = np.ones((8, 10))
    with pytest.raises(ValueError, match="needs signal_db"):
        speech_transmission_index(energy, DT, noise_db=40.0)
    with pytest.raises(ValueError, match="one per octave band"):
        speech_transmission_index(energy, DT, signal_db=[60.0, 60.0])
    with pytest.raises(ValueError, match="octave bands"):
        speech_transmission_index(np.ones((3, 10)), DT)
    # The seven STI bands work as well as the eight octave bands.
    assert speech_transmission_index(energy[1:], DT) == speech_transmission_index(energy, DT)
