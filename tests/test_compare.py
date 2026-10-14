import json
import math

import numpy as np
import pytest

from acoustic_spaces.compare import (
    JND,
    SUMMARY_BANDS_HZ,
    compare_impulse_responses,
    compare_parameters,
)
from acoustic_spaces.parameters import (
    ALL_PARAMETERS,
    PARAMETERS,
    RoomParameters,
    from_energy,
    from_impulse_response,
    octave_midbands,
)

DT = 1e-3


def _exponential(t60: float, length: float) -> np.ndarray:
    """Energy per slot of a decay e^(-k t) from 0 s on, integrated per slot."""
    k = 6.0 * math.log(10.0) / t60
    edges = np.arange(int(round(length / DT)) + 1) * DT
    return (np.exp(-k * edges[:-1]) - np.exp(-k * edges[1:])) / k


def _parameters(t60, length: float = 3.0, reference: float | None = None) -> RoomParameters:
    """Parameters of exponential decays: one T60 for every band, or one per band."""
    t60 = np.broadcast_to(np.asarray(t60, dtype=float), (8,))
    energy = np.array([_exponential(t, length) for t in t60])
    return from_energy(energy, DT, onset_s=0.0, reference_energy=reference)


def _tones(t60: float, sample_rate: int, length: float = 2.0) -> np.ndarray:
    """An impulse response that decays exactly exponentially in every octave band: a
    tone at each mid-band frequency under one envelope, after 10 ms of silence."""
    t = np.arange(int(length * sample_rate)) / sample_rate
    tones = sum(np.cos(2 * np.pi * f * t + 0.7 * b) for b, f in enumerate(octave_midbands()))
    decay = tones * np.exp(-3.0 * math.log(10.0) * t / t60)
    return np.concatenate([np.zeros(sample_rate // 100), decay])


def test_jnds_of_iso_3382_annex_a() -> None:
    assert set(PARAMETERS) <= set(JND) <= set(ALL_PARAMETERS)
    assert JND["t30"] == JND["t20"] == JND["edt"] == (0.05, True)  # 5 % of the value
    assert JND["t30_x"] == JND["t30_y"] == JND["t30_z"] == JND["t30"]
    assert JND["lf"] == JND["lfc"] == (0.05, False)
    assert not {"lj", "iacc_e", "iacc_l", "st_early", "st_late"} & set(JND)  # none known
    assert JND["c80"] == JND["c50"] == JND["g"] == (1.0, False)  # 1 dB
    assert JND["d50"] == (0.05, False) and JND["ts"] == (0.010, False)  # 10 ms
    assert SUMMARY_BANDS_HZ[0] == 125 and SUMMARY_BANDS_HZ[-1] == 4000


def test_decays_a_tenth_longer_are_two_jnds_apart() -> None:
    k_sim, k_meas = (6.0 * math.log(10.0) / t60 for t60 in (1.1, 1.0))
    # G against the measured decay's total energy: 0 dB measured, 10 log 1.1 simulated.
    c = compare_parameters(
        _parameters(1.1, reference=1.0 / k_meas), _parameters(1.0, reference=1.0 / k_meas)
    )
    for name in ("edt", "t20", "t30"):
        p = c[name]
        np.testing.assert_allclose(p.simulated, 1.1, rtol=1e-6)
        np.testing.assert_allclose(p.difference, 0.1, rtol=1e-5)
        # 10 % of the measured value, at 5 % per JND.
        np.testing.assert_allclose(p.difference_jnd, 2.0, rtol=1e-5)
        np.testing.assert_allclose(p.jnd, 0.05, rtol=1e-6)
        assert p.missing == [None] * 8
        assert (p.compared, p.within_jnd) == (8, 0)
        assert p.mean_abs_jnd == pytest.approx(2.0, rel=1e-5)

    def c80(k: float) -> float:
        return 10.0 * math.log10(math.exp(k * 0.08) - 1.0)

    def d50(k: float) -> float:
        return 1.0 - math.exp(-k * 0.05)

    # In JNDs: 1 dB of C80, 0.05 of D50, 10 ms of Ts.
    expected = {
        "c80": c80(k_sim) - c80(k_meas),
        "d50": (d50(k_sim) - d50(k_meas)) / 0.05,
        "ts": (1.0 / k_sim - 1.0 / k_meas) / 0.010,
    }
    for name, jnds in expected.items():
        np.testing.assert_allclose(c[name].difference_jnd, jnds, atol=0.05)
    np.testing.assert_allclose(c["g"].difference_jnd, 10.0 * math.log10(1.1), atol=1e-6)
    assert c["g"].within_jnd == 8


def test_bands_within_one_jnd_and_the_mean_from_125_hz_to_4_khz() -> None:
    # 0, 0.4, 0.8, 1.2, 1.6, 2, 4 and 6 JNDs longer than measured.
    t60 = [1.0, 1.02, 1.04, 1.06, 1.08, 1.1, 1.2, 1.3]
    c = compare_parameters(_parameters(t60), _parameters(1.0))
    t30 = c["t30"]
    np.testing.assert_allclose(t30.difference_jnd, [0, 0.4, 0.8, 1.2, 1.6, 2, 4, 6], atol=1e-6)
    assert (t30.compared, t30.within_jnd) == (8, 3)
    # The 63 Hz and 8 kHz bands are left out of the mean.
    assert t30.mean_abs_jnd == pytest.approx((0.4 + 0.8 + 1.2 + 1.6 + 2.0 + 4.0) / 6, rel=1e-5)
    summary = c.summary().splitlines()
    assert "within 1 JND" in summary[0] and "mean |JNDs|" in summary[0]
    assert summary[3].split() == ["T30", "(s)", "3", "of", "8", "1.7"]
    # Beyond one JND is marked, within is not.
    row = c.table().splitlines()[9]  # under the header, EDT and T20: T30's JNDs
    cells = ["+0.0", "+0.4", "+0.8", "+1.2*", "+1.6*", "+2.0*", "+4.0*", "+6.0*"]
    assert row.split() == ["JNDs", *cells]


def test_g_needs_a_reference_on_both_sides() -> None:
    c = compare_parameters(_parameters(1.1, reference=0.1), _parameters(1.0))
    g = c["g"]
    assert np.all(np.isfinite(g.simulated)) and np.all(np.isnan(g.measured))
    assert np.all(np.isnan(g.difference)) and np.all(np.isnan(g.difference_jnd))
    assert g.missing == ["no reference for the measured G"] * 8
    assert (g.compared, g.within_jnd) == (0, 0) and math.isnan(g.mean_abs_jnd)
    assert c["t30"].compared == 8  # the rest is compared all the same
    assert "G in every band: no reference for the measured G" in c.table()
    report = c.as_dict()["parameters"]["g"]
    assert report["difference_jnd"] == [None] * 8 and report["mean_abs_jnd"] is None


def test_a_response_that_stops_early_is_missing_not_zero() -> None:
    # The measured decay stops after 6 dB: it has no parameters, so nothing compares.
    c = compare_parameters(_parameters(2.0), _parameters(2.0, length=0.2))
    for name in PARAMETERS:
        p = c[name]
        assert np.all(np.isnan(p.difference_jnd)), name
        assert (p.compared, p.within_jnd) == (0, 0), name
        assert all("the measured response decays by less than 20 dB" in r for r in p.missing)
    table = c.table()
    assert "in every band: the measured response decays by less than 20 dB" in table
    assert "+0.0" not in table
    assert "0 of 0" in c.summary()
    # After 25 dB, only T30 is missing: its fit runs down to -35 dB.
    c = compare_parameters(_parameters(2.0), _parameters(2.0, length=0.833))
    assert c["t30"].missing == ["the measured decay does not reach -35 dB for T30"] * 8
    for name in ("edt", "t20", "c50", "c80", "d50", "ts"):
        assert c[name].compared == c[name].within_jnd == 8, name


def test_the_reasons_of_both_sides_are_given() -> None:
    c = compare_parameters(_parameters(2.0, length=0.2), _parameters(2.0, length=0.833))
    assert c["t30"].missing[0] == (
        "the simulated response decays by less than 20 dB; "
        "the measured decay does not reach -35 dB for T30"
    )


def test_the_report_is_plain_json() -> None:
    c = compare_parameters(_parameters(1.1, reference=0.1), _parameters(1.0, length=0.2))
    report = json.loads(json.dumps(c.as_dict(), allow_nan=False))
    assert report["bands_hz"] == [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    assert report["summary_bands_hz"] == list(SUMMARY_BANDS_HZ)
    t30 = report["parameters"]["t30"]
    assert t30["jnd"] == 0.05 and t30["jnd_relative"] is True
    assert t30["simulated"][0] == pytest.approx(1.1) and t30["measured"] == [None] * 8
    assert t30["missing"][0] == "the measured response decays by less than 20 dB"
    assert (t30["compared"], t30["within_jnd"], t30["mean_abs_jnd"]) == (0, 0, None)


def test_parameters_need_every_octave_band() -> None:
    one_band = from_energy(_exponential(1.0, 3.0)[None, :], DT, onset_s=0.0)
    with pytest.raises(ValueError, match="the measured parameters need one value per octave"):
        compare_parameters(_parameters(1.0), one_band)


def test_impulse_responses_at_their_own_sample_rates() -> None:
    simulated, measured = _tones(1.1, 44_100), _tones(1.0, 48_000)
    c = compare_impulse_responses(
        simulated,
        44_100,
        measured[:, None],  # as read from a mono WAV file
        48_000,
        simulated_reference=0.01,
        measured_reference=0.01,
    )
    for name in ("edt", "t20", "t30"):
        np.testing.assert_allclose(c[name].difference_jnd, 2.0, atol=0.02)
    # The parameters are those the analysis of each response gives, G included.
    sim = from_impulse_response(simulated, 44_100, 0.01)
    meas = from_impulse_response(measured, 48_000, 0.01)
    for name in PARAMETERS:
        np.testing.assert_allclose(getattr(c.simulated, name), getattr(sim, name))
        np.testing.assert_allclose(c[name].difference, getattr(sim, name) - getattr(meas, name))
    # Without a reference for one of them, G is not compared.
    c = compare_impulse_responses(simulated, 44_100, measured, 48_000, simulated_reference=0.01)
    assert c["g"].missing == ["no reference for the measured G"] * 8
    assert c["t30"].compared == 8


def test_the_noise_floor_leaves_out_decay_times_with_the_reason() -> None:
    fs, rng = 48_000, np.random.default_rng(4)
    t = np.arange(2 * fs) / fs
    decay = rng.standard_normal(len(t)) * np.exp(-3.0 * math.log(10.0) * t / 1.0)
    measured = np.concatenate([np.zeros(480), decay])
    measured += 10 ** (-40.0 / 20.0) * rng.standard_normal(len(measured))
    c = compare_impulse_responses(_tones(1.0, fs), fs, measured, fs)
    # 40 dB of decay above the noise: enough for T20 (35 dB), not for T30 (45 dB).
    assert c["t20"].compared >= 6
    assert c["t30"].compared == 0
    for reason in c["t30"].missing:
        assert reason.startswith("the measured decay clears the noise by ")
        assert reason.endswith(" dB, T30 needs 45 dB")
    # Without the noise detection, the noise only stretches the decay.
    raw = compare_impulse_responses(_tones(1.0, fs), fs, measured, fs, noise=False)
    assert raw["t30"].compared == 8


def test_a_short_impulse_response_is_missing_on_its_side() -> None:
    c = compare_impulse_responses(_tones(1.0, 48_000), 48_000, _tones(2.0, 48_000, 0.15), 48_000)
    for name in PARAMETERS:
        assert c[name].compared == 0, name
        assert np.all(np.isfinite(c[name].simulated)) or name == "g"
    assert c["c80"].missing == ["the measured response decays by less than 20 dB"] * 8
