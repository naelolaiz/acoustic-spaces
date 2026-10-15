"""Sources: sound power per band, speech spectra, the feed's gain, equalizer and delay,
and what they do to levels, impulse responses, the STI and coverage maps."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.air import energy_attenuation_per_m
from acoustic_spaces.engine import coverage_map
from acoustic_spaces.materials.bands import gain_levels, power_levels
from acoustic_spaces.parameters import (
    A_WEIGHTING_DB,
    SPEECH_SPECTRUM_DB,
    a_weighted,
    speech_transmission_index,
)
from acoustic_spaces.scene import Coverage, Source

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
FS = 48_000


def _source(**fields) -> Source:
    return Source(name="s", position=(0.0, 0.0, 0.0), **fields)


def test_sound_power_per_band() -> None:
    flat = _source(level_db=94.0).power_db()
    np.testing.assert_allclose(flat, 94.0 - 10 * math.log10(8))
    speech = _source(level_db=70.0, spectrum="speech_male")
    power = speech.power_db()
    assert 10 * np.log10(np.sum(10 ** (power / 10))) == pytest.approx(70.0)
    np.testing.assert_allclose(np.diff(power), np.diff(SPEECH_SPECTRUM_DB["male"]))
    assert speech.talker == "male" and _source(spectrum="speech_female").talker == "female"
    given = [80.0, 82, 84, 86, 88, 90, 92, 94]
    fed = _source(level_db=given, gain_db=3.0, eq_db=[0, 0, 0, -6, 0, 0, 0, 0])
    np.testing.assert_allclose(fed.power_db(), np.add(given, [3, 3, 3, -3, 3, 3, 3, 3]))
    # Third octaves split an octave's power, and keep its gain.
    third = fed.power_db("third_octave")
    assert third.shape == (24,)
    np.testing.assert_allclose(power_levels(third, "octave"), fed.power_db())
    np.testing.assert_allclose(fed.feed_db("third_octave")[9:12], -3.0)
    gains = gain_levels([0.0, 0.0, -100.0] * 8, "octave")
    np.testing.assert_allclose(gains, 10 * np.log10(2 / 3))  # the mean of the power gains


def test_sources_check_their_levels() -> None:
    with pytest.raises(ValidationError, match="or 24 .third-octave bands., got 3"):
        _source(level_db=[80.0, 80.0, 80.0])
    with pytest.raises(ValidationError, match="a level per band has its own"):
        _source(level_db=[80.0] * 8, spectrum="speech_male")
    with pytest.raises(ValidationError, match="eq_db"):
        _source(eq_db=[0.0] * 7)
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _source(delay_ms=-1.0)


def test_speech_spectra_of_iec_60268_16() -> None:
    # Relative to the A-weighted level, so their A-weighted sum is 0 dB.
    for talker, levels in SPEECH_SPECTRUM_DB.items():
        assert a_weighted(levels) == pytest.approx(0.0, abs=0.1), talker
    assert a_weighted(np.zeros(8)) == pytest.approx(
        10 * np.log10(np.sum(10 ** (np.array(A_WEIGHTING_DB) / 10)))
    )


def test_sti_weighted_for_female_speech() -> None:
    perfect = np.zeros((8, 2000))
    perfect[:, 0] = 1.0
    for talker in ("male", "female"):
        assert speech_transmission_index(perfect, 1e-3, talker=talker) == pytest.approx(1.0)
    # A decay that blurs the bass and keeps the treble: female speech, weighted to the
    # treble, comes out more intelligible.
    t = np.arange(2000) * 1e-3
    t60 = np.array([3.0, 3.0, 2.0, 1.0, 0.6, 0.4, 0.3, 0.3])
    energy = np.exp(-13.8 * t[None, :] / t60[:, None])
    male = speech_transmission_index(energy, 1e-3)
    female = speech_transmission_index(energy, 1e-3, talker="female")
    assert female > male + 0.01
    with pytest.raises(ValueError, match="unknown talker"):
        speech_transmission_index(energy, 1e-3, talker="child")


def _free_field(tmp_path: Path, **source) -> Scene:
    # One absorbing triangle 100 m away: free field for a listener 4 m from the source.
    (tmp_path / "far.obj").write_text("v 100 0 0\nv 100 0.1 0\nv 100 0 0.1\nf 1 2 3\n")
    return Scene.model_validate(
        {
            "geometry": {"file": str(tmp_path / "far.obj")},
            "materials": {"absorber": {"absorption": [1.0] * 8, "scattering": 0.0}},
            "default_material": "absorber",
            "sources": [{"name": "s", "position": [0, 0, 0], **source}],
            "listeners": [{"name": "l", "position": [4, 0, 0]}],
            "simulation": {
                "ir_length_s": 0.2,
                "late": {"rays": 500},
                "early": {"diffraction": "none"},
                "seed": 1,
            },
        }
    )


def test_free_field_sound_pressure_level(tmp_path: Path) -> None:
    given = [80.0, 82, 84, 86, 88, 90, 92, 94]
    eq = [0.0, 0, 0, -6, 0, 0, 0, 0]
    scene = _free_field(tmp_path, level_db=given, gain_db=3.0, eq_db=eq)
    response = next(iter(simulate(Room.from_scene(scene))))
    parameters = response.parameters()
    # Lp = Lw - 10 log10(4 pi r^2), less the air's absorption over the 4 m.
    air = 10 * np.log10(math.e) * energy_attenuation_per_m(20.0, 50.0) * 4.0
    expected = np.add(given, 3.0) + eq - 10 * math.log10(4 * math.pi * 16) - air
    np.testing.assert_allclose(response.levels(scene.sources[0].power_db(), parameters), expected)
    # G is the room's: the feed's gain does not change it.
    plain = next(iter(simulate(Room.from_scene(_free_field(tmp_path, level_db=given)))))
    np.testing.assert_allclose(parameters.g, plain.parameters().g)
    np.testing.assert_allclose(
        parameters.g,
        10 * np.log10(100 / 16)
        - 10 * np.log10(math.e) * (energy_attenuation_per_m(20.0, 50.0) * 4.0),
    )
    ir, reference = response.impulse_response(FS), plain.impulse_response(FS)
    k = int(np.argmax(np.abs(reference)))
    assert ir[k] / reference[k] == pytest.approx(10 ** (3 / 20), rel=0.1)  # mostly treble


def _shoebox(**source) -> Scene:
    scene = Scene.from_yaml(SHOEBOX)
    scene.simulation.seed = 1
    scene.simulation.ir_length_s = 1.0
    scene.simulation.late.rays = 2000
    scene.sources[0] = scene.sources[0].model_copy(update=source)
    return scene


def test_delay_and_gain_shift_and_scale_the_response() -> None:
    plain = next(iter(simulate(Room.from_scene(_shoebox()))))
    eq = [0.0, 0, 0, 0, -6, 0, 0, 0]
    fed = next(iter(simulate(Room.from_scene(_shoebox(gain_db=6.0, eq_db=eq, delay_ms=12.5)))))
    assert fed.direct_delay_s == pytest.approx(plain.direct_delay_s + 0.0125)
    assert fed.transition_s == pytest.approx(plain.transition_s + 0.0125)
    assert fed.length_s == pytest.approx(plain.length_s + 0.0125)
    gain = 10 ** (np.add(6.0, eq) / 10)
    np.testing.assert_allclose(fed.early.energy, plain.early.energy * gain)
    np.testing.assert_allclose(fed.early.delay_s, plain.early.delay_s + 0.0125)
    np.testing.assert_allclose(fed.late.sum(axis=1), plain.late.sum(axis=1) * gain)
    a, b = plain.parameters(), fed.parameters()
    for name in ("g", "t30", "edt", "c80", "d50"):
        np.testing.assert_allclose(getattr(b, name), getattr(a, name), atol=0.02)
    np.testing.assert_allclose(b.ts, a.ts, atol=1e-3)
    ir, reference = fed.impulse_response(FS), plain.impulse_response(FS)
    lag = int(np.argmax(np.abs(ir))) - int(np.argmax(np.abs(reference)))
    assert lag == 600  # 12.5 ms at 48 kHz
    level = fed.levels(_shoebox(gain_db=6.0, eq_db=eq).sources[0].power_db(), b)
    np.testing.assert_allclose(level - plain.levels(94.0, a), np.add(6.0, eq), atol=0.02)


def test_delay_and_gain_of_the_wave_bands() -> None:
    def run(**source):
        scene = _shoebox(**source)
        scene.simulation.ir_length_s = 0.5
        scene.simulation.late.rays = 500
        scene.simulation.wave.enabled = True
        scene.simulation.wave.max_band_hz = 63
        return next(iter(simulate(Room.from_scene(scene))))

    plain, fed = run(), run(gain_db=6.0, delay_ms=10.0)
    assert fed.wave.delay_s == pytest.approx(0.01)
    a, b = plain.energy()[0], fed.energy()[0]
    assert b.sum() == pytest.approx(a.sum() * 10**0.6, rel=1e-9)
    # 10 slots later, give or take a sample of the solver's on a slot's edge.
    np.testing.assert_allclose(
        np.cumsum(b)[10 : 10 + len(a)], np.cumsum(a) * 10**0.6, atol=0.02 * b.sum()
    )
    t = np.arange(len(b)) * 1e-3
    assert (t[: len(a)] @ a) / a.sum() + 0.01 == pytest.approx((t @ b) / b.sum(), abs=1e-4)
    field, reference = fed.wave.field(FS, 24_000), plain.wave.field(FS, 24_000)
    # 480 samples later (10 ms at 48 kHz), apart from the last 10 ms of the run, which
    # the response's 0.5 s cut off: the filters ring a little round that cut.
    peak = np.abs(reference).max()
    error = np.abs(field[480:20_000] - reference[:19_520] * 10**0.3).max()
    assert error < 0.01 * peak, error / peak


def test_combined_coverage_takes_the_delays() -> None:
    def maps(delay_ms: float):
        scene = _shoebox()
        scene.simulation.late.method = "statistical"
        scene.simulation.early.max_order = 1
        second = scene.sources[0].model_copy(update={"name": "copy", "delay_ms": delay_ms})
        scene.sources.append(second)
        return coverage_map(Room.from_scene(scene), settings=Coverage(spacing=2.0))

    together, apart = maps(0.0), maps(150.0)
    single = together.get("talker").values
    same = together.get().values
    # Two alike sources together: 3 dB louder, as clear as one.
    np.testing.assert_allclose(same["spl"], single["spl"] + 10 * math.log10(2), atol=0.01)
    np.testing.assert_allclose(same["c80"], single["c80"], atol=0.01)
    # The copy 150 ms late arrives as late sound: less clear, less intelligible.
    late = apart.get().values
    np.testing.assert_allclose(apart.get("copy").values["c80"], single["c80"], atol=0.01)
    assert np.all(late["c80"] < single["c80"] - 2.0)
    assert np.all(late["sti"] < same["sti"])


def test_coverage_takes_the_speech_spectrum() -> None:
    scene = _shoebox(level_db=72.0, spectrum="speech_female")
    scene.simulation.late.method = "statistical"
    scene.simulation.early.max_order = 1
    result = coverage_map(Room.from_scene(scene), settings=Coverage(spacing=2.0))
    values = result.get().values
    power = scene.sources[0].power_db()
    reference = 10 * math.log10(1 / (4 * math.pi * 100))
    expected = np.broadcast_to(power + reference, values["spl"].shape)
    np.testing.assert_allclose(values["spl"] - values["g"], expected, atol=0.01)
