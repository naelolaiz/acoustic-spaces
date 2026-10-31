"""Sources: sound power per band, speech spectra, the feed's gain, equalizer and delay,
line and surface sources, and what they do to levels, impulse responses, the STI and
coverage maps."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.air import energy_attenuation_per_m
from acoustic_spaces.directivity import lambert
from acoustic_spaces.engine import coverage_map, early_paths
from acoustic_spaces.materials.bands import gain_levels, power_levels
from acoustic_spaces.parameters import (
    A_WEIGHTING_DB,
    SPEECH_SPECTRUM_DB,
    a_weighted,
    speech_transmission_index,
)
from acoustic_spaces.scene import Coverage, Source
from acoustic_spaces.scene.model import MAX_ELEMENTS

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
    # The power in all leaves the feed out: the responses carry it.
    assert speech.sound_power_db == 70.0 and _source(gain_db=6.0).sound_power_db == 94.0
    in_all = 10 * math.log10(np.sum(10 ** (np.array(given) / 10)))
    assert fed.sound_power_db == pytest.approx(in_all)
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


def _free_field(tmp_path: Path, listeners=((4.0, 0.0, 0.0),), **source) -> Scene:
    # One absorbing triangle 100 m away: free field for a listener 4 m from the source.
    (tmp_path / "far.obj").write_text("v 100 0 0\nv 100 0.1 0\nv 100 0 0.1\nf 1 2 3\n")
    return Scene.model_validate(
        {
            "geometry": {"file": str(tmp_path / "far.obj")},
            "materials": {"absorber": {"absorption": [1.0] * 8, "scattering": 0.0}},
            "default_material": "absorber",
            "sources": [{"name": "s", "position": [0, 0, 0], **source}],
            "listeners": [{"name": f"l{i}", "position": p} for i, p in enumerate(listeners)],
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
    unfed = plain.levels(_shoebox().sources[0].power_db(), a)
    np.testing.assert_allclose(level - unfed, np.add(6.0, eq), atol=0.02)


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


def test_line_and_surface_sources_are_made_of_points() -> None:
    line = _source(line_m=2.0, spacing_m=0.5)
    np.testing.assert_allclose(line.elements()[:, 1], [-0.75, -0.25, 0.25, 0.75])
    np.testing.assert_allclose(line.elements()[:, [0, 2]], 0.0)
    # Along the source's left and up axes: turned 90 degrees, the line runs along -x.
    turned = _source(line_m=2.0, spacing_m=0.5, orientation={"yaw": 90.0})
    np.testing.assert_allclose(turned.elements()[:, 0], [0.75, 0.25, -0.25, -0.75], atol=1e-12)
    surface = _source(surface_m=(1.0, 0.6), spacing_m=0.5)
    assert surface.extended and not _source().extended
    assert surface.elements().shape == (4, 3)  # 2 across, 2 up
    np.testing.assert_allclose(np.unique(surface.elements()[:, 2]), [-0.15, 0.15])
    np.testing.assert_allclose(_source().elements(), [[0.0, 0.0, 0.0]])
    with pytest.raises(ValidationError, match="not both"):
        _source(line_m=1.0, surface_m=(1.0, 1.0))
    with pytest.raises(ValidationError, match=f"more than {MAX_ELEMENTS} point sources"):
        _source(surface_m=(20.0, 20.0), spacing_m=0.5)


def test_lambert_radiation() -> None:
    balloon = lambert()
    forward, sideways = np.array([[1.0, 0, 0]]), np.array([[0.0, 1, 0]])
    np.testing.assert_allclose(balloon.energy(forward), 4.0, rtol=0.01)
    np.testing.assert_allclose(balloon.energy(sideways), 0.0, atol=0.05)
    np.testing.assert_allclose(balloon.energy(-forward), 0.0, atol=1e-12)
    np.testing.assert_allclose(balloon.energy(np.array([[0.5, 0.0, 0.75**0.5]])), 2.0, rtol=0.02)
    np.testing.assert_allclose(balloon.index(), 10 * np.log10(4), atol=0.05)


def test_line_source_falls_3_db_per_doubling(tmp_path: Path) -> None:
    scene = _free_field(tmp_path, listeners=[(2.0, 0, 0), (4.0, 0, 0)], line_m=40.0)
    result = simulate(Room.from_scene(scene))
    assert result.details["s"]["elements"] == 80
    near, far = (response.parameters().g for response in result)
    # Close to a long line its sound spreads as a cylinder, not a sphere: about 3 dB per
    # doubling of distance (3.3 dB for 40 m at 2 and 4 m), not a point source's 6 dB;
    # more in the treble, which the air takes from the farther points.
    np.testing.assert_allclose((near - far)[:5], 3.31, atol=0.02)
    # The sum of its uncorrelated points, each with an equal share of the power.
    points = scene.sources[0].elements()
    m = energy_attenuation_per_m(20.0, 50.0)
    for g, x in ((near, 2.0), (far, 4.0)):
        d = np.linalg.norm(points - [x, 0, 0], axis=1)[:, None]
        expected = 10 * np.log10(100 / len(points) * np.sum(np.exp(-d * m) / d**2, axis=0))
        np.testing.assert_allclose(g, expected, atol=1e-9)
    paths = early_paths(Room.from_scene(scene), "s", "l0")
    assert len(paths) == 80 and np.all(paths.order == 0)
    np.testing.assert_allclose(paths.energy.sum(axis=0), result[("s", "l0")].early.energy.sum(0))


def test_surface_source_radiates_forward(tmp_path: Path) -> None:
    listeners = [(1.0, 0, 0), (2.0, 0, 0), (-2.0, 0, 0)]
    scene = _free_field(tmp_path, listeners, surface_m=(8.0, 8.0), directivity="lambert")
    result = simulate(Room.from_scene(scene))
    near, far, behind = (response.parameters().g[3] for response in result)
    # A surface's points radiate by Lambert's law: the continuous square's integral,
    # which falls slowly with distance near it, and nothing behind.
    side = (np.arange(800) + 0.5) / 100 - 4.0
    y, z = np.meshgrid(side, side)
    for g, x in ((near, 1.0), (far, 2.0)):
        d2 = x**2 + y**2 + z**2
        energy = np.sum(4 * x / np.sqrt(d2) / (4 * math.pi * d2)) / side.size**2
        assert g == pytest.approx(10 * np.log10(energy * 4 * math.pi * 100), abs=0.15)
    assert near - far < 1.5
    assert np.isnan(behind) or behind < -60


def test_extended_sources_in_the_room() -> None:
    scene = _shoebox(line_m=3.0, spacing_m=1.0)
    scene.simulation.late.rays = 3000
    room = Room.from_scene(scene)
    assert room.problems() == []
    result = simulate(room)
    info = result.details[scene.sources[0].name]
    assert info["elements"] == 3 and info["rays"] == 3000  # 1000 rays per band each
    # The same room, as loud as a point source of the same power: a few rays' decays are
    # too noisy to compare (at 30,000 rays per band their T30 agree within 10 %). The
    # line's points are uncorrelated, their paths add in energy even at low frequencies.
    plain = _shoebox()
    plain.simulation.early.coherent_max_band_hz = None
    point = next(iter(simulate(Room.from_scene(plain))))
    np.testing.assert_allclose(next(iter(result)).parameters().g, point.parameters().g, atol=0.6)
    # Coverage maps take the line's points as well.
    scene.simulation.late.method = "statistical"
    scene.simulation.early.max_order = 1
    values = coverage_map(Room.from_scene(scene), settings=Coverage(spacing=2.0)).get().values
    assert np.all(np.isfinite(values["spl"]))
    long = _shoebox(line_m=40.0)
    assert any("reaches outside the room" in p for p in Room.from_scene(long).problems())
    with pytest.raises(ValueError, match="the wave solver takes point sources only"):
        wave = _shoebox(line_m=1.0)
        wave.simulation.wave.enabled = True
        simulate(Room.from_scene(wave))


def test_sources_and_listeners_left_out() -> None:
    """A source or listener with ``enabled: false`` stays in the scene but is left out
    of simulations, maps and checks; the others get what a run of all gives them."""
    scene = _shoebox()
    scene.simulation.late.method = "statistical"
    scene.simulation.early.max_order = 1
    source, listener = scene.sources[0], scene.listeners[0]
    scene.sources.append(
        source.model_copy(update={"name": "copy", "position": np.add(source.position, 0.5)})
    )
    scene.listeners.append(
        listener.model_copy(update={"name": "other", "position": np.add(listener.position, -0.5)})
    )
    whole = simulate(Room.from_scene(scene))
    together = coverage_map(Room.from_scene(scene), settings=Coverage(spacing=2.0))
    scene.sources[1] = scene.sources[1].model_copy(update={"enabled": False})
    scene.listeners[1] = scene.listeners[1].model_copy(update={"enabled": False})
    room = Room.from_scene(scene)
    part = simulate(room)
    pair = (source.name, listener.name)
    assert list(part.responses) == [pair]
    np.testing.assert_array_equal(part[pair].early.energy, whole[pair].early.energy)
    np.testing.assert_array_equal(part[pair].late, whole[pair].late)
    # Asked for by name, a source left out is simulated all the same.
    assert list(simulate(room, sources=["copy"]).responses) == [("copy", listener.name)]
    alone = coverage_map(room, settings=Coverage(spacing=2.0))
    assert list(alone.maps) == [source.name]
    np.testing.assert_array_equal(
        alone.get().values["spl"], together.get(source.name).values["spl"]
    )
    # A source left out is not checked: outside the room, it stops nothing.
    scene.sources[1] = scene.sources[1].model_copy(update={"position": (100.0, 0.0, 0.0)})
    assert Room.from_scene(scene).problems() == []
    scene.sources[1] = scene.sources[1].model_copy(update={"enabled": None})
    assert "outside the room" in "; ".join(Room.from_scene(scene).problems())
    scene.sources[0] = scene.sources[0].model_copy(update={"enabled": False})
    scene.sources[1] = scene.sources[1].model_copy(update={"enabled": False})
    with pytest.raises(ValueError, match="every source is left out"):
        simulate(Room.from_scene(scene))
