"""Turbulence outdoors: coherence of paths, moments of the eddies, shadows, flicker."""

import dataclasses
import math

import numpy as np
import pytest
from pydantic import ValidationError
from test_refraction import field

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine.coherent import total_energy
from acoustic_spaces.engine.coverage import coverage_map
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.refraction import Profile, node_heights
from acoustic_spaces.engine.turbulence import (
    GROUND_K_RHO,
    GROUND_POWER,
    Eddies,
    Gusts,
    Wander,
    ground_share,
    von_karman,
)
from acoustic_spaces.render import auralize, flickers, gusts_for
from acoustic_spaces.scene import Atmosphere, Coverage, Turbulence
from acoustic_spaces.scene.model import TURBULENCE_PRESETS

C = 343.2
#: Kolmogorov's constant: G(rho) = 2.91 Cn² rho^(5/3) well inside the outer scale.
KOLMOGOROV = 2.914


def eddies(cn2: float = 1e-5, scale: float = 10.0) -> Eddies:
    return Eddies(cn2, scale, C)


def in_turbulence(preset="strong", atmosphere=None, **kwargs) -> Scene:
    atmosphere = dict(atmosphere or {})
    if preset is not None:
        atmosphere["turbulence"] = preset
    return field(atmosphere, **kwargs)


# -- the scene -----------------------------------------------------------------------


def test_presets_and_their_fields() -> None:
    weak, moderate, strong = (Turbulence.model_validate(p) for p in TURBULENCE_PRESETS)
    assert weak.wind_structure < moderate.wind_structure < strong.wind_structure
    assert moderate.temperature_structure == TURBULENCE_PRESETS["moderate"]["temperature_structure"]
    assert moderate.outer_scale_m == 10.0 and not moderate.calm
    mine = Turbulence.model_validate({"preset": "weak", "wind_structure": 0.2})
    assert mine.wind_structure == 0.2 and mine.temperature_structure == 0.001
    assert Turbulence().calm
    atmosphere = Atmosphere.model_validate({"turbulence": "moderate"})
    assert atmosphere.still and atmosphere.turbulence == moderate
    with pytest.raises(ValidationError, match="did you mean strong"):
        Turbulence.model_validate({"preset": "stronk"})
    with pytest.raises(ValidationError):
        Turbulence.model_validate({"wind_structure": -1})
    # Weaker with height: the strength holds at a height and falls by a power.
    assert moderate.height_m is None and moderate.falloff == pytest.approx(2.0 / 3.0)
    tall = Turbulence.model_validate({"preset": "moderate", "height_m": 2, "falloff": 1.2})
    assert tall.height_m == 2.0 and tall.falloff == 1.2 and tall.wind_structure == 0.12
    for bad in ({"height_m": 0}, {"height_m": 200}, {"falloff": -0.1}, {"falloff": 3}):
        with pytest.raises(ValidationError):
            Turbulence.model_validate({"preset": "weak", **bad})


@pytest.mark.parametrize(
    "atmosphere",
    [
        {"turbulence": "weak"},
        {"preset": "windy", "turbulence": {"preset": "strong", "outer_scale_m": 4}},
        {"turbulence": {"temperature_structure": 0.02, "wind_structure": 0.0}},
        {"turbulence": {"preset": "moderate", "height_m": 2.0, "falloff": 1.0}},
    ],
)
def test_turbulence_is_written_as_it_reads(atmosphere) -> None:
    scene = field(atmosphere)
    again = Scene.model_validate(scene.to_data())
    assert again.atmosphere == scene.atmosphere


# -- coherence -----------------------------------------------------------------------


def test_the_structure_function() -> None:
    rho = np.array([1e-3, 1e-2, 0.1])
    kolmogorov = KOLMOGOROV * 2e-6 * rho ** (5 / 3)
    np.testing.assert_allclose(eddies(2e-6, 1e6).structure(rho) / kolmogorov, 1, rtol=0.01)
    # Nearer the outer scale it falls below, as 1 - 0.805 (rho / L)^(1/3) to begin with.
    x = rho / 10.0
    ratio = eddies(2e-6, 10.0).structure(rho) / kolmogorov
    np.testing.assert_allclose(ratio, 1 - 0.805 * x ** (1 / 3), rtol=0.005)
    # Far apart the eddies of the two waves are independent: G levels off.
    far = eddies(2e-6, 10.0).structure(np.array([200.0, 600.0]))
    assert far[1] == pytest.approx(far[0], rel=1e-4)
    assert von_karman(np.array([0.0]))[0] == 0.0
    assert np.all(np.diff(von_karman(np.geomspace(1e-5, 50, 200))) >= 0)


def test_two_parallel_paths_drift_apart_as_the_markov_approximation_has_it() -> None:
    e = eddies()
    length, apart = 120.0, 0.7
    points = [
        np.array([[0.0, 0.0, 2.0], [length, 0.0, 2.0]]),
        np.array([[0.0, apart, 2.0], [length, apart, 2.0]]),
    ]
    variance = e.variances(points)
    assert variance[0, 0] == 0.0
    assert variance[0, 1] * C**2 == pytest.approx(length * float(e.structure(apart)), rel=1e-9)


def test_the_direct_sound_and_its_reflection_off_the_ground() -> None:
    # Well within the outer scale: D = (3/8) 2.91 k² Cn² R rho^(5/3), rho = 2 zs zr /
    # (zs + zr).
    e = eddies(1e-5, 1e6)
    zs, zr, r = 2.0, 1.5, 200.0
    points = [
        np.array([[0.0, 0.0, zs], [r, 0.0, zr]]),
        np.array([[0.0, 0.0, zs], [r * zs / (zs + zr), 0.0, 0.0], [r, 0.0, zr]]),
    ]
    rho = 2 * zs * zr / (zs + zr)
    expected = 3 / 8 * KOLMOGOROV * 1e-5 * r * rho ** (5 / 3) / C**2
    assert e.variances(points)[0, 1] == pytest.approx(expected, rel=0.03)


def test_the_ground_share() -> None:
    assert ground_share(1e6) == pytest.approx(1.0, abs=1e-6)
    assert ground_share(GROUND_K_RHO) == pytest.approx(0.5)
    small = 0.01 * GROUND_K_RHO
    assert ground_share(small) == pytest.approx(0.01**GROUND_POWER, rel=1e-3)


def test_a_wander_slices_stacks_and_weakens_ground_pairs() -> None:
    variance = np.array([[0.0, 4.0, 9.0], [4.0, 0.0, 1.0], [9.0, 1.0, 0.0]]) * 1e-9
    apart = np.zeros((3, 3))
    apart[0, 1] = apart[1, 0] = 0.5
    wander = Wander(variance, apart, C, 3.0)
    f = np.array([100.0, 4000.0])
    structure = wander.structure(f)
    k = 2 * math.pi * f / C
    np.testing.assert_allclose(
        structure[0, 1], (2 * math.pi * f) ** 2 * 4e-9 * ground_share(0.5 * k)
    )
    np.testing.assert_allclose(structure[0, 2], (2 * math.pi * f) ** 2 * 9e-9)
    np.testing.assert_allclose(np.diagonal(wander.coherence(f)), 1.0)
    part = wander[[2, 0]]
    assert part.variance[0, 1] == pytest.approx(9e-9) and part.correlation_s == 3.0
    both = Wander.stacked([wander, None, part], [3, 1, 2])
    assert both.variance.shape == (6, 6) and both.variance[0, 3:].sum() == 0.0
    assert both.variance[4, 5] == pytest.approx(9e-9)
    assert Wander.stacked([None, None], [1, 2]) is None


def test_moments_delay_the_paths_by_their_variances() -> None:
    corners = [
        np.array([[0.0, 0.0, 2.0], [80.0, 0.0, 1.5]]),
        np.array([[0.0, 0.0, 2.0], [45.7, 0.0, 0.0], [80.0, 0.0, 1.5]]),
        np.array([[0.0, 0.0, 2.0], [40.0, 30.0, 4.0], [80.0, 0.0, 1.5]]),
    ]
    variance = eddies().variances(corners)
    apart = np.zeros((3, 3))
    apart[0, 1] = apart[1, 0] = 0.85
    wander = Wander(variance, apart, C)
    centres = np.array([125.0, 1000.0])
    roots = wander.roots(centres, np.zeros(3, dtype=np.int64))
    rng = np.random.default_rng(1)
    delays = np.array([Wander.delays(roots, [rng.standard_normal(3)], 3) for _ in range(20000)])
    expected = wander.structure(centres) / (2 * math.pi * centres) ** 2
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        spread = np.var(delays[:, i] - delays[:, j], axis=0)
        np.testing.assert_allclose(spread, expected[i, j], rtol=0.05)


# -- in the engine -------------------------------------------------------------------


def test_no_eddies_change_nothing() -> None:
    listeners = ((60.0, 0.0, 1.5), (-90.0, 5.0, 1.5))
    for atmosphere in (None, {"preset": "windy"}):
        plain = simulate(Room.from_scene(in_turbulence(None, atmosphere, listeners=listeners)))
        calm = {"temperature_structure": 0.0, "wind_structure": 0.0}
        still = simulate(Room.from_scene(in_turbulence(calm, atmosphere, listeners=listeners)))
        for key in plain.responses:
            a, b = plain[key], still[key]
            assert b.early.wander is None
            assert np.array_equal(a.early.delay_s, b.early.delay_s)
            assert np.array_equal(a.early.energy, b.early.energy)
            assert np.array_equal(a.early.spectrum, b.early.spectrum)
            assert np.array_equal(a.late, b.late)
            assert np.array_equal(a.impulse_response(8000), b.impulse_response(8000))
    assert Eddies.of(TraceModel.from_room(Room.from_scene(field())), C) is None


def test_eddies_fill_the_ground_dip() -> None:
    listeners = ((150.0, 0.0, 1.5),)
    still = simulate(Room.from_scene(in_turbulence(None, listeners=listeners)))["s", "l0"]
    gusty = simulate(Room.from_scene(in_turbulence("strong", listeners=listeners)))["s", "l0"]
    assert np.array_equal(still.early.delay_s, gusty.early.delay_s)
    wander = gusty.early.wander
    assert wander is not None and wander.apart[0, 1] > 0.0
    plain = 10 * np.log10(total_energy(still.early))
    blurred = 10 * np.log10(total_energy(gusty.early))
    apart = 10 * np.log10(gusty.early.energy.sum(axis=0))
    # Low bands keep their interference, the dip fills in towards the paths added in energy.
    assert blurred[0] == pytest.approx(plain[0], abs=0.05)
    dip = int(np.argmin(plain - apart))
    assert plain[dip] < blurred[dip] <= apart[dip] + 0.5
    assert np.all(np.abs(blurred - apart) <= np.abs(plain - apart) + 1e-6)
    # The echogram agrees with it.
    level = 10 * np.log10(gusty.energy().sum(axis=1))
    assert level[dip] > 10 * np.log10(still.energy().sum(axis=1))[dip]


def test_an_impulse_response_is_a_moment_of_the_eddies() -> None:
    gusty = simulate(Room.from_scene(in_turbulence("strong", listeners=((150.0, 0.0, 1.5),))))
    response = gusty["s", "l0"]
    moment = response.moment(4)
    assert moment.wander is None and moment.lag.shape == response.early.energy.shape
    assert np.all(moment.lag[:, -1] == 0.0)
    assert not np.array_equal(moment.delay_s, response.early.delay_s)
    assert np.array_equal(response.moment(4).delay_s, moment.delay_s)
    assert np.array_equal(response.impulse_response(8000), response.impulse_response(8000))
    assert not np.array_equal(
        response.impulse_response(8000, seed=2), response.impulse_response(8000)
    )
    # On average over the moments, the echogram's energy.
    mean = np.mean([total_energy(response.moment(s)) for s in range(300)], axis=0)
    np.testing.assert_allclose(
        10 * np.log10(mean), 10 * np.log10(total_energy(response.early)), atol=0.4
    )


def test_no_moment_has_a_path_before_its_emission(monkeypatch) -> None:
    response = simulate(Room.from_scene(in_turbulence("strong", listeners=((150.0, 0.0, 1.5),))))[
        "s", "l0"
    ]
    early = response.early
    # Eddies that would hold every path back by more than it takes (no draw comes near).
    held = float(early.delay_s.max()) + 1.0
    bands = early.energy.shape[1]
    monkeypatch.setattr(
        Wander, "delays", staticmethod(lambda roots, draws, n: np.full((n, bands), -held))
    )
    moment = response.moment(4)
    assert np.all(moment.delay_s == 0.0) and np.all(moment.lag == 0.0)


# -- weaker with height ----------------------------------------------------------------


def test_the_eddies_strength_falls_with_height_from_the_outer_scale_up() -> None:
    same = eddies()
    assert np.array_equal(same.strength([0.0, 1.0, 50.0]), [1.0, 1.0, 1.0])
    tall = dataclasses.replace(same, height_m=2.0, falloff=2.0 / 3.0)
    z = np.array([0.0, 0.05, 0.5, 2.0, 10.0, 16.0, 160.0])
    # No nearer the ground than the outer scale (10 m).
    expected = (np.maximum(z, 10.0) / 2.0) ** (-2.0 / 3.0)
    np.testing.assert_allclose(tall.strength(z), expected)
    assert tall.strength(0.0) == tall.strength(2.0) == tall.strength(10.0) < 1.0
    assert tall.strength(16.0) == pytest.approx(0.25)
    assert tall.strength(160.0) == pytest.approx(0.25 * 10.0 ** (-2.0 / 3.0))
    steep = dataclasses.replace(tall, falloff=4.0 / 3.0)
    assert steep.strength(16.0) == pytest.approx(1.0 / 16.0)
    # Given at the outer scale or above, the strength holds down to the ground.
    aloft = dataclasses.replace(same, height_m=10.0)
    np.testing.assert_allclose(aloft.strength([0.0, 1.0, 10.0]), 1.0)
    assert aloft.strength(80.0) == pytest.approx(0.25)
    # The scene's fields reach the engine, with the ground's height.
    scene = in_turbulence({"preset": "strong", "height_m": 3.0, "falloff": 1.0})
    found = Eddies.of(TraceModel.from_room(Room.from_scene(scene)), C)
    assert found.height_m == 3.0 and found.falloff == 1.0 and found.ground == 0.0
    assert found.strength(6.0) == found.strength(10.0) == pytest.approx(0.3)
    assert found.strength(30.0) == pytest.approx(0.1)


def test_paths_near_the_ground_take_the_eddies_at_the_outer_scale_and_aloft_weaker() -> None:
    """Two paths 1 to 2 m up: with the strength given 10 m up (the outer scale) they
    part exactly as with eddies the same at every height, with it given 0.5 m up through
    eddies (10 / 0.5)^(-2/3) as strong; paths aloft take Cn² at their height."""
    same = eddies(1e-5)
    points = [
        np.array([[0.0, 0.0, 2.0], [150.0, 0.0, 1.5]]),
        np.array([[0.0, 0.0, 2.0], [150.0 * 2.0 / 3.5, 0.0, 0.0], [150.0, 0.0, 1.5]]),
    ]
    plain = same.variances(points)
    at_outer = dataclasses.replace(same, height_m=10.0).variances(points)
    below = dataclasses.replace(same, height_m=0.5).variances(points)
    np.testing.assert_allclose(at_outer, plain, rtol=1e-12)
    np.testing.assert_allclose(below, plain * 20.0 ** (-2.0 / 3.0), rtol=1e-12)
    # Two level paths at one height above the outer scale take exactly the strength at
    # that height.
    level = [
        np.array([[0.0, 0.0, 40.0], [100.0, 0.0, 40.0]]),
        np.array([[0.0, 0.0, 40.0], [100.0, 1.0, 40.0]]),
    ]
    tall = dataclasses.replace(same, height_m=20.0)
    np.testing.assert_allclose(
        tall.variances(level)[0, 1], same.variances(level)[0, 1] * 2.0 ** (-2.0 / 3.0), rtol=1e-12
    )
    # The ground's height counts from the scene's coordinates.
    raised = dataclasses.replace(tall, ground=20.0)
    np.testing.assert_allclose(
        raised.variances(level)[0, 1], same.variances(level)[0, 1], rtol=1e-12
    )


def test_eddies_weaker_with_height_scatter_less_into_a_high_shadow() -> None:
    z = node_heights(None)
    upward = Profile(z, 340.0 - 0.1 * z)
    same = eddies(1e-7)
    centres = np.array([250.0, 1000.0, 4000.0])
    plain = same.scattered(upward, 2.0, 1.5, 300.0, centres, 60.0)
    low = dataclasses.replace(same, height_m=0.5).scattered(upward, 2.0, 1.5, 300.0, centres, 60.0)
    high = dataclasses.replace(same, height_m=20.0).scattered(
        upward, 2.0, 1.5, 300.0, centres, 60.0
    )
    assert np.all(low > 0.0) and high.max() < 1.0
    assert np.all(low < plain) and np.all(plain < high)
    # The single scattering alone, from the air both see (from a metre or two up to tens
    # of metres), takes the eddies up there: weaker than at half a metre.
    single = same._single(upward, 2.0, 1.5, 300.0, centres)
    weaker = dataclasses.replace(same, height_m=0.5)._single(upward, 2.0, 1.5, 300.0, centres)
    assert np.all(weaker < single)


def test_eddies_the_same_at_every_height_are_as_before() -> None:
    listeners = ((150.0, 0.0, 1.5), (-120.0, 0.0, 1.5))
    windy = {"preset": "windy"}
    plain = simulate(Room.from_scene(in_turbulence("strong", windy, listeners=listeners)))
    given = {"preset": "strong", "falloff": 1.0}  # a fall-off without a height: no profile
    again = simulate(Room.from_scene(in_turbulence(given, windy, listeners=listeners)))
    weaker = {"preset": "strong", "height_m": 2.0}
    tall = simulate(Room.from_scene(in_turbulence(weaker, windy, listeners=listeners)))
    for key in plain.responses:
        assert np.array_equal(plain[key].early.wander.variance, again[key].early.wander.variance)
        assert np.array_equal(plain[key].energy(), again[key].energy())
        # With a height, the lit listener's paths part differently and the one in the
        # shadow gets different scattered sound.
        assert not np.array_equal(plain[key].energy(), tall[key].energy())


def test_eddies_scatter_sound_into_a_shadow() -> None:
    listeners = ((-150.0, 0.0, 1.5), (80.0, 0.0, 1.5))
    windy = {"preset": "windy"}
    still = simulate(Room.from_scene(in_turbulence(None, windy, listeners=listeners)))
    gusty = simulate(Room.from_scene(in_turbulence("moderate", windy, listeners=listeners)))
    shadow = [10 * np.log10(r["s", "l0"].energy().sum(axis=1)) for r in (still, gusty)]
    lit = [10 * np.log10(r["s", "l1"].energy().sum(axis=1)) for r in (still, gusty)]
    # Upwind in the shadow the high bands come up a lot; downwind, little changes.
    assert np.all(shadow[1][4:] > shadow[0][4:] + 10.0)
    np.testing.assert_allclose(lit[1][:3], lit[0][:3], atol=1.0)


def test_scattering_into_a_shadow_grows_with_the_eddies() -> None:
    z = node_heights(None)
    upward = Profile(z, 340.0 - 0.1 * z)
    weak, strong = eddies(2e-7), eddies(2e-5)
    centres = np.array([125.0, 500.0, 2000.0, 8000.0])
    a = weak.scattered(upward, 2.0, 1.5, 300.0, centres, 60.0)
    b = strong.scattered(upward, 2.0, 1.5, 300.0, centres, 60.0)
    assert np.all(a > 0.0) and np.all(b > a) and np.all(b <= 1.0)
    # Nothing in the shadow's way: a downward profile has none.
    assert weak._single(Profile(z, 340.0 + 0.1 * z), 2.0, 1.5, 300.0, centres).sum() == 0.0


def test_coverage_maps_take_the_eddies() -> None:
    settings = Coverage(spacing=10.0, rays=200, height=1.5)
    maps = [
        coverage_map(
            Room.from_scene(in_turbulence(p, listeners=((50.0, 0.0, 1.5),))), settings=settings
        )
        for p in (None, "strong")
    ]
    a, b = (m.get("s").values["spl"] for m in maps)
    assert a.shape == b.shape and np.all(np.isfinite(b))
    assert not np.allclose(a, b)


# -- flicker -------------------------------------------------------------------------


def test_gusts_change_as_an_ornstein_uhlenbeck_process() -> None:
    gusts = Gusts(1.0, 0.1, 40000, seed=3)
    x = gusts._column(0, 0)
    assert np.var(x) == pytest.approx(1.0, abs=0.05)
    assert np.corrcoef(x[:-1], x[1:])[0, 1] == pytest.approx(math.exp(-0.1), abs=0.01)
    roots = [(np.array([0, 2]), None), (np.array([1]), None)]
    groups = np.array([0, 1, 0])
    first = gusts.draws(roots, groups, 5)
    assert [len(d) for d in first] == [2, 1]
    assert first[0][0] == gusts._column(0, 0)[5] and first[1][0] == gusts._column(1, 0)[5]
    again = Gusts(1.0, 0.1, 40000, seed=3).draws(roots, groups, 5)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))


def test_sound_flickers_through_the_eddies() -> None:
    # Eddies 2 m across, in still air: they change within about two seconds.
    turbulence = {"preset": "strong", "outer_scale_m": 2.0}
    scene = in_turbulence(turbulence, listeners=((140.0, 0.0, 1.5),), ir_length_s=0.6)
    result = simulate(Room.from_scene(scene))
    response = result["s", "l0"]
    assert flickers(response) and response.early.wander.correlation_s == 2.0
    rate = 8000
    t = np.arange(8 * rate) / rate
    tone = np.sin(2 * np.pi * 1000.0 * t)
    wet = auralize(result, {"s": tone}, sample_rate=rate)[:, 0]
    frame = rate // 20
    level = 10 * np.log10(np.mean(wet[: 160 * frame].reshape(160, frame)[10:] ** 2, axis=1))
    still = dataclasses.replace(response, early=dataclasses.replace(response.early, wander=None))
    result.responses["s", "l0"] = still
    steady = auralize(result, {"s": tone}, sample_rate=rate)[:, 0]
    plain = 10 * np.log10(np.mean(steady[: 160 * frame].reshape(160, frame)[10:] ** 2, axis=1))
    assert np.ptp(level) > 3.0 > np.ptp(plain)
    assert len(wet) == len(steady)


def tone_at(signal: np.ndarray, f: float, rate: int, t: float, half: float = 0.005) -> complex:
    """The complex amplitude of a sine at ``f`` (Hz) in ``signal`` over the ``2 half``
    seconds about ``t``: the response at ``f`` of whatever the sine went through."""
    k = np.arange(int(round((t - half) * rate)), int(round((t + half) * rate)))
    return complex(2j * np.mean(signal[k] * np.exp(-2j * np.pi * f * k / rate)))


def half_way_errors(response, wet: np.ndarray, f: float, rate: int, gusts, seed, moments):
    """How far a sine at ``f`` played through ``response`` in turbulence (``wet``) is,
    half-way between each two of ``moments`` and the next, from the response whose
    paths are delayed half-way (the eddies' delays are linear in their draws): the
    largest difference, in dB of the response's rms amplitude (-40 dB is 1 %)."""
    from acoustic_spaces.spatial import Mono

    early = response.early
    roots = early.wander_roots()
    errors, scale = [], []
    for j in moments:
        a, b = gusts.draws(roots, early.point, j), gusts.draws(roots, early.point, j + 1)
        paths = early.at_moment(roots, [(x + y) / 2.0 for x, y in zip(a, b, strict=True)])
        ir = dataclasses.replace(response, early=paths).spatial_impulse_response(
            Mono(), rate, seed
        )[:, 0]
        expected = np.sum(ir * np.exp(-2j * np.pi * f * np.arange(len(ir)) / rate))
        errors.append(abs(tone_at(wet, f, rate, (j + 0.5) * gusts.step_s) - expected))
        scale.append(abs(expected) ** 2)
    return 20.0 * math.log10(max(errors) / math.sqrt(np.mean(scale)))


def test_half_way_between_two_moments_each_path_is_half_way_there() -> None:
    """Between two moments of the eddies each path's delay goes from one moment's to
    the next's, so half-way a tone has the response whose paths are delayed half-way.
    Crossfading the two moments' responses instead mixed each path at two delays,
    which half-way cancels the treble: at 8 kHz here the largest difference was 19 % of
    the tone's rms amplitude (now 0.05 %)."""
    turbulence = {"preset": "strong", "outer_scale_m": 2.0}
    scene = in_turbulence(turbulence, listeners=((140.0, 0.0, 1.5),), ir_length_s=0.6)
    result = simulate(Room.from_scene(scene))
    response = result["s", "l0"]
    rate, f = 48_000, 8000.0
    dry = np.sin(2 * np.pi * f * np.arange(3 * rate) / rate)
    wet = auralize(result, {"s": dry}, sample_rate=rate)[:, 0]
    n = int(math.ceil(response.length_s * rate))
    gusts = gusts_for(response, (len(dry) + n) / rate)
    moments = range(4, int(2.5 / gusts.step_s))
    assert half_way_errors(response, wet, f, rate, gusts, None, moments) < -40.0


def test_a_walk_goes_through_the_eddies() -> None:
    from acoustic_spaces.walkthrough import render_walk, simulate_walk

    data = in_turbulence({"preset": "strong", "outer_scale_m": 2.0}, ir_length_s=0.4).to_data()
    data["listeners"][0]["position"] = [60.0, 0.0, 1.5]
    data["listeners"][0]["walk"] = {"to": [[64.0, 0.0, 1.5]], "speed_m_s": 2.0}
    data["sources"].append({"name": "t", "position": [-30.0, 5.0, 2.0]})
    walked = simulate_walk(Room.from_scene(Scene.model_validate(data)), step_m=1.0)
    assert all(flickers(r) for r in walked.result)
    rate = 8000
    tone = np.sin(2 * np.pi * 1000.0 * np.arange(3 * rate) / rate)
    signals = {"s": tone, "t": tone}
    wet = render_walk(walked, signals, sample_rate=rate)
    assert np.all(np.isfinite(wet)) and np.array_equal(
        wet, render_walk(walked, signals, sample_rate=rate)
    )
    for key, response in walked.result.responses.items():
        early = dataclasses.replace(response.early, wander=None)
        walked.result.responses[key] = dataclasses.replace(response, early=early)
    steady = render_walk(walked, signals, sample_rate=rate)
    assert wet.shape == steady.shape and not np.allclose(wet, steady)


def test_inspect_names_the_eddies(tmp_path, capsys) -> None:
    from acoustic_spaces.cli import main

    path = tmp_path / "scene.yaml"
    in_turbulence("moderate").to_yaml(path)
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "still air: sound goes in straight lines" in out
    assert "eddies: moderate, C_T² 0.008 K² m^-2/3, C_v² 0.12 m^4/3 s^-2, 10 m across" in out
    in_turbulence({"preset": "moderate", "height_m": 2.0}).to_yaml(path)
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "C_v² 0.12 m^4/3 s^-2 at 2 m, weakening with height as z^-0.67, 10 m across" in out
