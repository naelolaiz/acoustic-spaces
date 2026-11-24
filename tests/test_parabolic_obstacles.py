"""Barriers and buildings inside the parabolic equation: thin screens and rigid roofs."""

import math

import numpy as np
import pytest

from acoustic_spaces.engine import parabolic, wide_angle
from acoustic_spaces.engine.diffraction import band_frequencies
from acoustic_spaces.materials.ground import surface_impedance

F = band_frequencies()[0]
C0 = 343.0


def log_wind(u10: float):
    """The effective speed of sound in a logarithmic wind of ``u10`` m/s at 10 m over a
    roughness length of 5 cm."""
    a = u10 / math.log(201.0)
    return lambda z: C0 + a * np.log1p(np.maximum(np.asarray(z, dtype=float), 0.0) / 0.05)


def band_db(p: np.ndarray, ranges, heights, zs: float = 1.0) -> np.ndarray:
    """The band level over free field of the pressure ``p`` (F, n)."""
    ranges, heights = np.asarray(ranges), np.asarray(heights)
    return 10 * np.log10(np.mean(np.abs(p) ** 2, axis=0) * (ranges**2 + (heights - zs) ** 2))


def grass(f):
    zg = np.conj(surface_impedance(f, 200.0))
    return lambda r: zg


@pytest.mark.parametrize("solver", [parabolic, wide_angle])
@pytest.mark.parametrize("independent", [False, True])
def test_no_obstacle_and_one_beyond_reach_change_nothing(solver, independent) -> None:
    f = F[3]
    ranges = np.array([30.0, 60.0, 12.0, 45.0])
    heights = np.array([1.5, 4.0, 8.0, 1.5])
    plain = solver.march(
        f, log_wind(5.0), grass(f), 1.0, ranges, heights, 30.0, independent=independent
    )
    for obstacles in [(), [(80.0, 80.0, 6.0)], [(70.0, 90.0, 6.0)]]:
        got = solver.march(
            f,
            log_wind(5.0),
            grass(f),
            1.0,
            ranges,
            heights,
            30.0,
            independent=independent,
            obstacles=obstacles,
        )
        assert np.array_equal(got, plain)


@pytest.mark.parametrize("solver", [parabolic, wide_angle])
def test_a_listener_at_a_screen_hears_what_passes_its_top_however_the_run_goes(solver) -> None:
    f = F[2]
    ranges, heights = np.array([20.0, 20.0, 60.0]), np.array([2.0, 5.0, 1.5])
    runs = [
        solver.march(
            f,
            log_wind(5.0),
            None,
            1.0,
            ranges,
            heights,
            30.0,
            independent=independent,
            obstacles=[(20.0, 20.0, 3.0)],
        )
        for independent in (False, True)
    ]
    for got in runs:
        assert np.all(got[:, 0] == 0.0)  # below the top, on the screen
        assert np.all(np.abs(got[:, 1]) > 0.0)
    np.testing.assert_allclose(
        band_db(runs[1][:, 1:], ranges[1:], heights[1:]),
        band_db(runs[0][:, 1:], ranges[1:], heights[1:]),
        atol=0.05,
    )


def test_the_wide_angle_equation_starts_before_a_near_screen() -> None:
    """A screen nearer than the wide-angle equation's first step (8.2 m at 125 Hz)
    screens from where it stands, as it does in the narrow-angle equation."""
    f = F[1]
    ranges, heights = np.array([30.0, 60.0, 100.0]), np.array([1.5, 1.5, 4.0])
    screen = [(5.0, 5.0, 4.0)]
    narrow = parabolic.march(
        f, log_wind(5.0), None, 1.0, ranges, heights, 30.0, obstacles=screen, refracting=False
    )
    wide = wide_angle.march(f, log_wind(5.0), None, 1.0, ranges, heights, 30.0, obstacles=screen)
    open_ground = parabolic.march(f, log_wind(5.0), None, 1.0, ranges, heights, 30.0)
    assert np.all(band_db(narrow, ranges, heights) < band_db(open_ground, ranges, heights) - 15.0)
    np.testing.assert_allclose(
        band_db(wide, ranges, heights), band_db(narrow, ranges, heights), atol=0.3
    )


@pytest.mark.parametrize("solver", [parabolic, wide_angle])
def test_screens_and_buildings_in_any_order(solver) -> None:
    f = F[2]
    ranges, heights = np.array([30.0, 70.0, 120.0]), np.array([1.5, 4.0, 1.5])
    obstacles = [(20.0, 20.0, 3.0), (40.0, 52.0, 6.0), (60.0, 60.0, 5.0)]
    runs = [
        solver.march(f, log_wind(5.0), None, 1.0, ranges, heights, 30.0, obstacles=order)
        for order in (obstacles, obstacles[::-1], [obstacles[1], obstacles[2], obstacles[0]])
    ]
    for other in runs[1:]:
        assert np.array_equal(other, runs[0])


def test_the_field_changes_smoothly_with_a_screen_s_height() -> None:
    """At 500 Hz the height step is 5 cm: in 3 mm steps across two of them, the level
    behind the screen and the wind's effect on it change by at most 0.03 dB a step."""
    f = F[3]
    ranges, heights = np.array([60.0]), np.array([1.5])
    levels = {}
    for u in (0.0, 5.0):
        levels[u] = np.array(
            [
                band_db(
                    parabolic.march(
                        f,
                        log_wind(u),
                        None,
                        1.0,
                        ranges,
                        heights,
                        30.0,
                        obstacles=[(20.0, 20.0, top)],
                        refracting=False,
                    ),
                    ranges,
                    heights,
                )[0]
                for top in 4.0 + np.arange(-30, 31, 3) * 1e-3
            ]
        )
    assert np.abs(np.diff(levels[5.0])).max() <= 0.03  # 3 mm steps
    assert np.abs(np.diff(levels[5.0] - levels[0.0])).max() <= 0.03


@pytest.mark.parametrize("solver", [parabolic, wide_angle])
def test_a_roof_s_wind_effect_against_the_full_wave_solution(solver) -> None:
    """A rigid block 40 to 52 m from a source 1 m up, its roof 6 m up, over rigid
    ground, at 125 Hz: the wind's effect on the level 80 and 150 m away, 1.5 m up, is
    within 0.2 dB of a full-wave solution's (mode matching in range, Evans 1983, with
    the same roof height), in both equations: 0.12 dB in the narrow-angle one and 0.12
    in the wide-angle one. Two zero-field screens in its place, with no roof between
    them, are 0.29 dB off in the narrow-angle equation and 0.27 dB in the wide-angle
    one."""
    f = F[1]
    ranges, heights = np.array([80.0, 150.0]), np.array([1.5, 1.5])
    roof = 6.0087  # the full-wave solution's roof, on its grid
    full_wave = {5.0: np.array([-0.06, 1.08]), -5.0: np.array([-0.10, -2.15])}
    flat = {"refracting": False} if solver is parabolic else {}

    def level(u, obstacles):
        p = solver.march(
            f, log_wind(u), None, 1.0, ranges, heights, 30.0, obstacles=obstacles, **flat
        )
        return band_db(p, ranges, heights)

    block = [(40.0, 52.0, roof)]
    for u, expected in full_wave.items():
        effect = level(u, block) - level(0.0, block)
        np.testing.assert_allclose(effect, expected, atol=0.2)
    # Below the roof the field behind the front face is gone, and comes back behind it.
    over = solver.march(
        f,
        log_wind(5.0),
        None,
        1.0,
        np.array([46.0, 46.0]),
        np.array([3.0, 8.0]),
        30.0,
        obstacles=block,
    )
    assert np.all(over[:, 0] == 0.0) and np.all(np.abs(over[:, 1]) > 0.0)


@pytest.mark.parametrize("solver", [parabolic, wide_angle])
@pytest.mark.parametrize("f", [125.0, 500.0, 2000.0])
def test_a_source_over_a_roof_against_its_two_paths(solver, f) -> None:
    """A source 1 m over a rigid roof 6 m up that reaches past the points, in still
    air: the source and its image in the roof, 2 to 14 degrees down from the source,
    within 0.15 dB in both equations (the narrow-angle one within 0.11 dB, the
    wide-angle one 0.01; with the image in the ground the narrow-angle one was 0.63 dB
    over at 125 Hz, and the wide-angle one, with no roof under the source, missed the
    image)."""
    ranges = np.array([10.0, 20.0, 30.0, 30.0, 60.0])
    heights = np.array([7.5, 6.5, 7.5, 10.0, 7.5])
    roof, zs = 6.0, 7.0
    flat = {"refracting": False} if solver is parabolic else {}
    p = solver.march(
        np.array([f]),
        lambda z: np.full(np.shape(z), C0),
        None,
        zs,
        ranges,
        heights,
        40.0,
        independent=True,
        obstacles=[(0.0, 200.0, roof)],
        **flat,
    )[0]
    k = 2.0 * math.pi * f / C0
    direct = np.hypot(ranges, heights - zs)
    image = np.hypot(ranges, heights - (2.0 * roof - zs))
    exact = np.exp(1j * k * direct) / direct + np.exp(1j * k * image) / image
    np.testing.assert_allclose(20.0 * np.log10(np.abs(p / exact)), 0.0, atol=0.15)


def test_refracting_takes_the_wide_angle_equation_at_steep_points_in_still_air(monkeypatch) -> None:
    calls = []
    march = wide_angle.march

    def counted(*args, **kwargs):
        calls.append(len(args[4]))
        return march(*args, **kwargs)

    monkeypatch.setattr(wide_angle, "march", counted)
    f = F[2]
    ranges, heights = np.array([60.0, 30.0]), np.array([1.5, 12.0])  # the second is steep
    still = log_wind(0.0)
    alone = parabolic.march(f, still, None, 1.0, ranges, heights, 30.0)
    assert calls == []
    both = parabolic.march(f, still, None, 1.0, ranges, heights, 30.0, refracting=True)
    assert calls == [1]
    assert np.array_equal(both[:, 0], alone[:, 0])


def test_a_coarser_grid_keeps_the_wind_s_effect_behind_a_screen() -> None:
    """At 1 kHz a quarter wavelength's height steps instead of a tenth's change the
    wind's effect behind a 4 m screen by at most 0.1 dB."""
    f = F[4]
    ranges, heights = np.array([40.0, 80.0]), np.array([1.5, 4.0])
    effects = []
    for steps in (parabolic.STEPS_PER_WAVELENGTH, 4):
        levels = [
            band_db(
                parabolic.march(
                    f,
                    log_wind(u),
                    None,
                    1.0,
                    ranges,
                    heights,
                    25.0,
                    obstacles=[(15.0, 15.0, 4.0)],
                    steps_per_wavelength=steps,
                ),
                ranges,
                heights,
            )
            for u in (0.0, 5.0)
        ]
        effects.append(levels[1] - levels[0])
    np.testing.assert_allclose(effects[1], effects[0], atol=0.1)


def test_runs_sharing_the_march_over_the_open_ground_get_what_they_get_alone(monkeypatch) -> None:
    """Runs that differ in their obstacles, points and grounds take one trunk one after
    another, in three orders: each has the bytes of the run by itself, and together they
    take fewer transforms."""
    f = F[3]
    wind = log_wind(5.0)
    lawn, hard = grass(f), (lambda r: np.full(len(f), 30.0 + 20.0j))
    there = lawn(0.0)
    patchy = lambda r: there if r < 20.0 else hard(r)  # noqa: E731
    runs = [
        (lawn, [(45.0, 45.0, 4.0)], [63.0, 93.0], True),
        (lawn, [(45.0, 45.0, 4.0), (55.0, 55.0, 4.0)], [50.0, 93.0], True),
        (lawn, [(45.0, 55.0, 6.0)], [63.0, 63.0], True),
        (lawn, [(30.0, 30.0, 3.0)], [20.0, 93.0], False),
        (lawn, [(-2.0, 3.0, 5.0), (60.0, 60.0, 4.0)], [93.0], True),
        (patchy, [(45.0, 45.0, 4.0)], [63.0, 93.0], True),
        (patchy, [(60.0, 60.0, 4.0)], [93.0], True),
        (hard, [(45.0, 45.0, 4.0)], [63.0], True),
        (None, [(60.0, 60.0, 4.0)], [93.0], True),
    ]
    transforms = []
    fft = np.fft.fft

    def counted(a, *args, **kwargs):
        transforms.append(len(a))
        return fft(a, *args, **kwargs)

    monkeypatch.setattr(np.fft, "fft", counted)

    def march(run, trunk):
        ground, obstacles, ranges, independent = run
        return parabolic.march(
            f,
            wind,
            ground,
            1.0,
            np.array(ranges),
            np.full(len(ranges), 1.5),
            25.0,
            independent=independent,
            obstacles=obstacles,
            trunk=trunk,
        )

    alone = [march(run, None) for run in runs]
    apart = sum(transforms)
    for order in (range(len(runs)), reversed(range(len(runs))), (3, 4, 2, 0, 6, 5, 1, 7, 8)):
        transforms.clear()
        trunk = parabolic.Trunk()
        for i in order:
            assert march(runs[i], trunk).tobytes() == alone[i].tobytes()
        assert sum(transforms) < apart


def test_a_run_with_steep_points_drops_the_march_kept(monkeypatch) -> None:
    """A run with points for the wide-angle equation clears the trunk it is given: it
    gets what it gets alone, and so does the run after it, which marches from the source
    again."""
    f = F[3]
    wind = log_wind(5.0)
    lawn = grass(f)
    transforms = []
    fft = np.fft.fft

    def counted(a, *args, **kwargs):
        transforms.append(len(a))
        return fft(a, *args, **kwargs)

    monkeypatch.setattr(np.fft, "fft", counted)

    def march(points, trunk):
        ranges, heights = points
        return parabolic.march(
            f,
            wind,
            lawn,
            1.0,
            np.array(ranges),
            np.array(heights),
            25.0,
            independent=True,
            obstacles=[(45.0, 45.0, 4.0)],
            trunk=trunk,
        )

    flat = ([63.0, 93.0], [1.5, 1.5])
    steep = ([50.0, 93.0], [20.0, 1.5])
    alone = march(flat, None)
    from_the_source = sum(transforms)
    high = march(steep, None)
    trunk = parabolic.Trunk()
    assert march(flat, trunk).tobytes() == alone.tobytes()
    transforms.clear()
    assert march(flat, trunk).tobytes() == alone.tobytes()
    assert sum(transforms) < from_the_source
    assert march(steep, trunk).tobytes() == high.tobytes()
    transforms.clear()
    assert march(flat, trunk).tobytes() == alone.tobytes()
    assert sum(transforms) == from_the_source


def still_air(z):
    return np.full(np.shape(z), C0)


@pytest.mark.parametrize("band", [2, 4])
def test_in_bounded_steps_the_level_behind_a_screen_does_not_go_with_the_layer_start(band) -> None:
    """15 and 60 m from a source half a metre up, behind a 7.43 m screen 4 m from it, in
    still air, the band level of a march in bounded steps is the same within 0.001 dB
    under absorbing layers that start 21.5, 60 and 240 m up (to four decimals of a dB at
    250 Hz and 1 kHz). The plain march's goes with the start, by 1.15 and 0.27 dB at
    250 Hz and by 0.23 and 0.06 dB at 1 kHz: the sound the screen's top bends almost
    straight up comes back down from the top of its grid."""
    f = F[band]
    ranges, heights = np.array([15.0, 60.0]), np.array([1.5, 1.5])
    spread = {}
    for bounded in (False, True):
        levels = [
            band_db(
                parabolic.march(
                    f,
                    still_air,
                    None,
                    0.5,
                    ranges,
                    heights,
                    top,
                    independent=True,
                    obstacles=[(4.0, 4.0, 7.43)],
                    refracting=False,
                    bounded=bounded,
                ),
                ranges,
                heights,
                0.5,
            )
            for top in (21.5, 60.0, 240.0)
        ]
        spread[bounded] = np.ptp(levels, axis=0)
    assert np.all(spread[True] < 0.001)
    assert spread[False][0] > (1.0 if band == 2 else 0.2)
    assert spread[False][1] > (0.2 if band == 2 else 0.05)


@pytest.mark.parametrize("zs", [2.0, 7.0])
def test_in_bounded_steps_each_point_gets_what_it_gets_alone(zs) -> None:
    """A march in bounded steps past a screen and a building, in wind over grass, gives
    each point the bytes a march to it alone gives: short of the screen (the plain
    march), behind it, at the building's front, over its roof, a metre and 30 m behind
    it (the first by a step of its own from the building's back) and beyond; with the
    source over a roof too, and from a march kept for the runs after it."""
    f = F[1][::3]
    lawn = grass(f)
    obstacles = [(20.0, 20.0, 3.0), (40.0, 60.0, 8.0)]
    if zs > 5.0:
        obstacles = [(-5.0, 10.0, 6.0), *obstacles]
    ranges = np.array([90.0, 12.0, 40.0, 50.0, 39.9, 61.0, 140.0, 20.5, 61.0])
    heights = np.array([1.5, 2.0, 9.0, 9.5, 4.0, 3.0, 1.5, 0.5, 1.5])

    def march(chosen, trunk=None):
        return parabolic.march(
            f,
            log_wind(5.0),
            lawn,
            zs,
            ranges[chosen],
            heights[chosen],
            30.0,
            independent=True,
            obstacles=obstacles,
            refracting=False,
            bounded=True,
            trunk=trunk,
        )

    together = march(slice(None))
    kept = parabolic.Trunk()
    assert march(slice(None), kept).tobytes() == together.tobytes()
    for i in range(len(ranges)):
        assert np.array_equal(march(slice(i, i + 1))[:, 0], together[:, i])
        assert np.array_equal(march(slice(i, i + 1), kept)[:, 0], together[:, i])
    assert np.all(np.isfinite(together))


def test_a_bounded_march_over_open_ground_is_the_plain_one() -> None:
    """With no obstacle on the way, or none short of the farthest point, a march asked
    for in bounded steps gives the plain march's bytes."""
    f = F[3][::2]
    ranges, heights = np.array([30.0, 60.0, 12.0, 45.0]), np.array([1.5, 4.0, 1.0, 1.5])
    for obstacles in [(), [(80.0, 80.0, 6.0)], [(70.0, 90.0, 6.0)]]:
        got = [
            parabolic.march(
                f,
                log_wind(5.0),
                grass(f),
                1.0,
                ranges,
                heights,
                30.0,
                independent=True,
                obstacles=obstacles,
                bounded=bounded,
            )
            for bounded in (False, True)
        ]
        assert got[1].tobytes() == got[0].tobytes()


def test_bounded_steps_begin_at_the_source_and_take_each_point_by_its_own_step() -> None:
    f = F[1][:2]
    at = (np.array([30.0]), np.array([1.5]), 30.0)
    with pytest.raises(ValueError, match="its own last step"):
        parabolic.march(f, still_air, None, 1.0, *at, bounded=True)
    more = {"independent": True, "refracting": False, "bounded": True}
    for wrong in (
        {"layer": 200.0},
        {"edges": []},
        {"start": (10.0, lambda z: np.ones((len(f), len(z)), dtype=complex))},
    ):
        with pytest.raises(ValueError, match="begins at the source"):
            parabolic.march(f, still_air, None, 1.0, *at, **more, **wrong)


@pytest.mark.parametrize("band", [0, 1, 2, 3, 4, 5, 6, 7])
def test_a_bounded_step_s_table_is_the_plain_step_s_for_sound_that_goes_gently(band) -> None:
    """Under the absorbing layer of a run with its top 30 m up, a full bounded step's
    table differs from the plain step's by less than 1e-5 for waves up to 30 degrees
    from the horizontal, 1e-4 up to 45 degrees and 2e-4 up to the steepest that the step
    takes whole (4e-6, 4e-5 and 9e-5 at most): 51 degrees in the bands from 63 Hz to
    1 kHz, 55 at 2 kHz, 60 at 4 kHz and 64 at 8 kHz. No entry is larger than 1.05
    (1.0495: the waves some 65 to 80 degrees up, of which the window lets a part
    through)."""
    f = F[band]
    steps = parabolic._Steps(
        f, still_air, None, 1.0, 30.0, None, None, parabolic.STEPS_PER_WAVELENGTH
    )
    table = steps.propagator(steps.step, steps.n, 0.0, None)
    plain = parabolic._exp_mirrored(1j * steps.step * (steps.vertical - steps.ka[:, None]))
    sine = np.abs(steps.kz)[None, :] / steps.ka[:, None]
    off = np.abs(table - plain)
    steepest = max(d for d in range(30, 90) if steps.takes(math.tan(math.radians(d)), steps.step))
    assert steepest == {5: 55, 6: 60, 7: 64}.get(band, 51)
    for degrees, within in ((30.0, 1e-5), (45.0, 1e-4), (steepest, 2e-4)):
        assert steps.takes(math.tan(math.radians(degrees)), steps.step)
        assert off[sine < math.sin(math.radians(degrees))].max() < within
    assert np.abs(table).max() < 1.05


def test_a_bounded_step_s_table_is_held_where_a_wave_would_grow_too_far() -> None:
    """The factor that keeps a wave from growing by more than BOUNDED_GROWTH between the
    ground and the layer's top changes a full step's table of the 1 kHz band by less
    than 1e-6 under a layer whose top is 1 km up (1.3e-7: the waves along the ground,
    which the window makes larger than 1 by that), and by more than 0.1 where it is 4 km
    up (0.14: the waves some 65 to 80 degrees up). Asked to let nothing grow, it brings
    every entry down to 1."""
    f = F[4]
    steps = parabolic._Steps(
        f, still_air, None, 1.0, 30.0, None, None, parabolic.STEPS_PER_WAVELENGTH
    )
    thick = steps.thickness
    window = (parabolic.BOUNDED_WHOLE * thick, parabolic.BOUNDED_WINDOW * thick)
    args = (steps.ka, steps.n, steps.dz, steps.step, *window)
    bare = parabolic._bounded(*args)
    smooth = parabolic.BOUNDED_SMOOTH * thick
    change = [
        np.abs(
            parabolic._bounded(*args, smooth, math.log(parabolic.BOUNDED_GROWTH) / tall) - bare
        ).max()
        for tall in (1000.0, 4000.0)
    ]
    assert change[0] < 1e-6
    assert change[1] > 0.1
    assert np.abs(parabolic._bounded(*args, smooth, 0.0)).max() < 1.0 + 1e-12
    assert np.abs(bare).max() > 1.04


@pytest.mark.parametrize("band", [2, 4])
def test_a_point_close_under_an_edge_gets_one_step_of_its_own(band, monkeypatch) -> None:
    """6, 12 and 20 m behind a 20 m edge 40 m from the source, 1.5 m up, the sound from
    the edge's top comes down more steeply than a full bounded step takes whole, to the
    nearest at 74 degrees: such a point gets one step of its own from the field at the
    edge. Its band level at 250 Hz and 1 kHz is within 0.001 dB in still air (0.0003 dB
    at most) and within 0.2 dB in a wind of 10 m/s (0.12 dB at 6 m, 0.06 dB or less at
    the others) of a march in full steps under an absorbing layer 320 m thick, whose
    steps take that sound whole."""
    f = F[band][::2]
    ranges, heights = np.array([46.0, 52.0, 60.0]), np.array([1.5, 1.5, 1.5])
    cut = [(40.0, 40.0, 20.0)]
    for speed in (still_air, log_wind(10.0)):
        levels = []
        for least in (20.0, 320.0):
            monkeypatch.setattr(parabolic, "LAYER_MIN_M", least)
            steps = parabolic._Steps(
                f, speed, None, 1.0, 60.0, None, None, parabolic.STEPS_PER_WAVELENGTH
            )
            slope = (20.0 + 1.5) / (ranges - 40.0)
            whole = [
                steps.takes(s, min(steps.step, r - 40.0))
                for s, r in zip(slope, ranges, strict=True)
            ]
            assert whole[0] == all(whole) == (least == 320.0)
            p = parabolic.march(
                f,
                speed,
                None,
                1.0,
                ranges,
                heights,
                60.0,
                independent=True,
                obstacles=cut,
                refracting=False,
                bounded=True,
            )
            levels.append(band_db(p, ranges, heights))
        within = 0.001 if speed is still_air else 0.2
        assert np.all(np.abs(levels[0] - levels[1]) < within)


def test_a_tall_top_s_steep_sound_over_a_lower_wall_is_taken_whole(monkeypatch) -> None:
    """Behind a 20 m edge 40 m from the source and a 2 m wall 2 m behind it, the sound
    from the edge's top comes down over the wall more steeply than from the wall's own
    top: the steps go by the steepest way from every top met, so that 7, 10, 15 and
    30 m behind the edge, 1.5 m up, the 1 kHz band in still air is within 0.1 dB of a
    march under an absorbing layer 320 m thick (0.05 dB at most; 0.17 dB going by the
    last top alone)."""
    f = F[4][::2]
    ranges, heights = np.array([47.0, 50.0, 55.0, 70.0]), np.full(4, 1.5)
    levels = []
    for least in (20.0, 320.0):
        monkeypatch.setattr(parabolic, "LAYER_MIN_M", least)
        p = parabolic.march(
            f,
            still_air,
            None,
            1.0,
            ranges,
            heights,
            60.0,
            independent=True,
            obstacles=[(40.0, 40.0, 20.0), (42.0, 42.0, 2.0)],
            refracting=False,
            bounded=True,
        )
        levels.append(band_db(p, ranges, heights))
    assert np.all(np.abs(levels[0] - levels[1]) < 0.1)


def test_a_steep_point_gets_its_own_step_where_the_march_has_come_to_its_range() -> None:
    """Three full steps behind a 3 m screen 10 m from the source, a point 15 m up, which
    the screen's top reaches by the ground more steeply than a full step takes whole,
    gets the bytes it gets alone with a point 1.5 m up at the same range before it or
    after it, for which the march has come to that range in full steps."""
    f = F[3][:1]
    steps = parabolic._Steps(
        f, still_air, None, 1.0, 30.0, None, None, parabolic.STEPS_PER_WAVELENGTH
    )
    # (As the march sums its steps, to land within its 1e-9 of the range.)
    r = 10.0
    r += steps.step
    r += steps.step
    goal = r + 5e-10 + steps.step
    assert not steps.takes((3.0 + 15.0) / (goal - 10.0), steps.step)
    assert steps.takes((3.0 + 1.5) / (goal - 10.0), steps.step)

    def march(ranges, heights):
        return parabolic.march(
            f,
            still_air,
            None,
            1.0,
            np.array(ranges),
            np.array(heights),
            30.0,
            independent=True,
            obstacles=[(10.0, 10.0, 3.0)],
            refracting=False,
            bounded=True,
        )

    alone = march([goal], [15.0])[:, 0]
    assert np.array_equal(march([goal, goal], [1.5, 15.0])[:, 1], alone)
    assert np.array_equal(march([goal, goal], [15.0, 1.5])[:, 0], alone)


def two_paths(ka: float, zs: float, r: float, z: float) -> complex:
    """The pressure of a source ``zs`` m over rigid ground ``r`` m away at the height
    ``z``: the direct sound and the reflection."""
    direct, bounced = math.hypot(r, z - zs), math.hypot(r, z + zs)
    return np.exp(1j * ka * direct) / direct + np.exp(1j * ka * bounced) / bounced


def own_step(impedance, zs: float, r: float, z: float) -> tuple[complex, float]:
    """The pressure at 2 kHz of one step of a point's own from the starting field of a
    source ``zs`` m up to the height ``z`` ``r`` m away, under a layer that starts 49 m
    up; and the wavenumber."""
    steps = parabolic._Steps(np.array([2000.0]), still_air, impedance, zs, 49.0, None, None, 10)
    steps.bound = True
    out = np.empty_like(steps.start())
    steps.own(steps.start(), 0.0, r, out, z)
    return parabolic._at_height(steps.pressure(out, r), steps.z, z)[0], float(steps.ka[0])


@pytest.mark.parametrize(
    ("zs", "r", "z", "within"),
    [(45.0, 60.0, 10.0, 0.2), (15.0, 120.0, 30.0, 0.1), (5.0, 30.0, 2.0, 0.1)],
)
def test_a_point_s_own_step_holds_the_reflection_of_every_height(zs, r, z, within) -> None:
    """One step of a point's own takes the sound of every height of the grid straight
    and by the ground, to a high point as to a low one: at 2 kHz over rigid ground,
    under a layer that starts 49 m up on a grid 70 m high, from the starting field of a
    source 45 m up to a point 10 m up 60 m away, of one 15 m up to a point 30 m up 120 m
    away and of one 5 m up to a point 2 m up 30 m away, it gives the level of the direct
    sound and the reflection within 0.2 dB (0.07, 0.001 and 0.01 dB; with a window about
    the point, which misses the reflection from as far below the ground as the grid has
    heights above the point, the first two were 2.2 and 5.0 dB off)."""
    p, ka = own_step(None, zs, r, z)
    assert abs(20 * np.log10(abs(p) / abs(two_paths(ka, zs, r, z)))) < within


def test_a_point_s_own_step_over_hard_ground_is_the_one_over_rigid_ground() -> None:
    """Where the field is continued below a hard ground (an impedance of 2000 + 2000i at
    2 kHz), a point's own step keeps it whole as deep as the layer starts above the
    ground: 2 and 10 m up 60 m from a source 45 m up, under a layer that starts 49 m
    up on a grid 70 m high, the step gives the level it gives over rigid ground within
    0.02 dB (0.003 and 0.013 dB; with the field faded from half the grid's height down,
    as the march's steps have it, 0.8 and 0.9 dB off)."""
    hard = np.array([2000.0 + 2000.0j])
    steps = parabolic._Steps(
        np.array([2000.0]), still_air, lambda r: hard, 45.0, 49.0, None, None, 10
    )
    here = steps.ground(0.0)
    assert here.image.all()
    assert here.fade_own[steps.z <= 49.0].min() == 1.0
    assert here.fade_below[0, steps.z > 36.0].max() < 1.0
    for z in (2.0, 10.0):
        over_hard, _ = own_step(lambda r: hard, 45.0, 60.0, z)
        over_rigid, _ = own_step(None, 45.0, 60.0, z)
        assert abs(20 * np.log10(abs(over_hard) / abs(over_rigid))) < 0.02


def fine_reference(monkeypatch, f, impedance, speed_at, ranges, heights, top, cut):
    """The band's level (dB) of a plain march in quarter-wavelength steps under an
    absorbing layer 320 m thick, in still air or the airs ``speed_at`` gives."""
    with monkeypatch.context() as patch:
        patch.setattr(parabolic, "LAYER_MIN_M", 320.0)
        patch.setattr(parabolic, "RANGE_STEP_WAVELENGTHS", 0.25)
        patch.setattr(parabolic, "RANGE_STEP_MIN_M", 0.0)
        p = parabolic.march(
            f,
            still_air,
            impedance,
            1.0,
            ranges,
            heights,
            top,
            speed_at=speed_at,
            independent=True,
            obstacles=cut,
            refracting=False,
        )
    return band_db(p, ranges, heights)


@pytest.mark.parametrize(
    ("what", "band", "edge", "change", "behind", "within"),
    [
        ("ground", 2, 20.0, 8.0, 12.0, 0.1),
        ("air", 4, 20.0, 4.0, 6.0, 0.05),
        ("ground", 3, 40.0, 3.0, 6.0, 0.15),
        ("air", 4, 40.0, 3.0, 6.0, 0.35),
    ],
)
def test_a_point_s_own_step_takes_the_ground_its_sound_lands_on_and_the_airs_on_its_way(
    what, band, edge, change, behind, within, monkeypatch
) -> None:
    """A point's own step takes each height's reflected sound over the ground it lands
    on (the sound of the heights above an edge's top lands between where the top's does
    and the point), and the air's phase along each height's straight way through the
    airs it meets. Behind an edge 40 m from the source, 1.5 m up, against a plain march
    in quarter-wavelength steps under a layer 320 m thick: 12 m behind a 20 m edge over
    a ground that turns from rigid to grass 8 m behind it, the 250 Hz band is within
    0.07 dB (0.84 dB with the ground at the step's middle); 6 m behind it where still
    air gives way to a wind of 10 m/s 4 m behind it, the 1 kHz band within 0.02 dB
    (0.63 dB with the air at the step's middle); and 6 m behind a 40 m edge, which its
    top reaches 82 degrees down, with the change 3 m behind the edge, the 500 Hz band
    within 0.10 dB over the ground and the 1 kHz band within 0.27 dB in the airs, where
    with one ground and one air all the way they are 0.05 and 0.07 dB off."""
    f = F[band][::2]
    at = 40.0 + change
    impedance, speed_at = None, None
    if what == "ground":
        lawn = grass(f)(0.0)
        impedance = lambda r: None if r < at else lawn  # noqa: E731
    else:
        windy = log_wind(10.0)
        speed_at = lambda r: still_air if r < at else windy  # noqa: E731
    ranges, heights = np.array([40.0 + behind]), np.array([1.5])
    cut = [(40.0, 40.0, edge)]
    taken = []
    own = parabolic._Steps.own
    monkeypatch.setattr(
        parabolic._Steps, "own", lambda self, *a: (taken.append(a[1]), own(self, *a))[1]
    )
    p = parabolic.march(
        f,
        still_air,
        impedance,
        1.0,
        ranges,
        heights,
        1.5 * edge,
        speed_at=speed_at,
        independent=True,
        obstacles=cut,
        refracting=False,
        bounded=True,
    )
    assert taken == [40.0]
    level = band_db(p, ranges, heights)
    reference = fine_reference(
        monkeypatch, f, impedance, speed_at, ranges, heights, 1.5 * edge, cut
    )
    assert abs(level[0] - reference[0]) < within


def test_a_point_s_own_step_takes_a_ground_whole_close_behind_a_change(monkeypatch) -> None:
    """A point's own step takes each height's reflected sound whole over the ground it
    lands on, however close behind a change of ground that is: 1.5 m up 6 m behind a
    40 m edge 40 m from the source, over rigid ground that turns to grass 5.5 m behind
    the edge, 0.28 m short of where the top's reflection lands, the 500 Hz band is
    within 0.15 dB of a plain march in quarter-wavelength steps under a layer 320 m
    thick with the grass from 3 m behind the edge on (0.10 dB), and 0.62 dB from that
    march over the ground as it is, which is 0.11 dB from what it is over rigid ground
    all the way."""
    f = F[3][::2]
    lawn = grass(f)(0.0)
    ranges, heights = np.array([46.0]), np.array([1.5])
    cut = [(40.0, 40.0, 40.0)]
    p = parabolic.march(
        f,
        still_air,
        lambda r: None if r < 45.5 else lawn,
        1.0,
        ranges,
        heights,
        60.0,
        independent=True,
        obstacles=cut,
        refracting=False,
        bounded=True,
    )
    level = band_db(p, ranges, heights)[0]
    mixed, rigid, soft = (
        fine_reference(
            monkeypatch,
            f,
            (lambda r, at=at: None if r < at else lawn),
            None,
            ranges,
            heights,
            60.0,
            cut,
        )[0]
        for at in (45.5, 1e9, 43.0)
    )
    assert 0.4 < abs(level - mixed) < 0.8
    assert abs(level - soft) < 0.15
    assert min(rigid, soft) - 0.05 < mixed < max(rigid, soft) + 0.05


def test_a_point_s_own_step_in_sight_of_the_source_takes_each_height_s_ground(monkeypatch) -> None:
    """In sight of the source over a wall, the sound of the heights above the wall's top
    lands further on than the top's own: 1.5 m up 10 m behind a 3 m wall 10 m from a
    source 30 m up, the top's reflection lands 6.7 m behind the wall and the source's
    own 9.0 m behind it. Over rigid ground that turns to grass 7.8 m behind the wall,
    between the two, the 1 kHz band of the point's own step from the wall is within
    0.15 dB of a plain march in quarter-wavelength steps under a layer 320 m thick
    (0.10 dB), which is 1.4 dB under what it is over rigid ground all the way (the level
    of a step over the one ground the top's reflection lands on)."""
    f = F[4][::2]
    lawn = grass(f)(0.0)
    ranges, heights = np.array([20.0]), np.array([1.5])
    cut = [(10.0, 10.0, 3.0)]
    taken = []
    own = parabolic._Steps.own
    monkeypatch.setattr(
        parabolic._Steps, "own", lambda self, *a: (taken.append(a[1]), own(self, *a))[1]
    )

    def level(impedance, bounded: bool) -> float:
        with monkeypatch.context() as patch:
            if not bounded:
                patch.setattr(parabolic, "LAYER_MIN_M", 320.0)
                patch.setattr(parabolic, "RANGE_STEP_WAVELENGTHS", 0.25)
                patch.setattr(parabolic, "RANGE_STEP_MIN_M", 0.0)
            p = parabolic.march(
                f,
                still_air,
                impedance,
                30.0,
                ranges,
                heights,
                60.0,
                independent=True,
                obstacles=cut,
                refracting=False,
                bounded=bounded,
            )
        return float(band_db(p, ranges, heights, 30.0)[0])

    mixed = lambda r: None if r < 17.8 else lawn  # noqa: E731
    stepped = level(mixed, True)
    assert taken == [10.0]
    reference, rigid = level(mixed, False), level(None, False)
    assert abs(stepped - reference) < 0.15
    assert 1.2 < rigid - reference < 1.6


def test_a_point_s_own_step_over_two_stretches_of_one_ground_is_the_step_over_that_ground(
    monkeypatch,
) -> None:
    """The sound of each height taken over the ground it lands on, with the same ground
    in two stretches of a point's own step, is the step over that ground: every
    height's sound lands on one of the two, and the step's field is the same to 1e-9 of
    its size."""
    f = F[3][::4]
    lawn = grass(f)(0.0)
    steps = parabolic._Steps(f, still_air, lambda r: lawn, 1.0, 30.0, None, None, 10)
    steps.bound = True
    assert steps.grounds(10.0, 7.0) == [(1.0, 10.0 + 0.5 * steps.step)]
    field = steps.start()
    one = steps.own(field, 10.0, 7.0, np.empty_like(field), 6.0).copy()
    monkeypatch.setattr(
        parabolic._Steps, "grounds", lambda self, r, dr: [(0.3, 11.0), (0.8, 14.0), (1.0, 16.5)]
    )
    two = steps.own(field, 10.0, 7.0, np.empty_like(field), 6.0)
    assert np.abs(two - one).max() < 1e-9 * np.abs(one).max()


def test_a_point_s_own_step_through_two_stretches_of_one_air_is_the_step_in_that_air(
    monkeypatch,
) -> None:
    """The air's phase along each height's way through the airs along a point's own
    step, with the same air in two stretches of it, is the phase in that air: the
    step's field is the same to 1e-9 of its size."""
    f = F[3][::4]
    windy = log_wind(10.0)
    steps = parabolic._Steps(f, windy, None, 1.0, 30.0, None, None, 10)
    steps.bound = True
    assert steps.airs(10.0, 7.0) == []
    field = steps.start()
    one = steps.own(field, 10.0, 7.0, np.empty_like(field), 6.0).copy()
    monkeypatch.setattr(
        parabolic._Steps,
        "airs",
        lambda self, r, dr: [(0.3, self.delta.real), (1.0, self.delta.real)],
    )
    two = steps.own(field, 10.0, 7.0, np.empty_like(field), 6.0)
    assert np.abs(two - one).max() < 1e-9 * np.abs(one).max()


def test_the_airs_along_a_step_end_where_the_air_changes() -> None:
    """Where the air changes along the way, a step's airs are those at the middles of
    the steps the march would take there, full ones and a shorter last one, and each
    stretch in one air ends at the share of the step at which the air changes, found
    to a thousandth of a full step; none where the air is the same all along, given by
    one function or by two. The grounds along a step are found the same way."""
    f = np.array([500.0])
    windy, other = log_wind(10.0), log_wind(10.0)
    lawn = grass(f)(0.0)
    steps = parabolic._Steps(
        f,
        still_air,
        lambda r: None if r < 15.0 else lawn,
        1.0,
        30.0,
        None,
        lambda r: still_air if r < 15.0 else windy,
        10,
    )
    step = steps.step
    assert 10.0 + 0.5 * step < 15.0 < 10.0 + 1.5 * step
    (first, calm), (second, wind) = steps.airs(10.0, 8.5)
    assert abs(10.0 + first * 8.5 - 15.0) < step / 1000.0
    assert second == 1.0
    assert np.all(calm == 0.0)
    assert np.all(wind[:, 1:] < 0.0)
    (ends, before), (last, after) = steps.grounds(10.0, 8.5)
    assert ends == first
    assert last == 1.0
    assert before < 15.0 < after < 18.5
    assert steps.airs(10.0, 3.0) == []
    assert steps.grounds(10.0, 3.0) == [(1.0, 11.5)]
    steps.speed_at = lambda r: windy if r < 15.0 else other
    assert steps.airs(10.0, 8.5) == []


def test_a_high_point_s_level_does_not_go_with_the_layer_start() -> None:
    """14 and 16 m from a source 5 m up, 20 and 22 m up behind a 1 m wall 2 m from the
    source, the 1 kHz band of a march in bounded steps is the same within 0.001 dB under
    layers that start 25, 60 and 120 m up, in still air and in a wind of 10 m/s (to four
    decimals of a dB): each point's own step holds the reflection off the ground too,
    which comes from further below the ground than the lowest of those grids has
    heights above the point."""
    f = F[4][::2]
    ranges, heights = np.array([14.0, 16.0]), np.array([20.0, 22.0])
    for speed in (still_air, log_wind(10.0)):
        levels = [
            band_db(
                parabolic.march(
                    f,
                    speed,
                    None,
                    5.0,
                    ranges,
                    heights,
                    top,
                    independent=True,
                    obstacles=[(2.0, 2.0, 1.0)],
                    refracting=False,
                    bounded=True,
                ),
                ranges,
                heights,
                5.0,
            )
            for top in (25.0, 60.0, 120.0)
        ]
        assert np.all(np.ptp(levels, axis=0) < 0.001)


def test_a_point_s_own_step_holds_over_a_ground_between_soft_and_hard(monkeypatch) -> None:
    """Over a ground between soft and hard the field continued below the ground grows
    with depth, and the march fades it out over the lower half of the depth at which
    it has grown tenfold: with an impedance of 20 + 140i in the 500 Hz band from 14 to
    24 m down on, and as deep as a layer 60 m up starts it has grown by up to 138
    times. A point's own step keeps it whole that deep. Behind a 40 m edge 40 m from
    the source, 0.3 m up 4 m behind it, 1.5 m up 6, 12 and 20 m behind it and 10 m up
    20 m behind it, the band is within 0.01 dB of what it is under a layer that starts
    120 m up, on a grid twice as high, where that ground is taken by its reflection
    factor (0.001 dB; in shorter steps the first two points are 2.4 and 1.1 dB off, and
    with one step of the field as the march fades it the five are 4.7, 4.6, 3.6, 0.16
    and 2.3 dB off)."""
    f = F[3][::2]
    ground = np.full(len(f), 20.0 + 140.0j)
    ranges = np.array([44.0, 46.0, 52.0, 60.0, 60.0])
    heights = np.array([0.3, 1.5, 1.5, 1.5, 10.0])
    taken = []
    own = parabolic._Steps.own
    monkeypatch.setattr(
        parabolic._Steps, "own", lambda self, *a: (taken.append(self.top), own(self, *a))[1]
    )
    levels = []
    for top in (60.0, 120.0):
        steps = parabolic._Steps(
            f, still_air, lambda r: ground, 1.0, top, None, None, parabolic.STEPS_PER_WAVELENGTH
        )
        here = steps.ground(0.0)
        assert bool(here.image.all()) == (top == 60.0)
        assert bool(here.image.any()) == (top == 60.0)
        if top == 60.0:
            assert here.fade_below[:, steps.z > 25.0].max() < 1.0
            assert np.exp((-steps.ka / ground).imag * 60.0).max() > 100.0
            assert here.fade_own[steps.z <= 60.0].min() == 1.0
        p = parabolic.march(
            f,
            still_air,
            lambda r: ground,
            1.0,
            ranges,
            heights,
            top,
            independent=True,
            obstacles=[(40.0, 40.0, 40.0)],
            refracting=False,
            bounded=True,
        )
        levels.append(band_db(p, ranges, heights))
    assert taken == [60.0] * 5 + [120.0] * 5
    assert np.all(np.abs(levels[0] - levels[1]) < 0.01)


def test_the_growth_guard_covers_an_excess_its_scaled_smoothing_would_overshoot() -> None:
    """Where a narrow excess stands apart from a wide one, the smoothed excess scaled up
    to cover the narrow one passes 1 over the wide one, and 1 less it would turn the
    table's sign there. The factor made for that case (:func:`parabolic._covered`) lies
    between 0 and 1, takes off the excess everywhere, and is 1 less the transform of a
    function that ends where the bump does, so that the step's reach stays bounded."""
    n = 4096
    index = np.arange(2 * n)
    apart = np.minimum(index, 2 * n - index) * 0.02
    bump = parabolic._bump(apart / 6.0)
    kz = np.abs(np.fft.fftfreq(2 * n))
    excess = 0.3 * np.exp(-0.5 * ((kz - 0.35) / 0.02) ** 2)
    excess[np.abs(kz - 0.1) < 3e-4] = 0.5
    cover = np.maximum(np.fft.fft(np.fft.ifft(excess) * bump).real, 0.0)
    above = excess > 1e-12
    scale = float(np.max(excess[above] / cover[above]))
    assert np.min(1.0 - scale * cover) < -0.5
    factor = parabolic._covered(excess, bump)
    assert factor.min() >= 0.0
    assert factor.max() <= 1.0
    assert np.all(factor <= 1.0 - excess + 1e-9)
    kernel = np.abs(np.fft.ifft(1.0 - factor))
    assert kernel[bump == 0.0].max() < 1e-12 * kernel.max()


def test_a_point_s_own_step_over_a_roof_lower_than_a_height_step(monkeypatch) -> None:
    """A point's own step is taken over a roof of any height, one lower than the first
    height above the ground too: at 50 and 63 Hz (a height step of 0.54 m), over a roof
    from 10 to 40 m, points 20 and 25 m up 2 and 4 m past its front get their own steps,
    and their level goes smoothly with the roof's height through the height step
    (0.003 dB between roofs 2 % lower and higher than it, 0.06 dB between roofs 0.3 and
    0.8 m high)."""
    f = np.array([50.0, 63.0])
    dz = parabolic._Steps(f, still_air, None, 1.0, 30.0, None, None, 10).dz
    calls = []
    plain = parabolic._Steps.own
    monkeypatch.setattr(
        parabolic._Steps, "own", lambda self, *a: (calls.append(self.roof), plain(self, *a))[1]
    )

    def levels(roof):
        p = parabolic.march(
            f,
            still_air,
            None,
            1.0,
            np.array([12.0, 14.0]),
            np.array([20.0, 25.0]),
            30.0,
            independent=True,
            obstacles=[(10.0, 40.0, roof)],
            refracting=False,
            bounded=True,
        )
        return 20 * np.log10(np.abs(p))

    assert 0.3 < dz < 0.8
    low, under, over, high = (levels(roof) for roof in (0.3, 0.98 * dz, 1.02 * dz, 0.8))
    assert calls == [0.3, 0.3, 0.98 * dz, 0.98 * dz, 1.02 * dz, 1.02 * dz, 0.8, 0.8]
    assert np.all(np.abs(under - over) < 0.01)
    assert np.all(np.abs(low - high) < 0.1)


def test_a_run_keeps_the_tables_of_its_full_steps_alone(monkeypatch) -> None:
    """The tables of the points' own steps come and go, those as long as a full step
    included: three runs on one trunk, each with ten points of other heights one full
    step behind a screen, leave no more tables than one run does."""
    f = np.array([500.0])

    def air(z):
        return np.full(np.shape(z), 340.0)

    # (In air of 340 m/s the points 13.4 m away are just a full step past the screen.)
    assert parabolic._Steps(f, air, None, 1.0, 30.0, None, None, 10).step == 13.4 - 10.0
    seen = []
    plain = parabolic._Steps.run
    monkeypatch.setattr(
        parabolic._Steps, "run", lambda self, *a, **k: (seen.append(self), plain(self, *a, **k))[1]
    )
    trunk = parabolic.Trunk()
    kept = []
    for batch in range(3):
        parabolic.march(
            f,
            air,
            None,
            1.0,
            np.full(10, 13.4),
            np.linspace(8.0, 25.0, 10) + 0.01 * batch,
            30.0,
            independent=True,
            obstacles=[(10.0, 10.0, 3.0)],
            refracting=False,
            bounded=True,
            trunk=trunk,
        )
        kept.append(len(seen[-1].steps))
    assert seen[0] is seen[-1]
    assert kept[0] == kept[1] == kept[2] <= 5


def test_a_full_step_under_a_layer_kilometres_up_takes_the_guard_s_second_way(monkeypatch) -> None:
    """Under an absorbing layer that starts 8 km up (1 kHz, a quarter wavelength's
    height step) the full bounded step's table has an excess at a few wavenumbers that
    its scaled smoothing would overshoot: the table is made by
    :func:`parabolic._covered` there, and it takes no sound further up or down than
    the window and the smoothing reach, the layer's thickness."""
    covered = []
    plain = parabolic._covered
    monkeypatch.setattr(
        parabolic, "_covered", lambda excess, bump: (covered.append(1), plain(excess, bump))[1]
    )
    steps = parabolic._Steps(np.array([1000.0]), still_air, None, 1.0, 8000.0, None, None, 4)
    steps.bound = True
    table = steps.propagator(steps.step, steps.n, 0.0, None)
    assert covered == [1]
    kernel = np.abs(np.fft.ifft(table[0]))
    index = np.arange(2 * steps.n)
    apart = np.minimum(index, 2 * steps.n - index) * steps.dz
    reach = (parabolic.BOUNDED_WINDOW + parabolic.BOUNDED_SMOOTH) * steps.thickness
    assert reach == steps.thickness
    assert kernel[apart > reach].max() < 1e-12 * kernel.max()
