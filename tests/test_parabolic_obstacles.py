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


@pytest.mark.parametrize("band", [0, 1, 2, 3, 4, 5])
def test_a_bounded_step_s_table_is_the_plain_step_s_for_sound_that_goes_gently(band) -> None:
    """Under the absorbing layer of a run with its top 30 m up, a full bounded step's
    table differs from the plain step's by less than 1e-5 for waves up to 30 degrees
    from the horizontal, 1e-4 up to 45 degrees and 2e-4 up to the steepest that the step
    takes whole (4e-6, 4e-5 and 9e-5 at most): 51 degrees in the bands from 63 Hz to
    1 kHz and 55 at 2 kHz. No entry is larger than 1.05 (1.0495: the waves some 65 to 80
    degrees up, of which the window lets a part through)."""
    f = F[band]
    steps = parabolic._Steps(
        f, still_air, None, 1.0, 30.0, None, None, parabolic.STEPS_PER_WAVELENGTH
    )
    table = steps.propagator(steps.step, steps.n, 0.0, None)
    plain = parabolic._exp_mirrored(1j * steps.step * (steps.vertical - steps.ka[:, None]))
    sine = np.abs(steps.kz)[None, :] / steps.ka[:, None]
    off = np.abs(table - plain)
    steepest = max(d for d in range(30, 90) if steps.takes(math.tan(math.radians(d)), steps.step))
    assert steepest == (55 if band == 5 else 51)
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
