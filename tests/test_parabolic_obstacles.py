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


def test_a_roof_s_wind_effect_against_the_full_wave_solution() -> None:
    """A rigid block 40 to 52 m from a source 1 m up, its roof 6 m up, over rigid
    ground, at 125 Hz: the wind's effect on the level 80 and 150 m away, 1.5 m up, is
    within 0.2 dB of a full-wave solution's (mode matching in range, Evans 1983, with
    the same roof height; 0.15 dB here), where two zero-field screens in its place are
    0.28 dB off."""
    f = F[1]
    ranges, heights = np.array([80.0, 150.0]), np.array([1.5, 1.5])
    roof = 6.0087  # the full-wave solution's roof, on its grid
    full_wave = {5.0: np.array([-0.06, 1.08]), -5.0: np.array([-0.10, -2.15])}

    def level(u, obstacles):
        p = parabolic.march(f, log_wind(u), None, 1.0, ranges, heights, 30.0, obstacles=obstacles)
        return band_db(p, ranges, heights)

    block = [(40.0, 52.0, roof)]
    for u, expected in full_wave.items():
        effect = level(u, block) - level(0.0, block)
        np.testing.assert_allclose(effect, expected, atol=0.2)
    # Below the roof the field behind the front face is gone, and comes back behind it.
    over = parabolic.march(
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
