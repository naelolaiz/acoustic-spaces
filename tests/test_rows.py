"""Sound over rows of barriers and buildings: the march over the whole row plus what
the rigid edges add (:mod:`acoustic_spaces.engine.rows`)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.engine import parabolic, rows
from acoustic_spaces.materials.ground import spherical_reflection, surface_impedance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import edge_diffraction  # noqa: E402

C = edge_diffraction.C
#: (4, 8) the frequencies of the 63 to 500 Hz octave bands the validation's references
#: are taken at.
F = (edge_diffraction.K * C / (2.0 * math.pi)).reshape(len(edge_diffraction.FREQUENCIES), -1)[:4]
#: Rows without a ground stand this high (m) on a rigid one in the 500 Hz band, and as
#: many wavelengths up in the others: the ground's reflection would come up to the tops
#: so nearly straight from below (88.6 degrees from an end 15 m from the first of them)
#: that the solver leaves it out.
LIFT = 300.0


def still(z):
    return np.full(np.shape(np.asarray(z, dtype=float)), C)


def wind(z):
    return C + 0.5 * np.log1p(np.maximum(np.asarray(z, dtype=float), 0.0) / 0.05)


def level(p: np.ndarray, distance: float) -> float:
    """The band level (dB) of the pressures ``p`` over free field ``distance`` m away."""
    return 10.0 * math.log10(float(np.mean(np.abs(p) ** 2)) * distance**2)


def over(obstacles, source, listener, lift: float = LIFT, ground=None, bands=F) -> list[rows.Row]:
    """The row's sound in the four bands (or in ``bands``) for a ``source`` and a
    ``listener`` (x, z), the ``obstacles`` (near x, far x, top z) lifted ``lift`` m
    with them."""
    reach = listener[0] - source[0]
    found = []
    for f in bands:
        up = lift * F[-1].min() / f.min()
        standing = [(a - source[0], b - source[0], up + h) for a, b, h in obstacles]
        ends = (up + source[1], reach, up + listener[1])
        # Lifted, the ground's reflection is too steep for the march, and is left out.
        steep = "leave out" if lift else "raise"
        found.append(rows.over_row(f, C, ground, *ends, standing, steep_reflections=steep))
    return found


def levels(found: list[rows.Row], source, listener, part: str = "pressure") -> np.ndarray:
    distance = math.hypot(listener[0] - source[0], listener[1] - source[1])
    return np.array([level(getattr(row, part), distance) for row in found])


# -- the march's field at the tops -----------------------------------------------------


@pytest.mark.parametrize("speed, shared", [(still, False), (wind, False), (still, True)])
def test_the_march_gives_the_field_at_each_top_before_the_obstacle_changes_it(
    speed, shared
) -> None:
    """With or without a trunk shared by the runs."""
    f = F[2]
    obstacles = [(20.0, 20.0, 3.0), (40.0, 52.0, 6.0)]
    ranges, heights = np.array([70.0, 30.0]), np.array([1.5, 4.0])
    common = dict(refracting=False, independent=True, trunk=parabolic.Trunk() if shared else None)
    found: list = []
    got = parabolic.march(
        f, speed, None, 1.0, ranges, heights, 30.0, obstacles=obstacles, edges=found, **common
    )
    plain = parabolic.march(
        f, speed, None, 1.0, ranges, heights, 30.0, obstacles=obstacles, **common
    )
    assert np.array_equal(got, plain)
    assert [(at, what, top) for at, what, top, _ in found] == [
        (20.0, "screen", 3.0),
        (40.0, "roof", 6.0),
        (52.0, "behind", 6.0),
    ]

    def at(point, standing):
        return parabolic.march(
            f, speed, None, 1.0, point[:1], point[1:], 30.0, obstacles=standing, **common
        )[:, 0]

    # The screen's top and the near corner without them, the far corner on the roof.
    np.testing.assert_allclose(found[0][3], at(np.array([20.0, 3.0]), []), rtol=1e-12)
    np.testing.assert_allclose(found[1][3], at(np.array([40.0, 6.0]), obstacles[:1]), rtol=1e-12)
    longer = [obstacles[0], (40.0, 60.0, 6.0)]
    np.testing.assert_allclose(found[2][3], at(np.array([52.0, 6.0]), longer), rtol=1e-12)
    assert all(np.all(np.abs(value) > 0.0) for *_, value in found)
    # Of a roof under the source, the far corner alone: the near one is behind it.
    under = [(-5.0, 10.0, 6.0), (20.0, 20.0, 9.0)]
    found = []
    parabolic.march(
        f, speed, None, 8.0, ranges, heights, 30.0, obstacles=under, edges=found, **common
    )
    assert [(at, what, top) for at, what, top, _ in found] == [
        (10.0, "behind", 6.0),
        (20.0, "screen", 9.0),
    ]
    assert all(np.all(np.isfinite(value)) for *_, value in found)


def test_the_field_at_the_tops_needs_the_narrow_angle_march() -> None:
    for refracting in (None, True):
        with pytest.raises(ValueError, match="refracting=False"):
            parabolic.march(
                F[1],
                still,
                None,
                1.0,
                np.array([30.0]),
                np.array([1.5]),
                30.0,
                obstacles=[(20.0, 20.0, 3.0)],
                refracting=refracting,
                edges=[],
            )


# -- a march from a given field --------------------------------------------------------


def point_source(f: np.ndarray, zs: float):
    """``field(r, z)``: (F, n) the pressure of a point source ``zs`` m above a rigid
    ground, at the heights ``z`` r m along the ground from it."""
    k = 2.0 * math.pi * f[:, None] / C

    def field(r: float, z) -> np.ndarray:
        z = np.atleast_1d(np.asarray(z, dtype=float))
        direct, image = np.hypot(r, z - zs), np.hypot(r, z + zs)
        return np.exp(1j * k * direct) / direct + np.exp(1j * k * image) / image

    return field


def test_a_march_from_a_point_source_s_field_gives_that_field_further_on() -> None:
    """From the exact field 10 m from a source 30 m above a rigid ground, the march is
    within 0.35 dB of the exact field 40 and 100 m away, up to 37 degrees above and
    below the source (between two of its heights it takes the straight line through
    them). The march from its own starting field is as close where the reflection off
    the ground comes up less steeply than 40 degrees, and more than 1.3 dB off (up to
    2.5) where it comes up more steeply."""
    f, zs = F[3], 30.0
    field = point_source(f, zs)
    ranges = np.array([40.0] * 6 + [100.0] * 3)
    heights = np.array([30.0, 40.0, 50.0, 60.0, 10.0, 2.0, 30.0, 60.0, 5.0])
    exact = np.stack([field(r, h)[:, 0] for r, h in zip(ranges, heights, strict=True)], axis=1)
    common = (f, still, None, zs, ranges, heights, 80.0)
    given = parabolic.march(*common, refracting=False, start=(10.0, lambda z: field(10.0, z)))
    usual = parabolic.march(*common, refracting=False)

    def off(p: np.ndarray) -> np.ndarray:
        return np.max(np.abs(20.0 * np.log10(np.abs(p) / np.abs(exact))), axis=0)

    assert np.all(off(given) < 0.35)
    steep = (zs + heights) / ranges > math.tan(math.radians(40.0))
    assert np.all(off(usual)[steep] > 1.3) and np.all(off(usual)[~steep] < 0.35)
    assert steep.sum() == 6


def test_a_march_from_a_given_field_meets_what_stands_at_its_start_and_past_it() -> None:
    f = F[2]
    field = point_source(f, 2.0)

    def run(obstacles, found=None) -> np.ndarray:
        return parabolic.march(
            f,
            still,
            None,
            2.0,
            np.array([40.0]),
            np.array([3.0]),
            40.0,
            obstacles=obstacles,
            refracting=False,
            edges=found,
            start=(10.0, lambda z: field(10.0, z)),
        )

    # What stands before the start does nothing.
    past = [(20.0, 20.0, 4.0)]
    assert np.array_equal(run([(5.0, 5.0, 30.0), *past]), run(past))
    # The start lies beyond the source, and no point before it.
    for begin, ranges, why in (
        (0.0, [40.0], "begins beyond the source"),
        (-1.0, [40.0], "begins beyond the source"),
        (10.0, [40.0, 9.9], "no point before its start"),
    ):
        with pytest.raises(ValueError, match=why):
            parabolic.march(
                f,
                still,
                None,
                2.0,
                np.array(ranges),
                np.full(len(ranges), 3.0),
                40.0,
                refracting=False,
                start=(begin, lambda z: field(10.0, z)),
            )
    at_it = parabolic.march(
        f,
        still,
        None,
        2.0,
        np.array([10.0]),
        np.array([3.0]),
        40.0,
        refracting=False,
        start=(10.0, lambda z: field(10.0, z)),
    )
    np.testing.assert_allclose(at_it, field(10.0, 3.0), rtol=2e-3)
    # A screen at the start is met at once, the given field at its top.
    found: list = []
    behind_it = run([(10.0, 10.0, 3.0), *past], found)
    assert [(at, what, top) for at, what, top, _ in found] == [
        (10.0, "screen", 3.0),
        (20.0, "screen", 4.0),
    ]
    np.testing.assert_allclose(found[0][3], field(10.0, 3.0)[:, 0], rtol=2e-3)
    assert np.all(np.abs(behind_it / run(past) - 1.0) > 0.1)
    # A building over the start keeps its roof under the field: the march of one that
    # begins at the start, without its near corner.
    over_it: list = []
    from_it: list = []
    assert np.array_equal(run([(4.0, 15.0, 3.0)], over_it), run([(10.0, 15.0, 3.0)], from_it))
    assert [(at, what) for at, what, *_ in over_it] == [(15.0, "behind")]
    assert [(at, what) for at, what, *_ in from_it] == [(10.0, "roof"), (15.0, "behind")]


def test_a_given_start_and_a_thicker_layer_need_the_narrow_angle_march() -> None:
    f = F[2]
    field = point_source(f, 1.0)
    for refracting in (None, True):
        for more, match in (
            ({"start": (5.0, lambda z: field(5.0, z))}, "a march from a given field"),
            ({"layer": 200.0}, "a thicker absorbing layer"),
        ):
            with pytest.raises(ValueError, match=match):
                parabolic.march(
                    f,
                    still,
                    None,
                    1.0,
                    np.array([30.0]),
                    np.array([1.5]),
                    30.0,
                    refracting=refracting,
                    **more,
                )


def test_a_thicker_absorbing_layer_takes_the_waves_that_rise_steeply() -> None:
    """A source 2 m up on a rigid ground, 1 m before a screen 10 m high: the field at
    the screen rises at 83 degrees and more. Under the usual layer, 50 wavelengths, it
    comes back down from the top of the grid, and the field 5 m behind the screen is
    not that under a layer of 400 m by its own size; 30 m behind the screen what came
    back has died out."""
    f = F[3]
    assert parabolic.layer_thickness(f, C) == 50.0 * C / f.min()
    assert parabolic.layer_thickness(np.array([4000.0, 8000.0]), C) == parabolic.LAYER_MIN_M
    field = point_source(f, 2.0)

    def run(layer, start, obstacles, ranges, heights) -> np.ndarray:
        return parabolic.march(
            f,
            still,
            None,
            2.0,
            np.array(ranges),
            np.array(heights),
            30.0,
            obstacles=obstacles,
            refracting=False,
            start=start,
            layer=layer,
        )

    close = ((1.0, lambda z: field(1.0, z)), [(1.0, 1.0, 10.0)], [6.0, 31.0], [9.0, 9.0])
    usual, thick = run(None, *close), run(400.0, *close)
    change = np.max(np.abs(usual / thick - 1.0), axis=0)
    assert change[0] > 0.5 and change[1] < 0.01
    # A layer no thicker than the usual is the usual one.
    assert np.array_equal(run(1.0, *close), usual)
    # A march without steep waves is the same under either, to a part in a thousand.
    far = (None, [(20.0, 20.0, 4.0)], [50.0, 50.0], [2.0, 8.0])
    np.testing.assert_allclose(run(200.0, *far), run(None, *far), rtol=1e-3)


# -- against closed forms and the exact edge integral ----------------------------------


@pytest.mark.parametrize("count", [1, 2, 3])
def test_screens_level_with_both_ends_let_through_one_over_their_number_plus_one(count) -> None:
    """The march alone, over screens without a reflection 5 m apart with the source and
    the listener as far before and past them, all at one height: 1 / (N + 1) of the
    free field (Vogler's multiple knife edges, in the Fresnel approximation)."""
    reach = 5.0 * (count + 1)
    screens = [(5.0 * (i + 1), 5.0 * (i + 1), 0.0) for i in range(count)]
    for row in over(screens, (0.0, 0.0), (reach, 0.0))[1:]:
        np.testing.assert_allclose(np.abs(row.marched) * reach, 1.0 / (count + 1), rtol=0.02)
        # A rigid top sends on more.
        assert np.all(np.abs(row.pressure) > np.abs(row.marched))


def over_one_screen(source, listener, heights_from: float = 0.0) -> np.ndarray:
    """(4,) the level (dB re free field) in the four bands behind a rigid half-plane
    with its top at x = 0, z = ``heights_from``. In free space the exact solution: the
    direct sound, where the listener sees the source, and the Biot-Tolstoy-Medwin
    integral over the edge. With ``heights_from`` above 0, on a rigid ground at z = 0:
    the four paths over the top, by the images of the source and the listener in the
    ground, without the sound that goes from the top down the screen to the ground
    and back."""
    edge = edge_diffraction.Edge([0, 0, heights_from], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    signs = (1.0, -1.0) if heights_from > 0.0 else (1.0,)
    distance = math.hypot(listener[0] - source[0], listener[1] - source[1])
    out = []
    for band in 2.0 * math.pi * F / C:
        p = np.zeros(len(band), dtype=complex)
        for zs in (sign * source[1] for sign in signs):
            for zl in (sign * listener[1] for sign in signs):
                a, b = np.array([source[0], 0.0, zs]), np.array([listener[0], 0.0, zl])
                p += edge_diffraction.btm_first(edge, a, b, band)
                if zs + (zl - zs) * -source[0] / (listener[0] - source[0]) > heights_from:
                    way = float(np.linalg.norm(b - a))
                    p += np.exp(-1j * band * way) / way  # in sight over the top
        out.append(level(p, distance))
    return np.array(out)


@pytest.mark.parametrize(
    "zs, zr",
    [(0.01, 0.01), (0.5, 0.01), (0.5, -0.5), (2.0, -2.0), (0.01, -2.0), (-1.0, -1.0), (3.0, 1.0)],
)
def test_one_rigid_screen_against_the_exact_edge_integral(zs: float, zr: float) -> None:
    """A rigid half-plane, the source 10 m before it and the listener 15 m past it, up
    to 3 m above its top and 2 m below: within 0.2 dB of the direct sound plus the
    Biot-Tolstoy-Medwin integral over the edge, which the march alone, whose screen
    reflects nothing, misses by up to 1.5 dB."""
    source, listener = (-10.0, zs), (15.0, zr)
    exact = over_one_screen(source, listener)
    found = over([(0.0, 0.0, 0.0)], source, listener)
    np.testing.assert_allclose(levels(found, source, listener), exact, atol=0.2)
    if zs + (zr - zs) * 10.0 / 25.0 <= 0.5:
        assert np.max(np.abs(levels(found, source, listener, "marched") - exact)) > 0.5


@pytest.mark.parametrize(
    "source, listener",
    [
        ((-10.0, -8.0), (15.0, -22.0)),
        ((-10.0, -8.0), (15.0, 0.5)),
        ((-10.0, 2.0), (15.0, -22.0)),
        ((-5.0, -8.0), (15.0, -22.0)),
        ((-3.0, -8.0), (15.0, -3.0)),
        ((-10.0, -3.0), (3.0, -8.0)),
        ((-0.5, -8.0), (15.0, -1.0)),
    ],
)
def test_one_rigid_screen_far_below_its_top_against_the_exact_edge_integral(
    source, listener
) -> None:
    """A rigid half-plane, the source or the listener 39 to 86 degrees below its top (a
    source half a metre before a screen 8 m above it): within 0.1 dB of the exact
    solution in every band, where the march alone is 3 to 11 dB low. (With the top's
    place within the march's cells the level moves by up to 0.19 dB where the sound
    turns by 80 to 115 degrees; here the top lies a tenth of a cell from the edge of
    one, where the level is closest.)"""
    exact = over_one_screen(source, listener)
    found = over([(0.0, 0.0, 0.0)], source, listener)
    np.testing.assert_allclose(levels(found, source, listener), exact, atol=0.1)
    assert np.all(levels(found, source, listener, "marched") < exact - 2.9)


@pytest.mark.parametrize(
    "top, source, listener",
    [
        (6.0, (-10.0, 1.5), (15.0, 1.5)),
        (6.0, (-2.0, 1.5), (3.0, 1.5)),
        (6.0, (-1.0, 1.5), (15.0, 1.5)),
        (6.0, (-0.5, 0.5), (0.5, 0.5)),
        (20.0, (-30.0, 0.5), (1.0, 4.0)),
        (3.0, (-5.0, 0.0), (20.0, 1.5)),
        (10.0, (-20.0, 4.0), (40.0, 12.0)),
    ],
)
def test_one_rigid_screen_on_a_rigid_ground_against_the_exact_edge_integrals(
    top, source, listener
) -> None:
    """A rigid screen 3 to 20 m high on a rigid ground, the source and the listener
    from tens of metres off to half a metre from its foot (85 degrees below its top;
    their images in the ground steeper still): within 0.25 dB in every band of the
    four paths over its top, each by the exact edge integral (:func:`over_one_screen`),
    where the march alone is 1.7 to 28 dB low."""
    paths = over_one_screen(source, listener, top)
    reach = listener[0] - source[0]
    found = [
        rows.over_row(f, C, None, source[1], reach, listener[1], [(-source[0], -source[0], top)])
        for f in F
    ]
    np.testing.assert_allclose(levels(found, source, listener), paths, atol=0.25)
    assert np.all(levels(found, source, listener, "marched") < paths - 1.5)


def test_the_absorbing_layer_grows_for_what_a_march_s_start_needs() -> None:
    f = F[3]
    usual = parabolic.layer_thickness(f, C)
    by_slope = 1.5 * math.log(1.0 / rows.STEEP_LEFT) / parabolic.LAYER_STRENGTH
    longest = C / f.min()
    # The usual layer takes waves that rise up to 4.4 m a metre, the thickest 35.
    assert usual / by_slope == pytest.approx(4.4, abs=0.05)
    assert rows.LAYER_MOST == 8.0
    onwards = (50.0, 0.0)

    def layer(first: float, top: float, z0: float, left_out: bool = False) -> float:
        strings = [[(0.0, z), (first, top), onwards] for z in (z0, -z0)]
        return rows._layer(f, C, first, top, strings, left_out)

    def longer(first: float, top: float, z0: float, height: float, to=onwards) -> float:
        """How much longer (m) the way from the end to the point ``to`` is over the
        start at ``height`` than over the top."""

        def way(z: float) -> float:
            return math.hypot(first, z - z0) + math.hypot(to[0] - first, to[1] - z)

        return way(height) - way(top)

    # The usual layer for a start 10 m from an end half a metre below the top.
    assert layer(10.0, 1.0, 0.5) == usual
    # An end half a metre up, a metre before a wall 8 m high. Its image in the ground
    # needs the start higher than the end does: from the top up to where the way over
    # the start is half a wavelength longer than over the top (the upper end of the
    # string's first Fresnel zone there), and round that three radii of the zone of a
    # string straight across, sqrt(wavelength times the start's distance).
    around = 3.0 * math.sqrt(longest * 1.0)
    thick = layer(1.0, 8.0, 0.5)
    highest = (thick / by_slope - around) * 1.0 - 0.5
    assert longer(1.0, 8.0, -0.5, highest) == pytest.approx(0.5 * longest, rel=1e-6)
    assert 8.3 < highest < 8.5 and thick > 2.5 * usual
    assert rows._layer(f, C, 1.0, 8.0, [[(0.0, 0.5), (1.0, 8.0), onwards]]) < thick
    # The way on is the string's as far as its next bend: up to a top 20 m high 2 m
    # past the wall, the zone reaches 9 m above the wall's top.
    bent = [(0.0, -0.5), (1.0, 8.0), (3.0, 20.0), onwards]
    highest = (rows._layer(f, C, 1.0, 8.0, [bent]) / by_slope - around) * 1.0 - 0.5
    assert longer(1.0, 8.0, -0.5, highest, (3.0, 20.0)) == pytest.approx(0.5 * longest, rel=1e-6)
    assert 17.0 < highest < 17.6
    # An end in sight of the other, 3 m above the top a metre before it: the start is
    # kept from the top up, whose edge the end lights, 3 m down from the end.
    in_sight = [(0.0, 4.0), onwards]
    assert rows._layer(f, C, 1.0, 1.0, [in_sight]) == pytest.approx((3.0 + around) * by_slope)
    # A wavelength round what the start needs at least: a start 5 cm from the end.
    assert longest > 3.0 * math.sqrt(longest * 0.05)
    level = [[(0.0, 1.0), (0.05, 1.0), onwards]]
    close = rows._layer(f, C, 0.05, 1.0, level)
    highest = (close / by_slope - longest / 0.05) * 0.05 + 1.0
    assert longer(0.05, 1.0, 1.0, highest) == pytest.approx(0.5 * longest, rel=1e-6)
    assert 0.75 * rows.LAYER_MOST * usual < close < rows.LAYER_MOST * usual
    # More than the thickest is out of reach: a start 3 cm from an end level with the
    # top, a top almost straight above an end close to it or far from it.
    with pytest.raises(rows.OutOfReach, match="0.03 m before .* would need 10.5 times"):
        layer(0.03, 1.0, 1.0, left_out=True)
    with pytest.raises(rows.OutOfReach, match="0.25 m before .* way on needs up to 88.3 degrees"):
        layer(0.25, 9.0, 1.0, left_out=True)
    with pytest.raises(rows.OutOfReach, match="10 m before .* way on needs up to 88.6 degrees"):
        layer(10.0, 400.0, 1.0, left_out=True)
    # So is the image's string, but where that is to be left out: it gets the thickest
    # layer, or none where the top is itself too steep from the image for that.
    steep = "reflection off the ground needs up to 88.3 degrees.* need 8.09 times.* takes 8"
    with pytest.raises(rows.OutOfReach, match=steep):
        layer(1.0, 31.5, 1.0)
    assert layer(1.0, 31.5, 1.0, left_out=True) == rows.LAYER_MOST * usual
    with pytest.raises(rows.OutOfReach, match="off the ground needs up to 89.3 degrees steep"):
        layer(10.0, 400.0, 399.5)
    assert layer(10.0, 400.0, 399.5, left_out=True) == usual


def test_the_first_fresnel_zone_on_a_march_s_start() -> None:
    """Where the way over a march's start is at most half a wavelength longer than the
    shortest one (:func:`rows._zone`): for a string straight across the start, midway
    between two points 2 d apart, up to sqrt(d wavelength / 2 + wavelength^2 / 16)
    above it; for a string almost along the start, from a point 5 m below to one 8 m
    above (or the other way round) 2 cm apart, up to a quarter of a wavelength past
    the higher point; and never below the top there, from which the start is kept."""
    excess = 0.45
    for d in (0.05, 1.0, 10.0):
        closed = math.sqrt(d * excess + excess**2 / 4.0)
        for up in (0.0, 3.0):
            zone = rows._zone(d, -5.0, (0.0, up), (2.0 * d, up), excess)
            assert zone == pytest.approx(up + closed)
    for start, onwards in (((0.0, -5.0), (0.02, 8.0)), ((0.0, 8.0), (0.02, -5.0))):
        assert rows._zone(0.01, -9.0, start, onwards, excess) == pytest.approx(
            8.0 + 0.5 * excess, abs=1e-3
        )
    # From the top up: the ways over the start from 2 m above the string on.
    above = rows._zone(1.0, 2.0, (0.0, 0.0), (2.0, 0.0), excess)
    assert 2.0 * math.hypot(1.0, above) - 2.0 * math.hypot(1.0, 2.0) == pytest.approx(excess)
    assert 2.0 < above < 2.0 + math.sqrt(1.0 * excess + excess**2 / 4.0)


def test_each_march_gets_the_layer_its_start_needs(monkeypatch) -> None:
    """A source half a metre before a wall 8 m high on a rigid ground and a listener
    15 m behind it: the start of the march from the source needs a layer more than
    four times the usual, for the way up to the top from the source's image in the
    ground; that of the march from the listener the usual. The way up from the
    source is steeper than 30 degrees, so both get a grid eight times as high above
    the lower end, by a layer thicker still."""
    f = F[3]
    march, given = parabolic.march, []

    def spy(*args, **more):
        given.append(more["layer"])
        return march(*args, **more)

    monkeypatch.setattr(parabolic, "march", spy)
    rows.over_row(f, C, None, 0.5, 15.5, 1.5, [(0.5, 0.5, 8.0)])
    usual = parabolic.layer_thickness(f, C)
    strings = [[(0.0, 0.5), (0.5, 8.0), (15.5, 1.5)], [(0.0, -0.5), (0.5, 8.0), (15.5, 1.5)]]
    needs = rows._layer(f, C, 0.5, 8.0, strings)
    assert needs > 4.0 * usual
    air = rows.layer_top(0.5, 1.5, 15.5, [(0.5, 0.5, 8.0)]) - 0.5
    assert (rows.STEEP_END_DEG, rows.GRID_TIMES) == (30.0, 8.0)
    assert given == pytest.approx([8.0 * (air + needs) - air, 8.0 * (air + usual) - air])


def test_a_steep_way_at_an_end_gets_a_higher_grid(monkeypatch) -> None:
    """A wall 20 m high on a rigid ground, the source 5 m before it half a metre up and
    the listener 5 m past it 4 m up: 46 dB below the free field in the 500 Hz band,
    and within 0.3 dB of the four paths over the top with the marches' grid eight
    times as high above the lower end; without, 4.6 dB off them. With the source
    1.5 m up, within 0.4 dB, and 0.7 dB off them with a grid four times as high.
    Where neither end's way to its first bend is steeper than 30 degrees (28 degrees
    up to a wall 6 m high from a source 1.5 m up, 8.5 m before it), the layer stays
    as it is; where the layer starts higher than 20 m above the wall, it is made
    thicker by less; and a march that begins at its end gets the grid too."""
    f = F[3]
    march, given = parabolic.march, []

    def spy(*args, **more):
        given.append(more["layer"])
        return march(*args, **more)

    monkeypatch.setattr(parabolic, "march", spy)
    source, listener, wall = (-5.0, 0.5), (5.0, 4.0), [(5.0, 5.0, 20.0)]
    paths = over_one_screen(source, listener, 20.0)[3]
    assert paths == pytest.approx(-46.2, abs=0.1)

    def off(height: float = 0.5) -> float:
        found = [rows.over_row(f, C, None, height, 10.0, 4.0, wall)]
        at = (source[0], height)
        return abs(levels(found, at, listener)[0] - over_one_screen(at, listener, 20.0)[3])

    assert off() < 0.3
    assert off(1.5) < 0.4
    usual = parabolic.layer_thickness(f, C)
    assert min(given) > 8.0 * usual
    monkeypatch.setattr(rows, "GRID_TIMES", 1.0)
    assert off() > 4.0
    monkeypatch.setattr(rows, "GRID_TIMES", 4.0)
    assert off(1.5) > 0.6
    monkeypatch.undo()
    monkeypatch.setattr(parabolic, "march", spy)
    given.clear()
    rows.over_row(f, C, None, 1.5, 25.0, 1.5, [(8.5, 8.5, 6.0)])
    assert given == [usual, usual]
    rows.over_row(f, C, None, 1.5, 25.0, 1.5, [(7.5, 7.5, 6.0)])
    assert min(given[2:]) > 8.0 * usual
    # The grid reaches 8 times as high as with the layer 20 m above the wall: a layer
    # that starts higher (a long way, or as it is given) is made thicker by less.
    row, air = [(18.0, 18.0, 6.0)], 6.0 + 20.0 - 1.5
    by_height = {}
    for top in (None, 300.0, 750.0):
        given.clear()
        found = rows.over_row(f, C, None, 1.5, 20.0, 1.5, row, top=top)
        by_height[top] = (list(given), level(found.pressure, 20.0))
    # The listener's start, 66 degrees below the top, needs more than the usual layer.
    strings = [[(0.0, 1.5), (2.0, 6.0), (20.0, 1.5)], [(0.0, -1.5), (2.0, 6.0), (20.0, 1.5)]]
    needs = rows._layer(f, C, 2.0, 6.0, strings)
    assert needs > 1.3 * usual
    least = np.array([usual, needs])
    assert by_height[None][0] == pytest.approx(8.0 * (air + least) - air)
    assert by_height[300.0][0] == pytest.approx(8.0 * (air + least) - 298.5)
    assert by_height[750.0][0] == pytest.approx(least)
    for top in (300.0, 750.0):
        assert abs(by_height[top][1] - by_height[None][1]) < 0.05
    # A long way, whose layer starts 0.15 of its 400 m up: the grid is no higher for it.
    low = F[0]
    given.clear()
    rows.over_row(low, C, None, 1.5, 400.0, 1.5, [(2.0, 2.0, 6.0)])
    thick = 8.0 * (air + parabolic.layer_thickness(low, C)) - (0.15 * 400.0 - 1.5)
    assert given == pytest.approx([thick, thick])

    # A march that begins at its end (the ground changes before the wall, 3 m high and
    # 10 m on) gets it as the one from the wall, whose way up is 72 degrees steep.
    def changes(r):
        return None if r < 5.0 else np.full(len(f), 1e9 + 0j)

    row = [(10.0, 10.0, 3.0)]
    assert rows._first(f, changes, 11.0, row) == (None, 0.0)
    given.clear()
    rows.over_row(f, C, changes, 1.0, 11.0, 0.0, row)
    assert given[0] == pytest.approx(8.0 * (23.0 + usual) - 23.0)
    assert given[1] > 8.0 * usual


def test_a_start_the_thickest_layer_cannot_hold_is_out_of_reach(monkeypatch) -> None:
    """A source 20 cm before a screen 8 m above it, 88.6 degrees up: out of reach in
    the 500 Hz band; in the three bands below, whose layers are thicker, within 0.2 dB
    of the exact solution, as in every band from 30 cm before the screen (87.9
    degrees). A source 10 cm before a screen and 3 m above its top is out of reach in
    the 500 Hz band too: its march's start is kept from the top up, 88.1 degrees down
    from the source. A source 2 cm before a screen is out of reach in every band, and
    so is a listener 1 cm past one, before the march from the source has run."""
    screen, listener = [(0.0, 0.0, 0.0)], (15.0, -1.0)
    for source, steep in (((-0.2, -8.0), "88.6"), ((-0.1, 3.0), "88.1")):
        refused = f"{-source[0]} m before .* way on needs up to {steep} degrees steep"
        with pytest.raises(rows.OutOfReach, match=refused):
            over(screen, source, listener, bands=F[3:])
        got = levels(over(screen, source, listener, bands=F[:3]), source, listener)
        np.testing.assert_allclose(got, over_one_screen(source, listener)[:3], atol=0.2)
    source = (-0.3, -8.0)
    got = levels(over(screen, source, listener), source, listener)
    np.testing.assert_allclose(got, over_one_screen(source, listener), atol=0.2)
    marched = []
    monkeypatch.setattr(parabolic, "march", lambda *given, **more: marched.append(given))
    for band in F:
        with pytest.raises(rows.OutOfReach, match="0.02 m before its first obstacle.* takes 8"):
            over(screen, (-0.02, 0.3), listener, bands=[band])
        with pytest.raises(rows.OutOfReach, match="0.01 m before its first obstacle.* takes 8"):
            over(screen, (-10.0, 0.5), (0.01, 1.0), bands=[band])
    # Both ends are checked before either march runs.
    assert marched == []


@pytest.mark.parametrize(
    "source, listener, bands",
    [((-0.3, -8.0), (0.3, 5.0), 3), ((-3.0, 5.0), (0.3, -3.0), 4), ((-0.3, 5.0), (0.3, -8.0), 3)],
)
def test_a_way_almost_along_a_march_s_start_takes_the_start_up_to_its_fresnel_zone(
    source, listener, bands
) -> None:
    """A rigid half-plane, the source 30 cm before it and 8 m below its top, the
    listener 30 cm past it and 5 m above (the same swapped; a source 3 m before it and
    5 m above, the listener 30 cm past it and 3 m below): the sound goes up or down
    almost along the start of a march, where the string's first Fresnel zone reaches
    metres past the top. Within 0.2 dB of the exact solution in the bands in reach;
    with the start kept three Fresnel radii of a string straight across round the
    string alone, the levels would be up to 3.6 dB low."""
    exact = over_one_screen(source, listener)[:bands]
    found = over([(0.0, 0.0, 0.0)], source, listener, bands=F[:bands])
    np.testing.assert_allclose(levels(found, source, listener), exact, atol=0.2)
    for f in F[bands:]:
        with pytest.raises(rows.OutOfReach, match="0.3 m before .* needs up to 8[78]"):
            over([(0.0, 0.0, 0.0)], source, listener, bands=[f])


def test_rows_lifted_high_above_the_ground_stand_in_free_space() -> None:
    """The ground's reflection would reach the first top at more than 88.6 degrees from
    either end, too steep for the thickest layer, and is left out where the solver is
    asked to: the row twice as high above the ground has the same level within 0.02
    dB."""
    for row, source, listener in (
        (ROOF_AND_SCREEN, (-9.99, 0.01), (15.01, -1.99)),
        ([(0.0, 0.0, 0.0)], (-10.0, 0.5), (15.0, -1.0)),
    ):
        once = levels(over(row, source, listener), source, listener)
        twice = levels(over(row, source, listener, lift=2.0 * LIFT), source, listener)
        np.testing.assert_allclose(twice, once, atol=0.02)
    for f in F:
        up = LIFT * F[-1].min() / f.min()
        strings = [[(0.0, up), (15.0, up), (25.0, up)], [(0.0, -up), (15.0, up), (25.0, up)]]
        assert rows._layer(f, C, 15.0, up, strings, True) == parabolic.layer_thickness(f, C)
    # Unless asked for, a reflection too steep for the march is not left out.
    lifted = (F[3], C, None, LIFT + 0.5, 25.0, LIFT - 1.0, [(10.0, 10.0, LIFT)])
    with pytest.raises(rows.OutOfReach, match="off the ground needs up to 89.0 degrees steep"):
        rows.over_row(*lifted)
    with pytest.raises(ValueError, match="steep_reflections"):
        rows.over_row(*lifted, steep_reflections="keep")


@pytest.mark.parametrize(
    "source, listener, bands",
    [
        ((-0.2, 0.1), (15.0, -1.0), 4),
        ((-0.2, 1.0), (15.0, -1.0), 4),
        ((-0.1, 3.0), (15.0, 2.0), 3),
        ((-0.05, 0.5), (15.0, -2.0), 4),
        ((-10.0, 0.5), (0.1, 3.0), 3),
        ((-1.0, 0.5), (15.0, -3.0), 4),
        ((-1.0, 3.0), (15.0, -1.0), 4),
    ],
)
def test_an_end_close_beside_a_screen_against_the_exact_edge_integral(
    source, listener, bands
) -> None:
    """A source or a listener 5 cm to 1 m before a rigid half-plane, far closer than a
    wavelength, 10 cm to 3 m above its top: within 0.2 dB of the exact solution in
    every band in reach (10 cm before the screen and 3 m above its top, the 500 Hz
    band is not). The march begins at the screen from the field in two dimensions that
    the end has there, a Hankel function, which so close to the end is not yet
    exp(ikR) / sqrt(R), and takes it a wavelength round what the start needs at
    least."""
    exact = over_one_screen(source, listener)[:bands]
    found = over([(0.0, 0.0, 0.0)], source, listener, bands=F[:bands])
    np.testing.assert_allclose(levels(found, source, listener), exact, atol=0.2)


@pytest.mark.parametrize(
    "source, listener",
    [
        ((-0.05, -0.15), (15.0, -1.0)),
        ((-0.05, -0.05), (15.0, -1.0)),
        ((-0.05, 0.1), (15.0, -1.0)),
        ((-0.1, -0.2), (15.0, 0.5)),
        ((-10.0, 0.5), (0.05, -0.15)),
    ],
)
def test_an_end_within_centimetres_of_a_screen_s_top(source, listener) -> None:
    """A source or a listener 5 or 10 cm before a rigid half-plane, from 20 cm below
    its top to 10 cm above it: within 0.45 dB of the exact solution in every band
    (0.39 dB low in the 63 Hz band 15 cm below the top, 0.25 dB high 10 cm above
    it), with the top where the lift puts it, 0.59 of a cell above a height of the
    march's grid. At other places of the top within a cell such ends are up to
    0.49 dB low and 0.34 dB high."""
    exact = over_one_screen(source, listener)
    found = over([(0.0, 0.0, 0.0)], source, listener)
    np.testing.assert_allclose(levels(found, source, listener), exact, atol=0.45)


def test_the_march_that_arrives_close_past_a_top_gets_finer_cells(monkeypatch) -> None:
    """A source 5 cm before a screen and 15 cm below its top, 0.4 of a cell of the 63 Hz
    band's march from the top: the march that arrives at it gets cells of half its
    distance to the top, 5 times as many a wavelength as usual, the march from it the
    usual ones; level with the top, where it would be 16 times as many, ``FINEST`` 8
    times what the march is asked for. With the usual cells for both, the level is
    0.39 dB above the exact solution, and 0.81 dB above it with the top three
    eighths of a cell higher; with the finer ones 0.39 and 0.48 dB below it. It is
    the top nearest to the end that counts, not the last on the way."""
    f, usual = F[0], parabolic.STEPS_PER_WAVELENGTH
    cell = C / f.max() / usual
    assert (rows.CELLS_TO_TOP, rows.FINEST) == (2.0, 8.0)
    source, listener, screen = (-0.05, -0.15), (15.0, -1.0), [(0.0, 0.0, 0.0)]
    assert math.hypot(*source) / cell == pytest.approx(0.4, abs=0.01)
    march, given = parabolic.march, []

    def spy(*args, **more):
        given.append(more["steps_per_wavelength"])
        return march(*args, **more)

    monkeypatch.setattr(parabolic, "march", spy)

    exact = over_one_screen(source, listener)[0]

    def off() -> list[float]:
        """The level less the exact solution's (dB), as the row stands and three
        eighths of a cell higher."""
        return [
            levels(over(screen, source, listener, lift=lift, bands=[f]), source, listener)[0]
            - exact
            for lift in (LIFT, LIFT + 0.375 * cell * f.min() / F[-1].min())
        ]

    low = off()
    assert -0.5 < min(low) and max(low) < -0.3 and abs(low[1] - low[0]) < 0.15
    fine = 2.0 * C / f.max() / math.hypot(*source)
    assert given == pytest.approx([usual, fine, usual, fine])
    assert fine / usual == pytest.approx(5.0, abs=0.1)
    given.clear()
    over(screen, (-0.05, 0.0), listener, bands=[f])
    over(screen, (-10.0, 0.5), (0.05, 0.0), bands=[f])
    assert given == [usual, 8.0 * usual, 8.0 * usual, usual]
    # Asked for twice the usual cells: those from the far end, and up to 8 times those.
    given.clear()
    up = LIFT * F[-1].min() / f.min()
    lifted = (f, C, None, up, 15.05, up - 1.0, [(0.05, 0.05, up)])
    rows.over_row(*lifted, steps_per_wavelength=2.0 * usual, steep_reflections="leave out")
    assert given == pytest.approx([2.0 * usual, 2.0 * C / f.max() / 0.05])
    assert 8.0 * usual < given[1] < 16.0 * usual
    # A lower screen stands between the listener and the screen it is close past.
    given.clear()
    over([(0.0, 0.0, 0.0), (0.1, 0.1, -1.0)], (-10.0, 0.0), (0.3, 0.0), bands=[f])
    assert given == pytest.approx([2.0 * C / f.max() / 0.3, usual])
    monkeypatch.setattr(rows, "FINEST", 1.0)
    high = off()
    assert min(high) > 0.3 and max(high) > 0.7 and abs(high[1] - high[0]) > 0.3


def test_ends_closer_over_the_tops_than_half_a_wavelength_are_out_of_reach() -> None:
    """The march's field in two dimensions becomes a point source's by the root of the
    length of the path over the tops, which holds far from the ends. A source and a
    listener on either side of a rigid half-plane, 0.55 of the wavelength of the
    band's lowest frequency from each other over its top (in sight of each other just
    over the top, and in its shadow): within 0.25 dB of the exact solution; 0.45 of
    it: out of reach. (A tenth of a wavelength apart, in sight far above the top, the
    level would be 0.5 dB low; a twentieth, 1.2 dB.)"""
    assert rows.STRING_WAVELENGTHS == 0.5
    screen = [(0.0, 0.0, 0.0)]
    for band, f in enumerate(F):
        longest = C / f.min()
        for along, up in ((0.5, 0.2), (0.4, -0.3)):
            # The path's length over the top, for one wavelength.
            path = 2.0 * math.hypot(along, min(up, 0.0)) * longest
            source = (-0.55 * along * longest, 0.55 * up * longest)
            listener = (-source[0], source[1])
            got = levels(over(screen, source, listener, bands=[f]), source, listener)[0]
            assert abs(got - over_one_screen(source, listener)[band]) < 0.25
            source = (-0.45 * along * longest, 0.45 * up * longest)
            refused = (
                f"ends {0.45 * path:.3g} m from each other over the tops: at {f.min():.0f} Hz "
                f"and above the solver takes none closer than {0.5 * longest:.3g} m"
            )
            with pytest.raises(rows.OutOfReach, match=refused):
                over(screen, source, (-source[0], source[1]), bands=[f])


def test_one_march_gives_the_direct_sound_and_the_reflection_each_by_its_own_distance() -> None:
    """A source 6 m above a rigid ground, a wall a millimetre high 10 m from it and a
    point 6 m up, 10 cm past the wall: the march from the source begins at the wall,
    from the source's field and its image's there, and gives the direct sound and the
    reflection off the ground, each weaker by its own distance (15.7 m to 10.1 m),
    within 0.1 dB in every band. With the image's wave weakened by the root of the
    direct path's length in place of its own, the level would be 0.6 to 0.7 dB too
    high."""
    zs, zr, wall, reach = 6.0, 6.0, 10.0, 10.1
    row = [(wall, wall, 0.001)]
    direct, image = math.hypot(reach, zr - zs), math.hypot(reach, zr + zs)
    top = rows.layer_top(zs, zr, reach, row)
    for f in F:
        k = 2.0 * math.pi * f / C
        exact = np.exp(1j * k * direct) / direct + np.exp(1j * k * image) / image
        way = rows._march_from(
            f, C, reach, top, parabolic.STEPS_PER_WAVELENGTH, None, zs, zr, row
        )()
        assert way.first == wall
        assert abs(level(way.pressure, direct) - level(exact, direct)) < 0.1
        np.testing.assert_allclose(way.pressure, exact, rtol=0.05)


# -- against a wave calculation --------------------------------------------------------

#: Rows (near x, far x, top z; a screen where far is near), a source and a listener
#: (x, z), and the level (dB re free field) in the 63 to 500 Hz octave bands of a
#: two-dimensional wave calculation turned into a point source's (finite differences in
#: time on a 2 cm grid, ``validation/rows.py``); then how far the row's level may be
#: from it (dB). The first rows stand in free space, the last on a rigid ground.
ROOF_AND_SCREEN = [(0.0, 9.0, 0.0), (10.01, 10.01, 0.0)]
TWO_ROOFS = [(0.0, 4.0, 0.0), (6.0, 10.0, 0.0)]
THREE_SCREENS = [(0.01, 0.01, 0.0), (5.01, 5.01, 0.0), (10.01, 10.01, 0.0)]
THREE_ROOFS = [(0.0, 2.0, 0.0), (4.0, 6.0, 0.0), (8.0, 10.0, 0.0)]
WAVE = [
    ([(0.0, 9.0, 0.0), (10.01, 10.01, -0.02)], 0.01, 0.01, [-6.24, -6.46, -6.70, -6.88], 0.3),
    ([(0.0, 9.0, 0.0), (10.01, 10.01, 0.02)], 0.01, 0.01, [-6.28, -6.53, -6.79, -7.02], 0.3),
    ([(0.0, 9.0, 0.0), (10.01, 10.01, 0.5)], 0.01, 0.01, [-6.80, -7.20, -7.94, -8.80], 0.3),
    (ROOF_AND_SCREEN, 2.01, -0.49, [-4.88, -4.53, -4.13, -3.59], 0.3),
    (TWO_ROOFS, 0.01, -0.49, [-7.46, -8.26, -9.10, -10.11], 0.3),
    (THREE_SCREENS, 0.01, -0.49, [-10.67, -11.44, -12.68, -13.98], 0.4),
    (THREE_ROOFS, 0.01, 0.01, [-7.64, -8.19, -8.46, -8.66], 0.3),
    ([(0.01, 0.01, 0.0), (10.01, 10.01, 0.0)], 0.01, 0.01, [-8.19, -8.75, -9.12, -9.40], 0.3),
    ([(0.0, 4.0, 0.0)], 0.51, 0.01, [-4.68, -4.61, -4.38, -3.91], 0.3),
    ([(6.0, 10.0, 0.0)], 0.01, 0.01, [-5.17, -5.41, -5.56, -5.70], 0.3),
    # The sound bends down by 22 degrees behind the row.
    (TWO_ROOFS, 2.01, -1.99, [-7.73, -8.61, -9.53, -10.45], 0.3),
    (ROOF_AND_SCREEN, 0.01, -1.99, [-10.04, -12.02, -14.37, -17.26], 0.3),
    ([(0.0, 10.0, 0.0)], 0.01, -1.99, [-8.79, -10.55, -12.62, -15.07], 0.3),
]
ON_GROUND = [
    ([(0.0, 4.0, 6.0), (6.0, 10.0, 6.0)], 6.01, 5.51, [-6.95, -7.51, -9.34, -9.80], 0.8),
    ([(0.0, 9.0, 6.0), (10.01, 10.01, 6.0)], 1.51, 6.01, [-11.54, -15.21, -13.61, -16.39], 0.8),
    ([(0.0, 10.0, 6.0)], 6.51, 6.01, [-4.46, -3.95, -4.25, -3.27], 0.5),
    ([(10.01, 10.01, 6.0)], 1.51, 5.51, [-3.50, -9.17, -10.98, -10.87], 0.6),
    # The sound comes up 24 degrees from 1.5 m above the ground, or goes down 42 degrees
    # to there, or both.
    ([(0.0, 10.0, 6.0)], 1.51, 6.01, [-10.31, -14.18, -12.42, -14.68], 0.3),
    ([(0.0, 10.0, 6.0)], 6.51, 1.51, [-14.89, -10.64, -15.18, -18.19], 0.3),
    ([(0.0, 10.0, 6.0)], 1.51, 1.51, [-20.29, -20.56, -23.94, -28.04], 0.4),
    ([(10.01, 10.01, 6.0)], 6.01, 1.51, [-19.34, -10.25, -13.02, -17.50], 0.7),
    ([(10.01, 10.01, 6.0)], 1.51, 1.51, [-13.01, -13.35, -15.54, -17.93], 0.6),
    ([(0.0, 4.0, 6.0), (6.0, 10.0, 6.0)], 1.51, 1.51, [-22.46, -22.04, -26.77, -30.24], 0.9),
]


@pytest.mark.parametrize("row, zs, zr, wave, within", WAVE)
def test_rows_in_free_space_against_a_wave_calculation(row, zs, zr, wave, within) -> None:
    source, listener = (-9.99, zs), (15.01, zr)
    found = over(row, source, listener)
    got = levels(found, source, listener)
    np.testing.assert_allclose(got, wave, atol=within)
    # The march alone, whose screens and walls reflect nothing, is further off.
    alone = levels(found, source, listener, "marched")
    assert np.max(np.abs(alone - wave)) > max(2.0 * np.max(np.abs(got - wave)), 0.5)


@pytest.mark.parametrize("row, zs, zr, wave, within", ON_GROUND)
def test_rows_on_a_rigid_ground_against_a_wave_calculation(row, zs, zr, wave, within) -> None:
    source, listener = (-9.99, zs), (15.01, zr)
    found = over(row, source, listener, lift=0.0)
    np.testing.assert_allclose(levels(found, source, listener), wave, atol=within)


def test_between_rigid_screens_on_a_rigid_ground_the_row_is_low() -> None:
    """Three rigid screens 5 m apart, 6 m high on a rigid ground: the level just over
    them comes out up to 1.1 dB below the wave calculation's (a march one way has no
    sound that goes to and fro between them; no run shows that this is what lacks)."""
    row = [(0.01, 0.01, 6.0), (5.01, 5.01, 6.0), (10.01, 10.01, 6.0)]
    source, listener = (-9.99, 6.51), (15.01, 6.01)
    got = levels(over(row, source, listener, lift=0.0), source, listener)
    off = got - np.array([-7.91, -8.15, -7.93, -7.69])
    assert np.all(off < -0.3) and np.all(off > -1.2) and off.min() < -0.9


def _parapet(height: float) -> list:
    """A roof that goes on behind the source without end, with a parapet ``height`` m
    high on its edge at x = 10.01 (the middle of the wave calculation's cell)."""
    return [(-30.0, 10.01, 0.0), (10.01, 10.01, height)]


#: A source (x, z) over a roof that goes on behind it without end, up to the roof's
#: edge or with a parapet on that edge, and the listener's height at x = 15.01: the wave
#: calculation's level (dB re free field), how far the row's may be from it (dB) and
#: how far the march alone is from it at least (none where it is the closer). A parapet
#: lower than a wavelength is further off than the roof's edge alone: the edges' terms
#: take it for a half-plane, and have no sound that goes on between its top and the
#: roof.
ROOF_TO_ITS_EDGE = [(-30.0, 10.02, 0.0)]
OVER_A_ROOF = [
    (ROOF_TO_ITS_EDGE, (9.51, 0.01), -1.99, [0.25, -0.56, -1.58, -2.83], 0.15, 1.5),
    (ROOF_TO_ITS_EDGE, (9.51, 1.51), -1.99, [0.16, -0.28, -0.29, 0.09], 0.25, 0.7),
    (ROOF_TO_ITS_EDGE, (7.01, 0.01), 1.01, [1.97, 2.35, 3.02, 4.05], 0.1, 0.3),
    (_parapet(1.1), (9.51, 0.01), -1.99, [-3.13, -6.72, -9.20, -12.37], 0.9, 8.0),
    (_parapet(1.1), (7.01, 1.51), 1.01, [-2.20, -7.58, -2.58, -3.18], 0.6, 2.0),
    (_parapet(0.3), (9.51, 0.01), -1.99, [-0.25, -0.95, -2.97, -6.46], 0.7, 4.5),
    (_parapet(0.3), (9.51, 1.51), 1.01, [-0.74, -0.62, 0.36, 0.07], 1.2, None),
    (_parapet(0.1), (9.51, 0.01), 1.01, [1.72, 1.66, 1.70, 1.43], 1.2, 1.5),
]


@pytest.mark.parametrize("row, source, zr, wave, within, alone_off", OVER_A_ROOF)
def test_a_source_over_a_roof_against_a_wave_calculation(
    row, source, zr, wave, within, alone_off
) -> None:
    listener = (15.01, zr)
    found = over(row, source, listener)
    np.testing.assert_allclose(levels(found, source, listener), wave, atol=within)
    if alone_off is not None:
        alone = levels(found, source, listener, "marched")
        assert np.max(np.abs(alone - wave)) > alone_off


def test_a_barrier_far_lower_than_a_wavelength_on_a_rigid_roof_is_off() -> None:
    """A barrier 10 cm high on a roof under both ends, half a metre past the source:
    the row's level is 2.3 dB above the wave calculation's in the 63 Hz band from a
    source on the roof, and 3.0 dB below it in the 125 Hz band from one 1.5 m up,
    where the march alone is within 0.4 dB. The edges' terms take the barrier for a
    half-plane, and have no sound that goes on between its top and the roof."""
    row = [(-30.0, 60.0, 0.0), (10.01, 10.01, 0.1)]
    listener = (15.01, 1.51)
    low, high = (9.51, 0.01), (9.51, 1.51)
    off = levels(over(row, low, listener), low, listener) - np.array([6.04, 6.16, 6.44, 5.84])
    assert 2.1 < off[0] < 2.6 and 1.0 < off[1] < 1.5
    found, wave = over(row, high, listener), np.array([4.42, 0.96, -1.96, 3.00])
    assert -3.3 < (levels(found, high, listener) - wave)[1] < -2.7
    assert np.max(np.abs(levels(found, high, listener, "marched") - wave)) < 0.45


# -- what the row keeps ----------------------------------------------------------------


def test_what_stands_before_the_source_or_past_the_listener_does_nothing() -> None:
    """The march goes one way: a barrier behind the source and a building past the
    listener, higher than the row between them, leave the sound as it is."""
    row = [(0.0, 4.0, 0.0), (6.0, 6.0, 0.3)]
    source, listener = (-9.99, 0.5), (15.01, 0.01)
    outside = [(-30.0, -30.0, 5.0), (20.0, 28.0, 9.0)]
    for alone, among in zip(
        over(row, source, listener), over(row + outside, source, listener), strict=True
    ):
        assert np.array_equal(alone.marched, among.marched)
        assert np.array_equal(alone.edges, among.edges)


def test_what_stands_at_an_end_but_not_on_its_way_does_nothing() -> None:
    """On a rigid ground, a source 3 m up and a listener 1.5 m up 30 m from it, a
    barrier between them: a barrier under an end (its top at the end's height or
    below), a building whose wall stands just behind the source and one that begins
    at the listener, both higher than they, leave both marches as they are."""
    f = F[2]
    ends, row = (f, C, None, 3.0, 30.0, 1.5), [(12.0, 12.0, 5.0)]
    alone = rows.over_row(*ends, row)
    for at_an_end in (
        [(0.0, 0.0, 3.0)],
        [(0.0, 0.0, 2.0), (30.0, 30.0, 1.5)],
        [(-8.0, 0.0, 9.0)],
        [(-8.0, 1e-10, 9.0)],
        [(30.0, 40.0, 9.0)],
        [(30.0, 30.0, 1.0), (-8.0, 0.0, 9.0), (30.0, 40.0, 9.0), (0.0, 0.0, 0.5)],
    ):
        among = rows.over_row(*ends, row + at_an_end)
        assert np.array_equal(alone.marched, among.marched)
        assert np.array_equal(alone.edges, among.edges)


@pytest.mark.parametrize(
    "end, obstacle",
    [
        ("source", (0.0, 10.0, 8.0)),  # against the wall it would have to climb
        ("source", (-5.0, 5.0, 8.0)),  # in the building
        ("source", (0.0, 0.0, 8.0)),  # in a barrier's plane
        ("source", (5e-10, 5e-10, 8.0)),  # and a hair off it, at one place with it
        ("listener", (20.0, 30.0, 8.0)),
        ("listener", (25.0, 35.0, 8.0)),
        ("listener", (30.0, 30.0, 8.0)),
        ("listener", (30.0 - 5e-10, 30.0 - 5e-10, 8.0)),
    ],
)
def test_an_end_in_or_against_an_obstacle_below_its_top_is_refused(end, obstacle) -> None:
    """An end in a building, or against a wall or in the plane of a barrier that it
    would have to climb, has no place in a row: a ValueError, and not
    :class:`rows.OutOfReach` (an end whose march cannot hold what its start needs)."""
    row = [(12.0, 12.0, 5.0), obstacle]
    with pytest.raises(ValueError, match=f"the {end} stands in or against an obstacle") as error:
        rows.over_row(F[2], C, None, 1.5, 30.0, 1.5, row)
    assert not isinstance(error.value, rows.OutOfReach)


def test_the_absorbing_layer_s_start_is_checked_and_an_obstacle_s_sides() -> None:
    ends = (F[3], C, None, 1.5, 300.0, 1.5)
    with pytest.raises(ValueError, match="starts above both ends and every obstacle"):
        rows.over_row(*ends, [(100.0, 150.0, 60.0)], top=3.0)
    with pytest.raises(ValueError, match="starts above both ends and every obstacle"):
        rows.over_row(*ends, [(100.0, 150.0, 60.0)], top=60.0)
    with pytest.raises(ValueError, match="far side lies before its near side"):
        rows.over_row(*ends, [(150.0, 100.0, 60.0)])


def test_a_given_start_of_the_absorbing_layer_is_the_marches(monkeypatch) -> None:
    """A ``top`` above both ends and what stands between them is where both marches'
    layers start, whatever stands higher before the source or past the listener."""
    ends = (F[3], C, None, 1.5, 30.0, 1.5)
    tops = []
    march = parabolic.march

    def spy(*given, **more):
        tops.append(given[6])
        return march(*given, **more)

    monkeypatch.setattr(rows.parabolic, "march", spy)
    alone = rows.over_row(*ends, [(12.0, 12.0, 5.0)], top=30.0)
    outside = [(-30.0, -30.0, 50.0), (40.0, 50.0, 50.0)]
    among = rows.over_row(*ends, [(12.0, 12.0, 5.0), *outside], top=30.0)
    assert tops == [30.0] * 4
    assert np.array_equal(alone.pressure, among.pressure)
    rows.over_row(*ends, [(12.0, 12.0, 5.0)])
    assert tops[4:] == [rows.layer_top(1.5, 1.5, 30.0, [(12.0, 12.0, 5.0)])] * 2


@pytest.mark.parametrize("row", [ROOF_AND_SCREEN, THREE_SCREENS, TWO_ROOFS])
def test_the_sound_is_the_same_with_the_ends_swapped(row) -> None:
    f = F[3]
    reach, zs, zr = 25.0, LIFT + 0.5, LIFT - 1.0
    standing = [(a + 9.99, b + 9.99, LIFT + h) for a, b, h in row]
    mirrored = [(reach - b, reach - a, h) for a, b, h in standing]
    one = rows.over_row(f, C, None, zs, reach, zr, standing, steep_reflections="leave out")
    other = rows.over_row(f, C, None, zr, reach, zs, mirrored, steep_reflections="leave out")
    np.testing.assert_allclose(other.pressure, one.pressure, rtol=1e-6)
    np.testing.assert_allclose(other.edges, one.edges, rtol=1e-6)


def test_the_march_from_the_listener_takes_the_ground_the_other_way_round() -> None:
    """Grass before a barrier and a rigid ground behind it: the same sound from either
    end, each march meeting the grounds in its own order."""
    f = F[2]
    grass = np.conj(surface_impedance(f, 200.0))
    reach, zs, zr = 60.0, 1.0, 4.0
    row = [(20.0, 20.0, 3.0), (30.0, 38.0, 5.0)]

    def ground(r):
        return grass if r < 20.0 else None

    one = rows.over_row(f, C, ground, zs, reach, zr, row)
    other = rows.over_row(
        f,
        C,
        lambda r: ground(reach - r),
        zr,
        reach,
        zs,
        [(reach - b, reach - a, h) for a, b, h in row],
    )
    np.testing.assert_allclose(other.pressure, one.pressure, rtol=1e-6)
    rigid = rows.over_row(f, C, None, zs, reach, zr, row)
    assert abs(level(one.pressure, reach) - level(rigid.pressure, reach)) > 1.0


def test_the_sound_is_the_same_with_the_ends_swapped_close_under_the_walls() -> None:
    """A source 1 m before a barrier 7.5 m above it and a listener 2 m behind a
    building 8 m above it, on a rigid ground: the same sound from either end."""
    f = F[3]
    reach, zs, zr = 17.0, 1.5, 4.0
    row = [(1.0, 1.0, 9.0), (6.0, 15.0, 12.0)]
    one = rows.over_row(f, C, None, zs, reach, zr, row)
    other = rows.over_row(f, C, None, zr, reach, zs, [(reach - b, reach - a, h) for a, b, h in row])
    np.testing.assert_allclose(other.pressure, one.pressure, rtol=1e-9)
    np.testing.assert_allclose(other.edges, one.edges, rtol=1e-9)


@pytest.mark.parametrize("flow", [200.0, 50.0])
def test_a_screen_5_cm_high_on_grass_leaves_the_open_ground_s_sound(flow: float) -> None:
    """Each march begins at the screen, from the field its end has there over the
    grass: the sound of a point source over open grass (the spherical wave's reflection
    factor), from 5 dB above the free field in the 63 Hz band to 13 dB below it in the
    dip, within 0.3 dB by the march. The edge's terms take the screen for a half-plane
    far higher than a wavelength, which one so low is not: with them the level is
    within 0.55 dB."""
    zs, zr, reach = 1.5, 1.5, 50.0
    direct, image = math.hypot(reach, zr - zs), math.hypot(reach, zr + zs)
    for f in F:
        ground = np.conj(surface_impedance(f, flow))
        k = 2.0 * math.pi * f / C
        reflected = np.conj(
            spherical_reflection(np.conj(1.0 / ground), (zr + zs) / image, k, image)
        )
        open_ground = np.exp(1j * k * direct) / direct + reflected * np.exp(1j * k * image) / image
        got = rows.over_row(f, C, lambda r, z=ground: z, zs, reach, zr, [(20.0, 20.0, 0.05)])
        assert abs(level(got.marched, direct) - level(open_ground, direct)) < 0.3
        assert abs(level(got.pressure, direct) - level(open_ground, direct)) < 0.55
    assert level(open_ground, direct) < -4.0  # the 500 Hz band, past the dip


@pytest.mark.parametrize("flow", [200.0, 50.0])
def test_a_source_lying_on_grass_lights_the_first_edge_with_its_image_s_reflection(
    flow: float,
) -> None:
    """A source on the grass 20 m before a screen 5 cm high and a listener 1.5 m up,
    50 m from it: the march within 0.25 dB of the open grass's sound, 0.55 dB with the
    edge's terms. The source's image, at the source's own place, lights the screen's
    top with the grass's reflection factor."""
    zr, reach = 1.5, 50.0
    distance = math.hypot(reach, zr)
    top = rows.layer_top(0.0, zr, reach, [(20.0, 20.0, 0.05)])
    for f in F:
        ground = np.conj(surface_impedance(f, flow))
        k = 2.0 * math.pi * f / C
        reflected = np.conj(spherical_reflection(np.conj(1.0 / ground), zr / distance, k, distance))
        open_ground = (1.0 + reflected) * np.exp(1j * k * distance) / distance
        row = [(20.0, 20.0, 0.05)]
        got = rows.over_row(f, C, lambda r, z=ground: z, 0.0, reach, zr, row)
        assert abs(level(got.marched, distance) - level(open_ground, distance)) < 0.25
        assert abs(level(got.pressure, distance) - level(open_ground, distance)) < 0.55
        way = rows._march_from(
            f, C, reach, top, parabolic.STEPS_PER_WAVELENGTH, lambda r, z=ground: z, 0.0, zr, row
        )()
        (own, a, _, _), (image, also, _, _) = rows._lit(way, k, ("screen", 20.0, 0.05))
        assert a == also == math.hypot(20.0, 0.05)
        at_top = np.conj(spherical_reflection(np.conj(1.0 / ground), 0.05 / a, k, a))
        np.testing.assert_allclose(image / own, at_top, rtol=1e-12)
        assert np.all(np.abs(at_top - 1.0) > 0.3)


def test_where_the_ground_changes_before_the_first_obstacle_the_march_begins_at_its_end() -> None:
    """Grass up to 10 m from the source and grass of all but the same flow resistivity
    from there to a barrier at 20 m: the march cannot begin at the barrier from the
    source's field over one ground, and begins at the source with the parabolic
    equation's own starting field, which near the horizontal gives the level over one
    grass within 0.06 dB."""
    zs, zr, reach = 1.0, 4.0, 60.0
    row = [(20.0, 20.0, 3.0), (30.0, 38.0, 5.0)]
    for f in F:
        one = np.conj(surface_impedance(f, 200.0))
        other = np.conj(surface_impedance(f, 200.0 * (1.0 + 1e-9)))
        assert not np.array_equal(one, other)
        same = rows.over_row(f, C, lambda r, z=one: z, zs, reach, zr, row)
        changing = rows.over_row(
            f, C, lambda r, a=one, b=other: a if r < 10.0 else b, zs, reach, zr, row
        )
        apart = abs(level(changing.pressure, reach) - level(same.pressure, reach))
        assert 1e-4 < apart < 0.06
        assert rows._first(f, lambda r, z=one: z, reach, row) == (20.0, 0.0)
        changes = rows._first(f, lambda r, a=one, b=other: a if r < 10.0 else b, reach, row)
        assert changes == (None, 0.0)
    # Nor can it begin at an obstacle without one on the way.
    assert rows._first(F[0], None, reach, []) == (None, 0.0)
    assert rows._first(F[0], None, reach, [(70.0, 70.0, 3.0)]) == (None, 0.0)


def test_over_a_roof_a_march_begins_at_the_first_top_past_its_end() -> None:
    """An end over a roof has its own field, with its image's in the roof, up to the
    first top past it: a barrier on the roof, what stands against the roof's far side,
    or that side. Where the roof goes on to the other end with no barrier between
    them, the march begins at the end."""
    reach = 60.0
    assert rows._first(F[0], None, reach, [(0.0, 8.0, 0.5), (30.0, 38.0, 5.0)]) == (8.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 8.0, 0.5)]) == (8.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 8.0, 0.5), (3.0, 3.0, 1.5)]) == (3.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 8.0, 0.5), (8.0, 8.0, 1.5)]) == (8.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 8.0, 0.5), (8.0, 20.0, 4.0)]) == (8.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 8.0, 0.5), (9.0, 9.0, 4.0)]) == (8.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 80.0, 0.5)]) == (None, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 60.0, 0.5)]) == (None, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 80.0, 0.5), (59.0, 59.0, 1.0)]) == (59.0, 0.5)
    assert rows._first(F[0], None, reach, [(-5.0, 80.0, 0.5), (61.0, 61.0, 1.0)]) == (None, 0.5)
    # The ground under the building does not count, whatever it is.
    grass = np.conj(surface_impedance(F[0], 200.0))
    assert rows._first(F[0], lambda r: grass if r < 4.0 else None, reach, [(-5.0, 8.0, 0.5)]) == (
        8.0,
        0.5,
    )


@pytest.mark.parametrize(
    "angle, low, high",
    [(20.0, -0.15, 0.1), (34.0, -0.4, -0.1), (40.0, -0.55, -0.25), (50.0, -1.15, -0.85)],
)
def test_a_march_from_its_end_takes_ways_within_35_degrees_of_the_horizontal(
    monkeypatch, angle: float, low: float, high: float
) -> None:
    """One rigid screen, the source 10 m before it and 20 to 50 degrees below its top.
    With each march from the screen on, the level is within 0.1 dB of the exact
    solution. With each from its end, as where the ground changes before the first
    obstacle, the parabolic equation's starting field holds near the horizontal:
    within 0.15 dB at 20 degrees and 0.1 to 0.4 dB low at 34; beyond 35 degrees the
    end is out of reach, where such a march is 0.25 to 0.55 dB low at 40 degrees and
    0.85 to 1.15 dB low at 50. (The row stands high above a ground, whose reflection
    a march from its end cannot leave out: within 35 degrees the end is out of reach
    for that, and the levels are those without the limit.)"""
    source, listener = (-10.0, -10.0 * math.tan(math.radians(angle))), (15.0, -1.0)
    exact = over_one_screen(source, listener)
    from_the_screen = levels(over([(0.0, 0.0, 0.0)], source, listener), source, listener)
    np.testing.assert_allclose(from_the_screen, exact, atol=0.1)
    monkeypatch.setattr(rows, "_first", lambda *given: (None, 0.0))
    steep = f"way on {angle:.1f}" if angle > rows.FROM_END_DEG else "reflection off the ground 89.9"
    with pytest.raises(rows.OutOfReach, match=f"{steep} degrees steep.*takes 35"):
        over([(0.0, 0.0, 0.0)], source, listener)
    monkeypatch.setattr(rows, "FROM_END_DEG", 90.0)
    off = levels(over([(0.0, 0.0, 0.0)], source, listener), source, listener) - exact
    assert np.all(off > low) and np.all(off < high)


def test_a_march_from_its_end_keeps_to_what_its_starting_field_holds() -> None:
    """An end with the ground changing before its first obstacle, or with nothing on
    its way (on the ground, or on a roof that goes on to the other end), is marched
    from the end itself with the parabolic equation's starting field. That holds
    within 35 degrees of the horizontal and from a wavelength on: a first top, or the
    other end with no top before it, closer than a wavelength of the band's lowest
    frequency, and a way on from the end or from its image in the ground or the roof
    steeper than 35 degrees, are out of reach, whatever is asked about steep
    reflections. Within those the levels here are within 0.45 dB of the edge
    integrals over the paths, and with nothing between the ends within 0.3 dB of the
    direct sound and the reflection."""

    def ground_changing(before: float, band: int, changing: bool = True) -> float:
        # A screen 8 m high on a rigid ground, which near it is rigid by an impedance.
        f, reach = F[band], before + 15.0

        def ground(r):
            return None if r < 0.5 * before or r > before + 1.0 else np.full(len(f), 1e9 + 0j)

        row = [(before, before, 8.0)]
        found = rows.over_row(f, C, ground if changing else None, 1.5, reach, 1.5, row)
        paths = over_one_screen((-before, 1.5), (15.0, 1.5), heights_from=8.0)
        return level(found.pressure, reach) - paths[band]

    assert abs(ground_changing(6.0, 3, changing=False)) < 0.1  # from the screen on
    with pytest.raises(rows.OutOfReach, match="ground changing.*way on 47.3 degrees steep"):
        ground_changing(6.0, 3)
    with pytest.raises(rows.OutOfReach, match="reflection off the ground 38.4 degrees steep"):
        ground_changing(12.0, 3)
    for band in range(4):
        assert abs(ground_changing(15.0, band)) < 0.45

    # A first top closer than a wavelength, with another one far enough behind it too.
    def changes(r):
        return None if r < 0.25 else np.full(8, 1e9 + 0j)

    refused = "ground changing.*0.5 m before its first top.*closer than 0.914"
    with pytest.raises(rows.OutOfReach, match=refused):
        rows.over_row(F[3], C, changes, 0.0, 15.5, 1.0, [(0.5, 0.5, 0.1), (10.5, 10.5, 0.2)])
    with pytest.raises(rows.OutOfReach, match="6 m before its first top.*47 Hz.*closer than 7.25"):
        rows.over_row(F[0], C, changes, 0.0, 21.0, 1.0, [(6.0, 6.0, 0.5)])

    # Nothing between the ends: the direct sound and the reflection off a rigid ground.
    def open_ground(zr: float, band: int) -> float:
        k = 2.0 * math.pi * F[band] / C
        direct, image = math.hypot(20.0, zr - 1.0), math.hypot(20.0, zr + 1.0)
        both = np.exp(1j * k * direct) / direct + np.exp(1j * k * image) / image
        found = rows.over_row(F[band], C, None, 1.0, 20.0, zr, [])
        return level(found.pressure, direct) - level(both, direct)

    with pytest.raises(rows.OutOfReach, match="nothing between.*way on 43.5 degrees"):
        open_ground(20.0, 0)
    assert abs(open_ground(13.0, 0)) < 0.25 and abs(open_ground(13.0, 3)) < 0.25
    # The other end is as close as a first top may be: a wavelength.
    with pytest.raises(rows.OutOfReach, match="nothing between.*6 m before the other end.*7.25"):
        rows.over_row(F[0], C, None, 0.0, 6.0, 0.0, [])
    one_roof = [(-20.0, 40.0, 2.0)]
    with pytest.raises(rows.OutOfReach, match="over a roof.*0.5 m before the other end.*0.914"):
        rows.over_row(F[3], C, None, 2.0, 0.5, 2.0, one_roof)
    with pytest.raises(rows.OutOfReach, match="over a roof.*off the roof 45.0 degrees steep"):
        rows.over_row(F[3], C, None, 7.0, 10.0, 7.0, one_roof)
    for band in (0, 3):  # from a wavelength on, on the ground and on a roof
        reach = 1.02 * C / F[band].min()
        k = 2.0 * math.pi * F[band] / C
        twice = level(2.0 * np.exp(1j * k * reach) / reach, reach)
        on_the_ground = rows.over_row(F[band], C, None, 0.0, reach, 0.0, [])
        on_the_roof = rows.over_row(F[band], C, None, 2.0, reach, 2.0, one_roof)
        assert abs(level(on_the_ground.pressure, reach) - twice) < 0.3
        assert abs(level(on_the_roof.pressure, reach) - twice) < 0.3
    # A march from its end cannot leave a reflection out: asked to, it is refused too.
    ends = (F[3], C, None, 300.0, 20.0, 301.0, [])
    for steep in ("raise", "leave out"):
        refused = "nothing between.*reflection off the ground 88.1 degrees steep"
        with pytest.raises(rows.OutOfReach, match=refused):
            rows.over_row(*ends, steep_reflections=steep)

    # A way down is as steep as a way up: from 20 m above a top, past it to the listener.
    def nearly_rigid(r):
        return None if r < 5.0 else np.full(len(F[3]), 1e9 + 0j)

    high = (F[3], C, nearly_rigid, 320.0, 25.0, 300.0, [(10.0, 10.0, 300.0)])
    with pytest.raises(rows.OutOfReach, match="way on 38.7 degrees steep"):
        rows.over_row(*high, steep_reflections="leave out")
    # And a reflection off a ground that changes, 54 degrees steep under a way on of 7.
    near = (F[3], C, nearly_rigid, 5.0, 23.0, 1.5, [(8.0, 8.0, 6.0)])
    for steep in ("raise", "leave out"):
        refused = "ground changing.*reflection off the ground 54.0 degrees steep"
        with pytest.raises(rows.OutOfReach, match=refused):
            rows.over_row(*near, steep_reflections=steep)


def test_a_march_that_begins_at_its_end_takes_the_way_up_to_the_tops() -> None:
    """Over grass, a source 1 m up, a barrier 4 m high 8 m from it (21 degrees up, the
    reflection off the grass 32) and a building 5 m high behind it. With grass of all
    but the same flow resistivity from 2 m on, the march from the source begins at the
    source with the parabolic equation's starting field: a point source's along the
    ground, and the root of a path's cosine less up that path. The marched sound is
    then within 0.15 dB of that of the march from the barrier on, what the edges add
    within 0.5 dB and the level within 0.25 dB."""
    zs, zr, reach = 1.0, 4.0, 40.0
    row = [(8.0, 8.0, 4.0), (18.0, 26.0, 5.0)]
    for f in F:
        one = np.conj(surface_impedance(f, 200.0))
        other = np.conj(surface_impedance(f, 200.0 * (1.0 + 1e-9)))
        same = rows.over_row(f, C, lambda r, z=one: z, zs, reach, zr, row)
        changing = rows.over_row(
            f, C, lambda r, a=one, b=other: a if r < 2.0 else b, zs, reach, zr, row
        )
        assert abs(level(changing.marched, reach) - level(same.marched, reach)) < 0.15
        assert abs(level(changing.edges, reach) - level(same.edges, reach)) < 0.5
        assert abs(level(changing.pressure, reach) - level(same.pressure, reach)) < 0.25


def over_a_corner(source, listener, roof: float = -1.0) -> np.ndarray:
    """(4,) the exact level (dB re free field) in the four bands at the corner, at
    x = 0 and z = 0, of a rigid roof that goes on without end before it (``roof`` -1:
    its far corner) or past it (1: its near corner), with its wall below the corner:
    the direct sound and the reflection off the roof, where the listener sees the
    source and its image in the roof, and the Biot-Tolstoy-Medwin integral over the
    corner, a wedge open over three right angles."""
    edge = edge_diffraction.Edge([0, 0, 0], [0, 1, 0], [roof, 0, 0], [0, 0, 1], 1.5)
    a, b = np.array([source[0], 0.0, source[1]]), np.array([listener[0], 0.0, listener[1]])
    round_s = math.atan2(source[1], roof * source[0]) % (2.0 * math.pi)
    round_l = math.atan2(listener[1], roof * listener[0]) % (2.0 * math.pi)
    seen = (abs(round_l - round_s) < math.pi, round_l + round_s < math.pi)
    out = []
    for band in 2.0 * math.pi * F / C:
        p = edge_diffraction.btm_first(edge, a, b, band)
        for there, height in zip(seen, (a[2], -a[2]), strict=True):
            if there:
                way = float(np.linalg.norm(b - np.array([a[0], 0.0, height])))
                p = p + np.exp(-1j * band * way) / way
        out.append(level(p, float(np.linalg.norm(b - a))))
    return np.array(out)


@pytest.mark.parametrize(
    "source, listener, within",
    [
        ((-8.0, 1.0), (15.0, -2.0), 0.05),
        ((-5.0, 0.3), (15.0, -3.0), 0.05),
        ((-10.0, 2.0), (10.0, 0.5), 0.05),
        ((-8.0, 0.0), (12.0, -1.0), 0.05),
        ((-3.0, 1.5), (15.0, -6.0), 0.1),
        ((-1.0, 0.5), (15.0, -2.0), 0.05),
        ((-0.5, 1.0), (15.0, -2.0), 0.1),
        ((-0.3, 0.0), (15.0, -2.0), 0.1),
        ((-0.2, 0.5), (10.0, -1.0), 0.1),
        ((-1.0, 0.0), (2.0, -8.0), 0.1),
        ((-0.1, 0.0), (15.0, 0.5), 0.1),
        ((-1.0, 2.0), (5.0, -3.0), 0.2),
        ((-2.0, 3.0), (10.0, -8.0), 0.2),
    ],
)
def test_an_end_over_a_roof_against_the_exact_edge_integral(source, listener, within) -> None:
    """A source on a rigid roof or up to 3 m above it, 10 cm to 10 m before the roof's
    far corner, and a listener past the corner, up to 8 m below the roof or half a
    metre above it: within 0.05 dB of the exact solution for the corner in five
    places, 0.1 dB in six and 0.2 dB in the two with the source 2 or 3 m up and the
    way down past the corner steeper than 30 degrees, from either end and in every
    band, where the march alone is 0.25 dB or more off. The march from the end over
    the roof begins at the corner, from that end's field and its image's in the
    roof."""
    exact = over_a_corner(source, listener)
    swapped = (-listener[0], listener[1]), (-source[0], source[1])
    found = over([(-30.0, 0.0, 0.0)], source, listener)
    np.testing.assert_allclose(levels(found, source, listener), exact, atol=within)
    assert np.max(np.abs(levels(found, source, listener, "marched") - exact)) > 0.25
    found = over([(0.0, 30.0, 0.0)], *swapped)
    np.testing.assert_allclose(levels(found, *swapped), exact, atol=within)


@pytest.mark.parametrize(
    "top, source, listener",
    [
        (2.0, (-10.0, 1.5), (15.0, 1.5)),
        (2.0, (-10.0, 0.5), (15.0, 3.0)),
        (1.0, (-10.0, 2.0), (15.0, 0.2)),
        (3.0, (-20.0, 1.0), (10.0, 1.0)),
    ],
)
def test_a_barrier_on_a_roof_under_both_ends_against_the_edge_integrals_of_its_paths(
    top, source, listener
) -> None:
    """A barrier 1 to 3 m high on a rigid roof that both ends stand over, 0.2 to 3 m
    above it and 10 m or more from the barrier: within 0.05 dB of the four paths
    over the top of a rigid screen on a rigid ground, the roof (the exact edge
    integral from the source and its image in the roof to the listener and its
    image, without the sound that goes from the top down the screen to the roof and
    back), where the march alone is up to 2.3 dB low."""
    paths = over_one_screen(source, listener, heights_from=top)
    found = over([(-40.0, 40.0, 0.0), (0.0, 0.0, top)], source, listener)
    np.testing.assert_allclose(levels(found, source, listener), paths, atol=0.05)
    assert np.min(levels(found, source, listener, "marched") - paths) < -1.0


@pytest.mark.parametrize(
    "top, source, listener, within",
    [
        (1.0, (-0.5, 0.5), (15.0, 1.5), 0.3),
        (1.0, (-0.3, 0.0), (15.0, 0.5), 0.3),
        (1.0, (-0.2, 0.9), (20.0, 2.0), 0.4),
        (3.0, (-0.3, 1.0), (15.0, 1.0), 0.2),
        (1.2, (-1.0, 1.5), (10.0, 1.5), 0.15),
        (0.5, (-0.5, 1.5), (15.0, 0.0), 0.15),
    ],
)
def test_an_end_over_a_roof_close_before_a_barrier_on_it(top, source, listener, within) -> None:
    """A source on a rigid roof or up to 1.5 m above it, 20 cm to 1 m before a barrier
    0.5 to 3 m high on the roof, and a listener over the roof past it: within 0.4 dB
    of the four paths over the top of a rigid screen on a rigid ground, from either
    end. The march from the source begins at the barrier, from the source's field and
    its image's in the roof."""
    paths = over_one_screen(source, listener, heights_from=top)
    row = [(-40.0, 40.0, 0.0), (0.0, 0.0, top)]
    found = over(row, source, listener)
    np.testing.assert_allclose(levels(found, source, listener), paths, atol=within)
    swapped = (-listener[0], listener[1]), (-source[0], source[1])
    np.testing.assert_allclose(levels(over(row, *swapped), *swapped), paths, atol=within)


def test_an_end_over_a_roof_too_close_to_its_first_top_is_out_of_reach() -> None:
    """Both ends over a roof 2 m up, the source a metre above it, a barrier 3 m high
    on the roof: 30 cm before the barrier the level is within 0.15 dB of the four
    paths over it in every band, 10 cm before it in the three lower bands. In the
    500 Hz band the start there needs the source's image in the roof up to 88.7
    degrees steep, and the source is out of reach; 5 cm before the barrier in every
    band, and so asked to leave steep reflections out: that is for the ground's, and
    a roof's is the source's own."""

    def on_a_roof(before: float, band: int, steep: str = "raise") -> float:
        reach, row = before + 15.0, [(-20.0, before + 35.0, 2.0), (before, before, 5.0)]
        found = rows.over_row(F[band], C, None, 3.0, reach, 3.0, row, steep_reflections=steep)
        paths = over_one_screen((-before, 1.0), (15.0, 1.0), heights_from=3.0)
        return level(found.pressure, reach) - paths[band]

    for band in range(4):
        assert abs(on_a_roof(0.3, band)) < 0.15
    for band in range(3):
        assert abs(on_a_roof(0.1, band)) < 0.15
    refused = "0.1 m before its first top.*reflection off the roof needs up to 88.7 degrees"
    with pytest.raises(rows.OutOfReach, match=refused):
        on_a_roof(0.1, 3)
    for steep in ("raise", "leave out"):
        for band in range(4):
            with pytest.raises(rows.OutOfReach, match="0.05 m before its first top.* takes 8"):
                on_a_roof(0.05, band, steep)
    with pytest.raises(rows.OutOfReach, match="reflection off the roof needs up to 89.6 degrees"):
        on_a_roof(0.05, 0, "leave out")
    # Over the ground, the reflection that is too steep is left out where that is asked.
    ground = (F[0], C, None, 1.0, 15.05, 1.0, [(0.05, 0.05, 3.0)])
    with pytest.raises(rows.OutOfReach, match="reflection off the ground needs up to 89.6"):
        rows.over_row(*ground)
    rows.over_row(*ground, steep_reflections="leave out")


def test_a_roof_s_edge_behind_an_end_does_nothing() -> None:
    """The march goes one way, and a roof under an end goes on behind it without end:
    with a source on the roof and a listener 1.5 m above it 20 m on, the same sound
    with the roof's edge 5 cm behind the source as 2 m or 30 m behind it. The exact
    solution for the corner behind the source has what the edge sends on: with the
    edge 2 m behind, the solver is within 0.4 dB of it; with the edge 5 cm behind, 2.6
    dB above it in the 63 Hz band and 2.0 dB in the 125 Hz band (at the edge itself the
    source sends 4/3 of the free field's pressure, not twice: 3.5 dB)."""
    listener = (20.0, 1.5)
    endless = over([(-30.0, 40.0, 0.0)], (0.0, 0.0), listener)
    for behind, low, high in (
        (0.05, (2.3, 1.8, 1.0, -0.2), (2.8, 2.3, 1.5, 0.4)),
        (2.0, -0.4, 0.4),
    ):
        found = over([(-behind, 40.0, 0.0)], (0.0, 0.0), listener)
        for one, other in zip(found, endless, strict=True):
            assert np.array_equal(one.pressure, other.pressure)
        exact = over_a_corner((behind, 0.0), (behind + listener[0], listener[1]), roof=1.0)
        off = levels(found, (0.0, 0.0), listener) - exact
        assert np.all(off > low) and np.all(off < high)


def test_the_string_over_the_tops() -> None:
    # Pulled taut from the source to the listener, it bends at the tops it rests on.
    tops = [(10.0, 3.0), (14.0, 9.0), (20.0, 9.0), (25.0, 4.0), (30.0, 6.5)]
    assert rows.string([(0.0, 1.0), *tops, (40.0, 2.0)]) == [
        (0.0, 1.0),
        (14.0, 9.0),
        (20.0, 9.0),
        (30.0, 6.5),
        (40.0, 2.0),
    ]
    # Straight over tops below the line of sight, and through one on it.
    assert rows.string([(0.0, 5.0), (10.0, 3.0), (30.0, 4.9)]) == [(0.0, 5.0), (30.0, 4.9)]
    assert rows.string([(0.0, 5.0), (10.0, 5.0), (30.0, 5.0)]) == [(0.0, 5.0), (30.0, 5.0)]
    # The angle the sound turns by at a top, from where it comes from to where it goes.
    assert rows._bend((-1.0, 0.0), (1.0, 0.0)) == 0.0
    assert abs(rows._bend((-1.0, -1.0), (1.0, -1.0))) == pytest.approx(0.5 * math.pi)
    assert abs(rows._bend((-1.0, 0.0), (0.0, -1.0))) == pytest.approx(0.5 * math.pi)


def test_the_row_s_skyline_and_the_edges_that_shed_sound() -> None:
    # Roofs level with each other that touch or overlap are one roof, in any order.
    assert rows.skyline([(16.0, 20.0, 6.0), (10.0, 13.0, 6.0), (12.0, 16.0, 6.0)]) == [
        (10.0, 20.0, 6.0)
    ]
    assert rows.edges_of([(10.0, 20.0, 6.0)]) == [("roof", 10.0, 6.0), ("behind", 20.0, 6.0)]
    # A building whose two sides are at one place is a barrier there.
    assert rows.skyline([(5.0, 5.0 + 1e-10, 3.0)]) == [(5.0, 5.0, 3.0)]
    assert rows.skyline([(5.0, 5.0 + 1e-10, 3.0), (2.0, 9.0, 2.0)]) == [
        (2.0, 9.0, 2.0),
        (5.0, 5.0, 3.0),
    ]
    # Over buildings that overlap, the highest roof; a lower roof against a wall has no
    # corner there, however little lower.
    for lower in (5.0, 5.999):
        stepped = rows.skyline([(10.0, 17.0, lower), (15.0, 20.0, 6.0), (15.0, 20.0, 6.0)])
        assert stepped == [(10.0, 15.0, lower), (15.0, 20.0, 6.0)]
        assert rows.edges_of(stepped) == [
            ("roof", 10.0, lower),
            ("roof", 15.0, 6.0),
            ("behind", 20.0, 6.0),
        ]
    inside = rows.skyline([(10.0, 20.0, 6.0), (12.0, 14.0, 9.0), (13.0, 30.0, 4.0)])
    assert inside == [
        (10.0, 12.0, 6.0),
        (12.0, 14.0, 9.0),
        (14.0, 20.0, 6.0),
        (20.0, 30.0, 4.0),
    ]
    assert rows.edges_of(inside) == [
        ("roof", 10.0, 6.0),
        ("roof", 12.0, 9.0),
        ("behind", 14.0, 9.0),
        ("behind", 20.0, 6.0),
        ("behind", 30.0, 4.0),
    ]
    # A barrier counts where it stands higher than a roof it is on or against, and
    # then hides the roof's corner under it; of two at one place, the higher.
    with_barriers = rows.skyline(
        [
            (10.0, 20.0, 6.0),
            (10.0, 10.0, 7.0),
            (20.0, 20.0, 6.0),
            (15.0, 15.0, 5.0),
            (15.0, 15.0, 8.0),
        ]
        + [(25.0, 25.0, 3.0), (25.0, 25.0, 2.0), (30.0, 30.0, 0.0), (40.0, 45.0, -1.0)]
    )
    with pytest.raises(ValueError, match="far side lies before its near side"):
        rows.skyline([(10.0, 20.0, 6.0), (40.0, 35.0, 5.0)])
    assert with_barriers == [
        (10.0, 10.0, 7.0),
        (10.0, 20.0, 6.0),
        (15.0, 15.0, 8.0),
        (25.0, 25.0, 3.0),
    ]
    assert rows.edges_of(with_barriers) == [
        ("screen", 10.0, 7.0),
        ("behind", 20.0, 6.0),
        ("screen", 15.0, 8.0),
        ("screen", 25.0, 3.0),
    ]


def test_the_row_is_marched_over_its_skyline() -> None:
    f = F[3]
    reach, zs, zr = 30.0, LIFT + 0.3, LIFT - 0.5

    def got(*row):
        standing = [(a, b, LIFT + h) for a, b, h in row]
        return rows.over_row(f, C, None, zs, reach, zr, standing, steep_reflections="leave out")

    whole = got((10.0, 20.0, 0.0))
    for same in (
        [(10.0, 15.0, 0.0), (15.0, 20.0, 0.0)],
        [(16.0, 20.0, 0.0), (10.0, 13.0, 0.0), (12.0, 16.0, 0.0)],
        [(10.0, 20.0, 0.0), (10.0, 20.0, 0.0), (12.0, 18.0, -3.0)],
        [(10.0, 20.0, 0.0), (10.0, 10.0, -0.2), (20.0, 20.0, 0.0)],
    ):
        np.testing.assert_array_equal(got(*same).pressure, whole.pressure)
    # A lower roof that runs on under a higher one ends at its wall.
    stepped = got((10.0, 15.0, -1.0), (15.0, 20.0, 0.0))
    np.testing.assert_array_equal(
        got((10.0, 17.0, -1.0), (15.0, 20.0, 0.0)).pressure, stepped.pressure
    )
    # Its corner at the wall sheds nothing: nearly the sound where it ends 10 cm short.
    apart = got((10.0, 14.9, -1.0), (15.0, 20.0, 0.0))
    assert abs(level(stepped.pressure, reach) - level(apart.pressure, reach)) < 0.1


def test_what_the_edges_shed_level_with_both_ends() -> None:
    """Far from the edge in wavelengths, along the line through a barrier's top a rigid
    half-plane sheds exp(i pi / 4) / (2 sqrt(2 pi k)) beyond one without a reflection,
    and along a roof its right-angled corner 2 / (3 sqrt(3)) of that."""
    k = 2.0 * math.pi * F[3] / C
    far = k * 1e9
    screen = np.exp(0.25j * math.pi) / (2.0 * np.sqrt(2.0 * math.pi * k))
    np.testing.assert_allclose(rows._screen(k, far, (-1.0, 0.0), (1.0, 0.0)), screen, rtol=1e-6)
    corner = screen * 2.0 / (3.0 * math.sqrt(3.0))
    np.testing.assert_allclose(rows._corner(k, far, (-1.0, 0.0), 1.0), corner, rtol=1e-6)
    np.testing.assert_allclose(rows._corner(k, far, (1.0, 0.0), -1.0), corner, rtol=1e-6)
    # A barrier's top sheds the same from either end.
    low, high = (-3.0, -1.0), (2.0, 0.5)
    np.testing.assert_allclose(
        rows._screen(k, k * 2.0, low, high),
        rows._screen(k, k * 2.0, (-high[0], high[1]), (-low[0], low[1])),
    )


def test_what_the_edges_shed_round_a_right_angle() -> None:
    """Far from the edge in wavelengths, sound that comes up to a barrier's top at 45
    degrees and goes down from it at 45: the half-plane's two terms of the reflections
    and 1 - cos(45 degrees) of its two terms of the incident wave, of which the two
    marches' mean sheds cos(45 degrees); and up at 45 degrees to a roof's near corner
    and on along the roof: the right-angled wedge's two terms of the incident wave less
    cos(22.5 degrees) of the half-plane's."""
    k = 2.0 * math.pi * F[3] / C
    far = k * 1e9
    unit = 1.0 / np.sqrt(2.0 * math.pi * k)

    def cot(x: float) -> float:
        return 1.0 / math.tan(x)

    # Round the top from the screen: the source at pi / 4, the listener at 7 pi / 4.
    incident = (cot((math.pi + 1.5 * math.pi) / 4.0) + cot((math.pi - 1.5 * math.pi) / 4.0)) / 4.0
    reflected = (cot((math.pi + 2.0 * math.pi) / 4.0) + cot((math.pi - 2.0 * math.pi) / 4.0)) / 4.0
    screen = abs(reflected + (1.0 - math.cos(0.25 * math.pi)) * incident) * unit
    np.testing.assert_allclose(
        np.abs(rows._screen(k, far, (-1.0, -1.0), (1.0, -1.0))), screen, rtol=1e-6
    )
    # Round the corner from the roof: the source at 5 pi / 4; round the wall's top from
    # the wall, the roof at 3 pi / 2 and the source at pi / 4.
    wedge = (cot((math.pi + 1.25 * math.pi) / 3.0) + cot((math.pi - 1.25 * math.pi) / 3.0)) / 3.0
    plane = (cot((math.pi + 1.25 * math.pi) / 4.0) + cot((math.pi - 1.25 * math.pi) / 4.0)) / 4.0
    corner = abs(wedge - math.cos(0.125 * math.pi) * plane) * unit
    np.testing.assert_allclose(np.abs(rows._corner(k, far, (-1.0, -1.0), 1.0)), corner, rtol=1e-6)
    np.testing.assert_allclose(np.abs(rows._corner(k, far, (1.0, -1.0), -1.0)), corner, rtol=1e-6)


def test_what_a_roof_s_corner_sheds_of_the_own_wave_of_an_end_over_the_roof() -> None:
    """Far from the corner in wavelengths, the wave of a source 30 degrees above a roof
    on its way to the roof's far corner, which goes down from it at 45 degrees (round
    the corner from the roof: the source at pi / 6, the listener at 5 pi / 4): the
    right-angled wedge's two terms of the incident wave, less cos(7.5 degrees) of the
    half-plane's, for a wave that turns by 15 degrees. And the wave of the source's
    image in the roof, 30 degrees below it: the wedge's two terms of the reflection
    off the roof, less cos(37.5 degrees) of the half-plane's, for one that turns by
    75. The near corner sheds the same seen from the other end, and a wave along the
    roof is what the march brings there."""
    k = 2.0 * math.pi * F[3] / C
    far = k * 1e9
    unit = 1.0 / np.sqrt(2.0 * math.pi * k)

    def cot(x: float) -> float:
        return 1.0 / math.tan(x)

    def shed(between: float) -> float:
        wedge = (cot((math.pi + between) / 3.0) + cot((math.pi - between) / 3.0)) / 3.0
        plane = (cot((math.pi + between) / 4.0) + cot((math.pi - between) / 4.0)) / 4.0
        return abs(wedge - math.cos(0.5 * (between - math.pi)) * plane)

    up, down = (math.cos(math.pi / 6.0), 0.5), (math.cos(math.pi / 6.0), -0.5)
    own, image = shed(1.25 * math.pi - math.pi / 6.0), shed(1.25 * math.pi + math.pi / 6.0)
    assert image > 2.0 * own
    for along, outside in ((-1.0, (1.0, -1.0)), (1.0, (-1.0, -1.0))):
        for inside, expected in ((up, own), (down, image)):
            inside = (along * inside[0], inside[1])
            got = rows._corner(k, far, outside, along, inside)
            np.testing.assert_allclose(np.abs(got), expected * unit, rtol=1e-6)
        level_with_it = rows._corner(k, k * 3.0, outside, along, (along, 0.0))
        assert np.array_equal(level_with_it, rows._corner(k, k * 3.0, outside, along))
    # Steeply down past the corner, the image's wave turns by more than the wall's
    # three right angles are from the roof: 60 degrees below the roof, and on at 255.
    steep = (math.cos(math.radians(255.0)), math.sin(math.radians(255.0)))
    below = (math.cos(math.radians(60.0)), -math.sin(math.radians(60.0)))
    got = rows._corner(k, far, (-steep[0], steep[1]), -1.0, (-below[0], below[1]))
    np.testing.assert_allclose(np.abs(got), shed(math.radians(315.0)) * unit, rtol=1e-6)


def test_an_end_over_a_roof_lights_its_first_top_with_its_image_in_the_roof(monkeypatch) -> None:
    """A source half a metre above a roof 2 m up, 4 m before a barrier 1 m high on the
    roof: the march from the source begins at the barrier, from the source's field
    and its image's in the roof, which is rigid whatever the ground is; and the
    barrier's top is lit by the two apart, each from its own direction. Level with
    the roof, the source and its image are one."""
    f = F[3]
    k = 2.0 * math.pi * f / C
    grass = np.conj(surface_impedance(f, 200.0))
    row, reach, zr = [(-5.0, 30.0, 2.0), (4.0, 4.0, 3.0), (10.0, 10.0, 2.5)], 20.0, 2.5
    top = rows.layer_top(2.5, zr, reach, row)
    march, given = parabolic.march, []

    def spy(*args, **more):
        given.append(more["start"])
        return march(*args, **more)

    monkeypatch.setattr(parabolic, "march", spy)
    ready = (f, C, reach, top, parabolic.STEPS_PER_WAVELENGTH, lambda r: grass)
    way = rows._march_from(*ready, 2.5, zr, row)()
    assert (way.first, way.under) == (4.0, 2.0) and np.all(np.isinf(way.ground))
    (own, a, to_own, apart), (image, b, to_image, also) = rows._lit(way, k, ("screen", 4.0, 3.0))
    assert apart and also
    assert (a, to_own) == (math.hypot(4.0, 0.5), (-4.0, -0.5))
    assert (b, to_image) == (math.hypot(4.0, 1.5), (-4.0, -1.5))
    np.testing.assert_allclose(own, np.exp(1j * k * a) / math.sqrt(a))
    np.testing.assert_allclose(image, np.exp(1j * k * b) / math.sqrt(b))
    # The start: the two waves on the heights of the barrier's plane.
    at, field = given[0]
    assert at == 4.0
    z = np.array([3.0, 3.5, 5.0])
    longer = math.sqrt(
        (math.hypot(4.0, 0.5) + math.hypot(16.0, 0.5))
        / (math.hypot(4.0, 1.5) + math.hypot(16.0, 0.5))
    )
    both = rows._wave(k, np.hypot(4.0, z - 2.5)) + longer * rows._wave(k, np.hypot(4.0, z - 1.5))
    np.testing.assert_allclose(field(z), both / 2.0, rtol=1e-12)
    on_it = rows._march_from(*ready, 2.0, zr, row)()
    np.testing.assert_allclose(given[1][1](z), 2.0 * rows._wave(k, np.hypot(4.0, z - 2.0)) / 2.0)
    (own, a, to_own, _), (image, b, to_image, _) = rows._lit(on_it, k, ("screen", 4.0, 3.0))
    assert (a, to_own) == (b, to_image) and np.array_equal(own, image)
    # Any other edge is lit by the marched field.
    assert [wave[3] for wave in rows._lit(way, k, ("screen", 10.0, 2.5))] == [False]


def test_a_roof_s_corner_is_told_the_own_waves_of_an_end_over_the_roof(monkeypatch) -> None:
    """A roof's corner is told where the wave on the roof's side comes from for the
    own wave of an end over that roof and for the wave of the end's image in it, and
    for no other: a wave that the march brings along the roof has no such direction,
    nor has the wave of an end before a building, which meets the corner from outside."""
    f = F[3]
    told = []
    real = rows._corner

    def spy(k, kl, outside, along, inside=None):
        told.append((outside, along, inside))
        return real(k, kl, outside, along, inside)

    monkeypatch.setattr(rows, "_corner", spy)
    # A source 1 m above a roof 2 m up, 2 m before its far corner; a listener 6 m past
    # the corner, 1 m above the ground. Each of the source's two waves, with each of the
    # listener's (its own, and its image's in the ground).
    rows.over_row(f, C, None, 3.0, 8.0, 1.0, [(-5.0, 2.0, 2.0)])
    down, further_down = (6.0, -1.0), (6.0, -3.0)
    assert told == [
        (down, -1.0, (-2.0, 1.0)),
        (further_down, -1.0, (-2.0, 1.0)),
        (down, -1.0, (-2.0, -1.0)),
        (further_down, -1.0, (-2.0, -1.0)),
    ]
    # The two swapped: the roof's near corner, from the listener over the roof.
    del told[:]
    rows.over_row(f, C, None, 1.0, 8.0, 3.0, [(6.0, 13.0, 2.0)])
    down, further_down = (-6.0, -1.0), (-6.0, -3.0)
    assert told == [
        (down, 1.0, (2.0, 1.0)),
        (down, 1.0, (2.0, -1.0)),
        (further_down, 1.0, (2.0, 1.0)),
        (further_down, 1.0, (2.0, -1.0)),
    ]
    # A building between two ends on the ground: each corner from outside, twice.
    del told[:]
    rows.over_row(f, C, None, 1.0, 20.0, 1.0, [(6.0, 13.0, 2.0)])
    assert [(along, inside) for _, along, inside in told] == [(1.0, None)] * 2 + [(-1.0, None)] * 2
    # A barrier on the roof between the source and the far corner: the march brings
    # the corner its wave along the string from the barrier's top.
    del told[:]
    rows.over_row(f, C, None, 3.0, 11.0, 1.0, [(-5.0, 5.0, 2.0), (2.0, 2.0, 3.5)])
    assert [(along, inside) for _, along, inside in told] == [(-1.0, None)] * 2


def test_the_edges_take_the_distance_parameter_of_a_point_source(monkeypatch) -> None:
    """L = a b / (a + b), a and b the distances from the edge to the source and to the
    listener (it changes the validation's levels by less than 0.1 dB)."""
    f = F[3]
    taken = []
    real = rows._screen
    monkeypatch.setattr(
        rows, "_screen", lambda k, kl, *directions: taken.append(kl) or real(k, kl, *directions)
    )
    row = (f, C, None, LIFT + 3.0, 30.0, LIFT - 4.0)
    rows.over_row(*row, [(12.0, 12.0, LIFT)], steep_reflections="leave out")
    a, b = math.hypot(12.0, 3.0), math.hypot(18.0, 4.0)
    k = 2.0 * math.pi * f / C
    np.testing.assert_allclose(taken[0], k * a * b / (a + b))
    # A roof's corners too, on a rigid ground: the near one from the source and from
    # its image in the ground, the far one to the listener and to its image, the other
    # way along the string over the roof.
    real_corner = rows._corner
    monkeypatch.setattr(
        rows, "_corner", lambda k, kl, *more: taken.append(kl) or real_corner(k, kl, *more)
    )
    del taken[:]
    rows.over_row(f, C, None, 9.0, 30.0, 2.0, [(12.0, 20.0, 6.0)])
    past_the_roof = 8.0 + math.hypot(10.0, 4.0)
    ways = [
        (math.hypot(12.0, 3.0), past_the_roof),
        (math.hypot(12.0, 15.0), past_the_roof),
        (math.hypot(20.0, 3.0), math.hypot(10.0, 4.0)),
        (math.hypot(20.0, 3.0), math.hypot(10.0, 8.0)),
    ]
    assert len(taken) == len(ways)
    for kl, (a, b) in zip(taken, ways, strict=True):
        np.testing.assert_allclose(kl, k * a * b / (a + b))


def test_the_level_changes_smoothly_as_a_top_passes_the_plane_of_the_others() -> None:
    """A barrier 1 m past a roof 9 m wide, its top from 5 cm below the roof's plane to
    5 cm above it, the source and the listener a centimetre above the plane: the level
    falls by less than 0.05 dB a centimetre."""
    source, listener = (-9.99, 0.01), (15.01, 0.01)
    tops = np.arange(-0.05, 0.0501, 0.01)
    got = np.array(
        [
            levels(over([(0.0, 9.0, 0.0), (10.0, 10.0, top)], source, listener), source, listener)
            for top in tops
        ]
    )
    steps = np.diff(got, axis=0)
    assert np.all(steps < 0.0) and np.all(steps > -0.05)


def test_the_absorbing_layer_starts_above_everything_and_rises_with_the_range() -> None:
    assert rows.layer_top(1.5, 4.0, 100.0, [(40.0, 50.0, 8.0)]) == 8.0 + rows.HEADROOM_M
    assert rows.layer_top(30.0, 4.0, 100.0, []) == 30.0 + rows.HEADROOM_M
    assert rows.layer_top(1.5, 1.5, 1000.0, [(40.0, 50.0, 8.0)]) == rows.TOP_SLOPE * 1000.0
