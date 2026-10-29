"""Helicopters (:mod:`acoustic_spaces.rotorcraft`): the trim, the blade tones against
Gutin's formula and the integral round the turn, Pegg's broadband noise against his
worked example, and the sound a helicopter sends."""

import math

import numpy as np
import pytest
from scipy.special import jv

from acoustic_spaces import rotorcraft as rc
from acoustic_spaces.air import density
from acoustic_spaces.directivity import _sphere
from acoustic_spaces.emission import Mixture, Periodic, Sum

MAIN = rc.Rotor(3, 5.345, 0.35, 394.0, turning="clockwise", thickness=0.09)
TAIL = rc.Rotor(2, 0.93, 0.185, 2086.0, hub_m=(-6.67, 0.0, -1.0), thrust=(0.0, 1.0, 0.0))
LIGHT = rc.Helicopter(2250.0, MAIN, TAIL, drag_area_m2=0.674)


def test_a_trim_balances_the_helicopter() -> None:
    trim = rc.trim(LIGHT, 60.0, 25.0)
    rho = density(25.0)
    weight, drag = 2250.0 * rc.GRAVITY, 0.5 * rho * 60.0**2 * 0.674
    thrust = trim.main.thrust_n * trim.main.axis
    np.testing.assert_allclose(thrust, [drag, 0.0, weight], rtol=1e-12, atol=1e-9)
    # The tail rotor holds the main rotor's torque.
    assert trim.tail.thrust_n * LIGHT.tail_arm_m == pytest.approx(trim.main.torque_nm)
    assert LIGHT.tail_arm_m == pytest.approx(6.67)
    assert trim.power_w > trim.main.torque_nm * MAIN.omega > 0.0
    assert trim.lift_coefficient("main") == pytest.approx(
        6.0 * trim.main.thrust_n / (rho * MAIN.blade_area_m2 * MAIN.tip_speed**2)
    )
    # Hovering, momentum theory's induced velocity.
    hover = rc.trim(LIGHT, 0.0)
    assert hover.main.induced_m_s == pytest.approx(
        math.sqrt(weight / (2.0 * density(15.0) * MAIN.disc_m2)), rel=1e-9
    )
    # Flying, the induced power falls and the parasite power grows.
    assert trim.main.induced_m_s < hover.main.induced_m_s


def _hub_loads(thrust: float, torque: float, radius: float, blades: int) -> rc._Blade:
    """All of a rotor's thrust and torque on one station, no thickness: Gutin's rotor."""
    none = np.zeros((1, 1))
    return rc._Blade(
        np.array([radius]),
        np.array([thrust / blades]),
        np.array([torque / (blades * radius)]),
        none,
        none,
        none,
    )


@pytest.mark.parametrize("spin", [1, -1])
def test_tones_in_hover_as_gutin_gives_them(spin: int) -> None:
    blades, omega, c = 3, 41.3, 343.0
    radius, thrust, torque = 0.8 * 5.35, 22000.0, 9000.0
    theta = np.radians(np.arange(5.0, 180.0, 10.0))
    own = np.column_stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)])
    amplitudes = rc._harmonics(
        own,
        own,
        np.array([1.0, 0.0, 0.0]),
        blades=blades,
        omega=omega,
        spin=spin,
        blade=_hub_loads(thrust, torque, radius, blades),
        speed=0.0,
        rho=1.2,
        c=c,
        count=6,
    )
    m = np.arange(1, 7)[None, :]
    # Gutin (1936): the rms far field R p of harmonic m, thrust along +z.
    gutin = (
        m * blades * omega / (2.0 * math.sqrt(2.0) * math.pi * c)
        * np.abs(-thrust * np.cos(theta)[:, None] + torque * c / (omega * radius**2))
        * np.abs(jv(m * blades, m * blades * omega * radius * np.sin(theta)[:, None] / c))
    )  # fmt: skip
    np.testing.assert_allclose(math.sqrt(2.0) * np.abs(amplitudes), gutin, rtol=1e-9, atol=1e-12)


def _round_the_turn(trim: rc.Trim, which: str, ways: np.ndarray, count: int) -> np.ndarray:
    """The harmonics as the far field's integral over a turn, summed numerically over
    the blade's points where they truly are (the straight chord), each a moving force
    or source: Farassat's formulation 1A far from the rotor."""
    rotor, load = trim._rotor(which)
    axes = rc._basis(load.axis)
    blade = rc._blade(rotor, load)
    c, v, omega, s = trim.sound_speed, trim.airspeed_m_s, rotor.omega, rotor.spin
    a, e1, e2 = axes[2], axes[0], axes[1]
    n_t = 2048
    psi = s * omega * np.arange(n_t) * (2.0 * math.pi / omega) / n_t
    radial = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
    ahead = s * np.cross(a, radial)
    doppler = 1.0 - v * ways[:, 0] / c
    k = rotor.blades * np.arange(1, count + 1)
    t = np.arange(n_t) / n_t * 2.0 * math.pi / omega

    def integral(strength: np.ndarray, where: np.ndarray) -> np.ndarray:
        tau = t[None, :] - (where @ ways.T).T / (c * doppler[:, None])
        return (
            np.einsum(
                "nt,nmt->nm", strength, np.exp(-1j * k[None, :, None] * omega * tau[:, None, :])
            )
            / n_t
        )

    total = np.zeros((len(ways), count), dtype=complex)
    xi = (blade.angles, blade.outward)
    for i, r in enumerate(blade.radii):
        force = -blade.lift[i] * a[None, :] + blade.drag[i] * ahead
        total += integral((force @ ways.T).T / c, r * radial)
        speed = omega * r + v * ahead[:, 0]
        for g in range(blade.areas.shape[1]):
            offset = math.tan(xi[0][i, g]) * r
            where = r * radial + offset * ahead
            total += integral(
                np.broadcast_to(trim.density * speed * blade.areas[i, g], (len(ways), n_t)), where
            )
    amplitudes = rotor.blades * 1j * k * omega / (4.0 * math.pi * doppler[:, None] ** 2) * total
    own = ways @ axes.T
    return amplitudes * np.exp(1j * k * s * np.arctan2(own[:, 1], own[:, 0])[:, None])


@pytest.mark.parametrize("which", ["main", "tail"])
def test_tones_as_the_integral_round_the_turn(which: str) -> None:
    trim = rc.trim(LIGHT, 60.0, 25.0)
    rotor, load = trim._rotor(which)
    axes = rc._basis(load.axis)
    ways = _sphere(12)
    count = 8
    ours = rc._harmonics(
        ways @ axes.T,
        ways,
        axes @ np.array([1.0, 0.0, 0.0]),
        blades=rotor.blades,
        omega=rotor.omega,
        spin=rotor.spin,
        blade=rc._blade(rotor, load),
        speed=60.0,
        rho=trim.density,
        c=trim.sound_speed,
        count=count,
    )
    exact = _round_the_turn(trim, which, ways, count)
    loudest = np.max(np.abs(exact), axis=0)
    assert np.max(np.abs(ours - exact) / loudest) < 1e-4


def test_pegg_s_worked_example() -> None:
    # Pegg (NASA TM-80200), example I: a hovering transport helicopter, 61.6 m from
    # the main rotor 5 degrees under its disc, 62.2 m from the tail rotor 10 degrees off
    # its axis.
    main = rc.pegg_level_db(208.0, 344.0, 18.58, 0.438) + 10.0 * math.log10(
        (math.cos(math.radians(85.0)) ** 2 + 0.1) / 61.6**2
    )
    tail = rc.pegg_level_db(202.0, 344.0, 3.44, 0.182) + 10.0 * math.log10(
        (math.cos(math.radians(10.0)) ** 2 + 0.1) / 62.2**2
    )
    assert main == pytest.approx(84.5, abs=0.05) and tail == pytest.approx(82.5, abs=0.05)
    assert rc.pegg_peak_hz(69420.0, 208.0) == pytest.approx(289.0, abs=0.5)
    assert rc.pegg_peak_hz(5206.5, 202.0) == pytest.approx(544.0, abs=1.0)
    assert rc.pegg_lift_db(0.48) == pytest.approx(rc.pegg_lift_db(0.48 + 1e-12), abs=0.01)
    centres, levels = rc.pegg_bands(289.0)
    # The peak's band is the 315 Hz one; its example's bands an octave apart.
    peak = int(np.argmax(levels))
    assert centres[peak] == pytest.approx(316.2, abs=0.1)
    np.testing.assert_allclose(
        levels[peak - 9 : peak + 16 : 3],
        [-19.5, -15.3, -11.7, -7.5, -11.5, -12.1, -16.5, -17.0, -21.8],
    )


def test_the_broadband_noise_in_its_bands() -> None:
    trim = rc.trim(LIGHT, 60.0, 25.0)
    rate, n = 48_000, 4 * 48_000
    noise = rc.broadband(trim, samples=n, sample_rate=rate, grid_deg=10.0)
    assert isinstance(noise, Mixture) and noise.signals.shape == (2, n)
    peak = rc.pegg_peak_hz(trim.main.thrust_n, MAIN.tip_speed)
    centres, levels = rc.pegg_bands(peak)
    spectrum = np.abs(np.fft.rfft(noise.signals[0])) ** 2 * 2.0 / n**2
    f = np.fft.rfftfreq(n, 1.0 / rate)
    for centre, level in zip(centres, levels, strict=True):
        inside = (f >= centre * 10**-0.05) & (f < centre * 10**0.05)
        if centre < rate / 2 * 10**-0.05 and inside.any():
            assert 10.0 * math.log10(spectrum[inside].sum()) == pytest.approx(level, abs=1e-6)
    # Straight down, Pegg's pattern at its loudest, the convective amplification none.
    below = noise.sent(0, 1, np.array([[0.0, 0.0, -1.0]]))[0]
    level_db = rc.pegg_level_db(
        MAIN.tip_speed, trim.sound_speed, MAIN.blade_area_m2, trim.lift_coefficient("main")
    )
    cos = trim.main.axis[2]
    weight = rc.REFERENCE_PA * 10 ** (level_db / 20) * math.sqrt(cos**2 + 0.1)
    assert noise.weights[0, 0, 0] == pytest.approx(weight)
    assert below == pytest.approx(
        weight * noise.signals[0, 0] + noise.weights[0, 0, 1] * noise.signals[1, 0]
    )


def test_the_sound_a_helicopter_sends() -> None:
    trim = rc.trim(LIGHT, 60.0, 25.0)
    sent = rc.emission(trim, seconds=0.5, grid_deg=30.0)
    assert isinstance(sent, Sum) and sent.length == 24_000
    main, tail, noise = sent.parts
    assert isinstance(main, Periodic) and isinstance(tail, Periodic) and isinstance(noise, Mixture)
    assert main.period_s == pytest.approx(60.0 / (394.0 * 3))
    assert tail.period_s == pytest.approx(60.0 / (2086.0 * 2))
    np.testing.assert_allclose(tail.centre, TAIL.hub_m)
    # The harmonics kept reach QUIET_DB under the loudest.
    assert main.table.shape[-1] % 16 == 0 and main.table.shape[-1] >= 16 * 10
    balloon = sent.directivity()
    assert balloon.table.shape[-1] == 8 and np.all(balloon.table > 0)
    assert np.mean(sent.averaged() ** 2) > 0


def test_what_a_rotor_needs() -> None:
    with pytest.raises(ValueError, match="counter-clockwise or clockwise"):
        rc.Rotor(3, 5.0, 0.3, 400.0, turning="anticlockwise")
    with pytest.raises(ValueError, match="two blades"):
        rc.Rotor(1, 5.0, 0.3, 400.0)
    with pytest.raises(ValueError, match="zero or more"):
        rc.trim(LIGHT, -1.0)
    with pytest.raises(ValueError, match="'main' and 'tail'"):
        rc.trim(LIGHT, 10.0).lift_coefficient("front")
