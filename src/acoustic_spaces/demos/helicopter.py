"""The helicopter of a scene's source of a rotorcraft model (``model: {method:
rotorcraft}``, the flyover's and the stadium's): the sound its rotors send each way, and
the sound power and directivity that go with it in the scene.

The helicopter is a source, a point at the main rotor's hub
with a trajectory at one height and one speed, facing the way it flies (``face_forward``). Its
rotors are trimmed for level flight (:func:`acoustic_spaces.rotorcraft.trim`) at its
airspeed: the trajectory's speed over the ground less the wind at its height, from the scene's
atmosphere (its own weather, not its zones'), in the scene's air. The trim faces the
way it moves through the air; facing along the trajectory instead, the helicopter does not
turn into a wind from the side (it would fly crabwise), so a scene's wind is best along
the trajectory or none. What the rotors send (:func:`acoustic_spaces.rotorcraft.emission`)
plays down each early path; the scene's source gets the sound power and the directivity
of that emission (:meth:`~acoustic_spaces.emission.Emission.sound_power_db`,
:meth:`~acoustic_spaces.emission.Emission.directivity`), for the levels the simulation
reports and the late part. Nothing of the simulation uses this module.
"""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np

#: The noise's seed, so that every step makes the same sound.
SEED = 0
#: Seconds of the rotors' sound its sound power and directivity are taken from: the
#: tones repeat, and the noise has its bands' power exactly in any length.
POWER_S = 1.0


#: Its broadband noise's models, in words.
BROADBAND = {
    "bpm": "Brooks, Pope and Marcolini's airfoil self-noise of each blade section",
    "pegg": "Pegg's method (NASA TM-80200)",
}


#: The rate the sound power and the directivity of a flight are found at (Hz): every
#: octave band lies under its half.
POWER_RATE = 48_000


def helicopters() -> dict:
    """The helicopters a scene's rotorcraft model flies, by its name of them
    (:class:`acoustic_spaces.rotorcraft.Helicopter`)."""
    from acoustic_spaces import rotorcraft

    return {"as350_b3": rotorcraft.AS350_B3}


@functools.lru_cache(maxsize=4)
def _rotors(
    airspeed_m_s: float,
    temperature_c: float,
    pressure_kpa: float,
    rate: int,
    broadband: str,
    helicopter: str = "as350_b3",
):
    """What of ``helicopter``'s sound does not depend on how long it lasts, found once for
    all the lengths :func:`_sent` makes: its trim, its two rotors' tones (one period in
    each direction) and, by ``broadband`` "bpm", its broadband noise's bands
    (:func:`acoustic_spaces.rotorcraft.bpm_powers`)."""
    from acoustic_spaces import rotorcraft

    flier = dataclasses.replace(helicopters()[helicopter], broadband=broadband)
    trim = rotorcraft.trim(flier, airspeed_m_s, temperature_c, pressure_kpa)
    tones = tuple(
        rotorcraft.tones(trim, which, samples=0, sample_rate=rate) for which in ("main", "tail")
    )
    bands = rotorcraft.bpm_powers(trim) if broadband == "bpm" else None
    return trim, tones, bands


@functools.lru_cache(maxsize=4)
def _sent(
    airspeed_m_s: float,
    temperature_c: float,
    pressure_kpa: float,
    rate: int,
    n: int,
    broadband: str,
    helicopter: str = "as350_b3",
):
    """``helicopter``'s trim and the ``n`` samples it sends at ``rate``, its broadband
    noise by ``broadband``: :func:`acoustic_spaces.rotorcraft.emission`, sample for
    sample, from :func:`_rotors` and ``n`` samples of noise."""
    from acoustic_spaces import rotorcraft
    from acoustic_spaces.emission import Sum

    trim, tones, bands = _rotors(
        airspeed_m_s, temperature_c, pressure_kpa, rate, broadband, helicopter
    )
    noise = rotorcraft.broadband(trim, samples=n, sample_rate=rate, seed=SEED, bands=bands)
    return trim, Sum((*(dataclasses.replace(tone, samples=n) for tone in tones), noise))


@functools.lru_cache(maxsize=16)
def flight_sound(
    helicopter: str,
    broadband: str,
    airspeed_m_s: float,
    temperature_c: float,
    pressure_kpa: float,
):
    """The sound power per octave band (dB re 1 pW, a tuple) and the directivity (a
    :class:`~acoustic_spaces.directivity.Balloon`, octave bands) of what ``helicopter``'s
    rotors send in level flight at ``airspeed_m_s`` in air at ``temperature_c`` and
    ``pressure_kpa``, its broadband noise by ``broadband``: of POWER_S of it at
    POWER_RATE (:meth:`~acoustic_spaces.emission.Emission.sound_power_db`,
    :meth:`~acoustic_spaces.emission.Emission.directivity`), what a scene's rotorcraft
    model gives its source."""
    rate = POWER_RATE
    trim, sent = _sent(
        airspeed_m_s,
        temperature_c,
        pressure_kpa,
        rate,
        round(POWER_S * rate),
        broadband,
        helicopter,
    )
    level = sent.sound_power_db(trim.density, trim.sound_speed)
    name = f"{helicopters()[helicopter].name}'s rotors at {airspeed_m_s:g} m/s through the air"
    return tuple(round(float(v), 6) for v in level), sent.directivity(name=name)


def flight_emission(source, rate: int, temperature_c: float, pressure_kpa: float):
    """What ``source`` (a scene source with a rotorcraft model, its trajectory as the
    run plays it) sends each way on the run's clock at ``rate``: silent until its
    trajectory sets off, then its rotors' sound along its pass, faded in and out over
    FADE_S, as loud as a made sound is for its sound power (:func:`emission_gain`)."""
    from acoustic_spaces.demos.models import FADE_S, pass_times
    from acoustic_spaces.emission import Delayed, Faded

    model = source.model
    start, end = pass_times(source)
    n = int(round((end - start) * rate))
    trim, sent = _sent(
        source.model_speed_m_s(),
        temperature_c,
        pressure_kpa,
        int(rate),
        n,
        model.broadband,
        model.helicopter,
    )
    faded = Faded(sent, min(int(FADE_S * rate), n // 2), n)
    gain = emission_gain(source.level_db, trim.density, trim.sound_speed)
    return Delayed(faded.scaled(gain), int(round(start * rate)))


def emission_gain(power_db, density: float, sound_speed: float) -> float:
    """The gain that brings an emission in pascals 1 m away whose sound power per octave
    band is ``power_db`` (dB re 1 pW) to a made sound's units: the mean square over the
    sphere of what it sends in those bands to SOUND_RMS squared, as a made sound's RMS
    is its sound power (:mod:`acoustic_spaces.demos.models`)."""
    from acoustic_spaces.demos.models import SOUND_RMS

    watts = float(np.sum(10.0 ** (np.asarray(power_db, dtype=float) / 10.0))) * 1e-12
    mean_square = watts * density * sound_speed / (4.0 * math.pi)
    return SOUND_RMS / math.sqrt(mean_square)


def below_sound(source, rate: int, temperature_c: float, pressure_kpa: float) -> np.ndarray:
    """What ``source`` (as :func:`flight_emission`) sends straight down, dry, on the
    run's clock: its feed, to hear."""
    sent = flight_emission(source, rate, temperature_c, pressure_kpa)
    n = sent.length
    return sent.sent(0, n, np.tile([0.0, 0.0, -1.0], (n, 1)))


__all__ = [
    "BROADBAND",
]
