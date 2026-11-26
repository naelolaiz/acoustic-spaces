"""The sounds the demo of a scene makes for its sources with a model (a scene source's
``model``): the sound its method's sound power is made from, on the run's clock.

A road vehicle of CNOSSOS-EU (:class:`~acoustic_spaces.scene.RoadVehicleModel`) plays
tyre noise and the firing of its engine, each with the octave spectrum of its part of
the method's sound power at the speed its trajectory goes, in the scene's air
(:func:`acoustic_spaces.demos.road.vehicle_sound`). The sound is silent until the
trajectory sets off, fades in over FADE_S as it does and out over the last FADE_S of
it, and is silent after: the vehicle sounds along its pass, at an RMS of 0.1 where it
sounds, which the feeds take as its sound power."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

#: How long a made sound fades in at the start of its source's trajectory and out at its
#: end (s).
FADE_S = 1.0


def speed_kmh(source) -> float:
    """The speed ``source`` (a scene source with a model) goes at along its trajectory
    (km/h)."""
    from acoustic_spaces.scene.model import steady_speed_m_s

    return 3.6 * steady_speed_m_s(source.position, source.trajectory, source.model.needs)


def pass_times(source) -> tuple[float, float]:
    """When ``source``'s trajectory sets off and when it ends, on the run's clock (s)."""
    from acoustic_spaces.walkthrough import knots

    start = float(source.trajectory.start_s)
    return start, float(knots(source).duration_s)


def made_key(source, rate: int, temperature_c: float, rpm: float | None = None) -> str:
    """What the sound ``source`` makes hangs on, as a short digest: its model (with
    ``rpm`` for a road vehicle's engine, when a run gives it), its speed, its pass on the
    run's clock, the air's temperature and the sample rate."""
    model = source.model.model_dump(mode="json")
    if rpm is not None and "rpm" in model:
        model["rpm"] = float(rpm)
    start, end = pass_times(source)
    what = {
        "model": model,
        "speed_kmh": round(speed_kmh(source), 6),
        "pass_s": [round(start, 6), round(end, 6)],
        "temperature_c": float(temperature_c),
        "rate": int(rate),
    }
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def made_file(folder: Path, source, rate: int, temperature_c: float, rpm=None) -> Path:
    """Where the sound ``source`` makes is kept in ``folder``, named after it and
    :func:`made_key`."""
    return Path(folder) / f"{source.name}_{made_key(source, rate, temperature_c, rpm)}.wav"


def made_sound(source, rate: int, temperature_c: float, rpm: float | None = None) -> np.ndarray:
    """The sound ``source`` (a scene source with a model, its trajectory as the run plays
    it) makes on the run's clock at ``rate``, mono: silent until its trajectory sets off,
    then its model's sound along the pass, faded in and out over FADE_S, at an RMS of 0.1
    (``rpm``, when given, turning a road vehicle's engine)."""
    from acoustic_spaces.demos.passby import faded
    from acoustic_spaces.demos.road import vehicle_power, vehicle_sound

    model = source.model
    start, end = pass_times(source)
    if model.method != "cnossos-eu":  # pragma: no cover - the only method so far
        raise ValueError(f"no sound is made for a {model.method} model")
    power = vehicle_power(model.category, speed_kmh(source), temperature_c)
    x = vehicle_sound(
        power,
        int(rate),
        end - start,
        rpm=float(model.rpm if rpm is None else rpm),
        cylinders=model.cylinders,
    )
    lead = np.zeros(int(round(start * rate)))
    return np.concatenate([lead, faded(x, int(rate), FADE_S)])


def made_words(source, rpm: float | None = None) -> str:
    """What ``source`` plays, in words: "CNOSSOS-EU light motor vehicles at 50 km/h, made
    to the method's sound power (an engine of 4 cylinders at 2000 rpm)"."""
    from acoustic_spaces.demos.road import CATEGORIES

    model = source.model
    kind = CATEGORIES[model.category].name
    turning = float(model.rpm if rpm is None else rpm)
    return (
        f"CNOSSOS-EU {kind} at {round(speed_kmh(source), 1):g} km/h, made to the method's "
        f"sound power (tyres and an engine of {model.cylinders} cylinders at {turning:g} rpm)"
    )


def model_line(source, temperature_c: float) -> str:
    """The stills' line on ``source`` (a scene source with a model, its trajectory as the
    run plays it): its speed and the A-weighted sound power its model gives it there, in
    air at ``temperature_c``: "Sound power of car at 50 km/h: 98.4 dB(A) (CNOSSOS-EU)"."""
    from acoustic_spaces.demos.road import vehicle_power

    model = source.model
    kmh = speed_kmh(source)
    power = vehicle_power(model.category, kmh, temperature_c).sound_power_a_db
    return f"Sound power of {source.name} at {round(kmh, 1):g} km/h: {power:.1f} dB(A) (CNOSSOS-EU)"


def made_credit(source, rpm: float | None = None) -> str:
    """The stills' credit for the sound ``source`` makes."""
    from acoustic_spaces.demos.passby import CYLINDER_WORDS
    from acoustic_spaces.demos.road import CATEGORIES

    model = source.model
    kind = CATEGORIES[model.category]
    cylinders = CYLINDER_WORDS.get(model.cylinders, str(model.cylinders))
    turning = float(model.rpm if rpm is None else rpm)
    tyres = "tyre noise and a" if kind.rolling_a is not None else "a"
    parts = "each" if kind.rolling_a is not None else "its sound"
    return (
        f"Sound made up for {source.name}: {tyres} {cylinders}-cylinder engine at {turning:g} "
        f"rpm, {parts} with the octave spectrum of CNOSSOS-EU's {kind.name} (category "
        f"{model.category}) at {round(speed_kmh(source), 1):g} km/h (Directive 2002/49/EC as "
        f"amended in 2021); faded in and out over {FADE_S:g} s"
    )


__all__ = [
    "FADE_S",
    "made_credit",
    "made_file",
    "made_key",
    "made_sound",
    "made_words",
    "model_line",
    "pass_times",
    "speed_kmh",
]
