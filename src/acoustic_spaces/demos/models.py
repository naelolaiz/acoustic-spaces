"""The sounds the demo of a scene makes for its sources with a model (a scene source's
``model``): the sound its method's sound power is made from, on the run's clock.

- A road vehicle of CNOSSOS-EU (:class:`~acoustic_spaces.scene.RoadVehicleModel`) plays
  tyre noise and the firing of its engine, each with the octave spectrum of its part of
  the method's sound power at the speed its trajectory goes, in the scene's air
  (:func:`acoustic_spaces.demos.road.vehicle_sound`).
- A siren (:class:`~acoustic_spaces.scene.SirenModel`) plays its square wave in its
  pattern (:func:`acoustic_spaces.demos.siren.siren_sound`).
- A horn (:class:`~acoustic_spaces.scene.HornModel`) plays its blasts
  (:func:`acoustic_spaces.demos.horn.horn_sound`), from the start of the first.
- A train or a tram of Schall 03 (:class:`~acoustic_spaces.scene.RailVehicleModel`)
  plays noise at each of its points at its height, each with the octave spectrum of its
  sound power (:func:`acoustic_spaces.demos.rail.point_sound`), a column each: the
  first the source's own sound, the others those of the points behind it
  (:func:`trail`), each at its sound power against the first's.

The sound is silent until the source's trajectory sets off and after it ends, and
plays along its pass at an RMS of 0.1, which the feeds take as its sound power: over
the whole of its sounding for a sound that sounds all along, faded in over FADE_S as
the trajectory sets off and out over its last FADE_S; over whole periods of a horn's
blasts and pauses, which rise and fall of themselves. The run's options set what the
model leaves to a run (an engine's rpm, a siren's pattern, a horn's signal) in the
source's model before it is made (:func:`acoustic_spaces.demos.scene_demo.changed`)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

#: How long a made sound that sounds all along fades in at the start of its source's
#: trajectory and out at its end (s).
FADE_S = 1.0
#: The RMS of a made sound (full scale): over its sounding, which is its sound power.
SOUND_RMS = 0.1
#: What each method's made sound is in a few words, for the messages.
METHOD_WORDS = {
    "cnossos-eu": "a road vehicle",
    "siren": "a siren",
    "horn": "a horn",
    "schall-03": "a rail vehicle",
}
#: Each rail model's height, for the seeds of its points' noise: the old pass-by demo's
#: order of a train's sources.
RAIL_SEEDS = {"rail_top": 0, "roof": 1, "pantograph": 2}


def speed_kmh(source) -> float | None:
    """The speed ``source`` (a scene source with a model) goes at along its trajectory
    (km/h), where its method takes one (None for one that takes no speed)."""
    from acoustic_spaces.scene.model import steady_speed_m_s

    if source.model.needs is None:
        return None
    return 3.6 * steady_speed_m_s(source.position, source.trajectory, source.model.needs)


def driven_kmh(source) -> float:
    """The speed (km/h) ``source``, whose model takes one (a road vehicle's), goes at
    along its trajectory (:func:`speed_kmh`)."""
    kmh = speed_kmh(source)
    if kmh is None:
        raise ValueError(f"{source.name}'s {source.model.method} model takes no speed")
    return kmh


def rail_points(source) -> list:
    """The points of ``source`` (a scene source with a rail model, its trajectory as the
    run plays it) at the speed it goes: :meth:`~acoustic_spaces.scene.RailVehicleModel.points`."""
    return source.model.points(driven_kmh(source) / 3.6)


def trail(source) -> tuple[float, tuple[tuple[float, np.ndarray], ...]]:
    """How the points of ``source`` (a scene source with a model, its trajectory as the
    run plays it) follow its trajectory: how long after its trajectory its first point
    gets to a place (s: how far behind the front it is, at its speed), and each other
    point's lag behind the first (s) with its sound power over the first's per octave
    band (dB); (0, ()) for a model of one point, a rail model's alone having more."""
    if source.model.method != "schall-03":
        return 0.0, ()
    points = rail_points(source)
    speed = driven_kmh(source) / 3.6
    first = points[0]
    behind = tuple(
        ((p.behind_m - first.behind_m) / speed, np.asarray(p.power_db) - first.power_db)
        for p in points[1:]
    )
    return first.behind_m / speed, behind


def recorded_trail(source, x: np.ndarray) -> np.ndarray:
    """(samples, points behind) what the points behind the first of ``source`` (a scene
    source with a model) play of ``x``, the recording its first plays in place of its
    made sound: each from another place in it on, round to its start, its RMS against
    the first's as its sound power is in all; none for a model of one point."""
    if source.model.method != "schall-03":
        return np.zeros((len(x), 0))
    points = rail_points(source)
    first = np.sum(10.0 ** (np.asarray(points[0].power_db) / 10.0))
    n = len(x)
    columns = [
        np.roll(x, k * n // len(points))
        * math.sqrt(float(np.sum(10.0 ** (np.asarray(p.power_db) / 10.0)) / first))
        for k, p in enumerate(points[1:], start=1)
    ]
    return np.column_stack(columns) if columns else np.zeros((n, 0))


def pass_times(source) -> tuple[float, float]:
    """When ``source``'s trajectory sets off and when it ends, on the run's clock (s)."""
    from acoustic_spaces.walkthrough import knots

    start = float(source.trajectory.start_s)
    return start, float(knots(source).duration_s)


def made_key(source, rate: int, temperature_c: float) -> str:
    """What the sound ``source`` makes hangs on, as a short digest: its model (as the run
    sets it), its speed where the method takes one, its pass on the run's clock, the
    air's temperature and the sample rate."""
    speed = speed_kmh(source)
    start, end = pass_times(source)
    what = {
        "model": source.model.model_dump(mode="json"),
        "speed_kmh": None if speed is None else round(speed, 6),
        "pass_s": [round(start, 6), round(end, 6)],
        "temperature_c": float(temperature_c),
        "rate": int(rate),
    }
    return hashlib.sha256(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def made_file(folder: Path, source, rate: int, temperature_c: float) -> Path:
    """Where the sound ``source`` makes is kept in ``folder``, named after it and
    :func:`made_key`."""
    return Path(folder) / f"{source.name}_{made_key(source, rate, temperature_c)}.wav"


def horn_kind(model):
    """The :class:`~acoustic_spaces.demos.horn.Kind` a horn model sounds."""
    from acoustic_spaces.demos.horn import KINDS

    return KINDS[(model.kind, model.signal)]


def fade_s(model) -> float:
    """How long the made sound of ``model`` fades in and out over (s): FADE_S, or a
    horn's rise, its blasts rising and falling of themselves."""
    return horn_kind(model).rise_s if model.method == "horn" else FADE_S


def made_sound(source, rate: int, temperature_c: float) -> np.ndarray:
    """The sound ``source`` (a scene source with a model, its trajectory and model as the
    run plays them) makes on the run's clock at ``rate``: silent until its trajectory
    sets off, then its model's sound along the pass at an RMS of SOUND_RMS, faded in and
    out (:func:`fade_s`); mono, but for a rail model of several points, a column each
    (its first the source's, :func:`trail`), each faded so, the others at their sound
    power against the first's."""
    from acoustic_spaces.demos.passby import faded

    model = source.model
    start, end = pass_times(source)
    duration = end - start
    if model.method == "cnossos-eu":
        from acoustic_spaces.demos.road import vehicle_power, vehicle_sound

        power = vehicle_power(model.category, driven_kmh(source), temperature_c)
        x = vehicle_sound(
            power, int(rate), duration, rpm=float(model.rpm), cylinders=model.cylinders
        )
    elif model.method == "siren":
        from acoustic_spaces.demos.siren import siren_sound

        x = siren_sound(model.pattern, int(rate), duration)
    elif model.method == "horn":
        from acoustic_spaces.demos.horn import horn_sound

        x = horn_sound(int(rate), duration, horn_kind(model))
    else:
        from acoustic_spaces.demos.rail import point_sound

        points = rail_points(source)
        reference = np.asarray(points[0].power_db)
        seed = 1000 * RAIL_SEEDS[model.height]
        x = np.column_stack(
            [
                point_sound(p.power_db, int(rate), duration, reference_db=reference, seed=seed + k)
                for k, p in enumerate(points)
            ]
        )
        x = x[:, 0] if x.shape[1] == 1 else x
    lead = np.zeros((int(round(start * rate)), *x.shape[1:]))
    return np.concatenate([lead, faded(x, int(rate), fade_s(model))])


def input_gain(source, x: np.ndarray, rate: int) -> float | None:
    """The gain of ``x``, a recording that ``source`` (a scene source with a model) plays
    in place of its made sound, so that it plays as loud as that sound: a horn's blasts
    as loud as the made horn's, whatever its pauses
    (:func:`~acoustic_spaces.demos.horn.recorded_gain`); any other its gated RMS brought
    to SOUND_RMS. None when it is silent."""
    if source.model.method == "horn":
        from acoustic_spaces.demos.horn import recorded_gain

        return recorded_gain(x, int(rate), horn_kind(source.model))
    from acoustic_spaces.demos.audio import gated_rms

    rms = gated_rms(x, int(rate))
    return SOUND_RMS / rms if rms > 0.0 else None


def _speed_words(source) -> str:
    speed = speed_kmh(source)
    return "" if speed is None else f" at {round(speed, 1):g} km/h"


def made_words(source) -> str:
    """What ``source`` plays, in words: "CNOSSOS-EU light motor vehicles at 50 km/h, made
    to the method's sound power (tyres and an engine of 4 cylinders at 2000 rpm)", "a
    siren's wail, made to the sound power of the 100 W siren NBS measured"."""
    from acoustic_spaces.demos.road import CATEGORIES

    model = source.model
    if model.method == "schall-03":
        return (
            f"{_rail_train(source)}: noise at {_rail_points(source)}, each made to Schall 03's "
            "sound power there"
        )
    if model.method == "siren":
        return f"a siren's {_pattern(model.pattern)}, made to {_siren_loudness(model)}"
    if model.method == "horn":
        kind = horn_kind(model)
        noun = _horn_noun(model)
        own = "'" if noun.endswith("s") else "'s"
        return f"{noun}{own} blasts, made to {_horn_loudness(model)}"
    kind = CATEGORIES[model.category].name
    return (
        f"CNOSSOS-EU {kind}{_speed_words(source)}, made to the method's sound power (tyres "
        f"and an engine of {model.cylinders} cylinders at {model.rpm:g} rpm)"
    )


def model_line(source, temperature_c: float) -> str:
    """The stills' line on ``source`` (a scene source with a model, its trajectory as the
    run plays it): its speed where the method takes one and the A-weighted sound power
    its model gives it, in air at ``temperature_c``: "Sound power of car at 50 km/h:
    98.4 dB(A) (CNOSSOS-EU)", "Sound power of police_car_siren: 130.4 dB(A) (a siren,
    wail)"; a rail model's, that of all its points: "Sound power of tram at 50 km/h:
    113.5 dB(A) (Schall 03, its 4 bogies at the rail top)"."""
    from acoustic_spaces.parameters import a_weighted

    model = source.model
    if model.method == "schall-03":
        from acoustic_spaces.demos.rail import train_power_db

        power = a_weighted(train_power_db(rail_points(source)))
        what = f"Schall 03, {_rail_points(source)}"
    elif model.method == "cnossos-eu":
        from acoustic_spaces.demos.road import vehicle_power

        kmh = driven_kmh(source)
        power = vehicle_power(model.category, kmh, temperature_c).sound_power_a_db
        what = "CNOSSOS-EU"
    else:
        power = a_weighted(np.asarray(model.power_db(None, temperature_c), dtype=float))
        what = (
            f"a siren, {_pattern(model.pattern)}"
            if model.method == "siren"
            else f"{_horn_noun(model)} in blasts, on average"
        )
    return f"Sound power of {source.name}{_speed_words(source)}: {power:.1f} dB(A) ({what})"


def made_credit(source) -> str:
    """The stills' credit for the sound ``source`` makes."""
    from acoustic_spaces.demos.passby import CYLINDER_WORDS
    from acoustic_spaces.demos.road import CATEGORIES

    model = source.model
    if model.method == "schall-03":
        return (
            f"Sound made up for {source.name}: noise at {_rail_points(source)}, each with the "
            "octave spectrum of Schall 03's sources there (16. BImSchV, Anlage 2, 2014), "
            f"{_rail_train(source)}; faded in and out over {FADE_S:g} s"
        )
    if model.method == "siren":
        return (
            f"Sound made up for {source.name}: a siren's {_pattern(model.pattern)}, a square "
            f"wave, {_siren_loudness(model)}; faded in and out over {FADE_S:g} s"
        )
    if model.method == "horn":
        kind = horn_kind(model)
        tones = " and ".join(f"{f:g}" for f in kind.hz)
        how = " in turn," if kind.in_turn else " together,"
        return (
            f"Sound made up for {source.name}: {_horn_noun(model)}, {tones} Hz"
            f"{how} in blasts, {_horn_loudness(model)}"
        )
    kind = CATEGORIES[model.category]
    cylinders = CYLINDER_WORDS.get(model.cylinders, str(model.cylinders))
    tyres = "tyre noise and a" if kind.rolling_a is not None else "a"
    parts = "each" if kind.rolling_a is not None else "its sound"
    speed = driven_kmh(source)
    return (
        f"Sound made up for {source.name}: {tyres} {cylinders}-cylinder engine at "
        f"{model.rpm:g} rpm, {parts} with the octave spectrum of CNOSSOS-EU's {kind.name} "
        f"(category {model.category}) at {round(speed, 1):g} km/h (Directive 2002/49/EC as "
        f"amended in 2021); faded in and out over {FADE_S:g} s"
    )


def made_brief(source) -> str:
    """The sound ``source`` makes in a few words, for a credit of several made sounds
    (:func:`made_credits`): "tyres and a four-cylinder engine at 2000 rpm (CNOSSOS-EU's
    light motor vehicles at 50 km/h)", "a siren's wail, a square wave, as loud as the
    100 W siren NBS measured", "disc horns, 360 and 430 Hz in turn, in blasts, 110 dB(A)
    at 2 m (Hella's S77)", "noise at its 4 bogies at the rail top (Schall 03's low-floor
    tram at 50 km/h)"."""
    from acoustic_spaces.demos.passby import CYLINDER_WORDS
    from acoustic_spaces.demos.rail import TRAINS
    from acoustic_spaces.demos.road import CATEGORIES

    model = source.model
    if model.method == "schall-03":
        # Its short name without its article: "A low-floor tram", Schall 03's low-floor tram.
        train = TRAINS[model.train].short.split(" ", 1)[1]
        return (
            f"noise at {_rail_points(source)} (Schall 03's {train} at "
            f"{round(driven_kmh(source), 1):g} km/h)"
        )
    if model.method == "siren":
        nbs = "the 100 W siren NBS measured"
        if model.gain_db == 0.0:
            loud = f"as loud as {nbs}"
        else:
            louder = "louder" if model.gain_db > 0.0 else "quieter"
            loud = f"{abs(model.gain_db):g} dB {louder} than {nbs}"
        return f"a siren's {_pattern(model.pattern)}, a square wave, {loud}"
    if model.method == "horn":
        kind = horn_kind(model)
        tones = " and ".join(f"{f:g}" for f in kind.hz)
        how = " in turn," if kind.in_turn else " together,"
        each = " each note" if kind.in_turn else ""
        return (
            f"{_horn_noun(model)}, {tones} Hz{how} in blasts, "
            f"{kind.on_axis_db:g} dB({kind.weighting}){each} at {kind.at_m:g} m "
            f"({HORN_BRIEF[model.kind]})"
        )
    kind = CATEGORIES[model.category]
    tyres = "tyres and " if kind.rolling_a is not None else ""
    cylinders = CYLINDER_WORDS.get(model.cylinders, str(model.cylinders))
    return (
        f"{tyres}a {cylinders}-cylinder engine at {model.rpm:g} rpm (CNOSSOS-EU's "
        f"{kind.name} at {round(driven_kmh(source), 1):g} km/h)"
    )


def made_credits(made: list[tuple[str, str, bool]]) -> str:
    """The stills' credit of several made sounds, ``made`` each source's name, its sound
    in a few words (:func:`made_brief`) and whether it fades in and out (a horn's blasts
    rise and fall of themselves): "Sounds made up for police_car: tyres and ...;
    police_car_siren: a siren's wail, ...; faded in and out over 1 s"."""
    said = "; ".join(f"{name}: {brief}" for name, brief, _fades in made)
    unfaded = [name for name, _brief, fades in made if not fades]
    fade = f"; faded in and out over {FADE_S:g} s"
    if unfaded and len(unfaded) < len(made):
        fade += ", not " + " and ".join(unfaded)
    elif unfaded:
        fade = ""
    return f"Sounds made up for {said}{fade}"


def _rail_train(source) -> str:
    """A rail model's train in words: "a low-floor tram with air conditioning at 50 km/h
    on track set in the street"."""
    from acoustic_spaces.demos.rail import TRACKS, TRAINS, TRAM_TRACKS

    model = source.model
    train = TRAINS[model.train]
    lay = (TRAM_TRACKS if train.units[0].kind.tram else TRACKS)[model.track].what
    return f"{train.title}{_speed_words(source)} on {lay}"


def _rail_points(source) -> str:
    """A rail model's points in words: "its 4 bogies at the rail top", "2 points on its
    roof, 4 m up", "its pantograph's head, 5 m up"."""
    n = len(rail_points(source))
    height = source.model.height
    if height == "rail_top":
        return f"its {n} bogies at the rail top" if n > 1 else "its bogie at the rail top"
    if height == "roof":
        return f"{n} points on its roof, 4 m up" if n > 1 else "its roof, 4 m up"
    return f"its {n} pantographs' heads, 5 m up" if n > 1 else "its pantograph's head, 5 m up"


def _article(noun: str) -> str:
    return "an" if noun[:1] in "aeiou" else "a"


def _pattern(pattern: str) -> str:
    return pattern.replace("_", "-")


def _siren_loudness(model) -> str:
    """How loud a siren model is, in words."""
    nbs = "the 100 W electronic siren NBS measured (SP 480-28, 1978)"
    if model.gain_db == 0.0:
        return f"the sound power of {nbs}"
    louder = "louder" if model.gain_db > 0.0 else "quieter"
    return f"{abs(model.gain_db):g} dB {louder} than {nbs}"


#: Where each horn's level comes from, in words.
HORN_RULES = {
    "air_horn": "the most the EU's rule for vehicles' horns allows on its bench",
    "disc_horns": "Hella's S77, approved to UN Regulation 28",
    "train_horn": "EN 15153-2's most",
}


#: Where each horn's level comes from, in a few words (:func:`made_brief`).
HORN_BRIEF = {
    "air_horn": "its rule's most",
    "disc_horns": "Hella's S77",
    "train_horn": "EN 15153-2's most",
}


#: Each horn in words, and those that are a pair.
HORN_NOUNS = {
    "air_horn": "an air horn",
    "disc_horns": "a pair of disc horns",
    "train_horn": "a train horn",
}
PLURAL_HORNS = ("disc_horns",)


def _horn_noun(model) -> str:
    """A horn model in words: "an air horn", "a pair of disc horns"."""
    return HORN_NOUNS[model.kind]


def _horn_loudness(model) -> str:
    """How loud a horn model is, in words: "118 dB(A) 2 m in front of it (...)"."""
    kind = horn_kind(model)
    each = " each note" if kind.in_turn else ""
    return (
        f"{kind.on_axis_db:g} dB({kind.weighting}){each} {kind.at_m:g} m in front of "
        f"{'them' if model.kind in PLURAL_HORNS else 'it'} "
        f"({HORN_RULES[model.kind]})"
    )


__all__ = [
    "FADE_S",
    "HORN_BRIEF",
    "HORN_NOUNS",
    "HORN_RULES",
    "METHOD_WORDS",
    "RAIL_SEEDS",
    "SOUND_RMS",
    "driven_kmh",
    "fade_s",
    "horn_kind",
    "input_gain",
    "made_brief",
    "made_credit",
    "made_credits",
    "made_file",
    "made_key",
    "made_sound",
    "made_words",
    "model_line",
    "pass_times",
    "rail_points",
    "recorded_trail",
    "speed_kmh",
    "trail",
]
