"""The street example's road works demo: a breaker, a generator and a dumper reversing
with its alarm on, heard at each of the street's listeners in turn, and a video that
tours them.

The machines are the street's left-out sources ``breaker``, ``generator``, ``dumper``
and ``dumper_alarm``, their sound power from the Defra noise database and the alarm's
rule (:mod:`acoustic_spaces.demos.plant`). ``scene`` writes the street with them alone
(``scene_works.yaml``), ``feeds`` makes what each plays (``--blows`` how fast the
breaker strikes; ``--input NAME=WAV`` a dry recording of yours instead), ``simulate``
simulates the scene, and ``render`` and ``video`` are the scene demo's
(:mod:`acoustic_spaces.demos.scene_demo`): every listener for ``--seconds`` while the
works go on, binaural, one gain for all, each stop's caption giving the level there.
``--indoors`` hears them in a house at each listener instead: through its closed
window, with the rumble of the breaker's and the dumper's vibration, or either alone
(:class:`acoustic_spaces.demos.indoors.StillHouses`).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Step,
    common_options,
    indoors,
    input_option,
    listeners_in,
    pipeline,
    plant,
    scene_demo,
)
from acoustic_spaces.demos.audio import mono, read, resampled, write

#: The works' sources in the street example, in the order the demo lists them.
SOURCES = (plant.BREAKER.name, plant.GENERATOR.name, plant.DUMPER.name, plant.ALARM)
#: What the feeds step keeps beside the feeds.
WORKS = "works.json"
#: What ``--indoors`` calls the works.
NOUN = "works"


def machine_of(name: str) -> plant.Machine | None:
    """The machine a source of the works is, or None for the alarm."""
    return plant.MACHINES.get(name)


def level_of(name: str, rate: int) -> list[float]:
    """The sound power per octave band of a source of the works (dB re 1 pW, to a tenth
    of a decibel, as the street's scene gives it)."""
    machine = machine_of(name)
    level = plant.alarm_power(rate) if machine is None else machine.sound_power()
    return [round(float(v), 1) for v in level]


def what_plays(name: str, args: argparse.Namespace) -> str:
    """What a source of the works plays unless ``--input`` gives it a file, in words."""
    machine = machine_of(name)
    if machine is None:
        return f"its reversing alarm: {plant.ALARM_HZ:g} Hz beeps, {plant.ALARM_PER_S:g} a second"
    if machine.made == "blows":
        return f"{machine.noun}: {float(args.blows):g} blows a minute"
    return f"{machine.noun}: its engine at {machine.rpm:g} rpm"


def plays(args: argparse.Namespace) -> dict[str, str]:
    return {name: what_plays(name, args) for name in SOURCES}


def problems(demo: Demo, args: argparse.Namespace) -> str | None:
    if float(args.blows) <= 0.0:
        return f"{demo.option('blows').option} is a positive number, not {args.blows!r}"
    if float(args.seconds) <= 0.0:
        return f"{demo.option('seconds').option} is a positive number, not {args.seconds!r}"
    try:
        indoors.House.of(getattr(args, "indoors", None), NOUN)
    except ValueError as exc:
        return str(exc)
    return None


def _point(position) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in position)
    return x, y, z


def houses(args: argparse.Namespace) -> indoors.StillHouses | None:
    """The house at each listener ``--indoors`` asks for (the works heard through its
    window, the rumble of the breaker's and the dumper's vibration, or both), or None
    for the works outdoors."""
    from acoustic_spaces import Scene

    choice = getattr(args, "indoors", None)
    house = indoors.House.of(choice, NOUN)
    if house is None:
        return None
    found = Scene.from_yaml(args.scene)
    playing = [s for s in found.sources if s.in_use and s.name in SOURCES]
    shaking = []
    for source in playing:
        machine = machine_of(source.name)
        if machine is not None and machine.vibration is not None:
            shaking.append(indoors.Plant(source.name, machine.vibration, _point(source.position)))
    return indoors.StillHouses(
        house,
        str(choice),
        tuple(shaking),
        {lis.name: _point(lis.position) for lis in found.listeners},
        max(s.sound_power_db for s in playing),
        scene_demo.format_of(args, found),
    )


# --------------------------------------------------------------------------- the scene


def scene(args: argparse.Namespace) -> None:
    """``--scene`` (scene_works.yaml) from ``--from`` (scene.yaml): the street with its
    road works alone, each of their sources at its sound power."""
    data = yaml.safe_load(Path(args.source).read_text(encoding="utf-8"))
    rate = int(data.get("simulation", {}).get("sample_rate", 48000))
    found = {s["name"]: s for s in data.get("sources", [])}
    missing = [name for name in SOURCES if name not in found]
    if missing:
        raise SystemExit(f"error: {args.source} has no {', '.join(missing)}")
    data["name"] = data["name"] + " (road works)"
    data["sources"] = [
        {k: v for k, v in found[name].items() if k != "enabled"}
        | {"level_db": level_of(name, rate)}
        for name in SOURCES
    ]
    pipeline.write_scene(
        args.scene,
        data,
        "# The street example with its road works alone, written by works_demo.py.\n",
        Path(args.source).parent,
    )
    print(f"wrote {args.scene}: {', '.join(SOURCES)}")


def scene_done(args: argparse.Namespace) -> bool:
    scene_file, source = Path(args.scene), Path(args.source)
    return scene_file.exists() and scene_file.stat().st_mtime >= source.stat().st_mtime


# --------------------------------------------------------------------------- the feeds


def length_s(args: argparse.Namespace) -> float:
    """How long the works play: ``--seconds`` for each listener in use."""
    from acoustic_spaces import Scene

    found = Scene.from_yaml(args.scene)
    heard = [lis for lis in found.listeners if lis.in_use]
    return float(args.seconds) * max(1, len(heard))


def made(args: argparse.Namespace) -> dict:
    """What the feeds are made of: the rate, the length, the blows and the inputs."""
    from acoustic_spaces import Scene

    rate = int(Scene.from_yaml(args.scene).simulation.sample_rate)
    inputs = {k: str(Path(v).resolve()) for k, v in sorted((args.inputs or {}).items())}
    return {
        "rate": rate,
        "length_s": round(length_s(args), 6),
        "blows_per_minute": float(args.blows),
        "inputs": inputs,
    }


def input_gain(name: str, x: np.ndarray, rate: int) -> float:
    """The gain of ``x``, a recording a source of the works plays instead of its made
    sound: a machine's brought to its made sound's RMS, the alarm's to its made sound's
    while it sounds (:func:`~acoustic_spaces.demos.horn.sounding_rms` of both), whatever
    its pauses."""
    from acoustic_spaces.demos.horn import sounding_rms

    if machine_of(name) is not None:
        rms = float(np.sqrt(np.mean(x**2)))
        if rms <= 0.0:
            raise SystemExit(f"error: --input {name}: the file is silent")
        return plant.SOUND_RMS / rms
    rms = sounding_rms(x, rate)
    if rms <= 0.0:
        raise SystemExit(f"error: --input {name}: the file is silent")
    return sounding_rms(plant.alarm_sound(rate, plant.ALARM_POWER_S), rate) / rms


def feeds(args: argparse.Namespace) -> None:
    """One WAV per source of the works in the work folder's ``feeds``, ``--seconds`` for
    each listener long: its made sound (:func:`~acoustic_spaces.demos.plant.machine_sound`,
    :func:`~acoustic_spaces.demos.plant.alarm_sound`) or the file ``--input`` gives it,
    mixed down to mono and cut or looped to that length; each at the made sounds' RMS,
    so that the scene demo's render plays it at its source's sound power."""
    facts = made(args)
    rate, seconds = facts["rate"], facts["length_s"]
    n = int(round(seconds * rate))
    out = Path(args.work) / "feeds"
    table: dict[str, dict] = {}
    for k, name in enumerate(SOURCES):
        given = facts["inputs"].get(name)
        machine = machine_of(name)
        if given is not None:
            x, r = read(Path(given))
            x = resampled(mono(x), r, rate)
            x = np.resize(x, n) * input_gain(name, x, rate)
            said = Path(given).name
        elif machine is None:
            x = plant.alarm_sound(rate, seconds)
            said = "made"
        else:
            x = plant.machine_sound(
                machine, rate, seconds, blows_per_minute=facts["blows_per_minute"], seed=k
            )
            said = "made"
        write(out / f"{name}.wav", x, rate)
        table[name] = {"signal": said, "channel": None, "length_s": n / rate}
        print(f"{name:>14}: {what_plays(name, args) if given is None else given}")
    pipeline.write_json(
        out / scene_demo.FEEDS, {"rate": rate, "length_s": n / rate, "sources": table, "silent": []}
    )
    pipeline.write_json(out / WORKS, facts)
    print(f"wrote {len(table)} feeds to {out}, {seconds:.0f} s each")


def feeds_done(args: argparse.Namespace) -> bool:
    out = Path(args.work) / "feeds"
    kept = out / WORKS
    if not kept.exists() or not Path(args.scene).exists():
        return False
    return (
        pipeline.read_json(kept) == made(args)
        and kept.stat().st_mtime >= Path(args.scene).stat().st_mtime
        and all((out / f"{name}.wav").exists() for name in SOURCES)
    )


# --------------------------------------------------------------------------- the video


def credit(args: argparse.Namespace) -> str:
    """The stills' line on what plays: each source's made sound or file."""
    facts = pipeline.read_json(Path(args.work) / "feeds" / WORKS)
    said = []
    for name in SOURCES:
        given = facts["inputs"].get(name)
        machine = machine_of(name)
        if given is not None:
            said.append(f"{name}: {Path(given).name}")
        elif machine is None:
            said.append(f"alarm: {plant.ALARM_HZ:g} Hz beeps, SAE J994 Type B")
        else:
            how = (
                f"{facts['blows_per_minute']:g} blows a minute"
                if machine.made == "blows"
                else f"its engine at {machine.rpm:g} rpm"
            )
            said.append(f"{name}: {how}")
    return "Made to Defra's noise database (2005): " + "; ".join(said)


def video(shown: Demo, args: argparse.Namespace) -> None:
    """The scene demo's video of the works, its stills crediting the made sounds."""
    if not args.credit:
        args = argparse.Namespace(**{**vars(args), "credit": credit(args)})
    scene_demo.video(args, shown, houses(args))


def demo(script: Path, example: str = "street") -> Demo:
    """The road works demo of ``example``, run by ``script`` (its ``works_demo.py``)."""
    shared = {o.name: o for o in scene_demo.DEMO.options}
    made = Demo(
        name="works",
        example=example,
        title="Road works in the street",
        description=__doc__,
        script=Path(script),
        steps=[],
        options=[
            Option(
                "blows",
                "how many blows a minute the breaker strikes (Epiroc's MB 1650, a 1650 kg "
                f"breaker, strikes {plant.BLOWS_RANGE[0]:g} to {plant.BLOWS_RANGE[1]:g})",
                plant.BLOWS_PER_MINUTE,
                "float",
                steps=("feeds",),
                unit="/min",
            ),
            dataclasses.replace(shared["seconds"], steps=("feeds", "video")),
            Option(
                "indoors",
                indoors.help_of(NOUN, "the works"),
                indoors.OFF,
                "choice",
                indoors.choices(NOUN),
                steps=("render", "video"),
            ),
            shared["listeners"],
            shared["credit"],
            Option(
                "source",
                "the example's scene",
                Path("scene.yaml"),
                "file",
                steps=("scene",),
                flag="--from",
                filter="Scene files (*.yaml *.yml)",
            ),
            Option(
                "scene", "the scene with the road works alone", Path("scene_works.yaml"), "path"
            ),
            *common_options("demo_works"),
            dataclasses.replace(
                input_option(("feeds",)),
                help="a WAV file a source of the works plays instead of its made sound, as "
                "NAME=WAV (breaker, generator, dumper or dumper_alarm): a dry recording, "
                "mixed down to mono and looped to the length of the works, a machine's "
                "at its sound power over the whole file, the alarm's beeps as loud as the "
                "made alarm's, whatever its pauses",
            ),
        ],
        simulations=scene_demo.simulations,
        scene_from="scene",
        problems=problems,
        plays=plays,
        # The works' scene keeps the street's listeners: read them before it is made.
        listeners=lambda args: listeners_in(args.source),
    )
    made.steps = [
        Step("scene", scene, "scene_works.yaml: the street with its road works alone", scene_done),
        Step("feeds", feeds, "what each machine and the alarm play, dry", feeds_done),
        SIMULATE,
        Step(
            "render",
            lambda args: scene_demo.render(args, made, houses(args)),
            "every listener, with one gain for all",
        ),
        Step(
            "video",
            lambda args: video(made, args),
            "tour.wav, the stills and tour.mp4 in the work folder",
        ),
    ]
    return made


__all__ = ["NOUN", "SOURCES", "demo", "feeds", "houses", "level_of", "plays", "scene"]
