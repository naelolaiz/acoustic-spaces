"""A car (its horn too, if wanted), a bus, a motorcycle, a police car with its siren on, a
fire engine with its siren and its air horn or a tram going along the street between the
houses and the office block, heard binaurally on the pavement, on a balcony and in the
garden behind the noise barrier, its pitch falling as it goes by (Doppler); and a video
that follows.

This is the demo of the street's scene (``acoustic-spaces demo scene.yaml --sources
car``, This scene in the editor): a preset of it, which keeps the command lines of the
script it was. The vehicles are the scene's own sources, left out of its simulations
until picked: ``car`` and its horn ``car_horn`` (the group ``Passenger car``), ``bus``,
``motorcycle``, ``police_car`` and its siren ``police_car_siren`` (``Police car``),
``fire_engine`` with ``fire_engine_siren`` and ``fire_engine_horn`` (``Fire engine``),
and ``tram`` with ``tram_roof`` (``Tram``). ``--sources`` picks what goes past, the car
alone by default, each name or group in a part of its own, one after the other in one
video; names joined with ``+`` go past together (``--sources car+car_horn``, the car
sounding its horn). The script's ``--vehicle`` still picks one (car, bus, motorcycle,
police_car, fire_engine or tram), and ``--car`` (car, car_and_horn or horn),
``--police-car`` (car_and_siren or siren) and ``--fire-engine`` (truck_siren_and_horn,
truck_and_siren, siren_and_horn or siren) what that vehicle plays.

They go at 50 km/h (``--speed`` sets another), and each has a model in the scene. The
car, the bus, the motorcycle, the police car and the fire engine send their tyres and
engine from 0.05 m above the road with CNOSSOS-EU's sound power per octave band for
their category at their speed, in the scene's air (Directive 2002/49/EC, Annex II,
Appendix F, Table F-1, as amended in 2021): its light motor vehicles (the car and the
police car), its medium heavy vehicles with two axles (the bus, a city bus, and the fire
engine) and its motorcycles; their sound is made to that power: tyre noise (none for the
motorcycle), and the firing of their engine (four cylinders at 2000 rpm, six at 1500 and
two at 4000; ``--rpm``), each with the spectrum of its part. The car's horn is a pair of
disc horns at 360 and 430 Hz, their tone discs ringing near 3 kHz, 0.5 m up, in two
short blasts every 3 s, 110 dB(A) 2 m in front of them, as Hella's S77 (approved to UN
Regulation 28). The police car's siren is on its roof, 1.5 m up (``--siren``: wail, yelp
or two_tone), at the sound power of a 100 W electronic siren the National Bureau of
Standards measured. The fire engine's siren is on its cab's roof, 3.0 m up, playing the
two tones as loud as the Swiss rule lets them be on a vehicle, and its air horn 1.2 m
up, in blasts as loud as the rule for vehicles' horns lets one be on its bench. The
sirens and horns face the way their vehicle drives, with the directivity of the siren
NBS measured (``siren.csv``). The tram is a low-floor tram with air conditioning on its
track set in the street, its sound power at the rail top and on its roof Schall 03's
(Germany's method for the noise of railways and tramways, 2014), its bogies and the
halves of its roof each played as far behind its front as they are. ``--indoors`` hears
what goes past in the room behind each listener that stands in front of a building's
wall (the pavement, 2 m in front of the houses, and the balcony, 1 m in front of the
offices): through its closed window, from the level 2 m in front of the wall, and the
rumble the tram's wheels make in it, or either alone (``airborne_and_rumble``,
``airborne``, ``rumble``; the script's ``tram_and_rumble`` and ``tram`` read as the
first two). ``--input NAME=WAV`` plays a dry recording of yours instead of a source's
made sound (one of a vehicle going past has its Doppler and the road in it already).
Run it from the folder of the street example (``acoustic-spaces examples --copy .
street && cd street``), in the Python environment acoustic-spaces is installed in, with
ffmpeg on the PATH::

    python passby_demo.py all      # every step, leaving out what is there already
    python passby_demo.py feeds    # the vehicle's sound, made to its sound power
    python passby_demo.py render   # the pass every metre along the street, at every listener
    python passby_demo.py video    # the video, its audio and its stills

The script's steps run as the ones that took them over: ``sound`` the feeds, ``scene``
the simulation of the scene with the vehicle taken in, ``walk`` the mixes. Each listener
hears the vehicle through every early path the scene traces (reflections off the road,
the facades and the barrier up to the third order, and the direct sound and the first
reflections diffracted round their edges), each through its own delay as it changes, so
the direct sound and each reflection shift in pitch as the vehicle passes, each its own
way; the late sound between the buildings is crossfaded from place to place. The stills
give the level heard at each moment, the loudest (L_Amax) and the sound exposure level
(SEL) of the pass, as simulated, how close the vehicle comes and how far the pitch of
its direct sound falls; the vehicle's trajectory, edited in the editor's Movement panel,
is driven as edited.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acoustic_spaces.demos.scene_demo import preset
from acoustic_spaces.i18n import N_

#: What the script's ``--vehicle`` picked: each of its vehicles and the sources it plays
#: as the demo's ``--sources`` takes them, a source, a group or names joined with ``+``;
#: for a vehicle that plays more than one way, its flag (the first choice its default).
VEHICLES: dict[str, str | tuple[str, dict[str, str]]] = {
    "car": (
        "--car",
        {"car": "car", "car_and_horn": "Passenger car", "horn": "car_horn"},
    ),
    "bus": "bus",
    "motorcycle": "motorcycle",
    "police_car": (
        "--police-car",
        {"car_and_siren": "Police car", "siren": "police_car_siren"},
    ),
    "fire_engine": (
        "--fire-engine",
        {
            "truck_siren_and_horn": "Fire engine",
            "truck_and_siren": "fire_engine+fire_engine_siren",
            "siren_and_horn": "fire_engine_siren+fire_engine_horn",
            "siren": "fire_engine_siren",
        },
    ),
    "tram": "Tram",
}
#: The vehicle the script played unless ``--vehicle`` said, and so the demo's pick.
VEHICLE = "car"
#: What the script's ``--indoors`` took for the tram, as the demo's takes it: the tram
#: through the window is the airborne sound.
INDOORS = {"tram_and_rumble": "airborne_and_rumble", "tram": "airborne"}
#: The flags of the vehicles that play more than one way, and their ``dest``.
PLAYS_DESTS = {
    entry[0]: "old_" + name for name, entry in VEHICLES.items() if isinstance(entry, tuple)
}


def old_flags() -> tuple[tuple[str, dict], ...]:
    """The script's flags that pick what plays, which read together (:func:`reads_old`):
    ``--vehicle`` and the flag of each vehicle that plays more than one way."""
    out: list[tuple[str, dict]] = [
        (
            "--vehicle",
            {
                "dest": "old_vehicle",
                "help": "the vehicle that goes past, as the script took it: "
                + ", ".join(VEHICLES)
                + " (comma separated: each in a part of its own); --sources picks the same",
            },
        )
    ]
    for name, entry in VEHICLES.items():
        if isinstance(entry, tuple):
            flag, plays = entry
            out.append(
                (
                    flag,
                    {
                        "dest": PLAYS_DESTS[flag],
                        "choices": tuple(plays),
                        "help": f"what the {name.replace('_', ' ')} plays when --vehicle "
                        "picks it, as the script took it: " + ", ".join(plays),
                    },
                )
            )
    return tuple(out)


def reads_old(args: argparse.Namespace) -> str | None:
    """Set ``--sources`` to what the script's ``--vehicle`` and its vehicles' flags pick,
    each vehicle in a part of its own, and what plays of it as its flag says (a flag for
    a vehicle that is not picked changes nothing, as in the script); what is wrong with
    them, or None."""
    dests = ("old_vehicle", *PLAYS_DESTS.values())
    given = {dest: getattr(args, dest) for dest in dests if hasattr(args, dest)}
    for dest in given:
        delattr(args, dest)
    if not given:
        return None
    if str(args.sources) != VEHICLE:
        return (
            "--sources picks what --vehicle, --car, --police-car and --fire-engine pick: "
            "give one or the other"
        )
    names = [n.strip() for n in str(given.get("old_vehicle", VEHICLE)).split(",") if n.strip()]
    unknown = [n for n in names if n not in VEHICLES]
    if unknown or not names:
        return f"--vehicle is one of {', '.join(VEHICLES)}, not {','.join(unknown)!r}"
    picks = []
    for name in names:
        entry = VEHICLES[name]
        if isinstance(entry, str):
            picks.append(entry)
            continue
        flag, plays = entry
        picks.append(plays[given.get(PLAYS_DESTS[flag]) or next(iter(plays))])
    args.sources = ",".join(picks)
    return None


def old_indoors(value: str) -> str:
    """A value of the script's ``--indoors`` as the demo's takes it (:data:`INDOORS`)."""
    return INDOORS.get(value, value)


DEMO = preset(
    name="passby",
    example="street",
    title=N_("A vehicle driving down the street"),
    description=__doc__,
    script=Path(__file__),
    # The script's car, and its vehicles each in a part of their own.
    defaults={"sources": VEHICLE, "sources_play": "in-turn"},
    renamed={"scene": (("--from", str),), "indoors": (("--indoors", old_indoors),)},
    steps={"feeds": ("sound",), "simulate": ("scene",), "render": ("walk",)},
    old_flags=old_flags(),
    reads_old=reads_old,
)

__all__ = ["DEMO", "INDOORS", "VEHICLES", "old_indoors", "reads_old"]


if __name__ == "__main__":
    sys.exit(DEMO.main())
