"""How much the trains shake a house beside the railway, and how loud the rumble is in
it, by the US Federal Transit Administration's general method.

    python vibration.py [scene.yaml] [--trains-per-day N] [--building small_masonry]
                        [--floor 1] [--ground normal] [--peak low]
                        [--listeners a,b]

For each of the railway demo's trains at its own speed, and a house standing where
each of the scene's listeners in use is (or those --listeners names), as far from the
track as it, or the scene's house a listener stands in front of (house_18m), as far as
its wall behind the listener: the vibration of the ground outside, of a floor inside,
and the rumble the room's surfaces radiate (the ground-borne noise), each against the
FTA's limits for residences; and how far from the track a house must be for both to
hold. The method is the General Vibration Assessment of the FTA's Transit Noise and
Vibration Impact Assessment Manual (FTA Report No. 0123, 2018, section 6.4;
:mod:`acoustic_spaces.vibration`): its base curve for the kind of train, adjusted for
its speed, the ground and the building. It gives overall levels, not their spectra,
and it rests on North American trains and soils.

The trains per day set the limits (the manual's table 6-2): the scene's line source is
one regional train an hour, 24 a day, which are infrequent events. A freight train's
locomotive counts as an infrequent event and its wagons as frequent ones, the wagons
taken 3 dB under the locomotive, the least of the 3 to 8 dB the manual gives. The house
is a small masonry building whose structure is not known, for which the manual gives a
conservative 3 dB more indoors than outdoors; --building picks another of its
buildings and --floor its floor. The vibration peaks below 30 Hz, as the manual says
it does for most surface track; --peak mid or high for stiff clayey soil. The faster
trains run above the manual's table of speeds (up to 60 mph, 97 km/h), so their levels
extrapolate its 20 log10(speed) (eq. 6-4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from acoustic_spaces import vibration
from acoustic_spaces.demos.rail import TRAINS
from acoustic_spaces.scene import Scene

#: The scene's train: its walk is the track.
TRAIN = "train"
#: Residences and buildings where people normally sleep (the manual's land use 2).
RESIDENCES = 2
#: The distances (m) over which the limits are looked for: as near and as far as the
#: manual draws its curves, 10 to 300 ft.
NEAREST_M, FARTHEST_M = vibration.NEAREST_M, vibration.FARTHEST_M

TITLE = "Vibration of the trains in a house beside the track"


def parser() -> argparse.ArgumentParser:
    """Its arguments, which the editor's Run demo window asks for too."""
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument(
        "scene", nargs="?", default=Path(__file__).with_name("scene.yaml"), help="the scene"
    )
    found.add_argument(
        "--trains-per-day", type=int, default=24, help="trains a day (the scene's one an hour)"
    )
    found.add_argument(
        "--building",
        choices=tuple(vibration.BUILDINGS),
        default="small_masonry",
        help="the house",
    )
    found.add_argument("--floor", type=int, default=1, help="its floor, 1 above the basement")
    found.add_argument(
        "--ground",
        choices=vibration.GROUNDS,
        default="normal",
        help="the ground: normal soil, soil that propagates efficiently, or a rock layer",
    )
    found.add_argument(
        "--peak",
        choices=tuple(vibration.NOISE_DB),
        default="low",
        help="where the vibration peaks: below 30 Hz, 30 to 60 Hz or above 60 Hz",
    )
    found.add_argument(
        "--listeners",
        default=None,
        help="only the houses where these listeners are, in this order, comma separated "
        "(default: where every listener in use is)",
    )
    return found


def places_of(scene: Scene, track, wanted: str | None) -> list[tuple[str, float]]:
    """The houses: where each listener in use is (those ``wanted`` names, in its order),
    each as far from the track as it, or the scene's building a listener stands in front
    of (:func:`~acoustic_spaces.demos.indoors.facing`), as far as its wall behind the
    listener."""
    from acoustic_spaces.demos import indoors, listener_names

    here = {lst.name: lst for lst in scene.listeners if lst.in_use}
    names = listener_names(wanted)
    if names is None:
        names = list(here)
    unknown = [name for name in names if name not in here]
    if unknown or not names:
        raise SystemExit(
            f"error: --listeners: {', '.join(unknown) or 'none named'}; the scene's "
            f"listeners in use are {', '.join(here) or 'none'}"
        )
    places = []
    for name in names:
        room = indoors.facing(scene, here[name].position)
        at = here[name].position if room is None else room.wall
        label = name if room is None else f"{name} ({room.name})"
        places.append((label, vibration.horizontal_distance(at, track)))
    return places


def track_of(scene: Scene) -> list[tuple[float, float, float]]:
    """The track: the walk of the scene's train, from its position."""
    train = next((s for s in scene.sources if s.name == TRAIN), None)
    if train is None or train.walk is None:
        raise SystemExit(f"error: the scene has no {TRAIN!r} with a walk")
    return [tuple(train.position), *(tuple(p) for p in train.walk.to)]


def rows(name: str, args: argparse.Namespace) -> list[tuple[str, float, vibration.Criterion]]:
    """The train ``name``'s parts heard apart: (what, change to its curve (dB), its
    limits), a freight train's locomotive and wagons apart."""
    if name == "freight":
        return [
            ("its locomotive", 0.0, vibration.criterion(RESIDENCES, "infrequent")),
            (
                "its wagons",
                vibration.FREIGHT_WAGONS_DB,
                vibration.criterion(RESIDENCES, "frequent"),
            ),
        ]
    kind = vibration.events(args.trains_per_day)
    return [("", 0.0, vibration.criterion(RESIDENCES, kind))]


#: The ground and where the vibration peaks, in words.
GROUND_WORDS = {
    "normal": "normal soil",
    "efficient": "soil that propagates it efficiently",
    "rock": "a rock layer",
}
PEAK_WORDS = {"low": "below 30 Hz", "mid": "at 30 to 60 Hz", "high": "above 60 Hz"}


def holds(level, limit: float) -> str:
    """From how far the level ``level`` (of the distance) is within ``limit``, in words."""
    found = vibration.distance_for(level, limit, NEAREST_M, FARTHEST_M)
    if found is None:
        return f"over its limit as far as the curve goes, {FARTHEST_M:.0f} m"
    if found == NEAREST_M:
        return f"within its limit from {NEAREST_M:.1f} m, where the curve starts"
    return (
        f"within its limit from {found:.0f} m"
        if found >= 10
        else f"within its limit from {found:.1f} m"
    )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    scene = Scene.from_yaml(args.scene)
    track = track_of(scene)
    line = vibration.Line(ground=args.ground)
    indoors = vibration.indoor_db(args.building, args.floor)
    places = places_of(scene, track, args.listeners)
    print(
        "Ground-borne vibration by the FTA's general method (FTA Report No. 0123, 2018), "
        "in the house a listener stands in front of, as if one stood where the others are:"
    )
    print(
        f"  the house {vibration.BUILDINGS[args.building]}: on its floor {args.floor} "
        f"{indoors:+.0f} dB from the ground outside;"
    )
    print(f"  {GROUND_WORDS[args.ground]}, the vibration peaking {PEAK_WORDS[args.peak]};")
    print(
        f"  residences, {args.trains_per_day} trains a day: "
        f"{vibration.events(args.trains_per_day)} events."
    )
    print("Levels in VdB (dB re 1 micro-inch/s), the rumble in dB(A).")
    noise = indoors + vibration.NOISE_DB[args.peak]
    for name, train in TRAINS.items():
        curve = vibration.CURVES[train.curve]
        print(f"\n{train.short} at {train.speed_kmh:.0f} km/h, on the curve of {curve.what}:")
        for what, change, limit in rows(name, args):

            def ground(
                d: float,
                change: float = change,
                curve: str = train.curve,
                kmh: float = train.speed_kmh,
            ) -> float:
                return vibration.ground_level(curve, d, kmh, line) + change

            label = f"{what}: " if what else ""
            print(
                f"  {label}limits {limit.vibration_vdb:.0f} VdB and {limit.noise_dba:.0f} dB(A) "
                "indoors"
            )
            print(f"    {'where':<22}{'distance':>10}{'ground':>9}{'indoors':>9}{'rumble':>9}")
            for place, distance in places:
                outside = ground(distance)
                over = [
                    w
                    for w, level, top in (
                        ("vibration", outside + indoors, limit.vibration_vdb),
                        ("rumble", outside + noise, limit.noise_dba),
                    )
                    if top is not None and level > top
                ]
                if len(over) == 2:
                    note = "  both over their limits"
                else:
                    note = "".join(f"  {w} over its limit" for w in over)
                print(
                    f"    {place:<22}{distance:8.1f} m{outside:9.1f}{outside + indoors:9.1f}"
                    f"{outside + noise:9.1f}{note}"
                )
            parts = [
                f"its vibration is {holds(lambda d: ground(d) + indoors, limit.vibration_vdb)}"
            ]
            if limit.noise_dba is not None:
                parts.append(f"its rumble is {holds(lambda d: ground(d) + noise, limit.noise_dba)}")
            print(f"    Indoors, {'; '.join(parts)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
