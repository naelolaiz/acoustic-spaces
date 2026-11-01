"""How much the tram and the road works shake the street's buildings, by the US
Federal Transit Administration's general method.

    python vibration.py [scene.yaml] [--tram-events frequent] [--building small_masonry]
                        [--floor 1] [--peak low]

The tram runs down the middle of the street at its own speed, 50 km/h. For the houses,
as far from its track as the scene's footprint is: the vibration of the ground outside,
of a floor inside, and the rumble the rooms' surfaces radiate (the ground-borne noise),
against the FTA's limits for residences. That is the General Vibration Assessment of
the FTA's Transit Noise and Vibration Impact Assessment Manual (FTA Report No. 0123,
2018, section 6.4; :mod:`acoustic_spaces.vibration`) with its rapid transit curve,
which the manual gives streetcars. Its passes are frequent events (more than 70 a
day), as the manual says most rapid transit's are; --tram-events picks another. The
houses are small masonry buildings whose structure is not known, for which the manual
gives a conservative 3 dB more indoors than outdoors. The office block has no limits in
the method: the manual counts offices among its institutions only where something in
them is sensitive to vibration.

The road works' breaker and dumper, as far from each building as the scene puts them:
their peak particle velocity against the manual's damage criteria for four kinds of
building, and their vibration level at the houses against the limit for residences
(the manual's section 7.2: its table 7-4, eq. 7-2 and 7-3, table 7-5). The breaker is
the table's hoe ram, the dumper its loaded trucks, the nearest rows; the generator is
left out, as the manual counts air compressors and light trucks among plant that makes
little vibration. Work going on for hours is taken as frequent events; the manual does
not say which its construction vibration is.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from acoustic_spaces import vibration
from acoustic_spaces.demos.plant import MACHINES
from acoustic_spaces.demos.rail import TRAINS
from acoustic_spaces.scene import Scene

#: The scene's tram (its walk is its track), houses and office block.
TRAM, HOUSES, OFFICES = "tram", "houses", "offices"
#: Residences and buildings where people normally sleep (the manual's land use 2).
RESIDENCES = 2
PEAK_WORDS = {"low": "below 30 Hz", "mid": "at 30 to 60 Hz", "high": "above 60 Hz"}

TITLE = "Vibration of the tram and the road works in the buildings"


def parser() -> argparse.ArgumentParser:
    """Its arguments, which the editor's Run demo window asks for too."""
    found = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    found.add_argument(
        "scene", nargs="?", default=Path(__file__).with_name("scene.yaml"), help="the scene"
    )
    found.add_argument(
        "--tram-events",
        choices=("frequent", "occasional", "infrequent"),
        default="frequent",
        help="the tram's passes: more than 70 a day, 30 to 70, or fewer",
    )
    found.add_argument(
        "--building",
        choices=tuple(vibration.BUILDINGS),
        default="small_masonry",
        help="the houses",
    )
    found.add_argument("--floor", type=int, default=1, help="their floor, 1 above the basement")
    found.add_argument(
        "--peak",
        choices=tuple(vibration.NOISE_DB),
        default="low",
        help="where the tram's vibration peaks: below 30 Hz, 30 to 60 Hz or above 60 Hz",
    )
    return found


def _named(items, name: str):
    """The scene's item ``name``."""
    found = next((item for item in items if item.name == name), None)
    if found is None:
        raise SystemExit(f"error: the scene has no {name!r}")
    return found


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    scene = Scene.from_yaml(args.scene)
    tram = _named(scene.sources, TRAM)
    if tram.walk is None:
        raise SystemExit(f"error: the scene's {TRAM!r} has no walk")
    track = [tuple(tram.position), *(tuple(p) for p in tram.walk.to)]
    buildings = {
        name: _named(scene.geometry.buildings, name).footprint for name in (HOUSES, OFFICES)
    }
    train = TRAINS[TRAM]
    indoors = vibration.indoor_db(args.building, args.floor)
    limit = vibration.criterion(RESIDENCES, args.tram_events)
    distance = _between(track, buildings[HOUSES])
    outside = vibration.ground_level(train.curve, distance, train.speed_kmh)
    inside = outside + indoors
    rumble = vibration.noise_level(inside, args.peak)
    print(
        "Ground-borne vibration by the FTA's general method (FTA Report No. 0123, 2018). "
        "Levels in VdB (dB re 1 micro-inch/s), the rumble in dB(A)."
    )
    print(
        f"\n{train.short} at {train.speed_kmh:.0f} km/h, on the curve of "
        f"{vibration.CURVES[train.curve].what}, {distance:.1f} m from the houses, "
        f"{vibration.BUILDINGS[args.building]}:"
    )
    print(
        f"  the ground outside {outside:.1f} VdB; on floor {args.floor} ({indoors:+.0f} dB) "
        f"{inside:.1f} VdB; the rumble {rumble:.1f} dB(A), the vibration peaking "
        f"{PEAK_WORDS[args.peak]}."
    )
    print(
        f"  Residences, {args.tram_events} events: limits {limit.vibration_vdb:.0f} VdB and "
        f"{limit.noise_dba:.0f} dB(A): the vibration is "
        f"{_against(inside, limit.vibration_vdb)}, the rumble "
        f"{_against(rumble, float(limit.noise_dba or 0.0))}."
    )
    residences = vibration.criterion(RESIDENCES, "frequent")
    print(
        "\nThe road works: their peak particle velocity (PPV) against the damage "
        "criteria, and their level at the houses against the limit for residences, "
        f"{residences.vibration_vdb:.0f} VdB."
    )
    for name, machine in MACHINES.items():
        if machine.vibration is None:
            print(f"  {name}: left out, plant that makes little vibration")
            continue
        plant = vibration.PLANT[machine.vibration]
        source = _named(scene.sources, name)
        print(f"  {name}, {machine.noun}: table 7-4's row for {plant.what}")
        for building, footprint in buildings.items():
            d = vibration.horizontal_distance(source.position, footprint, closed=True)
            ppv = vibration.plant_ppv(machine.vibration, d)
            level = vibration.plant_level(machine.vibration, d)
            over = [key for key, damage in vibration.DAMAGE.items() if ppv > damage.ppv_in_s]
            if len(over) > 1:
                damage = (
                    f"over the damage criteria of categories {', '.join(over[:-1])} and {over[-1]}"
                )
            elif over:
                damage = f"over the damage criterion of category {over[0]}"
            else:
                damage = "under every damage criterion"
            line = (
                f"    at the {building}, {d:.1f} m: PPV {vibration.mm_s(ppv):.2f} mm/s "
                f"({ppv:.3f} in/s), {level:.1f} VdB; {damage}"
            )
            if building == HOUSES:
                line += f"; {_against(level, residences.vibration_vdb)} for residences"
            print(line)
    print(
        "\nThe damage criteria (PPV): "
        + "; ".join(
            f"{key} {damage.what}, {vibration.mm_s(damage.ppv_in_s):.2f} mm/s"
            for key, damage in vibration.DAMAGE.items()
        )
        + "."
    )
    return 0


def _between(track, footprint) -> float:
    """The distance (m) in plan between the track and ``footprint``'s outline: the
    nearest of their corners to the other, as for any two lines that do not cross."""
    return min(
        *(vibration.horizontal_distance(corner, track) for corner in footprint),
        *(vibration.horizontal_distance(point, footprint, closed=True) for point in track),
    )


def _against(level: float, limit: float) -> str:
    """How ``level`` stands against ``limit``, in words."""
    if level > limit:
        return f"{level - limit:.1f} dB over the limit"
    return f"{limit - level:.1f} dB under the limit"


if __name__ == "__main__":
    raise SystemExit(main())
