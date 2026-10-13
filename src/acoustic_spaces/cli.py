"""Command-line interface for acoustic_spaces."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from acoustic_spaces import __version__, native_available
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.room import Room


def _band_header(label: str, width: int) -> str:
    bands = "".join(f"{f if f < 1000 else str(f // 1000) + 'k':>6}" for f in OCTAVE_BANDS_HZ)
    return f"{label:<{width}}{bands}"


def _band_row(label: str, values: Sequence[float], width: int, fmt: str = "6.2f") -> str:
    return f"{label:<{width}}" + "".join(f"{v:{fmt}}" for v in values)


def _load_room(path: str) -> Room | None:
    """Load a scene and its geometry, printing a readable error on failure."""
    try:
        return Room.from_yaml(path)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
    except ValidationError as exc:
        print(f"error: {path} is not a valid scene:", file=sys.stderr)
        for err in exc.errors():
            where = ".".join(str(part) for part in err["loc"]) or "scene"
            message = err["msg"].removeprefix("Value error, ")
            print(f"  {where}: {message}", file=sys.stderr)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
    return None


def _print_checks(room: Room) -> bool:
    """Print problems and warnings; return True when there are no problems."""
    for warning in room.warnings():
        print(f"warning: {warning}", file=sys.stderr)
    problems = room.problems()
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return not problems


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"acoustic_spaces {__version__}")
    print(f"compiled core: {'available' if native_available() else 'missing'}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    room = _load_room(args.scene)
    if room is None:
        return 1
    ok = _print_checks(room)
    scene = room.scene
    print(
        f"{scene.name}: {len(scene.materials)} materials, "
        f"{len(scene.sources)} sources, {len(scene.listeners)} listeners, "
        f"{room.report.faces} triangles"
    )
    return 0 if ok else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    room = _load_room(args.scene)
    if room is None:
        return 1
    scene = room.scene
    print(scene.name)
    print(f"geometry: {scene.geometry.file.name}")
    for line in room.report.summary().splitlines():
        print(f"  {line}")

    width = max([len(n) for n in room.material_names] + [12]) + 2
    print("\nmaterials (absorption per octave band, area)")
    print(_band_header("", width) + "    area m2")
    areas = room.area_by_material()
    for name, material in scene.materials.items():
        print(_band_row(name, material.absorption, width) + f"  {areas[name]:9.2f}")
        if material.source:
            print(f"  {material.source}")
    unassigned = room.unassigned_labels()
    if unassigned:
        print(_band_row("(no material)", [], width) + f"  {sum(unassigned.values()):9.2f}")

    if room.volume_m3 is not None:
        print("\nstatistical reverberation time (s), without air absorption")
        print(_band_header("", width))
        print(_band_row("Sabine", room.reverberation_time("sabine"), width))
        print(_band_row("Eyring", room.reverberation_time("eyring"), width))

    print("\npositions")
    for kind, items, inside in (
        ("source", scene.sources, room.sources_inside),
        ("listener", scene.listeners, room.listeners_inside),
    ):
        for k, item in enumerate(items):
            where = "" if inside is None else ("  inside" if inside[k] else "  OUTSIDE")
            x, y, z = item.position
            print(f"  {kind:<9}{item.name:<{width}}({x:.2f}, {y:.2f}, {z:.2f}){where}")
    print()
    return 0 if _print_checks(room) else 1


def _cmd_materials(args: argparse.Namespace) -> int:
    db = builtin_database()
    entries = db.search(" ".join(args.query), kind=args.kind)
    if not entries:
        print("no matching materials", file=sys.stderr)
        return 1
    width = max(len(e.id) for e in entries) + 2
    category = None
    print(_band_header("", width))
    for entry in sorted(entries, key=lambda e: (e.kind, e.category)):
        if (entry.kind, entry.category) != category:
            category = (entry.kind, entry.category)
            print(f"\n{entry.kind}: {entry.category}")
        values, _ = entry.octave_bands()
        print(_band_row(entry.id, values, width) + f"  {entry.description}")
    print(
        "\nValues from the pyroomacoustics materials table; 63 Hz and missing high "
        "octaves are copied from the nearest octave."
    )
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    room = _load_room(args.scene)
    if room is None:
        return 1
    _print_checks(room)
    try:
        from acoustic_spaces.viz import show_room

        show_room(room, screenshot=args.screenshot)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.screenshot:
        print(f"saved {args.screenshot}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustic-spaces", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show the installed version and compiled components")
    info.set_defaults(func=_cmd_info)

    validate = sub.add_parser("validate", help="check a scene file and its geometry")
    validate.add_argument("scene", help="path to a scene YAML file")
    validate.set_defaults(func=_cmd_validate)

    inspect = sub.add_parser(
        "inspect", help="geometry report, materials, reverberation estimates and positions"
    )
    inspect.add_argument("scene", help="path to a scene YAML file")
    inspect.set_defaults(func=_cmd_inspect)

    materials = sub.add_parser("materials", help="list or search the built-in materials")
    materials.add_argument("query", nargs="*", help="words to look for, e.g. carpet concrete")
    materials.add_argument("--kind", choices=["absorption", "scattering"], default=None)
    materials.set_defaults(func=_cmd_materials)

    view = sub.add_parser("view", help="show the room in 3D (needs the viz extra)")
    view.add_argument("scene", help="path to a scene YAML file")
    view.add_argument("--screenshot", metavar="PNG", help="save an image instead of a window")
    view.set_defaults(func=_cmd_view)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        # Material descriptions contain characters a Windows console code page may lack.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
