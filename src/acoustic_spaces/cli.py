"""Command-line interface for acoustic_spaces."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from acoustic_spaces import __version__, native_available
from acoustic_spaces.scene import Scene


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"acoustic_spaces {__version__}")
    print(f"compiled core: {'available' if native_available() else 'missing'}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        scene = Scene.from_yaml(args.scene)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValidationError as exc:
        print(f"error: {args.scene} is not a valid scene:", file=sys.stderr)
        for err in exc.errors():
            where = ".".join(str(part) for part in err["loc"]) or "scene"
            print(f"  {where}: {err['msg']}", file=sys.stderr)
        return 1

    if not scene.geometry.file.exists():
        print(f"error: geometry file not found: {scene.geometry.file}", file=sys.stderr)
        return 1
    print(
        f"{scene.name}: {len(scene.materials)} materials, "
        f"{len(scene.sources)} sources, {len(scene.listeners)} listeners"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acoustic-spaces", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show the installed version and compiled components")
    info.set_defaults(func=_cmd_info)

    validate = sub.add_parser("validate", help="check a scene file")
    validate.add_argument("scene", help="path to a scene YAML file")
    validate.set_defaults(func=_cmd_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
