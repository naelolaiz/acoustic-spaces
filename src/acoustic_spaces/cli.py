"""Command-line interface for acoustic_spaces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

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
    ok = _print_checks(room)
    paths = None
    if args.paths:
        if not ok:
            return 1
        from acoustic_spaces.engine import early_paths

        try:
            paths = early_paths(
                room, args.source, args.listener, max_order=args.max_order, threads=args.threads
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        counts = ", ".join(
            f"{(paths.order == k).sum()} of order {k}" for k in range(int(paths.order.max()) + 1)
        )
        print(f"{len(paths)} specular paths: {counts}")
    try:
        from acoustic_spaces.viz import show_room

        show_room(room, screenshot=args.screenshot, paths=paths)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.screenshot:
        print(f"saved {args.screenshot}")
    return 0


def _file_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unnamed"


def _cmd_simulate(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.engine import simulate
    from acoustic_spaces.materials import OCTAVE_BANDS_HZ

    room = _load_room(args.scene)
    if room is None or not _print_checks(room):
        return 1
    settings = room.scene.simulation.model_copy(deep=True)
    if args.seed is not None:
        settings.seed = args.seed
    if args.rays is not None:
        settings.late.rays = args.rays
    if args.max_order is not None:
        settings.early.max_order = args.max_order
    if args.length is not None:
        settings.ir_length_s = args.length
    sample_rate = args.sample_rate or settings.sample_rate
    try:
        result = simulate(
            room,
            settings=settings,
            threads=args.threads,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    irs = {key: r.impulse_response(sample_rate) for key, r in result.responses.items()}
    peak = max(float(abs(ir).max(initial=0.0)) for ir in irs.values())
    gain = 10.0 ** (-1.0 / 20.0) / peak if args.normalize and peak > 0 else 1.0

    report = {
        "scene": room.scene.name,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "sample_rate": sample_rate,
        "ir_gain": gain,
        "settings": settings.model_dump(mode="json"),
        "details": result.details,
        "responses": [],
    }
    for (source, listener), response in result.responses.items():
        stem = f"{_file_name(source)}__{_file_name(listener)}"
        write_wav(out / f"{stem}.wav", irs[(source, listener)] * gain, sample_rate)
        params = response.parameters()
        print(
            f"\n{source} -> {listener}: direct sound at {response.direct_delay_s * 1e3:.1f} ms, "
            f"{len(response.early)} early paths"
        )
        print(params.table())
        entry = {
            "source": source,
            "listener": listener,
            "wav": f"{stem}.wav",
            "direct_delay_s": response.direct_delay_s,
            "early_paths": len(response.early),
            "parameters": params.as_dict(),
        }
        if args.plot:
            try:
                from acoustic_spaces.viz import plot_response

                plot_response(response, out / f"{stem}.png")
                entry["plot"] = f"{stem}.png"
            except ImportError as exc:
                print(f"warning: {exc}", file=sys.stderr)
                args.plot = False
        report["responses"].append(entry)
    (out / "parameters.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(irs)} impulse responses and parameters.json to {out}")
    if peak * gain > 1.0:
        print(
            "note: the float WAV files exceed full scale (a listener is closer than 1 m to a "
            "source); use --normalize to scale them for playback",
            file=sys.stderr,
        )
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
    view.add_argument("--paths", action="store_true", help="draw early reflection paths")
    view.add_argument("--source", help="source for --paths (default: the first)")
    view.add_argument("--listener", help="listener for --paths (default: the first)")
    view.add_argument("--max-order", type=int, help="highest reflection order for --paths")
    view.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
    view.set_defaults(func=_cmd_view)

    simulate = sub.add_parser("simulate", help="simulate impulse responses and ISO 3382 parameters")
    simulate.add_argument("scene", help="path to a scene YAML file")
    simulate.add_argument("--out", required=True, metavar="DIR", help="folder for the results")
    simulate.add_argument("--sample-rate", type=int, help="override simulation.sample_rate")
    simulate.add_argument("--length", type=float, help="override simulation.ir_length_s")
    simulate.add_argument("--seed", type=int, help="override simulation.seed")
    simulate.add_argument("--rays", type=int, help="override simulation.late.rays")
    simulate.add_argument("--max-order", type=int, help="override simulation.early.max_order")
    simulate.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
    simulate.add_argument(
        "--normalize", action="store_true", help="scale all responses to a -1 dBFS peak"
    )
    simulate.add_argument(
        "--plot", action="store_true", help="save echogram and decay plots (needs viz extra)"
    )
    simulate.set_defaults(func=_cmd_simulate)
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
