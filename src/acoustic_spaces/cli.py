"""Command-line interface for acoustic_spaces."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from acoustic_spaces import __version__, native_available
from acoustic_spaces.engine.ray_device import DEVICES, TRIAL_FLAG
from acoustic_spaces.gltf import PARTICLE_COUNT, SLOW_MOTION
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.room import Room

#: What the map command can plot (the keys of engine.coverage.MAP_PARAMETERS).
MAP_PARAMETER_NAMES = ("spl", "spl_a", "g", "c50", "c80", "d50", "edt", "t20", "t30", "ts", "sti")
#: Edge diffraction settings (engine.diffraction.DIFFRACTION_MODES).
DIFFRACTION_MODES = ("none", "direct", "first_order", "second_order")
#: Highest octave bands the wave solver can compute (scene.model.WaveStage.max_band_hz).
WAVE_BANDS_HZ = (63, 125, 250, 500)
#: Choices for the highest band the early paths add in pressure in
#: (scene.model.EarlyStage.coherent_max_band_hz).
COHERENT_BANDS = ("none", "63", "125", "250", "500")
#: Rays the leak finder traces by default (engine.leaks.RAYS).
LEAK_RAYS = 5000
#: Walkthrough defaults (walkthrough.STEP_M, LATE_STEP_M and CROSSFADE_S).
STEP_M = 0.25
LATE_STEP_M = 2.0
CROSSFADE_S = 0.05


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
    print(f"CPU cores for ray tracing: {os.cpu_count() or 1}")
    from acoustic_spaces.engine.gpu import gpu_name

    print(f"GPU for the wave solver: {gpu_name() or 'none (needs the gpu extra and a driver)'}")
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
    if args.leaks:
        from acoustic_spaces.engine.leaks import find_leaks

        try:
            print(f"leaks: {find_leaks(room, rays=args.rays).summary()}")
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0 if ok else 1


#: Arau-Puchades this many times Eyring in some band: the absorption is spread too
#: unevenly for a diffuse field.
_UNEVEN_ABSORPTION = 1.25


def _uneven_absorption_note(ratio: Sequence[float]) -> str | None:
    """Why the decay will be longer than Sabine and Eyring say, when it will."""
    finite = [(r, k) for k, r in enumerate(ratio) if math.isfinite(r)]
    if not finite or max(finite)[0] <= _UNEVEN_ABSORPTION:
        return None
    worst, band = max(finite)
    frequency = OCTAVE_BANDS_HZ[band]
    where = f"{frequency // 1000} kHz" if frequency >= 1000 else f"{frequency} Hz"
    text = (
        f"note: the absorption is spread unevenly (Arau-Puchades, which allows for that, is "
        f"{worst:.1f} times Eyring at {where}). Sabine and Eyring assume a diffuse field and "
        "underestimate the decay of such rooms: sound travelling parallel to the absorbing "
        "surfaces dies away slowly, the more so the less the other surfaces scatter. "
        "simulate traces it."
    )
    return textwrap.fill(text, width=88, subsequent_indent="  ")


def _cmd_inspect(args: argparse.Namespace) -> int:
    room = _load_room(args.scene)
    if room is None:
        return 1
    scene = room.scene
    print(scene.name)
    print(f"geometry: {scene.geometry.describe()}")
    for line in room.report.summary().splitlines():
        print(f"  {line}")

    width = max([len(n) for n in room.material_names] + [12]) + 2
    print("\nmaterials (absorption per octave band, area)")
    print(_band_header("", width) + "    area m2")
    areas = room.area_by_material()
    for name, material in scene.materials.items():
        absorption = material.coefficients("octave")[0]
        print(_band_row(name, absorption, width) + f"  {areas[name]:9.2f}")
        passing = material.transmission_coefficients("octave")
        if passing is not None:
            reduction = [max(0.0, -10.0 * math.log10(t)) if t > 0 else math.inf for t in passing]
            print(_band_row("  R (dB)", reduction, width, "6.1f") + "  sound reduction")
        if material.audience:
            print("  audience seating: sound arriving low over it is attenuated (500 Hz-2 kHz)")
        if len(material.absorption) > len(absorption):
            print("  third-octave data, shown as octave averages")
        if material.source:
            print(f"  {material.source}")
    unassigned = room.unassigned_labels()
    if unassigned:
        print(_band_row("(no material)", [], width) + f"  {sum(unassigned.values()):9.2f}")

    print("\nsources (directivity index per octave band, dB)")
    print(_band_header("", width))
    for source, directivity in zip(scene.sources, room.directivities, strict=True):
        print(_band_row(source.name, directivity.in_bands("octave").index(), width, "6.1f"))
        print(f"  {_directivity_text(directivity)}")

    print("\nsound power (dB re 1 pW per octave band, the feed included)")
    print(_band_header("", width) + "  dB(A)")
    from acoustic_spaces.parameters import a_weighted

    for source in scene.sources:
        power = source.power_db("octave")
        print(_band_row(source.name, power, width, "6.1f") + f"  {a_weighted(power):5.1f}")
        feed = _feed_text(source)
        if feed:
            print(f"  {feed}")

    if room.volume_m3 is not None:
        print("\nstatistical reverberation time (s), without air absorption")
        print(_band_header("", width))
        times = {}
        for label in ("Sabine", "Eyring", "Arau-Puchades"):
            times[label] = room.reverberation_time(label.lower().replace("-", "_"))
            print(_band_row(label, times[label], width))
        note = _uneven_absorption_note(times["Arau-Puchades"] / times["Eyring"])
        if note:
            print(note)

    print("\npositions")
    for kind, items, inside in (
        ("source", scene.sources, room.sources_inside),
        ("listener", scene.listeners, room.listeners_inside),
    ):
        for k, item in enumerate(items):
            where = "" if inside is None else ("  inside" if inside[k] else "  OUTSIDE")
            x, y, z = item.position
            print(
                f"  {kind:<9}{item.name:<{width}}({x:.2f}, {y:.2f}, {z:.2f}){where}"
                + _extent_text(item)
                + _walk_text(item)
            )
    print()
    return 0 if _print_checks(room) else 1


def _extent_text(item) -> str:
    """'  line 3 m, 6 points' or '  surface 3 x 2 m, 24 points' for an extended source."""
    if not getattr(item, "extended", False):
        return ""
    count = len(item.elements())
    if item.line_m is not None:
        return f"  line {item.line_m:g} m, {count} points"
    width, height = item.surface_m
    return f"  surface {width:g} x {height:g} m, {count} points"


def _walk_text(item) -> str:
    """'  walks 25.0 m to (4.00, 16.00, 1.60) in 20.8 s' for a listener or source with a
    walk."""
    if getattr(item, "walk", None) is None:
        return ""
    corners = [item.position, *item.walk.to]
    length = sum(math.dist(a, b) for a, b in zip(corners, corners[1:], strict=False))
    x, y, z = item.walk.to[-1]
    seconds = length / item.walk.speed_m_s
    return f"  walks {length:.1f} m to ({x:.2f}, {y:.2f}, {z:.2f}) in {seconds:.1f} s"


def _directivity_text(directivity) -> str:
    text = directivity.name or "omni"
    if directivity.extrapolated:
        bands = ", ".join(str(f) for f in directivity.extrapolated)
        text += f" ({bands} Hz copied from the nearest octave)"
    return text


def _cmd_directivity(args: argparse.Namespace) -> int:
    import numpy as np

    from acoustic_spaces.directivity import PATTERNS, FirstOrder, lambert, read_directivity

    try:
        if args.source == "lambert":
            directivity = lambert()
        elif args.source in PATTERNS:
            directivity = FirstOrder.pattern(args.source)
        else:
            directivity = read_directivity(args.source, args.measurement)
    except (ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    width = 22
    print(_directivity_text(directivity))
    directivity = directivity.in_bands("octave")  # an array's is in third octaves
    print(_band_header("", width))
    print(_band_row("directivity index dB", directivity.index(), width, "6.1f"))
    front, back = directivity.energy([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    with np.errstate(divide="ignore"):
        print(_band_row("front to back dB", 10.0 * np.log10(front / back), width, "6.1f"))
    if args.plot:
        try:
            from acoustic_spaces.viz import plot_directivity

            plot_directivity(directivity, args.plot)
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"saved {args.plot}")
    return 0


def _library_materials(args: argparse.Namespace) -> int:
    """The materials of a CATT-Acoustic or ODEON library, with the words searched for."""
    from acoustic_spaces.materials import read_library, to_octave_bands

    try:
        materials = read_library(args.file)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    words = [w.lower() for w in args.query]
    materials = [
        m for m in materials if all(w in f"{m.label} {m.category or ''}".lower() for w in words)
    ]
    if not materials:
        print("no matching materials", file=sys.stderr)
        return 1
    width = max(len("  scattering"), *(len(m.label) for m in materials)) + 2
    print(_band_header("", width))
    category = None
    for material in materials:
        if material.category is not None and material.category != category:
            category = material.category
            print(f"\n{category}")
        values, _ = to_octave_bands(material.frequencies, material.absorption)
        print(_band_row(material.label, values, width))
        if material.scattering is not None:
            values, _ = to_octave_bands(material.frequencies, material.scattering)
            print(_band_row("  scattering", values, width))
    text = (
        f"In a scene, a material takes one with file: {Path(args.file).name} and entry: its "
        "name (or ODEON's number). Bands the library has no values for take the nearest "
        "band's."
    )
    print("\n" + textwrap.fill(text, width=88))
    return 0


def _cmd_materials(args: argparse.Namespace) -> int:
    if args.file is not None:
        return _library_materials(args)
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


def _cmd_examples(args: argparse.Namespace) -> int:
    import textwrap

    from acoustic_spaces._extras import install_hint
    from acoustic_spaces.examples import copy_examples, list_examples

    try:
        if args.copy:
            for scene_file in copy_examples(args.copy, args.names or None):
                print(scene_file)
            return 0
        examples = list_examples()
        unknown = sorted(set(args.names) - {e.name for e in examples})
        if unknown:
            raise ValueError(f"no example named {', '.join(unknown)}")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for example in examples:
        if args.names and example.name not in args.names:
            continue
        print(f"{example.name}: {example.title}")
        print(textwrap.fill(example.description, 88, initial_indent="  ", subsequent_indent="  "))
        print(f"  {example.path}")
        extra = example.missing_extra
        if extra:
            print(f"  needs the {extra} extra: {install_hint(extra)}")
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    try:
        from acoustic_spaces.gui import main as editor

        argv = [args.scene] if args.scene else []
        if args.no_device_trial:
            argv.append(TRIAL_FLAG)
        return editor(argv)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_view(args: argparse.Namespace) -> int:
    room = _load_room(args.scene)
    if room is None:
        return 1
    ok = _print_checks(room)
    paths = leaks = None
    if args.leaks:
        from acoustic_spaces.engine.leaks import find_leaks

        try:
            leaks = find_leaks(room, rays=args.rays)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"leaks: {leaks.summary()}")
    if args.paths:
        if not ok:
            return 1
        from acoustic_spaces.engine import early_paths

        try:
            paths = early_paths(
                room,
                args.source,
                args.listener,
                max_order=args.max_order,
                diffraction=args.diffraction,
                threads=args.threads,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"{len(paths)} paths: {_path_counts(paths)}")
    try:
        from acoustic_spaces.viz import show_room

        show_room(room, screenshot=args.screenshot, paths=paths, leaks=leaks)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.screenshot:
        print(f"saved {args.screenshot}")
    return 0


def _cmd_animate(args: argparse.Namespace) -> int:
    from acoustic_spaces.engine import trace_particles

    room = _load_room(args.scene)
    if room is None or not _print_checks(room):
        return 1
    try:
        particles = trace_particles(
            room,
            args.source,
            count=args.particles,
            duration=args.duration / 1e3,
            frames=args.frames,
            band=OCTAVE_BANDS_HZ.index(args.band),
            seed=args.seed,
        )
        from acoustic_spaces.viz import animate_particles

        animate_particles(room, particles, args.out, fps=args.fps, color=args.color)
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        print(f"saved {args.out}: {args.frames} frames, {args.duration:g} ms of sound")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from acoustic_spaces.engine import early_paths, trace_particles
    from acoustic_spaces.gltf import export_gltf

    room = _load_room(args.scene)
    if room is None:
        return 1
    if not _print_checks(room) and (args.paths or args.particles):
        return 1
    scene = room.scene
    paths = particles = None
    try:
        if args.paths:
            paths = early_paths(
                room,
                args.source,
                args.listener,
                max_order=args.max_order,
                diffraction=args.diffraction,
                threads=args.threads,
            )
        if args.particles:
            particles = trace_particles(
                room,
                args.source,
                count=args.count,
                duration=args.duration / 1e3,
                frames=args.frames,
                band=OCTAVE_BANDS_HZ.index(args.band),
                seed=args.seed,
            )
        export_gltf(room, args.out, paths=paths, particles=particles, slow_motion=args.slow_motion)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    size = Path(args.out).stat().st_size
    size_text = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} kB"
    surfaces = len(set(room.mesh.face_labels.tolist()))
    print(
        f"saved {args.out} ({size_text}): {surfaces} surfaces, "
        f"{len(scene.sources)} sources, {len(scene.listeners)} listeners"
    )
    if paths is not None:
        source = args.source or scene.sources[0].name
        listener = args.listener or scene.listeners[0].name
        print(f"  {len(paths)} paths from {source} to {listener}: {_path_counts(paths)}")
    if particles is not None:
        seconds = args.slow_motion * float(particles.times[-1])
        print(
            f"  {args.count} particles from {particles.source}: {args.duration:g} ms of sound "
            f"over {seconds:g} s ({round(24 * seconds)} frames at 24 fps)"
        )
    print("Open it in Blender with File > Import > glTF 2.0.")
    return 0


def _with_coherent(settings, choice: str | None):
    """``settings`` (``scene.simulation``) with the highest band the early paths add in
    pressure in, as ``--coherent-band`` gives it (None: as they are)."""
    if settings is None or choice is None:
        return settings
    band = None if choice == "none" else int(choice)
    early = settings.early.model_copy(update={"coherent_max_band_hz": band})
    return settings.model_copy(update={"early": early})


def _override(model, options: dict[str, tuple[str, object]]):
    """A copy of ``model`` with the fields the command line sets, checked as in a scene.

    ``options`` maps each option (``--rays``) to the field it sets (``late.rays``) and
    the value given, None when it was not. A value out of range is printed as an error
    naming its option, and None returned.
    """
    data = model.model_dump()
    for field, value in options.values():
        if value is not None:
            *parents, name = field.split(".")
            target = data
            for key in parents:
                target = target[key]
            target[name] = value
    try:
        return type(model).model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            where = ".".join(str(part) for part in err["loc"])
            option = next(
                (o for o, (field, _) in options.items() if f"{where}.".startswith(f"{field}.")),
                where or "options",
            )
            message = err["msg"].removeprefix("Value error, ")
            print(f"error: {option}: {message}", file=sys.stderr)
        return None


def _output_spec(text: str):
    """An outputs entry from the command line: mono, binaural, ambisonics[:ORDER] or
    loudspeakers:LAYOUT."""
    from acoustic_spaces.scene import Output

    name, _, value = text.partition(":")
    entry: dict = {"format": name}
    if name == "ambisonics" and value:
        entry["order"] = int(value)
    elif name == "loudspeakers":
        entry["layout"] = value or None
    elif value:
        raise ValueError(f"--output {text}: {name} takes no option")
    return Output.model_validate(entry)


def _outputs(room: Room, args: argparse.Namespace) -> list:
    """The scene's outputs, or those given with --output; --hrtf applies to binaural."""
    outputs = [_output_spec(t) for t in args.output] if args.output else room.scene.outputs
    if args.hrtf:
        if not any(o.format == "binaural" for o in outputs):
            outputs = [*outputs, _output_spec("binaural")]
        outputs = [
            o.model_copy(update={"hrtf": Path(args.hrtf)}) if o.format == "binaural" else o
            for o in outputs
        ]
    return outputs


THREADS_HELP = "worker threads (default: the scene's simulation.performance.threads; 0: every core)"


def _threads(value: str) -> int:
    """--threads: a count of worker threads, 0 for every core."""
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("give 0 (every core) or more threads")
    return count


def _add_performance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threads", type=_threads, help=THREADS_HELP)
    parser.add_argument(
        "--ray-tree",
        choices=("sah", "median"),
        help="override simulation.performance.ray_tree: split the tree that finds the "
        "surface a ray meets by the surface area heuristic (sah, fastest) or at the "
        "middle face (median, as before; for comparison)",
    )


def _performance_options(args: argparse.Namespace) -> dict[str, tuple[str, object]]:
    return {
        "--threads": ("performance.threads", args.threads),
        "--ray-tree": ("performance.ray_tree", args.ray_tree),
    }


TRIAL_HELP = (
    "with simulation.late.device auto, skip the short trial that times the CPU and the GPU "
    "on the scene, and take the choice saved from the last trial on this computer (the "
    "CPU without one)"
)


def _add_trial_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(TRIAL_FLAG, action="store_true", help=TRIAL_HELP)


def _cmd_simulate(args: argparse.Namespace) -> int:
    from acoustic_spaces.engine import simulate
    from acoustic_spaces.results import REPORT_FILE, write_results
    from acoustic_spaces.spatial import from_output

    room = _load_room(args.scene)
    if room is None or not _print_checks(room):
        return 1
    settings = _override(
        room.scene.simulation,
        {
            "--seed": ("seed", args.seed),
            "--bands": ("bands", args.bands),
            "--rays": ("late.rays", args.rays),
            "--max-order": ("early.max_order", args.max_order),
            "--diffraction": ("early.diffraction", args.diffraction),
            "--wave": ("wave.enabled", args.wave),
            "--wave-band": ("wave.max_band_hz", args.wave_band),
            "--wave-device": ("wave.device", args.wave_device),
            "--late-device": ("late.device", args.late_device),
            "--length": ("ir_length_s", args.length),
            "--sample-rate": ("sample_rate", args.sample_rate),
            **_performance_options(args),
        },
    )
    settings = _with_coherent(settings, args.coherent_band)
    if settings is None:
        return 1
    try:
        outputs = _outputs(room, args)
        formats = [from_output(o) for o in outputs]
    except (ValueError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.no_device_trial:
        from acoustic_spaces.engine import ray_device

        ray_device.set_trial(False)
    try:
        result = simulate(
            room,
            settings=settings,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    report = write_results(
        result,
        out,
        formats,
        sample_rate=settings.sample_rate,
        normalize=args.normalize,
        plots=args.plot,
        progress=lambda message: print(message, file=sys.stderr),
    )
    from acoustic_spaces.parameters import RoomParameters

    for entry, ((source, listener), response) in zip(
        report["responses"], result.responses.items(), strict=True
    ):
        wave = ""
        if response.wave is not None:
            centres = response.band_hz
            wave = f", {centres[0]}-{centres[response.wave.bands - 1]} Hz from the wave solver"
        print(
            f"\n{source} -> {listener}: direct sound at {response.direct_delay_s * 1e3:.1f} ms, "
            f"{_early_text(response.early)}{wave}"
        )
        print(RoomParameters.from_dict(entry["parameters"]).table())
        print(_level_text(entry))
        print(_speech_text(entry["speech"], entry["echo"]))
    files = sum(len(entry["outputs"]) for entry in report["responses"])
    print(f"\nwrote {files} impulse response files and {REPORT_FILE} to {out}")
    if report["peak"] > 1.0:
        print(
            "note: the float WAV files exceed full scale (a listener is closer than 1 m to a "
            "source); use --normalize to scale them for playback",
            file=sys.stderr,
        )
    return 0


def _feed_text(source) -> str:
    """What shapes a source's level beyond its sound power: 'speech_male spectrum,
    gain +3.0 dB, equalizer, 12.0 ms delay', or nothing."""
    parts = []
    if source.spectrum != "flat":
        parts.append(f"{source.spectrum} spectrum")
    if source.gain_db:
        parts.append(f"gain {source.gain_db:+.1f} dB")
    if source.eq_db is not None:
        parts.append("equalizer")
    if source.delay_ms:
        parts.append(f"{source.delay_ms:g} ms delay")
    return ", ".join(parts)


def _level_text(entry: dict) -> str:
    """One line: the A-weighted sound pressure level and that of each octave band."""
    bands = " ".join(_number(v, ".1f") for v in entry["spl_db"])
    return f"SPL {_number(entry['spl_a_db'], '.1f')} dB(A); per octave band, dB: {bands}"


def _number(value, fmt: str = ".2f") -> str:
    return "-" if value is None else f"{value:{fmt}}"


def _speech_text(speech: dict | None, echo: dict) -> str:
    """One line: STI and STIPA (when given), and the echo criteria with the values at
    which half of listeners hear an echo."""
    from acoustic_spaces.parameters import ECHO_CRITERIA

    parts = []
    if speech is not None:
        parts.append(f"STI {_number(speech['sti'])} (STIPA {_number(speech['stipa'])})")
    criteria = ", ".join(
        f"{_number(echo[kind]['ec'])} for {kind} (echo above {ECHO_CRITERIA[kind][3][1]:.1f})"
        for kind in ECHO_CRITERIA
    )
    parts.append(f"echo criterion {criteria}")
    return ", ".join(parts)


def _path_counts(paths) -> str:
    """'3 of order 0, 5 of order 1, 2 diffracted' for early paths."""
    specular = paths.order[~paths.diffracted]
    top = int(specular.max()) if len(specular) else -1
    parts = [f"{int((specular == k).sum())} of order {k}" for k in range(top + 1)]
    if paths.diffracted.any():
        parts.append(f"{int(paths.diffracted.sum())} diffracted")
    return ", ".join(parts) or "none"


def _early_text(paths) -> str:
    """'12 early paths (3 diffracted)'."""
    diffracted = int(paths.diffracted.sum())
    return f"{len(paths)} early paths" + (f" ({diffracted} diffracted)" if diffracted else "")


def _cmd_map(args: argparse.Namespace) -> int:
    import numpy as np

    from acoustic_spaces.engine import coverage_map
    from acoustic_spaces.engine.coverage import MAP_PARAMETERS
    from acoustic_spaces.scene import Coverage

    room = _load_room(args.scene)
    if room is None:
        return 1
    for warning in room.warnings():
        print(f"warning: {warning}", file=sys.stderr)
    scene = room.scene
    simulation = _override(
        scene.simulation,
        {
            "--seed": ("seed", args.seed),
            "--max-order": ("early.max_order", args.max_order),
            "--diffraction": ("early.diffraction", args.diffraction),
            "--length": ("ir_length_s", args.length),
            **_performance_options(args),
        },
    )
    simulation = _with_coherent(simulation, args.coherent_band)
    noise = args.noise[0] if args.noise is not None and len(args.noise) == 1 else args.noise
    settings = _override(
        scene.coverage or Coverage(),
        {
            "--surfaces": ("surfaces", args.surfaces),
            "--height": ("height", args.height),
            "--spacing": ("spacing", args.spacing),
            "--rays": ("rays", args.rays),
            "--noise": ("noise_db", noise),
        },
    )
    if simulation is None or settings is None:
        return 1
    try:
        result = coverage_map(
            room,
            settings=settings,
            simulation=simulation,
            sources=[args.source] if args.source else None,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    grid = result.grid
    band = OCTAVE_BANDS_HZ.index(args.band)
    band_text = f"{args.band} Hz" if args.band < 1000 else f"{args.band // 1000} kHz"
    levels = grid.index.shape[2]
    print(
        f"\n{len(grid)} map points {grid.cell:g} m apart, {settings.height:g} m above the "
        f"audience surfaces" + (f", on {levels} levels" if levels > 1 else "")
    )
    for chosen in [*result.maps.values(), *([result.combined] if result.combined else [])]:
        print(f"\n{chosen.source} (bands: {band_text})")
        print(f"{'':<14}{'min':>8}{'median':>8}{'max':>8}")
        for name, info in MAP_PARAMETERS.items():
            values = chosen.value(name, band if info.per_band else None)
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                row = ["-"] * 3
            else:
                row = [info.text(float(v)) for v in (finite.min(), np.median(finite), finite.max())]
            print(f"{info.label:<14}" + "".join(f"{cell:>8}" for cell in row))

    shown = result.get(args.source)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    try:
        if args.plot:
            from acoustic_spaces.viz import plot_map

            plot_map(shown, args.plot, parameter=args.parameter, band=band, room=room)
            print(f"saved {args.plot}")
        if args.screenshot:
            from acoustic_spaces.viz import show_room

            show_room(
                room,
                screenshot=args.screenshot,
                coverage=shown,
                parameter=args.parameter,
                band=band,
            )
            print(f"saved {args.screenshot}")
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _sweep_from(args: argparse.Namespace):
    from acoustic_spaces.measurement import Sweep

    return Sweep(
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        f_start=args.f_start,
        f_stop=args.f_stop,
        level_db=args.level,
        silence_s=args.silence,
        repeats=args.repeats,
    )


def _describe(sweep) -> str:
    seconds = sweep.period * sweep.repeats / sweep.sample_rate
    return (
        f"{sweep.repeats} x {sweep.sweep_length / sweep.sample_rate:.2f} s sweep, "
        f"{sweep.f_start:g} Hz to {sweep.f_stop:g} Hz at {sweep.level_db:g} dBFS, "
        f"{sweep.silence_s:g} s silence after each ({seconds:.1f} s in all, "
        f"{sweep.sample_rate} Hz)"
    )


def _cmd_sweep(args: argparse.Namespace) -> int:
    try:
        sweep = _sweep_from(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        sweep.save(out)
    except OSError as exc:
        print(f"error: cannot write {out}: {exc}", file=sys.stderr)
        return 1
    print(_describe(sweep))
    print(f"wrote {out} and {out.with_suffix('.json')}")
    return 0


def _load_sweep(path: str):
    from acoustic_spaces.measurement import Sweep

    try:
        return Sweep.load(path)
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: cannot read the sweep settings for {path}: {exc}", file=sys.stderr)
        return None


def _channel_index(channel: int, count: int, what: str) -> int | None:
    if not 1 <= channel <= count:
        print(f"error: {what} {channel} is not among channels 1 to {count}", file=sys.stderr)
        return None
    return channel - 1


def _cmd_deconvolve(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import read_wav, write_wav
    from acoustic_spaces.measurement import deconvolve, remove_latency

    sweep = _load_sweep(args.sweep)
    if sweep is None:
        return 1
    try:
        recording, sample_rate = read_wav(args.recording)
        if sample_rate != sweep.sample_rate:
            raise ValueError(
                f"the recording is at {sample_rate} Hz, the sweep at {sweep.sample_rate} Hz"
            )
        responses = deconvolve(recording, sweep, length_s=args.length)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.loopback is not None:
        column = _channel_index(args.loopback, responses.shape[1], "loopback channel")
        if column is None:
            return 1
        responses, lag = remove_latency(responses, column)
        print(f"latency from the loopback channel: {lag / sample_rate * 1e3:.2f} ms")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out, responses, sample_rate)
    print(
        f"wrote {responses.shape[1]} impulse response(s) of "
        f"{len(responses) / sample_rate:.2f} s to {args.out}"
    )
    return 0


def _reference_energy(args: argparse.Namespace, sample_rate: int, side: str = ""):
    """Reference for G from --reference-energy or --free-field and --distance; with
    ``side``, from that response's own options, as --measured-free-field."""
    from acoustic_spaces.audio import read_wav
    from acoustic_spaces.parameters import free_field_reference

    prefix = f"{side}_" if side else ""
    free_field, distance = (getattr(args, prefix + name) for name in ("free_field", "distance"))
    if free_field is None:
        return getattr(args, f"{prefix}reference_energy")
    if distance is None:
        option = f"--{side}-" if side else "--"
        raise ValueError(f"{option}free-field needs {option}distance")
    data, rate = read_wav(free_field)
    if rate != sample_rate:
        response = f"the {side} response" if side else "the response"
        raise ValueError(f"{free_field} is at {rate} Hz, {response} at {sample_rate} Hz")
    return free_field_reference(data[:, 0], rate, distance, filters=args.filters)


def _analysis(ir, sample_rate: int, args: argparse.Namespace, reference, **more) -> dict:
    """Print and return the parameters of one impulse response; ``more`` are the lateral
    and directional responses for them, and ``iacc`` the IACC of the ears it is one of."""
    from acoustic_spaces.parameters import (
        ECHO_CRITERIA,
        band_decay,
        echo_criterion,
        from_impulse_response,
        sti_from_impulse_response,
    )

    iacc = more.pop("iacc", None)
    noise = not args.no_noise
    params = from_impulse_response(
        ir, sample_rate, reference, filters=args.filters, noise=noise, stage=args.stage, **more
    )
    if iacc is not None:
        params = dataclasses.replace(params, **iacc)
    decay = band_decay(ir, sample_rate, filters=args.filters, noise=noise)
    print(f"direct sound at {decay.onset_s * 1e3:.1f} ms")
    if noise:
        ranges = [f"{r:6.0f}" if r != float("inf") else f"{'-':>6}" for r in decay.decay_range_db]
        print(f"{'decay range (dB)':<18}" + "".join(ranges))
    print(params.table())
    speech = {
        name: sti_from_impulse_response(
            ir, sample_rate, filters=args.filters, noise=noise, stipa=name == "stipa"
        )
        for name in ("sti", "stipa")
    }
    echo = {}
    for kind in ECHO_CRITERIA:
        value, at = echo_criterion(ir, sample_rate, kind, filters=args.filters)
        finite = value == value
        echo[kind] = {"ec": value if finite else None, "at_s": at if finite else None}
    print(_speech_text(speech, echo))
    return {
        "onset_s": decay.onset_s,
        "crosspoint_s": (decay.crosspoint / sample_rate).tolist() if noise else None,
        "decay_range_db": [None if r == float("inf") else float(r) for r in decay.decay_range_db]
        if noise
        else None,
        "parameters": params.as_dict(),
        "speech": speech,
        "echo": echo,
    }


def _plot_ir(ir, sample_rate: int, path: Path, title: str, args: argparse.Namespace) -> bool:
    try:
        from acoustic_spaces.viz import plot_impulse_response

        path.parent.mkdir(parents=True, exist_ok=True)
        plot_impulse_response(
            ir, sample_rate, path, title=title, filters=args.filters, noise=not args.no_noise
        )
    except ImportError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return False
    return True


def _analysed_channels(data, args: argparse.Namespace) -> list[tuple[int, str, dict]] | None:
    """The channels ``analyze`` reports (0-based), each with its title and the further
    responses for its parameters; None after printing why the file does not fit."""
    from acoustic_spaces.parameters import interaural_cross_correlation

    count = data.shape[1]
    if args.ambisonic or args.binaural:
        need, what = (4, "--ambisonic") if args.ambisonic else (2, "--binaural")
        if count < need:
            print(f"error: {what} needs {need} channels, the file has {count}", file=sys.stderr)
            return None
        if args.channel is not None:
            print(f"error: --channel does not go with {what}", file=sys.stderr)
            return None
    if args.ambisonic:
        # AmbiX: W, Y, Z, X; the figure of eight with its null on the source is Y.
        more = {"lateral": data[:, 1], "directional": data[:, [3, 1, 2]]}
        return [(0, "W", more)]
    if args.binaural:
        iacc = interaural_cross_correlation(data[:, :2], args.sample_rate, filters=args.filters)
        return [(c, ear, {"iacc": iacc}) for c, ear in ((0, "left ear"), (1, "right ear"))]
    more = {}
    if args.lateral is not None:
        column = _channel_index(args.lateral, count, "lateral channel")
        if column is None:
            return None
        more["lateral"] = data[:, column]
    if args.channel is not None:
        if _channel_index(args.channel, count, "channel") is None:
            return None
        channels = [args.channel - 1]
    elif args.lateral is not None:
        channels = [0 if args.lateral != 1 else 1]
    else:
        channels = list(range(count))
    return [(c, f"channel {c + 1}", more) for c in channels]


def _cmd_analyze(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import read_wav

    try:
        data, sample_rate = read_wav(args.ir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    args.sample_rate = sample_rate
    channels = _analysed_channels(data, args)
    if channels is None:
        return 1
    try:
        reference = _reference_energy(args, sample_rate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    name = Path(args.ir).name
    report = {"file": name, "bands_hz": list(OCTAVE_BANDS_HZ), "channels": []}
    for c, title, more in channels:
        print(f"\n{name}, {title}")
        try:
            analysis = _analysis(data[:, c], sample_rate, args, reference, **more)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        entry = {"channel": c + 1, **analysis}
        if args.plot:
            plot = Path(args.plot)
            if len(channels) > 1:
                plot = plot.with_name(f"{plot.stem}_{c + 1}{plot.suffix}")
            if _plot_ir(data[:, c], sample_rate, plot, f"{name}, ch {c + 1}", args):
                print(f"saved {plot}")
        report["channels"].append(entry)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def _compared_response(args: argparse.Namespace, side: str):
    """The response, sample rate, channel (1-based) and reference for G of one side of
    ``compare``, or None after printing why it cannot be read."""
    from acoustic_spaces.audio import read_wav

    path, channel = getattr(args, side), getattr(args, f"{side}_channel")
    try:
        data, sample_rate = read_wav(path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    column = _channel_index(1 if channel is None else channel, data.shape[1], f"{side} channel")
    if column is None:
        return None
    if channel is None and data.shape[1] > 1:
        print(
            f"note: {path} has {data.shape[1]} channels; comparing the first "
            f"(--{side}-channel picks another)",
            file=sys.stderr,
        )
    try:
        reference = _reference_energy(args, sample_rate, side)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None
    return data[:, column], sample_rate, column + 1, reference


def _cmd_compare(args: argparse.Namespace) -> int:
    from acoustic_spaces.compare import compare_impulse_responses
    from acoustic_spaces.parameters import onset

    sides = {}
    for side in ("simulated", "measured"):
        response = _compared_response(args, side)
        if response is None:
            return 1
        sides[side] = response
    (sim, sim_rate, _, sim_reference), (meas, meas_rate, _, meas_reference) = sides.values()
    noise = not args.no_noise
    comparison = compare_impulse_responses(
        sim,
        sim_rate,
        meas,
        meas_rate,
        simulated_reference=sim_reference,
        measured_reference=meas_reference,
        filters=args.filters,
        noise=noise,
    )
    report = {}
    for side, (ir, sample_rate, channel, _) in sides.items():
        name, start = Path(getattr(args, side)).name, onset(ir) / sample_rate
        print(
            f"{side + ':':<11}{name}, channel {channel}, {sample_rate} Hz, "
            f"direct sound at {start * 1e3:.1f} ms"
        )
        report[side] = {
            "file": name,
            "channel": channel,
            "sample_rate": sample_rate,
            "onset_s": start,
        }
    print(f"\n{comparison.table()}\n\n{comparison.summary()}")
    if args.plot:
        # As with analyze, a plot that cannot be drawn costs a warning, not the results.
        plot = Path(args.plot)
        try:
            from acoustic_spaces.viz import plot_comparison

            plot.parent.mkdir(parents=True, exist_ok=True)
            names = (report["simulated"]["file"], report["measured"]["file"])
            plot_comparison(
                sim,
                sim_rate,
                meas,
                meas_rate,
                plot,
                title=f"{names[0]} (simulated) against {names[1]} (measured)",
                filters=args.filters,
                noise=noise,
            )
        except ImportError as exc:
            print(f"warning: {exc}", file=sys.stderr)
        else:
            print(f"\nsaved {plot}")
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.update(comparison.as_dict())
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
    return 0


def _measured_parameters(entry: dict):
    """RoomParameters from a JSON entry that may give only some parameters."""
    from acoustic_spaces.parameters import PARAMETERS, RoomParameters

    missing = [None] * len(OCTAVE_BANDS_HZ)
    return RoomParameters.from_dict({**{name: missing for name in PARAMETERS}, **entry})


def _measured_pair(path: str, args: argparse.Namespace):
    """The parameters of a measured impulse response WAV, or those of the first channel
    in a report written by ``analyze --json``."""
    from acoustic_spaces.audio import read_wav
    from acoustic_spaces.parameters import from_impulse_response

    if Path(path).suffix.lower() == ".json":
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return _measured_parameters(report["channels"][0]["parameters"])
        except (KeyError, IndexError, TypeError):
            raise ValueError(f"{path} is not a report written by analyze --json") from None
    data, sample_rate = read_wav(path)
    channel = args.channel or 1
    if not 1 <= channel <= data.shape[1]:
        raise ValueError(f"{path} has no channel {channel} (it has {data.shape[1]})")
    return from_impulse_response(
        data[:, channel - 1], sample_rate, filters=args.filters, noise=not args.no_noise
    )


def _measured_pairs(args: argparse.Namespace) -> dict:
    """The measured parameters per (source, listener), from --measured and
    --measured-json."""
    pairs = {}
    for source, listener, path in args.measured or []:
        pairs[source, listener] = _measured_pair(path, args)
    for path in args.measured_json or []:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = data.get("pairs") if isinstance(data, dict) else None
        if not isinstance(entries, dict):
            raise ValueError(f'{path} has no "pairs": {{"SOURCE LISTENER": {{...}}}}')
        for key, entry in entries.items():
            names = key.replace("/", " ").split()
            if len(names) != 2:
                raise ValueError(f'{path}: "{key}" is not "SOURCE LISTENER"')
            pairs[names[0], names[1]] = _measured_parameters(entry)
    return pairs


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from acoustic_spaces.calibration import calibrate

    room = _load_room(args.scene)
    if room is None or not _print_checks(room):
        return 1
    settings = _override(
        room.scene.simulation,
        {
            "--rays": ("late.rays", args.rays),
            "--seed": ("seed", args.seed),
            "--length": ("ir_length_s", args.length),
            **_performance_options(args),
        },
    )
    if settings is None:
        return 1
    own = room.scene.simulation
    room.scene.simulation = settings
    try:
        measured = _measured_pairs(args)
        if not measured:
            print("error: give --measured or --measured-json", file=sys.stderr)
            return 1
        found = calibrate(
            room,
            measured,
            args.materials,
            parameters=args.parameters,
            rounds=args.rounds,
            progress=lambda message: print(message, file=sys.stderr),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(measured)} measured pairs, {found.rounds} rounds\n")
    print(found.table())
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The scene's own settings, not those of the command line.
        found.fitted_scene().model_copy(update={"simulation": own}).to_yaml(path)
        print(f"\nwrote {path}")
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(found.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


def _cmd_devices(_: argparse.Namespace) -> int:
    from acoustic_spaces.measurement import list_devices

    try:
        print(list_devices())
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.measurement import measure

    try:
        sweep = _sweep_from(args)
        reference = _reference_energy(args, sweep.sample_rate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    print(_describe(sweep), file=sys.stderr)
    print("playing and recording...", file=sys.stderr)
    try:
        result = measure(
            sweep,
            inputs=args.input,
            output=args.output,
            loopback=args.loopback,
            device=device,
            length_s=args.length,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep.save(out / "sweep.wav")
    write_wav(out / "recording.wav", result.recording, sweep.sample_rate)
    write_wav(out / "ir.wav", result.impulse_responses, sweep.sample_rate)
    peak = float(abs(result.recording).max(initial=0.0))
    if peak >= 0.99:
        print("warning: the recording clips; lower --level or the gain", file=sys.stderr)
    report = {
        "sweep": "sweep.json",
        "recording": "recording.wav",
        "impulse_responses": "ir.wav",
        "inputs": list(args.input),
        "output": args.output,
        "loopback": args.loopback,
        "latency_s": result.latency_s,
        "recording_peak_dbfs": 20.0 * math.log10(peak) if peak > 0 else None,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "channels": [],
    }
    if result.latency_s is not None:
        print(f"latency from the loopback channel: {result.latency_s * 1e3:.2f} ms")
    for k, channel in enumerate(args.input):
        ir = result.impulse_responses[:, k]
        print(f"\ninput {channel}")
        entry = {"input": channel, **_analysis(ir, sweep.sample_rate, args, reference)}
        if args.plot and _plot_ir(
            ir, sweep.sample_rate, out / f"ir_{channel}.png", f"input {channel}", args
        ):
            entry["plot"] = f"ir_{channel}.png"
        report["channels"].append(entry)
    (out / "measurement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote sweep, recording, impulse responses and measurement.json to {out}")
    return 0


def _read_dry(path: str, sample_rate: int):
    """A dry signal, mixed down to mono and resampled to ``sample_rate``."""
    from acoustic_spaces.audio import read_wav
    from acoustic_spaces.render import resample

    dry, rate = read_wav(path)
    if dry.shape[1] > 1:
        print(f"note: mixed the {dry.shape[1]} channels of {path} down to mono", file=sys.stderr)
    if rate != sample_rate:
        print(f"note: resampled {path} from {rate} Hz to {sample_rate} Hz", file=sys.stderr)
    return resample(dry.mean(axis=1), rate, sample_rate)


def _write_wet(path: Path, data, sample_rate: int, mask: int | None, normalize: bool) -> None:
    from acoustic_spaces.audio import write_wav

    peak = float(abs(data).max(initial=0.0))
    gain = 10.0 ** (-1.0 / 20.0) / peak if normalize and peak > 0 else 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, data * gain, sample_rate, mask)
    level = f"{20.0 * math.log10(peak * gain):+.1f} dBFS" if peak > 0 else "silent"
    print(
        f"wrote {path}: {data.shape[1]} channel(s), {len(data) / sample_rate:.2f} s at "
        f"{sample_rate} Hz, peak {level}"
    )
    if peak * gain > 1.0:
        print("note: the float WAV exceeds full scale; use --normalize to scale it for playback")


def _cmd_convolve(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import channel_mask, read_wav
    from acoustic_spaces.render import convolve

    try:
        ir, rate = read_wav(args.ir)
        mask = channel_mask(args.ir)
        dry = _read_dry(args.signal, rate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_wet(Path(args.out), convolve(dry, ir), rate, mask, args.normalize)
    return 0


def _simulated_irs(folder: Path, listener: str | None, label: str) -> tuple[str, dict[str, Path]]:
    """The listener and, per source, the response file of format ``label`` in a folder
    written by ``simulate``."""
    report = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
    entries = report["responses"]
    listeners = sorted({e["listener"] for e in entries})
    if listener is None:
        if len(listeners) != 1:
            raise ValueError(f"pick a listener with --listener: {', '.join(listeners)}")
        listener = listeners[0]
    elif listener not in listeners:
        raise ValueError(f"no listener {listener!r}; the results have {', '.join(listeners)}")
    files, labels = {}, set()
    for entry in entries:
        if entry["listener"] != listener:
            continue
        outputs = entry.get("outputs") or [{"format": "mono", "wav": entry["wav"]}]
        labels.update(o["format"] for o in outputs)
        for output in outputs:
            if output["format"] == label:
                files[entry["source"]] = folder / output["wav"]
    if not files:
        raise ValueError(f"no {label} responses; the results have {', '.join(sorted(labels))}")
    return listener, files


def _cmd_auralize(args: argparse.Namespace) -> int:
    import numpy as np

    from acoustic_spaces.audio import channel_mask, read_wav
    from acoustic_spaces.render import convolve

    try:
        listener, files = _simulated_irs(Path(args.results), args.listener, args.format)
        pairs = [text.partition("=") for text in args.signals]
        bad = [text for text in args.signals if not text.partition("=")[2]]
        if bad:
            raise ValueError(f"expected SOURCE=WAV, got {bad[0]!r}")
        missing = [name for name, _, _ in pairs if name not in files]
        if missing:
            raise ValueError(f"no source {missing[0]!r}; the results have {', '.join(files)}")
        mix, rate, mask = None, None, None
        for name, _, wav in pairs:
            ir, ir_rate = read_wav(files[name])
            if rate is None:
                rate, mask = ir_rate, channel_mask(files[name])
            wet = convolve(_read_dry(wav, rate), ir)
            if mix is None:
                mix = np.zeros((0, wet.shape[1]))
            if len(wet) > len(mix):
                mix = np.concatenate([mix, np.zeros((len(wet) - len(mix), mix.shape[1]))])
            mix[: len(wet)] += wet
            print(f"{name} -> {listener}: {files[name].name}", file=sys.stderr)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_wet(Path(args.out), mix, rate, mask, args.normalize)
    return 0


def _cmd_walkthrough(args: argparse.Namespace) -> int:
    from acoustic_spaces.spatial import from_output
    from acoustic_spaces.walkthrough import render_walk, simulate_walk

    room = _load_room(args.scene)
    if room is None or not _print_checks(room):
        return 1
    settings = _override(
        room.scene.simulation,
        {
            "--seed": ("seed", args.seed),
            "--rays": ("late.rays", args.rays),
            "--length": ("ir_length_s", args.length),
        },
    )
    if settings is None:
        return 1
    rate = settings.sample_rate
    try:
        if args.format:
            output = _output_spec(args.format)
        else:
            output = room.scene.outputs[0] if room.scene.outputs else _output_spec("mono")
        if args.hrtf and output.format == "binaural":
            output = output.model_copy(update={"hrtf": Path(args.hrtf)})
        fmt = from_output(output)
        bad = [text for text in args.signals if not text.partition("=")[2]]
        if bad:
            raise ValueError(f"expected SOURCE=WAV, got {bad[0]!r}")
        if args.video:
            from acoustic_spaces.viz import check_video

            check_video(args.video)  # before the simulation, not after it
        signals = {
            name: _read_dry(wav, rate)
            for name, _, wav in (text.partition("=") for text in args.signals)
        }
        walk = simulate_walk(
            room,
            args.walker,
            listener=args.listener,
            step_m=args.step,
            late_step_m=args.late_step or None,
            settings=settings,
            threads=args.threads,
            progress=lambda message: print(message, file=sys.stderr),
        )
        wet = render_walk(walk, signals, fmt, crossfade_s=args.crossfade * 1e-3)
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{walk.name} walks for {walk.duration_s:.1f} s: responses at {len(walk.names)} places, "
        f"the late part traced at {len(walk.late_places)} of them",
        file=sys.stderr,
    )
    mask = fmt.channel_mask() if hasattr(fmt, "channel_mask") else None
    _write_wet(Path(args.out), wet, rate, mask, args.normalize)
    if args.video:
        from acoustic_spaces.viz import animate_walk

        # An MP4 takes the sound along when players can play it: in one or two channels.
        with_sound = Path(args.video).suffix.lower() == ".mp4" and wet.shape[1] <= 2
        try:
            animate_walk(
                room, walk, args.video, fps=args.fps, sound=Path(args.out) if with_sound else None
            )
        except (ImportError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        seconds = max(walk.duration_s, len(wet) / rate) if with_sound else walk.duration_s
        print(
            f"saved {args.video}: {seconds:.1f} s at {args.fps} frames per second"
            + (", with the sound" if with_sound else "")
        )
    return 0


def _cmd_play(args: argparse.Namespace) -> int:
    import numpy as np

    from acoustic_spaces.audio import read_wav
    from acoustic_spaces.measurement import _sounddevice
    from acoustic_spaces.render import play

    if (args.signal is None) == (not args.live):
        print("error: give a signal to play or --live, not both", file=sys.stderr)
        return 1
    try:
        _sounddevice()
        ir, rate = read_wav(args.ir)
        dry = None if args.live else _read_dry(args.signal, rate)
        tracking = _head_tracking(args, ir, rate) if args.head_tracking else None
    except (ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    gain = 10.0 ** (args.gain / 20.0)
    if not args.absolute:
        # Binaural decoding keeps the level of the omnidirectional channel, W.
        energy = float(((ir[:, :1] if tracking else ir) ** 2).sum(axis=0).max())
        if energy > 0:
            gain /= math.sqrt(energy)
            print(
                f"response scaled by {-10.0 * math.log10(energy):+.1f} dB so the result plays "
                "about as loud as the dry signal (--absolute keeps the simulated level)",
                file=sys.stderr,
            )
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    what = "live input" if args.live else args.signal
    if tracking is None:
        processor, receiver = ir.astype(np.float32), None
        print(
            f"playing {what} through {args.ir} ({ir.shape[1]} channel(s)); Ctrl-C stops",
            file=sys.stderr,
        )
    else:
        processor, receiver = tracking
        print(
            f"playing {what} through {args.ir} (order {processor.order} Ambisonics) to the "
            f"ears, turned by OSC head-tracker messages on {receiver.host} port "
            f"{receiver.port}; Ctrl-C stops",
            file=sys.stderr,
        )
    try:
        if receiver is not None:
            receiver.start()
        result = play(
            processor,
            rate,
            dry,
            block=args.block,
            gain=gain,
            device=device,
            seconds=args.seconds,
        )
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if receiver is not None:
            receiver.stop()
    print(
        f"played {result.frames / rate:.1f} s; convolution took up to {100 * result.load:.0f} % "
        f"of each {args.block}-frame block"
    )
    if receiver is not None:
        print(f"{receiver.received} head-tracker message(s) received")
        if receiver.rejected:
            print(f"warning: {receiver.rejected} packet(s) were not OSC", file=sys.stderr)
    if result.xruns:
        print(
            f"warning: {result.xruns} block(s) dropped out; try a larger --block",
            file=sys.stderr,
        )
    return 0


def _head_tracking(args: argparse.Namespace, ir, rate: int):
    """The head-tracked binaural renderer and the OSC receiver that turns it."""
    from acoustic_spaces.headtracking import HeadOrientation, HeadTrackedBinaural, OscReceiver
    from acoustic_spaces.spatial import read_sofa

    invert = [axis.strip().lower() for axis in args.invert.split(",") if axis.strip()]
    orientation = HeadOrientation(invert)
    hrtf = read_sofa(args.hrtf) if args.hrtf else None
    processor = HeadTrackedBinaural(
        ir, rate, hrtf=hrtf, orientation=orientation, block=args.block, decoder=args.decoder
    )
    return processor, OscReceiver(orientation, args.osc_port, args.osc_host)


def _add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sweep")
    group.add_argument("--sample-rate", type=int, default=48_000, help="Hz (default 48000)")
    group.add_argument("--duration", type=float, default=6.0, help="sweep length, s (default 6)")
    group.add_argument("--f-start", type=float, default=20.0, help="Hz (default 20)")
    group.add_argument("--f-stop", type=float, default=20_000.0, help="Hz (default 20000)")
    group.add_argument("--level", type=float, default=-6.0, help="peak dBFS (default -6)")
    group.add_argument(
        "--silence", type=float, default=3.0, help="silence after each sweep, s (default 3)"
    )
    group.add_argument("--repeats", type=int, default=1, help="sweeps to average (default 1)")


def _add_analysis_arguments(
    parser: argparse.ArgumentParser, references: bool = True, stage: bool = True
) -> None:
    group = parser.add_argument_group("analysis")
    group.add_argument(
        "--filters",
        choices=["iec", "crossover"],
        default="iec",
        help="octave filters: IEC 61260 Butterworth (default) or the synthesis crossovers",
    )
    group.add_argument(
        "--no-noise", action="store_true", help="skip the noise floor detection and correction"
    )
    if stage:
        group.add_argument(
            "--stage",
            action="store_true",
            help="stage support too, for a response 1 m from the source",
        )
    if references:
        _add_reference_arguments(group)


def _add_reference_arguments(group, side: str = "") -> None:
    """--reference-energy, or --free-field and --distance: the reference for G; with
    ``side``, as --SIDE-reference-energy and so on, for one of two responses."""
    option = f"--{side}-" if side else "--"
    source = f"the {side} response's source" if side else "the source"
    ref = group.add_mutually_exclusive_group()
    ref.add_argument(
        f"{option}reference-energy",
        type=float,
        metavar="REFERENCE_ENERGY",
        help=f"energy of a broadband impulse response of {source} at 10 m in free field, "
        "for G (0.01 for responses made by acoustic-spaces)",
    )
    ref.add_argument(
        f"{option}free-field", metavar="WAV", help=f"free-field response of {source}, for G"
    )
    group.add_argument(
        f"{option}distance",
        type=float,
        metavar="DISTANCE",
        help=f"distance of the {option}free-field response, m",
    )


def build_parser() -> argparse.ArgumentParser:
    from acoustic_spaces.headtracking import AXES, DEFAULT_OSC_PORT
    from acoustic_spaces.spatial.decoder import DECODERS

    parser = argparse.ArgumentParser(prog="acoustic-spaces", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show the installed version and compiled components")
    info.set_defaults(func=_cmd_info)

    validate = sub.add_parser("validate", help="check a scene file and its geometry")
    validate.add_argument("scene", help="path to a scene YAML file")
    validate.add_argument(
        "--leaks", action="store_true", help="trace rays and report where they escape"
    )
    validate.add_argument(
        "--rays", type=int, default=LEAK_RAYS, help=f"rays for --leaks (default {LEAK_RAYS})"
    )
    validate.set_defaults(func=_cmd_validate)

    inspect = sub.add_parser(
        "inspect", help="geometry report, materials, reverberation estimates and positions"
    )
    inspect.add_argument("scene", help="path to a scene YAML file")
    inspect.set_defaults(func=_cmd_inspect)

    materials = sub.add_parser("materials", help="list or search the built-in materials")
    materials.add_argument("query", nargs="*", help="words to look for, e.g. carpet concrete")
    materials.add_argument("--kind", choices=["absorption", "scattering"], default=None)
    materials.add_argument(
        "--file", metavar="LIBRARY", help="list a CATT-Acoustic or ODEON material library instead"
    )
    materials.set_defaults(func=_cmd_materials)

    directivity = sub.add_parser(
        "directivity", help="directivity index of a SOFA or CSV directivity file, or a pattern"
    )
    directivity.add_argument(
        "source", help="a .sofa or .csv file, or a pattern such as cardioid or lambert"
    )
    directivity.add_argument(
        "--measurement", type=int, help="SOFA: the measurement to use (default: all averaged)"
    )
    directivity.add_argument("--plot", metavar="PNG", help="save polar plots (needs matplotlib)")
    directivity.set_defaults(func=_cmd_directivity)

    examples = sub.add_parser("examples", help="list the example scenes, or copy them")
    examples.add_argument("names", nargs="*", help="examples to show or copy (default: all)")
    examples.add_argument("--copy", metavar="DIR", help="copy the examples into this folder")
    examples.set_defaults(func=_cmd_examples)

    edit = sub.add_parser("edit", help="open the graphical editor (needs the gui extra)")
    edit.add_argument("scene", nargs="?", help="scene YAML file to open")
    _add_trial_argument(edit)
    edit.set_defaults(func=_cmd_edit)

    view = sub.add_parser("view", help="show the room in 3D (needs the viz extra)")
    view.add_argument("scene", help="path to a scene YAML file")
    view.add_argument("--screenshot", metavar="PNG", help="save an image instead of a window")
    view.add_argument("--paths", action="store_true", help="draw early reflection paths")
    view.add_argument("--source", help="source for --paths (default: the first)")
    view.add_argument("--listener", help="listener for --paths (default: the first)")
    view.add_argument("--max-order", type=int, help="highest reflection order for --paths")
    view.add_argument(
        "--diffraction", choices=DIFFRACTION_MODES, help="edge diffraction for --paths"
    )
    view.add_argument("--threads", type=_threads, help=THREADS_HELP)
    view.add_argument(
        "--leaks", action="store_true", help="draw the rays that escape and the open edges"
    )
    view.add_argument(
        "--rays", type=int, default=LEAK_RAYS, help=f"rays for --leaks (default {LEAK_RAYS})"
    )
    view.set_defaults(func=_cmd_view)

    animate = sub.add_parser(
        "animate", help="animate sound particles leaving a source (needs the viz extra)"
    )
    animate.add_argument("scene", help="path to a scene YAML file")
    animate.add_argument("--out", metavar="FILE", help="save a .gif or .mp4 instead of a window")
    animate.add_argument("--source", help="source to emit from (default: the first)")
    animate.add_argument("--particles", type=int, default=4000, help="particles (default 4000)")
    animate.add_argument(
        "--duration", type=float, default=100.0, help="sound to show, ms (default 100)"
    )
    animate.add_argument("--frames", type=int, default=100, help="frames (default 100)")
    animate.add_argument("--fps", type=int, default=25, help="frames per second (default 25)")
    animate.add_argument(
        "--band",
        type=int,
        default=1000,
        choices=OCTAVE_BANDS_HZ,
        help="octave band whose absorption applies, Hz (default 1000)",
    )
    animate.add_argument(
        "--color", choices=["energy", "order"], default="energy", help="colour particles by"
    )
    animate.add_argument("--seed", type=int, default=0, help="seed for diffuse reflections")
    animate.set_defaults(func=_cmd_animate)

    export = sub.add_parser(
        "export", help="write the room as glTF (.glb) for Blender, with paths or particles"
    )
    export.add_argument("scene", help="path to a scene YAML file")
    export.add_argument("out", help="glTF binary file to write (.glb)")
    export.add_argument("--paths", action="store_true", help="add early reflection paths")
    export.add_argument("--source", help="source for paths and particles (default: the first)")
    export.add_argument("--listener", help="listener for --paths (default: the first)")
    export.add_argument("--max-order", type=int, help="highest reflection order for --paths")
    export.add_argument(
        "--diffraction", choices=DIFFRACTION_MODES, help="edge diffraction for --paths"
    )
    export.add_argument("--particles", action="store_true", help="add a particle animation")
    export.add_argument(
        "--count",
        type=int,
        default=PARTICLE_COUNT,
        help=f"particles (default {PARTICLE_COUNT})",
    )
    export.add_argument(
        "--duration", type=float, default=100.0, help="sound to animate, ms (default 100)"
    )
    export.add_argument("--frames", type=int, default=100, help="keyframes (default 100)")
    export.add_argument(
        "--band",
        type=int,
        default=1000,
        choices=OCTAVE_BANDS_HZ,
        help="octave band whose absorption applies, Hz (default 1000)",
    )
    export.add_argument(
        "--slow-motion",
        type=float,
        default=SLOW_MOTION,
        help=f"seconds of animation per second of sound (default {SLOW_MOTION:g})",
    )
    export.add_argument("--seed", type=int, default=0, help="seed for diffuse reflections")
    export.add_argument("--threads", type=_threads, help=THREADS_HELP)
    export.set_defaults(func=_cmd_export)

    simulate = sub.add_parser("simulate", help="simulate impulse responses and ISO 3382 parameters")
    simulate.add_argument("scene", help="path to a scene YAML file")
    simulate.add_argument("--out", required=True, metavar="DIR", help="folder for the results")
    simulate.add_argument("--sample-rate", type=int, help="override simulation.sample_rate")
    simulate.add_argument("--length", type=float, help="override simulation.ir_length_s")
    simulate.add_argument("--seed", type=int, help="override simulation.seed")
    simulate.add_argument(
        "--bands",
        choices=("octave", "third_octave"),
        help="override simulation.bands: octave or third-octave bands (parameters stay in octaves)",
    )
    simulate.add_argument("--rays", type=int, help="override simulation.late.rays")
    simulate.add_argument("--max-order", type=int, help="override simulation.early.max_order")
    simulate.add_argument(
        "--diffraction", choices=DIFFRACTION_MODES, help="override simulation.early.diffraction"
    )
    simulate.add_argument(
        "--coherent-band",
        choices=COHERENT_BANDS,
        metavar="HZ",
        help="override simulation.early.coherent_max_band_hz: the highest octave band in "
        "which the early paths add in pressure (63, 125, 250, 500 or none)",
    )
    simulate.add_argument(
        "--wave",
        action=argparse.BooleanOptionalAction,
        help="override simulation.wave.enabled: the lowest octave bands from the wave solver",
    )
    simulate.add_argument(
        "--wave-band",
        type=int,
        choices=WAVE_BANDS_HZ,
        metavar="HZ",
        help="override simulation.wave.max_band_hz: the highest band it computes (63, 125, "
        "250 or 500; each costs about 16 times the one below)",
    )
    simulate.add_argument(
        "--wave-device",
        choices=("cpu", "gpu", "auto"),
        help="override simulation.wave.device: step the wave solver's grid on the CPU, on "
        "a GPU (with the gpu extra) or on a hardware GPU when there is one",
    )
    simulate.add_argument(
        "--late-device",
        choices=DEVICES,
        help="override simulation.late.device: trace the late rays on the CPU, on a GPU "
        "(with the gpu extra), on both sharing the rays, or on the fastest of those here",
    )
    _add_trial_argument(simulate)
    _add_performance_arguments(simulate)
    simulate.add_argument(
        "--output",
        action="append",
        metavar="FORMAT",
        help=(
            "write this format instead of the scene's outputs (repeatable): mono, binaural, "
            "ambisonics[:ORDER] or loudspeakers:LAYOUT (stereo, 5.0, 7.0, 5.0.4, 7.0.4, "
            "5.1, 7.1, 5.1.4, 7.1.4)"
        ),
    )
    simulate.add_argument("--hrtf", metavar="SOFA", help="HRTF file for binaural output")
    simulate.add_argument(
        "--normalize", action="store_true", help="scale all responses to a -1 dBFS peak"
    )
    simulate.add_argument(
        "--plot", action="store_true", help="save echogram and decay plots (needs viz extra)"
    )
    simulate.set_defaults(func=_cmd_simulate)

    coverage = sub.add_parser(
        "map", help="coverage map: parameters on a grid over the audience area"
    )
    coverage.add_argument("scene", help="path to a scene YAML file")
    coverage.add_argument("--source", help="map only this source (default: all, and together)")
    coverage.add_argument(
        "--surfaces",
        nargs="+",
        metavar="NAME",
        help="surfaces to map over (default: names with 'audience', else 'floor')",
    )
    coverage.add_argument("--height", type=float, help="listening height above them, m")
    coverage.add_argument("--spacing", type=float, help="distance between map points, m")
    coverage.add_argument("--rays", type=int, help="rays per source and band")
    coverage.add_argument(
        "--noise",
        type=float,
        nargs="+",
        metavar="DB",
        help="background noise for the STI, dB SPL: one value, or one per octave band",
    )
    coverage.add_argument("--seed", type=int, help="override simulation.seed")
    coverage.add_argument("--max-order", type=int, help="override simulation.early.max_order")
    coverage.add_argument(
        "--diffraction", choices=DIFFRACTION_MODES, help="override simulation.early.diffraction"
    )
    coverage.add_argument(
        "--coherent-band",
        choices=COHERENT_BANDS,
        metavar="HZ",
        help="override simulation.early.coherent_max_band_hz (63, 125, 250, 500 or none)",
    )
    coverage.add_argument("--length", type=float, help="override simulation.ir_length_s")
    _add_performance_arguments(coverage)
    coverage.add_argument(
        "--band",
        type=int,
        default=1000,
        choices=OCTAVE_BANDS_HZ,
        help="octave band to report and plot, Hz (default 1000)",
    )
    coverage.add_argument(
        "--parameter",
        default="g",
        choices=MAP_PARAMETER_NAMES,
        help="parameter to plot (default g)",
    )
    coverage.add_argument("--plot", metavar="PNG", help="save the map seen from above")
    coverage.add_argument("--screenshot", metavar="PNG", help="save the map in the 3D view")
    coverage.add_argument("--json", metavar="FILE", help="write every map as JSON")
    coverage.set_defaults(func=_cmd_map)

    sweep = sub.add_parser("sweep", help="write an exponential sweep excitation (WAV and JSON)")
    sweep.add_argument("out", help="WAV file to write; its settings go next to it as .json")
    _add_sweep_arguments(sweep)
    sweep.set_defaults(func=_cmd_sweep)

    deconvolve = sub.add_parser(
        "deconvolve", help="turn a recording of a sweep into impulse responses"
    )
    deconvolve.add_argument("recording", help="WAV recorded while the sweep played")
    deconvolve.add_argument(
        "--sweep", required=True, help="the sweep's .json (or the .wav next to it)"
    )
    deconvolve.add_argument("--out", required=True, help="WAV file for the impulse responses")
    deconvolve.add_argument("--length", type=float, help="response length, s (default: silence)")
    deconvolve.add_argument(
        "--loopback", type=int, help="channel of the recording wired to the output (1-based)"
    )
    deconvolve.set_defaults(func=_cmd_deconvolve)

    analyze = sub.add_parser("analyze", help="ISO 3382 parameters of an impulse response WAV")
    analyze.add_argument("ir", help="impulse response WAV")
    analyze.add_argument("--channel", type=int, help="analyse only this channel (1-based)")
    kind = analyze.add_mutually_exclusive_group()
    kind.add_argument(
        "--lateral",
        type=int,
        metavar="CHANNEL",
        help="the channel of a figure of eight with its null on the source (1-based), for "
        "LF, LFC and LJ of the omnidirectional channel (--channel, else the first other)",
    )
    kind.add_argument(
        "--ambisonic",
        action="store_true",
        help="the file is first-order AmbiX (W, Y, Z, X) facing the source: LF, LFC and LJ "
        "from Y and the T30 of X, Y and Z",
    )
    kind.add_argument(
        "--binaural", action="store_true", help="the first two channels are the ears: IACC too"
    )

    analyze.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    analyze.add_argument("--plot", metavar="PNG", help="save decay plots (needs viz extra)")
    _add_analysis_arguments(analyze)
    analyze.set_defaults(func=_cmd_analyze)

    compare = sub.add_parser(
        "compare", help="compare a simulated impulse response with a measured one, in JNDs"
    )
    compare.add_argument("simulated", help="simulated impulse response WAV")
    compare.add_argument("measured", help="measured impulse response WAV")
    compare.add_argument("--json", metavar="FILE", help="also write the comparison as JSON")
    compare.add_argument(
        "--plot", metavar="PNG", help="save both decays per octave band (needs viz extra)"
    )
    _add_analysis_arguments(compare, references=False, stage=False)
    for side in ("simulated", "measured"):
        group = compare.add_argument_group(f"{side} response")
        group.add_argument(
            f"--{side}-channel",
            type=int,
            metavar="CHANNEL",
            help="compare this channel (1-based, default 1)",
        )
        _add_reference_arguments(group, side)
    compare.set_defaults(func=_cmd_compare)

    calibrate = sub.add_parser(
        "calibrate",
        help="fit the materials' absorption to parameters measured in the room",
        description="Fit the absorption of the scene's materials, octave band by octave "
        "band, so that the simulated parameters (T30, EDT and C80 unless --parameters "
        "says otherwise) match those measured at each source and listener pair, in JNDs. "
        "Each round traces the room once and fits by re-weighting the rays traced.",
    )
    calibrate.add_argument("scene", help="scene YAML file")
    calibrate.add_argument(
        "--measured",
        nargs=3,
        action="append",
        metavar=("SOURCE", "LISTENER", "FILE"),
        help="a measured impulse response WAV (or a report of analyze --json) for the "
        "scene's SOURCE and LISTENER; repeat for each pair",
    )
    calibrate.add_argument(
        "--measured-json",
        action="append",
        metavar="FILE",
        help='measured parameters per pair: {"pairs": {"SOURCE LISTENER": {"t30": [...], '
        "...}}}, eight octave values each (null where missing)",
    )
    calibrate.add_argument(
        "--materials", nargs="+", metavar="NAME", help="the materials to fit (default: all)"
    )
    calibrate.add_argument(
        "--parameters",
        nargs="+",
        default=["t30", "edt", "c80"],
        choices=["edt", "t20", "t30", "c50", "c80", "d50", "ts", "g"],
        metavar="NAME",
        help="parameters to match: edt, t20, t30, c50, c80, d50, ts, g (default t30 edt c80)",
    )
    calibrate.add_argument(
        "--rounds", type=int, default=4, help="traces at new absorption, at most (default 4)"
    )
    calibrate.add_argument("--channel", type=int, help="channel of the WAV files (default 1)")
    calibrate.add_argument("--rays", type=int, help="override simulation.late.rays")
    calibrate.add_argument("--seed", type=int, help="override simulation.seed")
    calibrate.add_argument("--length", type=float, help="override simulation.ir_length_s")
    _add_performance_arguments(calibrate)
    calibrate.add_argument("-o", "--out", metavar="YAML", help="write the fitted scene here")
    calibrate.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    _add_analysis_arguments(calibrate, references=False, stage=False)
    calibrate.set_defaults(func=_cmd_calibrate)

    measure = sub.add_parser(
        "measure", help="play a sweep, record it and analyse it (needs the measure extra)"
    )
    measure.add_argument("--out", required=True, metavar="DIR", help="folder for the results")
    measure.add_argument("--device", help="audio device name or number (see 'devices')")
    measure.add_argument(
        "--input", type=int, nargs="+", default=[1], help="input channels (1-based, default 1)"
    )
    measure.add_argument("--output", type=int, default=1, help="output channel (default 1)")
    measure.add_argument("--loopback", type=int, help="input wired to the output, for latency")
    measure.add_argument("--length", type=float, help="response length, s (default: silence)")
    measure.add_argument("--plot", action="store_true", help="save decay plots (needs viz)")
    _add_sweep_arguments(measure)
    _add_analysis_arguments(measure)
    measure.set_defaults(func=_cmd_measure)

    convolve = sub.add_parser("convolve", help="run a dry recording through an impulse response")
    convolve.add_argument("signal", help="dry WAV (mixed down to mono, resampled to the IR rate)")
    convolve.add_argument("ir", help="impulse response WAV, any number of channels")
    convolve.add_argument("--out", required=True, help="WAV file for the result")
    convolve.add_argument("--normalize", action="store_true", help="scale to a -1 dBFS peak")
    convolve.set_defaults(func=_cmd_convolve)

    auralize = sub.add_parser(
        "auralize", help="mix every source's dry signal as heard at a simulated listener"
    )
    auralize.add_argument("results", help="folder written by simulate")
    auralize.add_argument(
        "signals", nargs="+", metavar="SOURCE=WAV", help="dry signal of each source to play"
    )
    auralize.add_argument("--listener", help="listener (default: the only one)")
    auralize.add_argument(
        "--format",
        default="mono",
        help="output as written by simulate: mono, binaural, ambix3, 5.0, ... (default mono)",
    )
    auralize.add_argument("--out", required=True, help="WAV file for the result")
    auralize.add_argument("--normalize", action="store_true", help="scale to a -1 dBFS peak")
    auralize.set_defaults(func=_cmd_auralize)

    walkthrough = sub.add_parser(
        "walkthrough",
        help="hear dry signals as a listener or a source walks through the room",
        description="Simulate responses along the walk of a listener or source (its "
        "'walk' in the scene) and play dry signals through them, crossfading from each "
        "place to the next.",
    )
    walkthrough.add_argument("scene", help="path to a scene YAML file")
    walkthrough.add_argument(
        "signals", nargs="+", metavar="SOURCE=WAV", help="dry signal of each source to play"
    )
    walkthrough.add_argument("--out", required=True, help="WAV file for the result")
    walkthrough.add_argument(
        "--walker", help="listener or source that walks (default: the only one with a walk)"
    )
    walkthrough.add_argument(
        "--listener", help="listener that hears a walking source (default: the only one)"
    )
    walkthrough.add_argument(
        "--format",
        metavar="FORMAT",
        help="mono, binaural, ambisonics[:ORDER] or loudspeakers:LAYOUT (default: the "
        "scene's first output, else mono)",
    )
    walkthrough.add_argument("--hrtf", metavar="SOFA", help="HRTF file for binaural output")
    walkthrough.add_argument(
        "--step",
        type=float,
        default=STEP_M,
        metavar="M",
        help=f"distance between the places responses are simulated at (default {STEP_M:g} m)",
    )
    walkthrough.add_argument(
        "--late-step",
        type=float,
        default=LATE_STEP_M,
        metavar="M",
        help="distance between the places the late part is traced at, the others mixing "
        f"theirs (default {LATE_STEP_M:g} m; 0: at every place)",
    )
    walkthrough.add_argument(
        "--crossfade",
        type=float,
        default=1e3 * CROSSFADE_S,
        metavar="MS",
        help=f"crossfade from one place to the next (default {1e3 * CROSSFADE_S:g} ms)",
    )
    walkthrough.add_argument("--rays", type=int, help="override simulation.late.rays")
    walkthrough.add_argument("--length", type=float, help="override simulation.ir_length_s")
    walkthrough.add_argument("--seed", type=int, help="override simulation.seed")
    walkthrough.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
    walkthrough.add_argument(
        "--video",
        metavar="FILE",
        help="also save the walk as a .gif, or an .mp4 with the sound in it (viz extra)",
    )
    walkthrough.add_argument("--fps", type=int, default=25, help="video frames per second")
    walkthrough.add_argument("--normalize", action="store_true", help="scale to a -1 dBFS peak")
    walkthrough.set_defaults(func=_cmd_walkthrough)

    play = sub.add_parser(
        "play", help="real-time convolution on the sound card (needs the measure extra)"
    )
    play.add_argument("ir", help="impulse response WAV; each channel goes to an output")
    play.add_argument("signal", nargs="?", help="dry WAV to play through it")
    play.add_argument("--live", action="store_true", help="run the first input through it live")
    play.add_argument("--seconds", type=float, help="stop live input after this long")
    play.add_argument("--block", type=int, default=256, help="frames per block (default 256)")
    play.add_argument("--device", help="audio device name or number (see 'devices')")
    play.add_argument("--gain", type=float, default=0.0, help="extra gain, dB (default 0)")
    play.add_argument(
        "--absolute",
        action="store_true",
        help="keep the response's level (default: scaled to play as loud as the dry signal)",
    )
    tracking = play.add_argument_group(
        "head tracking",
        "an Ambisonics response decoded to the ears and turned against the head, as a "
        "head tracker reports it over OSC (/ypr, /quaternion, /yaw, /pitch, /roll)",
    )
    tracking.add_argument(
        "--head-tracking", action="store_true", help="turn on head-tracked binaural playback"
    )
    tracking.add_argument(
        "--osc-port",
        type=int,
        default=DEFAULT_OSC_PORT,
        help=f"UDP port the tracker sends to (default {DEFAULT_OSC_PORT})",
    )
    tracking.add_argument(
        "--osc-host",
        default="127.0.0.1",
        help="address to listen on (default 127.0.0.1, this computer; 0.0.0.0 for the network)",
    )
    tracking.add_argument(
        "--invert",
        default="",
        metavar="AXES",
        help=f"axes the tracker counts the other way round, as {','.join(AXES)}",
    )
    tracking.add_argument("--hrtf", help="SOFA file (default: spherical head model)")
    tracking.add_argument(
        "--decoder", choices=DECODERS, default=DECODERS[0], help="binaural decoder (default magls)"
    )
    play.set_defaults(func=_cmd_play)

    devices = sub.add_parser("devices", help="list audio devices (needs the measure extra)")
    devices.set_defaults(func=_cmd_devices)
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
