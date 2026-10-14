"""Command-line interface for acoustic_spaces."""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from acoustic_spaces import __version__, native_available
from acoustic_spaces.gltf import PARTICLE_COUNT, SLOW_MOTION
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, builtin_database
from acoustic_spaces.room import Room

#: What the map command can plot (the keys of engine.coverage.MAP_PARAMETERS).
MAP_PARAMETER_NAMES = ("spl", "spl_a", "g", "c50", "c80", "d50", "edt", "t20", "t30", "ts", "sti")
#: Edge diffraction settings (engine.diffraction.DIFFRACTION_MODES).
DIFFRACTION_MODES = ("none", "direct", "first_order")
#: Highest octave bands the wave solver can compute (scene.model.WaveStage.max_band_hz).
WAVE_BANDS_HZ = (63, 125, 250, 500)


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
        print(_band_row(name, material.absorption, width) + f"  {areas[name]:9.2f}")
        if material.source:
            print(f"  {material.source}")
    unassigned = room.unassigned_labels()
    if unassigned:
        print(_band_row("(no material)", [], width) + f"  {sum(unassigned.values()):9.2f}")

    print("\nsources (directivity index per octave band, dB)")
    print(_band_header("", width))
    for source, directivity in zip(scene.sources, room.directivities, strict=True):
        print(_band_row(source.name, directivity.index(), width, "6.1f"))
        print(f"  {_directivity_text(directivity)}")

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
            print(f"  {kind:<9}{item.name:<{width}}({x:.2f}, {y:.2f}, {z:.2f}){where}")
    print()
    return 0 if _print_checks(room) else 1


def _directivity_text(directivity) -> str:
    text = directivity.name or "omni"
    if directivity.extrapolated:
        bands = ", ".join(str(f) for f in directivity.extrapolated)
        text += f" ({bands} Hz copied from the nearest octave)"
    return text


def _cmd_directivity(args: argparse.Namespace) -> int:
    import numpy as np

    from acoustic_spaces.directivity import PATTERNS, FirstOrder, read_directivity

    try:
        if args.source in PATTERNS:
            directivity = FirstOrder.pattern(args.source)
        else:
            directivity = read_directivity(args.source, args.measurement)
    except (ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    width = 22
    print(_directivity_text(directivity))
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


def _cmd_examples(args: argparse.Namespace) -> int:
    import textwrap

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
    return 0


def _cmd_edit(args: argparse.Namespace) -> int:
    try:
        from acoustic_spaces.gui import main as editor

        return editor([args.scene] if args.scene else [])
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


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

        show_room(room, screenshot=args.screenshot, paths=paths)
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
            "--rays": ("late.rays", args.rays),
            "--max-order": ("early.max_order", args.max_order),
            "--diffraction": ("early.diffraction", args.diffraction),
            "--wave": ("wave.enabled", args.wave),
            "--wave-band": ("wave.max_band_hz", args.wave_band),
            "--length": ("ir_length_s", args.length),
            "--sample-rate": ("sample_rate", args.sample_rate),
        },
    )
    if settings is None:
        return 1
    try:
        outputs = _outputs(room, args)
        formats = [from_output(o) for o in outputs]
    except (ValueError, ImportError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

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
    report = write_results(
        result,
        out,
        formats,
        sample_rate=settings.sample_rate,
        normalize=args.normalize,
        plots=args.plot,
        progress=lambda message: print(message, file=sys.stderr),
    )
    for (source, listener), response in result.responses.items():
        wave = ""
        if response.wave is not None:
            top = OCTAVE_BANDS_HZ[response.wave.bands - 1]
            wave = f", {OCTAVE_BANDS_HZ[0]}-{top} Hz from the wave solver"
        print(
            f"\n{source} -> {listener}: direct sound at {response.direct_delay_s * 1e3:.1f} ms, "
            f"{_early_text(response.early)}{wave}"
        )
        print(response.parameters().table())
    files = sum(len(entry["outputs"]) for entry in report["responses"])
    print(f"\nwrote {files} impulse response files and {REPORT_FILE} to {out}")
    if report["peak"] > 1.0:
        print(
            "note: the float WAV files exceed full scale (a listener is closer than 1 m to a "
            "source); use --normalize to scale them for playback",
            file=sys.stderr,
        )
    return 0


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
        },
    )
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
            threads=args.threads,
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
    sweep.save(out)
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


def _reference_energy(args: argparse.Namespace, sample_rate: int):
    """Reference for G from --reference-energy or --free-field and --distance."""
    from acoustic_spaces.audio import read_wav
    from acoustic_spaces.parameters import free_field_reference

    if args.free_field is None:
        return args.reference_energy
    if args.distance is None:
        raise ValueError("--free-field needs --distance")
    data, rate = read_wav(args.free_field)
    if rate != sample_rate:
        raise ValueError(f"{args.free_field} is at {rate} Hz, the response at {sample_rate} Hz")
    return free_field_reference(data[:, 0], rate, args.distance, filters=args.filters)


def _analysis(ir, sample_rate: int, args: argparse.Namespace, reference) -> dict:
    """Print and return the parameters of one impulse response."""
    from acoustic_spaces.parameters import band_decay, from_impulse_response

    noise = not args.no_noise
    params = from_impulse_response(ir, sample_rate, reference, filters=args.filters, noise=noise)
    decay = band_decay(ir, sample_rate, filters=args.filters, noise=noise)
    print(f"direct sound at {decay.onset_s * 1e3:.1f} ms")
    if noise:
        ranges = [f"{r:6.0f}" if r != float("inf") else f"{'-':>6}" for r in decay.decay_range_db]
        print(f"{'decay range (dB)':<18}" + "".join(ranges))
    print(params.table())
    return {
        "onset_s": decay.onset_s,
        "crosspoint_s": (decay.crosspoint / sample_rate).tolist() if noise else None,
        "decay_range_db": [None if r == float("inf") else float(r) for r in decay.decay_range_db]
        if noise
        else None,
        "parameters": params.as_dict(),
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


def _cmd_analyze(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import read_wav

    try:
        data, sample_rate = read_wav(args.ir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    channels = range(data.shape[1]) if args.channel is None else [args.channel - 1]
    if args.channel is not None and _channel_index(args.channel, data.shape[1], "channel") is None:
        return 1
    try:
        reference = _reference_energy(args, sample_rate)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = {"file": Path(args.ir).name, "bands_hz": list(OCTAVE_BANDS_HZ), "channels": []}
    for c in channels:
        print(f"\n{Path(args.ir).name}, channel {c + 1}")
        entry = {"channel": c + 1, **_analysis(data[:, c], sample_rate, args, reference)}
        if args.plot:
            plot = Path(args.plot)
            if len(channels) > 1:
                plot = plot.with_name(f"{plot.stem}_{c + 1}{plot.suffix}")
            if _plot_ir(data[:, c], sample_rate, plot, f"{Path(args.ir).name}, ch {c + 1}", args):
                print(f"saved {plot}")
        report["channels"].append(entry)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")
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


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
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
    ref = group.add_mutually_exclusive_group()
    ref.add_argument(
        "--reference-energy",
        type=float,
        help="energy of a broadband impulse response of the source at 10 m in free field, "
        "for G (0.01 for responses made by acoustic-spaces)",
    )
    ref.add_argument("--free-field", metavar="WAV", help="free-field response of the source, for G")
    group.add_argument("--distance", type=float, help="distance of the --free-field response, m")


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

    directivity = sub.add_parser(
        "directivity", help="directivity index of a SOFA or CSV directivity file, or a pattern"
    )
    directivity.add_argument("source", help="a .sofa or .csv file, or a pattern such as cardioid")
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
    view.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
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
    export.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
    export.set_defaults(func=_cmd_export)

    simulate = sub.add_parser("simulate", help="simulate impulse responses and ISO 3382 parameters")
    simulate.add_argument("scene", help="path to a scene YAML file")
    simulate.add_argument("--out", required=True, metavar="DIR", help="folder for the results")
    simulate.add_argument("--sample-rate", type=int, help="override simulation.sample_rate")
    simulate.add_argument("--length", type=float, help="override simulation.ir_length_s")
    simulate.add_argument("--seed", type=int, help="override simulation.seed")
    simulate.add_argument("--rays", type=int, help="override simulation.late.rays")
    simulate.add_argument("--max-order", type=int, help="override simulation.early.max_order")
    simulate.add_argument(
        "--diffraction", choices=DIFFRACTION_MODES, help="override simulation.early.diffraction"
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
    simulate.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
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
    coverage.add_argument("--rays", type=int, help="rays per source")
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
    coverage.add_argument("--length", type=float, help="override simulation.ir_length_s")
    coverage.add_argument("--threads", type=int, default=0, help="worker threads (0: all cores)")
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
    analyze.add_argument("--json", metavar="FILE", help="also write the results as JSON")
    analyze.add_argument("--plot", metavar="PNG", help="save decay plots (needs viz extra)")
    _add_analysis_arguments(analyze)
    analyze.set_defaults(func=_cmd_analyze)

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
