"""Command-line interface for acoustic_spaces."""

from __future__ import annotations

import argparse
import json
import math
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
    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.engine import simulate
    from acoustic_spaces.materials import OCTAVE_BANDS_HZ
    from acoustic_spaces.spatial import from_output

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
    out.mkdir(parents=True, exist_ok=True)
    irs = {}
    for key, response in result.responses.items():
        for fmt in formats:
            print(f"{key[0]} -> {key[1]}: rendering {fmt.label}", file=sys.stderr)
            irs[key, fmt.label] = response.spatial_impulse_response(fmt, sample_rate)
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
        written = []
        for fmt in formats:
            name = f"{stem}.wav" if fmt.label == "mono" else f"{stem}.{_file_name(fmt.label)}.wav"
            mask = fmt.channel_mask() if hasattr(fmt, "channel_mask") else None
            write_wav(out / name, irs[(source, listener), fmt.label] * gain, sample_rate, mask)
            written.append({"format": fmt.label, "wav": name, "channels": fmt.channel_names})
        params = response.parameters()
        print(
            f"\n{source} -> {listener}: direct sound at {response.direct_delay_s * 1e3:.1f} ms, "
            f"{len(response.early)} early paths"
        )
        print(params.table())
        entry = {
            "source": source,
            "listener": listener,
            "outputs": written,
            "direct_delay_s": response.direct_delay_s,
            "early_paths": len(response.early),
            "parameters": params.as_dict(),
        }
        if args.plot:
            try:
                from acoustic_spaces.viz import plot_directions, plot_response

                plot_response(response, out / f"{stem}.png")
                plot_directions(response, out / f"{stem}.directions.png")
                entry["plot"] = f"{stem}.png"
                entry["directions_plot"] = f"{stem}.directions.png"
            except ImportError as exc:
                print(f"warning: {exc}", file=sys.stderr)
                args.plot = False
        report["responses"].append(entry)
    (out / "parameters.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {len(irs)} impulse response files and parameters.json to {out}")
    if peak * gain > 1.0:
        print(
            "note: the float WAV files exceed full scale (a listener is closer than 1 m to a "
            "source); use --normalize to scale them for playback",
            file=sys.stderr,
        )
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
    recording, sample_rate = read_wav(args.recording)
    if sample_rate != sweep.sample_rate:
        print(
            f"error: the recording is at {sample_rate} Hz, the sweep at {sweep.sample_rate} Hz",
            file=sys.stderr,
        )
        return 1
    responses = deconvolve(recording, sweep, length_s=args.length)
    if args.loopback is not None:
        column = _channel_index(args.loopback, responses.shape[1], "loopback channel")
        if column is None:
            return 1
        responses, lag = remove_latency(responses, column)
        print(f"latency from the loopback channel: {lag / sample_rate * 1e3:.2f} ms")
    write_wav(args.out, responses, sample_rate)
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

        plot_impulse_response(
            ir, sample_rate, path, title=title, filters=args.filters, noise=not args.no_noise
        )
    except ImportError as exc:
        print(f"warning: {exc}", file=sys.stderr)
        return False
    return True


def _cmd_analyze(args: argparse.Namespace) -> int:
    from acoustic_spaces.audio import read_wav

    data, sample_rate = read_wav(args.ir)
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
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
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
    except (ImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    gain = 10.0 ** (args.gain / 20.0)
    if not args.absolute:
        energy = float((ir**2).sum(axis=0).max())
        if energy > 0:
            gain /= math.sqrt(energy)
            print(
                f"response scaled by {-10.0 * math.log10(energy):+.1f} dB so the result plays "
                "about as loud as the dry signal (--absolute keeps the simulated level)",
                file=sys.stderr,
            )
    device = int(args.device) if args.device and args.device.isdigit() else args.device
    what = "live input" if args.live else args.signal
    print(
        f"playing {what} through {args.ir} ({ir.shape[1]} channel(s)); Ctrl-C stops",
        file=sys.stderr,
    )
    try:
        result = play(
            ir.astype(np.float32),
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
    print(
        f"played {result.frames / rate:.1f} s; convolution took up to {100 * result.load:.0f} % "
        f"of each {args.block}-frame block"
    )
    if result.xruns:
        print(
            f"warning: {result.xruns} block(s) dropped out; try a larger --block",
            file=sys.stderr,
        )
    return 0


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
        "--output",
        action="append",
        metavar="FORMAT",
        help=(
            "write this format instead of the scene's outputs (repeatable): mono, binaural, "
            "ambisonics[:ORDER] or loudspeakers:LAYOUT (stereo, 5.0, 7.0, 5.0.4, 7.0.4)"
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
