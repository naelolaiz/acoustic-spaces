"""The pipeline the demos share, from the simulation to the video.

The impulse responses ``simulate`` writes are per unit of sound power; ``auralize`` mixes
each listener from them and the feeds. :func:`render_mixes` does that for every stop
with one gain for all (the loudest peak at -1 dBFS), measures each mix (its gated level
in dB(A), its gated K-weighted loudness, when the sound arrives) and writes
``seats.json``. :func:`tour_audio` cuts the mixes into one piece on the tour's times, each
shifted earlier by its arrival so the music runs on across a cut, with equal-power
crossfades; :func:`video` writes the stills and, through ffmpeg, the MP4 with a scrolling
spectrogram over the bottom of them (the demos draw their thumbnails themselves).

With ``--hrtf``, the mixes ask for the responses simulated through that file
(``binaural-<its name>``) and refuse ones made through the spherical head, so that a mix
through a spherical head cannot pass for one through the listener's own ears.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from acoustic_spaces.demos import audio, shell_line

PARAMETERS = "parameters.json"
SEATS = "seats.json"
#: Where the files given with ``--input`` are made into feeds, beside the mixes' folder.
INPUTS = "inputs"
#: Where the video draws the spectrogram on a still: left, top, width, height.
SPECTRUM_BOX = (100, 892, 1720, 96)


@dataclass(frozen=True)
class Simulation:
    """A scene the demo simulates and the folder of its results; ``output`` is the
    format ``simulate --output`` writes, or None when the scene names its outputs."""

    scene: Path
    results: Path
    output: str | None = None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- simulation


#: The keys of a scene file whose values name files, relative to the scene's folder.
FILE_KEYS = ("file", "hrtf", "signal", "recording")


def write_scene(path: Path, data: dict, header: str, source_dir: Path) -> None:
    """``data`` as a scene file at ``path`` under the comment ``header``, the files it
    names relative to ``source_dir`` (the folder of the scene it was made from) named so
    that they are found from ``path``'s folder too: relative when they lie under it,
    else in full. A demo run elsewhere than its example (the editor's copy under the
    home folder) writes scenes that still find the example's model and data."""
    path = Path(path)
    dest = path.resolve().parent
    source_dir = Path(source_dir).resolve()

    def moved(value: str) -> str:
        name = Path(value)
        if name.is_absolute() or dest == source_dir:
            return value
        full = (source_dir / name).resolve()
        try:
            return full.relative_to(dest).as_posix()
        except ValueError:
            return full.as_posix()

    def free(key: str, value: str) -> bool:
        """Whether ``value`` is a free recording's name, not a file."""
        from acoustic_spaces.recordings import NAMES

        return key == "recording" and value in NAMES

    def fix(node):
        if isinstance(node, dict):
            return {
                k: moved(v) if k in FILE_KEYS and isinstance(v, str) and not free(k, v) else fix(v)
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [fix(v) for v in node]
        return node

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        header + yaml.safe_dump(fix(data), sort_keys=False, default_flow_style=None, width=100),
        encoding="utf-8",
    )


def hrtf_label(fmt: str, hrtf: str | Path | None) -> str:
    """The output a binaural mix through ``hrtf`` asks for: ``binaural-<its name>``, as
    ``simulate --hrtf`` writes it; other formats as they are."""
    if hrtf and fmt == "binaural":
        return f"binaural-{Path(hrtf).stem}"
    return fmt


def output_label(args: argparse.Namespace) -> str:
    """The output the mixes ask for: ``--format`` (binaural when the demo has no such
    option), or with ``--hrtf`` the binaural output simulated through that file."""
    return hrtf_label(getattr(args, "format", "binaural"), getattr(args, "hrtf", None))


def label_text(label: str | None) -> str:
    """An output (:func:`output_label`) in words, as mixes are made: ``binaural through
    a spherical head model``, ``binaural through the HRTFs of me`` (``binaural-me``, the
    SOFA file's name without its suffix), ``as mono``; ``of an output not kept`` for
    mixes that do not say."""
    if not label:
        return "of an output not kept"
    if label == "binaural":
        return "binaural through a spherical head model"
    if label.startswith("binaural-"):
        return f"binaural through the HRTFs of {label.removeprefix('binaural-')}"
    return f"as {label}"


def simulate_argv(simulation: Simulation, hrtf: str | Path | None = None) -> list[str]:
    """The arguments of ``acoustic-spaces simulate`` for a simulation."""
    argv = ["simulate", str(simulation.scene), "--out", str(simulation.results)]
    if simulation.output:
        argv += ["--output", simulation.output]
    if hrtf:
        argv += ["--hrtf", str(hrtf)]
    return argv


def simulate_command(simulation: Simulation, hrtf: str | Path | None = None) -> str:
    """The simulate command to run, as one line."""
    return "acoustic-spaces " + shell_line(simulate_argv(simulation, hrtf))


def outputs_of(entry: dict) -> list[dict]:
    """The outputs of a response in ``parameters.json``: its formats and WAV files (older
    results give one mono file)."""
    return entry.get("outputs") or [{"format": "mono", "wav": entry["wav"]}]


def labels_of(entries: list[dict]) -> set[str]:
    return {o["format"] for e in entries for o in outputs_of(e)}


def simulated(simulation: Simulation, args: argparse.Namespace) -> bool:
    """Whether the simulation's results are there, through ``--hrtf`` when given, and
    newer than the scene (a scene changed since is simulated again)."""
    path = simulation.results / PARAMETERS
    if not path.exists():
        return False
    scene = Path(simulation.scene)
    if scene.exists() and scene.stat().st_mtime > path.stat().st_mtime:
        return False
    label = output_label(args)
    fmt = getattr(args, "format", "binaural")
    return label == fmt or label in labels_of(read_json(path)["responses"])


def responses_of(simulation: Simulation, args: argparse.Namespace, first: str = "") -> list[dict]:
    """The responses of a simulation, checked for ``--hrtf``; without them, stop with the
    commands to run first: ``first`` (the step that writes the scene, when the scene is
    missing) and the simulate command."""
    hrtf = getattr(args, "hrtf", None)
    path = simulation.results / PARAMETERS
    if not path.exists():
        before = f"  {first}\n" if first and not simulation.scene.exists() else ""
        raise SystemExit(
            f"error: no simulation in {simulation.results}; run first:\n"
            f"{before}  {simulate_command(simulation, hrtf)}"
        )
    entries = read_json(path)["responses"]
    label = output_label(args)
    if label != getattr(args, "format", "binaural") and label not in labels_of(entries):
        raise SystemExit(
            f"error: {simulation.results} has no responses through {hrtf}"
            f" ({', '.join(sorted(labels_of(entries)))}); run first:\n"
            f"  {simulate_command(simulation, hrtf)}"
        )
    return entries


def response_rate(simulation: Simulation, entries: list[dict]) -> int:
    """The sample rate of the responses."""
    return audio.read(simulation.results / outputs_of(entries[0])[0]["wav"])[1]


def arrival_s(
    results: Path, entries: list[dict], listener: str, weights: dict[str, float] | None = None
) -> float:
    """When the loudest sound reaches ``listener`` (s): the peak, over 2 ms, of the energy
    of its responses added up, each weighted by how loud its source plays (``weights``,
    dB; only those sources, or every source alike). Cross-correlating the mix with the
    music would be simpler, but a song's beat repeats, and on the stands that picked a
    beat 0.25 s late."""
    total, rate = np.zeros(0), 1
    for entry in entries:
        if entry["listener"] != listener:
            continue
        if weights is not None and entry["source"] not in weights:
            continue
        h, rate = audio.read(results / outputs_of(entry)[0]["wav"])
        energy = (h**2).sum(axis=1) if h.ndim > 1 else h**2
        if weights is not None:
            energy = energy * 10.0 ** (weights[entry["source"]] / 10.0)
        total = np.pad(total, (0, max(0, len(energy) - len(total))))
        total[: len(energy)] += energy
    smooth = np.convolve(total, np.ones(max(1, int(0.002 * rate))), "same")
    return int(np.argmax(smooth)) / rate


# --------------------------------------------------------------------------- mixes


def auralize(
    results: Path,
    feeds: dict[str, Path],
    listener: str,
    label: str,
    out: Path,
    equal_power: bool = False,
) -> None:
    """``acoustic-spaces auralize``: the feeds through ``listener``'s responses, into
    ``out``; ``equal_power`` when the feeds carry their sources' levels already."""
    from acoustic_spaces.cli import main as cli

    pairs = [f"{name}={path}" for name, path in feeds.items()]
    argv = ["auralize", str(results), *pairs, "--listener", listener]
    argv += ["--format", label, "--out", str(out)]
    if equal_power:
        argv.append("--equal-power")
    out.parent.mkdir(parents=True, exist_ok=True)
    code = cli(argv)
    if code:
        raise SystemExit(code)


@dataclass
class Mix:
    """A stop's mix: a listener of a simulation hearing the feeds, kept as ``key``."""

    key: str
    simulation: Simulation
    listener: str
    feeds: dict[str, Path]
    #: Each source's gain (dB) for the arrival time, and which sources count; None: all.
    weights: dict[str, float] | None = None
    #: Kept with the mix in ``seats.json`` (its weather, its movement...).
    info: dict = field(default_factory=dict)


def input_feed(name: str, file: Path, replaced: Path, folder: Path) -> Path:
    """``file`` made into the feed of the source ``name`` in place of ``replaced``, as
    ``folder/<replaced's folder>/<name>.wav``: mixed down to mono (as the renderer mixes
    every feed), at the replaced feed's rate, brought to its gated RMS (so the source
    radiates the power the demo gives it) and cut or padded with silence to its length
    (so the times of the tour hold). Made again when ``file`` changes, or either file is
    newer than it."""
    out = folder / replaced.parent.name / f"{name}.wav"
    made_from = out.with_suffix(".json")
    record = {"file": str(Path(file).resolve()), "replaces": str(Path(replaced).resolve())}
    if (
        out.exists()
        and made_from.exists()
        and read_json(made_from) == record
        and all(out.stat().st_mtime >= p.stat().st_mtime for p in (file, replaced))
    ):
        return out
    y, rate = audio.read(replaced)
    x, r = audio.read(file)
    x = audio.resampled(audio.mono(x), r, rate)
    target, level = audio.gated_rms(y, rate), audio.gated_rms(x, rate)
    if level <= 0.0:
        raise SystemExit(f"error: --input {name}: {file} is silent")
    if target <= 0.0:
        raise SystemExit(
            f"error: --input {name}: the feed it replaces is silent, no level to match"
        )
    length, given = len(y), len(x)
    x = np.concatenate([x[:length] * (target / level), np.zeros(max(0, length - given))])
    audio.write(out, x, rate)
    write_json(made_from, record)
    fit = "cut" if given > length else "padded with silence" if given < length else "as long"
    print(
        f"{name}: plays {Path(file).name} instead of its feed, at the feed's gated RMS "
        f"({audio.db(target / level):+.1f} dB), {fit} to {length / rate:.1f} s"
    )
    return out


def with_inputs(mixes: Sequence[Mix], inputs: dict[str, Path] | None, folder: Path) -> list[Mix]:
    """The mixes with the feeds of the sources ``inputs`` names (``--input``) made from
    those files (:func:`input_feed`, into ``folder``); a name no mix feeds is an error."""
    if not inputs:
        return list(mixes)
    fed = sorted({name for mix in mixes for name in mix.feeds})
    for name in inputs:
        if name not in fed:
            raise SystemExit(
                f"error: --input {name}: no such source plays here; it has {', '.join(fed)}"
            )
    made: dict[Path, Path] = {}
    out = []
    for mix in mixes:
        feeds = dict(mix.feeds)
        for name, file in inputs.items():
            if name in feeds:
                key = Path(feeds[name])
                if key not in made:
                    made[key] = input_feed(name, Path(file), key, folder)
                feeds[name] = made[key]
        used = {name: str(file) for name, file in inputs.items() if name in mix.feeds}
        out.append(replace(mix, feeds=feeds, info={**mix.info, "inputs": used}))
    return out


def credited(credit: str, seats: dict, sources: Sequence[str]) -> str:
    """The stills' credit line for the mixes in ``seats`` (:func:`render_mixes`):
    ``credit``, the recordings', then what the sources ``--input`` gave a file to play
    instead; only those when they are all of ``sources``, as none of the recordings is
    heard then."""
    inputs: dict[str, str] = {}
    for seat in seats["seats"].values():
        inputs.update(seat.get("inputs", {}))
    if not inputs:
        return credit
    plays = ", ".join(f"{name}: {Path(file).name}" for name, file in sorted(inputs.items()))
    if set(sources) <= set(inputs):
        return f"Playing {plays}"
    return f"{credit} · instead, {plays}"


def render_mixes(
    mixes: Sequence[Mix],
    folder: Path,
    args: argparse.Namespace,
    *,
    equal_power: bool = False,
    first: str = "",
    inputs: dict[str, Path] | None = None,
    post: Callable[[Mix, np.ndarray, int], np.ndarray] | None = None,
) -> dict:
    """Every mix as ``folder/<key>.<format>.wav``, with one gain for all (the loudest
    peak at -1 dBFS), and ``folder/seats.json``: the rate, the format, the gain and
    each mix's ``lag_s`` (when its sound arrives), ``rms_db`` (gated, A-weighted),
    ``loudness_db`` (K-weighted), ``relative_db`` (against the loudest), length and the
    ``inputs`` it played. ``inputs`` (``--input``) replaces the feeds of the sources it
    names (:func:`with_inputs`). ``post`` changes each mix as it is made, before the
    gain (a house's window and rumble, :mod:`acoustic_spaces.demos.indoors`); a mix's
    ``info["post"]`` says how. Mixes that exist, newer than their feeds and responses
    and made with the same inputs and the same ``post``, are kept unless ``--again``,
    their earlier gain undone."""
    fmt, label = args.format, output_label(args)
    seats_path = folder / SEATS
    previous = read_json(seats_path) if seats_path.exists() else {}
    old_gain = float(previous.get("gain", 1.0))
    reuse = not getattr(args, "again", False)
    mixes = with_inputs(mixes, inputs, folder.parent / INPUTS)

    def path_of(mix: Mix) -> Path:
        return folder / f"{mix.key}.{fmt}.wav"

    def kept(mix: Mix) -> bool:
        """Whether the mix is there, made with the same inputs, and newer than its feeds
        and its responses."""
        out = path_of(mix)
        if not (reuse and out.exists() and mix.key in previous.get("seats", {})):
            return False
        before = previous["seats"][mix.key]
        if before.get("inputs", {}) != mix.info.get("inputs", {}):
            return False
        if before.get("post") != mix.info.get("post"):
            return False
        made = [mix.simulation.results / PARAMETERS, *mix.feeds.values()]
        return all(out.stat().st_mtime >= path.stat().st_mtime for path in made if path.exists())

    entries: dict[Path, list[dict]] = {}
    for mix in mixes:
        if mix.simulation.results not in entries:
            entries[mix.simulation.results] = responses_of(mix.simulation, args, first)
    keep = {mix.key: kept(mix) for mix in mixes}  # before any mix is written
    peaks, rate = {}, 0
    for mix in mixes:
        out = path_of(mix)
        if keep[mix.key]:
            print(f"{out}: there already")
        else:
            auralize(mix.simulation.results, mix.feeds, mix.listener, label, out, equal_power)
        x, rate = audio.read(out)
        if post is not None and not keep[mix.key]:
            x = post(mix, x, rate)
            audio.write(out, x, rate)
        peaks[mix.key] = float(np.abs(x).max()) / (old_gain if keep[mix.key] else 1.0)
    gain = 10.0 ** (-1.0 / 20.0) / max(peaks.values())
    info: dict[str, dict] = {}
    lags: dict[tuple, float] = {}
    for mix in mixes:
        out = path_of(mix)
        x = audio.read(out)[0] * (gain / old_gain if keep[mix.key] else gain)
        audio.write(out, x, rate)
        weights = None if mix.weights is None else tuple(sorted(mix.weights.items()))
        where = (mix.simulation.results, mix.listener, weights)
        if where not in lags:
            lags[where] = arrival_s(
                mix.simulation.results, entries[mix.simulation.results], mix.listener, mix.weights
            )
        info[mix.key] = {
            "listener": mix.listener,
            "lag_s": lags[where],
            "rms_db": audio.level_db(x, rate),
            "loudness_db": audio.loudness_db(x, rate),
            "length_s": len(x) / rate,
            **mix.info,
        }
    top = max(v["rms_db"] for v in info.values())
    width = max(len(key) for key in info)
    for key, v in info.items():
        v["relative_db"] = v["rms_db"] - top
        late = v["lag_s"] * 1e3
        print(f"{key:>{width}}: {v['relative_db']:+6.1f} dB(A), sound arrives {late:5.0f} ms late")
    seats = {"rate": rate, "format": fmt, "gain": gain, "seats": info}
    write_json(seats_path, seats)
    return seats


def seats_of(folder: Path, hint: str) -> dict:
    """``seats.json`` in the mixes' folder, or stop with the step to run."""
    path = folder / SEATS
    if not path.exists():
        raise SystemExit(f"error: no mixes in {folder}; run first: {hint}")
    return read_json(path)


# --------------------------------------------------------------------------- the tour


@dataclass
class TourStop:
    """A stop of the tour: the mix ``key`` from its time ``start`` to ``stop``, playing
    at ``at`` in the tour's time, turned up by ``gain_db``, crossfaded with its
    neighbours unless it starts or ends a piece (``joined_before``, ``joined_after``)."""

    key: str
    at: float
    start: float
    stop: float
    gain_db: float = 0.0
    joined_before: bool = True
    joined_after: bool = True

    @property
    def length(self) -> float:
        return self.stop - self.start

    @property
    def end(self) -> float:
        """When it ends in the tour's time."""
        return self.at + self.length


def tour_audio(
    stops: Sequence[TourStop],
    folder: Path,
    seats: dict,
    *,
    crossfade_s: float = 1.0,
    fade_out_s: float = 4.0,
    peak_db: float | None = -1.0,
    kind: str | None = None,
) -> np.ndarray:
    """The tour as stereo: each stop's mix shifted earlier by its arrival (``lag_s`` in
    ``seats``) so the music runs on across a cut, turned up as the stop is, equal-power
    crossfades of ``crossfade_s`` centred on the cuts between joined stops, a fade at
    the end and the peak at ``peak_db``. With a ``kind``, each stop's
    ``<listener>.<kind>.wav`` instead of its mix (what is felt there, ``"shaker"``),
    neither turned up nor brought to a peak, as it is."""
    rate, fmt = seats["rate"], seats["format"]
    length = round(stops[-1].end * rate)
    out = np.zeros((length, 2))
    half = int(crossfade_s * rate / 2)
    ramp = np.sin(np.linspace(0.0, np.pi / 2, 2 * half))  # equal power
    for k, stop in enumerate(stops):
        seat = seats["seats"][stop.key]
        x = audio.stereo(audio.read(folder / f"{stop.key}.{kind or fmt}.wav")[0])
        offset = round((stop.at - stop.start) * rate)  # the mix's start in the tour
        x = audio.shifted(x, round(seat.get("lag_s", 0.0) * rate) - offset, length)
        if kind is None:
            x = x * 10.0 ** (stop.gain_db / 20.0)
        a, b = round(stop.at * rate), round(stop.end * rate)
        before = stop.joined_before and k > 0
        after = stop.joined_after and k < len(stops) - 1
        w = np.zeros(length)
        w[(a - half if before else a) : (b + half if after else b)] = 1.0
        if before:
            w[a - half : a + half] = ramp
        if after:
            w[b - half : b + half] = ramp[::-1]
        out += x * w[:, None]
    fade = min(int(fade_out_s * rate), length)
    if fade:
        out[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None] ** 2
    peak = float(np.abs(out).max())
    if peak > 0 and peak_db is not None and kind is None:
        out *= 10.0 ** (peak_db / 20.0) / peak
    return out


def picked_tour(tour: Sequence[tuple], keep: Callable[[tuple], bool]) -> list[tuple]:
    """The stops of ``tour`` that ``keep`` keeps, each stop a tuple that ends in its
    start and its stop in time, in time order: a stop left out gives its time to the
    stop before it (those before the first kept one, to that one), and a stop at the
    same place as the one before it (all but its times alike) is joined to it, so that
    the tour lasts as long and cuts where it did between the stops it keeps. Empty
    when it keeps none."""
    out: list[tuple] = []
    begin = None
    for stop in tour:
        *place, start, end = stop
        if not keep(stop):
            if out:
                out[-1] = (*out[-1][:-1], end)
            elif begin is None:
                begin = start
            continue
        if out and list(out[-1][:-2]) == place:
            out[-1] = (*out[-1][:-1], end)
            continue
        out.append((*place, start if out or begin is None else begin, end))
    return out


def gains_of(seats: dict, keys: Sequence[str]) -> dict[str, float]:
    """The gain (dB) each mix of a tour plays at: up to the loudness of the loudest one,
    K-weighted (``loudness_db``), not A-weighted as the levels are: the bass, which the A
    weighting all but leaves out, is heard, and far places and the back of a stage keep
    little else. Mixes measured before ``loudness_db`` was kept fall back on ``rms_db``."""
    levels = {}
    for key in keys:
        seat = seats["seats"][key]
        levels[key] = seat.get("loudness_db", seat["rms_db"])
    top = max(levels.values())
    return {key: top - level for key, level in levels.items()}


def missing_mixes(seats: dict, keys: Sequence[str], hint: str) -> None:
    """Stop with the step to run when a stop of the tour has no mix."""
    missing = [key for key in keys if key not in seats["seats"]]
    if missing:
        raise SystemExit(f"error: no mix for {missing[0]}; run: {hint}")


# --------------------------------------------------------------------------- the video


def video(
    folder: Path,
    frames: Sequence[tuple[str, float]],
    draw: Callable[[int], object],
    audio_path: Path,
    *,
    no_mp4: bool = False,
    spectrum_box: tuple[int, int, int, int] = SPECTRUM_BOX,
    chapters: Sequence[str] | None = None,
) -> None:
    """One still per frame (``name``, seconds) drawn by ``draw(k)``, into
    ``folder/stills``, ``chapters.txt`` when given, and ``folder/tour.mp4`` from them and
    ``audio_path``: the stills with a scrolling spectrogram in ``spectrum_box`` (ffmpeg,
    unless ``no_mp4``)."""
    stills = folder / "stills"
    stills.mkdir(parents=True, exist_ok=True)
    lines = []
    for k, (name, duration) in enumerate(frames):
        path = stills / f"{k:02d}_{name}.png"
        draw(k).convert("RGB").save(path, optimize=True)
        lines += [f"file '{path.resolve().as_posix()}'", f"duration {duration:.3f}"]
    lines.append(lines[-2])  # the concat demuxer needs the last picture again
    (stills / "stills.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(frames)} stills to {stills}")
    if chapters:
        (folder / "chapters.txt").write_text("\n".join(chapters) + "\n", encoding="utf-8")
        print(f"wrote {folder / 'chapters.txt'}")
    if no_mp4:
        return
    out = folder / "tour.mp4"
    _ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(stills / "stills.txt")],
        audio_path,
        _spectrum(spectrum_box, 30, "fps=30,"),
        out,
    )
    print(f"wrote {out}")


def animation(
    folder: Path,
    duration_s: float,
    draw: Callable[[float], object],
    audio_path: Path,
    *,
    fps: int = 25,
    stills: Sequence[tuple[str, float]] = (),
    no_mp4: bool = False,
    spectrum_box: tuple[int, int, int, int] = SPECTRUM_BOX,
    chapters: Sequence[str] | None = None,
) -> None:
    """A moving picture: ``draw(t)`` at every frame (``t`` in s, ``fps`` a second) for
    ``duration_s``, piped to ffmpeg with ``audio_path`` and a scrolling spectrogram in
    ``spectrum_box``, into ``folder/tour.mp4``; and the frames at ``stills`` (name,
    ``t``) as pictures in ``folder/stills``, which is all that ``no_mp4`` writes, with
    ``chapters.txt`` when given."""
    pictures = folder / "stills"
    pictures.mkdir(parents=True, exist_ok=True)
    for k, (name, t) in enumerate(stills):
        draw(t).convert("RGB").save(pictures / f"{k:02d}_{name}.png", optimize=True)
    if stills:
        print(f"wrote {len(stills)} stills to {pictures}")
    if chapters:
        (folder / "chapters.txt").write_text("\n".join(chapters) + "\n", encoding="utf-8")
        print(f"wrote {folder / 'chapters.txt'}")
    if no_mp4:
        return
    frames = int(math.ceil(duration_s * fps))
    first = draw(0.0).convert("RGB")
    width, height = first.size
    out = folder / "tour.mp4"
    source = ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}"]
    source += ["-r", str(fps), "-i", "-"]

    def feed(stdin) -> None:
        stdin.write(first.tobytes())
        for k in range(1, frames):
            stdin.write(draw(k / fps).convert("RGB").tobytes())
            if k % (10 * fps) == 0:
                print(f"  {k / fps:.0f} of {duration_s:.0f} s", flush=True)

    _ffmpeg(source, audio_path, _spectrum(spectrum_box, fps, ""), out, feed)
    print(f"wrote {out}")


def _spectrum(box: tuple[int, int, int, int], fps: int, pictures: str) -> str:
    """The filter that draws the sound's scrolling spectrogram over the pictures."""
    left, top, width, height = box
    return (
        f"[1:a]showspectrum=s={width}x512:slide=scroll:color=magma:scale=log:fscale=log:"
        f"start=40:stop=16000:drange=80:overlap=0.5:fps={fps},scale={width}:{height},"
        "format=rgba,colorkey=0x000000:0.08:0.25[s];"
        f"[0:v]{pictures}format=rgba[b];[b][s]overlay={left}:{top}:shortest=1,format=yuv420p[v]"
    )


def _ffmpeg(source: list[str], audio_path: Path, spectrum: str, out: Path, feed=None) -> None:
    """ffmpeg: the pictures from ``source`` (or piped in by ``feed(stdin)``), the sound,
    the spectrogram over them, into an MP4 for the web."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("error: ffmpeg is needed for the video")
    command = [
        "ffmpeg", "-v", "error", "-stats", "-y", *source, "-i", str(audio_path),
        "-filter_complex", spectrum, "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-c:a", "aac", "-b:a", "384k", "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    if feed is None:
        subprocess.run(command, check=True)
        return
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        try:
            feed(process.stdin)
        finally:
            process.stdin.close()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


__all__ = [
    "INPUTS",
    "PARAMETERS",
    "SEATS",
    "SPECTRUM_BOX",
    "Mix",
    "Simulation",
    "TourStop",
    "animation",
    "arrival_s",
    "auralize",
    "credited",
    "gains_of",
    "hrtf_label",
    "label_text",
    "input_feed",
    "labels_of",
    "missing_mixes",
    "output_label",
    "outputs_of",
    "picked_tour",
    "read_json",
    "render_mixes",
    "response_rate",
    "responses_of",
    "seats_of",
    "simulate_argv",
    "simulate_command",
    "simulated",
    "tour_audio",
    "video",
    "with_inputs",
    "write_json",
    "write_scene",
]
