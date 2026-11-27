"""One video of the parts of a run that plays several of something
(:meth:`~acoustic_spaces.demos.Demo.parts`): both of the airport's flights, two speeds
of a car, each source of a scene alone::

    python flight_demo.py all --vehicle departure,arrival

Each part is an ordinary run, in its own folder with its own ``run.json``
(:func:`~acoustic_spaces.demos.record_run`), whose video plays here after a title card
that says what sets it apart and how it was made, as it is, at its own level: every part
brings its loudest peak to -1 dBFS, so the parts are not comparable in level with each
other, which the cards say. Parts that differ only in what is heard at one level
(:attr:`~acoustic_spaces.demos.Option.one_level`: the movements of a symphony) are
joined at the level of the loudest of them instead, as their records say what brought
each to its peak (``peak_gain_db``). The joined video is named after the run
(``flight_each-departure+arrival_binaural.mp4``), in the work folder beside the parts', with
the chapters of every part.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acoustic_spaces.demos import RUN_RECORD, product, record_run, run_folder

if TYPE_CHECKING:
    from acoustic_spaces.demos import Demo

#: How long each title card shows (s).
CARD_S = 4.0
#: The joined video's frame rate and size, and its sound's sample rate: each part is
#: brought to them.
FPS = 30
SIZE = (1920, 1080)
RATE = 48000


def part_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The record (``run.json``) of each part of the run with ``args`` (resolved, of
    several parts), with ``run``, its name, and ``path``, its video; an error for a part
    with no video yet."""
    out = []
    for part in getattr(args, "series_parts", None) or []:
        folder = run_folder(part)
        try:
            record = json.loads((folder / RUN_RECORD).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        video = record.get("video") if isinstance(record, dict) else None
        if not video or not (folder.parent / video).is_file():
            raise SystemExit(f"error: the part {folder.name} has no video yet: run its video")
        record["run"] = folder.name
        record["path"] = folder.parent / video
        out.append(record)
    return out


def _parts(record: dict) -> list[tuple[str, str]]:
    """The parts of a run's name as its record keeps them, or its name as one part."""
    parts = record.get("parts")
    if isinstance(parts, list) and all(isinstance(p, list) and len(p) == 2 for p in parts):
        return [(str(kind), str(text)) for kind, text in parts]
    return [("run", record["run"])]


def clock(t: float) -> str:
    """A chapter's time: ``M:SS``, or ``H:MM:SS`` from an hour on."""
    seconds = int(t)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


_CHAPTER = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d{1,2})\s+(.*)$")


def shifted_chapter(line: str, offset_s: float) -> str | None:
    """A chapter of a part (``1:05 Gate``) at its time in the joined video, ``offset_s``
    later; None for a line that is no chapter."""
    found = _CHAPTER.match(line)
    if found is None:
        return None
    hours, minutes, seconds, title = found.groups()
    t = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    return f"{clock(t + offset_s)} {title}"


def one_level(demo: Demo, args: argparse.Namespace, records: Sequence[dict]) -> bool:
    """Whether the parts of the run with ``args`` (``records``, each part's) are joined at
    one level: what they differ in is heard so (:attr:`Option.one_level`), and each says
    what brought it to its peak."""
    names = {option.name: option for option in demo.options}
    varying = demo.series_order(args)
    return (
        bool(varying)
        and all(name in names and names[name].one_level for name in varying)
        and all(isinstance(record.get("peak_gain_db"), (int, float)) for record in records)
    )


def card_lines(record: dict, k: int, n: int, together: bool = False) -> tuple[str, str, list[str]]:
    """What the title card of the ``k``-th of ``n`` parts says: the demo's title, the
    part's name, and lines on what it plays and how it was made, and whether it is at
    its own level or, ``together``, at one with the others (:func:`one_level`)."""
    from acoustic_spaces.demos.pipeline import label_text

    parts = _parts(record)
    plays = [text for kind, text in parts if kind == "plays"]
    sources = [text for kind, text in parts if kind == "sources"]
    listeners = [text for kind, text in parts if kind == "listeners"]
    changed = [text for kind, text in parts if kind == "changed"]
    label = next((text for kind, text in parts if kind == "label"), None)
    lines = [f"Part {k + 1} of {n}"]
    if plays:
        lines.append("Plays: " + ", ".join(plays))
    if sources:
        lines.append("Sources: " + ", ".join(sources))
    lines.append("Listeners: " + (", ".join(listeners) if listeners else "every one"))
    lines.append("Settings: " + (", ".join(changed) if changed else "at their defaults"))
    if label is not None:
        lines.append("Mixed " + label_text(label))
    if record.get("command"):
        lines.append(f"Run as: {record['command']}")
    lines.append(
        "At one level with the other parts (the loudest peak of them at -1 dBFS): the parts "
        "are comparable in level with each other."
        if together
        else "At the level of its own video (its loudest peak at -1 dBFS): the parts are not "
        "comparable in level with each other."
    )
    return str(record.get("title") or record["run"]), record["run"], lines


def draw_card(title: str, name: str, lines: Sequence[str]):
    """A title card (:data:`SIZE`): the demo's title, the part's name, and ``lines``."""
    from acoustic_spaces.demos import pictures as pics

    return pics.title_card(SIZE, title, name, lines)


def join_parts(demo: Demo, args: argparse.Namespace) -> Path:
    """The parts of the run with ``args`` (resolved, of several parts), each after its
    title card, into one video beside them, named after the run (:func:`product`), with
    the chapters of every part; the run's folder keeps the cards and its record. Returns
    the video's path."""
    from acoustic_spaces.demos.pipeline import chapters_file, encode

    runs = part_records(args)
    folder = run_folder(args)
    out = product(args, ".mp4")
    cards = folder / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    for old in cards.glob("*.png"):
        old.unlink()
    inputs: list[str] = []
    graph: list[str] = []
    chapters: list[str] = []
    starts: list[tuple[float, str]] = []
    t = 0.0
    scale = (
        f"scale={SIZE[0]}:{SIZE[1]}:force_original_aspect_ratio=decrease,"
        f"pad={SIZE[0]}:{SIZE[1]}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p"
    )
    sound = f"aresample={RATE},aformat=sample_fmts=fltp:channel_layouts=stereo"
    together = one_level(demo, args, runs)
    # At one level: each part turned down by as much as its gain to its peak exceeds the
    # loudest part's.
    common = min(float(run["peak_gain_db"]) for run in runs) if together else 0.0
    for k, run in enumerate(runs):
        title, run_name, lines = card_lines(run, k, len(runs), together)
        card = cards / f"{k:02d}_{run_name}.png"
        draw_card(title, run_name, lines).convert("RGB").save(card, optimize=True)
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{CARD_S:g}", "-i", str(card)]
        inputs += ["-f", "lavfi", "-t", f"{CARD_S:g}", "-i", f"anullsrc=r={RATE}:cl=stereo"]
        inputs += ["-i", str(run["path"])]
        base = 3 * k
        graph.append(f"[{base}:v]{scale}[c{k}]")
        graph.append(f"[{base + 1}:a]{sound}[s{k}]")
        graph.append(f"[{base + 2}:v]{scale}[v{k}]")
        level = f",volume={common - float(run['peak_gain_db']):.6f}dB" if together else ""
        graph.append(f"[{base + 2}:a]{sound}{level}[a{k}]")
        chapters.append(f"{clock(t)} {k + 1}. {run_name}")
        starts.append((t, run_name))
        t += CARD_S
        for line in run.get("chapters") or []:
            moved = shifted_chapter(str(line), t)
            if moved is not None:
                chapters.append(moved)
        t += float(run.get("length_s") or 0.0)
    order = "".join(f"[c{k}][s{k}][v{k}][a{k}]" for k in range(len(runs)))
    graph.append(f"{order}concat=n={2 * len(runs)}:v=1:a=1[v][a]")
    print(f"wrote {len(runs)} title cards to {cards}")
    chaptered = chapters_file(out)
    chaptered.write_text("\n".join(chapters) + "\n", encoding="utf-8")
    print(f"wrote {chaptered}")

    def shown(at: float) -> str:
        current = starts[0][1]
        for start, run_name in starts:
            if at >= start:
                current = run_name
        return current

    encode(
        [*inputs, "-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]"],
        out,
        length_s=t,
        shown=shown,
    )
    print(f"wrote {out}: {', '.join(run['run'] for run in runs)} in turn")
    record_run(
        folder,
        demo=demo.name,
        title=demo.title,
        run=folder.name,
        parts=[list(part) for part in demo.run_parts(args)],
        series=[run["run"] for run in runs],
        command=demo.command("all", args, names_only=True),
        video=out.name,
        length_s=round(t, 3),
        chapters=chapters,
    )
    return out


__all__ = [
    "CARD_S",
    "FPS",
    "RATE",
    "SIZE",
    "card_lines",
    "clock",
    "draw_card",
    "join_parts",
    "one_level",
    "part_records",
    "shifted_chapter",
]
