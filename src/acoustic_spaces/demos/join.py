"""One video of several finished runs of a demo in turn: the ``join`` step every demo has
(:data:`~acoustic_spaces.demos.JOIN`)::

    python flight_demo.py join --runs flight_departure_binaural,flight_arrival_binaural

A run is finished when its video is there: its folder in the work folder keeps a
``run.json`` (:func:`~acoustic_spaces.demos.record_run`) that names it. Each run's video
plays after a title card that says what the run plays and how it was made, as it is, at
its own level: every run brings its loudest peak to -1 dBFS, so the parts are not
comparable in level with each other, which the cards say. The joined video is named
after what the runs share and what sets them apart
(``flight_departure+arrival_binaural.mp4``), in the work folder beside them, with the
chapters of every part.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from acoustic_spaces.demos import (
    RUN_RECORD,
    joined_name,
    listener_names,
    run_folder,
)

if TYPE_CHECKING:
    from acoustic_spaces.demos import Demo

#: How long each title card shows (s).
CARD_S = 4.0
#: The joined video's frame rate and size, and its sound's sample rate: each part is
#: brought to them.
FPS = 30
SIZE = (1920, 1080)
RATE = 48000
#: What a part with no settings of its own is called in the joined video's name, where
#: the others' settings stand.
DEFAULT_PART = "default"


def work_of(args: argparse.Namespace) -> Path:
    """The work folder of a run with ``args`` (resolved): where its runs' folders are."""
    return run_folder(args).parent


def finished_runs(demo: Demo, args: argparse.Namespace) -> list[dict[str, Any]]:
    """The finished runs of ``demo`` in the work folder of ``args`` (resolved), by name:
    each run's record (``run.json``) with ``run``, its name, and ``path``, its video,
    for those whose record names a video that is there."""
    work = work_of(args)
    out = []
    for record_path in sorted(work.glob(f"*/{RUN_RECORD}")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("demo") != demo.name:
            continue
        video = record.get("video")
        if not video or not (work / video).is_file():
            continue
        record["run"] = record_path.parent.name
        record["path"] = work / video
        out.append(record)
    return out


def _parts(record: dict) -> list[tuple[str, str]]:
    """The parts of a run's name as its record keeps them, or its name as one part."""
    parts = record.get("parts")
    if isinstance(parts, list) and all(isinstance(p, list) and len(p) == 2 for p in parts):
        return [(str(kind), str(text)) for kind, text in parts]
    return [("run", record["run"])]


def joined_run_name(records: Sequence[dict]) -> str:
    """The name of the video that joins ``records``: the parts their names share in
    front and behind, and between them what sets each apart, joined by ``+``
    (``flight_departure+arrival_binaural``); a run with nothing of its own there is
    :data:`DEFAULT_PART`."""
    names = [_parts(record) for record in records]
    front = 0
    while all(len(p) > front for p in names) and len({tuple(p[front]) for p in names}) == 1:
        front += 1
    back = 0
    while (
        all(len(p) - front > back for p in names)
        and len({tuple(p[len(p) - 1 - back]) for p in names}) == 1
    ):
        back += 1
    middles = []
    for parts in names:
        own = parts[front : len(parts) - back]
        middles.append("_".join(text for _kind, text in own) or DEFAULT_PART)
    head = [text for _kind, text in names[0][:front]]
    tail = [text for _kind, text in names[0][len(names[0]) - back :]]
    return joined_name([*head, "+".join(middles), *tail])


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


def card_lines(record: dict, k: int, n: int) -> tuple[str, str, list[str]]:
    """What the title card of the ``k``-th of ``n`` runs says: the demo's title, the
    run's name, and lines on what it plays and how it was made."""
    from acoustic_spaces.demos.pipeline import label_text

    parts = _parts(record)
    plays = [text for kind, text in parts if kind == "plays"]
    listeners = [text for kind, text in parts if kind == "listeners"]
    changed = [text for kind, text in parts if kind == "changed"]
    label = next((text for kind, text in parts if kind == "label"), None)
    lines = [f"Part {k + 1} of {n}"]
    if plays:
        lines.append("Plays: " + ", ".join(plays))
    lines.append("Listeners: " + (", ".join(listeners) if listeners else "every one"))
    lines.append("Settings: " + (", ".join(changed) if changed else "at their defaults"))
    if label is not None:
        lines.append("Mixed " + label_text(label))
    if record.get("command"):
        lines.append(f"Run as: {record['command']}")
    lines.append(
        "At the level of its own video (its loudest peak at -1 dBFS): the parts are not "
        "comparable in level with each other."
    )
    return str(record.get("title") or record["run"]), record["run"], lines


def draw_card(title: str, name: str, lines: Sequence[str]):
    """A title card (:data:`SIZE`): the demo's title, the run's name, and ``lines``."""
    from acoustic_spaces.demos import pictures as pics

    img = pics.canvas(SIZE)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)
    pics.text(draw, (pics.MARGIN, 300), title, pics.font("Bold", 64, display=True))
    pics.text(draw, (pics.MARGIN, 390), name, pics.font("Regular", 34), pics.LISTENER)
    y = 480
    for line in lines:
        shown, face = pics.fitted(draw, line, 30, SIZE[0] - 2 * pics.MARGIN)
        pics.text(draw, (pics.MARGIN, y), shown, face, pics.MUTED)
        y += 52
    return img


def _picked(runs: list[dict], wanted: list[str] | None, demo: Demo, work: Path) -> list[dict]:
    """The runs ``--runs`` names, in its order; every run when it names none."""
    if wanted is None:
        return runs
    by_name = {run["run"]: run for run in runs}
    picked = []
    for name in wanted:
        if name not in by_name:
            have = ", ".join(by_name) or "none"
            raise SystemExit(
                f"error: --runs {name}: no finished run of the {demo.name} demo in {work} "
                f"(a run is finished once its video is there); it has {have}"
            )
        picked.append(by_name[name])
    return picked


def join(demo: Demo, args: argparse.Namespace) -> Path:
    """The ``join`` step: the finished runs ``--runs`` names (every one by default), each
    after its title card, into one video in the work folder, with the chapters of every
    part beside it. Returns the video's path."""
    from acoustic_spaces.demos.pipeline import chapters_file, encode

    work = work_of(args)
    runs = _picked(finished_runs(demo, args), listener_names(getattr(args, "runs", None)), demo, work)
    if len(runs) < 2:
        have = ", ".join(run["run"] for run in runs) or "none"
        raise SystemExit(
            f"error: join needs two finished runs of the {demo.name} demo or more, and "
            f"{work} has {len(runs)} ({have}): run the demo with other settings first, "
            f"through its video"
        )
    name = joined_run_name(runs)
    out = work / f"{name}.mp4"
    cards = work / f"{name}_cards"
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
    for k, run in enumerate(runs):
        title, run_name, lines = card_lines(run, k, len(runs))
        card = cards / f"{k:02d}_{run_name}.png"
        draw_card(title, run_name, lines).convert("RGB").save(card, optimize=True)
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", f"{CARD_S:g}", "-i", str(card)]
        inputs += ["-f", "lavfi", "-t", f"{CARD_S:g}", "-i", f"anullsrc=r={RATE}:cl=stereo"]
        inputs += ["-i", str(run["path"])]
        base = 3 * k
        graph.append(f"[{base}:v]{scale}[c{k}]")
        graph.append(f"[{base + 1}:a]{sound}[s{k}]")
        graph.append(f"[{base + 2}:v]{scale}[v{k}]")
        graph.append(f"[{base + 2}:a]{sound}[a{k}]")
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
    return out


__all__ = [
    "CARD_S",
    "DEFAULT_PART",
    "FPS",
    "RATE",
    "SIZE",
    "card_lines",
    "clock",
    "draw_card",
    "finished_runs",
    "join",
    "joined_run_name",
    "shifted_chapter",
    "work_of",
]
