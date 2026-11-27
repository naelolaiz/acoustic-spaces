"""The stills of a demo of any scene (:mod:`acoustic_spaces.demos.scene_demo`): the scene
from above, drawn from its own geometry, with its sources, its listeners and the
listener of the moment, what is heard there and the tour; the frame round it is the
demos' (:mod:`acoustic_spaces.demos.pictures`). When something moves, the frames of a
moving picture (:meth:`Pictures.frame`) draw every mover where it is at that moment,
along its trajectory."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from acoustic_spaces.demos.audiences import audience_lines, audiences
from acoustic_spaces.demos.pictures import (
    FAINT,
    LINE,
    LISTENER,
    MARGIN,
    MUTED,
    SOURCE,
    TEXT,
    THUMBNAIL,
    TOUR_BOTTOM,
    TOUR_ROW,
    Plan,
    axes,
    badge,
    canvas,
    fitted_facts,
    font,
    footer,
    head,
    heading,
    place,
    rays,
    ring,
    scale_bar,
    text,
    thumbnail_badge,
    tour_column,
    tour_room,
)
from acoustic_spaces.demos.pipeline import TourStop
from acoustic_spaces.parameters import sti_rating
from acoustic_spaces.walkthrough import moment_name

#: Where the plan is drawn on a still: left, top, width, height.
PLAN_BOX = (MARGIN, 172, 860, 640)
#: Outdoors, with the view from the side under it: the most the plan takes of that box's
#: height, and the side view's from under the plan's caption to SIDE_BOTTOM.
PLAN_OVER_SIDE = 360
SIDE_BOTTOM = 832
#: Where a source passes the stop's listener, the plan and, under it, the graph of the
#: level heard over the pass, in place of the view from the side: left, top, width,
#: height.
PASS_PLAN_BOX = (MARGIN, 172, 860, 470)
GRAPH_BOX = (MARGIN + 64, 700, 796, 150)
#: How far the graph reaches under its top (dB), and its grid (dB, s).
RANGE_DB = 40.0
GRID_DB = 10.0
GRID_S = 2.0
#: The least room (pixels) between the time axis's labels: where GRID_S would crowd
#: them (a long pass, an aircraft's), they come every 5, 10, 20, 50... s instead.
LABEL_GAP_PX = 60.0
#: The colours of the audience's people and of empty seats.
PERSON = (190, 176, 150)
SEAT = (82, 92, 122)
#: The octave bands the reverberation time is averaged over (Hz).
MID_BANDS = (500, 1000)
#: The blocks a source's signal is weighed in over a stop (s): its mean square in each
#: (:func:`block_powers`).
PLAYED_BLOCK_S = 0.1
#: How far below the level of its sound power a source's signal may be over a stop, and
#: the source still play in it (dB); further, it is silent there, as a stem is where its
#: player rests (the 70 dB of the first gate of
#: :func:`~acoustic_spaces.demos.audio.gated_rms`).
SILENT_DB = 70.0


def block_powers(y: np.ndarray, rate: float) -> np.ndarray:
    """The mean square of the signal ``y`` (one channel) in each PLAYED_BLOCK_S of it, the
    last block as long as what is left."""
    y = np.asarray(y, dtype=float)
    if not len(y):
        return np.zeros(0)
    starts = np.arange(0, len(y), max(1, int(round(PLAYED_BLOCK_S * rate))))
    sizes = np.diff(np.append(starts, len(y)))
    return np.add.reduceat(y**2, starts) / sizes


def played_db(powers: np.ndarray | None, start: float, stop: float) -> float:
    """How loud a source plays from ``start`` to ``stop`` (s of its signal) against the
    level of its sound power (dB): the mean of ``powers`` (:func:`block_powers` of its
    signal, over the mean square it plays at its sound power) over the blocks in that
    span, as far as its signal goes; minus infinity where it is SILENT_DB or more below
    (or its signal ends before), and 0 where ``powers`` is None (its signal unknown: at
    its sound power throughout)."""
    if powers is None:
        return 0.0
    first = max(int(math.floor(start / PLAYED_BLOCK_S + 1e-9)), 0)
    last = max(int(math.ceil(stop / PLAYED_BLOCK_S - 1e-9)), first + 1)
    span = powers[first:last]
    mean = float(np.mean(span)) if len(span) else 0.0
    return 10.0 * math.log10(mean) if mean > 10.0 ** (-SILENT_DB / 10.0) else -math.inf


def _mid(values: Sequence, bands: Sequence[float]) -> float | None:
    """The mean of ``values`` over MID_BANDS (ISO 3382-1's mid-frequency value), None
    unless both bands have one."""
    picked = [v for v, f in zip(values, bands, strict=False) if f in MID_BANDS and v is not None]
    return sum(picked) / len(picked) if len(picked) == len(MID_BANDS) else None


def time_step(span_s: float, width_px: float) -> float:
    """How far apart (s) the time axis of a graph ``width_px`` wide over ``span_s`` labels
    its seconds: :data:`GRID_S`, or the first of 5, 10, 20, 50... s that leaves
    :data:`LABEL_GAP_PX` between labels."""
    step, k = GRID_S, 0
    while step * width_px / max(span_s, 1e-9) < LABEL_GAP_PX:
        step = GRID_S * (2.5, 5.0, 10.0)[k % 3] * 10.0 ** (k // 3)
        k += 1
    return step


def pass_levels(heard: dict) -> np.ndarray:
    """The levels a pass (``heard``, a listener's) holds at its times (dB(A)), minus
    infinity where nothing is heard (written null)."""
    return np.asarray([-math.inf if v is None else v for v in heard["levels_a_db"]], dtype=float)


def turned_up_line(gain_db: float) -> str:
    """What a still says of a stop the video turns up by ``gain_db``
    (:func:`~acoustic_spaces.demos.pipeline.gains_of`): to the loudness of the tour's
    loudest stop, or by the most it turns one up."""
    from acoustic_spaces.demos.pipeline import MOST_GAIN_DB

    if gain_db >= MOST_GAIN_DB:
        return f"Turned up {gain_db:.0f} dB in the video, the most it turns a stop up"
    return f"Turned up {gain_db:.0f} dB in the video, as loud as its loudest stop"


def counted(n: int, word: str) -> str:
    """``1 source``, ``3 sources``."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def wrapped(draw, words: str, weight: str, size: int, width: float, lines: int) -> tuple:
    """``words`` in at most ``lines`` lines no wider than ``width`` pixels, the font made
    smaller until they fit (a long word is cut with an ellipsis at the smallest size):
    the lines and the font."""
    for s in range(size, 23, -4):
        f = font(weight, s, display=True)
        out: list[str] = []
        for word in words.split():
            if out and draw.textlength(f"{out[-1]} {word}", font=f) <= width:
                out[-1] += f" {word}"
            else:
                out.append(word)
        if len(out) <= lines and all(draw.textlength(line, font=f) <= width for line in out):
            return out, f
    f = font(weight, 24, display=True)
    line = words
    while line and draw.textlength(line + "…", font=f) > width:
        line = line[:-1]
    return [line.rstrip() + "…"], f


def wrapped_line(draw, words: str, width: float, size: int = 20) -> tuple:
    """``words`` in one line no wider than ``width`` pixels: in ``size`` pixels, smaller
    down to 14 where they need it, cut with an ellipsis where even that is too wide."""
    for s in range(size, 13, -1):
        f = font("SemiBold", s)
        if draw.textlength(words, font=f) <= width:
            return words, f
    line = words
    while line and draw.textlength(line + "…", font=f) > width:
        line = line[:-1]
    return line.rstrip() + "…", f


def nice_length(metres: float) -> float:
    """A round length (1, 2, 5, 10...) of about a quarter of ``metres``, for a scale bar."""
    target = max(metres / 4.0, 1e-3)
    power = 10.0 ** math.floor(math.log10(target))
    return min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(m * power - target)) * power


#: Where a label may go beside its mark, as :func:`laid_out` tries them: a source's on
#: its right first, a listener's under its ring, then the corners.
CORNERS = ("below right", "above right", "below left", "above left")
SOURCE_SIDES = ("right", "left", "below", "above", *CORNERS)
LISTENER_SIDES = ("below", "above", "right", "left", *CORNERS)

Box = tuple[float, float, float, float]


def label_at(draw, mark: Box, words: str, f, side: str) -> tuple[tuple[float, float], str, Box]:
    """Where the label ``words`` goes on ``side`` of a mark taking the box ``mark``
    (left, top, right, bottom): the point and the anchor to write it at, and the box it
    takes."""
    x0, y0, x1, y1 = mark
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    w, h, gap = draw.textlength(words, font=f), f.size + 2, 4
    if side == "right":
        return (x1 + gap, y), "lm", (x1 + gap, y - h / 2, x1 + gap + w, y + h / 2)
    if side == "left":
        return (x0 - gap, y), "rm", (x0 - gap - w, y - h / 2, x0 - gap, y + h / 2)
    if side == "below":
        return (x, y1 + gap), "ma", (x - w / 2, y1 + gap, x + w / 2, y1 + gap + h)
    if side == "above":
        return (x, y0 - gap), "md", (x - w / 2, y0 - gap - h, x + w / 2, y0 - gap)
    vertical, horizontal = side.split()
    top = y1 + gap if vertical == "below" else y0 - gap - h
    if horizontal == "right":
        return (x1 + gap, top), "la", (x1 + gap, top, x1 + gap + w, top + h)
    return (x0 - gap, top), "ra", (x0 - gap - w, top, x0 - gap, top + h)


def head_boxes(at: tuple[float, float], forward: tuple[float, float], r: float) -> list[Box]:
    """The boxes the head of the listener of the moment takes on the plan
    (:func:`~acoustic_spaces.demos.pictures.head` at ``at``, facing ``forward``, of
    radius ``r``): the head with its ears and their letters, and the arrow ahead of it."""
    x, y = at
    out: list[Box] = [(x - 2.7 * r, y - 2.7 * r, x + 2.7 * r, y + 2.7 * r)]
    for along in (3.0, 4.4):
        ax, ay = x + forward[0] * r * along, y + forward[1] * r * along
        out.append((ax - 1.2 * r, ay - 1.2 * r, ax + 1.2 * r, ay + 1.2 * r))
    return out


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def laid_out(
    draw,
    labels: Sequence[tuple[Box, str, Sequence[str]]],
    f,
    taken: Sequence[Box] = (),
    area: Box | None = None,
    written: Sequence[Box] = (),
) -> list[tuple[tuple[float, float], str, str]]:
    """Each of ``labels`` (the box its mark takes, its words and the sides it may take,
    in the order tried), in that order, on the side where it runs into none of the words
    on the picture and what is drawn over the labels after them (``written``, and the
    labels laid out before it) and the fewest of what is ``taken`` (the marks' boxes),
    the first of those tried on a tie, inside ``area`` where it fits there; in the font
    ``f``. A label that would run into words on every side is left out: words written
    over words cannot be read, and the labels that come first (what is past the panel,
    the places a tour goes, then the sources) keep theirs. Each as the point, the anchor
    and the words."""
    words_on: list[Sequence[float]] = list(written)
    out = []
    for mark, words, sides in labels:
        best: tuple[int, int, int] | None = None
        chosen: tuple[tuple[float, float], str, Box] | None = None
        for side in sides:
            xy, anchor, box = label_at(draw, mark, words, f, side)
            inside = area is None or (
                area[0] <= box[0] and box[2] <= area[2] and area[1] <= box[1] and box[3] <= area[3]
            )
            score = (
                sum(_overlap(box, b) for b in words_on),
                int(not inside),
                sum(_overlap(box, b) for b in taken),
            )
            if best is None or score < best:
                best, chosen = score, (xy, anchor, box)
            if score == (0, 0, 0):
                break
        if chosen is None or best is None or best[0]:
            continue
        words_on.append(chosen[2])
        out.append((chosen[0], chosen[1], words))
    return out


def source_labels(
    draw, marks: Sequence[tuple[tuple[float, float], float, str, str | None]], f
) -> list[tuple[Box, str]]:
    """What labels the sources on the plan and the box of what each labels, from
    ``marks`` (each source's mark: where, its radius, its name and its group): each its
    name beside its mark, but where the names of a group's sources would run into each
    other in the font ``f`` (sources close together, as a choir's or an orchestra's
    section), the group once, by the last part of its path and how many it has, beside
    the box round all of them."""
    height = f.size + 2

    def right_of(at: tuple[float, float], r: float, words: str) -> Box:
        left = at[0] + r + 4
        return (left, at[1] - height / 2, left + draw.textlength(words, font=f), at[1] + height / 2)

    groups: dict[str, list] = {}
    for mark in marks:
        if mark[3]:
            groups.setdefault(mark[3], []).append(mark)
    crowded = []
    for group, members in groups.items():
        boxes = [right_of(at, r, name) for at, r, name, _ in members]
        if any(_overlap(a, b) for k, a in enumerate(boxes) for b in boxes[k + 1 :]):
            crowded.append(group)
    out = [
        ((at[0] - r, at[1] - r, at[0] + r, at[1] + r), name)
        for at, r, name, group in marks
        if group not in crowded
    ]
    for group in crowded:
        members = groups[group]
        box = (
            min(at[0] - r for at, r, _, _ in members),
            min(at[1] - r for at, r, _, _ in members),
            max(at[0] + r for at, r, _, _ in members),
            max(at[1] + r for at, r, _, _ in members),
        )
        out.append((box, f"{group.rsplit('/', 1)[-1]} ({len(members)})"))
    return out


def _within(xy: tuple[float, float], area: tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = area
    return left <= xy[0] <= right and top <= xy[1] <= bottom


class Pictures:
    """One still per stop of the tour (:meth:`still`) and a :meth:`thumbnail`.

    ``scene`` is the scene as data, ``outline`` the model's edges from above ((E, 2, 2),
    :func:`acoustic_spaces.room.outline_edges`), ``report`` the results' parameters.json,
    ``seats`` the mixes' entries of seats.json, ``stops`` the tour, ``playing`` the
    sources that play a signal and ``played`` the signals (each one's
    :func:`block_powers` over the mean square it plays at its sound power): a stop
    counts the sources that play in its span of them, each at the level it plays at
    there (:func:`played_db`; at its sound power where its signal is not given), as
    "Dona nobis pacem" has the organ alone, the choir alone, then both with the organ
    quiet; ``credit`` and ``tool`` the footer's lines. When
    something moves, ``paths`` gives each mover's knots (("sources" or "listeners",
    name): :class:`~acoustic_spaces.walkthrough.Knots`) and ``movement`` the moving mixes'
    moments and the level of each source and the STI of each talker's speech at each
    listener at each (seats.json's ``movement``), which :meth:`frame` draws and says at
    any moment of the tour. A talker is a source with a speech spectrum: a still says
    the STI of the one that plays loudest there.
    ``indoors``, the mixes heard in the houses listeners stand in front of, or
    outdoors (:meth:`~acoustic_spaces.demos.indoors.StillHouses.pictures`, with
    ``report``'s levels those where they are heard): whether the sources are heard
    through a house's window (``airborne``), what to say of each listener's house
    (``about``), the rumble of its floors and walls (``rumble_a_db``), and how hard the
    ground or a house's floor shakes at each listener (``vibration_vdb``).
    ``room``, the scene's :class:`~acoustic_spaces.room.Room`: outdoors, the stills
    draw its ground, its weather and the view from the side
    (:mod:`~acoustic_spaces.demos.outdoor_pictures`), but those named in ``hide``
    (:data:`~acoustic_spaces.demos.scene_demo.HIDDEN`). Indoors they give the room's
    reverberation time and, where something other than speech plays, its clarity C80,
    by ISO 3382-1; outdoors, where those say nothing of a room, neither."""

    def __init__(
        self,
        scene: dict,
        outline: np.ndarray,
        report: dict,
        seats: dict,
        stops: Sequence[TourStop],
        *,
        playing: Sequence[str],
        credit: str = "",
        tool: Sequence[str] | str = (),
        fmt: str = "binaural",
        paths: Mapping[tuple[str, str], object] | None = None,
        movement: dict | None = None,
        indoors: dict | None = None,
        variants: Mapping[str, tuple[str, dict]] | None = None,
        room=None,
        hide: Sequence[str] = (),
        sections: Sequence[tuple[float, str]] = (),
        played: Mapping[str, np.ndarray] | None = None,
        frame: tuple[float, float, float, float] | None = None,
    ):
        self.scene, self.report, self.seats, self.stops = scene, report, seats, list(stops)
        #: The parts of the music the recording has in time (when each starts, s, and
        #: what plays in it): a stop's still says the one it starts in.
        self.sections = sorted(sections)
        self.hide = set(hide)
        self.outdoors = None
        #: The view from the side drawn last: what it showed, its caption and its pixels.
        self._side = None
        if room is not None and room.outdoor is not None:
            from acoustic_spaces.demos.outdoor_pictures import Outdoors

            self.outdoors = Outdoors(room)
        #: Where the scene's audience stands or sits, and seats no one sits in.
        self.audiences = audiences(room) if room is not None and "audience" not in self.hide else []
        #: The audience drawn once, see-through, by the plan it was drawn on.
        self._crowd: tuple | None = None
        #: Whether the scene is a room (no ground), where ISO 3382-1's parameters hold.
        self.indoor = "ground" not in (scene.get("geometry") or {})
        # The mixes of a listener in still air (audience.still): its name, its results.
        self.where = {key: listener for key, (listener, _) in (variants or {}).items()}
        #: how many listeners the stops are at (a listener heard twice, as in the
        #: scene and in still air, is one place)
        self.places = len({self.where.get(s.key, s.key) for s in self.stops})
        self.paths = dict(paths or {})
        self.moments = np.asarray((movement or {}).get("times", [0.0]), dtype=float)
        self.along = (movement or {}).get("levels_a_db", {})
        self.sti_along = (movement or {}).get("sti", {})
        #: Where sources move past a listener that stands, the pass heard there, by
        #: listener (:func:`~acoustic_spaces.demos.scene_demo.passes_of`).
        self.passes = (movement or {}).get("passes", {}) or {}
        self.span = max((v["heard_s"][-1] for v in self.passes.values()), default=0.0) + 0.5
        self.credit, self.tool, self.fmt = credit, tool, fmt
        self.indoors = dict(indoors or {})
        self.sources = [s for s in scene["sources"] if s.get("enabled", True) is not False]
        self.playing = [s for s in self.sources if s["name"] in set(playing)]
        #: How loud each source that plays does in each stop, against its sound power
        #: (dB; minus infinity where it is silent there).
        self.played = [
            {
                s["name"]: played_db(
                    None if played is None else played.get(s["name"]), stop.start, stop.stop
                )
                for s in self.playing
            }
            for stop in self.stops
        ]
        self.listeners = {item["name"]: item for item in scene["listeners"]}
        self.outline = np.asarray(outline, dtype=float).reshape(-1, 2, 2)
        # A source whose sound does not reach the listener (None) is left out. A mix in
        # still air takes its listener's numbers in its own results.
        self.levels, self.sti, self.t30, self.c80 = {}, {}, {}, {}
        own = {e["listener"]: e["listener"] for e in report["responses"]}
        for results, named in [
            (report, own),
            *((given, {listener: key}) for key, (listener, given) in (variants or {}).items()),
        ]:
            bands = results.get("bands_hz", [63, 125, 250, 500, 1000, 2000, 4000, 8000])
            for e in results["responses"]:
                if e["listener"] not in named:
                    continue
                at = (e["source"], named[e["listener"]])
                if e.get("spl_a_db") is not None:
                    self.levels[at] = e["spl_a_db"]
                self.sti[at] = (e.get("speech") or {}).get("sti")
                self.t30[at] = _mid(e.get("parameters", {}).get("t30", []), bands)
                self.c80[at] = _mid(e.get("parameters", {}).get("c80", []), bands)
        points = [self.outline.reshape(-1, 2)] if len(self.outline) else []
        points.append(np.array([s["position"][:2] for s in self.sources], dtype=float))
        points.append(np.array([lis["position"][:2] for lis in self.listeners.values()]))
        points += [path.positions[:, :2] for path in self.paths.values()]
        xy = np.concatenate([p for p in points if len(p)])
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        #: Whether what the plan shows is cut to its panel, the scene reaching past the
        #: part of it ``frame`` gives (what moves drawn at the panel's edge while it is
        #: past it: :meth:`draw_beyond`).
        self.framed = frame is not None
        if frame is not None:
            x0, y0, x1, y1 = (float(v) for v in frame)
        pad = max(0.05 * max(x1 - x0, y1 - y0), 1.0)
        self.box = (float(x0 - pad), float(y0 - pad), float(x1 + pad), float(y1 + pad))

    # -- the scene from above ---------------------------------------------------------

    #: Where a source's label may go beside its mark, in the order tried
    #: (:func:`laid_out`).
    source_sides: tuple[str, ...] = SOURCE_SIDES

    def plan(self, left: float, top: float, width: float, height: float) -> Plan:
        """The plan that fits the scene into the box, centred in it."""
        x0, y0, x1, y1 = self.box
        scale = min(width / (x1 - x0), height / (y1 - y0))
        plan = Plan(left, top, scale, self.box)
        w, h = plan.size
        out = Plan(left + (width - w) / 2, top + (height - h) / 2, scale, self.box)
        out.area = (left, top, left + width, top + height)  # where its labels may go
        return out

    def marker(self, plan: Plan) -> float:
        """A head's radius on the picture: a head's width in the scene's scale, kept
        between what is seen and what hides the plan."""
        return float(min(16.0, max(7.0, 0.25 * plan.s)))

    # -- where things are ----------------------------------------------------------

    def placed(self, kind: str, item: dict, t: float | None) -> dict:
        """``item`` (a source or listener as scene data) where its trajectory has it at ``t`` (s
        from the start), facing as it does then; as it is when it does not move or
        ``t`` is None."""
        path = self.paths.get((kind, item["name"])) if t is not None else None
        if path is None:
            return item
        position, angles = path.at(t)
        yaw, pitch, roll = (float(a) for a in angles)
        return {
            **item,
            "position": [float(v) for v in position],
            "orientation": {"yaw": yaw, "pitch": pitch, "roll": roll},
        }

    def moment(self, t: float) -> int:
        """The moment of the moving mixes that ``t`` falls in: the last one at or before."""
        return int(max(np.searchsorted(self.moments, t + 1e-9, side="right") - 1, 0))

    def playing_in(self, k: int | None) -> list[dict]:
        """The sources that play in stop ``k`` (:func:`played_db`): every one that plays
        a signal, where ``k`` is None."""
        if k is None:
            return list(self.playing)
        return [s for s in self.playing if math.isfinite(self.played[k][s["name"]])]

    def levels_of(self, listener: str, t: float | None, k: int | None = None) -> dict[str, float]:
        """The A-weighted level of each playing source at ``listener`` (dB), as simulated:
        at ``t`` when something moves, else as the still results give it; in stop ``k``,
        of each that plays in it at the level it plays at there (:func:`played_db`). A
        source whose sound does not reach the listener (None) is left out, and one silent
        in the stop."""
        if t is not None and listener in self.along:
            moment = self.moment(t)
            values = {name: v[moment] for name, v in self.along[listener].items()}
            values = {name: v for name, v in values.items() if v is not None}
        else:
            values = {
                s["name"]: self.levels[(s["name"], listener)]
                for s in self.playing
                if (s["name"], listener) in self.levels
            }
        if k is None:
            return values
        played = self.played[k]
        return {
            name: v + played.get(name, 0.0)
            for name, v in values.items()
            if math.isfinite(played.get(name, 0.0))
        }

    def draw_paths(self, draw, plan: Plan) -> None:
        """Each trajectory as a dotted line, every quarter of a metre or so along it."""
        for path in self.paths.values():
            count = max(2, int(math.ceil(path.length_m / 0.25)) + 1)
            positions, _ = path.at(np.linspace(0.0, path.duration_s, count))
            points = [plan(*p[:2]) for p in positions]
            for k in range(0, len(points) - 1, 2):
                draw.line([points[k], points[k + 1]], fill=MUTED, width=2)

    def draw_scene(
        self,
        img: Image.Image,
        plan: Plan,
        current: str | None,
        t: float | None = None,
        k: int | None = None,
    ) -> None:
        """The outline, the sources with their names, the listeners as rings and the
        listener of the moment as a head, with a ray from each playing source (those
        that play in stop ``k``, in colour, the others grey); with ``t``, the
        trajectories dotted and every mover where it is then. ``current`` is the mix of
        the moment: a listener's, or its in still air."""
        key, current = current, self.where.get(current, current)
        sources = [self.placed("sources", s, t) for s in self.sources]
        listeners = {name: self.placed("listeners", lis, t) for name, lis in self.listeners.items()}
        target, area = img, self.clip_of(plan)
        if self.framed:  # drawn apart, then cut to the panel
            img = Image.new("RGBA", target.size, (0, 0, 0, 0))
        if self.outdoors is not None and "ground" not in self.hide:
            self.outdoors.ground(img, plan)
        draw = ImageDraw.Draw(img)
        for (xa, ya), (xb, yb) in self.outline:
            draw.line([plan(xa, ya), plan(xb, yb)], fill=LINE, width=2)
        if self.audiences:
            self.draw_audience(img, plan)
            draw = ImageDraw.Draw(img)
        # What the wind's words take, which the labels keep clear of.
        worded: list[Box] = []
        if self.outdoors is not None and "wind" not in self.hide:
            clear = [s["position"][:2] for s in sources]
            clear += [lis["position"][:2] for lis in listeners.values()]
            worded = self.outdoors.wind(img, plan, key in self.where, clear)
            draw = ImageDraw.Draw(img)
        if t is not None:
            self.draw_paths(draw, plan)
        r = self.marker(plan)
        label = font("Regular", 16)
        playing = {s["name"] for s in self.playing_in(k)}
        if current is not None and current in listeners:
            rays(
                img,
                plan,
                self.levels_of(key, t, k),
                [s for s in sources if s["name"] in playing],
                plan(*listeners[current]["position"]),
                lambda _n: SOURCE,
            )
            draw = ImageDraw.Draw(img)
        marks, beyond = [], []
        for source, given in zip(sources, self.sources, strict=True):
            x, y = plan(*source["position"])
            colour = SOURCE if given["name"] in playing else MUTED
            if self.framed and not _within((x, y), area):
                beyond.append((source, colour))
                continue
            rs = max(4.0, r * 0.55)
            draw.ellipse([x - rs, y - rs, x + rs, y + rs], fill=colour, outline=colour)
            marks.append(((x, y), rs, source["name"], given.get("group")))
        # The listeners' names first, the places the tour goes, then the sources'.
        labels: list[tuple[Box, str, Sequence[str]]] = []
        taken: list[Box] = [(x - rs, y - rs, x + rs, y + rs) for (x, y), rs, _, _ in marks]
        # The head of the listener of the moment is drawn over the labels: none goes there.
        covered = list(worded)
        for name, lis in listeners.items():
            if lis.get("enabled", True) is False:
                continue
            x, y = plan(*lis["position"])
            if name == current:
                covered += head_boxes((x, y), axes(plan, lis)[0], r)
                continue
            taken.append((x - r * 1.2, y - r * 1.2, x + r * 1.2, y + r * 1.2))
            forward, _left = axes(plan, lis)
            ring(draw, (x, y), forward, r * 0.8, r * 1.2, r * 0.6)
            labels.append(((x - r, y - r, x + r, y + r), name, LISTENER_SIDES))
        for box, words in source_labels(draw, marks, label):
            labels.append((box, words, self.source_sides))
        within = plan.area
        if self.framed:  # cut to the panel; the labels on it, those of what is past it too
            box = tuple(int(round(v)) for v in area)
            target.alpha_composite(img.crop(box), dest=box[:2])
            img, within = target, area
            draw = ImageDraw.Draw(img)
            here = listeners.get(current) if current is not None else None
            # What is past the panel first: its edge says no more of where it is.
            edges = self.draw_beyond(draw, plan, area, beyond, here)
            taken += [mark for mark, _, _ in edges]
            labels[:0] = edges
        for xy, anchor, words in laid_out(draw, labels, label, taken, within, covered):
            text(draw, xy, words, label, MUTED, anchor, halo=2)
        if current is not None and current in listeners:
            lis = listeners[current]
            forward, left = axes(plan, lis)
            head(img, plan(*lis["position"]), forward, left, r)
        draw = ImageDraw.Draw(img)
        metres = nice_length(self.box[2] - self.box[0])
        unit = f"{metres:g} m"
        scale_bar(draw, plan.left + 12, plan.top + plan.size[1] - 14, metres * plan.s, unit)

    def draw_audience(self, img: Image.Image, plan: Plan) -> None:
        """A dot for every few people where the audience stands or sits, closer where
        more are on a square metre, and dashes along the rows of seats no one sits in;
        see-through, so the lines to the listener and the marks stay in sight."""
        key = (plan.left, plan.top, plan.s, img.size)
        if self._crowd is None or self._crowd[0] != key:
            layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(layer)
            r = min(max(0.32 * plan.s, 1.0), 2.6)
            for k, area in enumerate(self.audiences):
                for x, y in area.dots(seed=k):
                    px, py = plan(x, y)
                    if area.kind == "seats":
                        w = 0.6 * plan.s
                        ends = (
                            [(px - w, py), (px + w, py)]
                            if area.along_x
                            else [
                                (px, py - w),
                                (px, py + w),
                            ]
                        )
                        draw.line(ends, fill=SEAT + (255,), width=2)
                    else:
                        alpha = 170 if area.kind == "standing" else 120
                        draw.ellipse([px - r, py - r, px + r, py + r], fill=PERSON + (alpha,))
            self._crowd = (key, layer)
        img.alpha_composite(self._crowd[1])

    # -- words ---------------------------------------------------------------------------

    def level_at(self, listener: str, t: float | None = None, k: int | None = None) -> float | None:
        """The A-weighted level of every playing source together (dB), as simulated, at
        ``t`` when something moves; in stop ``k``, of those that play in it, each at the
        level it plays at there (:meth:`levels_of`: over the stop, the equivalent level,
        where nothing moves). A source whose sound does not reach the listener (None)
        adds nothing, and None when nothing of them is heard. In a house heard with its
        rumble (``indoors``), the rumble too: steady where nothing moves, in the pass
        where something does; with the rumble alone, the rumble alone."""
        values: list[float | None] = list(self.levels_of(listener, t, k).values())
        passing = t is not None and listener in self.passes
        if passing:  # the pass's, as it is heard then
            moving = {name for kind, name in self.paths if kind == "sources"}
            others = self.levels_of(listener, t, k)
            values = [v for name, v in others.items() if name not in moving]
            values.append(self.heard_at(listener, t))
        rumble = self.indoors.get("rumble_a_db", {}).get(listener)
        if listener in self.indoors.get("about", {}) and not self.indoors.get("airborne", True):
            # In the house, the rumble alone: the pass's own, where something moves.
            values = [self.heard_at(listener, t)] if passing else []
        energy = sum(10.0 ** (v / 10.0) for v in values if v is not None)
        if rumble is not None and not passing:  # a pass holds its rumble already
            energy += 10.0 ** (rumble / 10.0)
        return 10.0 * math.log10(energy) if energy > 0.0 else None

    def speech_at(
        self, listener: str, t: float | None = None, k: int | None = None
    ) -> tuple[str, float] | None:
        """The playing talker (a source with a speech spectrum) loudest at ``listener``,
        at ``t`` when something moves, of those that play in stop ``k``, and the STI of
        its speech there, as simulated; None when no talker that plays is heard there,
        or its STI is not defined."""
        levels = self.levels_of(listener, t, k)
        talkers = [
            s["name"]
            for s in self.playing_in(k)
            if s.get("spectrum", "flat") != "flat" and s["name"] in levels
        ]
        if not talkers:
            return None
        name = max(talkers, key=lambda n: levels[n])
        if t is not None and listener in self.along:
            values = self.sti_along.get(listener, {}).get(name)
            sti = None if values is None else values[self.moment(t)]
        else:
            sti = self.sti.get((name, listener))
        return None if sti is None else (name, float(sti))

    def t30_at(self, listener: str, k: int | None = None) -> float | None:
        """The reverberation time T30 (s, ISO 3382-1) at ``listener`` of the sources that
        play in stop ``k``, their mean; None where none has one."""
        values = [self.t30.get((s["name"], listener)) for s in self.playing_in(k)]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def c80_at(self, listener: str, k: int | None = None) -> float | None:
        """The clarity C80 (dB, ISO 3382-1) at ``listener`` of the sources that play in
        stop ``k`` and are not talkers, their mean; None where only speech plays (whose
        clarity the STI gives) or none has one."""
        values = [
            self.c80.get((s["name"], listener))
            for s in self.playing_in(k)
            if s.get("spectrum", "flat") == "flat"
        ]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def nearest(
        self, listener: str, t: float | None = None, k: int | None = None
    ) -> tuple[str, float] | None:
        """The source nearest the listener of those that play in stop ``k``, and how far
        it is (m), at ``t``."""
        at = self.placed("listeners", self.listeners[self.where.get(listener, listener)], t)
        at = at["position"]
        playing = self.playing_in(k)
        if not playing:
            return None
        placed = [self.placed("sources", s, t) for s in playing]
        name, source = min(
            ((s["name"], s) for s in placed), key=lambda p: math.dist(p[1]["position"], at)
        )
        return name, math.dist(source["position"], at)

    def lines(self, k: int, t: float | None = None) -> list:
        """What a still of stop ``k`` says, at ``t`` of the tour when something moves."""
        stop = self.stops[k]
        seat = self.seats.get(stop.key, {})
        lines: list = []
        section = self.section_at(stop.start)
        if section is not None:
            lines.append(section)
        level = self.level_at(stop.key, t, k)
        heard = self.passes.get(stop.key)
        if level is not None:
            lines.append(f"{level:.0f} dB(A) here{' now' if heard else ''}, as simulated")
        if heard is not None:
            who = self.passing_name()
            lines += [
                f"{who}: loudest {heard['max_a_db']:.1f} dB(A) (L_Amax), {heard['max_s']:.1f} s in",
                f"{who}: sound exposure level {heard['sel_a_db']:.1f} dB(A) (SEL)",
            ]
            if "max_as_db" in heard:
                lines.append(
                    f"As Doc 29 measures it: L_ASmax {heard['max_as_db']:.1f}, SEL "
                    f"{heard['sel_10db_down_a_db']:.1f} dB(A)"
                )
            lines.append(f"{who}: closest {heard['closest_m']:.1f} m away")
            if "pitch_semitones" in heard:
                lines.append(
                    f"{who}: its direct sound falls {heard['pitch_semitones']:.1f} semitones "
                    "(Doppler)"
                )
            under = self.under_the_rest(stop.key)
            if under is not None:
                side = "under" if under >= 0.0 else "over"
                lines.append(
                    f"{who} at its loudest: {abs(under):.0f} dB {side} the other sources here"
                )
        speech = (
            None if stop.key in self.indoors.get("about", {}) else self.speech_at(stop.key, t, k)
        )
        if speech is not None:
            name, sti = speech
            said = f"Speech of {name.replace('_', ' ')}: STI {sti:.2f}, {sti_rating(sti)}"
            lines.append(said)
            if len(self.playing_in(k)) > 1:
                lines.append(("Of that speech alone, the others not counted as noise", MUTED))
        # The moving mixes keep when the sound arrives apart, where nothing moves.
        lag = seat["arrival_s"] if "arrival_s" in seat else seat.get("lag_s")
        if lag is not None:
            lines.append(f"The sound arrives {lag * 1e3:.0f} ms after it leaves")
        t30 = self.t30_at(stop.key, k) if self.indoor else None
        where = " where the trajectories start" if seat.get("moves") else ""
        if t30 is not None:
            lines.append(f"Reverberation time T30 {t30:.1f} s{where} (mean of 500 Hz and 1 kHz)")
        c80 = self.c80_at(stop.key, k) if self.indoor else None
        if c80 is not None:
            lines.append(f"Clarity C80 {c80:+.1f} dB{where} (mean of 500 Hz and 1 kHz)")
        near = None if heard is not None else self.nearest(stop.key, t, k)
        if near is not None:
            lines.append(f"Nearest playing: {near[0].replace('_', ' ')}, {near[1]:.1f} m away")
        shaking = self.indoors.get("vibration_vdb", {}).get(stop.key)
        about = self.indoors.get("about", {}).get(stop.key)
        if shaking is not None:
            from acoustic_spaces.demos.indoors import shaking_line

            rumble = self.indoors.get("rumble_a_db", {}).get(stop.key)
            lines.append(shaking_line(shaking, about is not None, rumble))
        if stop.gain_db > 0.5:
            lines.append(turned_up_line(stop.gain_db))
        if stop.key in self.where:
            lines.append(("In still air: the scene without its wind and weather", MUTED))
            lines.append(("Turned up as much as in the scene's weather", MUTED))
        elif self.where:
            lines.append(("In the scene's own wind and weather", MUTED))
        if about is not None:
            lines += [(line, MUTED) for line in about.split("\n")]
        if heard is not None and "about" in heard:
            lines += [(line, MUTED) for line in heard["about"].split("\n")]
        if t is not None and self.paths:
            walking = ", ".join(name.replace("_", " ") for _kind, name in self.paths)
            lines.append((f"Moving along their trajectories: {walking}", MUTED))
        lines += [(line, MUTED) for line in audience_lines(self.audiences)]
        if self.fmt == "binaural":
            lines.append(("L and R mark your ears", MUTED))
        return lines

    def section_at(self, t: float) -> str | None:
        """What plays in the part of the music a stop starting at ``t`` (s, in the
        recording) is in, a second into it so that a stop starting just before a part
        is in that part; None for a recording without parts."""
        parts = [what for start, what in self.sections if start <= t + 1.0]
        return parts[-1] if parts else None

    def title(self, listener: str, k: int | None = None) -> tuple[str, str]:
        """The listener's name, in still air for its mix in still air, and where its stop
        comes in the tour: stop ``k`` (from 0), else the listener's first (a listener
        the tour visits twice has two stops)."""
        names = [stop.key for stop in self.stops]
        if k is not None:
            k += 1
        else:
            k = names.index(listener) + 1 if listener in names else 0
        name = self.where.get(listener, listener).replace("_", " ")
        if listener in self.where:
            name = f"{name}, in still air"
        return name, f"Stop {k} of {len(names)}"

    def chapters(self) -> list[str]:
        """The video's chapters: when each stop starts (m:ss), what plays then of a
        recording in parts, and the stop's listener, as the tour on the stills names it."""
        from acoustic_spaces.demos.audio import mmss

        out = []
        for stop in self.stops:
            title = self.title(stop.key)[0]
            section = self.section_at(stop.start)
            out.append(f"{mmss(stop.at)} " + (f"{section} · {title}" if section else title))
        return out

    # -- the pictures ------------------------------------------------------------------

    def clip_of(self, plan: Plan) -> tuple[float, float, float, float]:
        """Where on the picture ``plan`` is drawn (left, top, right, bottom): its panel,
        or the scene's box on it where it has none."""
        if plan.area is not None:
            return plan.area
        w, h = plan.size
        return (plan.left, plan.top, plan.left + w, plan.top + h)

    def draw_beyond(
        self, draw, plan: Plan, area: Box, beyond, listener: dict | None
    ) -> list[tuple[Box, str, Sequence[str]]]:
        """Each source of ``beyond`` (scene data where it is then, and its colour),
        past the panel ``area``, as an arrowhead on its edge pointing to it; its label
        for :func:`laid_out`: the box the arrowhead takes, its name and how far it is
        from ``listener`` (scene data), when one is heard, and the sides it may take,
        those towards the panel first."""
        left, top, right, bottom = area
        cx, cy = (left + right) / 2, (top + bottom) / 2
        inset, size = 14.0, 11.0
        out = []
        for source, colour in beyond:
            x, y = plan(*source["position"])
            dx, dy = x - cx, y - cy
            # Where the line from the panel's middle to it leaves the panel, inset.
            reach = min(
                ((right - inset - cx) / dx if dx > 0 else (left + inset - cx) / dx)
                if abs(dx) > 1e-9
                else math.inf,
                ((bottom - inset - cy) / dy if dy > 0 else (top + inset - cy) / dy)
                if abs(dy) > 1e-9
                else math.inf,
            )
            ex, ey = cx + reach * dx, cy + reach * dy
            ux, uy = dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)
            back = (ex - ux * size, ey - uy * size)
            wing = (-uy * size * 0.6, ux * size * 0.6)
            draw.polygon(
                [
                    (ex + ux * size * 0.4, ey + uy * size * 0.4),
                    (back[0] + wing[0], back[1] + wing[1]),
                    (back[0] - wing[0], back[1] - wing[1]),
                ],
                fill=colour,
            )
            words = source["name"]
            if listener is not None:
                away = math.dist(source["position"], listener["position"])
                words = f"{words}, {away:,.0f} m away"
            inward = ["left" if ux > 0.5 else "right" if ux < -0.5 else ""]
            inward += ["above" if uy > 0.5 else "below" if uy < -0.5 else ""]
            sides = [side for side in inward if side]
            sides += [side for side in SOURCE_SIDES if side not in sides]
            out.append(((ex - size, ey - size, ex + size, ey + size), words, sides))
        return out

    def still(self, k: int, t: float | None = None) -> Image.Image:
        """The still of stop ``k``; at ``t`` of the tour, with every mover where it is
        then."""
        stop = self.stops[k]
        img = canvas()
        draw = ImageDraw.Draw(img)
        name = self.scene.get("name", "scene")
        playing = (
            counted(len(self.sources), "source") + " playing"
            if len(self.playing) == len(self.sources)
            else f"{len(self.playing)} of {counted(len(self.sources), 'source')} playing"
        )
        heading(draw, f"{name}, simulated", f"{playing}, heard at {counted(self.places, 'place')}")
        passing = stop.key in self.passes
        side = self.outdoors is not None and "side" not in self.hide and not passing
        if passing:
            plan = self.plan(*PASS_PLAN_BOX)
        elif side:
            left, top, width, _height = PLAN_BOX
            plan = self.plan(left, top, width, PLAN_OVER_SIDE)
            plan = Plan(plan.left, top, plan.s, self.box)
        else:
            plan = self.plan(*PLAN_BOX)
        self.draw_scene(img, plan, stop.key, t, k)
        if side:
            self.draw_side(img, plan, stop.key, t, k)
        draw = ImageDraw.Draw(img)
        if passing:
            self.graph(draw, stop.key, t)
        x = 1010
        if self.fmt == "binaural":
            badge(draw, x, 172)
        y = place(draw, x, 244, *self.title(stop.key, k))
        entries = [(self.title(s.key)[0], s.at) for s in self.stops]
        # The facts as large as leaves the tour its title and three rows at the least.
        y = fitted_facts(draw, x, y, self.lines(k, t), TOUR_BOTTOM - tour_room(len(entries)))
        tour_column(draw, x, y + 22, entries, k, row=TOUR_ROW, bottom=TOUR_BOTTOM)
        footer(draw, self.credit, self.tool)
        return img

    def passing_name(self) -> str:
        """The sources whose pass the stills give, in words: 'Helicopter', 'Police car
        and siren'."""
        playing = {s["name"] for s in self.playing}
        moving = [name for kind, name in self.paths if kind == "sources"]
        names = [n.replace("_", " ") for n in moving if n in playing] or ["the pass"]
        words = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"
        return words[:1].upper() + words[1:]

    def under_the_rest(self, listener: str) -> float | None:
        """How far the pass heard at ``listener`` is under the other sources that play
        there at its loudest (dB; negative where it is over them): its L_Amax against
        the A-weighted level of the others together then, each as loud as it plays in
        the stop of the tour then, as simulated; None where no other is heard then."""
        heard = self.passes[listener]
        t = heard["max_s"]
        k = max((j for j, stop in enumerate(self.stops) if stop.start <= t), default=0)
        moving = {name for kind, name in self.paths if kind == "sources"}
        others = [v for name, v in self.levels_of(listener, t, k).items() if name not in moving]
        energy = sum(10.0 ** (v / 10.0) for v in others)
        if energy <= 0.0:
            return None
        return 10.0 * math.log10(energy) - heard["max_a_db"]

    def heard_at(self, listener: str, t: float) -> float | None:
        """The A-weighted level of the pass heard at ``listener`` at ``t`` (s of the
        trajectories), as simulated; None before its sound arrives and after its last
        place's, and while nothing of it is heard."""
        heard = self.passes[listener]
        times = heard["heard_s"]
        if not times[0] <= t <= times[-1]:
            return None
        levels = pass_levels(heard)
        k = int(np.searchsorted(times, t, side="right"))
        if not np.all(np.isfinite(levels[max(k - 1, 0) : k + 1])):
            return None
        return float(np.interp(t, times, levels))

    def graph(self, draw, listener: str, t: float | None, box=GRAPH_BOX) -> None:
        """The level of the pass heard at ``listener``, as simulated: what has passed
        bright, what is to come faint, the loudest moment marked and, at ``t``, where the
        pass is."""
        left, top, width, height = box
        heard = self.passes[listener]
        times = np.asarray(heard["heard_s"], dtype=float)
        levels = pass_levels(heard)
        ceiling = GRID_DB * math.ceil((heard["max_a_db"] + 2.0) / GRID_DB)
        floor = ceiling - RANGE_DB

        def at(time: float, level: float) -> tuple[float, float]:
            share = (ceiling - min(max(level, floor), ceiling)) / RANGE_DB
            return left + width * time / self.span, top + height * share

        small = font("Regular", 16)
        text(
            draw,
            (left - 64, top - 34),
            f"{self.passing_name()}: level here, dB(A), as simulated",
            font("Medium", 18),
            MUTED,
        )
        level = floor
        while level <= ceiling + 1e-9:
            y = at(0.0, level)[1]
            draw.line([(left, y), (left + width, y)], fill=LINE, width=1)
            text(draw, (left - 10, y), f"{level:.0f}", small, FAINT, "rm")
            level += GRID_DB
        step = time_step(self.span, width)
        for j in range(int(self.span / step + 1e-9) + 1):
            x = at(j * step, floor)[0]
            text(draw, (x, top + height + 8), f"{j * step:.0f} s", small, FAINT, "ma")
        points = [at(a, b) for a, b in zip(times, levels, strict=True)]
        if t is None:
            draw.line(points, fill=LISTENER, width=3)
        else:
            split = int(np.searchsorted(times, t))
            if split < len(points):
                draw.line(points[max(split - 1, 0) :], fill=FAINT, width=2)
            if split > 0:
                now = self.heard_at(listener, t)
                past = points[:split] + ([at(t, now)] if now is not None else [])
                if len(past) > 1:
                    draw.line(past, fill=LISTENER, width=3)
        x, y = at(heard["max_s"], heard["max_a_db"])
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=SOURCE)
        text(draw, (x + 10, y - 4), f"L_Amax {heard['max_a_db']:.1f}", small, SOURCE, "ld")
        if t is not None:
            x = at(min(t, self.span), floor)[0]
            draw.line([(x, top), (x, top + height)], fill=TEXT, width=1)
            now = self.heard_at(listener, t)
            if now is not None:
                x, y = at(t, now)
                draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=LISTENER)

    def side_source(self, key: str, t: float | None = None, k: int | None = None) -> dict | None:
        """The source the view from the side goes through at the mix ``key``, of those
        that play in stop ``k``: the one heard loudest there, as simulated (the nearest,
        where none is heard); where it is at ``t``."""
        playing = self.playing_in(k)
        if not playing:
            return None
        levels = self.levels_of(key, t, k)
        heard = [s for s in playing if s["name"] in levels]
        if heard:
            source = max(heard, key=lambda s: levels[s["name"]])
        else:
            near = self.nearest(key, t, k)
            source = next(s for s in playing if near is not None and s["name"] == near[0])
        return self.placed("sources", source, t)

    def draw_side(
        self, img: Image.Image, plan: Plan, key: str, t: float | None = None, k: int | None = None
    ) -> None:
        """Under the plan, the view from the side through the source heard loudest at
        the mix ``key`` and its listener (:meth:`Outdoors.side`), of those that play in
        stop ``k``, with its caption."""
        name = self.where.get(key, key)
        source = self.side_source(key, t, k)
        if source is None or name not in self.listeners:
            return
        listener = self.placed("listeners", self.listeners[name], t)
        others = [
            self.placed("listeners", lis, t)
            for other, lis in self.listeners.items()
            if other != name and lis.get("enabled", True) is not False
        ]
        left, _top, width, _height = PLAN_BOX
        caption_y = plan.top + plan.size[1] + 22
        top = caption_y + 34
        box = (left, top, width, SIDE_BOTTOM - top)
        # A moving picture draws it again only where what it shows has moved (to a
        # centimetre): the frames of a listener standing still with sources that stand.
        where = tuple(
            tuple(round(float(v), 2) for v in item["position"])
            for item in (source, listener, *others)
        )
        cache = (key, source["name"], where, box)
        area = (int(left) - 40, int(top) - 8, int(left + width) + 40, int(SIDE_BOTTOM) + 16)
        if self._side is not None and self._side[0] == cache:
            said = self._side[1]
            img.paste(self._side[2], area[:2])
        else:
            said = self.outdoors.side(img, box, source, listener, others, still=key in self.where)
            self._side = (cache, said, img.crop(area))
        draw = ImageDraw.Draw(img)
        words = f"From the side, {source['name'].replace('_', ' ')} to you ({said})"
        lines, f = wrapped_line(draw, words, width)
        text(draw, (left, caption_y), lines, f, MUTED)

    def stop_at(self, t: float) -> tuple[int, float]:
        """The stop playing at ``t`` (s of the tour), and the moment of its mix then (s,
        the trajectories' time when something moves)."""
        k = max((j for j, stop in enumerate(self.stops) if stop.at <= t + 1e-9), default=0)
        stop = self.stops[k]
        return k, stop.start + (t - stop.at)

    def frame(self, t: float) -> Image.Image:
        """The moving picture at ``t`` (s of the tour): the still of the stop then, with
        every mover where it is and the level at that moment."""
        return self.still(*self.stop_at(t))

    def shown(self, k: int, t: float | None = None) -> str:
        """What the still of stop ``k`` shows, for the video's messages: the sources that
        play in it, then the stop's listener, each mover at ``t`` of its trajectory
        (:func:`~acoustic_spaces.walkthrough.moment_name`), as ``stage, car @ 3.40 s ->
        seat``; the sources of a group that play more than one, standing still, by
        the group once and how many of it play (``Choir (19)``), as the plan labels a
        crowded group."""

        def walks(kind: str, name: str) -> bool:
            return t is not None and (kind, name) in self.paths

        def named(kind: str, name: str) -> str:
            return moment_name(name, t) if walks(kind, name) else name

        playing = self.playing_in(k)
        groups: dict[str, int] = {}
        for source in playing:
            if source.get("group") and not walks("sources", source["name"]):
                groups[source["group"]] = groups.get(source["group"], 0) + 1
        words: list[str] = []
        for source in playing:
            group = source.get("group")
            if groups.get(group or "", 0) < 2 or walks("sources", source["name"]):
                words.append(named("sources", source["name"]))
            elif f"{group} ({groups[group]})" not in words:
                words.append(f"{group} ({groups[group]})")
        sources = ", ".join(words)
        listener = named("listeners", self.stops[k].key)
        return f"{sources} -> {listener}" if sources else listener

    def shown_at(self, t: float) -> str:
        """What the moving picture at ``t`` (s of the tour) shows (:meth:`shown`)."""
        return self.shown(*self.stop_at(t))

    def thumbnail(self) -> Image.Image:
        img = canvas(THUMBNAIL)
        plan = self.plan(560, 60, 680, 600)
        first = self.stops[0].key if self.stops else None
        self.draw_scene(img, plan, first, 0.0 if self.paths else None, 0 if self.stops else None)
        draw = ImageDraw.Draw(img)
        lines, f = wrapped(draw, self.scene.get("name", "scene"), "Black", 76, 470, 3)
        y = 60
        for line in lines:
            text(draw, (50, y), line, f)
            y += round(f.size * 1.12)
        text(draw, (50, y + 6), "simulated", font("Bold", 52, display=True), MUTED)
        y += 90
        if self.fmt == "binaural":
            thumbnail_badge(draw, 50, y)
            y += 90
        places = f"{counted(len(self.playing), 'source')}, {counted(self.places, 'place')}"
        text(draw, (50, y), places, font("SemiBold", 32), LISTENER)
        text(draw, (50, 655), "made with acoustic-spaces", font("Medium", 24), TEXT)
        return img


__all__ = [
    "CORNERS",
    "LISTENER_SIDES",
    "MID_BANDS",
    "PLAN_BOX",
    "Pictures",
    "SOURCE_SIDES",
    "counted",
    "label_at",
    "laid_out",
    "nice_length",
    "source_labels",
    "wrapped",
    "wrapped_line",
]
