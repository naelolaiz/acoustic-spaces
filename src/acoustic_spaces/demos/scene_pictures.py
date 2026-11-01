"""The stills of a demo of any scene (:mod:`acoustic_spaces.demos.scene_demo`): the scene
from above, drawn from its own geometry, with its sources, its listeners and the
listener of the moment, what is heard there and the tour; the frame round it is the
demos' (:mod:`acoustic_spaces.demos.pictures`). When something moves, the frames of a
moving picture (:meth:`Pictures.frame`) draw every walker where it is at that moment,
along its walk."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from acoustic_spaces.demos.pictures import (
    LINE,
    LISTENER,
    MARGIN,
    MUTED,
    SOURCE,
    TEXT,
    THUMBNAIL,
    Plan,
    axes,
    badge,
    canvas,
    facts,
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
)
from acoustic_spaces.demos.pipeline import TourStop

#: Where the plan is drawn on a still: left, top, width, height.
PLAN_BOX = (MARGIN, 172, 860, 640)
#: The octave bands the reverberation time is averaged over (Hz).
MID_BANDS = (500, 1000)


def _mid(values: Sequence, bands: Sequence[float]) -> float | None:
    """The mean of ``values`` over MID_BANDS (ISO 3382-1's mid-frequency value), None
    unless both bands have one."""
    picked = [v for v, f in zip(values, bands, strict=False) if f in MID_BANDS and v is not None]
    return sum(picked) / len(picked) if len(picked) == len(MID_BANDS) else None


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


def nice_length(metres: float) -> float:
    """A round length (1, 2, 5, 10...) of about a quarter of ``metres``, for a scale bar."""
    target = max(metres / 4.0, 1e-3)
    power = 10.0 ** math.floor(math.log10(target))
    return min((1.0, 2.0, 5.0, 10.0), key=lambda m: abs(m * power - target)) * power


class Pictures:
    """One still per stop of the tour (:meth:`still`) and a :meth:`thumbnail`.

    ``scene`` is the scene as data, ``outline`` the model's edges from above ((E, 2, 2),
    :func:`acoustic_spaces.room.outline_edges`), ``report`` the results' parameters.json,
    ``seats`` the mixes' entries of seats.json, ``stops`` the tour, ``playing`` the
    sources that play a signal, ``credit`` and ``tool`` the footer's lines. When
    something moves, ``paths`` gives each walker's knots (("sources" or "listeners",
    name): :class:`~acoustic_spaces.walkthrough.Knots`) and ``movement`` the moving mixes'
    moments and the level of each source at each listener at each (seats.json's
    ``movement``), which :meth:`frame` draws and says at any moment of the tour.
    ``indoors``, the mixes heard in a house at each listener
    (:meth:`~acoustic_spaces.demos.indoors.StillHouses.pictures`, with ``report``'s
    levels those in the houses): whether the sources are heard through its window
    (``airborne``), the rumble of its floors and walls at each listener
    (``rumble_a_db``, ``vibration_vdb``) and what to say of it (``about``)."""

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
    ):
        self.scene, self.report, self.seats, self.stops = scene, report, seats, list(stops)
        self.paths = dict(paths or {})
        self.moments = np.asarray((movement or {}).get("times", [0.0]), dtype=float)
        self.along = (movement or {}).get("levels_a_db", {})
        self.credit, self.tool, self.fmt = credit, tool, fmt
        self.indoors = dict(indoors or {})
        self.sources = [s for s in scene["sources"] if s.get("enabled", True) is not False]
        self.playing = [s for s in self.sources if s["name"] in set(playing)]
        self.listeners = {item["name"]: item for item in scene["listeners"]}
        self.outline = np.asarray(outline, dtype=float).reshape(-1, 2, 2)
        # A level the simulation could not give (None: a band without a decay) is left out.
        self.levels = {
            (e["source"], e["listener"]): e["spl_a_db"]
            for e in report["responses"]
            if e.get("spl_a_db") is not None
        }
        bands = report.get("bands_hz", [63, 125, 250, 500, 1000, 2000, 4000, 8000])
        self.t30 = {
            (e["source"], e["listener"]): _mid(e.get("parameters", {}).get("t30", []), bands)
            for e in report["responses"]
        }
        points = [self.outline.reshape(-1, 2)] if len(self.outline) else []
        points.append(np.array([s["position"][:2] for s in self.sources], dtype=float))
        points.append(np.array([lis["position"][:2] for lis in self.listeners.values()]))
        points += [path.positions[:, :2] for path in self.paths.values()]
        xy = np.concatenate([p for p in points if len(p)])
        (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
        pad = max(0.05 * max(x1 - x0, y1 - y0), 1.0)
        self.box = (float(x0 - pad), float(y0 - pad), float(x1 + pad), float(y1 + pad))

    # -- the scene from above ---------------------------------------------------------

    def plan(self, left: float, top: float, width: float, height: float) -> Plan:
        """The plan that fits the scene into the box, centred in it."""
        x0, y0, x1, y1 = self.box
        scale = min(width / (x1 - x0), height / (y1 - y0))
        plan = Plan(left, top, scale, self.box)
        w, h = plan.size
        return Plan(left + (width - w) / 2, top + (height - h) / 2, scale, self.box)

    def marker(self, plan: Plan) -> float:
        """A head's radius on the picture: a head's width in the scene's scale, kept
        between what is seen and what hides the plan."""
        return float(min(16.0, max(7.0, 0.25 * plan.s)))

    # -- where things are ----------------------------------------------------------

    def placed(self, kind: str, item: dict, t: float | None) -> dict:
        """``item`` (a source or listener as scene data) where its walk has it at ``t`` (s
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

    def levels_of(self, listener: str, t: float | None) -> dict[str, float]:
        """The A-weighted level of each playing source at ``listener`` (dB), as simulated:
        at ``t`` when something moves, else as the still results give it; a source the
        simulation gives none for (a response too short for its decay) is left out."""
        if t is not None and listener in self.along:
            k = self.moment(t)
            values = {name: v[k] for name, v in self.along[listener].items()}
            return {name: v for name, v in values.items() if v is not None}
        return {
            s["name"]: self.levels[(s["name"], listener)]
            for s in self.playing
            if (s["name"], listener) in self.levels
        }

    def draw_paths(self, draw, plan: Plan) -> None:
        """Each walk as a dotted line, every quarter of a metre or so along it."""
        for path in self.paths.values():
            count = max(2, int(math.ceil(path.length_m / 0.25)) + 1)
            positions, _ = path.at(np.linspace(0.0, path.duration_s, count))
            points = [plan(*p[:2]) for p in positions]
            for k in range(0, len(points) - 1, 2):
                draw.line([points[k], points[k + 1]], fill=MUTED, width=2)

    def draw_scene(
        self, img: Image.Image, plan: Plan, current: str | None, t: float | None = None
    ) -> None:
        """The outline, the sources with their names, the listeners as rings and the
        listener of the moment as a head, with a ray from each playing source; with
        ``t``, the walks dotted and every walker where it is then."""
        draw = ImageDraw.Draw(img)
        for (xa, ya), (xb, yb) in self.outline:
            draw.line([plan(xa, ya), plan(xb, yb)], fill=LINE, width=2)
        if t is not None:
            self.draw_paths(draw, plan)
        r = self.marker(plan)
        label = font("Regular", 16)
        sources = [self.placed("sources", s, t) for s in self.sources]
        listeners = {name: self.placed("listeners", lis, t) for name, lis in self.listeners.items()}
        if current is not None and current in listeners:
            playing = {s["name"] for s in self.playing}
            rays(
                img,
                plan,
                self.levels_of(current, t),
                [s for s in sources if s["name"] in playing],
                plan(*listeners[current]["position"]),
                lambda _n: SOURCE,
            )
            draw = ImageDraw.Draw(img)
        for source, given in zip(sources, self.sources, strict=True):
            x, y = plan(*source["position"])
            colour = SOURCE if given in self.playing else MUTED
            rs = max(4.0, r * 0.55)
            draw.ellipse([x - rs, y - rs, x + rs, y + rs], fill=colour, outline=colour)
            self.label_source(draw, (x, y), rs, source["name"], label)
        for name, lis in listeners.items():
            if lis.get("enabled", True) is False or name == current:
                continue
            forward, _left = axes(plan, lis)
            ring(draw, plan(*lis["position"]), forward, r * 0.8, r * 1.2, r * 0.6)
            text(
                draw,
                (plan(*lis["position"])[0], plan(*lis["position"])[1] + r + 4),
                name,
                label,
                MUTED,
                "ma",
            )
        if current is not None and current in listeners:
            lis = listeners[current]
            forward, left = axes(plan, lis)
            head(img, plan(*lis["position"]), forward, left, r)
        draw = ImageDraw.Draw(img)
        metres = nice_length(self.box[2] - self.box[0])
        unit = f"{metres:g} m"
        scale_bar(draw, plan.left + 12, plan.top + plan.size[1] - 14, metres * plan.s, unit)

    def label_source(self, draw, at: tuple[float, float], r: float, name: str, f) -> None:
        """A source's name beside its mark of radius ``r`` at ``at``, on its right."""
        text(draw, (at[0] + r + 4, at[1]), name, f, MUTED, "lm")

    # -- words ---------------------------------------------------------------------------

    def level_at(self, listener: str, t: float | None = None) -> float | None:
        """The A-weighted level of every playing source together (dB), as simulated, at
        ``t`` when something moves; None when the simulation gives none for one of them
        (a response too short for its decay), rather than a sum that leaves it out, and
        when nothing of them is heard."""
        if t is not None and listener in self.along:
            k = self.moment(t)
            values = [
                self.along[listener].get(s["name"], [None] * (k + 1))[k] for s in self.playing
            ]
        else:
            values = [self.levels.get((s["name"], listener)) for s in self.playing]
        rumble = self.indoors.get("rumble_a_db", {}).get(listener)
        if not self.indoors.get("airborne", True):
            values = []  # in the house, the rumble alone
        elif not values or any(v is None for v in values):
            return None
        energy = sum(10.0 ** (v / 10.0) for v in values)
        if rumble is not None:
            energy += 10.0 ** (rumble / 10.0)
        return 10.0 * math.log10(energy) if energy > 0.0 else None

    def t30_at(self, listener: str) -> float | None:
        values = [self.t30.get((s["name"], listener)) for s in self.playing]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def nearest(self, listener: str, t: float | None = None) -> tuple[str, float] | None:
        """The playing source nearest the listener, and how far it is (m), at ``t``."""
        at = self.placed("listeners", self.listeners[listener], t)["position"]
        if not self.playing:
            return None
        placed = [self.placed("sources", s, t) for s in self.playing]
        name, source = min(
            ((s["name"], s) for s in placed), key=lambda p: math.dist(p[1]["position"], at)
        )
        return name, math.dist(source["position"], at)

    def lines(self, k: int, t: float | None = None) -> list:
        """What a still of stop ``k`` says, at ``t`` of the tour when something moves."""
        stop = self.stops[k]
        seat = self.seats.get(stop.key, {})
        lines: list = []
        level = self.level_at(stop.key, t)
        if level is not None:
            lines.append(f"{level:.0f} dB(A) here, as simulated")
        # The moving mixes keep when the sound arrives apart, where nothing moves.
        lag = seat["arrival_s"] if "arrival_s" in seat else seat.get("lag_s")
        if lag is not None:
            lines.append(f"The sound arrives {lag * 1e3:.0f} ms after it leaves")
        t30 = self.t30_at(stop.key)
        if t30 is not None:
            where = " where the walks start" if seat.get("moves") else ""
            lines.append(f"Reverberation time T30 {t30:.1f} s{where} (mean of 500 Hz and 1 kHz)")
        near = self.nearest(stop.key, t)
        if near is not None:
            lines.append(f"Nearest playing: {near[0].replace('_', ' ')}, {near[1]:.1f} m away")
        shaking = self.indoors.get("vibration_vdb", {}).get(stop.key)
        if shaking is not None:
            rumble = self.indoors["rumble_a_db"][stop.key]
            lines.append(f"Ground shaking {shaking:.0f} VdB, its rumble {rumble:.1f} dB(A)")
        if stop.gain_db > 0.5:
            lines.append(f"Turned up {stop.gain_db:.0f} dB here so you can hear it")
        if "about" in self.indoors:
            lines += [(line, MUTED) for line in self.indoors["about"].split("\n")]
        if t is not None and self.paths:
            walking = ", ".join(name.replace("_", " ") for _kind, name in self.paths)
            lines.append((f"Moving along their walks: {walking}", MUTED))
        if self.fmt == "binaural":
            lines.append(("L and R mark your ears", MUTED))
        return lines

    def title(self, listener: str) -> tuple[str, str]:
        """The listener's name, and where it comes among the scene's listeners in use."""
        names = [n for n, lis in self.listeners.items() if lis.get("enabled", True) is not False]
        k = names.index(listener) + 1 if listener in names else 0
        return listener.replace("_", " "), f"Listener {k} of {len(names)}"

    # -- the pictures ------------------------------------------------------------------

    def still(self, k: int, t: float | None = None) -> Image.Image:
        """The still of stop ``k``; at ``t`` of the tour, with every walker where it is
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
        heading(
            draw, f"{name}, simulated", f"{playing}, heard at {counted(len(self.stops), 'place')}"
        )
        plan = self.plan(*PLAN_BOX)
        self.draw_scene(img, plan, stop.key, t)
        draw = ImageDraw.Draw(img)
        x = 1010
        if self.fmt == "binaural":
            badge(draw, x, 172)
        y = place(draw, x, 244, *self.title(stop.key))
        y = facts(draw, x, y, self.lines(k, t), size=23, step=32)
        entries = [(self.title(s.key)[0], s.at) for s in self.stops]
        tour_column(draw, x, y + 22, entries, k, row=28)
        footer(draw, self.credit, self.tool)
        return img

    def frame(self, t: float) -> Image.Image:
        """The moving picture at ``t`` (s of the tour): the still of the stop then, with
        every walker where it is and the level at that moment."""
        k = max((j for j, stop in enumerate(self.stops) if stop.at <= t + 1e-9), default=0)
        stop = self.stops[k]
        return self.still(k, stop.start + (t - stop.at))

    def thumbnail(self) -> Image.Image:
        img = canvas(THUMBNAIL)
        plan = self.plan(560, 60, 680, 600)
        first = self.stops[0].key if self.stops else None
        self.draw_scene(img, plan, first, 0.0 if self.paths else None)
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
        places = f"{counted(len(self.playing), 'source')}, {counted(len(self.stops), 'place')}"
        text(draw, (50, y), places, font("SemiBold", 32), LISTENER)
        text(draw, (50, 655), "made with acoustic-spaces", font("Medium", 24), TEXT)
        return img


__all__ = ["MID_BANDS", "PLAN_BOX", "Pictures", "counted", "nice_length", "wrapped"]
