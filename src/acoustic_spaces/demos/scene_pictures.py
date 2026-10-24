"""The stills of a demo of any scene (:mod:`acoustic_spaces.demos.scene_demo`): the scene
from above, drawn from its own geometry, with its sources, its listeners and the
listener of the moment, what is heard there and the tour; the frame round it is the
demos' (:mod:`acoustic_spaces.demos.pictures`)."""

from __future__ import annotations

import math
from collections.abc import Sequence

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
    sources that play a signal, ``credit`` and ``tool`` the footer's lines."""

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
    ):
        self.scene, self.report, self.seats, self.stops = scene, report, seats, list(stops)
        self.credit, self.tool, self.fmt = credit, tool, fmt
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

    def draw_scene(self, img: Image.Image, plan: Plan, current: str | None) -> None:
        """The outline, the sources with their names, the listeners as rings and the
        listener of the moment as a head, with a ray from each playing source."""
        draw = ImageDraw.Draw(img)
        for (xa, ya), (xb, yb) in self.outline:
            draw.line([plan(xa, ya), plan(xb, yb)], fill=LINE, width=2)
        r = self.marker(plan)
        label = font("Regular", 16)
        if current is not None and current in self.listeners:
            levels = {
                s["name"]: self.levels[(s["name"], current)]
                for s in self.playing
                if (s["name"], current) in self.levels
            }
            rays(
                img,
                plan,
                levels,
                self.playing,
                plan(*self.listeners[current]["position"]),
                lambda _n: SOURCE,
            )
            draw = ImageDraw.Draw(img)
        for source in self.sources:
            x, y = plan(*source["position"])
            colour = SOURCE if source in self.playing else MUTED
            rs = max(4.0, r * 0.55)
            draw.ellipse([x - rs, y - rs, x + rs, y + rs], fill=colour, outline=colour)
            text(draw, (x + rs + 4, y), source["name"], label, MUTED, "lm")
        for name, lis in self.listeners.items():
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
        if current is not None and current in self.listeners:
            lis = self.listeners[current]
            forward, left = axes(plan, lis)
            head(img, plan(*lis["position"]), forward, left, r)
        draw = ImageDraw.Draw(img)
        metres = nice_length(self.box[2] - self.box[0])
        unit = f"{metres:g} m"
        scale_bar(draw, plan.left + 12, plan.top + plan.size[1] - 14, metres * plan.s, unit)

    # -- words ---------------------------------------------------------------------------

    def level_at(self, listener: str) -> float | None:
        """The A-weighted level of every playing source together (dB), as simulated;
        None when the simulation gives none for one of them (a response too short for
        its decay), rather than a sum that leaves it out."""
        values = [self.levels.get((s["name"], listener)) for s in self.playing]
        if not values or any(v is None for v in values):
            return None
        return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in values))

    def t30_at(self, listener: str) -> float | None:
        values = [self.t30.get((s["name"], listener)) for s in self.playing]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def nearest(self, listener: str) -> tuple[str, float] | None:
        """The playing source nearest the listener, and how far it is (m)."""
        at = self.listeners[listener]["position"]
        if not self.playing:
            return None
        name, source = min(
            ((s["name"], s) for s in self.playing), key=lambda p: math.dist(p[1]["position"], at)
        )
        return name, math.dist(source["position"], at)

    def lines(self, k: int) -> list:
        stop = self.stops[k]
        seat = self.seats.get(stop.key, {})
        lines: list = []
        level = self.level_at(stop.key)
        if level is not None:
            lines.append(f"{level:.0f} dB(A) here, as simulated")
        lag = seat.get("lag_s")
        if lag is not None:
            lines.append(f"The sound arrives {lag * 1e3:.0f} ms after it leaves")
        t30 = self.t30_at(stop.key)
        if t30 is not None:
            lines.append(f"Reverberation time T30 {t30:.1f} s (mean of 500 Hz and 1 kHz)")
        near = self.nearest(stop.key)
        if near is not None:
            lines.append(f"Nearest playing: {near[0].replace('_', ' ')}, {near[1]:.1f} m away")
        if stop.gain_db > 0.5:
            lines.append(f"Turned up {stop.gain_db:.0f} dB here so you can hear it")
        if self.fmt == "binaural":
            lines.append(("L and R mark your ears", MUTED))
        return lines

    def title(self, listener: str) -> tuple[str, str]:
        """The listener's name, and where it comes among the scene's listeners in use."""
        names = [n for n, lis in self.listeners.items() if lis.get("enabled", True) is not False]
        k = names.index(listener) + 1 if listener in names else 0
        return listener.replace("_", " "), f"Listener {k} of {len(names)}"

    # -- the pictures ------------------------------------------------------------------

    def still(self, k: int) -> Image.Image:
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
        self.draw_scene(img, plan, stop.key)
        draw = ImageDraw.Draw(img)
        x = 1010
        if self.fmt == "binaural":
            badge(draw, x, 172)
        y = place(draw, x, 244, *self.title(stop.key))
        y = facts(draw, x, y, self.lines(k), size=23, step=32)
        entries = [(self.title(s.key)[0], s.at) for s in self.stops]
        tour_column(draw, x, y + 22, entries, k, row=28)
        footer(draw, self.credit, self.tool)
        return img

    def thumbnail(self) -> Image.Image:
        img = canvas(THUMBNAIL)
        plan = self.plan(560, 60, 680, 600)
        first = self.stops[0].key if self.stops else None
        self.draw_scene(img, plan, first)
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
