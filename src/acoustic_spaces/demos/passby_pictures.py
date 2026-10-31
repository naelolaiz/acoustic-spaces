"""The pictures of the pass-by demos (:mod:`acoustic_spaces.demos.passby`): the scene from
above with the vehicle where it is at that moment and the listener of the moment, under it
the level heard there over the pass, as simulated, and what the pass gives there; the
frame round them is the demos' (:mod:`acoustic_spaces.demos.pictures`)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from acoustic_spaces.demos import scene_pictures
from acoustic_spaces.demos.passby import CAR, kmh
from acoustic_spaces.demos.pictures import (
    FAINT,
    LINE,
    LISTENER,
    MARGIN,
    MUTED,
    PANEL,
    SOURCE,
    TEXT,
    THUMBNAIL,
    Plan,
    badge,
    canvas,
    facts,
    font,
    footer,
    heading,
    place,
    text,
    thumbnail_badge,
    tour_column,
)
from acoustic_spaces.demos.pipeline import TourStop
from acoustic_spaces.demos.scene_pictures import counted, wrapped

#: Where the plan and the graph under it are drawn: left, top, width, height.
PLAN_BOX = (MARGIN, 172, 860, 470)
GRAPH_BOX = (MARGIN + 64, 700, 796, 150)
#: How far the graph reaches under its top (dB), and its grid (dB, s).
RANGE_DB = 40.0
GRID_DB = 10.0
GRID_S = 2.0


class Pictures(scene_pictures.Pictures):
    """The frames of a pass-by tour (:meth:`frame`), a still of each stop at any moment
    of it (:meth:`still`) and a :meth:`thumbnail`.

    ``scene`` is scene_demo.yaml as data, ``outline`` the model's edges from above,
    ``facts`` the passes' walk.json (the vehicle's places and, per listener, the level
    heard along the pass and what it gives), ``seats`` their entries of seats.json,
    ``stops`` the tour and ``path`` the vehicle's knots
    (:class:`~acoustic_spaces.walkthrough.Knots`)."""

    def __init__(
        self,
        scene: dict,
        outline: np.ndarray,
        facts: dict,
        seats: dict,
        stops: Sequence[TourStop],
        *,
        path,
        credit: str = "",
        tool: Sequence[str] | str = (),
        fmt: str = "binaural",
    ):
        heard = facts["listeners"]
        noun = facts.get("source", CAR)
        # A vehicle of several sources (the police car and its siren) is drawn as one,
        # under its own name, whichever of them play.
        carried = set(facts.get("parts", [noun])) - {noun}
        sources = [
            {**s, "enabled": True} if s["name"] == noun else s
            for s in scene.get("sources", [])
            if s["name"] not in carried
        ]
        scene = {**scene, "sources": sources}
        movement = {
            "times": facts["times"],
            "levels_a_db": {name: {noun: v["levels_a_db"]} for name, v in heard.items()},
        }
        super().__init__(
            scene,
            outline,
            {"responses": []},
            seats,
            stops,
            playing=[noun],
            credit=credit,
            tool=tool,
            fmt=fmt,
            paths={("sources", noun): path},
            movement=movement,
        )
        self.facts, self.heard = facts, heard
        ground = (scene.get("geometry") or {}).get("ground") or {}
        self.zones = [(z.get("name", ""), z["polygon"]) for z in ground.get("zones") or []]
        self.span = max(max(v["heard_s"]) for v in heard.values()) + 0.5

    # -- the scene from above ---------------------------------------------------------

    def draw_scene(
        self, img: Image.Image, plan: Plan, current: str | None, t: float | None = None
    ) -> None:
        """The ground's zones (the road) under the scene's own picture."""
        draw = ImageDraw.Draw(img)
        label = font("Regular", 15)
        for name, polygon in self.zones:
            points = [plan(*p[:2]) for p in polygon]
            draw.polygon(points, fill=PANEL, outline=LINE)
            xs, ys = zip(*points, strict=True)
            text(draw, (min(xs) + 8, min(ys) + 6), name.replace("_", " "), label, FAINT)
        super().draw_scene(img, plan, current, t)

    def label_source(self, draw, at: tuple[float, float], r: float, name: str, f) -> None:
        """The vehicle's name under it, clear of the listeners beside its way."""
        text(draw, (at[0], at[1] + r + 3), name, f, MUTED, "ma")

    # -- what is heard -----------------------------------------------------------------

    def heard_at(self, listener: str, t: float) -> float | None:
        """The A-weighted level heard at ``listener`` ``t`` s into its pass (dB), as
        simulated; None before the vehicle's sound arrives and after its last place's."""
        times = self.heard[listener]["heard_s"]
        if not times[0] <= t <= times[-1]:
            return None
        return float(np.interp(t, times, self.heard[listener]["levels_a_db"]))

    def lines(self, k: int, t: float | None = None) -> list:
        """What a still of stop ``k`` says, ``t`` s into its pass."""
        stop = self.stops[k]
        heard = self.heard[stop.key]
        lines: list = []
        now = self.heard_at(stop.key, t) if t is not None else None
        if now is not None:
            lines.append((f"{now:.0f} dB(A) here now, as simulated", LISTENER))
        lines += [
            f"Loudest {heard['max_a_db']:.1f} dB(A) (L_Amax), {heard['max_s']:.1f} s in",
            f"Sound exposure level {heard['sel_a_db']:.1f} dB(A) (SEL)",
            f"Closest {heard['closest_m']:.1f} m away",
            f"Its direct sound falls {heard['pitch_semitones']:.1f} semitones (Doppler)",
        ]
        if stop.gain_db > 0.5:
            lines.append(f"Turned up {stop.gain_db:.0f} dB here so you can hear it")
        about = self.facts.get("about") or (
            f"A car at {kmh(self.facts['speed_kmh'])} km/h, "
            f"{self.facts['sound_power_a_db']:.1f} dB(A) of sound power (CNOSSOS-EU)"
        )
        lines += [(line, MUTED) for line in about.split("\n")]
        if self.fmt == "binaural":
            lines.append(("L and R mark your ears", MUTED))
        return lines

    def graph(self, draw, listener: str, t: float | None, box=GRAPH_BOX) -> None:
        """The level heard at ``listener`` over its pass, as simulated: what has passed
        bright, what is to come faint, the loudest moment marked and, at ``t``, where the
        pass is."""
        left, top, width, height = box
        heard = self.heard[listener]
        times = np.asarray(heard["heard_s"], dtype=float)
        levels = np.asarray(heard["levels_a_db"], dtype=float)
        ceiling = GRID_DB * math.ceil((heard["max_a_db"] + 2.0) / GRID_DB)
        floor = ceiling - RANGE_DB

        def at(time: float, level: float) -> tuple[float, float]:
            share = (ceiling - min(max(level, floor), ceiling)) / RANGE_DB
            return left + width * time / self.span, top + height * share

        small = font("Regular", 16)
        text(
            draw,
            (left - 64, top - 34),
            "Level heard here, dB(A), as simulated",
            font("Medium", 18),
            MUTED,
        )
        level = floor
        while level <= ceiling + 1e-9:
            y = at(0.0, level)[1]
            draw.line([(left, y), (left + width, y)], fill=LINE, width=1)
            text(draw, (left - 10, y), f"{level:.0f}", small, FAINT, "rm")
            level += GRID_DB
        second = 0.0
        while second <= self.span:
            x = at(second, floor)[0]
            text(draw, (x, top + height + 8), f"{second:.0f} s", small, FAINT, "ma")
            second += GRID_S
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

    # -- the pictures ------------------------------------------------------------------

    def still(self, k: int, t: float | None = None) -> Image.Image:
        """The still of stop ``k``, ``t`` s into its pass, the vehicle where it is then."""
        stop = self.stops[k]
        img = canvas()
        draw = ImageDraw.Draw(img)
        name = self.scene.get("name", "scene")
        speed = kmh(self.facts["speed_kmh"])
        heading(
            draw,
            f"{name}, simulated",
            f"At {speed} km/h, heard at {counted(len(self.stops), 'place')}, "
            "its pitch falling as it goes by (Doppler)",
        )
        self.draw_scene(img, self.plan(*PLAN_BOX), stop.key, t)
        draw = ImageDraw.Draw(img)
        self.graph(draw, stop.key, t)
        x = 1010
        if self.fmt == "binaural":
            badge(draw, x, 172)
        y = place(draw, x, 244, *self.title(stop.key))
        y = facts(draw, x, y, self.lines(k, t), size=23, step=32)
        entries = [(self.title(s.key)[0], s.at) for s in self.stops]
        tour_column(draw, x, y + 22, entries, k, row=28)
        footer(draw, self.credit, self.tool)
        return img

    def thumbnail(self) -> Image.Image:
        """The scene with the vehicle where the first listener hears it loudest."""
        img = canvas(THUMBNAIL)
        first = self.stops[0].key
        self.draw_scene(img, self.plan(560, 60, 680, 600), first, self.heard[first]["max_s"])
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
        speed = f"{kmh(self.facts['speed_kmh'])} km/h, Doppler"
        text(draw, (50, y), speed, font("SemiBold", 32), LISTENER)
        text(draw, (50, 655), "made with acoustic-spaces", font("Medium", 24), TEXT)
        return img


__all__ = ["GRAPH_BOX", "PLAN_BOX", "Pictures"]
