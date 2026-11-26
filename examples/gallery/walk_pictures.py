"""Pictures for walk_demo.py's video: the gallery from above with the guide, the walk and
the place the visitor looks round from (those of them the run plays); the visitor going
along, turning its head, or both, in the run's order;
the way the voice reaches them, straight or bent round the corner; and what they hear,
as simulated. Needs Pillow (which the viz extra brings); Inter is used when installed,
else DejaVu Sans.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from acoustic_spaces.demos.pictures import (
    BG,
    FAINT,
    LISTENER,
    MUTED,
    SOURCE,
    THUMBNAIL,
    Plan,
    arrow,
    badge,
    canvas,
    cone,
    facts,
    font,
    footer,
    head,
    heading,
    place,
    save,
    scale_bar,
    text,
    thumbnail_badge,
    tool_line,
    tour_column,
)
from acoustic_spaces.parameters import sti_rating as rating
from acoustic_spaces.walkthrough import facing_name, moment_name

FLOOR = (34, 38, 52)
WALL = (120, 128, 150)
GUIDE = SOURCE

__all__ = ["Pictures", "save"]


def inside(polygon: np.ndarray, x: float, y: float) -> bool:
    """Whether (x, y) lies in the polygon (n, 2), by the even-odd rule."""
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def in_sight(polygon: np.ndarray, a, b, step: float = 0.05) -> bool:
    """Whether the straight line from ``a`` to ``b`` (x, y) stays inside the floor
    plan: whether the guide is in sight, the corner not in the way."""
    n = max(2, int(math.dist(a[:2], b[:2]) / step))
    return all(
        inside(polygon, a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
        for k in range(1, n)
    )


def inner_corners(polygon: np.ndarray) -> list[tuple[float, float]]:
    """The corners of the floor plan that point into the room (where sound bends)."""
    area = 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0), strict=True)
    )
    found = []
    for k in range(len(polygon)):
        a, b, c = polygon[k - 1], polygon[k], polygon[(k + 1) % len(polygon)]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross * area < 0:  # turning against the polygon's way round
            found.append((float(b[0]), float(b[1])))
    return found


def bearing(at, yaw: float, target) -> tuple[str, float]:
    """Where ``target`` is for someone at ``at`` facing ``yaw`` (degrees): words, and
    the angle (degrees, to the left positive)."""
    angle = math.degrees(math.atan2(target[1] - at[1], target[0] - at[0])) - yaw
    angle = 180.0 - (180.0 - angle) % 360.0
    side = "left" if angle > 0 else "right"
    size = abs(angle)
    if size <= 25:
        words = "in front of you"
    elif size <= 65:
        words = f"ahead on your {side}"
    elif size <= 115:
        words = f"on your {side}"
    elif size <= 155:
        words = f"behind you on your {side}"
    else:
        words = "behind you"
    return words, angle


#: What each part is, for the heading (after "you" or "then") and for the thumbnail.
DOES = {"walk": "walk away round the corner", "turn": "stand and look round"}
SHORT = {"walk": "walk away", "turn": "turn round"}


class Pictures:
    """The video's frames: :meth:`frame` at any time of the video's sound, :meth:`moments` for
    its chapters and stills, and :meth:`thumbnail`. ``parts`` are those the video plays,
    in its order: each ``walk`` or ``turn``, its facts (walk.json, turn.json), and where
    its sound starts and ends in the video's (s)."""

    def __init__(
        self,
        scene: dict,
        parts: list[tuple[str, dict, float, float]],
        *,
        voice: tuple[str, float],
        credit: str,
        hrtf=None,
        pace: str = "at 1.2 m/s",
        tool=None,
    ):
        self.scene, self.parts, self.voice, self.pace = scene, parts, voice, pace
        found = {name: facts for name, facts, _, _ in parts}
        self.walked, self.turned = found.get("walk"), found.get("turn")
        self.credit, self.hrtf = credit, hrtf
        self.tool = tool or tool_line("gallery", scene, hrtf=hrtf)
        self.polygon = np.array(scene["geometry"]["floor_plan"], dtype=float)
        (x0, y0), (x1, y1) = self.polygon.min(axis=0), self.polygon.max(axis=0)
        self.box = (float(x0), float(y0), float(x1), float(y1))
        self.guide = next(s for s in scene["sources"] if s["name"] == "guide")
        self.corners = inner_corners(self.polygon)
        for _, part, _, _ in parts:
            part["times"] = np.asarray(part["times"], dtype=float)
            part["positions"] = np.asarray(part["positions"], dtype=float)
            part["unwrapped"] = np.degrees(np.unwrap(np.radians(part["yaws"])))
            part["sight"] = [
                in_sight(self.polygon, p, self.guide["position"]) for p in part["positions"]
            ]
        self.plan = self._plan(80, 172, 860, 660)
        self.base = self._base()

    # ------------------------------------------------------------------ the moment

    def state(self, t: float) -> dict:
        """Where the visitor is at ``t`` (s of the video's sound), the way it faces, and what the
        simulation says it hears there: in the first part whose sound goes on after ``t``
        (at its start in the silence before it), else at the end of the last."""
        name, part, at, _ = next((each for each in self.parts if t < each[3]), self.parts[-1])
        local = max(0.0, t - at)
        times = part["times"]
        k = int(np.clip(np.searchsorted(times, local, side="right") - 1, 0, len(times) - 1))
        if k + 1 < len(times) and local - times[k] > times[k + 1] - local:
            k += 1  # the nearest place, as the sound has it
        where = [float(np.interp(local, times, part["positions"][:, j])) for j in range(3)]
        return {
            "part": name,
            "local": local,
            "at": where,
            "yaw": float(np.interp(local, times, part["unwrapped"])),
            "level": part["level_a_db"][k],
            "sti": part["sti"][k],
            "sight": part["sight"][k],
        }

    def shown(self, t: float, visitor: str, standing: str) -> str:
        """What the picture at ``t`` (s of the video's sound) shows, for the video's messages: the
        guide, then the listener ``visitor`` at that moment of its walk, or ``standing``
        facing as the visitor's head turns there, as ``guide -> visitor @ 3.40 s`` or
        ``guide -> standing facing 90.0 deg``."""
        state = self.state(t)
        if state["part"] == "walk":
            heard = moment_name(visitor, state["local"])
        else:
            heard = facing_name(standing, state["yaw"])
        return f"{self.guide['name']} -> {heard}"

    def moments(self) -> list[tuple[str, float, str]]:
        """The tour's moments (a name, the time in the video's sound, a title), of the parts
        it plays: of the walk, setting off, the corner, out of sight, standing; of the
        turn, each hold and turn of the head."""
        found = []
        for name, part, start, _ in self.parts:
            if name == "walk":
                found += self._walk_moments(part, start)
            else:
                found += self._turn_moments(part, start)
        found.sort(key=lambda m: m[1])
        kept = []
        for moment in found:
            if kept and moment[1] - kept[-1][1] < 1.0:
                kept[-1] = (kept[-1][0], kept[-1][1], moment[2])
            else:
                kept.append(moment)
        return kept

    @staticmethod
    def _walk_moments(walked: dict, start: float) -> list[tuple[str, float, str]]:
        """The walk's moments, its sound starting at ``start`` (s of the video's)."""
        found = [("setting-off", start + 0.0, "Setting off, the guide behind you")]
        yaws, times = walked["unwrapped"], walked["times"]
        turning = np.flatnonzero(np.abs(yaws - yaws[0]) > 1.0)
        if turning.size:
            found.append(("corner", start + float(times[turning[0]]), "Into the short arm"))
        lost = [k for k, seen in enumerate(walked["sight"]) if not seen]
        if lost:
            found.append(("out-of-sight", start + float(times[lost[0]]), "The guide out of sight"))
        found.append(("standing", start + float(times[-1]), "Standing round the corner"))
        return found

    def _turn_moments(self, turned: dict, start: float) -> list[tuple[str, float, str]]:
        """The turn's moments, each hold and turn of the head, its sound starting at
        ``start`` (s of the video's)."""
        found = []
        keys, position = turned["keys"], turned["positions"][0]
        for (t0, y0), (_t1, y1) in zip(keys, [*keys[1:], keys[-1]], strict=True):
            at = start + t0
            if y1 != y0:
                way = "left" if y1 > y0 else "right"
                found.append((f"turning-{way}", at, f"Turning {way}"))
            else:
                words, _ = bearing(position, turned["facing"] + y0, self.guide["position"])
                found.append(("still", at, f"Still, the guide {words}"))
        return found

    # ------------------------------------------------------------------ drawing

    def _plan(self, left, top, width, height) -> Plan:
        x0, y0, x1, y1 = self.box
        return Plan(left, top, min(width / (x1 - x0), height / (y1 - y0)), self.box)

    def gallery(self, img, plan: Plan, labels: bool = True) -> None:
        """The floor, the walls, the walk and where the visitor looks round (those the
        run plays), the guide."""
        draw = ImageDraw.Draw(img)
        outline = [plan(x, y) for x, y in self.polygon]
        draw.polygon(outline, fill=FLOOR)
        draw.line([*outline, outline[0]], fill=WALL, width=4)
        if self.walked is not None:
            path = [plan(*p[:2]) for p in self.walked["positions"]]
            for k in range(0, len(path) - 1, 2):  # dotted
                draw.line([path[k], path[k + 1]], fill=MUTED, width=3)
        stand, r = None, 0.55 * plan.s
        if self.turned is not None:
            stand = plan(*self.turned["positions"][0][:2])
            draw.ellipse(
                [stand[0] - r, stand[1] - r, stand[0] + r, stand[1] + r], outline=MUTED, width=2
            )
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gx, gy = plan(*self.guide["position"][:2])
        yaw = math.radians(self.guide.get("orientation", {}).get("yaw", 0.0))
        cone(
            ImageDraw.Draw(layer),
            (gx, gy),
            plan.vec(math.cos(yaw), math.sin(yaw)),
            2.2 * plan.s,
            40.0,
            GUIDE + (90,),
        )
        img.alpha_composite(layer)
        d = 0.32 * plan.s
        draw.ellipse([gx - d, gy - d, gx + d, gy + d], fill=GUIDE, outline=BG, width=2)
        if not labels:
            return
        small = font("SemiBold", 17)
        outline_text = {"stroke_width": 4, "stroke_fill": BG}
        draw.text(
            (gx, gy - 0.9 * plan.s), "GUIDE", font=small, fill=GUIDE, anchor="md", **outline_text
        )
        x0, y0, x1, y1 = self.box
        draw.text(
            plan(0.5 * (x0 + x1) + 2.0, y0 + 0.9),
            "LONG ARM",
            font=small,
            fill=(128, 140, 172),
            anchor="mm",
            **outline_text,
        )
        draw.text(
            plan(4.0, y1 - 0.9),
            "SHORT ARM",
            font=small,
            fill=(128, 140, 172),
            anchor="mm",
            **outline_text,
        )
        if stand is not None:
            draw.text(
                (stand[0], stand[1] + r + 16),
                "where you look round",
                font=font("Medium", 15),
                fill=MUTED,
                anchor="ma",
                **outline_text,
            )

    def voice_path(self, img, plan: Plan, state: dict) -> None:
        """The way the voice reaches the visitor: straight while the guide is in sight,
        else bent round the nearest inner corner."""
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        guide, at = self.guide["position"], state["at"]
        width = max(2, round(0.12 * plan.s))
        if state["sight"]:
            draw.line([plan(*guide[:2]), plan(*at[:2])], fill=GUIDE + (200,), width=width)
        elif self.corners:
            corner = min(self.corners, key=lambda c: math.dist(c, at[:2]) + math.dist(c, guide[:2]))
            points = [plan(*guide[:2]), plan(*corner), plan(*at[:2])]
            draw.line(points, fill=GUIDE + (150,), width=width, joint="curve")
            cx, cy = points[1]
            draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=GUIDE + (230,))
        img.alpha_composite(layer)

    def visitor(self, img, plan: Plan, state: dict, size: float) -> None:
        yaw = math.radians(state["yaw"])
        forward = plan.vec(math.cos(yaw), math.sin(yaw))
        left = plan.vec(-math.sin(yaw), math.cos(yaw))
        head(img, plan(*state["at"][:2]), forward, left, size)

    def legend(self, draw, x: float, y: float) -> None:
        """In the empty corner of the L: what the marks are (the walk's dots when the run
        plays it), and a scale."""
        f = font("Medium", 18)
        rows = [
            ("guide", "the guide, the way they face"),
            ("you", "you, the way you face"),
            ("voice", "the voice: straight, or round the corner"),
        ]
        if self.walked is not None:
            rows.append(("path", "the walk"))
        for what, label in rows:
            if what == "guide":
                cone(draw, (x + 6, y + 9), (1.0, 0.0), 22, 40.0, tuple(c // 2 for c in GUIDE))
                draw.ellipse([x, y + 3, x + 12, y + 15], fill=GUIDE)
            elif what == "you":
                draw.ellipse([x, y + 2, x + 14, y + 16], fill=LISTENER)
                arrow(draw, (x + 14, y + 9), (1.0, 0.0), 14, 7, LISTENER, 2)
            elif what == "voice":
                draw.line([(x, y + 9), (x + 28, y + 9)], fill=GUIDE, width=3)
            else:
                for k in range(0, 28, 8):
                    draw.line([(x + k, y + 9), (x + k + 4, y + 9)], fill=MUTED, width=3)
            text(draw, (x + 44, y - 1), label, f, MUTED)
            y += 34
        scale_bar(draw, x, y + 34, 5.0 * self.plan.s, "5 m")

    def _base(self) -> Image.Image:
        img = canvas()
        draw = ImageDraw.Draw(img)
        does = [DOES[name] for name, _, _, _ in self.parts]
        heading(
            draw,
            "A walk through a gallery, simulated",
            f"The gallery example: a guide talks, you {', then '.join(does)}",
        )
        self.gallery(img, self.plan)
        draw = ImageDraw.Draw(img)
        x, y = self.plan(self.box[0] + 9.5, self.box[3] - 1.0)
        self.legend(draw, x, y)
        badge(draw, 1010, 172)
        footer(draw, self.credit, self.tool)
        return img

    def frame(self, t: float) -> Image.Image:
        """The picture at ``t`` (s of the video's sound)."""
        state = self.state(t)
        img = self.base.copy()
        self.voice_path(img, self.plan, state)
        self.visitor(img, self.plan, state, 0.42 * self.plan.s)
        draw = ImageDraw.Draw(img)
        guide = self.guide["position"]
        words, _ = bearing(state["at"], state["yaw"], guide)
        level, sti = state["level"], state["sti"]
        hears = f"Their voice at your ears: {level:.0f} dB(A), STI {sti:.2f} ({rating(sti)})"
        if state["part"] == "walk":
            title = "Walking away" if state["sight"] else "Round the corner"
            start = math.dist(self.walked["positions"][0][:2], guide[:2])
            where = f"Setting off {start:.0f} m from the guide, {self.pace}"
            distance = math.dist(state["at"][:2], guide[:2])
            if state["sight"]:
                lines = [f"The guide {words}, {distance:.1f} m away", hears]
            else:
                corner = min(self.corners, key=lambda c: math.dist(c, state["at"][:2]))
                came, _ = bearing(state["at"], state["yaw"], corner)
                lines = [
                    f"The guide out of sight, {distance:.1f} m away",
                    f"Their voice bends round the corner, {came}",
                    hears,
                ]
        else:
            title = "Looking round"
            distance = math.dist(state["at"][:2], guide[:2])
            where = f"Standing {distance:.0f} m in front of the guide, turning the head"
            facing = (state["yaw"] - self.turned["facing"]) % 360.0
            lines = [f"Turned {facing:.0f}° left: the guide {words}", hears]
            if not self.hrtf:
                lines.append(
                    ("A spherical head has no outer ears: behind sounds like in front", MUTED)
                )
        lines.append(("L and R mark your ears", MUTED))
        y = place(draw, 1010, 244, title, where)
        y = facts(draw, 1010, y, lines)
        moments = self.moments()
        current = max(k for k, (_, t0, _) in enumerate(moments) if t0 <= t + 1e-6)
        entries = [(title, t0) for _, t0, title in moments]
        tour_column(draw, 1010, max(y + 24, 520), entries, current, row=26, title="Moments")
        return img

    def thumbnail(self) -> Image.Image:
        img = canvas(THUMBNAIL)
        plan = self._plan(560, 60, 680, 600)
        self.gallery(img, plan, labels=False)
        part = self.walked if self.walked is not None else self.turned
        k = len(part["times"]) * 3 // 4  # well along the walk, or the turn
        state = {
            "at": list(part["positions"][k]),
            "yaw": float(part["unwrapped"][k]),
            "sight": part["sight"][k],
        }
        self.voice_path(img, plan, state)
        self.visitor(img, plan, state, 0.45 * plan.s)
        draw = ImageDraw.Draw(img)
        text(draw, (50, 70), "Gallery", font("Black", 92, display=True))
        what = "walk," if self.walked is not None else "look round,"
        text(draw, (50, 180), what, font("Bold", 56, display=True), MUTED)
        text(draw, (50, 244), "simulated", font("Bold", 56, display=True), MUTED)
        thumbnail_badge(draw, 50, 340)
        does = ", ".join(SHORT[name] for name, _, _, _ in self.parts)
        text(draw, (50, 430), does, font("SemiBold", 32), LISTENER)
        where = "round a corner" if self.walked is not None else "as you look round"
        text(draw, (50, 655), f"a guide's voice {where}", font("Medium", 26), FAINT)
        return img
