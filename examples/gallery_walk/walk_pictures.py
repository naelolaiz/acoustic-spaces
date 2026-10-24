"""Pictures for walk_demo.py's video: the gallery from above with the guide, the walk and
the place the visitor looks round from; the visitor going along, then turning its head;
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
    PA,
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

FLOOR = (34, 38, 52)
WALL = (120, 128, 150)
GUIDE = PA

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


def rating(sti: float) -> str:
    """The STI's words (IEC 60268-16)."""
    if not math.isfinite(sti):
        return "not defined"
    for top, words in ((0.30, "bad"), (0.45, "poor"), (0.60, "fair"), (0.75, "good")):
        if sti < top:
            return words
    return "excellent"


class Pictures:
    """The video's frames: :meth:`frame` at any time of tour.wav, :meth:`moments` for
    its chapters and stills, and :meth:`thumbnail`."""

    def __init__(
        self,
        scene: dict,
        walked: dict,
        turned: dict,
        *,
        walk_s: float,
        turn_at: float,
        voice: tuple[str, float],
        credit: str,
        hrtf=None,
        speed: float = 1.2,
    ):
        self.scene, self.walked, self.turned = scene, walked, turned
        self.walk_s, self.turn_at, self.voice, self.speed = walk_s, turn_at, voice, speed
        self.credit, self.hrtf = credit, hrtf
        self.tool = tool_line("gallery_walk", scene, hrtf=hrtf)
        self.polygon = np.array(scene["geometry"]["floor_plan"], dtype=float)
        (x0, y0), (x1, y1) = self.polygon.min(axis=0), self.polygon.max(axis=0)
        self.box = (float(x0), float(y0), float(x1), float(y1))
        self.guide = next(s for s in scene["sources"] if s["name"] == "guide")
        self.corners = inner_corners(self.polygon)
        for part in (walked, turned):
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
        """Where the visitor is at ``t`` (s of tour.wav), the way it faces, and what the
        simulation says it hears there."""
        if t < self.walk_s:
            part, local = self.walked, t
        else:
            part, local = self.turned, max(0.0, t - self.turn_at)
        times = part["times"]
        k = int(np.clip(np.searchsorted(times, local, side="right") - 1, 0, len(times) - 1))
        if k + 1 < len(times) and local - times[k] > times[k + 1] - local:
            k += 1  # the nearest place, as the sound has it
        where = [float(np.interp(local, times, part["positions"][:, j])) for j in range(3)]
        return {
            "part": "walk" if part is self.walked else "turn",
            "local": local,
            "at": where,
            "yaw": float(np.interp(local, times, part["unwrapped"])),
            "level": part["level_a_db"][k],
            "sti": part["sti"][k],
            "sight": part["sight"][k],
        }

    def moments(self) -> list[tuple[str, float, str]]:
        """The tour's moments (a name, the time in tour.wav, a title): setting off, the
        corner, out of sight, standing; then each hold and turn of the head."""
        walked, found = self.walked, [("setting-off", 0.0, "Setting off, the guide behind you")]
        yaws, times = walked["unwrapped"], walked["times"]
        turning = np.flatnonzero(np.abs(yaws - yaws[0]) > 1.0)
        if turning.size:
            found.append(("corner", float(times[turning[0]]), "Into the short arm"))
        lost = [k for k, seen in enumerate(walked["sight"]) if not seen]
        if lost:
            found.append(("out-of-sight", float(times[lost[0]]), "The guide out of sight"))
        found.append(("standing", float(times[-1]), "Standing round the corner"))
        keys = self.turned["keys"]
        position = self.turned["positions"][0]
        for (t0, y0), (_t1, y1) in zip(keys, [*keys[1:], keys[-1]], strict=True):
            at = self.turn_at + t0
            if y1 != y0:
                way = "left" if y1 > y0 else "right"
                found.append((f"turning-{way}", at, f"Turning {way}"))
            else:
                words, _ = bearing(position, self.turned["facing"] + y0, self.guide["position"])
                found.append(("still", at, f"Still, the guide {words}"))
        found.sort(key=lambda m: m[1])
        kept = []
        for moment in found:
            if kept and moment[1] - kept[-1][1] < 1.0:
                kept[-1] = (kept[-1][0], kept[-1][1], moment[2])
            else:
                kept.append(moment)
        return kept

    # ------------------------------------------------------------------ drawing

    def _plan(self, left, top, width, height) -> Plan:
        x0, y0, x1, y1 = self.box
        return Plan(left, top, min(width / (x1 - x0), height / (y1 - y0)), self.box)

    def gallery(self, img, plan: Plan, labels: bool = True) -> None:
        """The floor, the walls, the walk, where the visitor looks round, the guide."""
        draw = ImageDraw.Draw(img)
        outline = [plan(x, y) for x, y in self.polygon]
        draw.polygon(outline, fill=FLOOR)
        draw.line([*outline, outline[0]], fill=WALL, width=4)
        path = [plan(*p[:2]) for p in self.walked["positions"]]
        for k in range(0, len(path) - 1, 2):  # dotted
            draw.line([path[k], path[k + 1]], fill=MUTED, width=3)
        stand = plan(*self.turned["positions"][0][:2])
        r = 0.55 * plan.s
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
        """In the empty corner of the L: what the marks are, and a scale."""
        f = font("Medium", 18)
        rows = [
            ("guide", "the guide, the way they face"),
            ("you", "you, the way you face"),
            ("voice", "the voice: straight, or round the corner"),
            ("path", "the walk"),
        ]
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
        heading(
            draw,
            "A walk through a gallery, simulated",
            "The gallery_walk example: a guide talks, you walk away round the corner, "
            "then stand and look round",
        )
        self.gallery(img, self.plan)
        draw = ImageDraw.Draw(img)
        x, y = self.plan(self.box[0] + 9.5, self.box[3] - 1.0)
        self.legend(draw, x, y)
        badge(draw, 1010, 172)
        footer(draw, self.credit, self.tool)
        return img

    def frame(self, t: float) -> Image.Image:
        """The picture at ``t`` (s of tour.wav)."""
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
            where = f"From 3 m in front of the guide, at {self.speed:g} m/s"
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
        k = len(self.walked["times"]) * 3 // 4
        state = {
            "at": list(self.walked["positions"][k]),
            "yaw": float(self.walked["unwrapped"][k]),
            "sight": self.walked["sight"][k],
        }
        self.voice_path(img, plan, state)
        self.visitor(img, plan, state, 0.45 * plan.s)
        draw = ImageDraw.Draw(img)
        text(draw, (50, 70), "Gallery", font("Black", 92, display=True))
        text(draw, (50, 180), "walk,", font("Bold", 56, display=True), MUTED)
        text(draw, (50, 244), "simulated", font("Bold", 56, display=True), MUTED)
        thumbnail_badge(draw, 50, 340)
        text(draw, (50, 430), "walk away, turn round", font("SemiBold", 32), LISTENER)
        text(draw, (50, 655), "a guide's voice round a corner", font("Medium", 26), FAINT)
        return img
