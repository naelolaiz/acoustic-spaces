"""Stills for stadium_demo.py's video: the stadium from above with the loudspeakers, the
listener of the moment and how loud each loudspeaker reaches them, the stage close up,
and the tour. Needs Pillow (which the viz extra brings); Inter is used when installed,
else DejaVu Sans.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (13, 17, 28)
PANEL = (21, 27, 42)
LINE = (58, 68, 92)
TEXT = (232, 236, 244)
MUTED = (148, 158, 182)
FAINT = (96, 106, 132)
GROUND = (20, 30, 30)
GRASS = (30, 52, 40)
CROWD = (52, 58, 52)
STAND = (38, 48, 74)
STAND_EMPTY = (30, 34, 46)
STAGE = (78, 58, 42)
STAGE_EDGE = (150, 112, 78)
PA = (255, 166, 77)
MONITOR = (240, 110, 170)
LISTENER = (102, 217, 239)
PLAYER = (170, 226, 110)

#: The sources stadium_demo.py's ``band`` puts on the stage.
BAND = ("drum_kit", "singer_voice", "guitar_amp")

#: Where make_video draws the spectrogram: left, top, width, height.
SPECTRUM_BOX = (100, 892, 1720, 96)
CREDIT = (
    'Music: Rhythmussportgruppe, "Spicy Funk Cake" (Niklas Dahlheimer), recorded for the DEGA '
    "at Hochschule Düsseldorf, CC BY 4.0 · doi.org/10.5281/zenodo.3601032"
)
STAGE_BOX = (-60.0, -29.0, -37.0, 29.0)  # x0, y0, x1, y1 of the close-up

_FONTS: dict = {}


def font(weight: str, size: int, display: bool = False) -> ImageFont.FreeTypeFont:
    """Inter (Display for titles) when installed, else DejaVu Sans, else Pillow's own."""
    key = (weight, size, display)
    if key not in _FONTS:
        family = "InterDisplay" if display else "Inter"
        bold = weight in ("SemiBold", "Bold", "Black")
        for name in (f"{family}-{weight}.otf", f"{family}-{weight}.ttf"):
            try:
                _FONTS[key] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            try:
                _FONTS[key] = ImageFont.truetype(
                    "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
                )
            except OSError:
                _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def text(draw, xy, s, f, fill=TEXT, anchor="la") -> None:
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def arrow(draw, start, direction, length: float, head: float, colour, width: int) -> None:
    """A line from ``start`` along the unit vector ``direction``, with a head at its end."""
    (x, y), (dx, dy) = start, direction
    tip = (x + dx * length, y + dy * length)
    base = (tip[0] - dx * head, tip[1] - dy * head)
    draw.line([(x, y), base], fill=colour, width=width)
    draw.polygon(
        [
            tip,
            (base[0] - dy * head * 0.6, base[1] + dx * head * 0.6),
            (base[0] + dy * head * 0.6, base[1] - dx * head * 0.6),
        ],
        fill=colour,
    )


def cone(draw, apex, direction, length: float, half_angle: float, rgba) -> None:
    """A fan from ``apex`` along the unit vector ``direction``, ``half_angle`` degrees to
    either side: where a loudspeaker aims."""
    (x, y), (dx, dy) = apex, direction
    aim = math.atan2(dy, dx)
    rim = [
        (x + length * math.cos(aim + math.radians(a)), y + length * math.sin(aim + math.radians(a)))
        for a in range(-int(half_angle), int(half_angle) + 1, 4)
    ]
    draw.polygon([(x, y), *rim], fill=rgba)


def glow(img: Image.Image, shapes, blur: float) -> None:
    """Blurred discs, ``shapes`` as (x, y, radius, rgba), under what comes next."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, r, rgba in shapes:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))


def levels_at(results: Path) -> dict[tuple[str, str], float]:
    """A-weighted level (dB) of each source at each listener, as simulated at the
    source's sound power."""
    report = json.loads((results / "parameters.json").read_text(encoding="utf-8"))
    return {(e["source"], e["listener"]): e["spl_a_db"] for e in report["responses"]}


def read_obj(path: Path) -> list[tuple[str, list[tuple[float, float, float]]]]:
    """The faces of an OBJ file with the group each is in."""
    vertices, faces, group = [], [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append(tuple(float(v) for v in parts[1:4]))
        elif parts[0] in ("g", "o"):
            group = parts[1] if len(parts) > 1 else ""
        elif parts[0] == "f":
            faces.append((group, [vertices[int(p.split("/")[0]) - 1] for p in parts[1:]]))
    return faces


class Plan:
    """Scene x and y to pixels: from above, north (+y) up; or ``audience``: the stage as
    the pitch sees it, its back at the top and the audience's left (-y) on the left."""

    def __init__(self, left, top, scale, box, audience=False):
        self.left, self.top, self.s = left, top, scale
        self.x0, self.y0, self.x1, self.y1 = box
        self.audience = audience

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        if self.audience:
            return (self.left + (y - self.y0) * self.s, self.top + (x - self.x0) * self.s)
        return (self.left + (x - self.x0) * self.s, self.top + (self.y1 - y) * self.s)

    def vec(self, dx: float, dy: float) -> tuple[float, float]:
        return (dy, dx) if self.audience else (dx, -dy)

    @property
    def size(self) -> tuple[float, float]:
        w, h = (self.x1 - self.x0) * self.s, (self.y1 - self.y0) * self.s
        return (h, w) if self.audience else (w, h)


def kind(name: str) -> str:
    if name in BAND:
        return "band"
    if name.startswith(("vocal_wedges", "drum_fill", "side_fill")):
        return "monitor"
    return "sub" if "sub" in name else "pa"


COLOURS = {"pa": PA, "sub": PA, "monitor": MONITOR, "band": PLAYER}


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


class Pictures:
    def __init__(self, scene: dict, folder: Path, levels, seats, tour, titles, gains=None):
        self.scene, self.levels, self.seats, self.tour, self.titles = (
            scene,
            levels,
            seats,
            tour,
            titles,
        )
        self.gains = gains or {}  # dB each listener is turned up by in the video
        self.faces = read_obj(folder / scene["geometry"]["file"])
        self.sources = scene["sources"]
        self.listeners = {item["name"]: item for item in scene["listeners"]}
        ground = scene["geometry"].get("ground", {}).get("extent", [[-100, -82], [100, 82]])
        self.box = (ground[0][0], ground[0][1], ground[1][0], ground[1][1])
        zones = scene["geometry"].get("ground", {}).get("zones", [])
        self.zones = [z["polygon"] for z in zones]
        early = scene.get("simulation", {}).get("early", {})
        order = early.get("max_order")
        self.tool = (
            "Simulated with acoustic_spaces (stadium example): "
            + (f"image sources to order {order}, " if order else "")
            + "stochastic ray tracing for the late sound, binaural through a spherical head"
        )

    def title(self, listener: str) -> tuple[str, str]:
        return self.titles.get(listener, (listener.replace("_", " ").capitalize(), ""))

    # ------------------------------------------------------------------ drawing

    def stadium(self, img, plan: Plan, listener: str | None, close: bool = False) -> None:
        draw = ImageDraw.Draw(img)
        corners = [plan(self.box[0], self.box[1]), plan(self.box[2], self.box[3])]
        xs, ys = sorted(c[0] for c in corners), sorted(c[1] for c in corners)
        draw.rectangle([xs[0], ys[0], xs[1], ys[1]], fill=GROUND)
        grass = [plan(*p) for p in [(-60.5, -42.0), (60.5, -42.0), (60.5, 42.0), (-60.5, 42.0)]]
        draw.polygon(grass, fill=GRASS)
        for polygon in self.zones:
            draw.polygon([plan(*p) for p in polygon], fill=CROWD)
        # Faces without outlines (the model's corners are triangles), lowest first; the
        # walls as lines, the roofs see-through, the stage outlined as a whole.
        roofs, stage = [], []
        for group, face in sorted(self.faces, key=lambda f: sum(v[2] for v in f[1]) / len(f[1])):
            points = [plan(v[0], v[1]) for v in face]
            if group in ("roof", "stage_roof"):
                roofs.append(points)
            elif group.startswith("audience_"):
                draw.polygon(points, fill=STAND)
            elif group == "west_stand":
                draw.polygon(points, fill=STAND_EMPTY)
            elif group.startswith("stage"):
                draw.polygon(points, fill=STAGE)
                stage += points
            else:
                draw.line(points + points[:1], fill=LINE, width=2)
        if stage:
            xs, ys = [p[0] for p in stage], [p[1] for p in stage]
            draw.rectangle([min(xs), min(ys), max(xs), max(ys)], outline=STAGE_EDGE, width=2)
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        roof = ImageDraw.Draw(layer)
        for points in roofs:
            roof.polygon(points, fill=(150, 160, 190, 26))
        img.alpha_composite(layer)
        if listener is not None:
            self.rays(img, plan, listener)
        self.loudspeakers(img, plan, 7 if close else 5)
        self.people(img, plan, listener, 9 if close else 6, close)

    def rays(self, img, plan: Plan, listener: str) -> None:
        """A line from each loudspeaker to the listener, brighter the louder it is there
        (over the 24 dB below the loudest)."""
        here = {s: v for (s, lis), v in self.levels.items() if lis == listener}
        if not here:
            return
        top = max(here.values())
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        lx, ly = plan(*self.listeners[listener]["position"][:2])
        for source in sorted(self.sources, key=lambda s: here.get(s["name"], -999)):
            weight = 1.0 + (here.get(source["name"], -999) - top) / 24.0
            if weight <= 0:
                continue
            colour = COLOURS[kind(source["name"])]
            sx, sy = plan(*source["position"][:2])
            draw.line(
                [(sx, sy), (lx, ly)],
                fill=colour + (int(40 + 170 * weight),),
                width=max(1, int(1 + 3 * weight)),
            )
        img.alpha_composite(layer)

    def loudspeakers(self, img, plan: Plan, r: float) -> None:
        """Each loudspeaker as a dot with a see-through cone the way it aims (the band's
        aimed sources too); the subwoofers as squares, the band as diamonds."""
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        cones = ImageDraw.Draw(layer)
        for source in self.sources:
            what = kind(source["name"])
            if what == "sub" or (what == "band" and "orientation" not in source):
                continue
            x, y = plan(*source["position"][:2])
            yaw = math.radians(source.get("orientation", {}).get("yaw", 0.0))
            fx, fy = plan.vec(math.cos(yaw), math.sin(yaw))
            cone(cones, (x, y), (fx, fy), r * 5.0, 28.0, COLOURS[what] + (110,))
        img.alpha_composite(layer)
        draw = ImageDraw.Draw(img)
        for source in self.sources:
            x, y = plan(*source["position"][:2])
            what = kind(source["name"])
            colour = COLOURS[what]
            if what == "band":
                d = r * 1.1
                draw.polygon(
                    [(x, y - d), (x + d, y), (x, y + d), (x - d, y)], fill=colour, outline=BG
                )
            elif what == "sub":
                draw.rectangle([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=colour, outline=BG)
            else:
                d = r * 0.85
                draw.ellipse([x - d, y - d, x + d, y + d], fill=colour, outline=BG, width=2)

    def people(self, img, plan: Plan, current: str | None, r: float, close: bool) -> None:
        """The other listeners as rings with an arrow the way they face; the current one
        as a head."""
        draw = ImageDraw.Draw(img)
        for name, item in self.listeners.items():
            if name == current:
                continue
            x, y = plan(*item["position"][:2])
            if close and not self.on_stage(name):
                continue
            ring = r * 0.55
            draw.ellipse([x - ring, y - ring, x + ring, y + ring], outline=MUTED, width=2)
            (fx, fy), _ = self.axes(plan, item)
            arrow(draw, (x + fx * ring, y + fy * ring), (fx, fy), r * 2.2, r * 0.9, MUTED, 2)
        if current is not None and (not close or self.on_stage(current)):
            self.head(img, plan, self.listeners[current], r * (1.6 if close else 2.0))

    def on_stage(self, name: str) -> bool:
        x, y, z = self.listeners[name]["position"]
        return -58.0 <= x <= -42.0 and abs(y) <= 16.0 and z > 2.0

    @staticmethod
    def axes(plan: Plan, item: dict) -> tuple[tuple[float, float], tuple[float, float]]:
        """The way ``item`` faces and its left, as unit vectors on the picture: yaw turns
        from +x towards +y, so the left ear points to (-sin yaw, cos yaw)."""
        yaw = math.radians(item.get("orientation", {}).get("yaw", 0.0))
        return plan.vec(math.cos(yaw), math.sin(yaw)), plan.vec(-math.sin(yaw), math.cos(yaw))

    def head(self, img, plan: Plan, item: dict, r: float) -> None:
        """The listener: a head with a nose and an arrow the way it faces, and its ears
        marked L and R."""
        hx, hy = plan(*item["position"][:2])
        glow(img, [(hx, hy, 3 * r, LISTENER + (110,))], r)
        (fx, fy), (lx, ly) = self.axes(plan, item)
        draw = ImageDraw.Draw(img)
        width = max(3, round(r / 3))
        start = (hx + fx * r, hy + fy * r)
        arrow(draw, start, (fx, fy), r * 3.4, r * 1.2, BG, width + 4)  # an outline, over the rays
        arrow(draw, start, (fx, fy), r * 3.2, r * 1.0, LISTENER, width)
        nose = [
            (hx + fx * r * 1.45, hy + fy * r * 1.45),
            (hx + fx * r * 0.8 + lx * r * 0.45, hy + fy * r * 0.8 + ly * r * 0.45),
            (hx + fx * r * 0.8 - lx * r * 0.45, hy + fy * r * 0.8 - ly * r * 0.45),
        ]
        draw.polygon(nose, fill=LISTENER)
        for side in (1, -1):
            ex, ey = hx + lx * r * 0.9 * side, hy + ly * r * 0.9 * side
            draw.ellipse(
                [ex - r * 0.38, ey - r * 0.38, ex + r * 0.38, ey + r * 0.38],
                fill=LISTENER,
                outline=BG,
            )
        draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=LISTENER, outline=BG, width=2)
        letters = font("Bold", max(13, round(r * 1.1)))
        for side, letter in ((1, "L"), (-1, "R")):
            px, py = hx + lx * r * 2.0 * side, hy + ly * r * 2.0 * side
            draw.text(
                (px, py),
                letter,
                font=letters,
                fill=TEXT,
                anchor="mm",
                stroke_width=3,
                stroke_fill=BG,
            )

    def labels(self, img, plan: Plan) -> None:
        draw = ImageDraw.Draw(img)
        small, colour = font("SemiBold", 17), (128, 140, 172)
        for (x, y), label in [
            ((0, 72), "NORTH STAND"),
            ((0, -72), "SOUTH STAND"),
            ((90, 0), "EAST STAND"),
            ((-90, 0), "WEST STAND (EMPTY)"),
            ((7, 30), "PITCH · STANDING CROWD"),
        ]:
            px, py = plan(x, y)
            if "EAST" in label or "WEST" in label:
                layer = Image.new("RGBA", (300, 30), (0, 0, 0, 0))
                text(ImageDraw.Draw(layer), (150, 15), label, small, colour + (255,), "mm")
                layer = layer.rotate(90 if "WEST" in label else -90, expand=True)
                img.alpha_composite(layer, (int(px - layer.width / 2), int(py - layer.height / 2)))
            else:
                text(draw, (px, py), label, small, colour, "mm")
        x0, y0 = plan(self.box[0], self.box[1])
        bar = 50 * plan.s
        draw.line([(x0 + 12, y0 - 14), (x0 + 12 + bar, y0 - 14)], fill=MUTED, width=3)
        text(draw, (x0 + 12 + bar / 2, y0 - 20), "50 m", font("Medium", 16), MUTED, "md")

    def close_up(self, img, left: int, top: int, width: int, listener: str) -> None:
        """The stage from the pitch, with the monitors and the players."""
        draw = ImageDraw.Draw(img)
        text(draw, (left, top), "The stage, seen from the pitch", font("SemiBold", 22), MUTED)
        _x0, y0, _x1, y1 = STAGE_BOX
        scale = width / (y1 - y0)
        plan = Plan(left, top + 40, scale, STAGE_BOX, audience=True)
        w, h = plan.size
        frame = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(frame).rounded_rectangle(
            [left - 8, top + 32, left + w + 8, top + 48 + h], 10, PANEL
        )
        img.alpha_composite(frame)
        crop = Image.new("RGBA", img.size, (0, 0, 0, 0))
        self.stadium(crop, plan, listener, close=True)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [left - 4, top + 36, left + w + 4, top + 44 + h], 8, 255
        )
        img.paste(crop, (0, 0), Image.composite(crop.split()[3], mask, mask))
        draw = ImageDraw.Draw(img)
        text(
            draw,
            (left + w / 2, top + 52 + h),
            "front of the stage · audience this way ↓",
            font("Medium", 16),
            STAGE_EDGE,
            "ma",
        )

    def legend(self, draw, x: int, y: int) -> None:
        f = font("Medium", 18)
        for colour, label, step in ((PA, "sound system", 205), (MONITOR, "monitors", 160)):
            cone(draw, (x + 5, y + 7), (1.0, 0.0), 22, 28.0, tuple(c // 2 for c in colour))
            draw.ellipse([x, y + 2, x + 10, y + 12], fill=colour)
            text(draw, (x + 32, y - 2), label, f, MUTED)
            x += step
        draw.polygon([(x + 7, y), (x + 14, y + 7), (x + 7, y + 14), (x, y + 7)], fill=PLAYER)
        text(draw, (x + 24, y - 2), "band", f, MUTED)
        x += 95
        draw.ellipse([x, y, x + 14, y + 14], fill=LISTENER)
        arrow(draw, (x + 14, y + 7), (1.0, 0.0), 14, 7, LISTENER, 2)
        text(draw, (x + 36, y - 2), "you, facing", f, MUTED)
        x += 150
        draw.ellipse([x, y, x + 14, y + 14], outline=MUTED, width=2)
        arrow(draw, (x + 14, y + 7), (1.0, 0.0), 10, 6, MUTED, 2)
        text(draw, (x + 32, y - 2), "other places", f, MUTED)
        x += 180
        draw.line([(x, y + 7), (x + 26, y + 7)], fill=PA + (255,), width=3)
        text(draw, (x + 36, y - 2), "brighter: louder here", f, MUTED)

    def still(self, k: int) -> Image.Image:
        listener, _start, _stop = self.tour[k]
        img = Image.new("RGBA", (1920, 1080), BG)
        draw = ImageDraw.Draw(img)
        text(draw, (80, 44), "A concert in a stadium, simulated", font("Bold", 50, display=True))
        subtitle = (
            "The stadium example's sound system, stage monitors and band, "
            "heard binaurally at its listeners"
        )
        text(draw, (80, 112), subtitle, font("Regular", 26), MUTED)
        width, height = 860, 660
        scale = min(width / (self.box[2] - self.box[0]), height / (self.box[3] - self.box[1]))
        plan = Plan(80, 172, scale, self.box)
        self.stadium(img, plan, listener)
        self.labels(img, plan)
        draw = ImageDraw.Draw(img)
        self.legend(draw, 80, 172 + int(plan.size[1]) + 18)

        x = 1010
        pill, badge = font("SemiBold", 24), "BINAURAL  ·  USE HEADPHONES"
        draw.rounded_rectangle(
            [x, 172, x + draw.textlength(badge, font=pill) + 40, 218], 23, LISTENER
        )
        text(draw, (x + 20, 195), badge, pill, BG, "lm")
        title, where = self.title(listener)
        text(draw, (x, 244), title, font("Bold", 44, display=True))
        text(draw, (x, 302), where, font("Regular", 26), MUTED)
        seat = self.seats.get(listener, {})
        here = sorted(
            ((v, s) for (s, lis), v in self.levels.items() if lis == listener), reverse=True
        )
        names = ", ".join(s.replace("_", " ") for _, s in here[:3])
        relative = seat.get("relative_db", 0.0)
        facts = [
            f"The sound system reaches you {seat.get('lag_s', 0.0) * 1e3:.0f} ms after the desk",
            "The loudest place of the tour"
            if relative > -0.05
            else f"{relative:.1f} dB(A) against the loudest place of the tour",
            f"Loudest here: {names}",
            ("Facing the audience" if self.on_stage(listener) else "Facing the stage")
            + " · L and R mark your ears",
        ]
        if self.gains.get(listener, 0.0) > 0.5:
            facts.insert(2, f"Turned up {self.gains[listener]:.0f} dB here so you can hear it")
        y = 352
        for line in facts:
            text(draw, (x, y), line, font("Regular", 24), TEXT)
            y += 36
        self.close_up(img, x, y + 22, 400, listener)

        tx, ty = 1470, y + 22
        draw = ImageDraw.Draw(img)
        text(draw, (tx, ty), "The tour", font("SemiBold", 22), MUTED)
        ty += 36
        row = min(25, (860 - ty) // max(1, len(self.tour)))
        for j, (name, t0, _) in enumerate(self.tour):
            colour = LISTENER if j == k else (MUTED if j < k else FAINT)
            weight = "SemiBold" if j == k else "Regular"
            mark = "▶ " if j == k else "   "
            text(draw, (tx, ty), mark + self.title(name)[0], font(weight, 19), colour)
            text(draw, (1840, ty), mmss(t0), font(weight, 19), colour, "ra")
            ty += row

        draw.rounded_rectangle([80, 885, 1840, 995], 14, PANEL)  # the spectrogram goes here
        text(draw, (80, 1016), CREDIT, font("Regular", 19), MUTED)
        text(draw, (80, 1046), self.tool, font("Regular", 17), FAINT)
        return img

    def thumbnail(self) -> Image.Image:
        img = Image.new("RGBA", (1280, 720), BG)
        scale = min(700 / (self.box[2] - self.box[0]), 580 / (self.box[3] - self.box[1]))
        plan = Plan(540, 70, scale, self.box)
        self.stadium(img, plan, "mix_position" if "mix_position" in self.listeners else None)
        draw = ImageDraw.Draw(img)
        text(draw, (50, 70), "Stadium", font("Black", 92, display=True))
        text(draw, (50, 180), "concert,", font("Bold", 56, display=True), MUTED)
        text(draw, (50, 244), "simulated", font("Bold", 56, display=True), MUTED)
        pill = font("Bold", 34)
        draw.rounded_rectangle(
            [50, 340, 98 + draw.textlength("BINAURAL", font=pill), 400], 30, LISTENER
        )
        text(draw, (74, 370), "BINAURAL", pill, BG, "lm")
        text(
            draw,
            (50, 430),
            f"{len(self.listeners)} places, one song",
            font("SemiBold", 32),
            LISTENER,
        )
        text(draw, (50, 655), "sound system, monitors, band", font("Medium", 26), MUTED)
        return img
