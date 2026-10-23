"""Pictures for a video of the orchestra.

A cover with the hall and a seat, the seating on stage, and a thumbnail. Needs Pillow
(which matplotlib, and so the viz extra, brings).

    python make_images.py [--seat stalls-middle] [--movement 1] [--out .]

The cover (1920 x 1080) leaves a band at its bottom for the spectrogram make_video.sh
draws; the thumbnail is 1280 x 720, as YouTube wants it. Inter is used when installed,
else DejaVu Sans.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
HALL_X, HALL_Y, STAGE_X = 32.0, 20.0, 8.0  # the concert hall: 32 x 20 m, stage 8 m deep
CONDUCTOR = (7.3, 10.0)
MOVEMENTS = {
    1: "I. Allegro vivace e con brio",
    2: "II. Allegretto scherzando",
    4: "IV. Allegro vivace",
}

BG = (13, 17, 28)
PANEL = (21, 27, 42)
LINE = (58, 68, 92)
TEXT = (232, 236, 244)
MUTED = (148, 158, 182)
FAINT = (110, 120, 146)
STAGE = (58, 44, 34)
STAGE_EDGE = (120, 90, 64)
SEATS = (30, 38, 58)
LISTENER = (102, 217, 239)

SECTIONS = [  # source name prefix, label, colour
    ("vl1_", "Violins I", (246, 195, 91)),
    ("vl2_", "Violins II", (245, 152, 87)),
    ("vla_", "Violas", (232, 109, 96)),
    ("vc_", "Cellos", (214, 94, 156)),
    ("db_", "Double basses", (157, 110, 222)),
    ("fl", "Flutes", (99, 196, 238)),
    ("ob", "Oboes", (88, 168, 238)),
    ("cl", "Clarinets", (96, 140, 240)),
    ("bn", "Bassoons", (126, 120, 236)),
    ("hn", "Horns", (110, 214, 146)),
    ("tp", "Trumpets", (196, 226, 104)),
    ("timpani", "Timpani", (236, 236, 236)),
]
LABELS = {  # where the seating picture writes each section's name, scene x and y
    "Violins I": (7.05, 6.3),
    "Violins II": (7.05, 13.25),
    "Cellos": (4.25, 7.45),
    "Double basses": (3.35, 6.05),
    "Violas": (3.3, 12.4),
    "Flutes": (3.42, 9.1),
    "Oboes": (3.42, 10.9),
    "Clarinets": (1.55, 9.1),
    "Bassoons": (1.55, 10.9),
    "Horns": (1.55, 6.45),
    "Trumpets": (1.55, 13.55),
    "Timpani": (0.55, 14.6),
}
CREDIT = (
    "Anechoic recording: Orchester Wiener Akademie, cond. Martin Haselböck · TU Berlin "
    "(Böhm, Ackermann, Weinzierl), CC BY 4.0, doi.org/10.14279/depositonce-6729.2"
)
TOOL = (
    "Room simulated with acoustic_spaces: ray tracing for early reflections, stochastic ray "
    "tracing for the late tail"
)

_FONTS = {}


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


def section(name: str) -> tuple[str, tuple[int, int, int]]:
    for prefix, label, colour in SECTIONS:
        if name.startswith(prefix):
            return label, colour
    raise ValueError(f"no section for source {name!r}")


def colour_of(label: str) -> tuple[int, int, int]:
    return next(colour for _, name, colour in SECTIONS if name == label)


class Plan:
    """Scene x and y to pixels: from above with the stage on the left, or (rotated) the
    stage as the audience sees it, its front at the bottom."""

    def __init__(self, left, top, scale, x0=0.0, y0=0.0, y1=HALL_Y, rotated=False):
        self.left, self.top, self.s = left, top, scale
        self.x0, self.y0, self.y1, self.rotated = x0, y0, y1, rotated

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        if self.rotated:
            return (self.left + (y - self.y0) * self.s, self.top + (x - self.x0) * self.s)
        return (self.left + (x - self.x0) * self.s, self.top + (self.y1 - y) * self.s)

    def vec(self, dx: float, dy: float) -> tuple[float, float]:
        return (dy, dx) if self.rotated else (dx, -dy)

    def box(self, x0, y0, x1, y1) -> list[float]:
        (a, b), (c, d) = self(x0, y1), self(x1, y0)
        return [a, b, c, d]


def text(draw, xy, s, f, fill=TEXT, anchor="la") -> None:
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def glow(img: Image.Image, shapes, blur: float) -> None:
    """Blurred discs, ``shapes`` as (x, y, radius, rgba), under what comes next."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, r, rgba in shapes:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)))


def wall_hits(s, listener) -> list[tuple[float, float]]:
    """Where the first-order reflections off the four walls meet them, seen from above."""
    (sx, sy), (lx, ly) = s, listener
    return [
        (0.0, sy + (ly - sy) * sx / (sx + lx)),
        (HALL_X, sy + (ly - sy) * (HALL_X - sx) / (2 * HALL_X - sx - lx)),
        (sx + (lx - sx) * sy / (sy + ly), 0.0),
        (sx + (lx - sx) * (HALL_Y - sy) / (2 * HALL_Y - sy - ly), HALL_Y),
    ]


class Pictures:
    def __init__(self, scene: dict, seat: str, movement: int, facts: list[str] | None = None):
        self.facts = facts or []  # lines under the seat on the cover (orchestra_demo.py)
        self.players = [
            (s["name"], s["position"], s.get("orientation", {})) for s in scene["sources"]
        ]
        listeners = {item["name"]: item for item in scene["listeners"]}
        if seat not in listeners:
            raise SystemExit(f"no seat {seat!r}; the scene has {', '.join(listeners)}")
        self.listener = listeners[seat]
        self.seat = "on the conductor's podium" if seat == "conductor" else seat.replace("-", ", ")
        self.movement = MOVEMENTS[movement]
        self.counts: dict[str, int] = {}
        for name, _, _ in self.players:
            label = section(name)[0]
            self.counts[label] = self.counts.get(label, 0) + 1

    def players_on(self, img, plan, r, halo=True) -> None:
        if halo:
            shapes = []
            for name, (x, y, _), _ in self.players:
                shapes.append((*plan(x, y), r * 2.6, section(name)[1] + (110,)))
            glow(img, shapes, r * 1.6)
        draw = ImageDraw.Draw(img)
        for name, (x, y, _), orientation in self.players:
            cx, cy = plan(x, y)
            colour = section(name)[1]
            yaw = math.radians(orientation.get("yaw", 0.0))
            vx, vy = plan.vec(math.cos(yaw), math.sin(yaw))
            draw.line(
                [(cx, cy), (cx + vx * r * 1.9, cy + vy * r * 1.9)],
                fill=colour,
                width=max(2, r // 3),
            )
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour, outline=BG, width=2)
        cx, cy = plan(*CONDUCTOR)
        draw.regular_polygon((cx, cy, r * 1.3), 4, rotation=45, fill=TEXT, outline=BG)

    def hall(self, img, plan, r, labels=True) -> None:
        draw = ImageDraw.Draw(img)
        draw.rectangle(plan.box(0, 0, HALL_X, HALL_Y), fill=PANEL)
        x = 9.4
        while x < HALL_X - 0.6:  # rows of seats either side of the aisle
            for y0, y1 in ((0.9, 8.9), (11.1, 19.1)):
                draw.line([plan(x, y0), plan(x, y1)], fill=SEATS, width=max(2, int(plan.s * 0.32)))
            x += 0.9
        draw.rectangle(plan.box(0, 0, STAGE_X, HALL_Y), fill=STAGE)
        draw.line([plan(STAGE_X, 0), plan(STAGE_X, HALL_Y)], fill=STAGE_EDGE, width=3)
        draw.rectangle(plan.box(0, 0, HALL_X, HALL_Y), outline=LINE, width=3)
        # The direct sound and the first reflections off the walls, leader to seat.
        source = next((p for n, p, _ in self.players if n == "vl1_01"), self.players[0][1])[:2]
        seat = tuple(self.listener["position"][:2])
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        lines = ImageDraw.Draw(layer)
        for hit in wall_hits(source, seat):
            lines.line([plan(*source), plan(*hit), plan(*seat)], fill=LISTENER + (90,), width=2)
        lines.line([plan(*source), plan(*seat)], fill=LISTENER + (200,), width=3)
        img.alpha_composite(layer)
        self.players_on(img, plan, r)
        self.head(img, plan, r * 1.9)
        if labels:
            draw = ImageDraw.Draw(img)
            hx, hy = plan(*seat)
            text(
                draw,
                (hx, hy + r * 3),
                "your seat",
                font("SemiBold", int(plan.s * 0.62)),
                LISTENER,
                "ma",
            )
            small = font("SemiBold", int(plan.s * 0.5))
            text(draw, plan(STAGE_X / 2, HALL_Y - 0.5), "STAGE", small, STAGE_EDGE, "ma")
            text(draw, plan((STAGE_X + HALL_X) / 2, HALL_Y - 0.5), "STALLS", small, LINE, "ma")

    def head(self, img, plan, r) -> None:
        """The listener: a head with ears and a nose the way it faces."""
        hx, hy = plan(*self.listener["position"][:2])
        glow(img, [(hx, hy, 3 * r, LISTENER + (90,))], r)
        yaw = math.radians(self.listener.get("orientation", {}).get("yaw", 0.0))
        fx, fy = plan.vec(math.cos(yaw), math.sin(yaw))
        sx, sy = -fy, fx  # sideways
        draw = ImageDraw.Draw(img)
        nose = [
            (hx + fx * r * 1.45, hy + fy * r * 1.45),
            (hx + fx * r * 0.8 + sx * r * 0.45, hy + fy * r * 0.8 + sy * r * 0.45),
            (hx + fx * r * 0.8 - sx * r * 0.45, hy + fy * r * 0.8 - sy * r * 0.45),
        ]
        draw.polygon(nose, fill=LISTENER)
        for side in (1, -1):
            ex, ey = hx + sx * r * 0.85 * side, hy + sy * r * 0.85 * side
            draw.ellipse(
                [ex - r * 0.35, ey - r * 0.35, ex + r * 0.35, ey + r * 0.35], fill=LISTENER
            )
        draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=LISTENER, outline=BG, width=2)

    def legend(self, draw, x, y, size, gap) -> int:
        for _, label, colour in SECTIONS:
            draw.ellipse([x, y + 4, x + size - 6, y + size - 2], fill=colour)
            text(draw, (x + size + 8, y), label, font("Medium", size - 2))
            text(
                draw,
                (x + 300, y),
                str(self.counts.get(label, 0)),
                font("Regular", size - 2),
                MUTED,
                "ra",
            )
            y += gap
        return y

    def cover(self) -> Image.Image:
        img = Image.new("RGBA", (1920, 1080), BG)
        draw = ImageDraw.Draw(img)
        title = "Beethoven · Symphony No. 8 in F major, op. 93"
        text(draw, (80, 60), title, font("Bold", 54, display=True))
        subtitle = (
            f"{self.movement}  ·  {len(self.players)} anechoic tracks in a simulated concert hall"
        )
        text(draw, (80, 132), subtitle, font("Regular", 28), MUTED)
        self.hall(img, Plan(80, 205, 33), 7)
        draw = ImageDraw.Draw(img)
        x, pill = 1250, font("SemiBold", 26)
        badge = "BINAURAL  ·  USE HEADPHONES"
        draw.rounded_rectangle(
            [x, 205, x + draw.textlength(badge, font=pill) + 44, 255], 25, LISTENER
        )
        text(draw, (x + 22, 230), badge, pill, BG, "lm")
        text(draw, (x, 290), f"Seat: {self.seat}", font("SemiBold", 30))
        text(
            draw,
            (x, 330),
            f"German seating, {len(self.players)} players",
            font("Regular", 26),
            MUTED,
        )
        y = 372
        for line in self.facts:
            text(draw, (x, y), line, font("Regular", 24))
            y += 34
        self.legend(draw, x, y + 18, 26, 34)
        draw.rounded_rectangle([80, 885, 1840, 995], 14, PANEL)  # make_video.sh draws here
        text(draw, (80, 1018), CREDIT, font("Regular", 19), MUTED)
        text(draw, (80, 1046), TOOL, font("Regular", 17), FAINT)
        return img

    def seating(self) -> Image.Image:
        img = Image.new("RGBA", (1920, 1080), BG)
        draw = ImageDraw.Draw(img)
        text(draw, (80, 60), "The orchestra on stage", font("Bold", 54, display=True))
        subtitle = (
            "One source per anechoic track · German (antiphonal) seating · seen from the hall"
        )
        text(draw, (80, 132), subtitle, font("Regular", 28), MUTED)
        plan = Plan(80, 215, 78, y0=2.4, rotated=True)
        draw.rounded_rectangle([*plan(0.0, 2.4), *plan(8.0, 16.9)], 10, STAGE)
        draw.line([plan(8.0, 2.4), plan(8.0, 16.9)], fill=STAGE_EDGE, width=4)
        edge = "front of the stage  ·  audience this way  ↓"
        text(draw, plan(8.3, 9.65), edge, font("Medium", 22), STAGE_EDGE, "mm")
        self.players_on(img, plan, 13)
        draw = ImageDraw.Draw(img)
        cx, cy = plan(*CONDUCTOR)
        text(draw, (cx, cy + 22), "conductor", font("SemiBold", 21), anchor="ma")
        for label, (x, y) in LABELS.items():
            text(draw, plan(x, y), label, font("SemiBold", 21), colour_of(label), "mm")
        x = 1300
        text(draw, (x, 215), "Who plays where", font("SemiBold", 34))
        y = self.legend(draw, x, 275, 27, 38) + 22
        notes = [
            "Winds on 0.3 m risers, brass and",
            "timpani on 0.6 m. Horn bells point",
            "back, trumpet bells to the hall.",
            "Each player faces the conductor;",
            "directivity narrows above 1 kHz.",
            "Equal source power: the balance",
            "is the recording's own.",
        ]
        for line in notes:
            text(draw, (x, y), line, font("Regular", 23), MUTED)
            y += 32
        text(draw, (80, 1030), CREDIT, font("Regular", 19), MUTED)
        return img

    def thumbnail(self, note: str | None = None) -> Image.Image:
        img = Image.new("RGBA", (1280, 720), BG)
        self.hall(img, Plan(500, 172, 23.5), 5, labels=False)
        draw = ImageDraw.Draw(img)
        text(draw, (50, 60), "Beethoven 8", font("Black", 88, display=True))
        text(draw, (50, 165), "in a virtual", font("Bold", 52, display=True), MUTED)
        text(draw, (50, 225), "concert hall", font("Bold", 52, display=True), MUTED)
        pill = font("Bold", 34)
        draw.rounded_rectangle(
            [50, 320, 98 + draw.textlength("BINAURAL", font=pill), 380], 30, LISTENER
        )
        text(draw, (74, 350), "BINAURAL", pill, BG, "lm")
        text(draw, (50, 410), "use headphones", font("SemiBold", 34), LISTENER)
        tracks = note or f"{len(self.players)} anechoic tracks · simulated room"
        text(draw, (50, 655), tracks, font("Medium", 28), MUTED)
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", type=Path, default=HERE / "scene.yaml")
    parser.add_argument("--seat", default="stalls-middle", help="listener (default stalls-middle)")
    parser.add_argument("--movement", type=int, choices=sorted(MOVEMENTS), default=1)
    parser.add_argument("--out", type=Path, default=Path("."), help="folder for the PNG files")
    args = parser.parse_args()
    scene = yaml.safe_load(args.scene.read_text(encoding="utf-8"))
    pictures = Pictures(scene, args.seat, args.movement)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in ("cover", "seating", "thumbnail"):
        path = args.out / f"{name}.png"
        getattr(pictures, name)().convert("RGB").save(path, optimize=True)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
