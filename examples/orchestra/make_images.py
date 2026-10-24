"""Pictures for a video of the orchestra.

A cover with the hall and a seat, the seating on stage, and a thumbnail. The colours,
fonts and the frame round them come from :mod:`acoustic_spaces.demos.pictures`.

    python make_images.py [--seat stalls-middle] [--movement 1] [--out .]

The cover (1920 x 1080) leaves a band at its bottom for the spectrogram the video draws
over it; the thumbnail is 1280 x 720, as YouTube wants it.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from acoustic_spaces.demos.pictures import (
    BG,
    LINE,
    LISTENER,
    MARGIN,
    MUTED,
    PANEL,
    SIZE,
    TEXT,
    THUMBNAIL,
    Plan,
    axes,
    badge,
    canvas,
    facts,
    font,
    footer,
    glow,
    head,
    heading,
    save,
    text,
    thumbnail_badge,
    tool_line,
)

HERE = Path(__file__).resolve().parent
HALL_X, HALL_Y, STAGE_X = 32.0, 20.0, 8.0  # the concert hall: 32 x 20 m, stage 8 m deep
CONDUCTOR = (7.3, 10.0)
MOVEMENTS = {
    1: "I. Allegro vivace e con brio",
    2: "II. Allegretto scherzando",
    4: "IV. Allegro vivace",
}

STAGE = (58, 44, 34)
STAGE_EDGE = (120, 90, 64)
SEATS = (30, 38, 58)

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


def section(name: str) -> tuple[str, tuple[int, int, int]]:
    for prefix, label, colour in SECTIONS:
        if name.startswith(prefix):
            return label, colour
    raise ValueError(f"no section for source {name!r}")


def colour_of(label: str) -> tuple[int, int, int]:
    return next(colour for _, name, colour in SECTIONS if name == label)


def hall_plan(left, top, scale, *, y0=0.0, rotated=False) -> Plan:
    """The hall from above with the stage on the left, or (``rotated``) the stage as the
    audience sees it, its front at the bottom."""
    return Plan(left, top, scale, (0.0, y0, HALL_X, HALL_Y), rotated)


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
    def __init__(
        self,
        scene: dict,
        seat: str,
        movement: int,
        facts: list[str] | None = None,
        hrtf: str | Path | None = None,
    ):
        self.lines = facts or []  # lines under the seat on the cover (orchestra_demo.py)
        self.tool = tool_line("orchestra", scene, hrtf=hrtf)
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
        draw.rectangle(plan.rect(0, 0, HALL_X, HALL_Y), fill=PANEL)
        x = 9.4
        while x < HALL_X - 0.6:  # rows of seats either side of the aisle
            for y0, y1 in ((0.9, 8.9), (11.1, 19.1)):
                draw.line([plan(x, y0), plan(x, y1)], fill=SEATS, width=max(2, int(plan.s * 0.32)))
            x += 0.9
        draw.rectangle(plan.rect(0, 0, STAGE_X, HALL_Y), fill=STAGE)
        draw.line([plan(STAGE_X, 0), plan(STAGE_X, HALL_Y)], fill=STAGE_EDGE, width=3)
        draw.rectangle(plan.rect(0, 0, HALL_X, HALL_Y), outline=LINE, width=3)
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
        forward, left = axes(plan, self.listener)
        at = plan(*self.listener["position"][:2])
        head(img, at, forward, left, r, pointer=False, letters=False)

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
        img = canvas(SIZE)
        draw = ImageDraw.Draw(img)
        heading(
            draw,
            "Beethoven · Symphony No. 8 in F major, op. 93",
            f"{self.movement}  ·  {len(self.players)} anechoic tracks in a simulated concert hall",
            y=60,
        )
        self.hall(img, hall_plan(MARGIN, 205, 33), 7)
        draw = ImageDraw.Draw(img)
        x = 1250
        badge(draw, x, 205, size=26)
        text(draw, (x, 290), f"Seat: {self.seat}", font("SemiBold", 30))
        text(
            draw,
            (x, 330),
            f"German seating, {len(self.players)} players",
            font("Regular", 26),
            MUTED,
        )
        y = facts(draw, x, 372, self.lines, size=24, step=34)
        self.legend(draw, x, y + 18, 26, 34)
        footer(draw, CREDIT, self.tool)
        return img

    def seating(self) -> Image.Image:
        img = canvas(SIZE)
        draw = ImageDraw.Draw(img)
        heading(
            draw,
            "The orchestra on stage",
            "One source per anechoic track · German (antiphonal) seating · seen from the hall",
            y=60,
        )
        plan = hall_plan(MARGIN, 215, 78, y0=2.4, rotated=True)
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
        text(draw, (MARGIN, 1030), CREDIT, font("Regular", 19), MUTED)
        return img

    def thumbnail(self, note: str | None = None) -> Image.Image:
        img = canvas(THUMBNAIL)
        self.hall(img, hall_plan(500, 172, 23.5), 5, labels=False)
        draw = ImageDraw.Draw(img)
        text(draw, (50, 60), "Beethoven 8", font("Black", 88, display=True))
        text(draw, (50, 165), "in a virtual", font("Bold", 52, display=True), MUTED)
        text(draw, (50, 225), "concert hall", font("Bold", 52, display=True), MUTED)
        thumbnail_badge(draw, 50, 320)
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
        save(getattr(pictures, name)(), path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()


__all__ = ["CREDIT", "Pictures", "hall_plan", "save"]
