"""Stills for church_demo.py's video: the church from above and from the side with the
choir, the organ, the listener of the moment and how loud each source reaches them, the
choir close up, and the tour. Needs Pillow (which the viz extra brings); Inter is used
when installed, else DejaVu Sans.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (13, 17, 28)
PANEL = (21, 27, 42)
LINE = (58, 68, 92)
TEXT = (232, 236, 244)
MUTED = (148, 158, 182)
FAINT = (96, 106, 132)
FLOOR = (34, 38, 52)
PEWS = (52, 46, 40)
WALL = (120, 128, 150)
ALTAR = (190, 160, 110)
ORGAN = (255, 166, 77)
WOMEN = (186, 236, 120)
MEN = (110, 190, 90)
LISTENER = (102, 217, 239)

#: Where make_video draws the spectrogram: left, top, width, height.
SPECTRUM_BOX = (100, 892, 1720, 96)
CREDIT = (
    "Singers: VocalSet (Wilkins, Seetharaman, Wahl, Pardo 2018), CC BY 4.0 · "
    "Organ: VCSL pipe organ (Ivy Audio, Versilian Studios), CC0 · "
    '"Dona nobis pacem", traditional; organ harmonisation for this demo'
)
CHOIR_BOX = (28.9, -3.6, 33.0, 3.6)  # x0, y0, x1, y1 of the close-up

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
    either side: the way a source faces."""
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


class View:
    """Scene coordinates to pixels. ``plan``: from above, the altar (+x) to the right and
    north (+y) up. ``section``: from the south, x to the right and height up. ``nave``:
    from above as someone in the nave sees it, the altar at the top and north (their
    left) on the left."""

    def __init__(self, kind, left, top, scale, box):
        self.kind, self.left, self.top, self.s = kind, left, top, scale
        self.x0, self.y0, self.x1, self.y1 = box  # y is the height in a section

    def __call__(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        if self.kind == "section":
            return (self.left + (x - self.x0) * self.s, self.top + (self.y1 - z) * self.s)
        if self.kind == "nave":
            return (self.left + (self.y1 - y) * self.s, self.top + (self.x1 - x) * self.s)
        return (self.left + (x - self.x0) * self.s, self.top + (self.y1 - y) * self.s)

    def vec(self, dx: float, dy: float) -> tuple[float, float]:
        """A horizontal direction on the picture (unit length in a section only when
        along x)."""
        if self.kind == "section":
            return (dx, 0.0)
        if self.kind == "nave":
            return (-dy, -dx)
        return (dx, -dy)

    @property
    def size(self) -> tuple[float, float]:
        w, h = (self.x1 - self.x0) * self.s, (self.y1 - self.y0) * self.s
        return (h, w) if self.kind == "nave" else (w, h)


def kind(name: str) -> str:
    if name.startswith("organ"):
        return "organ"
    return "women" if name.startswith("singer_f") else "men"


COLOURS = {"organ": ORGAN, "women": WOMEN, "men": MEN}


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def mean_db(values) -> float:
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in values) / len(values))


class Pictures:
    def __init__(self, scene: dict, report: dict, seats, tour, titles, what_plays, playing, voices):
        self.scene, self.seats, self.tour, self.titles, self.what_plays = (
            scene,
            seats,
            tour,
            titles,
            what_plays,
        )
        self.playing, self.voices = playing, voices
        self.now = ("choir", "organ")  # what plays in the picture being drawn
        self.sources = scene["sources"]
        self.listeners = {item["name"]: item for item in scene["listeners"]}
        self.plan_xy = [tuple(p) for p in scene["geometry"]["floor_plan"]]
        self.height = float(scene["geometry"]["height"])
        xs, ys = [p[0] for p in self.plan_xy], [p[1] for p in self.plan_xy]
        self.box = (min(xs), min(ys), max(xs), max(ys))
        self.levels = {(e["source"], e["listener"]): e["spl_a_db"] for e in report["responses"]}
        bands = report.get("bands_hz", [63, 125, 250, 500, 1000, 2000, 4000, 8000])
        mid = [k for k, f in enumerate(bands) if 500 <= f <= 2000]
        self.c80 = {
            (e["source"], e["listener"]): sum(e["parameters"]["c80"][k] for k in mid) / len(mid)
            for e in report["responses"]
        }
        early = scene.get("simulation", {}).get("early", {})
        order = early.get("max_order")
        self.tool = (
            "Simulated with acoustic_spaces (church example): early reflections"
            + (f" to order {order}" if order else "")
            + ", stochastic ray tracing for the late sound, binaural through a spherical head"
        )

    def title(self, listener: str) -> tuple[str, str]:
        return self.titles.get(listener, (listener.replace("_", " ").capitalize(), ""))

    # ------------------------------------------------------------------ drawing

    def church(self, img, view: View, listener: str | None, r: float) -> None:
        """The church in ``view`` with the sources, the rays to ``listener`` and the
        listeners."""
        draw = ImageDraw.Draw(img)
        if view.kind == "section":
            x0, x1 = self.box[0], self.box[2]
            outline = [
                view(x0, 0, 0),
                view(x1, 0, 0),
                view(x1, 0, self.height),
                view(x0, 0, self.height),
            ]
            draw.polygon(outline, fill=FLOOR)
            draw.line(outline + outline[:1], fill=WALL, width=3)
            # The pews, as a band over the floor; the altar.
            draw.rectangle([*view(4.0, 0, 1.0), *view(28.0, 0, 0)], fill=PEWS)
            draw.rectangle([*view(34.6, 0, 1.0), *view(35.6, 0, 0)], fill=ALTAR)
            # Where the organ case stands (the gallery's floor is not in the model).
            draw.rectangle(
                [*view(0.0, 0, 13.0), *view(1.8, 0, 7.6)], outline=ORGAN + (255,), width=2
            )
        else:
            outline = [view(x, y) for x, y in self.plan_xy]
            draw.polygon(outline, fill=FLOOR)
            for y0, y1 in ((-6.2, -0.9), (0.9, 6.2)):  # pew blocks, for the eye
                for x in [4.0 + 1.0 * k for k in range(24)]:
                    draw.line(
                        [view(x, y0), view(x, y1)], fill=PEWS, width=max(2, round(view.s * 0.35))
                    )
            draw.line(outline + outline[:1], fill=WALL, width=3)
            a, b = view(34.6, -1.2), view(35.6, 1.2)
            draw.rectangle(
                [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], fill=ALTAR
            )
            a, b = view(0.0, -4.6), view(1.8, 4.6)
            draw.rectangle(
                [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])],
                outline=ORGAN,
                width=2,
            )
        if listener is not None:
            self.rays(img, view, listener)
        self.draw_sources(img, view, r)
        self.people(img, view, listener, r)

    def rays(self, img, view: View, listener: str) -> None:
        """A line from each source to the listener, brighter the louder it is there
        (over the 24 dB below the loudest)."""
        here = {
            s: v
            for (s, lis), v in self.levels.items()
            if lis == listener and ("organ" if s.startswith("organ") else "choir") in self.now
        }
        if not here:
            return
        top = max(here.values())
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        lx, ly = view(*self.listeners[listener]["position"])
        for source in sorted(self.sources, key=lambda s: here.get(s["name"], -999)):
            weight = 1.0 + (here.get(source["name"], -999) - top) / 24.0
            if weight <= 0:
                continue
            sx, sy = view(*source["position"])
            colour = COLOURS[kind(source["name"])]
            draw.line(
                [(sx, sy), (lx, ly)],
                fill=colour + (int(30 + 150 * weight),),
                width=max(1, int(1 + 2 * weight)),
            )
        img.alpha_composite(layer)

    def draw_sources(self, img, view: View, r: float) -> None:
        """Each source as a dot with a see-through cone the way it faces: the singers
        towards the nave, the organ's pipes out of the gallery."""
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        cones = ImageDraw.Draw(layer)
        for source in self.sources:
            what = kind(source["name"])
            x, y = view(*source["position"])
            yaw = math.radians(source.get("orientation", {}).get("yaw", 0.0))
            fx, fy = view.vec(math.cos(yaw), math.sin(yaw))
            size = r * (4.0 if what == "organ" else 2.6)
            cone(
                cones,
                (x, y),
                (fx, fy),
                size,
                30.0,
                COLOURS[what] + (70 if what != "organ" else 110,),
            )
        img.alpha_composite(layer)
        draw = ImageDraw.Draw(img)
        for source in sorted(self.sources, key=lambda s: s["position"][2]):
            what = kind(source["name"])
            x, y = view(*source["position"])
            d = r * (0.95 if what == "organ" else 0.6)
            if what == "organ":
                draw.rectangle([x - d * 0.7, y - d, x + d * 0.7, y + d], fill=ORGAN, outline=BG)
            else:
                draw.ellipse([x - d, y - d, x + d, y + d], fill=COLOURS[what], outline=BG)

    def people(self, img, view: View, current: str | None, r: float) -> None:
        """The other listeners as rings with an arrow the way they face; the current one
        as a head."""
        draw = ImageDraw.Draw(img)
        for name, item in self.listeners.items():
            if name == current:
                continue
            x, y = view(*item["position"])
            ring = r * 0.7
            draw.ellipse([x - ring, y - ring, x + ring, y + ring], outline=MUTED, width=2)
            (fx, fy), _ = self.axes(view, item)
            arrow(draw, (x + fx * ring, y + fy * ring), (fx, fy), r * 2.2, r * 0.9, MUTED, 2)
        if current is not None:
            self.head(img, view, self.listeners[current], r * 1.5)

    @staticmethod
    def axes(view: View, item: dict) -> tuple[tuple[float, float], tuple[float, float]]:
        """The way ``item`` faces and its left, as unit vectors on the picture: yaw turns
        from +x towards +y, so the left ear points to (-sin yaw, cos yaw)."""
        yaw = math.radians(item.get("orientation", {}).get("yaw", 0.0))
        return view.vec(math.cos(yaw), math.sin(yaw)), view.vec(-math.sin(yaw), math.cos(yaw))

    def head(self, img, view: View, item: dict, r: float) -> None:
        """The listener: a head with a nose and an arrow the way it faces, and, seen from
        above, its ears marked L and R."""
        hx, hy = view(*item["position"])
        glow(img, [(hx, hy, 3 * r, LISTENER + (110,))], r)
        (fx, fy), (lx, ly) = self.axes(view, item)
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
        if view.kind != "section":
            draw.polygon(nose, fill=LISTENER)
            for side in (1, -1):
                ex, ey = hx + lx * r * 0.9 * side, hy + ly * r * 0.9 * side
                draw.ellipse(
                    [ex - r * 0.38, ey - r * 0.38, ex + r * 0.38, ey + r * 0.38],
                    fill=LISTENER,
                    outline=BG,
                )
        draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=LISTENER, outline=BG, width=2)
        if view.kind != "section":
            letters = font("Bold", max(13, round(r * 1.1)))
            for side, letter in ((1, "L"), (-1, "R")):
                px, py = hx + lx * r * 2.1 * side, hy + ly * r * 2.1 * side
                draw.text(
                    (px, py),
                    letter,
                    font=letters,
                    fill=TEXT,
                    anchor="mm",
                    stroke_width=3,
                    stroke_fill=BG,
                )

    def labels(self, img, plan: View, section: View) -> None:
        draw = ImageDraw.Draw(img)
        small, colour = font("SemiBold", 16), (128, 140, 172)
        text(draw, plan(37.4, 0.0), "APSE", small, colour, "mm")
        text(draw, plan(16.0, 0.0), "NAVE", small, colour, "mm")
        x, y = plan(0.0, 7.0)
        text(
            draw,
            (x, y - 10),
            "FROM ABOVE · THE ENTRANCE ON THE LEFT, THE ALTAR ON THE RIGHT",
            small,
            colour,
            "ld",
        )
        text(draw, plan(35.1, -2.1), "ALTAR", font("SemiBold", 14), ALTAR, "mm")
        text(draw, plan(0.25, -5.75), "ORGAN GALLERY", font("SemiBold", 14), ORGAN, "lm")
        x, y = section(0.0, 0.0, self.height)
        text(draw, (x, y - 10), "FROM THE SIDE", small, colour, "ld")
        text(
            draw,
            section(2.6, 0.0, 13.6),
            "organ case (the gallery floor is not modelled)",
            font("Medium", 15),
            ORGAN,
            "la",
        )
        text(
            draw,
            section(20.0, 0.0, 9.0),
            f"vault {self.height:.0f} m up",
            font("Medium", 15),
            colour,
            "mm",
        )
        bar = 10 * plan.s
        x0, y0 = plan(self.box[2], self.box[1])
        draw.line([(x0 - bar, y0 + 16), (x0, y0 + 16)], fill=MUTED, width=3)
        text(draw, (x0 - bar / 2, y0 + 22), "10 m", font("Medium", 15), MUTED, "ma")

    def close_up(self, img, left: int, top: int, width: int, listener: str) -> None:
        """The choir from above, as the nave sees it: women in front, men behind."""
        draw = ImageDraw.Draw(img)
        text(draw, (left, top), "The choir, seen from the nave", font("SemiBold", 22), MUTED)
        scale = width / (CHOIR_BOX[3] - CHOIR_BOX[1])
        view = View("nave", left, top + 44, scale, CHOIR_BOX)
        w, h = view.size
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).rounded_rectangle(
            [left - 8, top + 34, left + w + 8, top + 54 + h], 10, PANEL
        )
        img.alpha_composite(layer)
        crop = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_crop = ImageDraw.Draw(crop)
        a, b = view(34.6, -1.2), view(35.6, 1.2)
        draw_crop.rectangle(
            [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])], fill=ALTAR
        )
        if listener in ("conductor", "in_the_choir"):
            self.rays(crop, view, listener)
        self.draw_sources(crop, view, 9)
        draw_crop = ImageDraw.Draw(crop)
        voice = {
            "soprano": "S",
            "mezzo-soprano": "Mz",
            "tenor": "T",
            "baritone": "Bar",
            "bass": "B",
            "bass-baritone": "BBar",
        }
        for source in self.sources:
            if source["name"].startswith("singer_"):
                x, y = view(*source["position"])
                part = voice.get(self.voices.get(source["name"].split("_", 1)[1], ""), "")
                if part:  # the front row's below its dots, the back row's above
                    front = source["name"].startswith("singer_f")
                    text(
                        draw_crop,
                        (x, y + 12 if front else y - 12),
                        part,
                        font("Medium", 13),
                        MUTED,
                        "ma" if front else "md",
                    )
        for name in ("conductor", "in_the_choir"):
            if name in self.listeners:
                item = self.listeners[name]
                if name == listener:
                    self.head(crop, view, item, 11)
                else:
                    x, y = view(*item["position"])
                    ImageDraw.Draw(crop).ellipse(
                        [x - 7, y - 7, x + 7, y + 7], outline=MUTED, width=2
                    )
                    (fx, fy), _ = self.axes(view, item)
                    arrow(ImageDraw.Draw(crop), (x + fx * 7, y + fy * 7), (fx, fy), 18, 8, MUTED, 2)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            [left - 4, top + 38, left + w + 4, top + 50 + h], 8, 255
        )
        img.paste(crop, (0, 0), Image.composite(crop.split()[3], mask, mask))
        draw = ImageDraw.Draw(img)
        text(
            draw,
            (left + w / 2, top + 58 + h),
            "the nave this way ↓ · north on the left",
            font("Medium", 15),
            FAINT,
            "ma",
        )

    def legend(self, draw, x: int, y: int) -> None:
        f = font("Medium", 17)
        for colour, label, step in ((WOMEN, "women", 112), (MEN, "men", 98)):
            draw.ellipse([x, y + 1, x + 12, y + 13], fill=colour)
            text(draw, (x + 20, y - 2), label, f, MUTED)
            x += step
        draw.rectangle([x + 2, y, x + 12, y + 14], fill=ORGAN)
        text(draw, (x + 20, y - 2), "organ pipes", f, MUTED)
        x += 135
        draw.ellipse([x, y, x + 14, y + 14], fill=LISTENER)
        arrow(draw, (x + 14, y + 7), (1.0, 0.0), 14, 7, LISTENER, 2)
        text(draw, (x + 36, y - 2), "you, facing", f, MUTED)
        x += 140
        draw.ellipse([x, y, x + 14, y + 14], outline=MUTED, width=2)
        arrow(draw, (x + 14, y + 7), (1.0, 0.0), 10, 6, MUTED, 2)
        text(draw, (x + 32, y - 2), "other places", f, MUTED)
        x += 150
        draw.line([(x, y + 7), (x + 26, y + 7)], fill=WOMEN, width=3)
        text(draw, (x + 34, y - 2), "playing; brighter: louder here", f, MUTED)

    def where(self, listener: str) -> str:
        """Where the choir and the organ are from the listener: distance, and in front,
        behind or to a side as the listener faces."""
        item = self.listeners[listener]
        px, py, pz = item["position"]
        yaw = math.radians(item.get("orientation", {}).get("yaw", 0.0))

        def describe(points) -> str:
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            cz = sum(p[2] for p in points) / len(points)
            dx, dy, dz = cx - px, cy - py, cz - pz
            ahead = dx * math.cos(yaw) + dy * math.sin(yaw)
            left = -dx * math.sin(yaw) + dy * math.cos(yaw)
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            if abs(ahead) >= abs(left):
                side = "in front of you" if ahead > 0 else "behind you"
            else:
                side = "to your left" if left > 0 else "to your right"
            up = f", {dz:.0f} m up" if dz > 3.0 else ""
            return f"{distance:.0f} m {side}{up}" if distance >= 1.5 else "all round you"

        choir = [s["position"] for s in self.sources if s["name"].startswith("singer_")]
        organ = [s["position"] for s in self.sources if s["name"].startswith("organ")]
        return f"The choir: {describe(choir)} · the organ: {describe(organ)}"

    def still(self, k: int) -> Image.Image:
        listener, start, _stop = self.tour[k]
        self.now = self.playing(start)
        img = Image.new("RGBA", (1920, 1080), BG)
        draw = ImageDraw.Draw(img)
        text(
            draw,
            (80, 44),
            "A choir and an organ in a church, simulated",
            font("Bold", 50, display=True),
        )
        subtitle = (
            "The church example of acoustic_spaces (a stone nave, 9500 m³), "
            "heard binaurally at six places"
        )
        text(draw, (80, 112), subtitle, font("Regular", 26), MUTED)
        scale = 19.5
        x0, y0, x1, y1 = self.box
        plan = View("plan", 80, 200, scale, self.box)
        section = View(
            "section", 80, 200 + (y1 - y0) * scale + 56, scale, (x0, 0.0, x1, self.height)
        )
        self.church(img, plan, listener, 6)
        self.church(img, section, listener, 6)
        self.labels(img, plan, section)
        draw = ImageDraw.Draw(img)
        self.legend(draw, 80, int(section.top + self.height * scale) + 16)

        x = 980
        pill, badge = font("SemiBold", 24), "BINAURAL  ·  USE HEADPHONES"
        draw.rounded_rectangle(
            [x, 172, x + draw.textlength(badge, font=pill) + 40, 218], 23, LISTENER
        )
        text(draw, (x + 20, 195), badge, pill, BG, "lm")
        title, where = self.title(listener)
        text(draw, (x, 244), title, font("Bold", 44, display=True))
        text(draw, (x, 302), where, font("Regular", 26), MUTED)
        seat = self.seats.get(listener, {})
        relative = seat.get("relative_db", 0.0)
        choir_c80 = [
            v for (s, lis), v in self.c80.items() if lis == listener and s.startswith("singer_")
        ]
        organ_c80 = [
            v for (s, lis), v in self.c80.items() if lis == listener and s.startswith("organ")
        ]
        facing = (
            "Facing the nave"
            if self.listeners[listener].get("orientation", {}).get("yaw", 0.0) == 180.0
            else "Facing the altar"
        )
        facts = [
            (self.what_plays(start), LISTENER),
            (self.where(listener), TEXT),
            (
                f"Clarity C80 here (0.5-2 kHz): choir {sum(choir_c80) / len(choir_c80):+.1f} dB, "
                f"organ {sum(organ_c80) / len(organ_c80):+.1f} dB",
                TEXT,
            ),
            (
                "The loudest place of the tour"
                if relative > -0.05
                else f"{relative:.1f} dB(A) against the loudest place of the tour",
                TEXT,
            ),
            (f"{facing} · L and R mark your ears", TEXT),
        ]
        y = 350
        for line, colour in facts:
            text(
                draw,
                (x, y),
                line,
                font("SemiBold" if colour == LISTENER else "Regular", 23),
                colour,
            )
            y += 32
        self.close_up(img, x, y + 14, 400, listener)

        tx, ty = 1500, y + 14
        draw = ImageDraw.Draw(img)
        text(draw, (tx, ty), "The tour", font("SemiBold", 22), MUTED)
        ty += 40
        for j, (name, t0, _) in enumerate(self.tour):
            colour = LISTENER if j == k else (MUTED if j < k else FAINT)
            weight = "SemiBold" if j == k else "Regular"
            mark = "▶ " if j == k else "   "
            text(draw, (tx, ty), mark + self.title(name)[0], font(weight, 19), colour)
            text(draw, (1840, ty), mmss(t0), font(weight, 19), colour, "ra")
            ty += 30

        draw.rounded_rectangle([80, 885, 1840, 995], 14, PANEL)  # the spectrogram goes here
        text(draw, (80, 1016), CREDIT, font("Regular", 18), MUTED)
        text(draw, (80, 1046), self.tool, font("Regular", 17), FAINT)
        return img

    def thumbnail(self) -> Image.Image:
        img = Image.new("RGBA", (1280, 720), BG)
        x0, y0, x1, y1 = self.box
        scale = 15.0
        plan = View("plan", 600, 90, scale, self.box)
        section = View(
            "section", 600, 90 + (y1 - y0) * scale + 30, scale, (x0, 0.0, x1, self.height)
        )
        listener = "nave_middle" if "nave_middle" in self.listeners else None
        self.now = ("choir", "organ")
        self.church(img, plan, listener, 5)
        self.church(img, section, listener, 5)
        draw = ImageDraw.Draw(img)
        text(draw, (50, 70), "Choir", font("Black", 92, display=True))
        text(draw, (50, 170), "& organ", font("Black", 92, display=True))
        text(draw, (50, 282), "in a church, simulated", font("Bold", 44, display=True), MUTED)
        pill = font("Bold", 34)
        draw.rounded_rectangle(
            [50, 360, 98 + draw.textlength("BINAURAL", font=pill), 420], 30, LISTENER
        )
        text(draw, (74, 390), "BINAURAL", pill, BG, "lm")
        singers = sum(1 for s in self.sources if s["name"].startswith("singer_"))
        text(draw, (50, 452), f"{singers} real singers, one organ", font("SemiBold", 32), LISTENER)
        text(
            draw,
            (50, 496),
            f"{len(self.listeners)} places in the nave and the choir",
            font("SemiBold", 32),
            LISTENER,
        )
        text(
            draw,
            (50, 655),
            "close-miked recordings through a simulated stone church",
            font("Medium", 24),
            MUTED,
        )
        return img
