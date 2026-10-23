"""Stills for lakeside_demo.py's video: the lake from above with the wind, the listener
of the moment and the other places, a slice through the scene along the wind with the
sound's rays bending in the evening's weather (or going straight in still air), and
the tour. Needs Pillow (which the viz extra brings); Inter is used when installed, else
DejaVu Sans.

The rays are drawn from the scene's own weathers: each one's effective speed of sound
along the slice (its temperature profile, and its wind's profile along the slice
added), the weather zones taking over where they lie, and a ray turning towards the
slower air as the ray equation has it. They show what the simulation's bent rays do,
in the plane of the wind; the sound heard is the simulation's.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BG = (13, 17, 28)
PANEL = (21, 27, 42)
LINE = (58, 68, 92)
TEXT = (232, 236, 244)
MUTED = (148, 158, 182)
FAINT = (96, 106, 132)
LAWN = (31, 54, 40)
GARDEN = (38, 58, 40)
WATER = (24, 52, 86)
WATER_EDGE = (52, 98, 150)
HOUSE = (110, 96, 84)
HOUSE_EDGE = (170, 150, 128)
STAGE = (78, 58, 42)
PA = (255, 166, 77)
LISTENER = (102, 217, 239)
WIND = (76, 201, 240)
COOL = (60, 120, 200)
STILL = (190, 196, 210)

#: Where make_video draws the spectrogram: left, top, width, height.
SPECTRUM_BOX = (100, 892, 1720, 96)
#: How much taller than wide the slice is drawn.
EXAGGERATION = 5.0
#: The angles (degrees above the horizontal) of the rays drawn each way from the stage.
FAN = np.arange(-6.0, 12.01, 1.0)
#: Of a ray's strength, what each bounce leaves: on water, and on the lawn.
BOUNCE = {"water": 0.95, "lawn": 0.3}

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


def inside(point, polygon) -> bool:
    """Whether ``point`` (x, y) lies in ``polygon`` [(x, y), ...] (even-odd rule)."""
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


class Plan:
    """Scene x and y to pixels, from above, north (+y) up."""

    def __init__(self, left, top, scale, box):
        self.left, self.top, self.s = left, top, scale
        self.x0, self.y0, self.x1, self.y1 = box

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return (self.left + (x - self.x0) * self.s, self.top + (self.y1 - y) * self.s)

    @property
    def size(self) -> tuple[float, float]:
        return (self.x1 - self.x0) * self.s, (self.y1 - self.y0) * self.s


class Slice:
    """Scene x and height z to pixels, the height ``EXAGGERATION`` times the length."""

    def __init__(self, left, top, scale, x0, x1, sky):
        self.left, self.top, self.s = left, top, scale
        self.x0, self.x1, self.sky = x0, x1, sky

    def __call__(self, x: float, z: float) -> tuple[float, float]:
        return (
            self.left + (x - self.x0) * self.s,
            self.top + (self.sky - z) * self.s * EXAGGERATION,
        )

    @property
    def size(self) -> tuple[float, float]:
        return (self.x1 - self.x0) * self.s, self.sky * self.s * EXAGGERATION


class Weather:
    """A scene's weathers along the line y = 0: which one holds at x, its profiles, and
    its effective speed of sound for sound going along +x or -x."""

    def __init__(self, scene: dict):
        from acoustic_spaces.scene import Scene

        model = Scene.model_validate(scene)
        self.atmosphere = model.atmosphere
        self.air_c = float(model.air.temperature_c)
        if self.atmosphere is None:
            self.weathers, self.grounds, self.zones = [], [], []
            return
        self.weathers = self.atmosphere.weathers()
        self.grounds = self.atmosphere.ground_temperatures(self.air_c)
        self.zones = [
            (zone.name, [tuple(p) for p in zone.polygon]) for zone in self.atmosphere.zones or []
        ]
        self.z = np.linspace(0.0, 60.0, 1201)
        self.tables = {}
        for k, (weather, ground) in enumerate(zip(self.weathers, self.grounds, strict=True)):
            for way in (1, -1):
                c = weather.sound_speed(self.z, ground, np.array([float(way), 0.0]))
                self.tables[(k, way)] = (c, np.gradient(c, self.z))

    def index(self, x: float, y: float = 0.0) -> int:
        for k, (_name, polygon) in enumerate(self.zones):
            if inside((x, y), polygon):
                return k + 1
        return 0

    def bending(self, x: np.ndarray, z: np.ndarray, way: int) -> np.ndarray:
        """(1/c) dc/dz at each ray's x and z, for rays going ``way`` (+1 or -1) along x."""
        out = np.zeros_like(z)
        if self.atmosphere is None:
            return out
        k = np.array([self.index(float(v)) for v in x])
        for j in np.unique(k):
            c, slope = self.tables[(int(j), way)]
            pick = k == j
            out[pick] = np.interp(z[pick], self.z, slope) / np.interp(z[pick], self.z, c)
        return out

    def label(self, k: int) -> str:
        weather = self.weathers[k]
        name = "park" if k == 0 else self.zones[k - 1][0]
        speed = weather.wind.speed_m_s if weather.wind is not None else 0.0
        parts = [f"{name}: wind {speed:g} m/s"]
        gradient = weather.temperature_gradient_k_m
        if k and abs(self.grounds[k] - self.grounds[0]) > 0.05 or gradient:
            parts.append(f"{self.grounds[k]:g} °C at the ground")
        if gradient:
            parts.append(f"+{10 * gradient:g} °C per 10 m up")
        return ", ".join(parts)


def rays(weather: Weather, start, extent, sky, water, step: float = 1.0):
    """Rays from ``start`` (x, z) along the fan both ways: a list of ((n, 2) points,
    strengths (n,)) each, bending as ``weather`` has it, bouncing on the ground (water
    where ``water(x)``) and stopping at the sky or the scene's ends."""
    out = []
    for way in (1, -1):
        theta = np.radians(FAN)
        x = np.full(len(FAN), float(start[0]))
        z = np.full(len(FAN), float(start[1]))
        strength = np.ones(len(FAN))
        going = np.ones(len(FAN), dtype=bool)
        paths = [[(x[j], z[j], 1.0)] for j in range(len(FAN))]
        for _ in range(int(2.0 * (extent[1] - extent[0]) / step)):
            if not going.any():
                break
            g = weather.bending(x, z, way)
            theta = theta - step * g * np.cos(theta)
            x = x + way * step * np.cos(theta)
            z = z + step * np.sin(theta)
            low = going & (z < 0.0)
            for j in np.flatnonzero(low):
                strength[j] *= BOUNCE["water" if water(x[j]) else "lawn"]
            z = np.where(low, -z, z)
            theta = np.where(low, -theta, theta)
            for j in np.flatnonzero(going):
                paths[j].append((x[j], min(z[j], sky), strength[j]))
            going &= (z < sky) & (x > extent[0]) & (x < extent[1]) & (strength > 0.05)
        for path in paths:
            data = np.array(path)
            out.append((data[:, :2], data[:, 2]))
    return out


class Pictures:
    def __init__(
        self, scenes: dict, seats, tour, titles, gains, credit: str, folder: Path | None = None
    ):
        self.scenes, self.seats, self.tour, self.titles, self.gains = (
            scenes,
            seats,
            tour,
            titles,
            gains,
        )
        self.credit = credit
        #: Where the scene's files (the arrays' directivity) are.
        self.folder = Path(folder) if folder is not None else Path(".")
        scene = scenes["evening"]
        ground = scene["geometry"]["ground"]
        (x0, y0), (x1, y1) = ground["extent"]
        self.box = (x0, y0, x1, y1)
        self.sky = float(ground.get("sky", 40.0))
        self.water = [
            [tuple(p) for p in zone["polygon"]]
            for zone in ground.get("zones", [])
            if scene.get("assignments", {}).get(zone["name"]) == "lake" or zone["name"] == "water"
        ]
        self.houses = scene["geometry"].get("buildings", [])
        self.sources = scene["sources"]
        self.listeners = {item["name"]: item for item in scene["listeners"]}
        self.weather = {w: Weather(s) for w, s in scenes.items()}
        # The rays and the distances start from the flown loudspeakers (the arrays), or
        # from every source when none is flown.
        flown = [s for s in self.sources if s["position"][2] > 2.0] or self.sources
        stage = np.mean([s["position"] for s in flown], axis=0)
        self.stage = stage
        self.rays = {
            w: rays(
                self.weather[w],
                (stage[0], stage[2]),
                (x0, x1),
                self.sky,
                lambda x: any(inside((x, 0.0), p) for p in self.water),
            )
            for w in scenes
        }
        early = scene.get("simulation", {}).get("early", {})
        order = early.get("max_order")
        self.tool = (
            "Simulated with acoustic_spaces (lakeside_concert example): "
            + (f"image sources to order {order}, " if order else "")
            + "bent rays for the late sound in the scene's wind and temperature, binaural "
            "through a spherical head"
        )

    def title(self, listener: str) -> tuple[str, str]:
        return self.titles.get(listener, (listener.replace("_", " ").capitalize(), ""))

    # ------------------------------------------------------------------ the plan

    def ground(self, img, plan: Plan, weather: str) -> None:
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = self.box
        draw.rectangle([*plan(x0, y1), *plan(x1, y0)], fill=LAWN)
        zones = self.weather["evening"].zones
        for name, polygon in zones:
            if name != "lake":
                draw.polygon([plan(*p) for p in polygon], fill=GARDEN)
        for polygon in self.water:
            draw.polygon([plan(*p) for p in polygon], fill=WATER, outline=WATER_EDGE)
        for house in self.houses:
            corners = [plan(*p) for p in house["footprint"]]
            draw.polygon(corners, fill=HOUSE, outline=HOUSE_EDGE)

    def wind(self, img, plan: Plan, weather: str) -> None:
        """Arrows the way the wind blows, as long as it is strong, and each weather's
        name and wind; nothing in still air."""
        state = self.weather[weather]
        draw = ImageDraw.Draw(img, "RGBA")
        if state.atmosphere is None:
            x0, y0, x1, y1 = self.box
            cx, cy = plan((x0 + x1) / 2, y1 - 16)
            text(
                draw,
                (cx, cy),
                "STILL AIR · NO WIND, ONE TEMPERATURE",
                font("SemiBold", 17),
                STILL,
                "mm",
            )
            return
        for _name, polygon in state.zones:
            points = [plan(*p) for p in polygon]
            for a, b in zip(points, points[1:] + points[:1], strict=True):
                length = math.dist(a, b)
                for t in np.arange(0.0, length, 12.0):
                    u, v = t / length, min(1.0, (t + 6.0) / length)
                    draw.line(
                        [
                            (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u),
                            (a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v),
                        ],
                        fill=WIND + (150,),
                        width=2,
                    )
        x0, y0, x1, y1 = self.box
        spacing = 50.0
        unit = 9.0  # pixels per m/s
        for gx in np.arange(x0 + spacing / 2, x1, spacing):
            for gy in np.arange(y0 + spacing / 2, y1 - 40.0, spacing):
                if math.dist((gx, gy), self.stage[:2]) < 30 or any(
                    math.dist((gx, gy), item["position"][:2]) < 28
                    for item in self.listeners.values()
                ):
                    continue
                k = state.index(gx, gy)
                weather_k = state.weathers[k]
                speed = float(weather_k.wind_speed(10.0)) if weather_k.wind else 0.0
                if speed <= 0.0:
                    continue
                dx, dy = weather_k.to_wind()
                px, py = plan(gx, gy)
                length = unit * speed
                arrow(
                    draw,
                    (px - dx * length / 2, py + dy * length / 2),
                    (dx, -dy),
                    length,
                    7,
                    WIND + (190,),
                    3,
                )
        small = font("SemiBold", 16)
        labels = [((x0 + 60.0) / 2 - 20, y1 - 14, 0)]
        for k, (_name, polygon) in enumerate(state.zones, start=1):
            xs = [p[0] for p in polygon]
            labels.append(((min(xs) + max(xs)) / 2, y1 - 14, k))
        for x, y, k in labels:
            px, py = plan(x, y)
            words = state.label(k).upper()
            half = draw.textlength(words, font=small) / 2 + 10
            draw.rounded_rectangle([px - half, py - 13, px + half, py + 13], 13, BG + (200,))
            text(draw, (px, py), words, small, WIND, "mm")

    @staticmethod
    def is_sub(source: dict) -> bool:
        """A subwoofer: on the ground, under the flown arrays."""
        return source["position"][2] < 2.0

    def stage_marks(self, img, plan: Plan) -> None:
        """The stage and its loudspeakers: a dot each, the arrays with a wide cone the
        way they aim (their boxes cover 90 degrees across), the subwoofers a narrower one
        (cardioids), as long as the plan's scale allows."""
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([*plan(-10, 12), *plan(0, -12)], fill=STAGE, outline=PA)
        cones = ImageDraw.Draw(img, "RGBA")
        for source in self.sources:
            x, y = source["position"][:2]
            px, py = plan(x, y)
            fx, fy = self.facing(source)
            qx, qy = plan(x + fx, y + fy)
            direction = (qx - px, qy - py)
            if self.is_sub(source):
                cone(cones, (px, py), direction, 16 * plan.s, 32.0, PA + (70,))
            else:
                cone(cones, (px, py), direction, 30 * plan.s, 45.0, PA + (70,))
        for source in self.sources:
            px, py = plan(*source["position"][:2])
            r = 4 if self.is_sub(source) else 6
            glow(img, [(px, py, 2 * r, PA + (120,))], 5)
            draw = ImageDraw.Draw(img, "RGBA")
            draw.ellipse([px - r, py - r, px + r, py + r], fill=PA, outline=BG)

    def aims(self, source: dict) -> tuple[float, float]:
        """The pitch (degrees, down negative) of an array's top and bottom boxes, from
        its directivity file; a source without one aims level."""
        elements = self.array_of(source).get("elements", [])
        pitches = [e.get("orientation", {}).get("pitch", 0.0) for e in elements]
        return (max(pitches), min(pitches)) if pitches else (0.0, 0.0)

    def hang(self, source: dict) -> float:
        """How tall an array hangs (m): the spread of its elements' heights."""
        elements = self.array_of(source).get("elements", [])
        heights = [e.get("position", [0, 0, 0])[2] for e in elements]
        return max(heights) - min(heights) if len(heights) > 1 else 0.0

    def array_of(self, source: dict) -> dict:
        """The array a source's directivity file describes, {} without one."""
        spec = source.get("directivity")
        path = spec.get("file") if isinstance(spec, dict) else None
        if not path:
            return {}
        with open(self.folder / path, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def facing(item: dict) -> tuple[float, float]:
        yaw = math.radians(item.get("orientation", {}).get("yaw", 0.0))
        return math.cos(yaw), math.sin(yaw)

    def people(self, img, plan: Plan, current: str) -> None:
        draw = ImageDraw.Draw(img, "RGBA")
        small = font("Medium", 15)
        for name, item in self.listeners.items():
            px, py = plan(*item["position"][:2])
            fx, fy = self.facing(item)
            if name == current:
                continue
            draw.ellipse([px - 7, py - 7, px + 7, py + 7], outline=MUTED, width=2)
            arrow(draw, (px + fx * 7, py - fy * 7), (fx, -fy), 12, 6, MUTED, 2)
            label = self.title(name)[0]
            x = item["position"][0]
            if math.dist(item["position"][:2], self.stage[:2]) < 60:
                text(draw, (px + 4, py + 12), label, small, MUTED, "la")
            elif x - self.box[0] < 60:
                text(draw, (px - 8, py + 12), label, small, MUTED, "la")
            else:
                text(draw, (px, py + 12), label, small, MUTED, "ma")
        item = self.listeners[current]
        px, py = plan(*item["position"][:2])
        sx, sy = plan(*self.stage[:2])
        draw.line([(sx, sy), (px, py)], fill=LISTENER + (120,), width=2)
        self.head(img, (px, py), self.facing(item), 11)

    def head(self, img, at, facing, r: float) -> None:
        """The listener: a head with a nose and an arrow the way it faces, and its ears
        marked L and R."""
        hx, hy = at
        glow(img, [(hx, hy, 3 * r, LISTENER + (110,))], r)
        fx, fy = facing[0], -facing[1]
        lx, ly = fy, -fx  # the left ear, a quarter turn from the facing (y down)
        draw = ImageDraw.Draw(img)
        start = (hx + fx * r, hy + fy * r)
        arrow(draw, start, (fx, fy), r * 3.4, r * 1.2, BG, 7)
        arrow(draw, start, (fx, fy), r * 3.2, r * 1.0, LISTENER, 4)
        draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=LISTENER, outline=BG, width=2)
        letters = font("Bold", 15)
        for side, letter in ((1, "L"), (-1, "R")):
            draw.text(
                (hx + lx * r * 2.0 * side, hy + ly * r * 2.0 * side),
                letter,
                font=letters,
                fill=TEXT,
                anchor="mm",
                stroke_width=3,
                stroke_fill=BG,
            )

    def scale_bar(self, img, plan: Plan) -> None:
        draw = ImageDraw.Draw(img)
        x0, y0, _x1, _y1 = self.box
        left, bottom = plan(x0, y0)
        bar = 100 * plan.s
        draw.line([(left + 12, bottom - 14), (left + 12 + bar, bottom - 14)], fill=MUTED, width=3)
        text(draw, (left + 12 + bar / 2, bottom - 20), "100 m", font("Medium", 15), MUTED, "md")

    # ------------------------------------------------------------------ the slice

    def slice(self, img, cut: Slice, weather: str, current: str) -> None:
        """The scene along y = 0 seen from the side, the height drawn taller: the ground
        and the water, the houses behind, the rays from the stage, the wind's profile
        at a few places, and the listener of the moment."""
        draw = ImageDraw.Draw(img, "RGBA")
        x0, x1 = cut.x0, cut.x1
        state = self.weather[weather]
        top_left, bottom_right = cut(x0, self.sky), cut(x1, 0.0)
        # The air: cooler over the water in the evening, drawn as a blue haze near it.
        draw.rectangle([*top_left, *bottom_right], fill=(16, 22, 36))
        if state.atmosphere is not None:
            haze = Image.new("RGBA", img.size, (0, 0, 0, 0))
            hd = ImageDraw.Draw(haze)
            for k, (_name, polygon) in enumerate(state.zones, start=1):
                cool = state.grounds[0] - state.grounds[k]
                if cool <= 0:
                    continue
                xs = [p[0] for p in polygon]
                for j in range(24):
                    z0, z1 = j * self.sky / 24, (j + 1) * self.sky / 24
                    alpha = int(min(120, 18 * cool) * (1 - j / 24) ** 2)
                    hd.rectangle([*cut(min(xs), z1), *cut(max(xs), z0)], fill=COOL + (alpha,))
            img.alpha_composite(haze)
            draw = ImageDraw.Draw(img, "RGBA")
        for house in self.houses:
            xs = [p[0] for p in house["footprint"]]
            draw.rectangle([*cut(min(xs), house["height"]), *cut(max(xs), 0.0)], fill=HOUSE + (90,))
        # The ground and the water.
        draw.rectangle([*cut(x0, 0.0), cut(x1, 0.0)[0], cut(x1, 0.0)[1] + 10], fill=LAWN)
        for polygon in self.water:
            xs = [p[0] for p in polygon]
            draw.rectangle(
                [*cut(min(xs), 0.0), cut(max(xs), 0.0)[0], cut(max(xs), 0.0)[1] + 10],
                fill=WATER_EDGE,
            )
        # The rays.
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        colour = PA if state.atmosphere is not None else STILL
        for points, strength in self.rays[weather]:
            pixels = [cut(x, z) for x, z in points]
            for j in range(0, len(pixels) - 1, 4):
                alpha = int(150 * strength[j])
                if alpha < 8:
                    continue
                ld.line(
                    [pixels[j], pixels[min(j + 4, len(pixels) - 1)]],
                    fill=colour + (alpha,),
                    width=2,
                )
        img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.6)))
        draw = ImageDraw.Draw(img, "RGBA")
        # The wind's profile: arrows up a mast in each weather.
        if state.atmosphere is not None:
            for x in (-90.0, 290.0, 640.0):
                k = state.index(x)
                weather_k = state.weathers[k]
                if weather_k.wind is None:
                    continue
                along = float(weather_k.to_wind()[0])
                base = cut(x, 0.0)
                draw.line([base, cut(x, self.sky * 0.92)], fill=TEXT + (120,), width=1)
                for h in (2.0, 5.0, 10.0, 20.0, 34.0):
                    speed = float(weather_k.wind_speed(h)) * along
                    if abs(speed) < 0.05:
                        continue
                    arrow(
                        draw,
                        cut(x, h),
                        (math.copysign(1.0, speed), 0.0),
                        abs(speed) * 14,
                        7,
                        WIND + (230,),
                        3,
                    )
        # The sky's edge.
        for px in range(int(top_left[0]), int(bottom_right[0]), 14):
            draw.line([(px, top_left[1]), (px + 6, top_left[1])], fill=FAINT, width=2)
        text(
            draw,
            (top_left[0] + 6, top_left[1] + 6),
            f"open sky, {self.sky:g} m up",
            font("Medium", 14),
            FAINT,
        )
        # The stage and its loudspeakers: each array hung from the roof, its boxes'
        # aims from the top one's to the bottom one's as a wedge to the lawn; the
        # subwoofers on the ground. Then the listeners.
        draw.rectangle([*cut(-10, 1.5), *cut(0, 0.0)], fill=STAGE)
        for source in self.sources:
            x, _y, z = source["position"]
            if self.is_sub(source):
                draw.rectangle([*cut(x - 0.6, z + 0.6), *cut(x + 0.6, 0.0)], fill=PA, outline=BG)
                continue
            top, bottom = self.aims(source)
            reach = [(x + z / math.tan(math.radians(-a)), 0.0) for a in (top, bottom) if a < 0]
            if len(reach) == 2:
                draw.polygon([cut(x, z), cut(*reach[0]), cut(*reach[1])], fill=PA + (60,))
            half = self.hang(source) / 2
            draw.line([cut(x, 0.0), cut(x, z + half + 0.5)], fill=PA + (120,), width=1)  # its mast
            draw.line([cut(x, z + half), cut(x, z - half)], fill=PA, width=5)
            draw.line([cut(x, z + half), cut(x, z - half)], fill=BG, width=1)
        px, py = cut(self.stage[0], self.stage[2])
        text(draw, (px - 12, py), "line arrays", font("Medium", 14), PA, "rm")
        px, py = cut(self.stage[0], 0.0)
        text(draw, (px - 12, py - 8), "subwoofers", font("Medium", 14), PA, "rm")
        for name, item in self.listeners.items():
            x, _y, z = item["position"]
            px, py = cut(x, z)
            if name == current:
                glow(img, [(px, py, 22, LISTENER + (140,))], 8)
                draw = ImageDraw.Draw(img, "RGBA")
                draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=LISTENER, outline=BG, width=2)
                text(draw, (px, py - 16), "you", font("SemiBold", 16), LISTENER, "md")
            else:
                draw.ellipse([px - 5, py - 5, px + 5, py + 5], outline=MUTED, width=2)

    # ------------------------------------------------------------------ the still

    def facts(self, listener: str, weather: str) -> list[str]:
        seat = self.seats[f"{listener}.{weather}"]
        lines = [f"The music reaches you {seat['lag_s']:.2f} s after it leaves the stage"]
        relative = seat["relative_db"]
        lines.append(
            "The loudest place of the tour"
            if relative > -0.05
            else f"{-relative:.1f} dB(A) quieter than the loudest place of the tour"
        )
        other = "still" if weather == "evening" else "evening"
        twin = self.seats.get(f"{listener}.{other}")
        if twin is not None:
            change = seat["rms_db"] - twin["rms_db"]
            if weather == "evening":
                lines.append(
                    f"In still air it would be {abs(change):.1f} dB "
                    + ("quieter" if change > 0 else "louder")
                )
            else:
                lines.append(
                    f"{abs(change):.1f} dB "
                    + ("louder" if change > 0 else "quieter")
                    + " than in the evening's weather, as you hear"
                )
        gain = self.gains[f"{listener}.{weather}"]
        if weather == "still":
            lines.append("Turned up as much as in the evening's weather, to compare")
        elif gain > 0.5:
            lines.append(f"Turned up {gain:.0f} dB here so you can hear it")
        lines.append("Facing the stage · L and R mark your ears")
        return lines

    def still(self, k: int) -> Image.Image:
        listener, weather, _start, _stop = self.tour[k]
        img = Image.new("RGBA", (1920, 1080), BG)
        draw = ImageDraw.Draw(img)
        text(draw, (80, 40), "A concert across a lake, simulated", font("Bold", 50, display=True))
        subtitle = (
            "The lakeside concert example on a summer evening, heard binaurally from the "
            "lawn to the far shore"
        )
        text(draw, (80, 106), subtitle, font("Regular", 26), MUTED)

        width = 1120
        scale = width / (self.box[2] - self.box[0])
        plan = Plan(80, 170, scale, self.box)
        self.ground(img, plan, weather)
        self.wind(img, plan, weather)
        self.stage_marks(img, plan)
        self.people(img, plan, listener)
        self.scale_bar(img, plan)
        _w, h = plan.size
        top = 170 + h + 50
        draw = ImageDraw.Draw(img)
        caption = "Seen from the side, along the wind (heights ×5): " + (
            "the sound bends down over the cool water, up behind the stage"
            if weather == "evening"
            else "in still air the sound goes straight"
        )
        text(draw, (80, top - 34), caption, font("SemiBold", 20), MUTED)
        cut = Slice(80, top, scale, self.box[0], self.box[2], self.sky)
        self.slice(img, cut, weather, listener)

        x = 1250
        draw = ImageDraw.Draw(img)
        pill, badge = font("SemiBold", 22), "BINAURAL  ·  USE HEADPHONES"
        draw.rounded_rectangle(
            [x, 170, x + draw.textlength(badge, font=pill) + 36, 212], 21, LISTENER
        )
        text(draw, (x + 18, 191), badge, pill, BG, "lm")
        title, where = self.title(listener)
        text(draw, (x, 236), title, font("Bold", 40, display=True))
        text(draw, (x, 290), where, font("Regular", 24), MUTED)
        tag = "This evening's weather" if weather == "evening" else "If the air were still"
        tag_colour = WIND if weather == "evening" else STILL
        tf = font("SemiBold", 20)
        draw.rounded_rectangle(
            [x, 330, x + draw.textlength(tag, font=tf) + 28, 364], 17, outline=tag_colour, width=2
        )
        text(draw, (x + 14, 347), tag, tf, tag_colour, "lm")
        y = 384
        for line in self.facts(listener, weather):
            text(draw, (x, y), line, font("Regular", 21), TEXT)
            y += 32

        ty = max(y + 24, 590)
        text(draw, (x, ty), "The tour", font("SemiBold", 21), MUTED)
        ty += 32
        row = min(26, (860 - ty) // max(1, len(self.tour)))
        for j, (name, w, t0, _) in enumerate(self.tour):
            colour = LISTENER if j == k else (MUTED if j < k else FAINT)
            weight = "SemiBold" if j == k else "Regular"
            mark = "▶ " if j == k else "   "
            label = self.title(name)[0] + (", still air" if w == "still" else "")
            text(draw, (x, ty), mark + label, font(weight, 18), colour)
            text(draw, (1840, ty), mmss(t0), font(weight, 18), colour, "ra")
            ty += row

        draw.rounded_rectangle([80, 885, 1840, 995], 14, PANEL)  # the spectrogram goes here
        text(draw, (80, 1016), self.credit, font("Regular", 19), MUTED)
        text(draw, (80, 1046), self.tool, font("Regular", 17), FAINT)
        return img

    def thumbnail(self) -> Image.Image:
        img = Image.new("RGBA", (1280, 720), BG)
        scale = 1180 / (self.box[2] - self.box[0])
        self.slice(img, Slice(50, 400, scale, self.box[0], self.box[2], self.sky), "evening", "")
        draw = ImageDraw.Draw(img)
        text(draw, (50, 50), "Concert across a lake,", font("Black", 72, display=True))
        text(draw, (50, 140), "simulated", font("Bold", 56, display=True), MUTED)
        pill = font("Bold", 30)
        draw.rounded_rectangle(
            [50, 230, 94 + draw.textlength("BINAURAL", font=pill), 284], 27, LISTENER
        )
        text(draw, (72, 257), "BINAURAL", pill, BG, "lm")
        text(
            draw,
            (50, 310),
            "how a summer evening carries the music 550 m over the water",
            font("SemiBold", 28),
            LISTENER,
        )
        return img
