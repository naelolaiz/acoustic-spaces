"""Stills for lakeside_demo.py's video: the lake from above with the wind, the listener
of the moment and the other places, a slice through the scene along the wind with the
sound's rays bending in the evening's weather (or going straight in still air), and
the tour. The frame round them comes from :mod:`acoustic_spaces.demos.pictures`.

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
from PIL import Image, ImageDraw, ImageFilter

from acoustic_spaces.demos.pictures import (
    BG,
    FAINT,
    LISTENER,
    MARGIN,
    MUTED,
    PA,
    SIZE,
    TEXT,
    THUMBNAIL,
    Plan,
    arrow,
    axes,
    badge,
    canvas,
    cone,
    facing,
    facts,
    font,
    footer,
    glow,
    head,
    heading,
    place,
    ring,
    save,
    scale_bar,
    text,
    thumbnail_badge,
    tool_line,
    tour_column,
)

LAWN = (31, 54, 40)
GARDEN = (38, 58, 40)
WATER = (24, 52, 86)
WATER_EDGE = (52, 98, 150)
HOUSE = (110, 96, 84)
HOUSE_EDGE = (170, 150, 128)
STAGE = (78, 58, 42)
WIND = (76, 201, 240)
COOL = (60, 120, 200)
STILL = (190, 196, 210)


#: How much taller than wide the slice is drawn.
EXAGGERATION = 5.0
#: The angles (degrees above the horizontal) of the rays drawn each way from the stage.
FAN = np.arange(-6.0, 12.01, 1.0)
#: Of a ray's strength, what each bounce leaves: on water, and on the lawn.
BOUNCE = {"water": 0.95, "lawn": 0.3}


def inside(point, polygon) -> bool:
    """Whether ``point`` (x, y) lies in ``polygon`` [(x, y), ...] (even-odd rule)."""
    x, y = point
    hit = False
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


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
        self,
        scenes: dict,
        seats,
        tour,
        titles,
        gains,
        credit: str,
        folder: Path | None = None,
        hrtf: str | Path | None = None,
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
        self.tool = tool_line(
            "lakeside_concert",
            scene,
            late="bent rays for the late sound in the scene's wind and temperature",
            hrtf=hrtf,
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
            fx, fy = facing(source)
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

    def people(self, img, plan: Plan, current: str) -> None:
        draw = ImageDraw.Draw(img, "RGBA")
        small = font("Medium", 15)
        for name, item in self.listeners.items():
            if name == current:
                continue
            px, py = plan(*item["position"][:2])
            forward, _ = axes(plan, item)
            ring(draw, (px, py), forward, 7, 12, 6)
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
        forward, left = axes(plan, item)
        head(img, (px, py), forward, left, 11, nose=False, ears=False)

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
        img = canvas(SIZE)
        draw = ImageDraw.Draw(img)
        heading(
            draw,
            "A concert across a lake, simulated",
            "The lakeside concert example on a summer evening, heard binaurally from the "
            "lawn to the far shore",
            y=40,
        )

        width = 1120
        scale = width / (self.box[2] - self.box[0])
        plan = Plan(MARGIN, 170, scale, self.box)
        self.ground(img, plan, weather)
        self.wind(img, plan, weather)
        self.stage_marks(img, plan)
        self.people(img, plan, listener)
        draw = ImageDraw.Draw(img)
        left, bottom = plan(self.box[0], self.box[1])
        scale_bar(draw, left + 12, bottom - 14, 100 * plan.s, "100 m")
        _w, h = plan.size
        top = 170 + h + 50
        caption = "Seen from the side, along the wind (heights ×5): " + (
            "the sound bends down over the cool water, up behind the stage"
            if weather == "evening"
            else "in still air the sound goes straight"
        )
        text(draw, (MARGIN, top - 34), caption, font("SemiBold", 20), MUTED)
        cut = Slice(MARGIN, top, scale, self.box[0], self.box[2], self.sky)
        self.slice(img, cut, weather, listener)

        x = 1250
        draw = ImageDraw.Draw(img)
        badge(draw, x, 170, size=22)
        place(draw, x, 236, *self.title(listener), size=40, where_size=24)
        tag = "This evening's weather" if weather == "evening" else "If the air were still"
        tag_colour = WIND if weather == "evening" else STILL
        tf = font("SemiBold", 20)
        draw.rounded_rectangle(
            [x, 330, x + draw.textlength(tag, font=tf) + 28, 364], 17, outline=tag_colour, width=2
        )
        text(draw, (x + 14, 347), tag, tf, tag_colour, "lm")
        y = facts(draw, x, 384, self.facts(listener, weather), size=21, step=32)
        entries = [
            (self.title(name)[0] + (", still air" if w == "still" else ""), t0)
            for name, w, t0, _ in self.tour
        ]
        tour_column(draw, x, max(y + 24, 590), entries, k, row=26, size=18)
        footer(draw, self.credit, self.tool)
        return img

    def thumbnail(self) -> Image.Image:
        img = canvas(THUMBNAIL)
        scale = 1180 / (self.box[2] - self.box[0])
        self.slice(img, Slice(50, 400, scale, self.box[0], self.box[2], self.sky), "evening", "")
        draw = ImageDraw.Draw(img)
        text(draw, (50, 50), "Concert across a lake,", font("Black", 72, display=True))
        text(draw, (50, 140), "simulated", font("Bold", 56, display=True), MUTED)
        thumbnail_badge(draw, 50, 230, size=30)
        text(
            draw,
            (50, 310),
            "how a summer evening carries the music 550 m over the water",
            font("SemiBold", 28),
            LISTENER,
        )
        return img


__all__ = ["Pictures", "Slice", "Weather", "save"]
