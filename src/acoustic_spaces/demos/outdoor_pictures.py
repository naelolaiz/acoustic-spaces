"""What the stills of an outdoor scene show besides its plan
(:mod:`acoustic_spaces.demos.scene_pictures`).

From above: the ground coloured by its materials, the buildings, barriers and plants on
it, and the wind of each of the atmosphere's weathers, with the zones it holds in
(:meth:`Outdoors.ground`, :meth:`Outdoors.wind`).

From the side (:meth:`Outdoors.side`): the vertical plane through a source and the
listener of the moment, across the whole ground, its height drawn taller: the ground
along it, what stands on it, the air cooler or warmer than the scene's, the wind's
component along the plane up a mast in each weather, and the sound's rays in it. The
bold ones are the eigenrays the simulation finds from the source to the listener over
the open ground (:class:`~acoustic_spaces.engine.refraction.Rays`), in the profile it
takes along that way: the mean of the weathers under it, each weighted by its length
(:class:`~acoustic_spaces.engine.weather.Weather`); the faint ones a fan from the
source, a ray every degree, each bending as the weather it is over at each step has it
and losing at each bounce the share of its energy the ground there does not reflect at
1 kHz (the plane wave's reflection factor of its impedance at that angle, or 1 minus its
absorption coefficient). The sound heard is the simulation's; the rays show where it
goes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from acoustic_spaces import products
from acoustic_spaces.demos.pictures import (
    BG,
    FAINT,
    LISTENER,
    MUTED,
    SOURCE,
    TEXT,
    arrow,
    font,
    glow,
    text,
)

WATER = (24, 52, 86)
WATER_EDGE = (52, 98, 150)
SOFT = (31, 54, 40)
HARD = (54, 58, 68)
PLANTS = (52, 104, 58)
BUILDING = (110, 96, 84)
BUILDING_EDGE = (170, 150, 128)
WIND = (76, 201, 240)
COOL = (60, 120, 200)
WARM = (214, 120, 64)
STILL = (190, 196, 210)
#: Where along the top of the ground, then along its bottom, the line that says the air is
#: still is tried (a fraction of the way across), and how far (pixels) it keeps from the
#: marks of the sources and listeners.
STILL_ACROSS = (0.5, 0.25, 0.75)
CLEAR_PX = 14.0
AIR = (16, 22, 36)
#: The colour of each kind of ground (:func:`ground_kind`).
GROUNDS = {"water": WATER, "hard": HARD, "soft": SOFT}

#: The fan's rays leave the source at these angles above the horizontal (degrees),
#: each way along the plane.
FAN_DEG = np.arange(-6.0, 12.01, 1.0)
#: The frequency (Hz) at which a fan ray's bounces take the ground's reflection.
BOUNCE_HZ = 1000.0
#: A fan ray ends when it keeps less than this share of its energy.
FAINTEST = 0.05
#: Steps along the plane a fan ray takes across the whole ground.
FAN_STEPS = 1600
#: A material whose name, or whose impedance's preset, holds one of these is water.
WATERS = ("water", "lake", "sea", "river", "pond")
#: A ground of at least this flow resistivity (kPa s/m²: Nord2000's class F, gravel, and
#: up) is drawn hard; one without an impedance, absorbing no more than this at 1 kHz.
HARD_RESISTIVITY = 2000.0
HARD_ABSORPTION = 0.1
#: Heights (m) at which a mast shows the wind, those under the view's top.
MAST_M = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
#: Pixels of arrow per m/s of wind along the plane, up a mast.
MAST_PX_PER_M_S = 14.0
#: How much taller than wide the side view may be drawn: the first that shows the
#: height wanted.
EXAGGERATIONS = (1, 2, 5, 10, 20, 50, 100)
#: A difference in the speed of sound (m/s) between 10 m up and the ground smaller than
#: this bends the sound too little to tell.
BENDS_M_S = 0.05


def ground_kind(name: str, material) -> str:
    """``water``, ``hard`` or ``soft``: how the plan colours a ground of ``material``
    (named ``name``)."""
    impedance = getattr(material, "impedance", None)
    words = f"{name} {getattr(impedance, 'preset', '') or ''}".casefold()
    if any(w in words for w in WATERS):
        return "water"
    if impedance is not None:
        return "hard" if impedance.resistivity >= HARD_RESISTIVITY else "soft"
    if material is None:
        return "soft"
    absorption = material.coefficients("octave")[0][4]  # 1 kHz
    return "hard" if absorption <= HARD_ABSORPTION else "soft"


def reflected(material, sine: float, speed_of_sound: float) -> float:
    """The share of a plane wave's energy that a ground of ``material`` reflects at
    BOUNCE_HZ, met at the grazing angle whose sine is ``sine``: |R|² of its impedance,
    or 1 minus its absorption coefficient at 1 kHz (none: all)."""
    from acoustic_spaces.materials.ground import plane_reflection

    if material is None:
        return 1.0
    impedance = getattr(material, "impedance", None)
    if impedance is not None:
        cosine = np.array([max(sine, 0.0)])
        beta = impedance.admittance(np.array([BOUNCE_HZ]), speed_of_sound, cosine=cosine)
        return float(np.abs(plane_reflection(beta, cosine))[0] ** 2)
    return 1.0 - float(material.coefficients("octave")[0][4])


def inside(point, polygon) -> bool:
    """Whether ``point`` (x, y) lies in ``polygon`` (n, 2)."""
    from acoustic_spaces.geometry.outdoor import in_polygon

    return bool(in_polygon(np.asarray([point], dtype=float), np.asarray(polygon, dtype=float))[0])


class Side:
    """Distance along a plane and height above the ground to pixels: ``t0`` to ``t1`` m
    along it across ``width`` pixels from ``left`` (from right to left where
    ``flipped``), and the heights up to ``top`` m, ``exaggeration`` times taller than
    wide, down to ``bottom`` pixels at the ground."""

    def __init__(self, left, bottom, width, t0, t1, top, exaggeration, flipped=False):
        self.left, self.bottom, self.t0, self.t1 = left, bottom, t0, t1
        self.s = width / (t1 - t0)
        self.top, self.exaggeration, self.flipped = top, exaggeration, flipped

    def __call__(self, t: float, h: float) -> tuple[float, float]:
        along = self.t1 - t if self.flipped else t - self.t0
        return (self.left + along * self.s, self.bottom - h * self.s * self.exaggeration)

    def rect(self, ta: float, ha: float, tb: float, hb: float) -> list[float]:
        """The rectangle between (``ta``, ``ha``) and (``tb``, ``hb``) as Pillow wants it:
        left, top, right, bottom."""
        (x0, y0), (x1, y1) = self(ta, ha), self(tb, hb)
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    @property
    def ahead(self) -> float:
        """Which way along the picture the plane's distances grow: 1 right, -1 left."""
        return -1.0 if self.flipped else 1.0


class Outdoors:
    """An outdoor scene's ground, what stands on it and its weather, from ``room`` (a
    :class:`~acoustic_spaces.room.Room` with ``outdoor``), for the stills."""

    def __init__(self, room) -> None:
        from acoustic_spaces.engine.weather import Weather
        from acoustic_spaces.geometry import UNIT_SCALE

        self.room = room
        self.layout = layout = room.outdoor
        scene = room.scene
        self.c = float(room.speed_of_sound())
        self.ground_z = float(layout.ground)
        self.sky = float(layout.sky) - self.ground_z
        self.flat = layout.terrain is None
        # The material of the ground outside the zones (first) and in each.
        labels = [("ground",), *((name, "zones", "ground") for name, _ in layout.zones)]
        self.materials = [self._material(label) for label in labels]
        self.kinds = [ground_kind(name or "", scene.materials.get(name)) for name in self.materials]
        atmosphere = scene.atmosphere
        self.atmosphere = None if atmosphere is None or atmosphere.still else atmosphere
        self.air = None
        self.weathers, self.grounds_c, self.names = [], [], []
        if self.atmosphere is not None and self.flat:
            scale = UNIT_SCALE[scene.geometry.units or "m"]
            ground_c = float(scene.air.temperature_c)
            self.air = Weather(self.atmosphere, ground_c, layout.extent, scale)
            self.weathers = self.atmosphere.weathers()
            self.grounds_c = self.atmosphere.ground_temperatures(ground_c)
            zones = self.atmosphere.zone_names()
            self.names = ["outside the zones" if zones else "the air", *zones]
        self._kept: dict = {}
        self._bounces: dict = {}

    def _material(self, label: tuple[str, ...]) -> str | None:
        """The material the scene gives the faces labelled ``label``, None if none."""
        mesh = self.room.mesh
        if label not in mesh.labels:
            return None
        faces = np.flatnonzero(mesh.face_labels == mesh.labels.index(label))
        index = int(self.room.face_materials[faces[0]]) if len(faces) else -1
        return self.room.material_names[index] if index >= 0 else None

    # -- from above ---------------------------------------------------------------------

    def ground(self, img: Image.Image, plan) -> None:
        """The ground's extent and zones, coloured by their materials, with what stands
        on it: plants, barriers and buildings."""
        draw = ImageDraw.Draw(img, "RGBA")
        x0, y0, x1, y1 = self.layout.extent
        draw.rectangle(plan.rect(x0, y0, x1, y1), fill=GROUNDS[self.kinds[0]])
        for (_name, polygon), kind in zip(self.layout.zones, self.kinds[1:], strict=True):
            edge = WATER_EDGE if kind == "water" else None
            draw.polygon([plan(*p) for p in polygon], fill=GROUNDS[kind], outline=edge)
        for plant in self.layout.vegetation:
            if plant.outline is not None:
                draw.polygon([plan(*p) for p in plant.outline], fill=PLANTS + (170,))
            else:
                (cx, cy), r = plan(*plant.centre[:2]), plant.radius * plan.s
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=PLANTS + (170,))
        for _name, line in self.layout.barriers:
            draw.line([plan(*p[:2]) for p in line], fill=BUILDING_EDGE, width=3)
        for _name, footprint, _roof in self.layout.buildings:
            draw.polygon([plan(*p) for p in footprint], fill=BUILDING, outline=BUILDING_EDGE)

    def legend(self) -> list[tuple[str, tuple]]:
        """The ground's materials, each once: its name and colour."""
        seen: dict[str, tuple] = {}
        for name, kind in zip(self.materials, self.kinds, strict=True):
            if name is not None and name not in seen:
                seen[name] = GROUNDS[kind]
        return list(seen.items())

    def weather_label(self, k: int) -> str:
        """What weather ``k`` is: its name, its wind 10 m up, the temperature at the
        ground where it differs from the scene's, and how it changes with height."""
        weather = self.weathers[k]
        parts = [f"{self.names[k]}: wind {float(weather.wind_speed(10.0)):.1f} m/s 10 m up"]
        ground_c = self.grounds_c[k]
        if k and abs(ground_c - self.grounds_c[0]) > 0.05:
            parts.append(f"{ground_c:g} °C at the ground")
        rise = float(weather.temperature(10.0, ground_c) - weather.temperature(0.0, ground_c))
        if abs(rise) >= 0.05:
            parts.append(f"{rise:+.1f} °C per 10 m up")
        return ", ".join(parts)

    def wind(
        self, img: Image.Image, plan, still: bool, keep_clear=()
    ) -> list[tuple[float, float, float, float]]:
        """The wind from above: an arrow the way it blows in each weather, as long as it
        is strong 10 m up, the weather zones dashed and each weather's label; in still
        air, a line that says so, along the top or the bottom of the ground where it
        covers none of the points ``keep_clear`` (x, y), the sources' and listeners'. No
        arrow within reach of those either. The boxes its words take (pixels: left, top,
        right, bottom), for the labels of the sources and listeners to keep clear of."""
        draw = ImageDraw.Draw(img, "RGBA")
        x0, y0, x1, y1 = self.layout.extent
        small = font("SemiBold", 15)
        if still or self.air is None:
            words = "STILL AIR · NO WIND, ONE TEMPERATURE"
            half = draw.textlength(words, font=small) / 2 + 4
            left, top, right, bottom = plan.rect(x0, y0, x1, y1)
            marks = [plan(*p) for p in keep_clear]
            box = None
            for py in (top + 16.0, bottom - 16.0):
                for across in STILL_ACROSS:
                    px = min(max(left + (right - left) * across, left + half), right - half)
                    tried = (px - half, py - 10.0, px + half, py + 10.0)
                    if box is None:
                        box = tried  # the top's middle, where nothing is clear
                    if not any(
                        tried[0] - CLEAR_PX < mx < tried[2] + CLEAR_PX
                        and tried[1] - CLEAR_PX < my < tried[3] + CLEAR_PX
                        for mx, my in marks
                    ):
                        box = tried
                        break
                else:
                    continue
                break
            assert box is not None
            text(draw, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), words, small, STILL, "mm")
            return [box]
        for polygon in self.air.polygons:
            if polygon is None:
                continue
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
        speeds = [float(w.wind_speed(10.0)) for w in self.weathers]
        strongest = max(speeds)
        span = max(x1 - x0, y1 - y0)
        spacing = span / 16.0
        if strongest > 0.0:
            clear = 0.6 * spacing
            for gx in np.arange(x0 + spacing / 2, x1, spacing):
                for gy in np.arange(y0 + spacing / 2, y1, spacing):
                    if any(math.dist((gx, gy), p) < clear for p in keep_clear):
                        continue
                    k = int(self.air.zone_at(np.array([[gx, gy]]))[0])
                    if speeds[k] <= 0.0:
                        continue
                    dx, dy = plan.vec(*self.weathers[k].to_wind())
                    length = 0.8 * spacing * plan.s * speeds[k] / strongest
                    px, py = plan(gx, gy)
                    start = (px - dx * length / 2, py - dy * length / 2)
                    arrow(draw, start, (dx, dy), length, 7, WIND + (190,), 3)
        # Each weather's label near the top of where it holds, a row lower where it
        # would cover one already placed.
        placed: list[tuple[float, float, float]] = []
        left, right = plan(x0, y0)[0], plan(x1, y0)[0]
        for k in range(len(self.weathers)):
            at = self._label_at(k)
            if at is None:
                continue
            px, py = plan(*at)
            words = self.weather_label(k).upper()
            half = draw.textlength(words, font=small) / 2 + 10
            px = min(max(px, left + half), right - half)
            while any(abs(py - y) < 26 and px - half < b and px + half > a for a, b, y in placed):
                py += 28
            placed.append((px - half, px + half, py))
            draw.rounded_rectangle([px - half, py - 12, px + half, py + 12], 12, BG + (210,))
            text(draw, (px, py), words, small, WIND, "mm")
        return [(a, y - 12.0, b, y + 12.0) for a, b, y in placed]

    def _label_at(self, k: int) -> tuple[float, float] | None:
        """Where weather ``k``'s label goes: near the top of where it holds, in the
        middle across."""
        if self.air is None:
            return None
        x0, y0, x1, y1 = self.layout.extent
        xs = np.linspace(x0, x1, 41)[1:-1]
        ys = np.linspace(y1, y0, 41)[1:-1]
        for gy in ys[1:]:
            row = np.column_stack([xs, np.full(len(xs), gy)])
            hit = xs[self.air.zone_at(row) == k]
            if len(hit):
                return float(np.median(hit)), float(gy)
        return None

    # -- from the side ------------------------------------------------------------------

    def chord(self, start, direction) -> tuple[float, float]:
        """How far back and forward (m) the line from ``start`` (2,) along the unit
        ``direction`` (2,) runs across the ground's extent."""
        x0, y0, x1, y1 = self.layout.extent
        lo, hi = -math.inf, math.inf
        for s, d, a, b in ((start[0], direction[0], x0, x1), (start[1], direction[1], y0, y1)):
            if abs(d) < 1e-12:
                continue
            t0, t1 = (a - s) / d, (b - s) / d
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
        return min(lo, 0.0), max(hi, 0.0)

    def profile(self, start, end):
        """The profile the simulation's eigenrays take from ``start`` to ``end`` (x, y),
        as :class:`~acoustic_spaces.engine.refracted.Refraction` makes it: with weather
        zones the mean of the weathers along the way, their shares rounded as it rounds
        them; with one weather, its profile along the way, the way's angle to the wind
        rounded to the degree it takes. None in still air."""
        from acoustic_spaces.engine.refracted import AZIMUTH_STEP_DEG, MIX_STEP

        if self.air is None:
            return None
        a, b = self.air.shares(np.array([start[:2], end[:2]], dtype=float))
        if self.air.zoned:
            a, b = np.round(a / MIX_STEP) * MIX_STEP, np.round(b / MIX_STEP) * MIX_STEP
        else:
            angle = math.degrees(math.acos(min(max(float(b[0]), -1.0), 1.0)))
            angle = round(angle / AZIMUTH_STEP_DEG) * AZIMUTH_STEP_DEG
            b = np.array([round(math.cos(math.radians(angle)), 12)])
        total = float(a.sum())
        return self.air.profile(a / total, b / total)

    def eigenrays(self, source, listener, still: bool) -> list[np.ndarray]:
        """The eigenrays from ``source`` to ``listener`` (x, y, z) over the open ground
        as the simulation finds them: (k, 2) distance along the way and height above
        the ground each; straight in still air (the direct sound and its reflection
        off the ground). None over a terrain."""
        from acoustic_spaces.engine.refracted import MAX_GROUND_BOUNCES
        from acoustic_spaces.engine.refraction import Profile, Rays

        if not self.flat:
            return []
        zs = max(float(source[2]) - self.ground_z, 0.0)
        zr = max(float(listener[2]) - self.ground_z, 0.0)
        distance = math.dist(source[:2], listener[:2])
        if distance <= 1e-6:
            return []
        profile = None if still else self.profile(source, listener)
        key = (
            None if profile is None else (profile.c.tobytes(), profile.z.tobytes()),
            zs,
        )
        rays = self._kept.get(key)
        if rays is None:
            rays = Rays(Profile.uniform(self.c) if profile is None else profile, zs)
            self._kept[key] = rays
        bounces = 1 if profile is None else MAX_GROUND_BOUNCES
        found = rays.eigenrays(zr, distance, bounces)
        return [np.asarray(rays.curve(zr, found, i)) for i in range(len(found))]

    def fan(self, source, direction, t0: float, t1: float, still: bool, top: float):
        """The fan's rays from ``source`` (x, y, z) each way along ``direction`` (2,,
        unit), from ``t0`` to ``t1`` m along it: a list of ((n, 2) distance and height,
        (n,) the share of its energy left), each bending in the weather under it at
        each step and stopping at ``top`` m up, at the ends, or once too faint."""
        origin = np.asarray(source[:2], dtype=float)
        direction = np.asarray(direction, dtype=float)
        step = (t1 - t0) / FAN_STEPS
        out = []
        # The weather along the plane, looked up as often as the simulation looks it up.
        from acoustic_spaces.engine.weather import WEATHER_STEP_M

        grid = np.arange(t0, t1 + WEATHER_STEP_M, WEATHER_STEP_M)
        if self.air is not None:
            zones = self.air.zone_at(
                origin[None, :] + (grid[:, None] + 0.5 * WEATHER_STEP_M) * direction[None, :]
            )
        for way in (1.0, -1.0):
            tables = self._tables(direction * way)
            theta = np.radians(FAN_DEG)
            n = len(theta)
            t = np.zeros(n)
            z = np.full(n, max(float(source[2]) - self.ground_z, 0.0))
            energy = np.ones(n)
            going = np.ones(n, dtype=bool)
            ts, zs, energies, alive = [t.copy()], [z.copy()], [energy.copy()], [going.copy()]
            for _ in range(2 * FAN_STEPS):
                if not going.any():
                    break
                if tables and not still:
                    cell = np.clip(((t - t0) / WEATHER_STEP_M).astype(int), 0, len(grid) - 1)
                    zone = zones[cell]
                    bend = np.zeros(n)
                    for k in np.unique(zone):
                        pick = zone == k
                        profile = tables[int(k)]
                        layer = profile.layer(z[pick])
                        bend[pick] = profile.gradient[layer] / profile.speed(z[pick])
                    theta = theta - step * bend * np.cos(theta)
                t = t + way * step * np.cos(theta)
                z = z + step * np.sin(theta)
                low = going & (z < 0.0)
                for j in np.flatnonzero(low).tolist():
                    energy[j] *= self._bounce(origin + t[j] * direction, -theta[j])
                z = np.where(low, -z, z)
                theta = np.where(low, -theta, theta)
                ts.append(t.copy())
                zs.append(np.minimum(z, top))
                energies.append(energy.copy())
                alive.append(going.copy())
                going &= (z < top) & (t > t0) & (t < t1) & (energy > FAINTEST)
            t_at, z_at, energy_at, alive_at = (np.array(a) for a in (ts, zs, energies, alive))
            for j in range(n):
                kept = alive_at[:, j]
                out.append((np.column_stack([t_at[kept, j], z_at[kept, j]]), energy_at[kept, j]))
        return out

    def _bounce(self, point, grazing: float) -> float:
        """The share of a fan ray's energy left after it bounces off the ground at
        ``point`` (x, y) at the ``grazing`` angle (radians), :func:`reflected`; kept by
        the ground's zone and the angle to a milliradian, as fan rays bounce alike."""
        zone = int(self.layout.zone_at(np.asarray([point[:2]], dtype=float))[0])
        key = (zone, round(float(grazing), 3))
        if key not in self._bounces:
            name = self.materials[zone + 1]
            material = self.room.scene.materials.get(name) if name is not None else None
            self._bounces[key] = reflected(material, math.sin(key[1]), self.c)
        return self._bounces[key]

    def _tables(self, towards) -> list:
        """Each weather's profile along ``towards`` (2,, unit): the speed of sound in
        still air and the wind's component along it, as the parabolic equation takes it
        over that weather."""
        from acoustic_spaces.engine.refraction import Profile

        if self.air is None:
            return []
        cosines = products.product(self.air.towards, np.asarray(towards, dtype=float))
        return [
            Profile(self.air.nodes, self.air.still[k] + float(cosines[k]) * self.air.wind[k])
            for k in range(len(self.air))
        ]

    def bending(self, source, listener, still: bool) -> str:
        """How the sound bends on its way from ``source`` to ``listener``, from the
        profile the simulation's eigenrays take: down where the effective speed of sound
        grows with height (10 m up against the ground), up where it falls."""
        profile = None if still else self.profile(source, listener)
        if profile is None:
            return "in still air the sound goes straight"
        rise = float(profile.speed(10.0) - profile.speed(0.0))
        if rise > BENDS_M_S:
            return "the sound bends down on its way to you"
        if rise < -BENDS_M_S:
            return "the sound bends up on its way to you"
        return "the sound goes almost straight on its way to you"

    def side(
        self,
        img: Image.Image,
        box: tuple[float, float, float, float],
        source: dict,
        listener: dict,
        others: Sequence[dict] = (),
        *,
        still: bool = False,
    ) -> str:
        """The vertical plane through ``source`` and ``listener`` (scene data) across the
        ground, in ``box`` (left, top, width, height) on ``img``; ``others``, the other
        listeners, drawn where they stand near the plane. Returns what it shows, for the
        caption: how high it is drawn and how the sound bends."""
        left, top_px, width, height = box
        s = np.asarray(source["position"], dtype=float)
        r = np.asarray(listener["position"], dtype=float)
        distance = math.dist(s[:2], r[:2])
        direction = (r[:2] - s[:2]) / distance if distance > 1e-6 else np.array([1.0, 0.0])
        t0, t1 = self.chord(s[:2], direction)
        crossed = self.layout.crossed_along(s[:2] + t0 * direction, direction, t1 - t0)
        rays = [] if distance <= 1e-6 else self.eigenrays(s, r, still)
        blocked = [found for found in crossed if t0 + found[0] < distance and t0 + found[1] > 0.0]
        kept = [ray for ray in rays if not self._through(ray, crossed, t0)]
        tallest = max(
            [float(s[2]), float(r[2])]
            + [found[2] for found in crossed]
            + [plant.high for plant in self.layout.vegetation]
        )
        tallest -= self.ground_z
        highest = max((float(ray[:, 1].max()) for ray in kept), default=0.0)
        wanted = min(max(3.0 * tallest, 1.3 * highest, 10.0), self.sky)
        scale = width / (t1 - t0)
        # The most the heights can be stretched with the height wanted still in view.
        exaggeration = max([e for e in EXAGGERATIONS if wanted * scale * e <= height] or [1])
        top = min(height / (scale * exaggeration), self.sky)
        # Seen so that what lies left on the plan (or below it, the plane running north
        # and south) lies left here too.
        flipped = bool(direction[0] < -1e-9 or (abs(direction[0]) <= 1e-9 and direction[1] < 0))
        view = Side(left, top_px + height, width, t0, t1, top, exaggeration, flipped)
        self._air(img, view, s[:2], direction, still)
        self._ground(img, view, s[:2], direction, crossed)
        fan = self.fan(s, direction, t0, t1, still, top) if self.flat else []
        sight = (
            []
            if self.flat
            else [np.array([[0.0, s[2] - self.ground_z], [distance, r[2] - self.ground_z]])]
        )
        self._rays(img, view, fan, kept + sight, still)
        if not still:
            self._masts(img, view, s[:2], direction)
        draw = ImageDraw.Draw(img, "RGBA")
        sky_y = view(t0, top)[1]
        if top >= self.sky - 1e-6:
            for px in range(int(left), int(left + width), 14):
                draw.line([(px, sky_y), (px + 6, sky_y)], fill=FAINT, width=2)
            words = f"open sky, {self.sky:g} m up"
            text(draw, (left + 6, sky_y + 6), words, font("Medium", 14), FAINT)
        self._people(img, view, s, r, direction, distance, others, source["name"])
        if blocked:
            names = self._blockers(s[:2], direction, distance)
            bends = f"{names} in the way: the sound comes over and round it"
        elif not self.flat:
            ts = np.linspace(0.0, distance, 257)[1:-1]
            ground = self.layout.ground_at(s[None, :2] + ts[:, None] * direction[None, :])
            sight = s[2] + (r[2] - s[2]) * ts / max(distance, 1e-9)
            hidden = bool(np.any(ground > sight))
            bends = "the terrain hides the line of sight" if hidden else "the line of sight"
        else:
            bends = self.bending(s, r, still)
        return f"heights ×{exaggeration:g}: {bends}"

    def _blockers(self, start, direction, distance) -> str:
        """The names of what stands between ``start`` and ``distance`` m along
        ``direction``."""
        end = np.asarray(start) + distance * np.asarray(direction)
        names = []
        for name, footprint, _roof in self.layout.buildings:
            line = np.linspace(start, end, 64)
            if any(inside(p, footprint) for p in line):
                names.append(name.replace("_", " "))
        for name, path in self.layout.barriers:
            from acoustic_spaces.geometry.outdoor import line_crossings

            if line_crossings(np.asarray(start), np.asarray(direction), distance, path[:, :2]):
                names.append(name.replace("_", " "))
        return ", ".join(names) or "something"

    def _through(self, ray: np.ndarray, crossed, t0: float) -> bool:
        """Whether the eigenray ``ray`` (distance from the source, height) passes through
        a barrier or building ``crossed`` along the plane (from ``t0``)."""
        for near, far, roof, _ in crossed:
            near, far = near + t0, far + t0
            ts = np.linspace(near, far, 8) if far > near else np.array([near])
            inside_ray = (ts >= ray[0, 0]) & (ts <= ray[-1, 0])
            if not inside_ray.any():
                continue
            heights = np.interp(ts[inside_ray], ray[:, 0], ray[:, 1])
            if np.any(heights < roof - self.ground_z):
                return True
        return False

    def _air(self, img, view: Side, start, direction, still: bool) -> None:
        """The air above the plane, tinted blue where a weather is cooler at the ground
        than the scene's, orange where warmer, fading with height."""
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle(view.rect(view.t0, view.top, view.t1, 0.0), fill=AIR)
        if still or self.air is None or len(self.weathers) < 2:
            return
        haze = Image.new("RGBA", img.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(haze)
        for k, a, b in self._stretches(start, direction, view.t0, view.t1):
            warmer = self.grounds_c[k] - self.grounds_c[0]
            if abs(warmer) < 0.5:
                continue
            colour = WARM if warmer > 0 else COOL
            for j in range(24):
                h0, h1 = j * view.top / 24, (j + 1) * view.top / 24
                alpha = int(min(120.0, 18.0 * abs(warmer)) * (1 - j / 24) ** 2)
                hd.rectangle(view.rect(a, h1, b, h0), fill=colour + (alpha,))
        img.alpha_composite(haze)

    def _stretches(self, start, direction, t0, t1) -> list[tuple[int, float, float]]:
        """The weather under the plane from ``t0`` to ``t1`` m along it: (weather, from,
        to) for each stretch of one."""
        ts = np.linspace(t0, t1, 801)
        if self.air is None:
            return [(0, float(t0), float(t1))]
        zone = self.air.zone_at(np.asarray(start)[None, :] + ts[:, None] * direction[None, :])
        cuts = np.flatnonzero(np.diff(zone)) + 1
        bounds = [0, *cuts.tolist(), len(ts) - 1]
        pairs = zip(bounds[:-1], bounds[1:], strict=True)
        return [(int(zone[a]), float(ts[a]), float(ts[b])) for a, b in pairs]

    def _ground(self, img, view: Side, start, direction, crossed) -> None:
        """The ground along the plane in its materials' colours, the terrain's heights
        where it has one, and what stands on it: buildings, barriers and plants."""
        draw = ImageDraw.Draw(img, "RGBA")
        start = np.asarray(start, dtype=float)
        ts = np.linspace(view.t0, view.t1, 801)
        points = start[None, :] + ts[:, None] * direction[None, :]
        if self.flat:
            heights = np.zeros(len(ts))
        else:
            heights = self.layout.ground_at(points) - self.ground_z
        zone = self.layout.zone_at(points) + 1
        for j in range(len(ts) - 1):
            colour: tuple[int, ...] = GROUNDS[self.kinds[int(zone[j])]]
            colour = (
                WATER_EDGE if colour == WATER else tuple(min(255, int(c * 1.6)) for c in colour)
            )
            a, b = view(ts[j], heights[j]), view(ts[j + 1], heights[j + 1])
            draw.polygon([a, b, (b[0], b[1] + 10), (a[0], a[1] + 10)], fill=colour)
        for plant in self.layout.vegetation:
            inside_plant = plant.inside(
                np.column_stack([points, np.full(len(points), 0.5 * (plant.low + plant.high))])
            )
            for i in np.flatnonzero(inside_plant).tolist():
                draw.rectangle(
                    view.rect(
                        ts[i],
                        plant.high - self.ground_z,
                        ts[min(i + 1, len(ts) - 1)],
                        plant.low - self.ground_z,
                    ),
                    fill=PLANTS + (110,),
                )
        for near, far, roof, _ in crossed:
            near, far = near + view.t0, far + view.t0
            if far - near > 1e-6:
                draw.rectangle(
                    view.rect(near, roof - self.ground_z, far, 0.0),
                    fill=BUILDING + (200,),
                    outline=BUILDING_EDGE,
                )
            else:
                draw.line(
                    [view(near, 0.0), view(near, roof - self.ground_z)],
                    fill=BUILDING_EDGE,
                    width=4,
                )

    def _rays(self, img, view: Side, fan, kept, still: bool) -> None:
        """The fan faint, as much as each ray keeps of its energy, and the eigenrays
        bold over it."""
        colour = STILL if still or self.air is None else SOURCE
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        for points, energy in fan:
            pixels = [view(t, h) for t, h in points]
            for j in range(0, len(pixels) - 1, 4):
                alpha = int(110 * energy[j])
                if alpha < 8:
                    continue
                ld.line(
                    [pixels[j], pixels[min(j + 4, len(pixels) - 1)]],
                    fill=colour + (alpha,),
                    width=1,
                )
        img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(0.5)))
        draw = ImageDraw.Draw(img, "RGBA")
        for ray in kept:
            pixels = [view(t, h) for t, h in ray]
            draw.line(pixels, fill=colour + (235,), width=3, joint="curve")

    def _masts(self, img, view: Side, start, direction) -> None:
        """A mast in the middle of each weather's longest stretch under the plane, with
        an arrow at each of MAST_M the wind's component along the plane there."""
        if self.air is None:
            return
        draw = ImageDraw.Draw(img, "RGBA")
        longest: dict[int, tuple[float, float]] = {}
        for k, a, b in self._stretches(start, direction, view.t0, view.t1):
            if k not in longest or b - a > longest[k][1] - longest[k][0]:
                longest[k] = (a, b)
        for k, (a, b) in sorted(longest.items()):
            weather = self.weathers[k]
            along = products.dot(weather.to_wind(), np.asarray(direction, dtype=float))
            heights: list[float] = []
            for h in MAST_M:  # those under the top, far enough apart to tell
                if h <= 0.85 * view.top and (
                    not heights or view(0.0, heights[-1])[1] - view(0.0, h)[1] >= 18
                ):
                    heights.append(h)
            speeds = [float(weather.wind_speed(h)) * along for h in heights]
            if not any(abs(v) >= 0.05 for v in speeds):
                continue
            t = 0.5 * (a + b)
            draw.line([view(t, 0.0), view(t, 0.9 * view.top)], fill=TEXT + (120,), width=1)
            for h, v in zip(heights, speeds, strict=True):
                if abs(v) < 0.05:
                    continue
                sign = view.ahead if v > 0 else -view.ahead
                arrow(draw, view(t, h), (sign, 0.0), abs(v) * MAST_PX_PER_M_S, 7, WIND + (230,), 3)

    def _people(self, img, view: Side, s, r, direction, distance, others, name) -> None:
        """The source, the other listeners near the plane and the listener of the
        moment ("you")."""
        draw = ImageDraw.Draw(img, "RGBA")
        px, py = view(0.0, float(s[2]) - self.ground_z)
        glow(img, [(px, py, 12, SOURCE + (120,))], 5)
        draw = ImageDraw.Draw(img, "RGBA")
        draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=SOURCE, outline=BG)
        label = name.replace("_", " ")
        if view.flipped:
            text(draw, (px + 10, py), label, font("Medium", 14), SOURCE, "lm")
        else:
            text(draw, (px - 10, py), label, font("Medium", 14), SOURCE, "rm")
        across = np.array([-direction[1], direction[0]])
        reach = 0.04 * (view.t1 - view.t0)
        for other in others:
            p = np.asarray(other["position"], dtype=float)
            if abs(products.dot(p[:2] - s[:2], across)) > reach:
                continue
            ox, oy = view(products.dot(p[:2] - s[:2], direction), float(p[2]) - self.ground_z)
            draw.ellipse([ox - 5, oy - 5, ox + 5, oy + 5], outline=MUTED, width=2)
        lx, ly = view(distance, float(r[2]) - self.ground_z)
        glow(img, [(lx, ly, 22, LISTENER + (140,))], 8)
        draw = ImageDraw.Draw(img, "RGBA")
        draw.ellipse([lx - 8, ly - 8, lx + 8, ly + 8], fill=LISTENER, outline=BG, width=2)
        text(draw, (lx, ly - 16), "you", font("SemiBold", 16), LISTENER, "md")


__all__ = ["Outdoors", "Side", "ground_kind", "reflected"]
