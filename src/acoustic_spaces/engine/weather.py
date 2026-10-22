"""The weather over an outdoor scene where it changes from place to place.

An atmosphere (:class:`~acoustic_spaces.scene.Atmosphere`) can have ``zones``: parts of
the scene with wind and temperature of their own, such as the lee of a row of buildings,
where the wind drops, or the water of a lake, over which the air lies cooler. Outside the
zones the atmosphere's own weather holds. Sound that crosses from one weather into
another bends in each in turn:

- The early paths (:mod:`~acoustic_spaces.engine.refracted`) take the mean of the
  profiles along their way, each weather weighted by the length of the way under it,
  the wind's component along each leg with it: the equivalent profile of the
  engineering methods (Nord2000, Harmonoise), exact for the travel time of a straight
  path and a close approximation for the bending, which depends on the gradients along
  the way as a whole.
- The parabolic equation marches through each weather in turn, taking the profile of
  the zone it is over at each step, as the Green's function parabolic equation allows
  (Gilbert and Di 1993; Salomons 2001, ch. 7): exact to the equation's own accuracy.
- The wave solver gives each cell of its grid the speed of sound of the zone the cell
  stands over.
- The late rays, which sweep the whole scene, take the mean weather over the ground's
  extent, each zone weighted by its area, with the wind's mean vector.
- The eddies are the atmosphere's everywhere.
"""

from __future__ import annotations

import math

import numpy as np

from acoustic_spaces.engine.refraction import Profile, node_heights
from acoustic_spaces.geometry import UNIT_SCALE
from acoustic_spaces.geometry.builders import plan_problem
from acoustic_spaces.geometry.outdoor import clip_polygon, in_polygon

#: The weather along a way is looked up this often (m).
WEATHER_STEP_M = 1.0
#: The mean weather over the ground's extent samples each side this many times.
MEAN_SAMPLES = 128


class Weather:
    """The weathers of an outdoor scene: the atmosphere's own (index 0) and each of its
    zones' (1 on), which holds where, and the profiles along the ways sound takes.
    ``extent`` is the ground's (x_min, y_min, x_max, y_max) in metres and ``scale`` the
    geometry's units in metres, for the zones' polygons."""

    def __init__(self, atmosphere, ground_c: float, extent, scale: float = 1.0) -> None:
        self.atmosphere = atmosphere
        self.zoned = bool(atmosphere.zones)
        weathers = atmosphere.weathers()
        grounds = atmosphere.ground_temperatures(float(ground_c))
        nodes = np.zeros(0)
        for weather in weathers:
            nodes = np.union1d(nodes, node_heights(weather))
        self.nodes = nodes
        self.still = np.array(
            [w.sound_speed(nodes, g) for w, g in zip(weathers, grounds, strict=True)], dtype=float
        )  # (K, N)
        self.wind = np.array([w.wind_speed(nodes) for w in weathers], dtype=float)  # (K, N)
        self.towards = np.array([w.to_wind() for w in weathers], dtype=float)  # (K, 2)
        self.extent = tuple(float(v) for v in extent)
        self.polygons: list[np.ndarray | None] = []
        names = atmosphere.zone_names()
        for name, zone in zip(names, atmosphere.zones or [], strict=True):
            polygon = float(scale) * np.asarray(zone.polygon, dtype=float)
            problem = plan_problem(polygon)
            if problem is not None:
                words = problem.replace("a floor plan", "a zone")
                words = words.replace("the floor plan", "the zone")
                raise ValueError(f"weather zone {name!r}: {words}")
            self.polygons.append(clip_polygon(polygon, self.extent))

    @classmethod
    def of(cls, model) -> Weather | None:
        """The weather of ``model``'s scene (a :class:`~acoustic_spaces.engine.model.
        TraceModel`), or None where sound goes straight: indoors, or outdoors in still
        air."""
        scene = model.room.scene
        atmosphere = getattr(scene, "atmosphere", None)
        if model.room.outdoor is None or atmosphere is None or atmosphere.still:
            return None
        scale = UNIT_SCALE[scene.geometry.units or "m"]
        return cls(atmosphere, float(scene.air.temperature_c), model.room.outdoor.extent, scale)

    def __len__(self) -> int:
        return len(self.still)

    def zone_at(self, points: np.ndarray) -> np.ndarray:
        """(n,) which weather holds at each point (n, 2 or 3): 0 for the atmosphere's
        own, k for its zone k (the first where zones overlap)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        out = np.zeros(len(points), dtype=np.int64)
        for k, polygon in enumerate(self.polygons, start=1):
            if polygon is not None:
                out[(out == 0) & in_polygon(points, polygon)] = k
        return out

    def shares(self, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Along the horizontal polyline ``corners`` (m + 1, 2): (K,) the share of its
        length under each weather, and (K,) that share times the cosine of the
        weather's wind to the leg, summed over the legs (the wind's component along the
        way, over its speed)."""
        corners = np.asarray(corners, dtype=float).reshape(-1, 2)
        legs = np.diff(corners, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        total = float(lengths.sum())
        a = np.zeros(len(self))
        b = np.zeros(len(self))
        if total <= 0.0:
            a[self.zone_at(corners[:1])[0]] = 1.0
            return a, b
        for start, leg, length in zip(corners[:-1], legs, lengths, strict=True):
            if length <= 1e-12:
                continue
            n = max(1, int(math.ceil(length / WEATHER_STEP_M)))
            t = (np.arange(n) + 0.5) / n
            zone = self.zone_at(start[None, :] + t[:, None] * leg[None, :])
            share = np.bincount(zone, minlength=len(self)) * (length / n / total)
            a += share
            b += share * (self.towards @ (leg / length))
        return a, b

    def profile(self, a: np.ndarray, b: np.ndarray) -> Profile:
        """The mean profile of the weathers with the shares ``a`` and ``b`` (K,) of
        :meth:`shares`: the still air's speed and the wind's component, each weather
        weighted by its share."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return Profile(self.nodes, a @ self.still + b @ self.wind)

    def along(self, start: np.ndarray, direction: np.ndarray, reach: float):
        """A function giving, at a distance along ``direction`` (2,, unit) from
        ``start`` (2,), the speed function (of the height) of the weather there, the
        same object wherever the weather is the same; up to ``reach`` and a little
        past it."""
        direction = np.asarray(direction, dtype=float)
        n = int(math.ceil(max(float(reach), 0.0) / WEATHER_STEP_M)) + 2
        at = (np.arange(n) + 0.5) * WEATHER_STEP_M
        zone = self.zone_at(np.asarray(start, dtype=float)[None, :] + at[:, None] * direction)
        cosines = self.towards @ direction
        speeds = [
            Profile(self.nodes, self.still[k] + float(cosines[k]) * self.wind[k]).speed
            for k in range(len(self))
        ]
        last = n - 1
        return lambda r: speeds[zone[min(int(r / WEATHER_STEP_M), last)]]

    def mean_rows(self) -> tuple[np.ndarray, np.ndarray]:
        """The mean weather over the ground's extent, each weather weighted by its area:
        (N, 3) rows of height, speed of sound in still air and the wind's speed along
        (2,) the mean wind's direction (+x where there is no wind)."""
        x0, y0, x1, y1 = self.extent
        n = MEAN_SAMPLES
        xs = x0 + (np.arange(n) + 0.5) / n * (x1 - x0)
        ys = y0 + (np.arange(n) + 0.5) / n * (y1 - y0)
        points = np.column_stack([np.repeat(xs, n), np.tile(ys, n)])
        share = np.bincount(self.zone_at(points), minlength=len(self)) / float(n * n)
        still = share @ self.still
        vector = np.einsum("k,kn,kj->nj", share, self.wind, self.towards)  # (N, 2)
        total = vector.sum(axis=0)
        norm = float(np.linalg.norm(total))
        towards = total / norm if norm > 0.0 else np.array([1.0, 0.0])
        return np.column_stack([self.nodes, still, vector @ towards]), towards


__all__ = ["MEAN_SAMPLES", "WEATHER_STEP_M", "Weather"]
