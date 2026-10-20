"""Vegetation outdoors: trees, hedges and belts of trees, as volumes that take sound from
the paths through them and scatter some of it on.

Each volume (:class:`~acoustic_spaces.geometry.outdoor.PlantShape`: a polygon raised
between two heights, or a lone tree's crown, a spheroid) holds one or two media:

- Its foliage (``Vegetation.foliage``): an attenuation per metre and band, of which a
  share ``albedo`` is scattered on, with a Henyey-Greenstein phase function of mean
  cosine ``asymmetry``, and the rest absorbed. The ``dense`` preset is ISO 9613-2's
  dense foliage (Annex A, Table A.1, its rates for 20 to 200 m of it), all absorbed:
  the table is what dense foliage takes from sound passing through it in all, with
  what it scatters onwards counted in, so it adds no scattered sound of its own.
- Its stems (``Vegetation.stems``): upright rigid cylinders, ``per_m2`` per square
  metre of ground and ``diameter`` across, which absorb nothing and scatter, per
  metre of horizontal travel, their number times the scattering width of a rigid
  cylinder (the exact series for sound at right angles to its axis, Morse and Ingard
  1968), averaged over the band (Bullen and Fricke 1982 treat a wood's trunks so).
  Sound going up or down meets them along its horizontal share only, and they scatter
  it sideways, keeping its slope (the cone a cylinder scatters into), the angle
  through which they turn it drawn from a wrapped Cauchy distribution with the
  cylinder's mean cosine.

Where volumes overlap their media add up.

**Early paths** (the direct sound, reflections, paths round edges) lose
exp(-sum mu l) of their energy, with mu each medium's attenuation per metre
(absorption and scattering; for stems times the horizontal share of the path's leg)
and l the length of the leg inside it: Beer-Lambert.

**Late rays** (traced by the compiled tracer, on the CPU) are absorbed in the same way
along their way, and scatter in the plants at random, as often as the media scatter:
from where one scatters, foliage sends its share straight to each listener the way the
phase function has it (as a surface's scattered sound reaches them), stems what the
ray's next leg would carry through the listener's detector sphere, averaged over the
turns, and the ray goes on in a new direction. The sound the plants scatter is late
sound: the early paths lose it, the late part brings it.

The wave solver's bands leave vegetation out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces.geometry.outdoor import PlantShape, in_polygon
from acoustic_spaces.materials.bands import Resolution, band_limits

#: dB per neper of energy: 10 log10(e).
DB_PER_NEPER = 10.0 / math.log(10.0)
#: Frequencies per band over which the stems' scattering is averaged.
BAND_SAMPLES = 9
#: Angles at which a cylinder's scattering is summed for its mean cosine.
ANGLES = 512


@dataclass
class Medium:
    """What fills a volume, per band: absorption and scattering per metre (of the
    energy) and the scattering's mean cosine. ``upright`` media (stems) take them
    times the horizontal share of a path and scatter sideways."""

    absorption: np.ndarray
    scattering: np.ndarray
    asymmetry: np.ndarray
    upright: bool = False

    @property
    def extinction(self) -> np.ndarray:
        """Absorption and scattering per metre: what a path through it loses."""
        return self.absorption + self.scattering


def foliage_medium(foliage, resolution: Resolution) -> Medium | None:
    """The medium of a volume's ``foliage`` (:class:`~acoustic_spaces.scene.Foliage`) in
    the bands at ``resolution``; None for none."""
    if foliage.bare:
        return None
    total = foliage.attenuation_db(resolution) / DB_PER_NEPER
    return Medium(
        absorption=(1.0 - foliage.albedo) * total,
        scattering=foliage.albedo * total,
        asymmetry=np.full(len(total), float(foliage.asymmetry)),
    )


def cylinder_scattering(ka: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a rigid cylinder of radius a at ``ka`` (k the wavenumber), sound arriving at
    right angles to its axis: its scattering width over its diameter (the scattering
    efficiency, which tends to 2 as ka grows and to 3 pi^2 (ka)^3 / 8 as it shrinks)
    and the mean cosine of the angle it scatters through."""
    from scipy.special import jvp, yvp

    ka = np.atleast_1d(np.asarray(ka, dtype=float))
    # The orders that count at each ka (the rest are vanishingly small, and Y_n of a
    # high order at a small ka overflows).
    limit = np.ceil(ka + 4.0 * np.cbrt(ka) + 10.0)
    n = np.arange(int(limit.max()) + 1)[:, None]
    order, at = np.nonzero(n <= limit[None, :])
    b = np.zeros((len(n), len(ka)), dtype=complex)  # each order's share of the scattered wave
    j, y = jvp(order, ka[at]), yvp(order, ka[at])
    b[order, at] = j / (j + 1j * y)
    weight = np.where(n == 0, 1.0, 2.0)
    efficiency = 2.0 * np.sum(weight * np.abs(b) ** 2, axis=0) / ka
    theta = 2.0 * np.pi * np.arange(ANGLES) / ANGLES
    pattern = np.abs(np.einsum("nk,nt->kt", weight * b, np.cos(n * theta[None, :]))) ** 2
    cosine = np.sum(pattern * np.cos(theta), axis=1) / np.sum(pattern, axis=1)
    return efficiency, cosine


def stems_medium(stems, resolution: Resolution, speed_of_sound: float) -> Medium:
    """The medium of a volume's ``stems`` (:class:`~acoustic_spaces.scene.Stems`) in the
    bands at ``resolution``, for sound at ``speed_of_sound`` (m/s)."""
    lower, upper = band_limits(resolution)
    f = np.exp(
        np.linspace(np.log(lower), np.log(upper), BAND_SAMPLES + 1)[:-1]
        + 0.5 * np.log(upper / lower) / BAND_SAMPLES
    )  # (samples, bands), even in log frequency over each band
    radius = 0.5 * stems.diameter
    efficiency, cosine = cylinder_scattering((2.0 * np.pi * f * radius / speed_of_sound).ravel())
    efficiency, cosine = efficiency.reshape(f.shape), cosine.reshape(f.shape)
    width = stems.diameter * efficiency.mean(axis=0)
    return Medium(
        absorption=np.zeros(len(width)),
        scattering=stems.per_m2 * width,
        asymmetry=np.sum(efficiency * cosine, axis=0) / np.sum(efficiency, axis=0),
        upright=True,
    )


def _spheroid_lengths(shape: PlantShape, p, d, length) -> np.ndarray:
    """(n,) metres of each segment (from p along unit d for length) inside a crown."""
    axes = np.array([shape.radius, shape.radius, shape.half_height])
    q, e = (p - shape.centre) / axes, d / axes
    a = np.einsum("ij,ij->i", e, e)
    b = np.einsum("ij,ij->i", q, e)
    c = np.einsum("ij,ij->i", q, q) - 1.0
    disc = b * b - a * c
    root = np.sqrt(np.maximum(disc, 0.0))
    t0 = np.maximum((-b - root) / a, 0.0)
    t1 = np.minimum((-b + root) / a, length)
    return np.where(disc > 0.0, np.maximum(t1 - t0, 0.0), 0.0)


def _prism_lengths(shape: PlantShape, p, d, length) -> np.ndarray:
    """(n,) metres of each segment (from p along unit d for length) inside a polygon
    raised between two heights."""
    dz = d[:, 2]
    flat = np.abs(dz) < 1e-12
    within = (p[:, 2] > shape.low) & (p[:, 2] < shape.high)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (shape.low - p[:, 2]) / dz
        b = (shape.high - p[:, 2]) / dz
    ta = np.maximum(0.0, np.where(flat, np.where(within, 0.0, np.inf), np.minimum(a, b)))
    tb = np.minimum(length, np.where(flat, np.where(within, np.inf, -np.inf), np.maximum(a, b)))
    start, end = shape.outline, np.roll(shape.outline, -1, axis=0)
    edge = end - start
    origin, way = p[:, :2], d[:, :2]
    denom = way[:, None, 0] * edge[None, :, 1] - way[:, None, 1] * edge[None, :, 0]
    w = start[None] - origin[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * edge[None, :, 1] - w[..., 1] * edge[None, :, 0]) / denom
        s = (w[..., 0] * way[:, None, 1] - w[..., 1] * way[:, None, 0]) / denom
    crossing = (np.abs(denom) > 1e-12) & (s >= 0.0) & (s < 1.0)
    crossing &= (t > ta[:, None]) & (t < tb[:, None])
    cuts = np.sort(np.column_stack([ta, np.where(crossing, t, np.nan), tb]), axis=1)
    widths = np.diff(cuts, axis=1)
    middle = 0.5 * (cuts[:, :-1] + cuts[:, 1:])
    good = np.isfinite(widths) & (widths > 0.0)
    at = origin[:, None] + way[:, None] * np.where(good, middle, 0.0)[..., None]
    inside = in_polygon(at.reshape(-1, 2), shape.outline).reshape(widths.shape)
    return np.where(good & inside, widths, 0.0).sum(axis=1)


@dataclass
class Plants:
    """An outdoor scene's vegetation for one set of bands: its volumes, the media in
    them, and which fills which (``fills``: pairs of a shape's and a medium's index)."""

    shapes: tuple[PlantShape, ...]
    media: list[Medium]
    fills: list[tuple[int, int]]

    @classmethod
    def of(cls, room, resolution: Resolution, speed_of_sound: float) -> Plants | None:
        """The vegetation of ``room`` (an outdoor scene's ``geometry.vegetation``), or
        None without any."""
        if room.outdoor is None or not room.outdoor.vegetation:
            return None
        media, fills = [], []
        for index, spec in enumerate(room.scene.geometry.vegetation):
            for medium in (
                foliage_medium(spec.foliage, resolution),
                None
                if spec.stems is None
                else stems_medium(spec.stems, resolution, speed_of_sound),
            ):
                if medium is not None:
                    fills.append((index, len(media)))
                    media.append(medium)
        return cls(tuple(room.outdoor.vegetation), media, fills)

    @property
    def bands(self) -> int:
        return len(self.media[0].absorption)

    def lengths(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """(n, shapes) metres of each segment from ``starts`` to ``ends`` (n, 3) inside
        each volume."""
        starts, ends = np.atleast_2d(starts).astype(float), np.atleast_2d(ends).astype(float)
        span = ends - starts
        length = np.linalg.norm(span, axis=1)
        d = span / np.maximum(length, 1e-300)[:, None]
        out = np.zeros((len(starts), len(self.shapes)))
        for k, shape in enumerate(self.shapes):
            lo, hi = shape.corners().min(axis=0), shape.corners().max(axis=0)
            near = np.all(np.minimum(starts, ends) < hi, axis=1)
            near &= np.all(np.maximum(starts, ends) > lo, axis=1) & (length > 0.0)
            if not near.any():
                continue
            measure = _spheroid_lengths if shape.outline is None else _prism_lengths
            out[near, k] = measure(shape, starts[near], d[near], length[near])
        return out

    def depth(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """(n, bands) what each segment from ``starts`` to ``ends`` (n, 3) loses in the
        plants, in nepers of energy: sum of mu l."""
        starts, ends = np.atleast_2d(starts).astype(float), np.atleast_2d(ends).astype(float)
        inside = self.lengths(starts, ends)
        span = ends - starts
        flat = np.hypot(span[:, 0], span[:, 1]) / np.maximum(np.linalg.norm(span, axis=1), 1e-300)
        out = np.zeros((len(starts), self.bands))
        for shape, medium in self.fills:
            share = flat if self.media[medium].upright else 1.0
            out += (inside[:, shape] * share)[:, None] * self.media[medium].extinction[None]
        return out

    def attenuate(self, paths):
        """``paths`` (:class:`~acoustic_spaces.engine.early.EarlyReflections`) with each
        one's energy times exp(-sum mu l) along its corners."""
        if len(paths) == 0 or not paths.points:
            return paths
        legs = [len(p) - 1 for p in paths.points]
        starts = np.vstack([p[:-1] for p in paths.points])
        ends = np.vstack([p[1:] for p in paths.points])
        loss = np.zeros((len(paths), paths.energy.shape[1]))
        np.add.at(loss, np.repeat(np.arange(len(paths)), legs), self.depth(starts, ends))
        if not loss.any():
            return paths
        from dataclasses import replace

        return replace(paths, energy=paths.energy * np.exp(-loss))

    def native(self, band: slice) -> dict:
        """The arguments of ``_native.trace_stochastic`` and ``trace_map`` that put the
        plants in the late rays' way, for the bands ``band``: one row per volume and
        medium in it (kind 0 for a polygon raised between two heights: low, high, its
        first corner in ``plant_outline`` and their number; kind 1 for a crown: its
        centre, horizontal and vertical semi-axes), and each medium's absorption,
        scattering and mean cosine per band and whether it stands upright."""
        rows, outline = [], []
        for shape_index, medium in self.fills:
            shape = self.shapes[shape_index]
            if shape.outline is None:
                x, y, z = shape.centre
                rows.append([1, medium, x, y, z, shape.radius, shape.half_height, 0.0])
            else:
                first = sum(len(o) for o in outline)
                rows.append([0, medium, shape.low, shape.high, first, len(shape.outline), 0, 0])
                outline.append(shape.outline)

        def table(name: str) -> np.ndarray:
            return np.ascontiguousarray(
                [getattr(m, name)[band] for m in self.media], dtype=np.float64
            )

        return {
            "plants": np.ascontiguousarray(rows, dtype=np.float64),
            "plant_outline": np.ascontiguousarray(
                np.vstack(outline) if outline else np.zeros((0, 2)), dtype=np.float64
            ),
            "plant_absorption": table("absorption"),
            "plant_scattering": table("scattering"),
            "plant_asymmetry": table("asymmetry"),
            "plant_upright": np.array([m.upright for m in self.media], dtype=np.uint8),
        }


def plant_arguments(model, band: slice) -> dict:
    """``model.plants.native(band)``, or nothing without vegetation."""
    plants = getattr(model, "plants", None)
    return {} if plants is None else plants.native(band)


__all__ = [
    "DB_PER_NEPER",
    "Medium",
    "Plants",
    "cylinder_scattering",
    "foliage_medium",
    "plant_arguments",
    "stems_medium",
]
