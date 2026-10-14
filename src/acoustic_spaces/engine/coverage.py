"""Coverage maps: room acoustic parameters on a grid over the audience area.

The audience surfaces, picked by name, are covered with square columns ``spacing``
metres wide. Each column holds a map point at listening ``height`` above the surface
(two or more where surfaces lie above each other, such as a balcony over the stalls),
with a slab of air around it. For every source:

- the direct sound and the specular reflections up to ``simulation.early.max_order``
  are found exactly at each point, as at a listener;
- everything else comes from one stochastic ray tracing run in which every leg of
  every ray adds its track length through each slab it crosses
  (:func:`~acoustic_spaces.engine.late.stochastic_map_energy`), so a map costs about
  as much as one listener; with the statistical late method, an exponential decay.

Each point's parameters follow from its energy per band and time slot as at a
listener (ISO 3382-1), together with the speech transmission index (IEC 60268-16) and
sound pressure levels from the sources' sound power.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

from acoustic_spaces import parameters as iso3382
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import late as late_stage
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.simulate import (
    _as_room,
    _candidates,
    _Edges,
    _pick,
    _seed,
    mixing_time,
)
from acoustic_spaces.geometry import points_inside
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Coverage, Scene
from acoustic_spaces.scene.model import Simulation

#: Time resolution of each point's energy response (s).
TIME_STEP = 1e-3
#: Sub-samples per column side for finding the surfaces under it.
_SAMPLES = 5
#: Least part of a column over the surfaces that gets a map point.
_MIN_COVER = 0.2
#: Surfaces under one column closer than this (m) belong to the same layer.
_LAYER_GAP = 1.5
#: Thickest slab of air around a map point (m).
_MAX_THICKNESS = 1.0
#: Lowest slab bottom above the surface (m).
_ABOVE_SURFACE = 0.05
#: A-weighting of the octave bands (dB, IEC 61672-1, rounded).
A_WEIGHTING_DB = (-26.2, -16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1)


class MapParameter(NamedTuple):
    """How a map parameter is shown: ``label`` with its unit, what it is, whether it
    has a value per octave band, and the ``scale`` and number ``format`` of the shown
    value (the stored value is in SI units: Ts in seconds shows in milliseconds)."""

    label: str
    description: str
    per_band: bool
    scale: float
    format: str

    def text(self, value: float) -> str:
        return self.format.format(value * self.scale) if math.isfinite(value) else "–"


#: The parameters of every map point.
MAP_PARAMETERS: dict[str, MapParameter] = {
    "spl": MapParameter("SPL (dB)", "sound pressure level", True, 1.0, "{:.1f}"),
    "spl_a": MapParameter("SPL (dB(A))", "A-weighted sound pressure level", False, 1.0, "{:.1f}"),
    "g": MapParameter("G (dB)", "strength", True, 1.0, "{:.1f}"),
    "c50": MapParameter("C50 (dB)", "clarity, 50 ms", True, 1.0, "{:.1f}"),
    "c80": MapParameter("C80 (dB)", "clarity, 80 ms", True, 1.0, "{:.1f}"),
    "d50": MapParameter("D50", "definition", True, 1.0, "{:.2f}"),
    "edt": MapParameter("EDT (s)", "early decay time", True, 1.0, "{:.2f}"),
    "t20": MapParameter("T20 (s)", "reverberation time from a 20 dB decay", True, 1.0, "{:.2f}"),
    "t30": MapParameter("T30 (s)", "reverberation time from a 30 dB decay", True, 1.0, "{:.2f}"),
    "ts": MapParameter("Ts (ms)", "centre time", True, 1e3, "{:.0f}"),
    "sti": MapParameter("STI", "speech transmission index", False, 1.0, "{:.2f}"),
}


@dataclass
class MapGrid:
    """Map points over the audience area: columns ``cell`` metres wide from the
    corner ``origin`` (x, y), each holding up to ``index.shape[2]`` points."""

    origin: tuple[float, float]
    cell: float
    #: (ny, nx, layers) point index of each column and layer, -1 for none.
    index: np.ndarray
    #: (n, 3) the points, at listening height above the column's centre.
    points: np.ndarray
    #: (n, 2) column (x, y) of each point.
    columns: np.ndarray
    #: (n,) height of the surface under each point (m).
    surface: np.ndarray
    #: (n,) bottom and top of each point's slab of air (m), and the volume of the slab
    #: inside the room (m^3).
    low: np.ndarray
    high: np.ndarray
    volume: np.ndarray
    #: (m, 3, 2) the surfaces under the map seen from above: x, y of their triangles.
    footprint: np.ndarray
    #: (n, 25) which fifths of the column, by x then y, lie over the surfaces.
    cover: np.ndarray

    def __len__(self) -> int:
        return len(self.points)

    def squares(self) -> np.ndarray:
        """(n, 4, 3) corners of each point's column, at the point's height."""
        x0, y0 = self.origin
        x = x0 + self.columns[:, 0] * self.cell
        y = y0 + self.columns[:, 1] * self.cell
        d = self.cell
        z = self.points[:, 2]
        corners = [(x, y), (x + d, y), (x + d, y + d), (x, y + d)]
        return np.stack([np.column_stack([cx, cy, z]) for cx, cy in corners], axis=1)

    def tiles(self) -> tuple[np.ndarray, np.ndarray]:
        """Squares at the points' height over the part of each column above the
        surfaces: (m, 4, 3) corners and the point each belongs to. A column only partly
        over them is cut into fifths along each side."""
        full = self.cover.all(axis=1)
        whole = self.squares()[full]
        owner = [np.flatnonzero(full)]
        parts = [whole]
        partial = np.flatnonzero(~full)
        if len(partial):
            k, sub = np.nonzero(self.cover[partial])
            k = partial[k]
            d = self.cell / _SAMPLES
            x = self.origin[0] + self.columns[k, 0] * self.cell + (sub // _SAMPLES) * d
            y = self.origin[1] + self.columns[k, 1] * self.cell + (sub % _SAMPLES) * d
            z = self.points[k, 2]
            corners = [(x, y), (x + d, y), (x + d, y + d), (x, y + d)]
            parts.append(np.stack([np.column_stack([cx, cy, z]) for cx, cy in corners], axis=1))
            owner.append(k)
        return np.concatenate(parts), np.concatenate(owner)


def audience_surfaces(room: Room) -> list[str]:
    """The surfaces a map covers when none are named: those whose names contain
    "audience", else "floor"."""
    names = room.mesh.label_names()
    audience = [n for n in names if "audience" in n.lower()]
    if audience:
        return audience
    if "floor" in names:
        return ["floor"]
    raise ValueError(
        "name the surfaces the coverage map lies over (coverage: surfaces); this model has: "
        + ", ".join(names)
    )


def map_grid(
    room: Room,
    surfaces: Sequence[str] | None = None,
    *,
    height: float = 1.2,
    spacing: float = 1.0,
) -> MapGrid:
    """Map points ``spacing`` metres apart, ``height`` metres above the named surfaces
    (default: :func:`audience_surfaces`). Only faces that can be walked on count, not
    walls, and only points inside the room are kept."""
    if not (height > 0 and spacing > 0):
        raise ValueError("height and spacing must be positive")
    mesh = room.mesh
    names = list(surfaces) if surfaces else audience_surfaces(room)
    known = set(mesh.label_names())
    missing = [n for n in names if n not in known]
    if missing:
        raise ValueError(
            f"no surface named {', '.join(missing)}; this model has: {', '.join(sorted(known))}"
        )
    wanted = np.array([any(n in label for n in names) for label in mesh.labels], dtype=bool)
    faces = np.flatnonzero(wanted[mesh.face_labels] & (np.abs(mesh.face_normals()[:, 2]) > 0.25))
    if len(faces) == 0:
        raise ValueError(f"{', '.join(names)}: no surface there faces up")
    tri = mesh.triangles[faces]

    lo, hi = tri[:, :, :2].reshape(-1, 2).min(axis=0), tri[:, :, :2].reshape(-1, 2).max(axis=0)
    nx, ny = (np.maximum(1, np.ceil((hi - lo) / spacing - 1e-9)).astype(int)).tolist()
    # Centre the grid on the surfaces.
    origin = lo - 0.5 * (np.array([nx, ny]) * spacing - (hi - lo))
    step = spacing / _SAMPLES

    # Every sub-sample of every column with the height of each surface above or below it.
    hits_x, hits_y, hits_z = [], [], []
    for a, b, c in tri:
        t_lo = np.min([a[:2], b[:2], c[:2]], axis=0)
        t_hi = np.max([a[:2], b[:2], c[:2]], axis=0)
        g_lo = np.maximum(np.ceil((t_lo - origin) / step - 0.5), 0).astype(int)
        g_hi = np.minimum(
            np.floor((t_hi - origin) / step - 0.5), [nx * _SAMPLES - 1, ny * _SAMPLES - 1]
        ).astype(int)
        if np.any(g_hi < g_lo):
            continue
        gx, gy = np.meshgrid(
            np.arange(g_lo[0], g_hi[0] + 1), np.arange(g_lo[1], g_hi[1] + 1), indexing="ij"
        )
        gx, gy = gx.ravel(), gy.ravel()
        px = origin[0] + (gx + 0.5) * step
        py = origin[1] + (gy + 0.5) * step
        # Barycentric coordinates in the xy plane.
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        w_a = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / det
        w_b = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / det
        w_c = 1.0 - w_a - w_b
        inside = (w_a >= -1e-9) & (w_b >= -1e-9) & (w_c >= -1e-9)
        hits_x.append(gx[inside])
        hits_y.append(gy[inside])
        hits_z.append((w_a * a[2] + w_b * b[2] + w_c * c[2])[inside])
    if not hits_x:
        raise ValueError("the map surfaces are too small for the spacing; use a smaller one")
    sx, sy, sz = (np.concatenate(v) for v in (hits_x, hits_y, hits_z))

    centre = _SAMPLES // 2
    column = (sy // _SAMPLES) * nx + sx // _SAMPLES
    order = np.argsort(column, kind="stable")
    sx, sy, sz, column = sx[order], sy[order], sz[order], column[order]
    starts = np.flatnonzero(np.r_[True, column[1:] != column[:-1]])
    ends = np.r_[starts[1:], len(column)]

    # Per column, each level of surface under it (stalls, a balcony above them, ...):
    # its height, the part of the column over it, and where the point goes across:
    # the column's centre, or the middle of the part over the surface when the centre
    # is not over it (columns along a slanted wall).
    rows: list[tuple[int, int, int, float, float, float, float]] = []
    covers: list[np.ndarray] = []
    for s, e in zip(starts, ends, strict=True):
        ix, iy = int(sx[s] // _SAMPLES), int(sy[s] // _SAMPLES)
        sub_x, sub_y, z = sx[s:e] % _SAMPLES, sy[s:e] % _SAMPLES, sz[s:e]
        z_sorted = np.sort(z)
        breaks = np.flatnonzero(np.diff(z_sorted) > _LAYER_GAP)
        bottoms = z_sorted[np.r_[0, breaks + 1]]
        tops = z_sorted[np.r_[breaks, len(z_sorted) - 1]]
        for k, (bottom, top) in enumerate(zip(bottoms, tops, strict=True)):
            on = (z >= bottom) & (z <= top)
            sub = np.unique(sub_x[on] * _SAMPLES + sub_y[on])
            at_centre = on & (sub_x == centre) & (sub_y == centre)
            if at_centre.any():
                px, py, height_at = centre + 0.5, centre + 0.5, float(z[at_centre][0])
            else:
                px = float(np.mean(sub // _SAMPLES)) + 0.5
                py = float(np.mean(sub % _SAMPLES)) + 0.5
                height_at = float(np.mean(z[on]))
            if len(sub) < _MIN_COVER * _SAMPLES**2:
                continue  # a sliver: too few rays would cross its slab
            x = origin[0] + (ix + px / _SAMPLES) * spacing
            y = origin[1] + (iy + py / _SAMPLES) * spacing
            rows.append((ix, iy, k, height_at, len(sub) / _SAMPLES**2, x, y))
            cover = np.zeros(_SAMPLES**2, dtype=bool)
            cover[sub] = True
            covers.append(cover)

    if not rows:
        raise ValueError("the map surfaces are too small for the spacing; use a smaller one")
    thickness = min(spacing, _MAX_THICKNESS)
    ix, iy, layer = (np.array([r[i] for r in rows], dtype=np.int64) for i in range(3))
    surface, fraction, px, py = (np.array([r[i] for r in rows]) for i in range(3, 7))
    points = np.column_stack([px, py, surface + height])
    low = np.maximum(points[:, 2] - thickness / 2, surface + _ABOVE_SURFACE)
    high = low + thickness

    keep = np.ones(len(points), dtype=bool)
    if room.report.closed:
        keep = np.asarray(points_inside(mesh, points), dtype=bool)
    if not keep.any():
        raise ValueError("no map point lies inside the room; check height and surfaces")
    cover = np.array(covers)
    ix, iy, layer, surface, fraction, points, low, high, cover = (
        v[keep] for v in (ix, iy, layer, surface, fraction, points, low, high, cover)
    )
    # A slab counts the part of it inside the room: past the edge of the surfaces it
    # may go on over a stage or an aisle, or end at a wall.
    inside = fraction.copy()
    partial = np.flatnonzero(fraction < 1.0)
    if room.report.closed and len(partial):
        g = (np.arange(_SAMPLES) + 0.5) / _SAMPLES
        gx, gy = (v.ravel() for v in np.meshgrid(g, g, indexing="ij"))
        samples = np.stack(
            [
                np.column_stack(
                    [
                        origin[0] + (ix[k] + gx) * spacing,
                        origin[1] + (iy[k] + gy) * spacing,
                        np.full(len(gx), 0.5 * (low[k] + high[k])),
                    ]
                )
                for k in partial
            ]
        )
        share = points_inside(mesh, samples.reshape(-1, 3)).reshape(len(partial), -1).mean(axis=1)
        inside[partial] = np.maximum(share, fraction[partial])
    # Renumber the layers of each column from 0 after dropping points.
    layers = int(layer.max()) + 1
    index = np.full((ny, nx, layers), -1, dtype=np.int64)
    for k in range(len(points)):
        slot = int(np.flatnonzero(index[iy[k], ix[k]] < 0)[0])
        index[iy[k], ix[k], slot] = k
    layers = int((index >= 0).sum(axis=2).max())
    return MapGrid(
        origin=(float(origin[0]), float(origin[1])),
        cell=float(spacing),
        index=np.ascontiguousarray(index[:, :, :layers]),
        points=points,
        columns=np.column_stack([ix, iy]),
        surface=surface,
        low=low,
        high=high,
        volume=spacing**2 * inside * (high - low),
        footprint=np.ascontiguousarray(tri[:, :, :2]),
        cover=cover,
    )


@dataclass
class CoverageMap:
    """Parameters at every map point for one source, or for all of them together."""

    source: str
    grid: MapGrid
    #: Parameter name (see :data:`MAP_PARAMETERS`) -> (points, bands) or (points,).
    values: dict[str, np.ndarray]

    def value(self, name: str, band: int | None = None) -> np.ndarray:
        """One value per point: a parameter in one band (index into
        ``OCTAVE_BANDS_HZ``), or a parameter that has no bands."""
        if name not in self.values:
            raise ValueError(f"unknown parameter {name!r}; use one of {', '.join(self.values)}")
        v = self.values[name]
        if v.ndim == 1:
            return v
        if band is None:
            raise ValueError(f"{name} is per octave band; give a band")
        return v[:, band]

    def as_dict(self) -> dict:
        """Points and values as plain lists, None where undefined (JSON friendly)."""

        def plain(a: np.ndarray):
            rounded = np.round(a.astype(float), 4)
            return np.where(np.isfinite(rounded), rounded, None).tolist()

        return {
            "source": self.source,
            "bands_hz": list(OCTAVE_BANDS_HZ),
            "points": self.grid.points.round(3).tolist(),
            "values": {name: plain(v) for name, v in self.values.items()},
        }


@dataclass
class CoverageResult:
    """Coverage maps of every source, and of all of them together."""

    room: Room
    settings: Coverage
    grid: MapGrid
    maps: dict[str, CoverageMap]
    #: All sources playing together at their levels; None with one source.
    combined: CoverageMap | None = None
    details: dict[str, dict[str, float]] = field(default_factory=dict)

    def get(self, source: str | None = None) -> CoverageMap:
        """The map of one source; None gives all sources together (or the only one)."""
        if source is None:
            return self.combined if self.combined is not None else next(iter(self.maps.values()))
        if source not in self.maps:
            raise ValueError(f"unknown source {source!r}; use one of {', '.join(self.maps)}")
        return self.maps[source]

    def as_dict(self) -> dict:
        """The settings, the points and every map as plain lists (JSON friendly)."""
        maps = list(self.maps.values())
        if self.combined is not None:
            maps.append(self.combined)
        return {
            "scene": self.room.scene.name,
            "settings": self.settings.model_dump(mode="json"),
            "bands_hz": list(OCTAVE_BANDS_HZ),
            "points": self.grid.points.round(3).tolist(),
            "parameters": {name: p.label for name, p in MAP_PARAMETERS.items()},
            "maps": {
                m.source: {k: v for k, v in m.as_dict().items() if k not in ("points", "bands_hz")}
                for m in maps
            },
            "details": self.details,
        }


@dataclass
class _SourceRun:
    """Everything one source contributes at each point."""

    level_db: float
    onset: np.ndarray  # (n,) arrival of the direct sound
    early: list[tuple[np.ndarray, np.ndarray]]  # per point: delays, energies (paths, bands)
    late: late_stage.MapEnergy | None
    #: Statistical tail: (T60 per band, start per point, volume, speed of sound).
    statistical: tuple[np.ndarray, np.ndarray, float, float] | None


def _curve(run: _SourceRun, k: int, length: float, bands: int) -> np.ndarray:
    """Energy (bands, slots of ``TIME_STEP``) at point k, slot 0 at its direct sound."""
    onset = run.onset[k]
    slots = max(1, int(math.ceil((length - onset) / TIME_STEP - 1e-9)))
    out = np.zeros((bands, slots))
    delays, energy = run.early[k]
    idx = np.floor((delays - onset) / TIME_STEP + 1e-9).astype(np.int64)
    keep = (idx >= 0) & (idx < slots)
    for b in range(bands):
        np.add.at(out[b], idx[keep], energy[keep, b])
    late = run.late
    if late is not None:
        fine = min(late.fine_bins, slots)
        out[:, :fine] += late.energy[k, :, :fine]
        ratio = int(round(late.coarse_step / TIME_STEP))
        coarse = late.energy[k, :, late.fine_bins :]
        spread = np.repeat(coarse / ratio, ratio, axis=1)
        end = min(slots, late.fine_bins + spread.shape[1])
        out[:, late.fine_bins : end] += spread[:, : end - late.fine_bins]
    if run.statistical is not None:
        t60, start, volume, c = run.statistical
        rate = 6.0 * math.log(10.0) / t60[:, None]
        edges = onset + np.arange(slots + 1) * TIME_STEP
        lo = np.maximum(edges[:-1], start[k])
        hi = np.maximum(edges[1:], start[k])
        out += c / volume / rate * (np.exp(-rate * lo) - np.exp(-rate * hi))
    return out


def _point_values(
    energy: np.ndarray, power_db: np.ndarray, reference: float, noise_db
) -> dict[str, np.ndarray]:
    """All map parameters of one point's energy response (slot 0 at its direct sound);
    ``power_db`` is the source's sound power per band, ``reference`` the free-field
    energy at 10 m of a unit source."""
    p = iso3382.from_energy(energy, TIME_STEP, 0.0, reference_energy=reference)
    out = {name: getattr(p, name) for name in ("g", "c50", "c80", "d50", "edt", "t20", "t30", "ts")}
    spl = power_db + p.g + 10.0 * math.log10(reference)
    out["spl"] = spl
    weighted = spl + np.array(A_WEIGHTING_DB)
    out["spl_a"] = np.array(10.0 * np.log10(np.sum(10.0 ** (weighted / 10.0))))
    finite = np.isfinite(spl)
    signal = np.where(finite, spl, -100.0)
    out["sti"] = np.array(
        iso3382.speech_transmission_index(energy, TIME_STEP, signal_db=signal, noise_db=noise_db)
    )
    return out


def _band_power_db(level_db: float, bands: int) -> np.ndarray:
    """A source's sound power level spread evenly over the octave bands."""
    return np.full(bands, level_db - 10.0 * math.log10(bands))


def coverage_map(
    room: Room | Scene | str | Path,
    *,
    settings: Coverage | None = None,
    simulation: Simulation | None = None,
    sources: Sequence[str] | None = None,
    threads: int = 0,
    progress: Callable[[str], None] | None = None,
) -> CoverageResult:
    """Coverage maps of the scene's sources (all by default) over its audience area.

    ``settings`` default to ``scene.coverage`` (surfaces, height, spacing, rays and
    background noise); ``simulation`` (default: ``scene.simulation``) gives the early
    order, late method, response length and seed as for
    :func:`~acoustic_spaces.engine.simulate`. Each
    source radiates its ``level_db`` of sound power spread evenly over the octave
    bands; the speech transmission index takes that as the speech level.
    """
    room = _as_room(room)
    # Listeners play no part in a map.
    problems = [p for p in room.problems() if not p.startswith("listener ")]
    if problems:
        raise ValueError("; ".join(problems))
    scene = room.scene
    settings = settings or scene.coverage or Coverage()
    sim = simulation or scene.simulation
    say = progress or (lambda _: None)
    chosen = _pick(scene.sources, sources, "source")

    started = time.perf_counter()
    grid = map_grid(room, settings.surfaces, height=settings.height, spacing=settings.spacing)
    say(f"Map: {len(grid)} points, {grid.cell:g} m apart")
    model = TraceModel.from_room(room)
    edges = _Edges(model)
    c = room.speed_of_sound()
    air = room.air_attenuation()
    volume = room.volume_m3
    length = sim.ir_length_s
    bands = model.absorption.shape[1]
    seed = sim.seed
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint32)[0])
    statistical = sim.late.method == "statistical"
    if statistical and volume is None:
        raise ValueError("the statistical late method needs a closed model")
    if sim.transition_ms is not None:
        transition = sim.transition_ms * 1e-3
    elif statistical:
        transition = mixing_time(volume)
    else:
        transition = math.inf
    rays = settings.rays or sim.late.rays
    reference = 1.0 / (4.0 * math.pi * 100.0)
    details: dict[str, dict[str, float]] = {"map": {"points": float(len(grid))}}

    runs: dict[str, _SourceRun] = {}
    names = {source.name for source in chosen}
    for s_index, source in enumerate(scene.sources):
        if source.name not in names:
            continue
        info: dict[str, float] = {"seed": _seed(seed, s_index)}
        src = np.array(source.position, dtype=float)
        directivity = room.directivity(source.name)
        orientation = np.array(source.orientation.rotation())
        onset = np.linalg.norm(grid.points - src, axis=1) / c
        cutoff = np.minimum(onset + transition, length)

        t = time.perf_counter()
        say(f"{source.name}: early reflections at {len(grid)} points")
        candidates = _candidates(
            model, src, sim.early.method, sim.early.max_order, sim.early.rays, threads
        )
        diffraction = edges.diffraction(sim.early.diffraction, src, room, source.name, orientation)
        early = []
        diffracted = 0
        for k in range(len(grid)):
            paths = early_stage.find_paths(
                model,
                src,
                grid.points[k],
                candidates,
                speed_of_sound=c,
                air=air,
                max_delay_s=float(cutoff[k]),
                directivity=directivity,
                orientation=orientation,
            )
            if diffraction is not None:
                paths = diffraction.apply(grid.points[k], paths, float(cutoff[k]))
                diffracted += int(paths.diffracted.sum())
            early.append((paths.delay_s, paths.energy))
        if diffraction is not None:
            info["edges"] = float(len(diffraction.edges))
            info["diffracted_paths"] = float(diffracted)
        info["early_s"] = time.perf_counter() - t

        t = time.perf_counter()
        late = None
        stat = None
        if statistical:
            t60 = room.reverberation_time("eyring", air_absorption=True)
            stat = (np.asarray(t60, dtype=float), onset + transition, float(volume), c)
        else:
            say(f"{source.name}: stochastic ray tracing over the map ({rays} rays)")
            late = late_stage.stochastic_map_energy(
                model,
                src,
                grid,
                onset,
                rays=rays,
                max_time=length,
                speed_of_sound=c,
                air=air,
                covered_order=sim.early.max_order,
                covered_time=cutoff,
                splitting=sim.late.splitting,
                seed=info["seed"],
                directivity=directivity,
                orientation=orientation,
                threads=threads,
            )
            info["rays"] = float(late.rays)
            info["traced_rays"] = float(late.traced_rays)
        info["late_s"] = time.perf_counter() - t
        details[source.name] = info
        runs[source.name] = _SourceRun(source.level_db, onset, early, late, stat)

    say("Map: parameters at every point")
    noise = settings.noise_db
    per_source: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in runs}
    combined: list[dict[str, np.ndarray]] = []
    loudest = max(run.level_db for run in runs.values())
    for k in range(len(grid)):
        curves = {}
        for name, run in runs.items():
            curve = _curve(run, k, length, bands)
            curves[name] = curve
            power = _band_power_db(run.level_db, bands)
            per_source[name].append(_point_values(curve, power, reference, noise))
        if len(runs) > 1:
            # All sources together, each at its level relative to the loudest, on the
            # time axis of the earliest direct sound.
            first = min(run.onset[k] for run in runs.values())
            slots = int(math.ceil((length - first) / TIME_STEP - 1e-9))
            total = np.zeros((bands, slots))
            for name, run in runs.items():
                shift = int(round((run.onset[k] - first) / TIME_STEP))
                curve = curves[name][:, : max(0, slots - shift)]
                weight = 10.0 ** ((run.level_db - loudest) / 10.0)
                total[:, shift : shift + curve.shape[1]] += weight * curve
            combined.append(_point_values(total, _band_power_db(loudest, bands), reference, noise))

    def stack(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        return {name: np.array([r[name] for r in rows]) for name in MAP_PARAMETERS}

    maps = {name: CoverageMap(name, grid, stack(rows)) for name, rows in per_source.items()}
    together = CoverageMap("all sources", grid, stack(combined)) if combined else None
    details["map"]["seconds"] = time.perf_counter() - started
    return CoverageResult(room, settings, grid, maps, together, details)


__all__ = [
    "A_WEIGHTING_DB",
    "MAP_PARAMETERS",
    "TIME_STEP",
    "CoverageMap",
    "CoverageResult",
    "MapGrid",
    "MapParameter",
    "audience_surfaces",
    "coverage_map",
    "map_grid",
]
