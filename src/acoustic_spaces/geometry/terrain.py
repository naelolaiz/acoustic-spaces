"""Map data for outdoor scenes: terrain as heights on a regular grid, and buildings as
footprints from a map.

A :class:`HeightGrid` holds the ground's height at the nodes of a square grid, from an
Esri ASCII grid (``.asc``, what GIS tools and ``gdal_translate -of AAIGrid`` write) or
from a table of ``x y z`` points on a regular grid (``.xyz``, ``.txt`` or ``.csv``), or
given in the scene file. Between the nodes the ground is bilinear; past the grid's edge
it goes on at the height of its edge. The scene's ground is the grid sampled over the
ground's extent (see :func:`acoustic_spaces.geometry.outdoor.terrain_box`).

Buildings come from GeoJSON (``.geojson`` or ``.json``): each ``Polygon`` feature, and
each part of a ``MultiPolygon``, is a building with its outer ring as its footprint and
its height in a property (``height`` by default). The coordinates must be in the
scene's units on a plane (a projected coordinate system, such as UTM), not in degrees of
longitude and latitude; :data:`LONLAT_EXTENT` tells the two apart.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Nodes this close (relative to the spacing) to a grid line lie on it.
_GRID_TOLERANCE = 1e-6
#: A footprint narrower than this (in the file's units) is a building drawn in degrees
#: of longitude and latitude, where a house is a ten-thousandth of a degree across.
LONLAT_EXTENT = 0.01


@dataclass(frozen=True, eq=False)
class HeightGrid:
    """Heights on a square grid: ``heights[j, i]`` at ``(origin[0] + i spacing,
    origin[1] + j spacing)``, rows along y and columns along x."""

    origin: tuple[float, float]
    spacing: float
    heights: np.ndarray

    def __post_init__(self) -> None:
        heights = np.asarray(self.heights, dtype=float)
        if heights.ndim != 2 or heights.shape[0] < 2 or heights.shape[1] < 2:
            raise ValueError("a terrain needs at least two rows and two columns of heights")
        if not np.all(np.isfinite(heights)):
            raise ValueError("every height of the terrain must be a number")
        if not self.spacing > 0.0:
            raise ValueError("the terrain's spacing must be positive")
        object.__setattr__(self, "heights", heights)
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))
        object.__setattr__(self, "spacing", float(self.spacing))

    @property
    def shape(self) -> tuple[int, int]:
        """(columns along x, rows along y)."""
        return self.heights.shape[1], self.heights.shape[0]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of the grid's nodes."""
        nx, ny = self.shape
        x0, y0 = self.origin
        return x0, y0, x0 + (nx - 1) * self.spacing, y0 + (ny - 1) * self.spacing

    @property
    def lowest(self) -> float:
        return float(self.heights.min())

    @property
    def highest(self) -> float:
        return float(self.heights.max())

    def scaled(self, scale: float, offset: float = 0.0) -> HeightGrid:
        """The grid with its coordinates and heights times ``scale`` and ``offset`` added
        to the heights."""
        return HeightGrid(
            (scale * self.origin[0], scale * self.origin[1]),
            scale * self.spacing,
            scale * self.heights + offset,
        )

    def shifted(self, dx: float, dy: float) -> HeightGrid:
        """The grid moved by (dx, dy)."""
        return HeightGrid((self.origin[0] + dx, self.origin[1] + dy), self.spacing, self.heights)

    def height_at(self, points) -> np.ndarray:
        """(n,) the ground's height at each point (n, 2 or 3): bilinear between the
        nodes, and past the grid's edge the height at its edge."""
        points = np.atleast_2d(np.asarray(points, dtype=float))[:, :2]
        nx, ny = self.shape
        u = np.clip((points[:, 0] - self.origin[0]) / self.spacing, 0.0, nx - 1.0)
        v = np.clip((points[:, 1] - self.origin[1]) / self.spacing, 0.0, ny - 1.0)
        i = np.minimum(np.floor(u).astype(np.int64), nx - 2)
        j = np.minimum(np.floor(v).astype(np.int64), ny - 2)
        fu, fv = u - i, v - j
        h = self.heights
        return (
            (1.0 - fu) * (1.0 - fv) * h[j, i]
            + fu * (1.0 - fv) * h[j, i + 1]
            + (1.0 - fu) * fv * h[j + 1, i]
            + fu * fv * h[j + 1, i + 1]
        )

    def _cells(self, extent) -> tuple[slice, slice]:
        """The rows and columns of the nodes of the cells covering ``extent``."""
        x0, y0, x1, y1 = extent
        nx, ny = self.shape
        d = self.spacing
        i0 = int(np.clip(math.floor((x0 - self.origin[0]) / d + _GRID_TOLERANCE), 0, nx - 1))
        i1 = int(np.clip(math.ceil((x1 - self.origin[0]) / d - _GRID_TOLERANCE), 0, nx - 1))
        j0 = int(np.clip(math.floor((y0 - self.origin[1]) / d + _GRID_TOLERANCE), 0, ny - 1))
        j1 = int(np.clip(math.ceil((y1 - self.origin[1]) / d - _GRID_TOLERANCE), 0, ny - 1))
        return slice(j0, j1 + 1), slice(i0, i1 + 1)

    def lowest_over(self, extent) -> float:
        """The lowest node of the cells covering ``extent`` (x_min, y_min, x_max,
        y_max): a bound on the ground's height there."""
        rows, columns = self._cells(extent)
        return float(self.heights[rows, columns].min())

    def highest_over(self, extent) -> float:
        """The highest node of the cells covering ``extent``."""
        rows, columns = self._cells(extent)
        return float(self.heights[rows, columns].max())

    def describe(self, units: str = "m") -> str:
        nx, ny = self.shape
        return (
            f"{nx} x {ny} points {self.spacing:g} {units} apart, from z = {self.lowest:g} to "
            f"{self.highest:g} {units}"
        )


#: The file types read as terrain, by suffix.
TERRAIN_SUFFIXES = (".asc", ".xyz", ".txt", ".csv")


def read_height_grid(path: str | Path) -> HeightGrid:
    """The grid in ``path``: an Esri ASCII grid (``.asc``) or ``x y z`` points on a
    regular grid (``.xyz``, ``.txt``, ``.csv``; whitespace or commas between them,
    ``#`` comments)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".asc":
        return _read_asc(path)
    if suffix in TERRAIN_SUFFIXES:
        return _read_xyz(path)
    raise ValueError(
        f"{path.name}: a terrain is an Esri ASCII grid (.asc) or x y z points on a regular "
        "grid (.xyz, .txt or .csv); convert other rasters with gdal_translate -of AAIGrid"
    )


def _read_asc(path: Path) -> HeightGrid:
    """An Esri ASCII grid: a header of ncols, nrows, the lower-left corner or centre,
    cellsize and a NODATA value, then the rows from the top (north) down."""
    header: dict[str, float] = {}
    values: list[float] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].lower()
            if not values and key in (
                "ncols",
                "nrows",
                "xllcorner",
                "yllcorner",
                "xllcenter",
                "yllcenter",
                "cellsize",
                "nodata_value",
                "dx",
                "dy",
            ):
                if len(parts) != 2:
                    raise ValueError(f"{path.name}: header line {line.strip()!r} needs one value")
                try:
                    header[key] = float(parts[1])
                except ValueError:
                    raise ValueError(f"{path.name}: {parts[0]} {parts[1]!r} is no number") from None
                continue
            try:
                values.extend(float(v) for v in parts)
            except ValueError:
                raise ValueError(
                    f"{path.name}: {line.strip()!r} holds something that is no number"
                ) from None
    for key in ("ncols", "nrows"):
        if key not in header:
            raise ValueError(f"{path.name}: the header lacks {key}")
    if "cellsize" in header:
        spacing = header["cellsize"]
    elif "dx" in header and "dy" in header:
        if not math.isclose(header["dx"], header["dy"], rel_tol=_GRID_TOLERANCE):
            raise ValueError(f"{path.name}: dx and dy differ; the grid must be square")
        spacing = header["dx"]
    else:
        raise ValueError(f"{path.name}: the header lacks cellsize")
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    if ncols < 2 or nrows < 2:
        raise ValueError(f"{path.name}: a terrain needs at least two rows and two columns")
    if len(values) != ncols * nrows:
        raise ValueError(
            f"{path.name}: {ncols} columns x {nrows} rows need {ncols * nrows} heights, "
            f"the file holds {len(values)}"
        )
    heights = np.array(values).reshape(nrows, ncols)[::-1]  # the first row is the north one
    if "nodata_value" in header:
        missing = int(np.count_nonzero(heights == header["nodata_value"]))
        if missing:
            raise ValueError(
                f"{path.name}: {missing} of the heights are NODATA; fill them in (the "
                "terrain must have a height everywhere)"
            )
    if "xllcorner" in header or "yllcorner" in header:
        if not ("xllcorner" in header and "yllcorner" in header):
            raise ValueError(f"{path.name}: the header needs both xllcorner and yllcorner")
        origin = (header["xllcorner"] + 0.5 * spacing, header["yllcorner"] + 0.5 * spacing)
    elif "xllcenter" in header and "yllcenter" in header:
        origin = (header["xllcenter"], header["yllcenter"])
    else:
        raise ValueError(f"{path.name}: the header lacks xllcorner and yllcorner")
    try:
        return HeightGrid(origin, spacing, heights)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None


def _read_xyz(path: Path) -> HeightGrid:
    """``x y z`` points on a regular grid, in any order, whitespace or commas between
    the numbers, lines starting with # left out, and a header line of names allowed."""
    rows = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for number, line in enumerate(fh, 1):
            text = line.split("#", 1)[0].strip()
            if not text:
                continue
            parts = text.replace(",", " ").replace(";", " ").split()
            try:
                values = [float(v) for v in parts]
            except ValueError:
                if number == 1 or not rows:
                    continue  # a header line of names
                raise ValueError(
                    f"{path.name}: line {number} holds something that is no number"
                ) from None
            if len(values) < 3:
                raise ValueError(f"{path.name}: line {number} needs x, y and z")
            rows.append(values[:3])
    if len(rows) < 4:
        raise ValueError(f"{path.name}: a terrain needs at least two rows and two columns")
    points = np.array(rows)
    xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError(f"{path.name}: a terrain needs at least two rows and two columns")
    dx, dy = np.diff(xs), np.diff(ys)
    spacing = float(min(dx.min(), dy.min()))
    if spacing <= 0.0:
        raise ValueError(f"{path.name}: the points' spacing is zero")
    # Every x and y must sit on a grid line, and every node must be there once.
    i = (points[:, 0] - xs[0]) / spacing
    j = (points[:, 1] - ys[0]) / spacing
    if not (
        np.all(np.abs(i - np.round(i)) <= _GRID_TOLERANCE)
        and np.all(np.abs(j - np.round(j)) <= _GRID_TOLERANCE)
    ):
        raise ValueError(
            f"{path.name}: the points are not on a regular grid (the same spacing along x "
            "and y, every point on a grid line)"
        )
    i, j = np.round(i).astype(np.int64), np.round(j).astype(np.int64)
    nx, ny = int(i.max()) + 1, int(j.max()) + 1
    heights = np.full((ny, nx), np.nan)
    heights[j, i] = points[:, 2]
    missing = int(np.count_nonzero(np.isnan(heights)))
    if missing:
        raise ValueError(
            f"{path.name}: the grid of {nx} x {ny} points lacks {missing} of them; every "
            "node needs a height"
        )
    if len(points) != nx * ny:
        raise ValueError(f"{path.name}: some points are given twice")
    try:
        return HeightGrid((float(xs[0]), float(ys[0])), spacing, heights)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None


def write_asc(path: str | Path, grid: HeightGrid) -> None:
    """Write ``grid`` as an Esri ASCII grid (the rows from the north one down)."""
    nx, ny = grid.shape
    lines = [
        f"ncols {nx}",
        f"nrows {ny}",
        f"xllcenter {grid.origin[0]:g}",
        f"yllcenter {grid.origin[1]:g}",
        f"cellsize {grid.spacing:g}",
        "NODATA_value -9999",
    ]
    lines += [" ".join(f"{v:g}" for v in row) for row in grid.heights[::-1]]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- buildings from a map ---------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Footprint:
    """A building read from a map: its name, footprint (n, 2) and height."""

    name: str
    footprint: np.ndarray
    height: float
    #: How many holes (courtyards) the footprint had, which are built over.
    holes: int = 0


def read_footprints(
    path: str | Path,
    height: str | float = "height",
    name: str = "name",
    default_height: float | None = None,
) -> list[Footprint]:
    """The buildings in a GeoJSON file: one per ``Polygon`` feature and per part of a
    ``MultiPolygon`` (named ``<name>_1``, ``<name>_2``...), with the property ``height``
    (or that height for all of them) and the property ``name`` (features without one are
    unnamed: ``""``). Features whose height is missing take ``default_height``, or are
    refused without one. Holes in a footprint are built over. The coordinates must be
    on a plane, not degrees of longitude and latitude."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: not GeoJSON (its top level is no object)")
    kind = data.get("type")
    if kind == "FeatureCollection":
        features = data.get("features")
        if not isinstance(features, list):
            raise ValueError(f"{path.name}: the FeatureCollection has no features list")
    elif kind == "Feature":
        features = [data]
    elif kind in ("Polygon", "MultiPolygon"):
        features = [{"type": "Feature", "geometry": data, "properties": {}}]
    else:
        raise ValueError(
            f"{path.name}: not GeoJSON buildings (a FeatureCollection of Polygon or "
            f"MultiPolygon features); its type is {kind!r}"
        )
    out: list[Footprint] = []
    for number, feature in enumerate(features, 1):
        if not isinstance(feature, dict):
            raise ValueError(f"{path.name}: feature {number} is no object")
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            continue  # a feature without a geometry: nothing to build
        label = str(properties.get(name) or "") if name else ""
        where = f"feature {number}" + (f" ({label!r})" if label else "")
        if isinstance(height, str):
            value = properties.get(height, default_height)
            if value is None:
                raise ValueError(
                    f"{path.name}: {where} has no property {height!r}; add it, name the "
                    "property that holds the height, or set a default height"
                )
            try:
                building_height = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{path.name}: {where}: height {value!r} is no number") from None
        else:
            building_height = float(height)
        if not building_height > 0.0:
            raise ValueError(f"{path.name}: {where}: the height must be positive")
        shape = geometry.get("type")
        if shape == "Polygon":
            polygons = [geometry.get("coordinates")]
        elif shape == "MultiPolygon":
            polygons = geometry.get("coordinates") or []
        else:
            continue  # points and lines are no buildings
        parts = []
        for polygon in polygons:
            if not isinstance(polygon, list) or not polygon:
                raise ValueError(f"{path.name}: {where}: the polygon has no rings")
            ring = _ring(polygon[0], path, where)
            parts.append((ring, len(polygon) - 1))
        for k, (ring, holes) in enumerate(parts):
            part_name = label if len(parts) == 1 else (f"{label}_{k + 1}" if label else "")
            out.append(Footprint(part_name, ring, building_height, holes))
    return out


def _ring(coordinates, path: Path, where: str) -> np.ndarray:
    """A ring's corners (n, 2), its closing point dropped, in the file's units."""
    try:
        ring = np.array([[float(p[0]), float(p[1])] for p in coordinates])
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{path.name}: {where}: the ring is no list of [x, y] points") from None
    if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
        ring = ring[:-1]
    if len(ring) < 3:
        raise ValueError(f"{path.name}: {where}: a footprint needs at least three corners")
    if np.all(np.ptp(ring, axis=0) < LONLAT_EXTENT) and np.all(np.abs(ring[:, 0]) <= 180.0):
        raise ValueError(
            f"{path.name}: {where} is {np.ptp(ring, axis=0).max():g} units across: the "
            "coordinates look like degrees of longitude and latitude; project the file to "
            "metres on a plane first (ogr2ogr -t_srs with a UTM or national grid)"
        )
    return ring


__all__ = [
    "LONLAT_EXTENT",
    "TERRAIN_SUFFIXES",
    "Footprint",
    "HeightGrid",
    "read_footprints",
    "read_height_grid",
    "write_asc",
]
