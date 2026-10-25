"""The scene being edited: its file, its room, and the edits made to it.

Every change goes through :meth:`Document.edit`, which changes a plain copy of the
scene, validates it with the scene schema, loads the room (reusing the mesh when the
geometry did not change) and only then makes it current. A change the schema or the
geometry rejects raises and leaves the document as it was, so the editor can show the
reason. Undo and redo step through whole scenes, which are small.
"""

from __future__ import annotations

import contextlib
import dataclasses
import filecmp
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from acoustic_spaces.directivity import array_files
from acoustic_spaces.geometry import UNIT_SCALE, RoomMesh, extruded_mesh, points_inside
from acoustic_spaces.geometry.outdoor import SKY
from acoustic_spaces.i18n import _, ngettext
from acoustic_spaces.room import Room, matching_name
from acoustic_spaces.scene import Scene
from acoustic_spaces.scene.model import INDOOR_COHERENT_MAX_BAND_HZ, directivity_entries

#: Heights above a floor for new sources and listeners (m): a talker, a seated ear.
SOURCE_HEIGHT = 1.5
LISTENER_HEIGHT = 1.2
#: Undo steps kept.
HISTORY = 200
#: Height of a drawn room when the scene had none to keep (m).
DRAWN_HEIGHT = 3.0
#: The ground of a new outdoor scene, metres across.
NEW_GROUND_SIZE = 200.0
#: Sizes (m) of the parts drawn in the plan of an outdoor scene, to change afterwards:
#: a building's height, a barrier's, a belt of trees', a hedge's height and width, and
#: a lone tree's height, where its crown starts and how wide it is.
NEW_PART_SIZES = {
    "building": {"height": 10.0},
    "barrier": {"height": 3.0},
    "belt": {"height": 12.0},
    "hedge": {"height": 2.0, "width": 1.5},
    "tree": {"height": 10.0, "base": 2.5, "crown_diameter": 8.0},
}
#: A dragged source or listener keeps its height above the floor while the floor under
#: it rises or falls by at most this much (m) from one place to the next (a raked
#: floor); past a bigger step (a balcony edge) it keeps its height instead.
FOLLOW_STEP = 1.0
#: Decimals kept in coordinates the editor writes (a micrometre).
DECIMALS = 6


class EditError(ValueError):
    """A change the scene schema or the geometry refused; the message says why."""


def _message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors():
            where = ".".join(str(p) for p in error["loc"])
            text = error["msg"].removeprefix("Value error, ")
            parts.append(f"{where}: {text}" if where else text)
        return "; ".join(parts)
    return str(exc)


def box_scene(size: tuple[float, float, float] = (8.0, 6.0, 3.0)) -> Scene:
    """A new rectangular room with one material, a source and a listener."""
    length, width, height = size
    return Scene.model_validate(
        {
            "name": "untitled",
            "geometry": {"box": list(size)},
            "materials": {
                "plaster": {"preset": "plasterboard", "scattering": 0.1},
                "floor": {"preset": "concrete_floor", "scattering": 0.05},
            },
            "assignments": {"floor": "floor"},
            "default_material": "plaster",
            "sources": [
                {"name": "source", "position": [length * 0.25, width * 0.5, SOURCE_HEIGHT]}
            ],
            "listeners": [
                {
                    "name": "listener",
                    "position": [length * 0.7, width * 0.5, LISTENER_HEIGHT],
                    "orientation": {"yaw": 180.0},
                }
            ],
        }
    )


def outdoor_scene(size: float = NEW_GROUND_SIZE) -> Scene:
    """A new outdoor scene: a flat meadow ``size`` m across under an open sky, with a
    source and a listener 50 m apart in the middle of it, facing each other."""
    half = size / 2.0
    return Scene.model_validate(
        {
            "name": "untitled",
            "geometry": {"ground": {"extent": [[-half, -half], [half, half]]}},
            "materials": {"meadow": {"impedance": "grass"}},
            "default_material": "meadow",
            "sources": [{"name": "source", "position": [-25.0, 0.0, SOURCE_HEIGHT]}],
            "listeners": [
                {
                    "name": "listener",
                    "position": [25.0, 0.0, LISTENER_HEIGHT],
                    "orientation": {"yaw": 180.0},
                }
            ],
        }
    )


def model_scene(path: str | Path) -> Scene:
    """A new scene around a model file: one material for every surface, and a source and
    a listener inside the model (when it is closed; otherwise around its middle)."""
    path = Path(path).resolve()
    data = {
        "name": path.stem,
        "geometry": {"file": str(path)},
        "materials": {"walls": {"preset": "plasterboard", "scattering": 0.1}},
        "default_material": "walls",
        "sources": [{"name": "source", "position": [0.0, 0.0, 0.0]}],
        "listeners": [{"name": "listener", "position": [0.0, 0.0, 0.0]}],
    }
    try:
        room = Room.from_scene(Scene.model_validate(data))
    except (ValidationError, ValueError, OSError) as exc:
        raise EditError(_message(exc)) from exc
    lo = np.array(room.report.bounds_min, dtype=float)
    hi = np.array(room.report.bounds_max, dtype=float)
    steps = [np.linspace(a, b, 9)[1:-1] for a, b in zip(lo, hi, strict=True)]
    grid = np.stack(np.meshgrid(*steps, indexing="ij"), axis=-1).reshape(-1, 3)
    if room.report.closed:
        inside = grid[points_inside(room.mesh, grid)]
        grid = inside if len(inside) else grid
    # On the lowest level of the grid, along the middle of the longest side: a quarter
    # and three quarters of the way.
    grid = grid[grid[:, 2] == grid[:, 2].min()]
    axis = int(np.argmax(hi[:2] - lo[:2]))
    across = np.abs(grid[:, 1 - axis] - (lo[1 - axis] + hi[1 - axis]) / 2)
    row = grid[across == across.min()]
    row = row if len(row) > 1 else grid
    row = row[np.argsort(row[:, axis], kind="stable")]
    source, listener = row[len(row) // 4], row[(3 * len(row)) // 4]
    data["sources"][0]["position"] = [float(v) for v in source]
    data["listeners"][0]["position"] = [float(v) for v in listener]
    return Scene.model_validate(data)


def plan_corners(scene: Scene) -> np.ndarray | None:
    """The corners (m) of a floor plan or box room seen from above, in the order of its
    walls (``wall_1`` runs from the first corner to the second); None for a model file."""
    geometry = scene.geometry
    scale = UNIT_SCALE[geometry.units or "m"]
    if geometry.floor_plan is not None:
        return scale * np.array(geometry.floor_plan, dtype=float)
    if geometry.box is not None:
        length, width = scale * geometry.box[0], scale * geometry.box[1]
        return np.array([[0.0, 0.0], [length, 0.0], [length, width], [0.0, width]])
    return None


def room_height(scene: Scene) -> float | None:
    """The height (m) of a floor plan or box room; None for a model file."""
    geometry = scene.geometry
    scale = UNIT_SCALE[geometry.units or "m"]
    if geometry.floor_plan is not None:
        return scale * geometry.height
    if geometry.box is not None:
        return scale * geometry.box[2]
    return None


def surface_heights(mesh: RoomMesh, x: float, y: float, *, up: bool = False) -> np.ndarray:
    """Heights of the floors, ceilings and other surfaces facing up or down (only up,
    with ``up``) that lie over or under the point (x, y)."""
    triangles = mesh.triangles
    normals = mesh.face_normals()
    flat = normals[:, 2] > 0.5 if up else np.abs(normals[:, 2]) > 0.5
    a, b, c = (triangles[flat, k] for k in range(3))
    # Barycentric coordinates of (x, y) in each triangle seen from above.
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    v2 = np.array([x, y]) - a[:, :2]
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    ok = np.abs(det) > 1e-12
    det = np.where(ok, det, 1.0)
    u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / det
    v = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / det
    tolerance = 1e-9
    hit = ok & (u >= -tolerance) & (v >= -tolerance) & (u + v <= 1 + tolerance)
    z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
    return np.unique(z[hit])


def floor_height(mesh: RoomMesh, x: float, y: float, below: float = np.inf) -> float | None:
    """The highest surface under (x, y) below the height ``below``, or None."""
    heights = surface_heights(mesh, x, y)
    heights = heights[heights <= below + 1e-9]
    return float(heights.max()) if len(heights) else None


def follow_floor(mesh: RoomMesh, position, xy) -> tuple[float, float, float]:
    """``position`` moved to the plan position ``xy``, at the same height above the floor
    when the floor changes by at most :data:`FOLLOW_STEP`, else at the same height."""
    x, y, z = (float(v) for v in position)
    new_x, new_y = (float(v) for v in xy)
    old = floor_height(mesh, x, y, z)
    if old is not None:
        heights = surface_heights(mesh, new_x, new_y)
        if len(heights):
            nearest = float(heights[np.argmin(np.abs(heights - old))])
            if abs(nearest - old) <= FOLLOW_STEP:
                z += nearest - old
    return (new_x, new_y, z)


def _rounded(values) -> list[float]:
    return [round(float(v), DECIMALS) + 0.0 for v in values]


def _free_spots(plan: np.ndarray, mesh: RoomMesh, height: float, count: int) -> np.ndarray:
    """Plan positions inside the floor plan ``plan`` (its room is ``mesh``) on a grid,
    half a metre or more from the walls when at least ``count`` of them are."""
    lo, hi = plan.min(axis=0), plan.max(axis=0)
    step = max(float(np.max(hi - lo)) / 40.0, 0.05)
    xs = np.arange(lo[0] + step / 2, hi[0], step)
    ys = np.arange(lo[1] + step / 2, hi[1], step)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    grid = grid[points_inside(mesh, np.column_stack([grid, np.full(len(grid), height / 2)]))]
    if len(grid) == 0:
        return grid
    a, d = plan, np.roll(plan, -1, axis=0) - plan  # the walls: start and direction
    t = np.einsum("pnk,nk->pn", grid[:, None] - a[None], d) / np.sum(d * d, axis=1)
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * d[None]
    distance = np.linalg.norm(grid[:, None] - closest, axis=2).min(axis=1)
    clear = grid[distance >= min(0.5, 0.5 * distance.max())]
    return clear if len(clear) >= count else grid


def referenced_files(scene: Scene) -> list[Path]:
    """The files ``scene`` uses (:meth:`~acoustic_spaces.scene.Scene.files`): its model,
    terrain and map of buildings, material tables, source directivities (with the files
    an array names for its element) and HRTFs."""
    return scene.files()


def _in_units(item: dict, scale: float) -> dict:
    """``item``, a building's, barrier's, vegetation's or zone's scene entry with its
    corners, points and sizes in metres, with them in the geometry's units (``scale``
    metres each)."""
    out = dict(item)
    for key in ("footprint", "path", "polygon"):
        if key in out:
            out[key] = [_rounded(np.asarray(p, dtype=float) / scale) for p in out[key]]
    if "tree" in out:
        out["tree"] = _rounded(np.asarray(out["tree"], dtype=float) / scale)
    for key in ("height", "base", "width", "crown_diameter"):
        if key in out:
            out[key] = round(float(out[key]) / scale, DECIMALS)
    return out


def unique_name(stem: str, taken: set[str]) -> str:
    """``stem``, or ``stem-2``, ``stem-3``... whichever is free."""
    if stem not in taken:
        return stem
    k = 2
    while f"{stem}-{k}" in taken:
        k += 1
    return f"{stem}-{k}"


class Document(QObject):
    """A scene, the file it lives in, and its edit history."""

    #: Emitted after every change of the scene: an edit, undo, redo or load.
    changed = Signal()
    #: Emitted when the file, the unsaved state or the undo history changes.
    state_changed = Signal()

    def __init__(self, scene: Scene, path: str | Path | None = None) -> None:
        super().__init__()
        self.path = None if path is None else Path(path)
        self.scene = scene
        self.room = Room.from_scene(scene)
        self._undo: list[tuple[str, Scene]] = []
        self._redo: list[tuple[str, Scene]] = []
        self._saved = scene
        #: The sources and listeners hidden in the 3D view and the plan, as (kind, name):
        #: how the editor shows the scene, not part of it.
        self.hidden: set[tuple[str, str]] = set()

    @classmethod
    def open(cls, path: str | Path) -> Document:
        """Load a scene file; raises :class:`EditError` with the reason if it is invalid."""
        try:
            scene = Scene.from_yaml(path)
            return cls(scene, path)
        except (ValidationError, ValueError, OSError) as exc:
            raise EditError(_message(exc)) from exc

    # -- state ----------------------------------------------------------------------

    @property
    def base_dir(self) -> Path:
        """The folder relative paths in edits are resolved against."""
        return self.path.parent if self.path is not None else Path.cwd()

    @property
    def modified(self) -> bool:
        return self.scene is not self._saved

    def undo_text(self) -> str | None:
        return self._undo[-1][0] if self._undo else None

    def redo_text(self) -> str | None:
        return self._redo[-1][0] if self._redo else None

    def names(self) -> set[str]:
        """Names of every source and listener; they share one namespace."""
        return {s.name for s in self.scene.sources} | {x.name for x in self.scene.listeners}

    # -- changes --------------------------------------------------------------------

    def edit(self, description: str, change: Callable[[dict], None]) -> None:
        """Apply ``change`` to a plain copy of the scene (the YAML structure as dicts and
        lists) and make the result current. Raises :class:`EditError` if the scene
        schema or the geometry rejects it; the document is then unchanged."""
        data = self.scene.to_data()
        change(data)
        try:
            scene = Scene.model_validate(data, context={"base_dir": self.base_dir})
            room = Room.from_scene(scene, reuse=self.room)
        except (ValidationError, ValueError, OSError) as exc:
            raise EditError(_message(exc)) from exc
        self._undo.append((description, self.scene))
        del self._undo[:-HISTORY]
        self._redo.clear()
        self._set(scene, room)

    def undo(self) -> None:
        if not self._undo:
            return
        description, scene = self._undo.pop()
        self._redo.append((description, self.scene))
        self._set(scene, Room.from_scene(scene, reuse=self.room))

    def redo(self) -> None:
        if not self._redo:
            return
        description, scene = self._redo.pop()
        self._undo.append((description, self.scene))
        self._set(scene, Room.from_scene(scene, reuse=self.room))

    def _set(self, scene: Scene, room: Room) -> None:
        self.scene, self.room = scene, room
        self.changed.emit()
        self.state_changed.emit()

    def save(self, path: str | Path | None = None, *, copy_files: bool = False) -> None:
        """Write the scene file (to ``path``, which becomes the document's file).

        With ``copy_files``, the files the scene uses from its current folder (a model,
        material tables, directivities, HRTFs) are copied along to the new folder,
        keeping their relative places, and the scene refers to the copies. Files already
        there with other contents are left alone; the scene then keeps using the
        originals.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError(_("no file to save to; give a path"))
        if copy_files and self.path is not None:
            self._copy_files(target.resolve().parent)
        self.scene.to_yaml(target)
        self.path = target
        self._saved = self.scene
        self.state_changed.emit()

    def _copy_files(self, folder: Path) -> None:
        old = self.base_dir.resolve()
        if folder == old:
            return
        copies: dict[Path, Path] = {}
        for file in referenced_files(self.scene):
            try:
                relative = file.resolve().relative_to(old)
            except ValueError:
                continue
            copy = folder / relative
            if copy.exists() and not filecmp.cmp(file, copy, shallow=False):
                continue
            copies[file] = copy
        # An array's YAML file names its element's file relative to itself: it moves only
        # along with every file it names, or its copy would look for them in vain.
        while True:
            moving = {file.resolve() for file in copies}
            stays = [f for f in copies if any(n.resolve() not in moving for n in array_files(f))]
            if not stays:
                break
            for file in stays:
                del copies[file]
        if not copies:
            return
        made: list[Path] = []
        folders: list[Path] = []
        try:
            for file, copy in copies.items():
                if copy.exists():
                    continue
                folders += [p for p in (copy.parent, *copy.parent.parents) if not p.exists()]
                copy.parent.mkdir(parents=True, exist_ok=True)
                made.append(copy)
                shutil.copy2(file, copy)
            self._use_copies({str(file): str(copy) for file, copy in copies.items()}, folder)
        except Exception:
            # A save that fails leaves nothing behind in the new folder.
            for copy in made:
                copy.unlink(missing_ok=True)
            for made_folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
                with contextlib.suppress(OSError):
                    made_folder.rmdir()
            raise

    def _use_copies(self, moved: dict[str, str], folder: Path) -> None:
        """Point the scene at the copies of its files (``moved``, original to copy) in
        ``folder``."""
        data = self.scene.to_data()
        if data["geometry"].get("file") in moved:
            data["geometry"]["file"] = moved[data["geometry"]["file"]]
        terrain = data["geometry"].get("ground", {}).get("terrain", {})
        if terrain.get("file") in moved:
            terrain["file"] = moved[terrain["file"]]
        buildings = data["geometry"].get("buildings_file", {})
        if buildings.get("file") in moved:
            buildings["file"] = moved[buildings["file"]]
        for material in data["materials"].values():
            if material.get("file") in moved:
                material["file"] = moved[material["file"]]
        for source in data["sources"]:
            for spec in directivity_entries(source.get("directivity")):
                if spec.get("file") in moved:
                    spec["file"] = moved[spec["file"]]
        for output in data["outputs"]:
            if output.get("hrtf") in moved:
                output["hrtf"] = moved[output["hrtf"]]
        scene = Scene.model_validate(data, context={"base_dir": folder})
        # The same scene with other copies of its files: no undo step. The copies hold
        # the same bytes, so the loaded geometry and directivities stay (an array's take
        # seconds to add up again).
        self.scene, self.room = scene, dataclasses.replace(self.room, scene=scene)
        self.changed.emit()

    # -- common edits -----------------------------------------------------------------

    def add_point(self, kind: str, position: tuple[float, float, float]) -> int:
        """Add a source or listener (``kind`` is "sources" or "listeners") at
        ``position`` and return its index."""
        stem = "source" if kind == "sources" else "listener"
        entry = {"name": unique_name(stem, self.names()), "position": list(position)}
        description = _("add source") if kind == "sources" else _("add listener")
        self.edit(description, lambda data: data[kind].append(entry))
        return len(getattr(self.scene, kind)) - 1

    def move_point(self, kind: str, index: int, position) -> None:
        """Move a source or listener to ``position`` (m)."""
        where = _rounded(position)
        description = _("move source") if kind == "sources" else _("move listener")
        self.edit(description, lambda data: data[kind][index].update(position=where))

    def update_point(self, kind: str, index: int, **fields) -> None:
        """Change fields of a source or listener (name, position, orientation...)."""
        old = getattr(self.scene, kind)[index].name

        def change(data: dict) -> None:
            data[kind][index].update(fields)

        self.edit(_("change source") if kind == "sources" else _("change listener"), change)
        if (kind, old) in self.hidden and fields.get("name", old) != old:
            self.hidden.discard((kind, old))
            self.hidden.add((kind, fields["name"]))

    def set_enabled(self, kind: str, indices, enabled: bool) -> None:
        """Use the sources or listeners (``kind`` is "sources" or "listeners") at
        ``indices`` in simulations and maps, or leave them out (``enabled: false``), in
        one step of the history."""
        items = getattr(self.scene, kind)
        changing = [k for k in indices if items[k].in_use != enabled]
        if not changing:
            return
        count = len(changing)
        if kind == "sources":
            text = (
                ngettext("use source", "use {count} sources", count)
                if enabled
                else ngettext("leave out source", "leave out {count} sources", count)
            )
        else:
            text = (
                ngettext("use listener", "use {count} listeners", count)
                if enabled
                else ngettext("leave out listener", "leave out {count} listeners", count)
            )

        def change(data: dict) -> None:
            for k in changing:
                if enabled:
                    data[kind][k].pop("enabled", None)
                else:
                    data[kind][k]["enabled"] = False

        self.edit(text.format(count=count), change)

    def set_walk_enabled(self, kind: str, index: int, enabled: bool) -> None:
        """Have the source or listener at ``index`` go along its walk in a movement
        (``walk.enabled``), or keep the walk but leave it standing where it starts."""
        item = getattr(self.scene, kind)[index]
        if item.walk is None:
            raise EditError(
                _("{name} has no walk yet: add a keyframe first").format(name=item.name)
            )
        if item.walk.enabled == enabled:
            return

        def change(data: dict) -> None:
            data[kind][index]["walk"]["enabled"] = enabled

        self.edit(_("turn the walk on") if enabled else _("turn the walk off"), change)

    def set_walk_doppler(self, kind: str, index: int, doppler: bool) -> None:
        """Have the walk of the source or listener at ``index`` take the early sound through
        delays that change as it goes, so its pitch shifts (``walk.doppler``), or crossfade
        from place to place."""
        item = getattr(self.scene, kind)[index]
        if item.walk is None:
            raise EditError(
                _("{name} has no walk yet: add a keyframe first").format(name=item.name)
            )
        if item.walk.doppler == doppler:
            return

        def change(data: dict) -> None:
            data[kind][index]["walk"]["doppler"] = doppler

        self.edit(_("turn Doppler on") if doppler else _("turn Doppler off"), change)

    def set_hidden(self, kind: str, indices, hidden: bool) -> None:
        """Hide the sources or listeners at ``indices`` in the 3D view and the plan, or
        show them again; this changes how the editor shows the scene, not the scene."""
        names = {(kind, getattr(self.scene, kind)[k].name) for k in indices}
        if hidden:
            self.hidden |= names
        else:
            self.hidden -= names

    def remove_point(self, kind: str, index: int) -> None:
        source = kind == "sources"
        if len(getattr(self.scene, kind)) == 1:
            raise EditError(
                _("a scene needs at least one source")
                if source
                else _("a scene needs at least one listener")
            )
        description = _("remove source") if source else _("remove listener")
        self.edit(description, lambda data: data[kind].pop(index))

    def set_floor_plan(self, corners, description: str | None = None) -> None:
        """New corners (m, seen from above) for a floor plan room. A box becomes a floor
        plan with the same walls, so its material assignments still apply."""
        geometry = self.scene.geometry
        if geometry.ground is not None:
            raise EditError(_("the scene is outdoors: draw a floor plan to replace it"))
        if geometry.file is not None:
            raise EditError(_("the room is a model file: draw a floor plan to replace it"))
        scale = UNIT_SCALE[geometry.units or "m"]
        height = geometry.height if geometry.floor_plan is not None else geometry.box[2]
        plan = [_rounded(np.asarray(corner, dtype=float) / scale) for corner in corners]

        def change(data: dict) -> None:
            data["geometry"].pop("box", None)
            data["geometry"].update(floor_plan=plan, height=height)

        self.edit(description or _("change floor plan"), change)

    def draw_floor_plan(self, corners, height: float | None = None) -> list[str]:
        """Make the room the floor plan ``corners`` (m), ``height`` metres high (by default
        as high as the room was, else :data:`DRAWN_HEIGHT`). Sources and listeners that
        would be outside move inside, at their height if it fits; returns their names.
        Surfaces no assignment covers (after a model file) get the material that covered
        most of the old room, as the scene's default."""
        height = float(height or room_height(self.scene) or DRAWN_HEIGHT)
        plan = [_rounded(corner) for corner in corners]
        try:
            mesh = extruded_mesh(np.array(plan, dtype=float), height)
        except ValueError as exc:
            raise EditError(str(exc)) from exc
        items = [
            (kind, k, item)
            for kind in ("sources", "listeners")
            for k, item in enumerate(getattr(self.scene, kind))
        ]
        inside = points_inside(mesh, [item.position for _, _, item in items])
        spots = _free_spots(np.array(plan), mesh, height, int((~inside).sum()))
        # A point outside goes to the free spot nearest its place in the old room's
        # footprint stretched over the new one's, so the layout stays recognisable.
        old_lo = np.array(self.room.report.bounds_min[:2], dtype=float)
        old_size = np.maximum(np.array(self.room.report.bounds_max[:2]) - old_lo, 1e-9)
        new_lo, new_hi = np.min(plan, axis=0), np.max(plan, axis=0)
        moved: dict[tuple[str, int], list[float]] = {}
        taken = np.zeros(len(spots), dtype=bool)
        for (kind, k, item), ok in zip(items, inside, strict=True):
            if ok or not len(spots):
                continue
            relative = np.clip((np.asarray(item.position[:2]) - old_lo) / old_size, 0.0, 1.0)
            target = new_lo + relative * (new_hi - new_lo)
            distance = np.linalg.norm(spots - target, axis=1)
            if not taken.all():
                distance[taken] = np.inf
            best = int(np.argmin(distance))
            taken[best] = True
            usual = SOURCE_HEIGHT if kind == "sources" else LISTENER_HEIGHT
            z = item.position[2] if 0.0 < item.position[2] < height else min(usual, height / 2)
            moved[kind, k] = _rounded([*spots[best], z])

        default = None
        names = set(self.scene.assignments) | set(self.scene.materials)
        if self.scene.default_material is None and any(
            matching_name(label, names) is None for label in mesh.labels
        ):
            areas = self.room.area_by_material()
            if self.room.outdoor is not None:
                areas.pop(SKY, None)  # an opening: walls of it would let everything out
            default = max(areas, key=areas.get) if areas else next(iter(self.scene.materials))

        def change(data: dict) -> None:
            data["geometry"] = {"floor_plan": plan, "height": height}
            # Outdoors the early paths may add in pressure in every band; a room's take
            # the default.
            early = data.get("simulation", {}).get("early", {})
            if (early.get("coherent_max_band_hz") or 0) > INDOOR_COHERENT_MAX_BAND_HZ:
                del early["coherent_max_band_hz"]
            # The wind and temperature, the weather classes, and the insertion loss of
            # buildings, barriers and vegetation, are for outdoor scenes.
            data.pop("atmosphere", None)
            data.pop("long_term", None)
            data.get("simulation", {}).pop("insertion_loss_of", None)
            for (kind, k), position in moved.items():
                data[kind][k]["position"] = position
            if default is not None:
                data["default_material"] = default

        self.edit(_("draw floor plan"), change)
        return [getattr(self.scene, kind)[k].name for kind, k in moved]

    # -- outdoors -------------------------------------------------------------------

    def add_part(self, kind: str, item: dict) -> int:
        """Add a building, a barrier or vegetation (``kind`` "buildings", "barriers" or
        "vegetation") to an outdoor scene: ``item`` is its scene entry with its corners,
        points and sizes in metres, which go into the geometry's units. Returns its
        index."""
        what = {
            "buildings": _("add building"),
            "barriers": _("add barrier"),
            "vegetation": _("add vegetation"),
        }
        entry = _in_units(item, UNIT_SCALE[self.scene.geometry.units or "m"])
        self.edit(what[kind], lambda data: data["geometry"].setdefault(kind, []).append(entry))
        return len(getattr(self.scene.geometry, kind)) - 1

    def add_zone(self, polygon) -> int:
        """Add a ground zone with these corners (m) to an outdoor scene; it takes the
        ground's material until one is assigned to it. Returns its index."""
        entry = _in_units({"polygon": polygon}, UNIT_SCALE[self.scene.geometry.units or "m"])
        self.edit(
            _("add zone"),
            lambda data: data["geometry"]["ground"].setdefault("zones", []).append(entry),
        )
        return len(self.scene.geometry.ground.zones) - 1

    def add_weather_zone(self, polygon) -> int:
        """Add a weather zone with these corners (m) to an outdoor scene; it keeps the
        scene's weather until given wind or temperature of its own. Returns its index."""
        entry = _in_units({"polygon": polygon}, UNIT_SCALE[self.scene.geometry.units or "m"])

        def change(data: dict) -> None:
            atmosphere = data.get("atmosphere") or {}
            atmosphere.setdefault("zones", []).append(entry)
            data["atmosphere"] = atmosphere

        self.edit(_("add weather zone"), change)
        return len(self.scene.atmosphere.zones) - 1

    def set_extent(self, extent) -> None:
        """Set where the ground of an outdoor scene ends: ``[[x_min, y_min], [x_max,
        y_max]]`` in the geometry's units, or None for the margin round what stands on
        it (or the whole terrain)."""

        def change(data: dict) -> None:
            ground = data["geometry"]["ground"]
            if extent is None:
                ground.pop("extent", None)
            else:
                ground["extent"] = [_rounded(corner) for corner in extent]

        self.edit(_("change ground"), change)

    def set_terrain(self, path: str | Path | None) -> None:
        """Make the ground of an outdoor scene follow the heights in ``path`` (an Esri
        ASCII grid, or x y z points on a grid), or flat again with None. A terrain sets
        the ground's size, so a set extent goes with it."""

        def change(data: dict) -> None:
            ground = data["geometry"]["ground"]
            if path is None:
                ground.pop("terrain", None)
            else:
                ground["terrain"] = {"file": str(Path(path).resolve())}
                ground.pop("extent", None)

        self.edit(_("set terrain") if path is not None else _("remove terrain"), change)

    def set_buildings_file(
        self,
        path: str | Path | None,
        height: str | float = "height",
        name: str = "name",
        default_height: float | None = None,
    ) -> None:
        """Add the buildings of the GeoJSON file ``path`` to an outdoor scene, each as
        high as its property ``height`` says (or that height for all of them, and
        ``default_height`` where the property is missing) and named by its property
        ``name``; None drops the file and its buildings."""
        geometry = self.scene.geometry
        gone = set(geometry.building_names()[len(geometry.buildings or []) :])

        def change(data: dict) -> None:
            if path is None:
                data["geometry"].pop("buildings_file", None)
                settings = data["simulation"]
                kept = [n for n in settings.get("insertion_loss_of") or [] if n not in gone]
                if kept:
                    settings["insertion_loss_of"] = kept
                else:
                    settings.pop("insertion_loss_of", None)
                return
            spec = {"file": str(Path(path).resolve()), "height": height, "name": name}
            if default_height is not None:
                spec["default_height"] = default_height
            data["geometry"]["buildings_file"] = spec

        self.edit(
            _("add buildings from a map") if path is not None else _("remove the map's buildings"),
            change,
        )

    def assign(self, surface: str, material: str | None) -> None:
        """Give the surfaces named ``surface`` a material (None removes the assignment)."""

        def change(data: dict) -> None:
            if material is None:
                data["assignments"].pop(surface, None)
            else:
                data["assignments"][surface] = material

        self.edit(_("assign {surface}").format(surface=surface), change)

    def set_material(self, name: str, fields: dict) -> None:
        """Replace a material's settings (add it when new)."""
        self.edit(
            _("change material {name}").format(name=name),
            lambda data: data["materials"].update({name: fields}),
        )

    def add_material(self, preset: str | None = None) -> str:
        """Add a material from a database entry (or a neutral one) and return its name."""
        name = unique_name(preset or "material", set(self.scene.materials))
        fields = {"preset": preset} if preset else {"absorption": [0.1] * 8}
        fields["scattering"] = 0.1
        self.edit(
            _("add material {name}").format(name=name),
            lambda data: data["materials"].update({name: fields}),
        )
        return name

    def add_library_material(self, path: str | Path, entry: str) -> str:
        """Add a material from an entry of a CATT-Acoustic or ODEON library file and
        return its name: the entry's, without its number."""
        from acoustic_spaces.materials import find_material

        try:
            found = find_material(path, entry)
        except (OSError, ValueError) as exc:
            raise EditError(str(exc)) from None
        base = re.sub(r"\W+", "_", found.name.lower()).strip("_") or "material"
        name = unique_name(base, set(self.scene.materials))
        entry_key = found.name if found.number is None else found.number
        fields: dict = {"file": str(path), "entry": entry_key}
        if found.scattering is None:
            fields["scattering"] = 0.1
        self.edit(
            _("add material {name}").format(name=name),
            lambda data: data["materials"].update({name: fields}),
        )
        return name

    def rename_material(self, old: str, new: str) -> None:
        if new == old:
            return
        if new in self.scene.materials:
            raise EditError(_("there is already a material named “{name}”").format(name=new))
        if not new:
            raise EditError(_("a material needs a name"))

        def change(data: dict) -> None:
            data["materials"] = {
                (new if key == old else key): value for key, value in data["materials"].items()
            }
            for surface, material in data["assignments"].items():
                if material == old:
                    data["assignments"][surface] = new
            if data.get("default_material") == old:
                data["default_material"] = new

        self.edit(_("rename material {name}").format(name=old), change)

    def remove_material(self, name: str) -> None:
        """Remove a material and the assignments that use it."""
        if len(self.scene.materials) == 1:
            raise EditError(_("a scene needs at least one material"))

        def change(data: dict) -> None:
            del data["materials"][name]
            data["assignments"] = {
                surface: material
                for surface, material in data["assignments"].items()
                if material != name
            }
            if data.get("default_material") == name:
                del data["default_material"]

        self.edit(_("remove material {name}").format(name=name), change)


__all__ = [
    "DRAWN_HEIGHT",
    "FOLLOW_STEP",
    "LISTENER_HEIGHT",
    "NEW_GROUND_SIZE",
    "NEW_PART_SIZES",
    "SOURCE_HEIGHT",
    "Document",
    "EditError",
    "box_scene",
    "floor_height",
    "follow_floor",
    "model_scene",
    "outdoor_scene",
    "plan_corners",
    "referenced_files",
    "room_height",
    "surface_heights",
    "unique_name",
]
