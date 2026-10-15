"""The scene being edited: its file, its room, and the edits made to it.

Every change goes through :meth:`Document.edit`, which changes a plain copy of the
scene, validates it with the scene schema, loads the room (reusing the mesh when the
geometry did not change) and only then makes it current. A change the schema or the
geometry rejects raises and leaves the document as it was, so the editor can show the
reason. Undo and redo step through whole scenes, which are small.
"""

from __future__ import annotations

import filecmp
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from PySide6.QtCore import QObject, Signal

from acoustic_spaces.geometry import UNIT_SCALE, RoomMesh, extruded_mesh, points_inside
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene

#: Heights above a floor for new sources and listeners (m): a talker, a seated ear.
SOURCE_HEIGHT = 1.5
LISTENER_HEIGHT = 1.2
#: Undo steps kept.
HISTORY = 200
#: Height of a drawn room when the scene had none to keep (m).
DRAWN_HEIGHT = 3.0
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
    """The files ``scene`` uses: its model, material tables, source directivities and
    HRTFs."""
    files = [scene.geometry.file] if scene.geometry.file is not None else []
    files += [m.file for m in scene.materials.values() if m.file is not None]
    files += [
        s.directivity.file
        for s in scene.sources
        if not isinstance(s.directivity, str) and s.directivity.file is not None
    ]
    files += [o.hrtf for o in scene.outputs if o.hrtf is not None]
    return [Path(f) for f in files]


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
        data = self.scene.model_dump(mode="json", exclude_none=True)
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
        material tables, HRTFs) are copied along to the new folder, keeping their
        relative places, and the scene refers to the copies. Files already there with
        other contents are left alone; the scene then keeps using the originals.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no file to save to; give a path")
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
        moved: dict[str, str] = {}
        for file in referenced_files(self.scene):
            try:
                relative = file.resolve().relative_to(old)
            except ValueError:
                continue
            copy = folder / relative
            if copy.exists() and not filecmp.cmp(file, copy, shallow=False):
                continue
            copy.parent.mkdir(parents=True, exist_ok=True)
            if not copy.exists():
                shutil.copy2(file, copy)
            moved[str(file)] = str(copy)
        if not moved:
            return
        data = self.scene.model_dump(mode="json", exclude_none=True)
        if data["geometry"].get("file") in moved:
            data["geometry"]["file"] = moved[data["geometry"]["file"]]
        for material in data["materials"].values():
            if material.get("file") in moved:
                material["file"] = moved[material["file"]]
        for source in data["sources"]:
            directivity = source.get("directivity")
            if isinstance(directivity, dict) and directivity.get("file") in moved:
                directivity["file"] = moved[directivity["file"]]
        for output in data["outputs"]:
            if output.get("hrtf") in moved:
                output["hrtf"] = moved[output["hrtf"]]
        scene = Scene.model_validate(data, context={"base_dir": folder})
        # The same scene with other copies of its files: no undo step.
        self.scene, self.room = scene, Room.from_scene(scene, reuse=self.room)
        self.changed.emit()

    # -- common edits -----------------------------------------------------------------

    def add_point(self, kind: str, position: tuple[float, float, float]) -> int:
        """Add a source or listener (``kind`` is "sources" or "listeners") at
        ``position`` and return its index."""
        stem = "source" if kind == "sources" else "listener"
        entry = {"name": unique_name(stem, self.names()), "position": list(position)}
        self.edit(f"add {stem}", lambda data: data[kind].append(entry))
        return len(getattr(self.scene, kind)) - 1

    def move_point(self, kind: str, index: int, position) -> None:
        """Move a source or listener to ``position`` (m)."""
        stem = "source" if kind == "sources" else "listener"
        where = _rounded(position)
        self.edit(f"move {stem}", lambda data: data[kind][index].update(position=where))

    def update_point(self, kind: str, index: int, **fields) -> None:
        """Change fields of a source or listener (name, position, orientation...)."""
        stem = "source" if kind == "sources" else "listener"

        def change(data: dict) -> None:
            data[kind][index].update(fields)

        self.edit(f"change {stem}", change)

    def remove_point(self, kind: str, index: int) -> None:
        stem = "source" if kind == "sources" else "listener"
        if len(getattr(self.scene, kind)) == 1:
            raise EditError(f"a scene needs at least one {stem}")
        self.edit(f"remove {stem}", lambda data: data[kind].pop(index))

    def set_floor_plan(self, corners, description: str = "change floor plan") -> None:
        """New corners (m, seen from above) for a floor plan room. A box becomes a floor
        plan with the same walls, so its material assignments still apply."""
        geometry = self.scene.geometry
        if geometry.file is not None:
            raise EditError("the room is a model file: draw a floor plan to replace it")
        scale = UNIT_SCALE[geometry.units or "m"]
        height = geometry.height if geometry.floor_plan is not None else geometry.box[2]
        plan = [_rounded(np.asarray(corner, dtype=float) / scale) for corner in corners]

        def change(data: dict) -> None:
            data["geometry"].pop("box", None)
            data["geometry"].update(floor_plan=plan, height=height)

        self.edit(description, change)

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
            not set(label) & names for label in mesh.labels
        ):
            areas = self.room.area_by_material()
            default = max(areas, key=areas.get) if areas else next(iter(self.scene.materials))

        def change(data: dict) -> None:
            data["geometry"] = {"floor_plan": plan, "height": height}
            for (kind, k), position in moved.items():
                data[kind][k]["position"] = position
            if default is not None:
                data["default_material"] = default

        self.edit("draw floor plan", change)
        return [getattr(self.scene, kind)[k].name for kind, k in moved]

    def assign(self, surface: str, material: str | None) -> None:
        """Give the surfaces named ``surface`` a material (None removes the assignment)."""

        def change(data: dict) -> None:
            if material is None:
                data["assignments"].pop(surface, None)
            else:
                data["assignments"][surface] = material

        self.edit(f"assign {surface}", change)

    def set_material(self, name: str, fields: dict) -> None:
        """Replace a material's settings (add it when new)."""
        self.edit(f"change material {name}", lambda data: data["materials"].update({name: fields}))

    def add_material(self, preset: str | None = None) -> str:
        """Add a material from a database entry (or a neutral one) and return its name."""
        name = unique_name(preset or "material", set(self.scene.materials))
        fields = {"preset": preset} if preset else {"absorption": [0.1] * 8}
        fields["scattering"] = 0.1
        self.edit(f"add material {name}", lambda data: data["materials"].update({name: fields}))
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
        self.edit(f"add material {name}", lambda data: data["materials"].update({name: fields}))
        return name

    def rename_material(self, old: str, new: str) -> None:
        if new == old:
            return
        if new in self.scene.materials:
            raise EditError(f"there is already a material named {new!r}")
        if not new:
            raise EditError("a material needs a name")

        def change(data: dict) -> None:
            data["materials"] = {
                (new if key == old else key): value for key, value in data["materials"].items()
            }
            for surface, material in data["assignments"].items():
                if material == old:
                    data["assignments"][surface] = new
            if data.get("default_material") == old:
                data["default_material"] = new

        self.edit(f"rename material {old}", change)

    def remove_material(self, name: str) -> None:
        """Remove a material and the assignments that use it."""
        if len(self.scene.materials) == 1:
            raise EditError("a scene needs at least one material")

        def change(data: dict) -> None:
            del data["materials"][name]
            data["assignments"] = {
                surface: material
                for surface, material in data["assignments"].items()
                if material != name
            }
            if data.get("default_material") == name:
                del data["default_material"]

        self.edit(f"remove material {name}", change)


__all__ = [
    "DRAWN_HEIGHT",
    "FOLLOW_STEP",
    "LISTENER_HEIGHT",
    "SOURCE_HEIGHT",
    "Document",
    "EditError",
    "box_scene",
    "floor_height",
    "follow_floor",
    "model_scene",
    "plan_corners",
    "referenced_files",
    "room_height",
    "surface_heights",
    "unique_name",
]
