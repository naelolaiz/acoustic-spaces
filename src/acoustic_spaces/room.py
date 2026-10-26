"""A scene with its geometry loaded: a material on every surface, and checks.

:class:`Room` is what the simulation stages consume. It also gives the classic
statistical estimates (Sabine and Eyring reverberation times) so a model can be
sanity-checked before any ray is traced.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from acoustic_spaces.air import energy_attenuation_per_m, speed_of_sound
from acoustic_spaces.directivity import (
    FAR_FIELD_HZ,
    Directivity,
    array_length,
    far_field_distance,
    load_directivity,
)
from acoustic_spaces.geometry import (
    UNIT_SCALE,
    MeshReport,
    RoomMesh,
    box_mesh,
    check_mesh,
    clean_mesh,
    extruded_mesh,
    load_mesh,
)
from acoustic_spaces.geometry.cells import MeshCells, cell_of
from acoustic_spaces.geometry.outdoor import SKY, Layout, outdoor_mesh
from acoustic_spaces.materials.bands import Resolution, band_centres
from acoustic_spaces.scene import Scene

UNASSIGNED = -1
#: A listener closer than this (metres) to a source has no response: the direct
#: sound's level grows without bound as the distance goes to zero.
MIN_SOURCE_DISTANCE_M = 0.01


def _label_text(label: tuple[str, ...]) -> str:
    return " / ".join(label) if label else "(unnamed)"


def matching_name(label: tuple[str, ...], names: Iterable[str]) -> str | None:
    """The first of ``names`` that a surface's label holds: as it is written, or else in
    any case (``brick_rough`` names SketchUp's ``Brick_Rough``); None for none."""
    names = list(names)
    for name in label:
        if name in names:
            return name
    folded = {}
    for name in names:
        folded.setdefault(name.casefold(), name)
    for name in label:
        if name.casefold() in folded:
            return folded[name.casefold()]
    return None


def _material_for(label: tuple[str, ...], scene: Scene) -> str | None:
    """Assignments win over names that match a material directly; then the default."""
    key = matching_name(label, scene.assignments)
    if key is not None:
        return scene.assignments[key]
    return matching_name(label, scene.materials) or scene.default_material


def _lets_through(material) -> bool:
    """Whether a material lets some sound through in some band (``transmission: 0``
    lets none)."""
    through = material.transmission_coefficients()
    return through is not None and any(t > 0.0 for t in through)


def _air_cells(
    cells: MeshCells, where: np.ndarray, face_materials, scene: Scene, names: list[str]
) -> np.ndarray:
    """The cells that hold air (see :attr:`Room.air_cells`), given the enclosed cell
    each source element and listener lies in (``where``, -1 for none)."""
    occupied = np.unique(where[where >= 0])
    in_rooms = np.isin(occupied, cells.rooms)
    if in_rooms.all():
        return cells.rooms
    # Cells joined by a surface that lets some sound through.
    passing = np.array([_lets_through(scene.materials[n]) for n in names] + [False])[face_materials]
    links = cells.side[passing]

    def joined(start) -> set[int]:
        air = set(np.asarray(start).tolist())
        grown = True
        while grown:
            grown = False
            for a, b in links.tolist():
                if (a in air) != (b in air) and cells.enclosed[a] and cells.enclosed[b]:
                    air |= {a, b}
                    grown = True
        return air

    if not in_rooms.any():
        # Walls drawn as a shell: the rooms are the cavities they stand in.
        return np.array(sorted(joined(occupied)), dtype=np.int64)
    # Some stand in the model's rooms. Another cell is air only where surfaces that let
    # sound through join it to a room (a booth with glass walls); otherwise it is the
    # inside of a solid (a column, a closed box), and what stands in it is in no room.
    air = set(cells.rooms.tolist())
    for cell in occupied[~in_rooms]:
        reached = joined([cell])
        if reached & air:
            air |= reached
    return np.array(sorted(air), dtype=np.int64)


def _directivity_key(spec, scene: Scene) -> str:
    if isinstance(spec, str):
        return spec
    # An array's phases depend on the speed of sound.
    return f"{spec.model_dump_json()} at {scene.air.temperature_c} C"


@dataclass
class Room:
    scene: Scene
    mesh: RoomMesh
    report: MeshReport
    #: Material names in ``scene.materials`` order; ``face_materials`` indexes this.
    material_names: list[str]
    #: (F,) index into ``material_names`` per face, ``UNASSIGNED`` (-1) when none matched.
    face_materials: np.ndarray
    #: Whether each source / listener is inside; None when the mesh is not closed.
    sources_inside: np.ndarray | None
    listeners_inside: np.ndarray | None
    #: Assignment keys that match no name in the geometry.
    unused_assignments: list[str]
    #: Each source's directivity, in ``scene.sources`` order.
    directivities: list[Directivity]
    #: (R,) the cells of ``mesh.cells()`` that hold air: the rooms. Usually the
    #: model's rooms (see :mod:`acoustic_spaces.geometry.cells`); when a source or a
    #: listener lies in another enclosed cell (a model with thick walls, whose rooms are
    #: cells inside the walls' shell), the cells they lie in, with those that surfaces
    #: letting sound through join to them. Empty when the mesh is not closed.
    air_cells: np.ndarray
    #: Where the ground and the sky lie in an outdoor scene (``geometry.ground``); None
    #: for a room.
    outdoor: Layout | None = None
    #: Whether each source (any of its point sources) / listener stands inside a closed
    #: solid: in a cell the faces enclose that holds no air (see ``air_cells``); None
    #: when the mesh is not closed.
    sources_in_solid: np.ndarray | None = None
    listeners_in_solid: np.ndarray | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> Room:
        return cls.from_scene(Scene.from_yaml(path))

    @classmethod
    def from_scene(cls, scene: Scene, *, reuse: Room | None = None) -> Room:
        """Load the scene's geometry and give every surface its material.

        ``reuse`` is an earlier room: when its geometry settings equal the scene's, its
        mesh is kept instead of being loaded and checked again (an editor changing
        materials or positions), and so are directivities read from the same settings.
        """
        geometry = scene.geometry
        outdoor = None
        if reuse is not None and reuse.scene.geometry == geometry:
            mesh, report, outdoor = reuse.mesh, reuse.report, reuse.outdoor
        else:
            scale = UNIT_SCALE[geometry.units or "m"]
            if geometry.ground is not None:
                objects = None
                if geometry.file is not None:
                    objects = load_mesh(
                        geometry.file, units=geometry.units, up_axis=geometry.up_axis
                    )
                raw, outdoor = outdoor_mesh(geometry, objects, scale)
            elif geometry.file is not None:
                raw = load_mesh(
                    geometry.file,
                    units=geometry.units,
                    up_axis=geometry.up_axis,
                    space=geometry.space,
                )
            elif geometry.box is not None:
                raw = box_mesh(tuple(scale * v for v in geometry.box))
            else:
                raw = extruded_mesh(scale * np.array(geometry.floor_plan), scale * geometry.height)
            mesh, merged, removed = clean_mesh(raw)
            if len(mesh.faces) == 0:
                raise ValueError(
                    f"{geometry.describe()} has no faces left: all {removed} of them have no "
                    "area (their corners lie in one place or on one line)"
                )
            report = check_mesh(mesh, merged, removed)

        material_names = list(scene.materials)
        index = {name: k for k, name in enumerate(material_names)}
        per_label = np.array(
            [index.get(_material_for(label, scene), UNASSIGNED) for label in mesh.labels]
            or [UNASSIGNED],
            dtype=np.int64,
        )
        face_materials = per_label[mesh.face_labels]

        sources_inside = listeners_inside = sources_in_solid = listeners_in_solid = None
        air = np.zeros(0, dtype=np.int64)
        if report.closed:
            cells = mesh.cells()
            elements = [s.elements() for s in scene.sources]
            points = np.vstack(
                [*elements, *[np.reshape(lis.position, (1, 3)) for lis in scene.listeners]]
                or [np.zeros((0, 3))]
            )
            enclosed = np.flatnonzero(cells.enclosed)
            found = cell_of(mesh.vertices, mesh.faces, cells, points, enclosed)
            where = np.where(found >= 0, enclosed[np.maximum(found, 0)], -1)
            # Sources and listeners left out (enabled: false) make no cell a room.
            used = np.repeat(
                [s.in_use for s in scene.sources] + [lis.in_use for lis in scene.listeners],
                [len(e) for e in elements] + [1] * len(scene.listeners),
            ).astype(bool)
            air = _air_cells(cells, where[used], face_materials, scene, material_names)
            inside = np.isin(where, air) & (where >= 0)
            solid = (where >= 0) & ~inside
            count = sum(len(e) for e in elements)
            # A line or surface source is inside when all its point sources are, and in a
            # solid when any of them is.
            ends = np.cumsum([len(e) for e in elements])[:-1]
            sources_inside = np.array(
                [part.all() for part in np.split(inside[:count], ends)] if elements else [], bool
            )
            sources_in_solid = np.array(
                [part.any() for part in np.split(solid[:count], ends)] if elements else [], bool
            )
            listeners_inside = inside[count:]
            listeners_in_solid = solid[count:]

        names = {name.casefold() for name in mesh.label_names()}
        unused = [key for key in scene.assignments if key.casefold() not in names]
        loaded = {}
        if reuse is not None:
            for source, directivity in zip(reuse.scene.sources, reuse.directivities, strict=True):
                loaded[_directivity_key(source.directivity, reuse.scene)] = directivity
        directivities = []
        c = speed_of_sound(scene.air.temperature_c)
        for source in scene.sources:
            key = _directivity_key(source.directivity, scene)
            if key not in loaded:
                try:
                    loaded[key] = load_directivity(source.directivity, c)
                except ImportError as exc:
                    raise ValueError(f"source {source.name!r}: {exc}") from exc
                except ValueError as exc:
                    raise ValueError(f"source {source.name!r}: {exc}") from None
            directivities.append(loaded[key])
        return cls(
            scene=scene,
            mesh=mesh,
            report=report,
            material_names=material_names,
            face_materials=face_materials,
            sources_inside=sources_inside,
            listeners_inside=listeners_inside,
            unused_assignments=unused,
            directivities=directivities,
            air_cells=air,
            outdoor=outdoor,
            sources_in_solid=sources_in_solid,
            listeners_in_solid=listeners_in_solid,
        )

    def directivity(self, source: str, bands: Resolution = "octave") -> Directivity:
        """The directivity of the source with this name, in octave or third-octave
        ``bands``."""
        for item, directivity in zip(self.scene.sources, self.directivities, strict=True):
            if item.name == source:
                return directivity.in_bands(bands)
        raise KeyError(f"no source named {source!r}")

    @property
    def bands(self) -> Resolution:
        """The bands the scene's simulation runs in: ``octave`` or ``third_octave``."""
        return self.scene.simulation.bands

    @property
    def band_centres(self) -> tuple[int, ...]:
        """Centre frequencies (Hz) of the bands the scene's simulation runs in."""
        return band_centres(self.bands)

    def sky_faces(self) -> np.ndarray:
        """(F,) the faces of an outdoor scene's open sky (all False for a room)."""
        if self.outdoor is None:
            return np.zeros(len(self.mesh.faces), dtype=bool)
        sky = np.array([label == (SKY,) for label in self.mesh.labels] or [False], dtype=bool)
        return sky[self.mesh.face_labels]

    def building_faces(self) -> np.ndarray:
        """(F,) the faces of an outdoor scene's buildings (``geometry.buildings``)."""
        if self.outdoor is None:
            return np.zeros(len(self.mesh.faces), dtype=bool)
        built = np.array(
            [bool(label) and label[-1] == "buildings" for label in self.mesh.labels] or [False],
            dtype=bool,
        )
        return built[self.mesh.face_labels]

    def shown_faces(self) -> np.ndarray:
        """(F,) the faces worth drawing: all but an outdoor scene's sky, which would hide
        the rest."""
        return ~self.sky_faces()

    # -- checks -------------------------------------------------------------------

    def unassigned_labels(self) -> dict[str, float]:
        """Surface area (m2) without a material, by the names the faces carry."""
        areas = self.mesh.face_areas()
        out: dict[str, float] = {}
        for label_id in np.unique(self.mesh.face_labels[self.face_materials == UNASSIGNED]):
            mask = (self.mesh.face_labels == label_id) & (self.face_materials == UNASSIGNED)
            out[_label_text(self.mesh.labels[label_id])] = float(areas[mask].sum())
        return out

    def problems(self) -> list[str]:
        """Issues that would make a simulation wrong; fix these first. Sources and
        listeners left out (``enabled: false``) are not checked."""
        out = []
        unassigned = self.unassigned_labels()
        if unassigned:
            listed = ", ".join(f"{name} ({area:.1f} m2)" for name, area in unassigned.items())
            out.append(
                f"surfaces without a material: {listed}; "
                "add them to assignments or set default_material"
            )
        for kind, every, every_inside, every_solid in (
            ("source", self.scene.sources, self.sources_inside, self.sources_in_solid),
            ("listener", self.scene.listeners, self.listeners_inside, self.listeners_in_solid),
        ):
            items = [item for item in every if item.in_use]
            inside = None
            if every_inside is not None:
                inside = [ok for item, ok in zip(every, every_inside, strict=True) if item.in_use]
            if every_solid is None:
                every_solid = np.zeros(len(every), dtype=bool)
            solid = [bad for item, bad in zip(every, every_solid, strict=True) if item.in_use]
            if self.outdoor is not None:
                for item in items:
                    points = item.elements() if kind == "source" else [item.position]
                    building = next((b for b in self.outdoor.building_at(points) if b), None)
                    if building is not None:
                        out.append(
                            f"{kind} {item.name!r} at {list(item.position)} is inside building "
                            f"{building!r}"
                        )
            if inside is None:
                continue
            for item, ok, in_solid in zip(items, inside, solid, strict=True):
                where = "reaches" if getattr(item, "extended", False) else "is"
                if not ok and self.outdoor is not None:
                    ground = "the ground" if self.outdoor.terrain is None else "the terrain"
                    out.append(
                        f"{kind} {item.name!r} at {list(item.position)} {where} outside the "
                        f"open air: below {ground}, above the sky or past the edge of the "
                        "ground (geometry.ground's extent, margin and sky)"
                    )
                elif not ok and in_solid:
                    into = "reaches into" if where == "reaches" else "is inside"
                    out.append(
                        f"{kind} {item.name!r} at {list(item.position)} {into} a closed "
                        "solid (the inside of a column, a thick wall or a closed box), not in "
                        "a room"
                    )
                elif not ok:
                    out.append(
                        f"{kind} {item.name!r} at {list(item.position)} {where} outside the room"
                    )
        for listener in self._used("listeners"):
            for source in self._used("sources"):
                gap = np.linalg.norm(source.elements() - np.asarray(listener.position), axis=1)
                if gap.min() < MIN_SOURCE_DISTANCE_M:
                    out.append(
                        f"listener {listener.name!r} at {list(listener.position)} is within "
                        f"{MIN_SOURCE_DISTANCE_M * 100:g} cm of source {source.name!r}, too close "
                        "for a response (the direct sound has no finite level there); move "
                        "them apart"
                    )
        return out

    def _used(self, kind: str) -> list:
        """The scene's sources or listeners (``kind``) in use: not left out."""
        return [item for item in getattr(self.scene, kind) if item.in_use]

    def warnings(self) -> list[str]:
        """Things worth a look that do not stop a simulation; about the sources and
        listeners in use."""
        r = self.report
        out = []
        if not r.closed:
            out.append(
                f"the model is not closed ({r.open_faces} faces have no room on either "
                f"side; {r.boundary_edges} boundary edges): rays can escape (the leak "
                "finder shows where: validate --leaks, or Show leaks in the editor's View "
                "menu), and the volume and inside checks are skipped"
            )
        out.extend(self._unreachable())
        if r.inconsistent_edges:
            out.append(
                f"{r.inconsistent_edges} edges join faces with opposite orientation; the "
                "rooms, the volume and the inside checks come from the sides of the faces, "
                "so only the direction of the normals is unknown"
            )
        for key in self.unused_assignments:
            out.append(f"assignment {key!r} matches no name in the geometry")
        for name, material in self.scene.materials.items():
            out.extend(f"material {name!r}: {note}" for note in material.clipped)
        out.extend(self._near_arrays())
        if self.outdoor is not None:
            out.extend(self.scene.geometry.footprint_notes())
        return out

    def _near_arrays(self) -> list[str]:
        """Listeners too close to a coherent array for its far-field pattern."""
        out = []
        c = self.speed_of_sound()
        for source in self._used("sources"):
            if getattr(source.directivity, "near_field", None):
                continue  # its elements add up one by one
            try:
                length = array_length(source.directivity)
            except ValueError:
                continue  # an unreadable file: loading the directivity said so already
            if not length:
                continue
            reach = far_field_distance(length, FAR_FIELD_HZ, c)
            for listener in self._used("listeners"):
                distance = float(np.linalg.norm(np.subtract(listener.position, source.position)))
                if distance < reach:
                    out.append(
                        f"listener {listener.name!r} is {distance:.1f} m from array "
                        f"{source.name!r} ({length:.2f} m long), whose far-field pattern "
                        f"holds from about {reach:.1f} m at {FAR_FIELD_HZ:g} Hz and farther "
                        "at higher frequencies; near_field: true adds up its elements one by "
                        "one"
                    )
        return out

    def _unreachable(self) -> list[str]:
        """Listeners in a room that no sound of a source can reach: no surface between
        their rooms lets sound through."""
        sources, listeners = self._used("sources"), self._used("listeners")
        if len(self.air_cells) < 2 or not listeners or not sources:
            return []
        through = self.face_transmission()
        links = np.zeros((0, 2), dtype=np.int64)
        if through is not None:
            pair = self.face_room[np.any(through > 0.0, axis=1)]
            links = pair[(pair >= 0).all(axis=1)]
        # Rooms joined to each room, step by step.
        joined = np.eye(len(self.air_cells), dtype=bool)
        joined[links[:, 0], links[:, 1]] = joined[links[:, 1], links[:, 0]] = True
        for _ in range(len(self.air_cells)):
            joined = (joined.astype(int) @ joined.astype(int)) > 0
        listener_rooms = self.room_of([lis.position for lis in listeners])
        out = []
        for source in sources:
            rooms = np.unique(self.room_of(source.elements()))
            rooms = rooms[rooms >= 0]
            for listener, room in zip(listeners, listener_rooms, strict=True):
                if room >= 0 and len(rooms) and not joined[rooms, room].any():
                    out.append(
                        f"listener {listener.name!r} is in another room than source "
                        f"{source.name!r}, and no surface between them lets sound through"
                    )
        return out

    # -- statistics ---------------------------------------------------------------

    @property
    def volume_m3(self) -> float | None:
        """The volume of air (m3): of all the rooms together; None when the model is not
        closed."""
        return float(self.room_volumes.sum()) if len(self.air_cells) else None

    @property
    def room_volumes(self) -> np.ndarray:
        """(R,) volume of each room (m3)."""
        return self.mesh.cells().volume[self.air_cells]

    @property
    def face_room(self) -> np.ndarray:
        """(F, 2) the room (an index into :attr:`room_volumes`) on the front (the side
        the face's normal points to) and on the back of each face, or -1 where a side
        faces no room: the outside, or the inside of a solid."""
        cells = self.mesh.cells()
        index = np.full(len(cells.volume), -1, dtype=np.int64)
        index[self.air_cells] = np.arange(len(self.air_cells))
        return index[cells.side]

    def room_of(self, points) -> np.ndarray:
        """The room each point lies in (an index into :attr:`room_volumes`), or -1."""
        mesh = self.mesh
        return cell_of(mesh.vertices, mesh.faces, mesh.cells(), points, self.air_cells)

    def face_transmission(self, bands: Resolution = "octave") -> np.ndarray | None:
        """(F, bands) share of sound each face lets through, or None when none does."""
        names = self.material_names
        rows = [self.scene.materials[n].transmission_coefficients(bands) for n in names]
        if all(r is None for r in rows):
            return None
        width = len(band_centres(bands))
        table = np.array([np.zeros(width) if r is None else r for r in rows] + [np.zeros(width)])
        return table[self.face_materials]

    def _sides(self, bands: Resolution) -> tuple[np.ndarray, np.ndarray]:
        """Each face's area as the rooms see it (m2), and the share of the sound meeting
        it that it takes out of them per band, both summed over its sides that face a
        room: a partition or a free-standing panel counts twice, an opening between two
        rooms not at all. A face between two rooms takes what it absorbs, not what it
        passes on. Without rooms (an open model), every face counts once."""
        areas = self.mesh.face_areas()
        alpha = np.array(
            [self.scene.materials[n].coefficients(bands)[0] for n in self.material_names]
            + [np.zeros(len(band_centres(bands)))]
        )[self.face_materials]
        if not len(self.air_cells):
            return areas, alpha
        sides = (self.face_room >= 0).sum(axis=1)
        through = self.face_transmission(bands)
        if through is None:
            return areas * sides, alpha * sides[:, None]
        between = sides == 2
        taken = np.where(between[:, None], alpha - through, alpha) * sides[:, None]
        opening = between & np.all(alpha >= 1.0, axis=1) & np.all(through >= 1.0, axis=1)
        return np.where(opening, 0.0, areas * sides), taken

    def area_by_material(self) -> dict[str, float]:
        """Surface area (m2) per material, in ``material_names`` order."""
        areas = self.mesh.face_areas()
        return {
            name: float(areas[self.face_materials == k].sum())
            for k, name in enumerate(self.material_names)
        }

    def absorption_area(self, bands: Resolution = "octave") -> np.ndarray:
        """Equivalent absorption area per octave band, or per third-octave band with
        ``bands="third_octave"`` (m2): the sum of S * alpha over the sides of the faces
        that face a room. With partitions or openings between rooms, it is that of all
        the rooms as one: what passes from one room to another is not lost."""
        return self.mesh.face_areas() @ self._sides(bands)[1]

    def air_attenuation(self, bands: Resolution = "octave") -> np.ndarray:
        """Energy attenuation of the scene's air per octave band, or per third-octave
        band with ``bands="third_octave"`` (1/m), ISO 9613-1."""
        air = self.scene.air
        return energy_attenuation_per_m(
            air.temperature_c, air.humidity_percent, air.pressure_kpa, bands
        )

    def speed_of_sound(self) -> float:
        return speed_of_sound(self.scene.air.temperature_c)

    def reverberation_time(
        self,
        formula: Literal["sabine", "eyring", "arau_puchades"] = "sabine",
        air_absorption: bool = False,
        bands: Resolution = "octave",
    ) -> np.ndarray:
        """Statistical reverberation time per octave band, or per third-octave band with
        ``bands="third_octave"`` (s).

        Sabine and Eyring assume a diffuse field; they are a sanity check, not a
        prediction. ``arau_puchades`` (Arau-Puchades, Acustica 65, 1988) allows for
        absorption spread unevenly between the room's directions: an Eyring time for each
        of the x, y and z axes from the surfaces facing it, combined as a geometric mean
        weighted by their area. Each face's area is shared between the axes by the
        squares of its normal's components, so in a box aligned with the axes every wall
        counts for the axis it faces. ``air_absorption`` adds the ``4 m V`` term of
        ISO 9613-1 air.
        """
        if self.volume_m3 is None:
            raise ValueError("reverberation time needs a closed model")
        volume = self.volume_m3
        constant = 24.0 * math.log(10.0) * volume / self.speed_of_sound()
        air = 4.0 * self.air_attenuation(bands) * volume if air_absorption else 0.0
        counted, taken = self._sides(bands)
        surface = float(counted.sum())

        def eyring(absorption: np.ndarray, area: float) -> np.ndarray:
            mean_alpha = np.clip(absorption / area, 0.0, 1.0 - 1e-12)
            with np.errstate(divide="ignore"):
                return constant / (-surface * np.log1p(-mean_alpha) + air)

        absorption = self.mesh.face_areas() @ taken
        if formula == "sabine":
            with np.errstate(divide="ignore"):
                return constant / (absorption + air)
        if formula == "eyring":
            return eyring(absorption, surface)
        if formula == "arau_puchades":
            # Each face's area shared between the axes; its sides that face a room count.
            share = self.mesh.face_areas()[:, None] * self.mesh.face_normals() ** 2  # (F, 3)
            sides = counted / np.maximum(self.mesh.face_areas(), 1e-300)
            total = float((share * sides[:, None]).sum())
            log_time = np.zeros(taken.shape[1])
            for area, axis_absorption in zip(
                (share * sides[:, None]).sum(axis=0), share.T @ taken, strict=True
            ):
                if area > 0:
                    log_time += area / total * np.log(eyring(axis_absorption, area))
            return np.exp(log_time)
        raise ValueError(f"unknown formula {formula!r}; use 'sabine', 'eyring' or 'arau_puchades'")


def outline_edges(room: Room) -> tuple[np.ndarray, np.ndarray]:
    """The model's edges seen from above, for a plan of the room: (E, 2, 2) end points
    (m) of the edges where the surface folds or ends, and (E,) the label of a face
    beside each. Vertical edges, points from above, are left out."""
    mesh = room.mesh
    faces = mesh.faces
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    unique, start, count = np.unique(edges, axis=0, return_index=True, return_counts=True)
    normals = mesh.face_normals()
    keep = count != 2
    pairs = count == 2
    first, second = owner[start[pairs]], owner[start[pairs] + 1]
    fold = np.abs(np.sum(normals[first] * normals[second], axis=1)) < math.cos(math.radians(10))
    keep[np.flatnonzero(pairs)[fold]] = True
    points = mesh.vertices[unique[keep]][:, :, :2]
    # Edges that are points from above (vertical edges) add nothing.
    long = np.linalg.norm(points[:, 1] - points[:, 0], axis=1) > 1e-9
    labels = mesh.face_labels[owner[start[keep]]]
    return points[long], labels[long]
