"""A scene with its geometry loaded: a material on every surface, and checks.

:class:`Room` is what the simulation stages consume. It also gives the classic
statistical estimates (Sabine and Eyring reverberation times) so a model can be
sanity-checked before any ray is traced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from acoustic_spaces.air import energy_attenuation_per_m, speed_of_sound
from acoustic_spaces.geometry import (
    UNIT_SCALE,
    MeshReport,
    RoomMesh,
    box_mesh,
    check_mesh,
    clean_mesh,
    extruded_mesh,
    load_mesh,
    points_inside,
)
from acoustic_spaces.scene import Scene

UNASSIGNED = -1


def _label_text(label: tuple[str, ...]) -> str:
    return " / ".join(label) if label else "(unnamed)"


def _material_for(label: tuple[str, ...], scene: Scene) -> str | None:
    """Assignments win over names that match a material directly; then the default."""
    for name in label:
        if name in scene.assignments:
            return scene.assignments[name]
    for name in label:
        if name in scene.materials:
            return name
    return scene.default_material


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

    @classmethod
    def from_yaml(cls, path: str | Path) -> Room:
        return cls.from_scene(Scene.from_yaml(path))

    @classmethod
    def from_scene(cls, scene: Scene, *, reuse: Room | None = None) -> Room:
        """Load the scene's geometry and give every surface its material.

        ``reuse`` is an earlier room: when its geometry settings equal the scene's, its
        mesh is kept instead of being loaded and checked again (an editor changing
        materials or positions).
        """
        geometry = scene.geometry
        if reuse is not None and reuse.scene.geometry == geometry:
            mesh, report = reuse.mesh, reuse.report
        else:
            scale = UNIT_SCALE[geometry.units]
            if geometry.file is not None:
                raw = load_mesh(geometry.file, units=geometry.units, up_axis=geometry.up_axis)
            elif geometry.box is not None:
                raw = box_mesh(tuple(scale * v for v in geometry.box))
            else:
                raw = extruded_mesh(scale * np.array(geometry.floor_plan), scale * geometry.height)
            mesh, merged, removed = clean_mesh(raw)
            report = check_mesh(mesh, merged, removed)

        material_names = list(scene.materials)
        index = {name: k for k, name in enumerate(material_names)}
        per_label = np.array(
            [index.get(_material_for(label, scene), UNASSIGNED) for label in mesh.labels]
            or [UNASSIGNED],
            dtype=np.int64,
        )
        face_materials = per_label[mesh.face_labels]

        sources_inside = listeners_inside = None
        if report.closed:
            sources_inside = points_inside(mesh, [s.position for s in scene.sources])
            listeners_inside = points_inside(mesh, [lis.position for lis in scene.listeners])

        names = set(mesh.label_names())
        unused = [key for key in scene.assignments if key not in names]
        return cls(
            scene=scene,
            mesh=mesh,
            report=report,
            material_names=material_names,
            face_materials=face_materials,
            sources_inside=sources_inside,
            listeners_inside=listeners_inside,
            unused_assignments=unused,
        )

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
        """Issues that would make a simulation wrong; fix these first."""
        out = []
        unassigned = self.unassigned_labels()
        if unassigned:
            listed = ", ".join(f"{name} ({area:.1f} m2)" for name, area in unassigned.items())
            out.append(
                f"surfaces without a material: {listed}; "
                "add them to assignments or set default_material"
            )
        for kind, items, inside in (
            ("source", self.scene.sources, self.sources_inside),
            ("listener", self.scene.listeners, self.listeners_inside),
        ):
            if inside is None:
                continue
            for item, ok in zip(items, inside, strict=True):
                if not ok:
                    out.append(f"{kind} {item.name!r} at {list(item.position)} is outside the room")
        return out

    def warnings(self) -> list[str]:
        """Things worth a look that do not stop a simulation."""
        r = self.report
        out = []
        if not r.closed:
            out.append(
                f"the model is not closed ({r.boundary_edges} boundary edges, "
                f"{r.non_manifold_edges} edges shared by more than two faces): rays can "
                "escape, and the volume and inside checks are skipped"
            )
        if r.inconsistent_edges:
            out.append(
                f"{r.inconsistent_edges} edges join faces with opposite orientation; "
                "absorption is two-sided, so this only matters for the volume"
            )
        for key in self.unused_assignments:
            out.append(f"assignment {key!r} matches no name in the geometry")
        return out

    # -- statistics ---------------------------------------------------------------

    @property
    def volume_m3(self) -> float | None:
        return self.report.volume_m3

    def area_by_material(self) -> dict[str, float]:
        """Surface area (m2) per material, in ``material_names`` order."""
        areas = self.mesh.face_areas()
        return {
            name: float(areas[self.face_materials == k].sum())
            for k, name in enumerate(self.material_names)
        }

    def absorption_area(self) -> np.ndarray:
        """Equivalent absorption area per octave band (m2): the sum of S * alpha."""
        total = np.zeros(len(self.scene.materials[self.material_names[0]].absorption))
        for name, area in self.area_by_material().items():
            total += area * np.asarray(self.scene.materials[name].absorption)
        return total

    def air_attenuation(self) -> np.ndarray:
        """Energy attenuation of the scene's air per octave band (1/m), ISO 9613-1."""
        air = self.scene.air
        return energy_attenuation_per_m(air.temperature_c, air.humidity_percent, air.pressure_kpa)

    def speed_of_sound(self) -> float:
        return speed_of_sound(self.scene.air.temperature_c)

    def reverberation_time(
        self, formula: Literal["sabine", "eyring"] = "sabine", air_absorption: bool = False
    ) -> np.ndarray:
        """Statistical reverberation time per octave band (s).

        These assume a diffuse field; they are a sanity check, not a prediction.
        ``air_absorption`` adds the ``4 m V`` term of ISO 9613-1 air.
        """
        if self.volume_m3 is None:
            raise ValueError("reverberation time needs a closed model")
        volume = self.volume_m3
        constant = 24.0 * math.log(10.0) * volume / self.speed_of_sound()
        absorption = self.absorption_area()
        air = 4.0 * self.air_attenuation() * volume if air_absorption else 0.0
        if formula == "sabine":
            with np.errstate(divide="ignore"):
                return constant / (absorption + air)
        if formula == "eyring":
            surface = self.report.surface_area_m2
            mean_alpha = np.clip(absorption / surface, 0.0, 1.0 - 1e-12)
            with np.errstate(divide="ignore"):
                return constant / (-surface * np.log1p(-mean_alpha) + air)
        raise ValueError(f"unknown formula {formula!r}; use 'sabine' or 'eyring'")
