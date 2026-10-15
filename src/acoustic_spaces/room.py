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
from acoustic_spaces.directivity import Directivity, load_directivity
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
from acoustic_spaces.materials.bands import Resolution, band_centres
from acoustic_spaces.scene import Scene

UNASSIGNED = -1
#: A listener closer than this (metres) to a source has no response: the direct
#: sound's level grows without bound as the distance goes to zero.
MIN_SOURCE_DISTANCE_M = 0.01


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


def _directivity_key(spec) -> str:
    return spec if isinstance(spec, str) else spec.model_dump_json()


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
        if reuse is not None and reuse.scene.geometry == geometry:
            mesh, report = reuse.mesh, reuse.report
        else:
            scale = UNIT_SCALE[geometry.units or "m"]
            if geometry.file is not None:
                raw = load_mesh(geometry.file, units=geometry.units, up_axis=geometry.up_axis)
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

        sources_inside = listeners_inside = None
        if report.closed:
            sources_inside = points_inside(mesh, [s.position for s in scene.sources])
            listeners_inside = points_inside(mesh, [lis.position for lis in scene.listeners])

        names = set(mesh.label_names())
        unused = [key for key in scene.assignments if key not in names]
        loaded = {}
        if reuse is not None:
            for source, directivity in zip(reuse.scene.sources, reuse.directivities, strict=True):
                loaded[_directivity_key(source.directivity)] = directivity
        directivities = []
        for source in scene.sources:
            key = _directivity_key(source.directivity)
            if key not in loaded:
                try:
                    loaded[key] = load_directivity(source.directivity)
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
        for listener in self.scene.listeners:
            for source in self.scene.sources:
                if math.dist(listener.position, source.position) < MIN_SOURCE_DISTANCE_M:
                    out.append(
                        f"listener {listener.name!r} at {list(listener.position)} is within "
                        f"{MIN_SOURCE_DISTANCE_M * 100:g} cm of source {source.name!r}, too close "
                        "for a response (the direct sound has no finite level there); move "
                        "them apart"
                    )
        return out

    def warnings(self) -> list[str]:
        """Things worth a look that do not stop a simulation."""
        r = self.report
        out = []
        if not r.closed:
            out.append(
                f"the model is not closed ({r.boundary_edges} boundary edges, "
                f"{r.odd_edges} edges shared by three or another odd number of faces): "
                "rays can escape, and the volume and inside checks are skipped"
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

    def absorption_area(self, bands: Resolution = "octave") -> np.ndarray:
        """Equivalent absorption area per octave band, or per third-octave band with
        ``bands="third_octave"`` (m2): the sum of S * alpha."""
        total = np.zeros(len(band_centres(bands)))
        for name, area in self.area_by_material().items():
            total += area * np.asarray(self.scene.materials[name].coefficients(bands)[0])
        return total

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
        surface = self.report.surface_area_m2

        def eyring(absorption: np.ndarray, area: float) -> np.ndarray:
            mean_alpha = np.clip(absorption / area, 0.0, 1.0 - 1e-12)
            with np.errstate(divide="ignore"):
                return constant / (-surface * np.log1p(-mean_alpha) + air)

        if formula == "sabine":
            with np.errstate(divide="ignore"):
                return constant / (self.absorption_area(bands) + air)
        if formula == "eyring":
            return eyring(self.absorption_area(bands), surface)
        if formula == "arau_puchades":
            share = self.mesh.face_areas()[:, None] * self.mesh.face_normals() ** 2  # (F, 3)
            alpha = np.array(
                [self.scene.materials[n].coefficients(bands)[0] for n in self.material_names]
            )
            face_alpha = np.where(
                (self.face_materials == UNASSIGNED)[:, None], 0.0, alpha[self.face_materials]
            )
            total = share.sum()
            log_time = np.zeros(face_alpha.shape[1])
            for area, absorption in zip(share.sum(axis=0), share.T @ face_alpha, strict=True):
                if area > 0:
                    log_time += area / total * np.log(eyring(absorption, area))
            return np.exp(log_time)
        raise ValueError(f"unknown formula {formula!r}; use 'sabine', 'eyring' or 'arau_puchades'")
