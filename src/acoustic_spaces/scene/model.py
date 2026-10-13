"""Pydantic models for the scene file.

A scene file is YAML with this shape::

    name: Small lecture room
    geometry:
      file: room.obj          # relative to the scene file
      units: m
    materials:
      plaster:
        absorption: [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05]
        scattering: 0.1
      audience:
        absorption: [0.3, 0.4, 0.6, 0.7, 0.8, 0.8, 0.8, 0.8]
    assignments:
      walls: plaster           # geometry group name -> material
      seats: audience
    sources:
      - name: speaker
        position: [2.0, 1.5, 1.2]
    listeners:
      - name: row-3
        position: [6.0, 1.5, 1.2]
        orientation: {yaw: 180, pitch: 0, roll: 0}
    simulation:
      early: {method: ray_tracing, max_order: 3}
      late: {method: stochastic_ray_tracing}

Coordinates are in metres after unit conversion, with +z up. Coefficient lists
always have one value per octave band in ``OCTAVE_BANDS_HZ``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Octave-band centre frequencies used for all frequency-dependent data (Hz).
OCTAVE_BANDS_HZ: tuple[int, ...] = (63, 125, 250, 500, 1000, 2000, 4000, 8000)

Vector3 = tuple[float, float, float]
Coefficient = Annotated[float, Field(ge=0.0, le=1.0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _per_band(name: str, value: list[float]) -> list[float]:
    if len(value) != len(OCTAVE_BANDS_HZ):
        raise ValueError(f"expected {len(OCTAVE_BANDS_HZ)} {name} values, got {len(value)}")
    return value


class Orientation(_Model):
    """Yaw, pitch and roll in degrees; all zero looks down +x with +z up."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class Geometry(_Model):
    file: Path = Field(description="mesh file (OBJ first; STL, PLY, glTF, DXF in M1)")
    units: Literal["m", "cm", "mm", "in", "ft"] = "m"


class Material(_Model):
    """Random-incidence absorption and scattering per octave band."""

    absorption: list[Coefficient]
    scattering: list[Coefficient] | Coefficient = 0.1
    source: str | None = Field(default=None, description="where the coefficients come from")

    @field_validator("absorption")
    @classmethod
    def _absorption_per_band(cls, value: list[float]) -> list[float]:
        return _per_band("absorption", value)

    @field_validator("scattering")
    @classmethod
    def _scattering_per_band(cls, value: list[float] | float) -> list[float]:
        if isinstance(value, float | int):
            return [float(value)] * len(OCTAVE_BANDS_HZ)
        return _per_band("scattering", value)


class Source(_Model):
    name: str
    position: Vector3
    orientation: Orientation = Field(default_factory=Orientation)
    directivity: Literal[
        "omni", "subcardioid", "cardioid", "supercardioid", "hypercardioid", "figure8"
    ] = "omni"
    level_db: float = Field(default=94.0, description="sound power level in dB re 1 pW")


class Listener(_Model):
    name: str
    position: Vector3
    orientation: Orientation = Field(default_factory=Orientation)


class Air(_Model):
    temperature_c: float = Field(default=20.0, ge=-20.0, le=50.0)
    humidity_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    pressure_kpa: float = Field(default=101.325, gt=0.0)


class EarlyStage(_Model):
    """Early reflections: exact specular paths with delay and arrival direction."""

    method: Literal["ray_tracing", "image_sources"] = "ray_tracing"
    max_order: int = Field(default=3, ge=0, le=20)
    rays: int = Field(default=20_000, ge=1, description="ray tracing only")


class LateStage(_Model):
    """Late reverberation: stochastic energy decay per band, time slot and direction."""

    method: Literal["stochastic_ray_tracing", "statistical"] = "stochastic_ray_tracing"
    rays: int = Field(default=100_000, ge=1, description="stochastic ray tracing only")
    diffuse_rain: bool = True


class Simulation(_Model):
    early: EarlyStage = Field(default_factory=EarlyStage)
    late: LateStage = Field(default_factory=LateStage)
    transition_ms: float | None = Field(
        default=None, gt=0.0, description="early to late handover; None uses the mixing time"
    )
    ir_length_s: float = Field(default=2.0, gt=0.0)
    sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    seed: int | None = Field(default=None, description="random seed, for reproducible runs")


class Scene(_Model):
    name: str = "untitled"
    geometry: Geometry
    materials: dict[str, Material] = Field(min_length=1)
    assignments: dict[str, str] = Field(
        default_factory=dict, description="geometry group name -> material name"
    )
    sources: list[Source] = Field(min_length=1)
    listeners: list[Listener] = Field(min_length=1)
    air: Air = Field(default_factory=Air)
    simulation: Simulation = Field(default_factory=Simulation)

    @model_validator(mode="after")
    def _check_references(self) -> Scene:
        unknown = sorted(set(self.assignments.values()) - set(self.materials))
        if unknown:
            raise ValueError(f"assignments reference unknown materials: {', '.join(unknown)}")
        names = [s.name for s in self.sources] + [listener.name for listener in self.listeners]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"source and listener names must be unique: {', '.join(duplicates)}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scene:
        """Load a scene file; the geometry path is resolved relative to it."""
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        scene = cls.model_validate(data)
        if not scene.geometry.file.is_absolute():
            scene.geometry.file = (path.parent / scene.geometry.file).resolve()
        return scene

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
