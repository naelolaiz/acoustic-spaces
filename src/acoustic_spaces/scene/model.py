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
        preset: audience_upholstered_chairs_1   # built-in database entry
        scattering: theatre_audience
      panels:
        file: panels.csv                        # BRAS-style coefficient table
    assignments:
      walls: plaster           # geometry group, material or layer name -> material
      seats: audience
    default_material: plaster  # for surfaces no assignment matches
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
always have one value per octave band in ``OCTAVE_BANDS_HZ``. Relative paths are
resolved against the scene file's folder.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from acoustic_spaces.materials import (
    OCTAVE_BANDS_HZ,
    builtin_database,
    read_coefficients_csv,
    to_octave_bands,
)

Vector3 = tuple[float, float, float]
Coefficient = Annotated[float, Field(ge=0.0, le=1.0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _per_band(name: str, value: list[float]) -> list[float]:
    if len(value) != len(OCTAVE_BANDS_HZ):
        raise ValueError(f"expected {len(OCTAVE_BANDS_HZ)} {name} values, got {len(value)}")
    return value


def _resolve_path(value: Path, info: ValidationInfo) -> Path:
    base_dir = (info.context or {}).get("base_dir")
    if base_dir is not None and not value.is_absolute():
        return (Path(base_dir) / value).resolve()
    return value


def _describe(origin: str, extrapolated: list[int]) -> str:
    if not extrapolated:
        return origin
    bands = ", ".join(str(f) for f in extrapolated)
    return f"{origin} ({bands} Hz copied from the nearest octave)"


class Orientation(_Model):
    """Yaw, pitch and roll in degrees; all zero looks down +x with +z up."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def forward(self) -> tuple[float, float, float]:
        """Unit vector of the look direction: yaw turns from +x towards +y, pitch up."""
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        return (
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            math.sin(pitch),
        )


class Geometry(_Model):
    file: Path = Field(description="mesh file: OBJ, STL, PLY, OFF, glTF/GLB or DXF")
    units: Literal["m", "cm", "mm", "in", "ft"] = "m"
    up_axis: Literal["y", "z"] | None = Field(
        default=None, description="vertical axis of the file; None uses the format's convention"
    )

    @field_validator("file")
    @classmethod
    def _resolve_file(cls, value: Path, info: ValidationInfo) -> Path:
        return _resolve_path(value, info)


DEFAULT_SCATTERING = 0.1


class Material(_Model):
    """Random-incidence absorption and scattering per octave band.

    Coefficients come from, in order of precedence: values given here, a
    coefficient file (``file``), or a built-in database entry (``preset``).
    ``scattering`` may also name a scattering entry of the built-in database.
    Data in other bands is mapped onto the octaves; octaves without data take the
    nearest octave's value, and ``source`` says which ones.
    """

    preset: str | None = Field(default=None, description="built-in database entry")
    file: Path | None = Field(default=None, description="CSV: frequencies, absorption, scattering")
    absorption: list[Coefficient] | None = None
    scattering: list[Coefficient] | Coefficient | str | None = None
    source: str | None = Field(default=None, description="where the coefficients come from")

    @field_validator("absorption")
    @classmethod
    def _absorption_per_band(cls, value: list[float] | None) -> list[float] | None:
        return None if value is None else _per_band("absorption", value)

    @field_validator("scattering")
    @classmethod
    def _scattering_per_band(cls, value: list[float] | float | str | None):
        if isinstance(value, float | int):
            return [float(value)] * len(OCTAVE_BANDS_HZ)
        if isinstance(value, list):
            return _per_band("scattering", value)
        return value

    @field_validator("file")
    @classmethod
    def _resolve_file(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def _resolve_coefficients(self) -> Material:
        if self.preset is not None and self.file is not None:
            raise ValueError("give either a preset or a file, not both")
        absorption = scattering = None
        origin = scattering_origin = None

        if self.preset is not None:
            entry = _database_entry(self.preset, "absorption")
            absorption, extrapolated = entry.octave_bands()
            origin = _describe(entry.source, extrapolated)
        elif self.file is not None:
            if not self.file.exists():
                raise ValueError(f"coefficient file not found: {self.file}")
            freqs, alpha, scatter = read_coefficients_csv(self.file)
            absorption, extrapolated = to_octave_bands(freqs, alpha)
            origin = _describe(f"file {self.file.name}", extrapolated)
            if scatter is not None:
                scattering, _ = to_octave_bands(freqs, scatter)

        if isinstance(self.scattering, str):
            entry = _database_entry(self.scattering, "scattering")
            scattering, extrapolated = entry.octave_bands()
            scattering_origin = "scattering: " + _describe(entry.source, extrapolated)
        elif self.scattering is not None:
            scattering = self.scattering

        if self.absorption is None:
            if absorption is None:
                raise ValueError("give absorption values, a preset or a file")
            self.absorption = _clip(absorption)
        else:
            origin = None
        self.scattering = _clip(scattering or [DEFAULT_SCATTERING] * len(OCTAVE_BANDS_HZ))
        if self.source is None:
            origins = [o for o in (origin, scattering_origin) if o]
            self.source = "; ".join(origins) or None
        return self


def _database_entry(name: str, kind: Literal["absorption", "scattering"]):
    try:
        return builtin_database().get(name, kind=kind)
    except KeyError as exc:
        raise ValueError(exc.args[0]) from None


def _clip(values: list[float]) -> list[float]:
    return [min(max(float(v), 0.0), 1.0) for v in values]


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
        default_factory=dict, description="geometry group, material or layer name -> material"
    )
    default_material: str | None = Field(
        default=None, description="material for surfaces that no assignment matches"
    )
    sources: list[Source] = Field(min_length=1)
    listeners: list[Listener] = Field(min_length=1)
    air: Air = Field(default_factory=Air)
    simulation: Simulation = Field(default_factory=Simulation)

    @model_validator(mode="after")
    def _check_references(self) -> Scene:
        referenced = set(self.assignments.values())
        if self.default_material is not None:
            referenced.add(self.default_material)
        unknown = sorted(referenced - set(self.materials))
        if unknown:
            raise ValueError(f"assignments reference unknown materials: {', '.join(unknown)}")
        names = [s.name for s in self.sources] + [listener.name for listener in self.listeners]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"source and listener names must be unique: {', '.join(duplicates)}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scene:
        """Load a scene file; relative paths in it are resolved against its folder."""
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls.model_validate(data, context={"base_dir": path.resolve().parent})

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)
