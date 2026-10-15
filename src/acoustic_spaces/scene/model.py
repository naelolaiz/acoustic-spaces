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
        directivity: cardioid  # or {pattern: [...per band]} or {file: speaker.sofa}
    listeners:
      - name: row-3
        position: [6.0, 1.5, 1.2]
        orientation: {yaw: 180, pitch: 0, roll: 0}
    simulation:
      early: {method: ray_tracing, max_order: 3, diffraction: first_order}
      late: {method: stochastic_ray_tracing}
    outputs:                   # formats for the simulate command (default: mono)
      - mono
      - binaural               # spherical head model; or {format: binaural, hrtf: x.sofa}
      - {format: ambisonics, order: 3}
      - {format: loudspeakers, layout: "5.0"}

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
    PositiveFloat,
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
    # Unknown keys are typing mistakes; NaN and infinity (YAML's .nan and .inf) are
    # no positions, angles or coefficients.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


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
    """Yaw, pitch and roll in degrees; all zero looks down +x with +z up.

    Yaw turns left (from +x towards +y), pitch looks up, and roll tilts the head to the
    right (the left ear goes up), applied in the order roll, pitch, yaw.
    """

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

    def rotation(self) -> list[list[float]]:
        """3 x 3 matrix whose columns are the forward, left and up axes in room
        coordinates; it turns head coordinates (x forward, y left, z up) into room ones."""
        cy, sy = math.cos(math.radians(self.yaw)), math.sin(math.radians(self.yaw))
        cp, sp = math.cos(math.radians(self.pitch)), math.sin(math.radians(self.pitch))
        cr, sr = math.cos(math.radians(self.roll)), math.sin(math.radians(self.roll))
        forward = (cp * cy, cp * sy, sp)
        left = (-cy * sp * sr - sy * cr, -sy * sp * sr + cy * cr, cp * sr)
        up = (
            forward[1] * left[2] - forward[2] * left[1],
            forward[2] * left[0] - forward[0] * left[2],
            forward[0] * left[1] - forward[1] * left[0],
        )
        return [[forward[i], left[i], up[i]] for i in range(3)]


class Geometry(_Model):
    """The room's surfaces: a CAD file, or a box or extruded floor plan built from
    dimensions (surfaces ``floor``, ``ceiling``, ``walls`` and ``wall_1``, ``wall_2``...)."""

    file: Path | None = Field(
        default=None,
        description="mesh file: OBJ, STL, PLY, OFF, glTF/GLB, DXF or SketchUp (sketchup extra)",
    )
    box: tuple[PositiveFloat, PositiveFloat, PositiveFloat] | None = Field(
        default=None, description="length (x), width (y) and height (z) of a rectangular room"
    )
    floor_plan: list[tuple[float, float]] | None = Field(
        default=None, min_length=3, description="(x, y) corners of the floor, in order"
    )
    height: PositiveFloat | None = Field(default=None, description="floor_plan: room height")
    units: Literal["m", "cm", "mm", "in", "ft"] | None = Field(
        default=None,
        description=(
            "unit of the coordinates and dimensions; None takes the units a DXF file "
            "records ($INSUNITS), and metres otherwise"
        ),
    )
    up_axis: Literal["y", "z"] | None = Field(
        default=None, description="vertical axis of the file; None uses the format's convention"
    )

    @field_validator("file")
    @classmethod
    def _resolve_file(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def _one_kind(self) -> Geometry:
        given = [k for k in ("file", "box", "floor_plan") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("geometry needs exactly one of file, box or floor_plan")
        if (self.floor_plan is None) != (self.height is None):
            raise ValueError("height goes with floor_plan, and floor_plan needs it")
        return self

    def describe(self) -> str:
        if self.file is not None:
            return self.file.name
        units = self.units or "m"
        if self.box is not None:
            return "box " + " x ".join(f"{v:g}" for v in self.box) + f" {units}"
        corners = len(self.floor_plan or [])
        return f"floor plan with {corners} corners, {self.height:g} {units} high"


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


PatternName = Literal[
    "omni", "subcardioid", "cardioid", "supercardioid", "hypercardioid", "figure8"
]


class DirectivitySpec(_Model):
    """A source's directivity per octave band: first-order patterns (a name, or the
    ``a`` of ``a + (1 - a) cos(theta)``, for each band) or a measured balloon from a
    SOFA file or a CSV table (see :mod:`acoustic_spaces.directivity`)."""

    pattern: list[PatternName | Coefficient] | None = Field(
        default=None, description="one pattern name or value of a per octave band"
    )
    file: Path | None = Field(
        default=None, description="measured data: a SOFA file (FreeFieldDirectivityTF) or CSV"
    )
    measurement: int | None = Field(
        default=None,
        ge=0,
        description="SOFA: the measurement to use (a note, say); None averages them all",
    )

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: list | None) -> list | None:
        return None if value is None else _per_band("pattern", value)

    @field_validator("file")
    @classmethod
    def _resolve_file(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def _one_kind(self) -> DirectivitySpec:
        if (self.pattern is None) == (self.file is None):
            raise ValueError("give a directivity pattern per band or a file, not both")
        if self.measurement is not None and (
            self.file is None or self.file.suffix.lower() != ".sofa"
        ):
            raise ValueError("measurement is for SOFA directivity files")
        return self


class Source(_Model):
    name: str
    position: Vector3
    orientation: Orientation = Field(default_factory=Orientation)
    directivity: PatternName | DirectivitySpec = Field(
        default="omni",
        description="a pattern for every band, or per-band patterns or measured data",
    )
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
    """Early reflections: exact specular paths with delay and arrival direction, and
    the sound diffracted at the edges of the room."""

    method: Literal["ray_tracing", "image_sources"] = "ray_tracing"
    max_order: int = Field(default=3, ge=0, le=20)
    rays: int = Field(default=20_000, ge=1, description="ray tracing only")
    diffraction: Literal["none", "direct", "first_order"] = Field(
        default="first_order",
        description=(
            "sound bending round edges (pillars, balcony fronts, screens): none, of the "
            "direct sound only, or of the direct sound and first-order reflections"
        ),
    )


class LateStage(_Model):
    """Late reverberation: stochastic energy decay per band, time slot and direction."""

    method: Literal["stochastic_ray_tracing", "statistical"] = "stochastic_ray_tracing"
    rays: int = Field(
        default=50_000,
        ge=1,
        description=(
            "stochastic ray tracing only: rays per octave band (vector-based scattering "
            "traces each band with rays of its own where the bands scatter differently)"
        ),
    )
    reception: Literal["diffuse_rain", "secondary_sources", "detector"] = Field(
        default="diffuse_rain",
        description=(
            "how reflected energy reaches listeners: diffuse_rain (scattered energy "
            "radiates to each visible listener, specular rays are caught by a detector "
            "sphere), secondary_sources (all reflected energy radiates, as in ODEON's late "
            "method; smoother but assumes a diffuse field) or detector (the sphere only)"
        ),
    )
    receiver_radius: float | None = Field(
        default=None,
        gt=0.0,
        description="detector sphere radius in metres; None sizes it from volume and rays",
    )
    directions: int = Field(
        default=64,
        ge=0,
        le=1024,
        description=(
            "arrival directions the late energy is sorted into, for spatial outputs "
            "(stochastic ray tracing; the statistical tail arrives evenly from all around)"
        ),
    )
    splitting: bool = Field(
        default=True,
        description=(
            "split rays that carry much more energy than usual and thin out those that "
            "carry much less (unbiased); keeps late tails smooth where few rays survive"
        ),
    )


class Scattering(_Model):
    """How reflections scatter, in the early and the late stage alike."""

    method: Literal["vector", "random"] = Field(
        default="vector",
        description=(
            "where a reflected ray goes: vector (the specular direction and a random "
            "Lambert direction added, weighted by the scattering coefficient, as in "
            "ODEON) or random (specularly, or with the scattering coefficient as "
            "probability in a Lambert direction; before 3.0)"
        ),
    )
    from_size: bool = Field(
        default=True,
        description=(
            "add the scattering of a surface's limited size: below a frequency set by its "
            "width and length, the angle of incidence and the distance the sound came "
            "from, less of a reflection is specular (Christensen and Rindel, 2005)"
        ),
    )
    by_band: bool = Field(
        default=True,
        description=(
            "vector only: where the bands scatter differently, trace each band with rays "
            "of its own (exact; slower), or turn one ray for all bands by their mean "
            "scattering (faster; scatters the treble like the bass)"
        ),
    )


class WaveStage(_Model):
    """The lowest octave bands from a wave-based solver (finite differences in time on a
    cubic grid) instead of rays: room modes, and sound bending round everything."""

    enabled: bool = Field(
        default=False, description="compute the lowest octave bands with the wave solver"
    )
    max_band_hz: Literal[63, 125, 250, 500] = Field(
        default=125,
        description=(
            "the highest octave band it computes; each octave higher costs about 16 times as much"
        ),
    )
    points_per_wavelength: float = Field(
        default=8.0,
        ge=5.0,
        le=20.0,
        description="grid cells per wavelength at the top of each band's crossover",
    )


LAYOUT_NAMES = ("stereo", "5.0", "7.0", "5.0.4", "7.0.4", "5.1", "7.1", "5.1.4", "7.1.4")


class Speaker(_Model):
    name: str
    azimuth: float = Field(description="degrees, positive to the left")
    elevation: float = Field(default=0.0, ge=-90.0, le=90.0, description="degrees, positive up")
    lfe: bool = Field(default=False, description="a low-frequency effects channel (silent)")


class Output(_Model):
    """A format every response is rendered in: ``mono``, ``binaural``, ``ambisonics``
    (AmbiX) or ``loudspeakers``. A bare name stands for the format with its defaults."""

    format: Literal["mono", "binaural", "ambisonics", "loudspeakers"]
    order: int = Field(default=1, ge=0, le=7, description="ambisonics: the order")
    hrtf: Path | None = Field(
        default=None, description="binaural: a SOFA file; None uses a spherical head model"
    )
    layout: str | None = Field(
        default=None, description=f"loudspeakers: one of {', '.join(LAYOUT_NAMES)}"
    )
    speakers: list[Speaker] | None = Field(
        default=None, description="loudspeakers: a layout of your own, in channel order"
    )

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, value):
        if isinstance(value, str):
            return {"format": value}
        if isinstance(value, dict) and isinstance(value.get("layout"), int | float):
            # YAML reads `layout: 5.0` as a number.
            return {**value, "layout": f"{float(value['layout']):.1f}"}
        return value

    @field_validator("hrtf")
    @classmethod
    def _resolve_hrtf(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def _check_format(self) -> Output:
        if self.format == "loudspeakers":
            if (self.layout is None) == (self.speakers is None):
                raise ValueError("loudspeakers need either a layout or a list of speakers")
            if self.layout is not None and self.layout not in LAYOUT_NAMES:
                raise ValueError(
                    f"unknown layout {self.layout!r}; use one of {', '.join(LAYOUT_NAMES)}"
                )
            if self.speakers is not None and len(self.speakers) < 2:
                raise ValueError("a loudspeaker layout needs at least two speakers")
        elif self.layout is not None or self.speakers is not None:
            raise ValueError("layout and speakers are for the loudspeakers format")
        if self.hrtf is not None and self.format != "binaural":
            raise ValueError("hrtf is for the binaural format")
        return self


class Simulation(_Model):
    early: EarlyStage = Field(default_factory=EarlyStage)
    late: LateStage = Field(default_factory=LateStage)
    scattering: Scattering = Field(default_factory=Scattering)
    wave: WaveStage = Field(default_factory=WaveStage)
    transition_ms: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "early paths end this long after the direct sound; None hands over by "
            "reflection order (stochastic) or at the mixing time sqrt(V) ms (statistical)"
        ),
    )
    ir_length_s: float = Field(default=2.0, gt=0.0)
    sample_rate: int = Field(default=48_000, ge=8_000, le=192_000)
    seed: int | None = Field(default=None, description="random seed, for reproducible runs")


class Coverage(_Model):
    """A coverage map: room parameters on a grid over the audience area (see
    :mod:`acoustic_spaces.engine.coverage`)."""

    surfaces: list[str] | None = Field(
        default=None,
        description=(
            "surfaces the map lies over; None takes those named with 'audience', else 'floor'"
        ),
    )
    height: float = Field(default=1.2, gt=0.0, description="listening height above them (m)")
    spacing: float = Field(default=1.0, gt=0.0, description="distance between map points (m)")
    rays: int | None = Field(
        default=None, ge=1, description="rays per source and band; None uses simulation.late.rays"
    )
    noise_db: list[float] | float | None = Field(
        default=None,
        description="background noise for the STI (dB SPL, one value or one per octave band)",
    )

    @field_validator("noise_db")
    @classmethod
    def _noise_per_band(cls, value):
        if isinstance(value, list):
            return _per_band("noise_db", value)
        return value


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
    outputs: list[Output] = Field(
        default_factory=lambda: [Output(format="mono")],
        description="formats to write every response in (simulate command)",
    )
    coverage: Coverage | None = Field(
        default=None, description="coverage map settings (map command); None uses the defaults"
    )

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
        """Write the scene file. Paths to files under its folder are written relative to
        it, so the folder can move; others stay absolute."""
        path = Path(path)
        data = self.model_dump(mode="json", exclude_none=True)
        base = path.resolve().parent
        geometry = data["geometry"]
        if "file" in geometry:
            geometry["file"] = _portable(geometry["file"], base)
        for material in data["materials"].values():
            if "file" in material:
                material["file"] = _portable(material["file"], base)
        for output in data["outputs"]:
            if "hrtf" in output:
                output["hrtf"] = _portable(output["hrtf"], base)
        for source in data["sources"]:
            directivity = source.get("directivity")
            if isinstance(directivity, dict) and "file" in directivity:
                directivity["file"] = _portable(directivity["file"], base)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)


def _portable(file: str, base: Path) -> str:
    """``file`` relative to ``base`` when it lies inside it, with forward slashes."""
    try:
        return Path(file).resolve().relative_to(base).as_posix()
    except ValueError:
        return file
