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

Coordinates are in metres after unit conversion, with +z up. Coefficient lists have
one value per octave band in ``OCTAVE_BANDS_HZ`` or per third-octave band in
``THIRD_OCTAVE_BANDS_HZ``; ``simulation.bands`` sets which bands a simulation runs in,
and each list converts to them. Relative paths are resolved against the scene file's
folder.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
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
from pydantic_core import PydanticCustomError

from acoustic_spaces.air import SOUND_SPEED_0C
from acoustic_spaces.materials import (
    OCTAVE_BANDS_HZ,
    THIRD_OCTAVE_BANDS_HZ,
    builtin_database,
    read_coefficients_csv,
)
from acoustic_spaces.materials.bands import (
    Resolution,
    convert,
    finer_than_octaves,
    gain_levels,
    power_levels,
    to_bands,
)
from acoustic_spaces.materials.ground import PRESETS as GROUND_PRESETS
from acoustic_spaces.materials.ground import band_absorption, surface_impedance
from acoustic_spaces.materials.libraries import find_material, read_library
from acoustic_spaces.parameters import SPEECH_SPECTRUM_DB

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


def _per_band_or_third(name: str, value: list) -> list:
    if len(value) not in (len(OCTAVE_BANDS_HZ), len(THIRD_OCTAVE_BANDS_HZ)):
        raise ValueError(
            f"expected {len(OCTAVE_BANDS_HZ)} {name} values (octave bands) or "
            f"{len(THIRD_OCTAVE_BANDS_HZ)} (third-octave bands), got {len(value)}"
        )
    return value


def _from_data(frequencies, values) -> tuple[list[float], list[int]]:
    """Coefficients at arbitrary frequencies on third-octave bands when the data has
    that detail, else on octave bands, and the bands copied from the nearest one."""
    resolution = "third_octave" if finer_than_octaves(frequencies) else "octave"
    return to_bands(frequencies, values, resolution)


def _resolve_path(value: Path, info: ValidationInfo) -> Path:
    base_dir = (info.context or {}).get("base_dir")
    if base_dir is not None and not value.is_absolute():
        return (Path(base_dir) / value).resolve()
    return value


def _describe(origin: str, extrapolated: list[int]) -> str:
    if not extrapolated:
        return origin
    bands = ", ".join(str(f) for f in extrapolated)
    return f"{origin} ({bands} Hz copied from the nearest band)"


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


#: Names an outdoor scene gives its surfaces by kind; buildings and barriers take others.
OUTDOOR_NAMES = ("ground", "sky", "walls", "roofs", "buildings", "barriers", "zones")
#: The material of the sky of an outdoor scene unless the scene names its own: an
#: opening, which lets all sound out of the scene.
SKY = "sky"


def sky_material() -> Material:
    """The open sky: a surface that lets all sound through and reflects none."""
    return Material(transmission=1.0, scattering=0.0, source="the open sky")


class GroundZone(_Model):
    """A part of the ground with a material of its own (a road through a field, a lawn,
    a pond): a polygon on the ground, its surfaces ``<name>``, ``zones`` and ``ground``
    (an assignment to the zone's name wins over one to ``ground``). The part of it past
    the ground's edge is left out."""

    name: str | None = Field(default=None, description="None: zone_1, zone_2...")
    polygon: list[tuple[float, float]] = Field(
        min_length=3, description="(x, y) corners of the zone, in order"
    )


class Ground(_Model):
    """Flat ground under an open sky: it makes the scene an outdoor one (see
    :mod:`acoustic_spaces.geometry.outdoor`). Lengths are in the geometry's units."""

    height: float = Field(default=0.0, description="height (z) of the ground")
    extent: tuple[tuple[float, float], tuple[float, float]] | None = Field(
        default=None,
        description=(
            "[[x_min, y_min], [x_max, y_max]] of the ground; None: the buildings, barriers "
            "and model file, with margin round them"
        ),
    )
    margin: PositiveFloat = Field(
        default=30.0,
        description=(
            "ground round the buildings, barriers and model file when extent is left out, "
            "and sky above the highest of them when sky is"
        ),
    )
    sky: PositiveFloat | None = Field(
        default=None,
        description=(
            "height of the open sky above the ground; None: margin above the highest "
            "building, barrier or model"
        ),
    )
    zones: list[GroundZone] | None = Field(
        default=None,
        description="parts of the ground with materials of their own; they may not overlap",
    )

    def zone_names(self) -> list[str]:
        """The zones' names, ``zone_<n>`` for those without one."""
        return [z.name or f"zone_{k + 1}" for k, z in enumerate(self.zones or [])]

    @model_validator(mode="after")
    def _extent_has_area(self) -> Ground:
        if self.extent is not None:
            (x0, y0), (x1, y1) = self.extent
            if not (x0 < x1 and y0 < y1):
                raise ValueError(
                    "ground extent is [[x_min, y_min], [x_max, y_max]], each min < max"
                )
        return self


class Building(_Model):
    """A building standing on the ground: its footprint raised to a height (surfaces
    ``<name>_wall_1``..., ``<name>_walls``, ``<name>_roof``, and ``walls``, ``roofs``
    and ``buildings`` for all of them)."""

    name: str | None = Field(default=None, description="None: building_1, building_2...")
    footprint: list[tuple[float, float]] = Field(
        min_length=3, description="(x, y) corners of the footprint, in order"
    )
    height: PositiveFloat = Field(description="height of the roof above the ground")


class Barrier(_Model):
    """A thin screen standing on the ground along a path (surfaces ``<name>`` and
    ``barriers``)."""

    name: str | None = Field(default=None, description="None: barrier_1, barrier_2...")
    path: list[tuple[float, float]] = Field(
        min_length=2, description="(x, y) points along the barrier, in order"
    )
    height: PositiveFloat = Field(description="height of its top above the ground")


#: The foliage presets: ``dense``, ISO 9613-2's dense foliage, and ``none``, bare stems.
FOLIAGE_PRESETS = ("dense", "none")
#: ISO 9613-2 (1996), Annex A, Table A.1: what dense foliage takes from sound passing 20
#: to 200 m through it, dB per metre in the octave bands of ``OCTAVE_BANDS_HZ``.
DENSE_FOLIAGE_DB_PER_M = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.12)


class Foliage(_Model):
    """What the leaves and twigs of a vegetation volume take from sound passing through
    it, per metre: a preset, an attenuation per band, or the leaves themselves, their
    area per cubic metre and their width, from which the engine works out what they
    scatter and absorb in each band. ``albedo`` is the share of an attenuation that
    they scatter on, the rest they absorb; ``asymmetry`` how far forward they scatter
    it (the mean cosine of the angle, as in a Henyey-Greenstein phase function); leaves
    given by their size and density work both out for themselves."""

    preset: Literal["dense", "none"] | None = Field(
        default=None,
        description=(
            "dense: ISO 9613-2's dense foliage (Annex A), its attenuation from 20 to 200 m "
            "of foliage per metre; none: no foliage (bare stems, winter)"
        ),
    )
    attenuation: list[Annotated[float, Field(ge=0.0, le=10.0)]] | None = Field(
        default=None,
        description="dB per metre per octave or third-octave band, when no preset says it",
    )
    leaf_area_per_m3: PositiveFloat | None = Field(
        default=None,
        le=50.0,
        description=(
            "the leaves' area (one side's), square metres per cubic metre of the volume, "
            "with leaf_width: the engine works out what they scatter and absorb in each "
            "band from thin rigid discs of that size, turned every way alike"
        ),
    )
    leaf_width: PositiveFloat | None = Field(
        default=None,
        le=1.0,
        description="a leaf's width (m): the diameter of a disc of a leaf's area",
    )
    albedo: Coefficient = Field(
        default=0.0,
        description="share of what the foliage takes that it scatters on rather than absorbs",
    )
    asymmetry: float = Field(
        default=0.0,
        ge=-0.9,
        le=0.9,
        description=(
            "mean cosine of the angle the foliage scatters sound through: 0 every way "
            "alike, towards 1 onwards"
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _from_preset(cls, data):
        if isinstance(data, str):
            data = {"preset": data}
        if isinstance(data, dict) and data.get("preset") is not None:
            preset = data["preset"]
            if preset not in FOLIAGE_PRESETS:
                close = difflib.get_close_matches(str(preset), list(FOLIAGE_PRESETS), n=1)
                hint = f"; did you mean {close[0]}?" if close else ""
                raise ValueError(
                    f"no foliage preset named {preset!r}{hint} (the presets are "
                    f"{', '.join(FOLIAGE_PRESETS)})"
                )
        return data

    @field_validator("attenuation")
    @classmethod
    def _bands(cls, value: list[float] | None) -> list[float] | None:
        return None if value is None else _per_band_or_third("attenuation", value)

    @model_validator(mode="after")
    def _one_way(self) -> Foliage:
        if (self.leaf_area_per_m3 is None) != (self.leaf_width is None):
            raise ValueError("leaf_area_per_m3 and leaf_width go together")
        ways = [
            w
            for w, given in (
                ("a preset", self.preset is not None),
                ("an attenuation per band", self.attenuation is not None),
                ("leaves by their area and width", self.leaf_area_per_m3 is not None),
            )
            if given
        ]
        if len(ways) != 1:
            raise ValueError(
                "foliage takes a preset, an attenuation per band or leaves by their area and "
                "width, one of them"
            )
        if self.preset == "none" and (self.albedo or self.asymmetry):
            raise ValueError("foliage none has nothing to scatter: leave out albedo and asymmetry")
        if self.leaf_area_per_m3 is not None and (self.albedo or self.asymmetry):
            raise ValueError(
                "leaves given by their area and width work out what they scatter: leave out "
                "albedo and asymmetry"
            )
        return self

    @property
    def bare(self) -> bool:
        """Whether there is no foliage."""
        if self.preset is not None:
            return self.preset == "none"
        if self.leaf_area_per_m3 is not None:
            return False
        return not any(self.attenuation)

    def attenuation_db(
        self, resolution: Resolution = "octave", speed_of_sound: float = 343.0
    ) -> np.ndarray:
        """dB per metre in each band at ``resolution``: what the foliage takes in all,
        scattered on or absorbed; leaves given by their area and width take it for sound
        at ``speed_of_sound`` (m/s)."""
        if self.leaf_area_per_m3 is not None:
            from acoustic_spaces.engine.vegetation import DB_PER_NEPER, leaves_medium

            medium = leaves_medium(
                self.leaf_area_per_m3, self.leaf_width, resolution, speed_of_sound
            )
            return medium.extinction * DB_PER_NEPER
        if self.preset == "none":
            values = [0.0] * len(OCTAVE_BANDS_HZ)
        elif self.preset == "dense":
            values = list(DENSE_FOLIAGE_DB_PER_M)
        else:
            values = self.attenuation
        return np.asarray(convert(values, resolution), dtype=float)

    def describe(self) -> str:
        if self.preset is not None:
            return f"{self.preset} foliage" if self.preset == "dense" else "no foliage"
        if self.leaf_area_per_m3 is not None:
            return f"leaves of {self.leaf_area_per_m3:g} m²/m³, {self.leaf_width:g} m wide"
        return "foliage of " + "/".join(f"{v:g}" for v in self.attenuation) + " dB/m"


class Stems(_Model):
    """The trunks and stems of a vegetation volume: upright rigid cylinders, which
    scatter sound sideways, all of it (Bullen and Fricke 1982)."""

    per_m2: PositiveFloat = Field(le=100.0, description="stems per square metre of ground")
    diameter: PositiveFloat = Field(le=5.0, description="their mean diameter (m)")


class Vegetation(_Model):
    """Plants standing on the ground outdoors: a volume through which sound loses some of
    itself, scattered on or absorbed. Its shape is one of a ``footprint`` raised from
    ``base`` to ``height`` (a tree belt, a wood), a ``path`` of a ``width`` (a hedge),
    or a lone tree's crown over ``tree``: a spheroid ``crown_diameter`` across from
    ``base`` to ``height``. Lengths and heights are in the geometry's units, over the
    ground; ``foliage`` and ``stems`` say what fills it."""

    name: str | None = Field(default=None, description="None: vegetation_1, vegetation_2...")
    footprint: list[tuple[float, float]] | None = Field(
        default=None, min_length=3, description="(x, y) corners of a belt or wood, in order"
    )
    path: list[tuple[float, float]] | None = Field(
        default=None, min_length=2, description="(x, y) points along a hedge, in order"
    )
    width: PositiveFloat | None = Field(default=None, description="a hedge's width")
    tree: tuple[float, float] | None = Field(default=None, description="(x, y) of a lone tree")
    crown_diameter: PositiveFloat | None = Field(
        default=None, description="how wide a lone tree's crown is"
    )
    height: PositiveFloat = Field(description="height of the top over the ground")
    base: float = Field(
        default=0.0,
        ge=0.0,
        description="height of the bottom over the ground; a tree's, where its crown starts",
    )
    foliage: Foliage = Field(
        default_factory=lambda: Foliage(preset="dense"),
        description=(
            "what the leaves and twigs take: a preset (dense, none), values, or the leaves "
            "by their area and width"
        ),
    )
    stems: Stems | None = Field(default=None, description="trunks and stems in it, if any")

    @model_validator(mode="after")
    def _one_shape(self) -> Vegetation:
        given = [k for k in ("footprint", "path", "tree") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("vegetation needs exactly one of footprint, path or tree")
        if (self.width is not None) != (self.path is not None):
            raise ValueError("width goes with path, and a path needs it")
        if (self.crown_diameter is not None) != (self.tree is not None):
            raise ValueError("crown_diameter goes with tree, and a tree needs it")
        if self.base >= self.height:
            raise ValueError(f"base ({self.base:g}) must be below height ({self.height:g})")
        if self.foliage.bare and self.stems is None:
            raise ValueError("vegetation with no foliage and no stems does nothing")
        return self

    @property
    def kind(self) -> Literal["belt", "hedge", "tree"]:
        """belt (a footprint), hedge (a path) or tree."""
        if self.footprint is not None:
            return "belt"
        return "hedge" if self.path is not None else "tree"

    def describe(self) -> str:
        """What fills it, as text: its foliage and stems."""
        parts = [] if self.foliage.bare else [self.foliage.describe()]
        if self.stems is not None:
            parts.append(f"{self.stems.per_m2:g} stems/m², {self.stems.diameter:g} m across")
        return ", ".join(parts) or "nothing"


class Geometry(_Model):
    """The room's surfaces: a CAD file, or a box or extruded floor plan built from
    dimensions (surfaces ``floor``, ``ceiling``, ``walls`` and ``wall_1``, ``wall_2``...).

    Outdoors, ``ground`` lays flat ground under an open sky, with ``buildings`` and
    ``barriers`` built on it and, from ``file``, any other objects standing on it.
    ``vegetation`` adds trees, hedges and belts of them, which are no surfaces: sound
    passes through them, losing some of itself."""

    file: Path | None = Field(
        default=None,
        description=(
            "mesh file: OBJ, STL, PLY, OFF, glTF/GLB, DXF, Collada, IFC (ifc extra) or "
            "SketchUp (sketchup extra)"
        ),
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
            "unit of the coordinates and dimensions; None takes the units a DXF "
            "($INSUNITS), IFC or Collada file records, and metres otherwise"
        ),
    )
    up_axis: Literal["y", "z"] | None = Field(
        default=None, description="vertical axis of the file; None uses the format's convention"
    )
    space: str | None = Field(
        default=None,
        description=(
            "IFC file: the room (IfcSpace, by name, long name or GlobalId) to take on its "
            "own, from its space boundaries; None reads every building element"
        ),
    )

    ground: Ground | None = Field(
        default=None, description="flat ground under an open sky: an outdoor scene"
    )
    buildings: list[Building] | None = Field(
        default=None, description="outdoors: buildings standing on the ground"
    )
    barriers: list[Barrier] | None = Field(
        default=None, description="outdoors: thin screens standing on the ground"
    )
    vegetation: list[Vegetation] | None = Field(
        default=None, description="outdoors: trees, hedges and belts of them"
    )

    @field_validator("file")
    @classmethod
    def _resolve_file(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def _one_kind(self) -> Geometry:
        if self.ground is not None:
            return self._outdoor()
        if self.buildings is not None or self.barriers is not None:
            raise ValueError("buildings and barriers stand on the ground: add geometry.ground")
        if self.vegetation is not None:
            raise ValueError("vegetation stands on the ground: add geometry.ground")
        given = [k for k in ("file", "box", "floor_plan") if getattr(self, k) is not None]
        if len(given) != 1:
            raise ValueError("geometry needs exactly one of file, box or floor_plan")
        if (self.floor_plan is None) != (self.height is None):
            raise ValueError("height goes with floor_plan, and floor_plan needs it")
        if self.space is not None and (self.file is None or self.file.suffix.lower() != ".ifc"):
            raise ValueError("space names a room of an IFC file: give one as file")
        return self

    def _outdoor(self) -> Geometry:
        if self.box is not None or self.floor_plan is not None or self.height is not None:
            raise ValueError(
                "an outdoor scene (geometry.ground) takes buildings, barriers and a model "
                "file, not a box or floor_plan"
            )
        if self.space is not None:
            raise ValueError("space picks a room of an IFC file; outdoors, leave it out")
        if self.ground.extent is None and not (
            self.file or self.buildings or self.barriers or self.vegetation
        ):
            raise ValueError(
                "with no buildings, barriers, vegetation or model file, the ground needs an "
                "extent: [[x_min, y_min], [x_max, y_max]]"
            )
        plants = self.vegetation_names()
        repeated = sorted({n for n in plants if plants.count(n) > 1})
        if repeated:
            raise ValueError(f"vegetation names must be unique: {', '.join(repeated)}")
        names = self.building_names() + self.barrier_names() + self.ground.zone_names()
        reserved = sorted(set(names) & set(OUTDOOR_NAMES))
        if reserved:
            raise ValueError(f"{reserved[0]!r} names every surface of its kind; pick another name")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"building, barrier and zone names must be unique: {', '.join(duplicates)}"
            )
        return self

    def building_names(self) -> list[str]:
        """The buildings' names, ``building_<n>`` for those without one."""
        return [b.name or f"building_{k + 1}" for k, b in enumerate(self.buildings or [])]

    def barrier_names(self) -> list[str]:
        """The barriers' names, ``barrier_<n>`` for those without one."""
        return [b.name or f"barrier_{k + 1}" for k, b in enumerate(self.barriers or [])]

    def vegetation_names(self) -> list[str]:
        """The vegetation's names, ``vegetation_<n>`` for those without one."""
        return [v.name or f"vegetation_{k + 1}" for k, v in enumerate(self.vegetation or [])]

    def part_names(self) -> list[str]:
        """The names of an outdoor scene's buildings, barriers and vegetation, in turn:
        what an insertion loss can be given for (``simulation.insertion_loss_of``)."""
        return [*self.building_names(), *self.barrier_names(), *self.vegetation_names()]

    def describe(self) -> str:
        if self.ground is not None:
            parts = ["outdoors"]
            for count, one, many in (
                (len(self.buildings or []), "building", "buildings"),
                (len(self.barriers or []), "barrier", "barriers"),
                *(
                    (sum(v.kind == kind for v in self.vegetation or []), kind, f"{kind}s")
                    for kind in ("tree", "hedge", "belt")
                ),
            ):
                if count:
                    parts.append(f"{count} {one if count == 1 else many}")
            if self.file is not None:
                parts.append(self.file.name)
            return ", ".join(parts)
        if self.file is not None:
            return self.file.name + (f" (space {self.space})" if self.space else "")
        units = self.units or "m"
        if self.box is not None:
            return "box " + " x ".join(f"{v:g}" for v in self.box) + f" {units}"
        corners = len(self.floor_plan or [])
        return f"floor plan with {corners} corners, {self.height:g} {units} high"


DEFAULT_SCATTERING = 0.1


class Impedance(_Model):
    """The acoustic impedance of a porous ground (grass, soil, snow, gravel, asphalt),
    from its flow resistivity by an empirical model (see
    :mod:`acoustic_spaces.materials.ground`): a ``preset``, one of the ground classes of
    Nord2000, or a ``flow_resistivity``; with a ``thickness``, a layer of it on a hard
    backing.

    A material with an impedance takes its absorption from it, and reflects sound as a
    ground does: the spherical wave's reflection (Weyl-van der Pol), with its phase and
    the ground wave near grazing incidence, in place of the absorption."""

    preset: str | None = Field(
        default=None,
        description="a ground class of Nord2000: " + ", ".join(GROUND_PRESETS),
    )
    model: Literal["miki", "delany_bazley", "two_parameter"] = Field(
        default="miki",
        description=(
            "miki (passive at every frequency), delany_bazley, or two_parameter (a "
            "porosity that falls with depth)"
        ),
    )
    flow_resistivity: PositiveFloat | None = Field(
        default=None,
        description="kPa s/m²; with two_parameter, the effective flow resistivity",
    )
    porosity_rate: float | None = Field(
        default=None,
        ge=0.0,
        description="two_parameter: how fast the porosity falls with depth (1/m)",
    )
    thickness: PositiveFloat | None = Field(
        default=None, description="m: a layer of the ground on a hard backing"
    )

    @model_validator(mode="after")
    def _complete(self) -> Impedance:
        if self.preset is not None and self.preset not in GROUND_PRESETS:
            close = difflib.get_close_matches(self.preset, list(GROUND_PRESETS), n=1)
            hint = f"; did you mean {close[0]}?" if close else ""
            raise ValueError(
                f"no ground preset named {self.preset!r}{hint} (the presets are "
                f"{', '.join(GROUND_PRESETS)})"
            )
        if (self.preset is None) == (self.flow_resistivity is None):
            raise ValueError(
                "give the ground a preset or a flow_resistivity (kPa s/m²), one of them"
            )
        if self.model == "two_parameter":
            if self.porosity_rate is None:
                raise ValueError("the two_parameter model needs a porosity_rate (1/m)")
            if self.thickness is not None:
                raise ValueError(
                    "the two_parameter model takes no thickness: its porosity falls with depth"
                )
        elif self.porosity_rate is not None:
            raise ValueError("porosity_rate is for the two_parameter model")
        return self

    @property
    def resistivity(self) -> float:
        """The flow resistivity (kPa s/m²), the preset's when a preset is given."""
        if self.flow_resistivity is not None:
            return float(self.flow_resistivity)
        return GROUND_PRESETS[self.preset][0]

    def admittance(self, f, speed_of_sound: float = 343.0) -> np.ndarray:
        """The normalised admittance (1 / impedance) at frequencies ``f`` (Hz), in the
        engine's sign convention, e^(jωt)."""
        return 1.0 / surface_impedance(
            f,
            self.resistivity,
            self.model,
            porosity_rate=self.porosity_rate or 0.0,
            thickness=self.thickness,
            speed_of_sound=speed_of_sound,
        )

    def describe(self) -> str:
        """In a few words: the preset or the flow resistivity, and the rest."""
        model = {"miki": "Miki", "delany_bazley": "Delany-Bazley", "two_parameter": "two-parameter"}
        words = [self.preset] if self.preset else []
        words.append(f"{self.resistivity:g} kPa s/m²")
        if self.porosity_rate is not None:
            words.append(f"{self.porosity_rate:g}/m")
        if self.thickness is not None:
            words.append(f"{self.thickness:g} m thick")
        return f"{model[self.model]}: " + ", ".join(words)


class Material(_Model):
    """Random-incidence absorption and scattering per octave or third-octave band, and
    the share of sound that passes through.

    Coefficients come from, in order of precedence: values given here (8 per octave
    band or 24 per third-octave band), a coefficient file (``file``), or a built-in
    database entry (``preset``). The file may also be a material library of
    CATT-Acoustic or ODEON, with ``entry`` naming the material in it (see
    :mod:`acoustic_spaces.materials.libraries`). ``scattering`` may also name a
    scattering entry of the built-in database. Data at other frequencies is mapped onto
    third-octave bands when it has more than one value in some octave band, else onto
    octave bands; bands without data take the nearest band's value, and ``source`` says
    which ones.
    :meth:`coefficients` gives both lists in the bands a simulation runs in.

    ``transmission`` (a share from 0 to 1) or ``transmission_loss`` (the sound reduction
    index R in dB, the transmission being 10^(-R/10)) makes a surface let sound through:
    a partition between two rooms, a door, or with a transmission of 1 an opening. What
    passes through counts as absorbed on the side it comes from, so the absorption must
    be at least the transmission in every band; without absorption values, a preset or a
    file, the absorption is the transmission (an opening absorbs 1, a heavy wall no more
    than it lets through). :meth:`transmission_coefficients` gives it in the bands a
    simulation runs in.
    """

    preset: str | None = Field(default=None, description="built-in database entry")
    file: Path | None = Field(
        default=None,
        description="CSV of frequencies, absorption and scattering, or a CATT or ODEON library",
    )
    entry: str | int | None = Field(
        default=None, description="the material in a CATT or ODEON library file: name or number"
    )
    absorption: list[Coefficient] | None = None
    scattering: list[Coefficient] | Coefficient | str | None = None
    transmission: list[Coefficient] | Coefficient | None = Field(
        default=None, description="share of the sound that passes through (1: an opening)"
    )
    transmission_loss: list[float] | float | None = Field(
        default=None, description="sound reduction index R (dB): transmission 10^(-R/10)"
    )
    audience: bool = Field(
        default=False,
        description=(
            "audience seating: sound that reaches a listener low over it loses more, "
            "as it passes heads and seats (grazing incidence)"
        ),
    )
    impedance: Impedance | None = Field(
        default=None,
        description=(
            "a porous ground's impedance (a preset such as grass, or its flow "
            "resistivity), which gives the absorption and the ground's reflection"
        ),
    )
    source: str | None = Field(default=None, description="where the coefficients come from")

    @field_validator("impedance", mode="before")
    @classmethod
    def _impedance_preset(cls, value):
        return {"preset": value} if isinstance(value, str) else value

    @field_validator("absorption")
    @classmethod
    def _absorption_per_band(cls, value: list[float] | None) -> list[float] | None:
        return None if value is None else _per_band_or_third("absorption", value)

    @field_validator("scattering")
    @classmethod
    def _scattering_per_band(cls, value: list[float] | float | str | None):
        if isinstance(value, float | int):
            return [float(value)] * len(OCTAVE_BANDS_HZ)
        if isinstance(value, list):
            return _per_band_or_third("scattering", value)
        return value

    @field_validator("transmission")
    @classmethod
    def _transmission_per_band(cls, value: list[float] | float | None):
        if isinstance(value, float | int):
            return [float(value)] * len(OCTAVE_BANDS_HZ)
        return None if value is None else _per_band_or_third("transmission", value)

    @field_validator("transmission_loss")
    @classmethod
    def _loss_per_band(cls, value: list[float] | float | None):
        if value is None:
            return None
        values = [float(value)] * len(OCTAVE_BANDS_HZ) if isinstance(value, float | int) else value
        values = _per_band_or_third("transmission_loss", values)
        if min(values) < 0.0:
            raise ValueError("transmission_loss is in dB and cannot be negative")
        return values

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

        if self.impedance is not None:
            if self.preset is not None or self.file is not None:
                raise ValueError(
                    "a material with an impedance takes its absorption from it: give no "
                    "preset or file"
                )
            if self.transmission is not None or self.transmission_loss is not None:
                raise ValueError("a material with an impedance lets no sound through")
            # Third-octave bands, as the absorption changes fast with frequency; any
            # absorption values written with it are replaced.
            self.absorption = None
            absorption = band_absorption(self.impedance.admittance, len(THIRD_OCTAVE_BANDS_HZ))
        elif self.preset is not None:
            entry = _database_entry(self.preset, "absorption")
            absorption, extrapolated = _from_data(entry.frequencies, entry.coefficients)
            origin = _describe(entry.source, extrapolated)
        elif self.file is not None:
            if not self.file.exists():
                raise ValueError(f"coefficient file not found: {self.file}")
            if self.entry is not None:
                found = find_material(self.file, self.entry)
                freqs, alpha, scatter = found.frequencies, found.absorption, found.scattering
                where = f"{found.label!r} of {self.file.name}"
            else:
                freqs, alpha, scatter = _read_coefficients(self.file)
                where = f"file {self.file.name}"
            absorption, extrapolated = _from_data(freqs, alpha)
            origin = _describe(where, extrapolated)
            if scatter is not None:
                scattering, _ = _from_data(freqs, scatter)
        if self.entry is not None and self.file is None:
            raise ValueError("entry names a material in a library file: give the file too")

        if isinstance(self.scattering, str):
            entry = _database_entry(self.scattering, "scattering")
            scattering, extrapolated = _from_data(entry.frequencies, entry.coefficients)
            scattering_origin = "scattering: " + _describe(entry.source, extrapolated)
        elif self.scattering is not None:
            scattering = self.scattering

        if self.transmission is not None and self.transmission_loss is not None:
            raise ValueError("give either transmission or transmission_loss, not both")
        passing = self._transmission()
        if self.absorption is None:
            if absorption is None and passing is not None:
                absorption = passing
            if absorption is None:
                raise ValueError("give absorption values, a preset or a file")
            self.absorption = _clip(absorption)
        else:
            origin = None
        self.scattering = _clip(scattering or [DEFAULT_SCATTERING] * len(OCTAVE_BANDS_HZ))
        if passing is not None:
            thirds = len(THIRD_OCTAVE_BANDS_HZ) in (len(passing), len(self.absorption))
            fine = "third_octave" if thirds else "octave"
            centres = THIRD_OCTAVE_BANDS_HZ if thirds else OCTAVE_BANDS_HZ
            through, alpha = convert(passing, fine), convert(self.absorption, fine)
            for f, t, a in zip(centres, through, alpha, strict=True):
                if t > a + 1e-9:
                    raise ValueError(
                        f"the transmission at {f:g} Hz ({t:.3g}) is more than the absorption "
                        f"({a:.3g}): the sound a surface lets through counts as absorbed on "
                        "the side it comes from, so the absorption must be at least the "
                        "transmission"
                    )
        if self.source is None:
            origins = [o for o in (origin, scattering_origin) if o]
            self.source = "; ".join(origins) or None
        return self

    def _transmission(self) -> list[float] | None:
        if self.transmission_loss is not None:
            return [10.0 ** (-r / 10.0) for r in self.transmission_loss]
        return None if self.transmission is None else list(self.transmission)

    def coefficients(self, bands: Resolution = "octave") -> tuple[list[float], list[float]]:
        """Absorption and scattering in octave or third-octave ``bands``: an octave
        value holds for its three third octaves, and an octave takes the mean of its
        three."""
        return convert(self.absorption, bands), convert(self.scattering, bands)

    def transmission_coefficients(self, bands: Resolution = "octave") -> list[float] | None:
        """The share of sound that passes through, in octave or third-octave ``bands``
        (converted as :meth:`coefficients` does), or None for a surface that lets
        nothing through."""
        passing = self._transmission()
        return None if passing is None else convert(passing, bands)


def _read_coefficients(path: Path):
    """A coefficient CSV file, or the one material of a library that has just one."""
    try:
        return read_coefficients_csv(path)
    except ValueError as exc:
        try:
            materials = read_library(path)
        except ValueError:
            raise exc from None
    if len(materials) != 1:
        names = ", ".join(m.label for m in materials[:12])
        more = f" and {len(materials) - 12} more" if len(materials) > 12 else ""
        raise ValueError(
            f"{path.name} is a library of {len(materials)} materials: name one with entry "
            f"({names}{more})"
        )
    return materials[0].frequencies, materials[0].absorption, materials[0].scattering


def _database_entry(name: str, kind: Literal["absorption", "scattering"]):
    try:
        return builtin_database().get(name, kind=kind)
    except KeyError as exc:
        raise ValueError(exc.args[0]) from None


def _clip(values: list[float]) -> list[float]:
    return [min(max(float(v), 0.0), 1.0) for v in values]


#: The most point sources a line or surface source may be made of.
MAX_ELEMENTS = 400

PatternName = Literal[
    "omni", "subcardioid", "cardioid", "supercardioid", "hypercardioid", "figure8"
]


#: The most elements a coherent array may have.
MAX_ARRAY_ELEMENTS = 256


class ArrayElement(_Model):
    """One loudspeaker or driver of a coherent array, in the array's (the source's)
    frame: x forward, y left, z up."""

    position: Vector3 = Field(description="where it sits (m), from the source's position")
    orientation: Orientation = Field(
        default_factory=Orientation, description="its aim, turned from the source's"
    )
    gain_db: float = Field(default=0.0, ge=-100.0, le=60.0, description="gain of its feed")
    delay_ms: float = Field(default=0.0, ge=0.0, le=1000.0, description="delay of its feed")
    invert: bool = Field(default=False, description="its polarity inverted")


class ArraySpec(_Model):
    """A coherent array: elements fed the same signal, each with its own gain, delay
    and polarity, whose sound adds up with its phase. Either ``elements`` one by one, or
    a straight column of ``count`` elements ``spacing_m`` apart along the up axis,
    centred on the source's position."""

    element: PatternName | Literal["lambert"] | DirectivitySpec = Field(
        default="omni", description="each element's own directivity, along its aim"
    )
    elements: list[ArrayElement] | None = Field(default=None, description="the elements")
    count: int | None = Field(
        default=None, ge=2, le=MAX_ARRAY_ELEMENTS, description="a column of this many"
    )
    spacing_m: PositiveFloat | None = Field(default=None, description="the column's spacing")
    steer_deg: float = Field(
        default=0.0,
        ge=-90.0,
        le=90.0,
        description=(
            "delays that tilt the beam up (positive) or down from the forward axis, "
            "on top of the elements' own"
        ),
    )

    @model_validator(mode="after")
    def _layout(self) -> ArraySpec:
        column = self.count is not None or self.spacing_m is not None
        if column == (self.elements is not None):
            raise ValueError("give an array's elements, or a count and spacing_m, not both")
        if column and (self.count is None or self.spacing_m is None):
            raise ValueError("a column of elements needs both count and spacing_m")
        if self.elements is not None and not 1 <= len(self.elements) <= MAX_ARRAY_ELEMENTS:
            raise ValueError(f"an array has 1 to {MAX_ARRAY_ELEMENTS} elements")
        if isinstance(self.element, DirectivitySpec) and self.element.array is not None:
            raise ValueError("an array's elements cannot be arrays themselves")
        return self

    def layout(self) -> list[ArrayElement]:
        """The elements, a column's laid out."""
        if self.elements is not None:
            return list(self.elements)
        offsets = (np.arange(self.count) - 0.5 * (self.count - 1)) * self.spacing_m
        return [ArrayElement(position=(0.0, 0.0, float(-z))) for z in offsets]


class DirectivitySpec(_Model):
    """A source's directivity per octave band: first-order patterns (a name, or the
    ``a`` of ``a + (1 - a) cos(theta)``, for each band), a measured balloon from a
    SOFA file or a CSV table, or the far-field pattern of a coherent array, given here
    or in a YAML file (see :mod:`acoustic_spaces.directivity`)."""

    pattern: list[PatternName | Coefficient] | None = Field(
        default=None,
        description="one pattern name or value of a per octave band (or third-octave band)",
    )
    file: Path | None = Field(
        default=None,
        description=(
            "measured data: a SOFA file (FreeFieldDirectivityTF) or CSV; or an array "
            "described in a YAML file (.yaml or .yml)"
        ),
    )
    array: ArraySpec | None = Field(default=None, description="a coherent array")
    measurement: int | None = Field(
        default=None,
        ge=0,
        description="SOFA: the measurement to use (a note, say); None averages them all",
    )

    @field_validator("pattern")
    @classmethod
    def _check_pattern(cls, value: list | None) -> list | None:
        return None if value is None else _per_band_or_third("pattern", value)

    @field_validator("file")
    @classmethod
    def _resolve_file(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        return None if value is None else _resolve_path(value, info)

    @model_validator(mode="after")
    def _one_kind(self) -> DirectivitySpec:
        if sum(x is not None for x in (self.pattern, self.file, self.array)) != 1:
            raise ValueError("give one of a directivity pattern per band, a file or an array")
        if self.measurement is not None and (
            self.file is None or self.file.suffix.lower() != ".sofa"
        ):
            raise ValueError("measurement is for SOFA directivity files")
        return self


ArraySpec.model_rebuild()


class Walk(_Model):
    """A path a listener or a point source walks along, for a walkthrough
    (:mod:`acoustic_spaces.walkthrough`): from its position through ``to``."""

    to: list[Vector3] = Field(min_length=1, description="the points it walks through, in order")
    speed_m_s: PositiveFloat = Field(default=1.2, description="walking speed (m/s)")
    face_forward: bool = Field(
        default=True,
        description="turn to face the way it walks (yaw); false keeps its orientation",
    )
    turn_deg_s: PositiveFloat = Field(
        default=90.0,
        description="how fast it turns through a corner facing forward (degrees per second)",
    )


class Source(_Model):
    name: str
    position: Vector3
    orientation: Orientation = Field(default_factory=Orientation)
    directivity: PatternName | Literal["lambert"] | DirectivitySpec = Field(
        default="omni",
        description=(
            "a pattern for every band (lambert: a vibrating surface's, nothing behind), "
            "or per-band patterns or measured data"
        ),
    )
    line_m: PositiveFloat | None = Field(
        default=None,
        description=(
            "a line source this long (m), along the source's left axis and centred on its "
            "position: uncorrelated point sources at most spacing_m apart share its power"
        ),
    )
    surface_m: tuple[PositiveFloat, PositiveFloat] | None = Field(
        default=None,
        description=(
            "a surface source this wide and high (m), across the source's left and up axes "
            "and centred on its position, facing forward: uncorrelated point sources at "
            "most spacing_m apart share its power"
        ),
    )
    spacing_m: PositiveFloat = Field(
        default=0.5, description="the most the point sources of a line or surface are apart (m)"
    )
    level_db: float | list[float] = Field(
        default=94.0,
        description=(
            "sound power level in dB re 1 pW: in all, spread over the bands as "
            "spectrum says, or per band (8 octave or 24 third-octave values)"
        ),
    )
    spectrum: Literal["flat", "speech_male", "speech_female"] = Field(
        default="flat",
        description=(
            "how a level given in all spreads over the bands: evenly over the octave "
            "bands, or as male or female speech (IEC 60268-16), which the STI then "
            "weights for"
        ),
    )
    gain_db: float = Field(
        default=0.0,
        ge=-100.0,
        le=60.0,
        description="gain of the source's feed (dB): on its level and its impulse responses",
    )
    eq_db: list[Annotated[float, Field(ge=-100.0, le=60.0)]] | None = Field(
        default=None,
        description=(
            "equalizer gain per band (dB, 8 octave or 24 third-octave values), on the "
            "level and the impulse responses"
        ),
    )
    delay_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=10_000.0,
        description=(
            "how long after emission starts the source plays (ms): its arrivals, impulse "
            "responses and its part in combined coverage maps come that much later"
        ),
    )
    walk: Walk | None = Field(
        default=None, description="a path to walk, for a walkthrough (point sources only)"
    )

    @field_validator("level_db", mode="before")
    @classmethod
    def _finite_level(cls, value):
        # One error for a NaN level, not one for each kind of value it could have been.
        if isinstance(value, float) and not math.isfinite(value):
            raise PydanticCustomError("finite_number", "Input should be a finite number")
        return value

    @field_validator("level_db")
    @classmethod
    def _level_per_band(cls, value: float | list[float]) -> float | list[float]:
        return _per_band_or_third("level_db", value) if isinstance(value, list) else value

    @field_validator("eq_db")
    @classmethod
    def _eq_per_band(cls, value: list[float] | None) -> list[float] | None:
        return None if value is None else _per_band_or_third("eq_db", value)

    @model_validator(mode="after")
    def _spectrum_of_a_total(self) -> Source:
        if isinstance(self.level_db, list) and self.spectrum != "flat":
            raise ValueError("spectrum shapes a level given in all; a level per band has its own")
        if self.line_m is not None and self.surface_m is not None:
            raise ValueError("a source is a line or a surface, not both")
        if self.walk is not None and self.extended:
            raise ValueError("only a point source walks, not a line or a surface")
        if len(self._grid()[0]) * len(self._grid()[1]) > MAX_ELEMENTS:
            raise ValueError(f"more than {MAX_ELEMENTS} point sources; make spacing_m larger")
        return self

    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Offsets (m) of the point sources along the left and the up axis."""

        def offsets(size: float | None) -> np.ndarray:
            if size is None:
                return np.zeros(1)
            n = max(1, math.ceil(size / self.spacing_m - 1e-9))
            return ((np.arange(n) + 0.5) / n - 0.5) * size

        if self.surface_m is not None:
            return offsets(self.surface_m[0]), offsets(self.surface_m[1])
        return offsets(self.line_m), np.zeros(1)

    @property
    def extended(self) -> bool:
        """Whether the source is a line or a surface rather than a point."""
        return self.line_m is not None or self.surface_m is not None

    def elements(self) -> np.ndarray:
        """Positions (n, 3) of the point sources that make up the source: its position
        alone for a point source; along a line or over a surface evenly spaced, each
        in the middle of an equal share of it."""
        left, up = self._grid()
        axes = np.array(self.orientation.rotation())
        offsets = left[:, None, None] * axes[:, 1] + up[None, :, None] * axes[:, 2]
        return np.asarray(self.position, dtype=float) + offsets.reshape(-1, 3)

    def outline(self) -> np.ndarray:
        """Corners (n, 3) of the source's extent: a line's two ends, a surface's four
        corners in turn, or a point source's position alone."""
        axes = np.array(self.orientation.rotation())
        centre = np.asarray(self.position, dtype=float)
        if self.line_m is not None:
            half = 0.5 * self.line_m * axes[:, 1]
            return np.array([centre - half, centre + half])
        if self.surface_m is not None:
            left, up = 0.5 * self.surface_m[0] * axes[:, 1], 0.5 * self.surface_m[1] * axes[:, 2]
            return centre + np.array([-left - up, left - up, left + up, -left + up])
        return centre[None, :]

    @property
    def talker(self) -> Literal["male", "female"]:
        """The speech the STI weights for: female for ``speech_female``, else male."""
        return "female" if self.spectrum == "speech_female" else "male"

    def feed_db(self, bands: Resolution = "octave") -> np.ndarray:
        """The gain of the feed per band (dB): ``gain_db`` and ``eq_db`` together."""
        eq = np.zeros(len(OCTAVE_BANDS_HZ)) if self.eq_db is None else self.eq_db
        return self.gain_db + gain_levels(eq, bands)

    @property
    def sound_power_db(self) -> float:
        """Sound power level in all (dB re 1 pW) that ``level_db`` gives, summed over its
        bands when given per band, without the feed's gain."""
        if isinstance(self.level_db, list):
            return 10.0 * math.log10(float(np.sum(10.0 ** (np.asarray(self.level_db) / 10.0))))
        return float(self.level_db)

    def power_db(self, bands: Resolution = "octave") -> np.ndarray:
        """Sound power level per band (dB re 1 pW), the feed's gain included: an octave's
        power is split evenly between its third octaves."""
        if isinstance(self.level_db, list):
            levels = np.asarray(self.level_db, dtype=float)
        else:
            if self.spectrum == "flat":
                shape = np.zeros(len(OCTAVE_BANDS_HZ))
            else:
                shape = np.array(SPEECH_SPECTRUM_DB[self.talker])
            total = 10.0 * math.log10(float(np.sum(10.0 ** (shape / 10.0))))
            levels = self.level_db + shape - total
        return power_levels(levels, bands) + self.feed_db(bands)


class Listener(_Model):
    name: str
    position: Vector3
    orientation: Orientation = Field(default_factory=Orientation)
    walk: Walk | None = Field(default=None, description="a path to walk, for a walkthrough")


class Air(_Model):
    temperature_c: float = Field(default=20.0, ge=-20.0, le=50.0)
    humidity_percent: float = Field(default=50.0, ge=0.0, le=100.0)
    pressure_kpa: float = Field(default=101.325, gt=0.0)


class Wind(_Model):
    """The wind over the ground: its speed at a height, the way it blows, and how the
    speed grows with height."""

    speed_m_s: float = Field(default=0.0, ge=0.0, le=40.0, description="speed at height_m (m/s)")
    direction_deg: float = Field(
        default=0.0,
        description=(
            "the way it blows to, in degrees counterclockwise from +x seen from above "
            "(0: along +x, 90: along +y)"
        ),
    )
    height_m: PositiveFloat = Field(default=10.0, description="the height the speed is at (m)")
    profile: Literal["log", "linear", "constant"] = Field(
        default="log",
        description=(
            "log: speed ln(1 + z/roughness) over the ground's friction, the usual profile; "
            "linear: in proportion to the height; constant: the same at every height"
        ),
    )


#: The atmosphere presets: the fields each sets, indicative values (the temperatures
#: as the ground's plus gradient z + log ln(1 + z / roughness)).
ATMOSPHERE_PRESETS: dict[str, dict] = {
    # No wind, the same temperature everywhere: sound goes in straight lines.
    "neutral": {},
    # A moderate wind, 5 m/s at 10 m over grass: sound bends down downwind and up
    # upwind.
    "windy": {"wind": {"speed_m_s": 5.0}},
    # Sunshine warming the ground: the air cools with height and sound bends up, with a
    # shadow zone a few hundred metres away.
    "sunny_day": {"temperature_log_k": -0.5, "temperature_gradient_k_m": -0.0098},
    # A clear night cooling the ground: the air warms with height (an inversion) and
    # sound bends down.
    "clear_night": {"temperature_log_k": 0.5, "temperature_gradient_k_m": 0.0},
}


#: The turbulence presets: the structure parameters of the temperature (C_T², K² m^-2/3)
#: and of the wind (C_v², m^4/3 s^-2) near the ground, indicative values.
TURBULENCE_PRESETS: dict[str, dict] = {
    # Overcast, a light wind: the air hardly stirs.
    "weak": {"temperature_structure": 0.001, "wind_structure": 0.03},
    # A breeze and some sun, the usual day.
    "moderate": {"temperature_structure": 0.008, "wind_structure": 0.12},
    # A strong wind or a hot sun: gusty air.
    "strong": {"temperature_structure": 0.08, "wind_structure": 0.5},
}


class Turbulence(_Model):
    """The eddies of the wind and of the temperature, which make sound outdoors flicker:
    the paths of one sound that take different ways stop adding in step (the ground dip
    fills in), and sound is scattered into shadow zones. Their strength is given by the
    structure parameters of the temperature and of the wind, C_T² and C_v², with a von
    Kármán spectrum whose largest eddies are ``outer_scale_m`` across."""

    preset: str | None = Field(
        default=None,
        description="a strength to start from: " + ", ".join(TURBULENCE_PRESETS),
    )
    temperature_structure: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description=(
            "the temperature's structure parameter C_T² (K² m^-2/3): about 0.001 to 1 near "
            "the ground"
        ),
    )
    wind_structure: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description=(
            "the wind's structure parameter C_v² (m^4/3 s^-2): about 0.01 to 1 near the ground"
        ),
    )
    outer_scale_m: PositiveFloat = Field(
        default=10.0,
        le=1000.0,
        description="the size of the largest eddies (m): some metres near the ground",
    )

    @model_validator(mode="before")
    @classmethod
    def _from_preset(cls, data):
        if isinstance(data, str):
            data = {"preset": data}
        if isinstance(data, dict) and data.get("preset") is not None:
            preset = data["preset"]
            if preset not in TURBULENCE_PRESETS:
                close = difflib.get_close_matches(str(preset), list(TURBULENCE_PRESETS), n=1)
                hint = f"; did you mean {close[0]}?" if close else ""
                raise ValueError(
                    f"no turbulence preset named {preset!r}{hint} (the presets are "
                    f"{', '.join(TURBULENCE_PRESETS)})"
                )
            data = {**TURBULENCE_PRESETS[preset], **data}
        return data

    @property
    def calm(self) -> bool:
        """Whether there are no eddies at all."""
        return self.temperature_structure == 0.0 and self.wind_structure == 0.0


class Atmosphere(_Model):
    """How the speed of sound changes with height outdoors, which bends sound: the
    temperature (warmer air carries sound faster) and the wind (sound goes faster with
    it). Both vary with height only. The scene's air temperature is the temperature at
    the ground. Heights are in metres above the ground. Eddies of both (``turbulence``)
    make the sound flicker."""

    preset: str | None = Field(
        default=None,
        description="a weather to start from: " + ", ".join(ATMOSPHERE_PRESETS),
    )
    wind: Wind | None = Field(default=None, description="None: no wind")
    temperature_gradient_k_m: float = Field(
        default=0.0,
        ge=-0.2,
        le=0.2,
        description="how the temperature changes per metre of height (K/m)",
    )
    temperature_log_k: float = Field(
        default=0.0,
        ge=-5.0,
        le=5.0,
        description=(
            "a change of temperature fastest near the ground, in K times ln(1 + z / "
            "roughness_m): negative on a sunny day, positive on a clear night"
        ),
    )
    roughness_m: PositiveFloat = Field(
        default=0.05,
        le=2.0,
        description=(
            "the ground's roughness length (m), for the log profiles: 0.01 to 0.1 over grass"
        ),
    )
    table: list[tuple[float, float, float]] | None = Field(
        default=None,
        description=(
            "measured profile in place of the formulas: [height above the ground (m), "
            "temperature (°C), wind speed (m/s)] rows, by height; between them linear, "
            "above the top as at the top"
        ),
    )
    turbulence: Turbulence | None = Field(
        default=None,
        description="the eddies of the wind and of the temperature; None: none",
    )

    @model_validator(mode="before")
    @classmethod
    def _from_preset(cls, data):
        if isinstance(data, dict) and data.get("preset") is not None:
            preset = data["preset"]
            if preset not in ATMOSPHERE_PRESETS:
                close = difflib.get_close_matches(str(preset), list(ATMOSPHERE_PRESETS), n=1)
                hint = f"; did you mean {close[0]}?" if close else ""
                raise ValueError(
                    f"no atmosphere preset named {preset!r}{hint} (the presets are "
                    f"{', '.join(ATMOSPHERE_PRESETS)})"
                )
            given = dict(data)
            data = {**ATMOSPHERE_PRESETS[preset], **given}
            if isinstance(given.get("wind"), dict) and "wind" in ATMOSPHERE_PRESETS[preset]:
                data["wind"] = {**ATMOSPHERE_PRESETS[preset]["wind"], **given["wind"]}
        return data

    @model_validator(mode="after")
    def _check_table(self) -> Atmosphere:
        if self.table is None:
            return self
        # Zeros are what a written-out scene has (Scene.to_data writes every field).
        formulas = self.temperature_gradient_k_m != 0.0 or self.temperature_log_k != 0.0
        if self.preset is None and formulas:
            raise ValueError(
                "give the atmosphere a table or the temperature's gradient and log terms, not both"
            )
        heights = [row[0] for row in self.table]
        if len(heights) < 2 or heights[0] < 0.0 or np.any(np.diff(heights) <= 0.0):
            raise ValueError(
                "an atmosphere table has two rows or more, by height from the ground up"
            )
        if any(row[2] < 0.0 for row in self.table):
            raise ValueError("an atmosphere table's wind speeds are not negative")
        return self

    def to_wind(self) -> np.ndarray:
        """(2,) the unit vector the wind blows along."""
        angle = math.radians(self.wind.direction_deg if self.wind is not None else 0.0)
        return np.array([math.cos(angle), math.sin(angle)])

    def temperature(self, z, ground_c: float) -> np.ndarray:
        """The temperature (°C) at heights ``z`` (m) above the ground, the ground's
        being ``ground_c`` (a table gives its own)."""
        z = np.maximum(np.asarray(z, dtype=float), 0.0)
        if self.table is not None:
            rows = np.asarray(self.table, dtype=float)
            return np.interp(z, rows[:, 0], rows[:, 1])
        return (
            ground_c
            + self.temperature_gradient_k_m * z
            + self.temperature_log_k * np.log1p(z / self.roughness_m)
        )

    def wind_speed(self, z) -> np.ndarray:
        """The wind's speed (m/s) at heights ``z`` (m) above the ground."""
        z = np.maximum(np.asarray(z, dtype=float), 0.0)
        if self.table is not None:
            rows = np.asarray(self.table, dtype=float)
            return np.interp(z, rows[:, 0], rows[:, 2])
        wind = self.wind
        if wind is None or wind.speed_m_s == 0.0:
            return np.zeros_like(z)
        if wind.profile == "constant":
            return np.full_like(z, wind.speed_m_s)
        if wind.profile == "linear":
            return wind.speed_m_s * z / wind.height_m
        return (
            wind.speed_m_s
            * np.log1p(z / self.roughness_m)
            / math.log1p(wind.height_m / self.roughness_m)
        )

    def sound_speed(self, z, ground_c: float, towards=None) -> np.ndarray:
        """The speed of sound (m/s) at heights ``z`` (m) above the ground: in still air,
        or the effective speed along the horizontal direction ``towards`` (2,), the
        wind's component along it added."""
        z = np.asarray(z, dtype=float)
        speed = SOUND_SPEED_0C * np.sqrt(1.0 + self.temperature(z, ground_c) / 273.15)
        if towards is not None:
            along = float(np.dot(self.to_wind(), towards))
            speed = speed + along * self.wind_speed(z)
        return speed

    @property
    def still(self) -> bool:
        """Whether sound goes in straight lines: no wind and one temperature."""
        if self.table is not None:
            rows = np.asarray(self.table, dtype=float)
            return bool(np.all(rows[:, 1] == rows[0, 1]) and np.all(rows[:, 2] == 0.0))
        calm = self.wind is None or self.wind.speed_m_s == 0.0
        return calm and self.temperature_gradient_k_m == 0.0 and self.temperature_log_k == 0.0


#: The highest ``simulation.early.coherent_max_band_hz`` in a room, and the default
#: outdoors.
INDOOR_COHERENT_MAX_BAND_HZ = 500
OUTDOOR_COHERENT_MAX_BAND_HZ = 8000


class EarlyStage(_Model):
    """Early reflections: exact specular paths with delay and arrival direction, and
    the sound diffracted at the edges of the room."""

    method: Literal["ray_tracing", "image_sources"] = "ray_tracing"
    max_order: int = Field(default=3, ge=0, le=20)
    rays: int = Field(default=20_000, ge=1, description="ray tracing only")
    diffraction: Literal["none", "direct", "first_order", "second_order"] = Field(
        default="first_order",
        description=(
            "sound bending round edges (pillars, balcony fronts, screens): none, of the "
            "direct sound only, of the direct sound and first-order reflections, or also "
            "of the direct sound over two edges in turn"
        ),
    )
    diffraction_reflections: Literal["one_side", "both_sides"] = Field(
        default="one_side",
        description=(
            "with first_order or second_order diffraction, where a path over one edge may "
            "reflect: on one side of the edge, before it or after it (off the ground on "
            "the way to a barrier, or off a façade behind it), or on both sides, before "
            "and after (off the ground on the way to a barrier and again behind it: the "
            "four paths over a barrier on the ground)"
        ),
    )
    coherent_max_band_hz: Literal[63, 125, 250, 500, 1000, 2000, 4000, 8000] | None = Field(
        default=125,
        description=(
            "the highest octave band in which the early paths add in pressure, with the "
            "phase of every reflection and of the sound bent round edges, as they do at "
            "low frequencies, rather than in energy; null for none. Above 500 Hz only "
            "outdoors, where every band is the default. The wave solver's bands take "
            "their place when it runs up to this band or higher"
        ),
    )


class LateStage(_Model):
    """Late reverberation: stochastic energy decay per band, time slot and direction."""

    method: Literal["stochastic_ray_tracing", "statistical"] = "stochastic_ray_tracing"
    rays: int = Field(
        default=50_000,
        ge=1,
        description=(
            "stochastic ray tracing only: rays per band (vector-based scattering "
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
    device: Literal["cpu", "gpu", "both", "auto"] | None = Field(
        default=None,
        description=(
            "stochastic ray tracing only: where it traces the rays: cpu, gpu (through "
            "WebGPU: Vulkan, Metal or Direct3D 12, with the gpu extra), both (the two share "
            "the rays; with a GPU, results can vary slightly from run to run) or auto (the "
            "fastest of those on this computer for the response length, as acoustic-spaces tune "
            "or a short trial found; the CPU when unsure). Not set: this computer's "
            "default, cpu unless acoustic-spaces tune made it auto. Sound through "
            "partitions and material calibration always run on the CPU"
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
    scheme: Literal["auto", "isotropic", "standard"] = Field(
        default="auto",
        description=(
            "the grid's stencil: isotropic (19 points; waves 0.67 % slow at 8 points per "
            "wavelength, alike in every direction) or standard (7 points; up to 1.7 % slow, "
            "along the axes); auto takes the isotropic one but round an open model, where "
            "only the standard one has the matched layer"
        ),
    )
    device: Literal["cpu", "gpu", "auto"] | None = Field(
        default=None,
        description=(
            "where it steps the grid: cpu, gpu (through WebGPU: Vulkan, Metal or Direct3D 12, "
            "with the gpu extra) or auto (a hardware GPU when wgpu finds one, else the CPU; "
            "once acoustic-spaces tune timed both, whichever is faster for the run's "
            "length). Not set: this computer's default, cpu unless acoustic-spaces tune "
            "made it auto"
        ),
    )


class Performance(_Model):
    """How a run uses the computer. Neither setting changes the results, beyond a ray
    that meets two faces at exactly the same point (a shared edge) taking the other one
    with the other tree."""

    threads: int = Field(
        default=0,
        ge=0,
        description=(
            "worker threads for ray tracing and the wave solver on the CPU; 0 uses every "
            "core. Fewer leave the computer free for other work"
        ),
    )
    ray_tree: Literal["sah", "median"] = Field(
        default="sah",
        description=(
            "how the tree that finds the surface a ray meets is split: sah (by the surface "
            "area heuristic; several times faster in detailed models) or median (at the "
            "middle face, as before; for comparison)"
        ),
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
    bands: Resolution = Field(
        default="octave",
        description=(
            "the bands it runs in: octave (8, 63 Hz to 8 kHz) or third_octave (24, 50 Hz "
            "to 10 kHz; with rays per band, about three times the work). Room parameters "
            "are given in octave bands either way"
        ),
    )
    early: EarlyStage = Field(default_factory=EarlyStage)
    late: LateStage = Field(default_factory=LateStage)
    scattering: Scattering = Field(default_factory=Scattering)
    wave: WaveStage = Field(default_factory=WaveStage)
    performance: Performance = Field(default_factory=Performance)
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
    seed: int | None = Field(default=None, ge=0, description="random seed, for reproducible runs")
    insertion_loss_of: list[str] | None = Field(
        default=None,
        min_length=1,
        description=(
            "outdoors: buildings, barriers or vegetation, by name, whose insertion loss the "
            "simulate and map commands give: the scene is also run without them, and the "
            "level without them less the level with them is the insertion loss"
        ),
    )


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
    atmosphere: Atmosphere | None = Field(
        default=None,
        description="outdoors: wind and temperature changing with height, which bend sound",
    )
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
        if self._default_sky() and SKY not in self.materials:
            self.materials[SKY] = sky_material()
        early = self.simulation.early
        if self.geometry.ground is not None:
            # Over the ground the direct sound and its reflection interfere in every
            # band (the ground dip), and the paths are few.
            if "coherent_max_band_hz" not in early.model_fields_set:
                early.coherent_max_band_hz = OUTDOOR_COHERENT_MAX_BAND_HZ
        elif self.atmosphere is not None:
            raise ValueError(
                "atmosphere is for outdoor scenes (geometry.ground): in a room the air is "
                "one temperature and still"
            )
        elif (early.coherent_max_band_hz or 0) > INDOOR_COHERENT_MAX_BAND_HZ:
            raise ValueError(
                f"simulation.early.coherent_max_band_hz goes up to "
                f"{INDOOR_COHERENT_MAX_BAND_HZ} Hz in a room, whose early paths are many; "
                "higher bands are for outdoor scenes"
            )
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
        removed = self.simulation.insertion_loss_of
        if removed is not None:
            if self.geometry.ground is None:
                raise ValueError(
                    "simulation.insertion_loss_of is for outdoor scenes (geometry.ground): "
                    "it names their buildings, barriers or vegetation"
                )
            parts = self.geometry.part_names()
            unknown = [name for name in removed if name not in parts]
            if unknown:
                known = ", ".join(parts) or "none"
                raise ValueError(
                    f"simulation.insertion_loss_of names {unknown[0]!r}, which is no "
                    f"building, barrier or vegetation of the scene (they are: {known})"
                )
            twice = sorted({name for name in removed if removed.count(name) > 1})
            if twice:
                raise ValueError(f"simulation.insertion_loss_of names {twice[0]!r} twice")
        return self

    def without(self, names: Sequence[str]) -> Scene:
        """This outdoor scene without the buildings, barriers and vegetation of these
        names (see :meth:`Geometry.part_names`), and without ``insertion_loss_of``."""
        parts = self.geometry.part_names()
        unknown = [name for name in names if name not in parts]
        if unknown:
            raise ValueError(f"no building, barrier or vegetation is named {unknown[0]!r}")
        data = self.to_data()
        geometry = data["geometry"]
        for kind, kept in (
            ("buildings", self.geometry.building_names()),
            ("barriers", self.geometry.barrier_names()),
            ("vegetation", self.geometry.vegetation_names()),
        ):
            if kind not in geometry:
                continue
            geometry[kind] = [
                item for item, name in zip(geometry[kind], kept, strict=True) if name not in names
            ]
            if not geometry[kind]:
                del geometry[kind]
        data["simulation"].pop("insertion_loss_of", None)
        return Scene.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scene:
        """Load a scene file; relative paths in it are resolved against its folder."""
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            try:
                data = yaml.safe_load(fh)
            except yaml.YAMLError as exc:
                raise ValueError(f"{path.name}: not valid YAML: {exc}") from None
        return cls.model_validate(data, context={"base_dir": path.resolve().parent})

    def to_data(self) -> dict:
        """The scene as plain YAML data, dicts and lists. Settings left unset are left out,
        apart from those whose null is not their default."""
        data = self.model_dump(mode="json", exclude_none=True)
        for name, material in self.materials.items():
            if material.impedance is not None:  # the impedance gives the absorption
                data["materials"][name].pop("absorption", None)
        if self.simulation.early.coherent_max_band_hz is None:
            data["simulation"]["early"]["coherent_max_band_hz"] = None
        # The open sky's material is added when the scene is read: left out, as written.
        if self._default_sky() and self.materials.get(SKY) == sky_material():
            del data["materials"][SKY]
        return data

    def _default_sky(self) -> bool:
        """Whether the sky of an outdoor scene takes the built-in open sky: the scene
        assigns the sky no material of its own."""
        return self.geometry.ground is not None and not any(
            key.casefold() == SKY for key in self.assignments
        )

    def to_yaml(self, path: str | Path) -> None:
        """Write the scene file. Paths to files under its folder are written relative to
        it, so the folder can move; others stay absolute."""
        path = Path(path)
        data = self.to_data()
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
            if not isinstance(directivity, dict):
                continue
            # An inline array's element may have a file of its own.
            element = directivity.get("array", {}).get("element")
            for spec in (directivity, element):
                if isinstance(spec, dict) and "file" in spec:
                    spec["file"] = _portable(spec["file"], base)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)


def _portable(file: str, base: Path) -> str:
    """``file`` relative to ``base`` when it lies inside it, with forward slashes."""
    try:
        return Path(file).resolve().relative_to(base).as_posix()
    except ValueError:
        return file
