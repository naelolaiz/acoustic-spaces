"""The uncertainty of the results from the materials' uncertain absorption and the rays'
random error.

A Monte Carlo propagation of distributions (GUM Supplement 1, JCGM 101:2008): each draw
takes every material's absorption from its distribution
(:class:`~acoustic_spaces.scene.model.AbsorptionUncertainty`), one deviate per material
that its bands share, and re-weights the traced responses for it
(:mod:`~acoustic_spaces.engine.reweight`), without tracing again. To each draw's
parameters it adds the rays' random error, as an input of its own: the standard error
of the late rays' batches (:meth:`~acoustic_spaces.engine.response.Response.standard_errors`,
a jackknife of n batches) times a deviate of Student's t with n - 1 degrees of freedom,
as JCGM 101 6.4.9 takes a quantity known from n observations, whose standard uncertainty
is the standard error times sqrt((n - 1) / (n - 3)). The two are independent, each
parameter's spread in a band being taken on its own. The parameters of the draws give
each one's mean, combined standard uncertainty (the draws' standard deviation, JCGM 101
7.6) and probabilistically symmetric coverage interval (JCGM 101 7.7), and the budget:
each input's standard uncertainty alone.

Not counted: the scattering's uncertainty, nor the method's (docs/validation.md), nor
the rays' error with fewer than ``MIN_BATCHES`` batches (Student's t has no variance
then) or in the parameters it is not found for (those from the late sound's directions).
Of the method's errors, those of the paths over edges are listed where a response has
such paths (:data:`EDGE_ERRORS_DB`), each with the largest difference of the early level
from the exact solution that the validation finds in each band, but not drawn: they are
measured for whole levels in a few geometries, not for each path.
A draw of the absorption below 0 is taken as 0 and one above ``MAX_ABSORPTION`` as that,
so near those ends the spread is a little less than the uncertainty gives. The
re-weighting is exact for the early paths and to second order for the late part
(:mod:`~acoustic_spaces.engine.reweight`): for uncertainties of a few hundredths, as
measured absorption has, that error is far below the spread.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

from acoustic_spaces.engine.reweight import log_ratio
from acoustic_spaces.parameters import RoomParameters

#: The highest absorption a draw takes: re-weighting needs some sound reflected.
MAX_ABSORPTION = 0.99
#: The draws :func:`material_spread` takes unless told otherwise.
DEFAULT_DRAWS = 400
#: The fewest batches of late rays whose error counts: Student's t with n - 1 degrees of
#: freedom has a variance from 3 up.
MIN_BATCHES = 4
#: The parts of a :class:`Spread`.
PARTS = ("mean", "standard", "low", "high", "absorption", "rays")
_NOT = math.nan
#: The method's errors where a response has paths over edges, which the spread lists but
#: does not draw: for each, the largest difference (dB) of the level from the exact
#: solution (BTM) that docs/validation.md finds in each octave band from 63 Hz, where the
#: band adds the paths in pressure and where in energy (NaN: not measured).
EDGE_ERRORS_DB = {
    # Over one edge: over a thin screen, at the shadow boundaries of a thin screen, a
    # right-angled corner and a wedge open over 225 degrees, and the paths that fade at
    # the edges of a screen's end, a building's corner, a strip and a gap, and over a
    # barrier with both_sides ("Edge diffraction against the exact solution",
    # "Reflections that fade at edges against the exact solution"; at 8 kHz not the
    # strip's 0.19 dB, which is the exact level's own error).
    "one edge": {
        "pressure": (0.08, 0.05, 0.03, 0.01, 0.03, 0.05, 0.04, 0.05),
        "energy": (3.02, 2.65, 2.22, 1.79, 1.38, 1.04, 0.77, 0.56),
    },
    # With one_side, behind a barrier on a reflecting ground: the path reflected off the
    # ground before the top and after it is left out.
    "one side": {
        "pressure": (2.20, 2.98, 2.60, 0.89, 0.80, 0.67, 0.76, 0.73),
        "energy": (_NOT,) * 8,
    },
    # Over two edges (second_order): blocks 0.3 to 3 m wide and two thin screens, and in
    # pressure a barrier 1 m thick on a reflecting ground, whose paths over both edges
    # take no reflection off the ground.
    "two edges": {
        "pressure": (9.85, 6.96, 3.83, 5.66, 5.54, 5.18, 5.08, 5.40),
        "energy": (1.86, 1.26, 0.74, 0.44, 0.30, 0.29, 0.23, 0.13),
    },
    # Where a path's point on an edge leaves the edge's end, as the UTD has no terms for
    # the corners of edges: at a thin screen's corner, with direct.
    "corners of edges": {
        "pressure": (1.83, 0.99, 1.06, 0.41, 0.21, 0.22, 0.05, 0.08),
        "energy": (_NOT,) * 8,
    },
}


@dataclasses.dataclass(frozen=True)
class Spread:
    """The spread of one response's ISO 3382-1 parameters over the draws, each a
    (bands,) array: ``mean``, ``standard`` (the combined standard uncertainty), and
    ``low`` and ``high``, the ends of the coverage interval of probability ``coverage``;
    and the budget, ``absorption`` and ``rays``, the standard uncertainty from the
    materials' absorption alone and from the rays' random error alone (NaN where it is
    not counted; from ``batches`` batches, 0 for none). NaN where a parameter is
    undefined in some draw. ``not_counted``: the method's errors over edges that the
    response has (:data:`EDGE_ERRORS_DB`), each the largest difference of the level
    from the exact solution in each band (dB; NaN where not measured)."""

    mean: RoomParameters
    standard: RoomParameters
    low: RoomParameters
    high: RoomParameters
    absorption: RoomParameters
    rays: RoomParameters
    coverage: float
    draws: int
    batches: int
    not_counted: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict:
        """Plain values, with None for undefined ones (JSON friendly)."""
        return {
            "draws": self.draws,
            "coverage": self.coverage,
            "batches": self.batches,
            **{part: getattr(self, part).as_dict() for part in PARTS},
            "not_counted": {
                name: [float(v) if math.isfinite(v) else None for v in bound]
                for name, bound in self.not_counted.items()
            },
        }


def absorption_draws(scene, materials: Sequence[str], draws: int, seed: int) -> np.ndarray:
    """(draws, materials, bands) the absorption of ``materials`` in each draw: each
    material's own, plus its uncertainty times one deviate per draw that its bands share
    (standard normal, or uniform on -1 to 1 for a tolerance); those without an
    uncertainty keep their own. Held to 0 to ``MAX_ABSORPTION``."""
    if draws < 2:
        raise ValueError("the spread needs 2 draws or more")
    rng = np.random.default_rng(seed)
    out = []
    for name in materials:
        material = scene.materials[name]
        own = np.asarray(material.coefficients("octave")[0], dtype=float)
        spec = material.uncertainty
        if spec is None:
            out.append(np.broadcast_to(own, (draws, own.size)))
            continue
        width = np.asarray(spec.absorption, dtype=float)
        if spec.distribution == "uniform":
            deviate = rng.uniform(-1.0, 1.0, draws)
        else:
            deviate = rng.standard_normal(draws)
        out.append(np.clip(own + deviate[:, None] * width, 0.0, MAX_ABSORPTION))
    return np.stack(out, axis=1)


def material_spread(
    result,
    scene,
    face_group: np.ndarray,
    materials: Sequence[str],
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    coverage: float = 0.95,
    rays: bool = True,
) -> dict:
    """{(source, listener): :class:`Spread`} the spread of ``result``'s parameters that
    the uncertainty of its materials' absorption leads to, with the rays' random error
    (unless ``rays`` is False). ``result`` is ``scene`` simulated with ``face_group`` for
    ``materials`` (:func:`~acoustic_spaces.engine.reweight.groups_of` and
    ``simulate(groups=...)``)."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("the coverage probability lies between 0 and 1")
    if all(scene.materials[name].uncertainty is None for name in materials):
        raise ValueError("no material on the surfaces has an absorption uncertainty")
    traced = np.array([scene.materials[n].coefficients("octave")[0] for n in materials])
    taken = absorption_draws(scene, materials, draws, seed)
    ratios = [log_ratio(traced, drawn) for drawn in taken]
    # The rays' deviates from a stream of their own, so that the absorption's draws stay
    # those of ``seed``.
    rng = np.random.default_rng([seed, 1])
    out = {}
    for key, response in result.responses.items():
        found = [
            response.reweighted(face_group, ratio, absorption=(traced, drawn)).parameters()
            for ratio, drawn in zip(ratios, taken, strict=True)
        ]
        batches = len(response.late_batches) if rays else 0
        errors = response.standard_errors() if batches >= MIN_BATCHES else None
        spread = _spread(found, coverage, errors, 0 if errors is None else batches, rng)
        out[key] = dataclasses.replace(spread, not_counted=edge_errors(response, scene))
    return out


def edge_errors(response, scene) -> dict[str, np.ndarray]:
    """{name: (bands,) dB} the method's errors over edges (:data:`EDGE_ERRORS_DB`) that
    ``response`` (of ``scene``) has, in each band as the band adds its early paths: none
    without paths over an edge; over two edges with ``second_order``, and with a path
    reflected on both sides of an edge left out with ``one_side``."""
    early = response.early
    if early.edge is None or not np.any(early.edge >= 0):
        return {}
    bands = early.energy.shape[1]
    if bands != len(EDGE_ERRORS_DB["one edge"]["pressure"]):
        return {}  # measured in octave bands only
    coherent = 0 if early.spectrum is None else early.spectrum.shape[1]
    settings = scene.simulation.early
    names = ["one edge"]
    if settings.diffraction != "direct" and settings.diffraction_reflections == "one_side":
        names.append("one side")
    if settings.diffraction == "second_order":
        names.append("two edges")
    names.append("corners of edges")
    out = {}
    for name in names:
        bound = EDGE_ERRORS_DB[name]
        out[name] = np.array(
            [bound["pressure"][b] if b < coherent else bound["energy"][b] for b in range(bands)]
        )
    return out


def with_spread(result, scene, face_group, materials, draws: int = DEFAULT_DRAWS, seed: int = 0):
    """``result`` with its ``spreads`` (:func:`material_spread`), or as it is when no
    material on the surfaces has an absorption uncertainty."""
    if all(scene.materials[name].uncertainty is None for name in materials):
        return result
    spreads = material_spread(result, scene, face_group, materials, draws=draws, seed=seed)
    return dataclasses.replace(result, spreads=spreads)


def _spread(
    found: list[RoomParameters],
    coverage: float,
    errors: RoomParameters | None,
    batches: int,
    rng: np.random.Generator,
) -> Spread:
    """The :class:`Spread` of the parameters ``found`` in the draws of the absorption,
    each with the rays' error added where ``errors`` (their standard errors, from
    ``batches`` batches) has it: Student's t of ``batches`` - 1 degrees of freedom, from
    ``rng``."""
    names = [f.name for f in dataclasses.fields(RoomParameters)]
    tail = 0.5 * (1.0 - coverage)
    parts: dict[str, dict[str, np.ndarray]] = {part: {} for part in PARTS}
    for name in names:
        values = np.stack([np.asarray(getattr(p, name), dtype=float) for p in found])
        defined = np.all(np.isfinite(values), axis=0)
        safe = np.where(defined, values, 0.0)
        absorption = safe.std(axis=0, ddof=1)
        rays = np.full(values.shape[1], np.nan)
        if errors is not None:
            error = np.asarray(getattr(errors, name), dtype=float)
            counted = np.isfinite(error)
            nu = batches - 1
            safe = safe + rng.standard_t(nu, size=safe.shape) * np.where(counted, error, 0.0)
            rays = np.where(counted, error * math.sqrt(nu / (nu - 2)), np.nan)
        for part, value in (
            ("mean", safe.mean(axis=0)),
            ("standard", safe.std(axis=0, ddof=1)),
            ("low", np.quantile(safe, tail, axis=0)),
            ("high", np.quantile(safe, 1.0 - tail, axis=0)),
            ("absorption", absorption),
            ("rays", rays),
        ):
            parts[part][name] = np.where(defined, value, np.nan)
    return Spread(
        **{part: RoomParameters(**values) for part, values in parts.items()},
        coverage=coverage,
        draws=len(found),
        batches=batches,
    )


__all__ = [
    "DEFAULT_DRAWS",
    "EDGE_ERRORS_DB",
    "MAX_ABSORPTION",
    "MIN_BATCHES",
    "Spread",
    "absorption_draws",
    "edge_errors",
    "material_spread",
    "with_spread",
]
