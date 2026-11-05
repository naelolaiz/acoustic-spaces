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


@dataclasses.dataclass(frozen=True)
class Spread:
    """The spread of one response's ISO 3382-1 parameters over the draws, each a
    (bands,) array: ``mean``, ``standard`` (the combined standard uncertainty), and
    ``low`` and ``high``, the ends of the coverage interval of probability ``coverage``;
    and the budget, ``absorption`` and ``rays``, the standard uncertainty from the
    materials' absorption alone and from the rays' random error alone (NaN where it is
    not counted; from ``batches`` batches, 0 for none). NaN where a parameter is
    undefined in some draw."""

    mean: RoomParameters
    standard: RoomParameters
    low: RoomParameters
    high: RoomParameters
    absorption: RoomParameters
    rays: RoomParameters
    coverage: float
    draws: int
    batches: int

    def as_dict(self) -> dict:
        """Plain values, with None for undefined ones (JSON friendly)."""
        return {
            "draws": self.draws,
            "coverage": self.coverage,
            "batches": self.batches,
            **{part: getattr(self, part).as_dict() for part in PARTS},
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
        out[key] = _spread(found, coverage, errors, 0 if errors is None else batches, rng)
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
    "MAX_ABSORPTION",
    "MIN_BATCHES",
    "Spread",
    "absorption_draws",
    "material_spread",
    "with_spread",
]
