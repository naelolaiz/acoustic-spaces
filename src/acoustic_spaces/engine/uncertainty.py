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
each one's mean, the standard uncertainty that the absorption leads to (their standard
deviation, JCGM 101 7.6) and, with the rays' deviates added, its probabilistically
symmetric coverage interval (JCGM 101 7.7). The combined standard uncertainty is the
root of the sum of the squares of the absorption's and the rays': the rays' term is
added to each draw on its own, so its variance adds exactly, and the standard deviation
of draws that hold its deviates would give the same less steadily (with 4 or 5 batches
far less: Student's t of 3 or 4 degrees of freedom has no fourth moment, which a sample
variance needs to settle). The budget is these two. An end of the interval, or a
one-sided bound, past what the parameter can be (a decay time below 0, a D50 above 1:
:data:`~acoustic_spaces.parameters.LIMITS`) is held there, as the Analytical Methods
Committee recommends for an interval near a natural limit (AMC Technical Brief 26A,
2008; Cowen and Ellison, Analyst 131 (2006) 710): the rays' error of a value near its
limit would else take it past it. They also give its one-sided bounds, for a
limit on one side (JCGM 106:2012's conformance probability): the value it is at least
with probability ``one_sided``, and the one it is at most with that probability, the
draws' quantiles of 1 - ``one_sided`` and of ``one_sided`` (for a normal spread at 95 %,
the mean less and plus 1.645 standard uncertainties).

The draws are a number given beforehand (JCGM 101 7.2.1), or, with ``draws`` adaptive, as
many as JCGM 101 7.9's adaptive procedure takes: batch after batch until, for every
parameter in every band of every response, twice the standard deviation of the batches'
mean, standard uncertainty and interval's ends, each over the root of the number of
batches, is no more than the numerical tolerance of the standard uncertainty at
``ADAPTIVE_DIGITS`` significant digits (7.9.2), and so for the one-sided bounds, which
the Supplement's test does not name. A batch is ``ADAPTIVE_BATCH`` draws, not
the 10 000 the Supplement names (a choice it calls arbitrary), as each draw re-weights
the whole run. The test is first made after ``ADAPTIVE_MIN_BATCHES`` batches, not 2: the
Supplement's first test comes after 20 000 draws, which settle one digit of a normal
spread whatever it is, while two batches of 100 pass by chance about one time in six
for a result that needs some 460 draws, and one time in twenty for one that needs a
thousand. The draws stop at ``ADAPTIVE_MAX_DRAWS`` if
the results are not stable by then (:attr:`Spread.stable`).

A material that gives no uncertainty of its own takes none, unless its absorption is a
built-in database entry's, whose source that database does not name
(:attr:`~acoustic_spaces.scene.model.Material.builtin`): then it takes the scene's
``simulation.uncertainty.presets`` (:func:`uncertainty_of`). By default that is a
tolerance of plus or minus ``PRESET_TOLERANCE`` in every band; ``iso_12999_2`` takes
twice the reproducibility standard deviation ISO 12999-2:2020 gives for absorption
measured to ISO 354 (:func:`octave_reproducibility`), doubled as such a table's values
are not a measurement of the material in the room.

Not counted: the scattering's uncertainty, nor the method's (docs/validation.md), nor
the rays' error with fewer than ``MIN_BATCHES`` batches (Student's t has no variance
then) or in the parameters it is not found for (those from the late sound's directions).
Nor a band in which a material absorbs everything: no energy is left there to
re-weight, so that band is not drawn, and the spread names it (:func:`held`). Nor an
absorption uncertainty given beside a ground's impedance (:func:`grounds_left_out`):
the impedance gives that ground's reflection, which re-weighting by 1 - a does not
follow. A draw stays at or above what its material lets through, which is absorbed
whatever the rest of the surface does.
Of the method's errors, those of the paths over edges are listed where a response has
such paths (:data:`EDGE_ERRORS_DB`), each with the largest difference of the early level
from the exact solution that the validation finds in each band, but not drawn: they are
measured for whole levels in a few geometries, not for each path.
A draw of the absorption below 0 is taken as 0 and one above ``MAX_ABSORPTION`` as that,
so near those ends the spread is a little less than the uncertainty gives. The
re-weighting is exact for the early paths; the late part's follows Eyring's decay with
the counts' covariance to second order (:mod:`~acoustic_spaces.engine.reweight`), an
estimate that stays near tracing again also for a draw far from the traced absorption.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

from acoustic_spaces.engine.reweight import (
    cannot_reweight,
    groups_of,
    log_ratio,
    reweighted_materials,
)
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import LIMITS, RoomParameters
from acoustic_spaces.scene.model import ONE_SIDED, SPREAD_DRAWS, AbsorptionUncertainty

#: The highest absorption a draw takes: re-weighting needs some sound reflected.
MAX_ABSORPTION = 0.99
#: The draws :func:`material_spread` takes unless told otherwise.
DEFAULT_DRAWS = SPREAD_DRAWS
#: The probability of the one-sided bounds :func:`material_spread` gives unless told
#: otherwise.
DEFAULT_ONE_SIDED = ONE_SIDED
#: ``draws`` for as many as the adaptive procedure of JCGM 101 7.9 takes.
ADAPTIVE = "adaptive"
#: The draws of each batch of the adaptive procedure.
ADAPTIVE_BATCH = 100
#: The batches the adaptive procedure takes before its first test.
ADAPTIVE_MIN_BATCHES = 10
#: The most draws the adaptive procedure takes.
ADAPTIVE_MAX_DRAWS = 20_000
#: The significant digits of a standard uncertainty that the adaptive procedure takes as
#: meaningful (n_dig of JCGM 101 7.9.2).
ADAPTIVE_DIGITS = 1
#: A standard uncertainty no larger than this is rounding, and stable as it is.
_NO_SPREAD = 1e-12
#: How far apart the absorptions lie that the adaptive procedure fits the admittance at,
#: in deviates of a material's distribution (a normal one's standard deviations, a
#: tolerance's half-widths): about what ``ADMITTANCE_NODES`` over the range of
#: ``DEFAULT_DRAWS`` draws gives.
ADMITTANCE_STEPS = {"normal": 0.75, "uniform": 0.25}
#: The absorptions of each material, over its draws' range, that the admittance is
#: fitted at for the phases of the paths that add in pressure (the draws take it between
#: the two nearest).
ADMITTANCE_NODES = 9
#: The fewest batches of late rays whose error counts: Student's t with n - 1 degrees of
#: freedom has a variance from 3 up.
MIN_BATCHES = 4
#: How many draws :func:`material_spread` takes between two calls of its ``progress``.
PROGRESS_DRAWS = 25
#: The parts of a :class:`Spread`.
PARTS = ("mean", "standard", "low", "high", "at_least", "at_most", "absorption", "rays")
#: The half-width of the tolerance a built-in material of unknown source takes in every
#: band (``simulation.uncertainty.presets: tolerance``, the default).
PRESET_TOLERANCE = 0.1
#: ISO 12999-2:2020 Table 1: the reproducibility standard deviation of an absorption
#: coefficient measured to ISO 354 in a third-octave band, sigma_R = m alpha_s + n, as
#: {band (Hz): (m, n)}. An octave band the standard gives no value for
#: (:data:`ISO_12999_2_OCTAVES`) takes the mean of the sigma_R of its third octaves the
#: table has (the 63 Hz octave those at 63 and 80 Hz, as the table has no 50 Hz row),
#: the 8 kHz octave, past the table's last band, that at 5 kHz. Below 100 Hz the rows
#: lie outside ISO 354's range, and the table comes from absorbers (suspended ceilings,
#: mineral wool, foams), so for hard surfaces it is taken beyond its data.
ISO_12999_2_THIRDS = {
    63: (0.450, 0.015),
    80: (0.330, 0.015),
    100: (0.240, 0.015),
    125: (0.180, 0.015),
    160: (0.140, 0.015),
    200: (0.110, 0.015),
    250: (0.090, 0.015),
    315: (0.075, 0.015),
    400: (0.060, 0.015),
    500: (0.050, 0.015),
    630: (0.045, 0.015),
    800: (0.040, 0.015),
    1000: (0.040, 0.015),
    1250: (0.040, 0.016),
    1600: (0.037, 0.018),
    2000: (0.035, 0.021),
    2500: (0.030, 0.026),
    3150: (0.030, 0.032),
    4000: (0.030, 0.040),
    5000: (0.026, 0.060),
}
#: ISO 12999-2:2020 Table 2: the reproducibility standard deviation of the practical
#: absorption coefficient (ISO 11654: the octave band's mean of its third octaves,
#: rounded to 0.05), sigma_R = m alpha_p + n, as {octave band (Hz): (m, n)}, which an
#: octave band's absorption takes where the standard gives it. That rounding is in it,
#: so for unrounded values it is a little more than the measurement's alone.
ISO_12999_2_OCTAVES = {
    250: (0.059, 0.016),
    500: (0.000, 0.040),
    1000: (0.000, 0.040),
    2000: (0.000, 0.040),
    4000: (0.000, 0.050),
}
#: How many times ISO 12999-2's reproducibility a built-in material takes with
#: ``iso_12999_2``.
ISO_12999_2_FACTOR = 2.0
#: What each choice of ``simulation.uncertainty.presets`` gives a built-in material.
PRESET_UNCERTAINTIES = {
    "tolerance": f"plus or minus {PRESET_TOLERANCE:g}",
    "iso_12999_2": "twice the reproducibility ISO 12999-2 gives for ISO 354",
}
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
    # Over two edges (second_order): rigid blocks 0.3 to 3 m wide and two thin screens, and
    # in pressure a barrier 1 m thick on a reflecting ground, whose paths over both edges
    # take no reflection off the ground.
    "two edges": {
        "pressure": (9.85, 6.96, 3.83, 5.66, 5.54, 5.18, 5.08, 5.40),
        "energy": (1.86, 1.26, 0.74, 0.44, 0.30, 0.29, 0.23, 0.13),
    },
    # Where a path's point on an edge reaches the edge's end, with direct: a thin
    # screen's edges fade out past their free ends, within 0.10 dB of BTM at a screen's
    # corner, but the edges of a solid block stop at its corners, as the UTD has no terms
    # for the corners of edges.
    "corners of edges": {
        "pressure": (2.29, 1.50, 1.62, 1.89, 1.44, 0.89, 0.58, 0.50),
        "energy": (_NOT,) * 8,
    },
}


@dataclasses.dataclass(frozen=True)
class Spread:
    """The spread of one response's ISO 3382-1 parameters over the draws, each a
    (bands,) array: ``mean``, ``standard`` (the combined standard uncertainty), and
    ``low`` and ``high``, the ends of the coverage interval of probability ``coverage``;
    ``at_least`` and ``at_most``, the one-sided bounds of probability ``one_sided`` (the
    value the parameter is at least with that probability, and the one it is at most);
    and the budget, ``absorption`` and ``rays``, the standard uncertainty from the
    materials' absorption alone and from the rays' random error alone (NaN where it is
    not counted; from ``batches`` batches, 0 for none). NaN where a parameter is
    undefined in some draw. ``not_counted``: the method's errors over edges that the
    response has (:data:`EDGE_ERRORS_DB`), each the largest difference of the level
    from the exact solution in each band (dB; NaN where not measured). ``assumed``: the
    built-in materials of unknown source that took ``simulation.uncertainty.presets``,
    with that choice (:func:`assumed`). ``held``: the bands (Hz) in which a material
    with an uncertainty absorbs everything and is not drawn (:func:`held`). ``stable``:
    with the draws chosen adaptively,
    whether every result of every response met JCGM 101 7.9's test before the draws
    reached their limit; None with a number of draws given beforehand, which is not
    tested."""

    mean: RoomParameters
    standard: RoomParameters
    low: RoomParameters
    high: RoomParameters
    at_least: RoomParameters
    at_most: RoomParameters
    absorption: RoomParameters
    rays: RoomParameters
    coverage: float
    one_sided: float
    draws: int
    batches: int
    not_counted: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    assumed: dict[str, str] = dataclasses.field(default_factory=dict)
    held: dict[str, list[int]] = dataclasses.field(default_factory=dict)
    stable: bool | None = None

    def as_dict(self) -> dict:
        """Plain values, with None for undefined ones (JSON friendly)."""
        return {
            "draws": self.draws,
            "stable": self.stable,
            "coverage": self.coverage,
            "one_sided": self.one_sided,
            "batches": self.batches,
            **{part: getattr(self, part).as_dict() for part in PARTS},
            "not_counted": {
                name: [float(v) if math.isfinite(v) else None for v in bound]
                for name, bound in self.not_counted.items()
            },
            "assumed": dict(self.assumed),
            "held": {name: list(bands) for name, bands in self.held.items()},
        }


def octave_reproducibility(absorption) -> np.ndarray:
    """(octave bands,) ISO 12999-2's reproducibility standard deviation of the octave
    band ``absorption``: its octave bands' (:data:`ISO_12999_2_OCTAVES`) where it gives
    them, else the mean of its third octaves' (:data:`ISO_12999_2_THIRDS`)."""
    alpha = np.asarray(absorption, dtype=float)
    out = np.empty(len(OCTAVE_BANDS_HZ))
    rows = sorted(ISO_12999_2_THIRDS)
    for b in range(len(OCTAVE_BANDS_HZ)):
        if OCTAVE_BANDS_HZ[b] in ISO_12999_2_OCTAVES:
            m, n = ISO_12999_2_OCTAVES[OCTAVE_BANDS_HZ[b]]
            out[b] = m * alpha[b] + n
            continue
        thirds = [f for f in THIRD_OCTAVE_BANDS_HZ[3 * b : 3 * b + 3] if f in ISO_12999_2_THIRDS]
        if not thirds:  # past the table: its nearest band
            centre = OCTAVE_BANDS_HZ[b]
            thirds = [min(rows, key=lambda f: abs(math.log(f / centre)))]
        out[b] = np.mean([m * alpha[b] + n for m, n in (ISO_12999_2_THIRDS[f] for f in thirds)])
    return out


def uncertainty_of(scene, name: str) -> AbsorptionUncertainty | None:
    """The uncertainty of material ``name``'s absorption that a spread takes: its own;
    for a built-in one of unknown source without one, the scene's
    ``simulation.uncertainty.presets``; else None. None for a ground with an impedance
    too (:func:`grounds_left_out`)."""
    material = scene.materials[name]
    if material.impedance is not None:
        return None
    if material.uncertainty is not None or not material.builtin:
        return material.uncertainty
    choice = scene.simulation.uncertainty.presets
    if choice == "none":
        return None
    source = f"{PRESET_UNCERTAINTIES[choice]}, for a built-in material of unknown source"
    if choice == "tolerance":
        return AbsorptionUncertainty(
            absorption=PRESET_TOLERANCE, distribution="uniform", source=source
        )
    own = material.coefficients("octave")[0]
    sigma = np.minimum(ISO_12999_2_FACTOR * octave_reproducibility(own), 1.0)
    return AbsorptionUncertainty(absorption=[float(v) for v in sigma], source=source)


def assumed(scene, materials: Sequence[str]) -> dict[str, str]:
    """{material: the choice of ``simulation.uncertainty.presets`` it takes} those of
    ``materials`` that take the default uncertainty of a built-in material
    (:func:`uncertainty_of`)."""
    choice = scene.simulation.uncertainty.presets
    return {
        name: choice
        for name in materials
        if scene.materials[name].uncertainty is None and uncertainty_of(scene, name) is not None
    }


def held(scene, materials: Sequence[str]) -> dict[str, list[int]]:
    """{material: octave bands (Hz)} where those of ``materials`` that have an
    uncertainty (:func:`uncertainty_of`) absorb everything: no draw moves such a band,
    as no energy is left in it to re-weight."""
    out = {}
    for name in materials:
        if uncertainty_of(scene, name) is None:
            continue
        own = scene.materials[name].coefficients("octave")[0]
        bands = [hz for hz, a in zip(OCTAVE_BANDS_HZ, own, strict=True) if a >= 1.0]
        if bands:
            out[name] = bands
    return out


def grounds_left_out(room) -> list[str]:
    """The materials on ``room``'s surfaces that give an absorption uncertainty beside
    a ground impedance. A spread leaves them out: the impedance gives the ground's
    reflection, which does not change with the absorption as 1 - a, the factor that
    re-weighting applies."""
    materials = room.scene.materials
    return [
        name
        for name in reweighted_materials(room)
        if materials[name].impedance is not None and materials[name].uncertainty is not None
    ]


def has_uncertainty(scene, materials: Sequence[str] | None = None) -> bool:
    """Whether some material of ``materials`` (default all of ``scene``'s) has an
    absorption uncertainty a spread takes (:func:`uncertainty_of`)."""
    names = scene.materials if materials is None else materials
    return any(uncertainty_of(scene, name) is not None for name in names)


def spread_groups(
    room, settings=None
) -> tuple[tuple[np.ndarray, list[str]] | None, str | None, bool]:
    """(groups, why, own) for the spread of a run of ``room`` with ``settings`` (default
    the scene's): the face groups and materials to trace it with
    (:func:`~acoustic_spaces.engine.reweight.groups_of`), or None. None with ``why``
    None: no material on its surfaces has an uncertainty a spread takes. Otherwise
    ``why`` is what the run would need to re-weight
    (:func:`~acoustic_spaces.engine.reweight.cannot_reweight`), and ``own`` whether some
    material's uncertainty is one of its own, not a built-in material's default."""
    uncertain = [n for n in reweighted_materials(room) if uncertainty_of(room.scene, n) is not None]
    if not uncertain:
        return None, None, False
    own = any(room.scene.materials[n].uncertainty is not None for n in uncertain)
    why = cannot_reweight(room, settings)
    return (None if why else groups_of(room, settings)), why, own


def absorption_draws(scene, materials: Sequence[str], draws: int, seed: int) -> np.ndarray:
    """(draws, materials, bands) the absorption of ``materials`` in each draw: each
    material's own, plus its uncertainty times one deviate per draw that its bands share
    (standard normal, or uniform on -1 to 1 for a tolerance); those without an
    uncertainty (:func:`uncertainty_of`) keep their own. Held to 0 to
    ``MAX_ABSORPTION``, or to the material's own absorption where that is more, and from
    below to what the material lets through, which counts as absorbed; a band that
    absorbs everything is not drawn (:func:`held`)."""
    return _draws(scene, materials, draws, seed)[0]


def _limits(material, own: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The least and the most absorption per band that a draw of ``material`` takes,
    ``own`` its absorption: from what it lets through, which is absorbed whatever the
    rest of the surface does (0 for a surface that lets nothing through), to
    ``MAX_ABSORPTION``, or to ``own`` where that is more."""
    passing = material.transmission_coefficients("octave")
    bottom = np.zeros(own.size) if passing is None else np.minimum(passing, own)
    return bottom, np.maximum(own, MAX_ABSORPTION)


def _draws(scene, materials, draws, seed):
    """:func:`absorption_draws`, with the (draws, materials) deviates and the
    (materials, bands) widths they scale (0 for a material without an uncertainty)."""
    if draws < 2:
        raise ValueError("the spread needs 2 draws or more")
    rng = np.random.default_rng(seed)
    out, deviates, widths = [], [], []
    for name in materials:
        own = np.asarray(scene.materials[name].coefficients("octave")[0], dtype=float)
        spec = uncertainty_of(scene, name)
        if spec is None:
            out.append(np.broadcast_to(own, (draws, own.size)))
            deviates.append(np.zeros(draws))
            widths.append(np.zeros(own.size))
            continue
        # A band that absorbs everything stays: no energy is left there to re-weight.
        width = np.where(own >= 1.0, 0.0, np.asarray(spec.absorption, dtype=float))
        if spec.distribution == "uniform":
            deviate = rng.uniform(-1.0, 1.0, draws)
        else:
            deviate = rng.standard_normal(draws)
        out.append(np.clip(own + deviate[:, None] * width, *_limits(scene.materials[name], own)))
        deviates.append(deviate)
        widths.append(width)
    return np.stack(out, axis=1), np.stack(deviates, axis=1), np.stack(widths)


class _Admittances:
    """The admittance of each draw's absorption for the phases of the paths that add in
    pressure (:func:`~acoustic_spaces.engine.coherent.absorption_phases`): for a
    material's draws, fitted when first asked for at nodes among their deviates
    (:func:`~acoustic_spaces.engine.impedance.fit_admittance`, about a second each) and
    taken between the two nearest, as fitting each draw's would take that for every
    draw. The nodes are ``ADMITTANCE_NODES`` over the range of the draws given at once
    (:meth:`over`), or, for draws that come batch by batch, every ``steps`` (one per
    material) from 0 (:meth:`along`), so that a later batch moves no earlier draw's.
    ``limits`` holds each material's draws (:func:`_limits`), and so the nodes. Other
    rows (as traced) are fitted as the trace fits them."""

    def __init__(self, traced, widths, limits) -> None:
        self._traced, self._widths, self._limits = traced, widths, limits
        self._fits: dict[tuple[int, float], object] = {}
        self._found: dict[bytes, tuple[int, float, float, float]] = {}

    def over(self, taken, deviates) -> None:
        """Take the draws ``taken`` (draws, materials, bands) of ``deviates`` (draws,
        materials), with each material's nodes over its deviates' range."""
        for k in range(self._traced.shape[0]):
            if not np.any(self._widths[k]):
                continue
            nodes = np.linspace(deviates[:, k].min(), deviates[:, k].max(), ADMITTANCE_NODES)
            for drawn, deviate in zip(taken[:, k], deviates[:, k], strict=True):
                j = int(np.clip(np.searchsorted(nodes, deviate) - 1, 0, len(nodes) - 2))
                t = float((deviate - nodes[j]) / (nodes[j + 1] - nodes[j]))
                self._found[np.ascontiguousarray(drawn).tobytes()] = (
                    k,
                    float(nodes[j]),
                    float(nodes[j + 1]),
                    t,
                )

    def along(self, taken, deviates, steps) -> None:
        """Take the draws ``taken`` of ``deviates``, with material k's nodes every
        ``steps[k]`` from 0."""
        for k in range(self._traced.shape[0]):
            if not np.any(self._widths[k]):
                continue
            for drawn, deviate in zip(taken[:, k], deviates[:, k], strict=True):
                j = math.floor(deviate / steps[k])
                t = float(deviate / steps[k] - j)
                self._found[np.ascontiguousarray(drawn).tobytes()] = (
                    k,
                    j * steps[k],
                    (j + 1) * steps[k],
                    t,
                )

    def _fit(self, k: int, node: float):
        from acoustic_spaces.engine.impedance import fit_admittance

        if (k, node) not in self._fits:
            row = np.clip(self._traced[k] + node * self._widths[k], *self._limits[k])
            self._fits[k, node] = fit_admittance(row, self._traced.shape[1])
        return self._fits[k, node]

    def __call__(self, row: np.ndarray):
        from acoustic_spaces.engine.impedance import fit_admittance

        found = self._found.get(np.ascontiguousarray(row, dtype=float).tobytes())
        if found is None:
            return fit_admittance(row.tolist(), self._traced.shape[1])
        k, below, above, t = found
        low, high = self._fit(k, below), self._fit(k, above)
        return lambda f: (1.0 - t) * low(f) + t * high(f)


def numerical_tolerance(standard, digits: int = 1) -> np.ndarray:
    """The numerical tolerance of each standard uncertainty ``standard`` with ``digits``
    significant digits taken as meaningful (JCGM 101 7.9.2): written as c 10^l with c a
    whole number of ``digits`` digits, half of 10^l. Infinite where there is no spread
    (``standard`` at most ``_NO_SPREAD``, rounding)."""
    standard = np.asarray(standard, dtype=float)
    out = np.full(standard.shape, math.inf)
    some = standard > _NO_SPREAD
    power = np.floor(np.log10(standard[some])) - (digits - 1)
    # 0.096 at one digit is 10 10^-2, which is 1 10^-1.
    power += np.round(standard[some] / 10.0**power) >= 10.0**digits
    out[some] = 0.5 * 10.0**power
    return out


class _Tally:
    """One response's parameters in the draws, batch by batch: the parameters each draw
    of the absorption gives, and the rays' error to add to each (their standard errors
    ``errors``, from ``batches`` batches of rays, times a deviate of Student's t of
    ``batches`` - 1 degrees of freedom; None for none). The mean and the absorption's
    standard uncertainty are those of the draws' parameters, the combined standard
    uncertainty has the rays' in closed form, and the interval's ends and the one-sided
    bounds are quantiles of the draws with the rays' deviates."""

    def __init__(
        self, errors: RoomParameters | None, batches: int, coverage: float, one_sided: float
    ) -> None:
        self._errors, self._batches, self._coverage = errors, batches, coverage
        # The draws' quantiles: the interval's ends, then the one-sided bounds.
        tail = 0.5 * (1.0 - coverage)
        self._quantiles = (tail, 1.0 - tail, 1.0 - one_sided, one_sided)
        self._one_sided = one_sided
        self._names = [f.name for f in dataclasses.fields(RoomParameters)]
        self._values: dict[str, list[np.ndarray]] = {name: [] for name in self._names}
        self._noise: dict[str, list[np.ndarray]] = {name: [] for name in self._names}
        # Of each batch, each parameter's (6, bands) mean and standard deviation over
        # the absorption's draws, and with the rays' deviates its interval's ends and
        # one-sided bounds; and whether it is defined in every draw of it.
        self._estimates: dict[str, list[np.ndarray]] = {name: [] for name in self._names}
        self._defined: dict[str, np.ndarray] = {}
        self.draws = 0

    def add(self, found: list[RoomParameters], rng: np.random.Generator) -> None:
        """Take a batch: the parameters ``found`` in its draws of the absorption, the
        rays' deviates from ``rng``."""
        self.draws += len(found)
        for name in self._names:
            values = np.stack([np.asarray(getattr(p, name), dtype=float) for p in found])
            defined = np.all(np.isfinite(values), axis=0)
            safe = np.where(defined, values, 0.0)
            self._values[name].append(values)
            mean, deviation = safe.mean(axis=0), safe.std(axis=0, ddof=1)
            if self._errors is not None:
                error = np.asarray(getattr(self._errors, name), dtype=float)
                nu = self._batches - 1
                noise = rng.standard_t(nu, size=values.shape) * np.where(
                    np.isfinite(error), error, 0.0
                )
                self._noise[name].append(noise)
                safe = safe + noise
            self._defined[name] = self._defined.get(name, True) & defined
            self._estimates[name].append(np.vstack([mean, deviation, self._ends(name, safe)]))

    def _ends(self, name: str, drawn: np.ndarray) -> np.ndarray:
        """(4, bands) the interval's ends and the one-sided bounds of parameter ``name``
        from ``drawn`` (draws, bands), the draws with the rays' deviates: their
        quantiles, held to what the parameter can be."""
        ends = np.quantile(drawn, self._quantiles, axis=0)
        return np.clip(ends, *LIMITS[name]) if name in LIMITS else ends

    def _rays(self, name: str) -> np.ndarray:
        """(bands,) the rays' standard uncertainty of parameter ``name``: its standard
        error times sqrt(nu / (nu - 2)), Student's t of nu degrees of freedom being
        that much wider than the normal; NaN where the rays' error is not counted."""
        if self._errors is None:
            return np.full(self._values[name][0].shape[1], np.nan)
        error = np.asarray(getattr(self._errors, name), dtype=float)
        nu = self._batches - 1
        return np.where(np.isfinite(error), error * math.sqrt(nu / (nu - 2)), np.nan)

    def unstable(self) -> tuple[float, str, int] | None:
        """None when every parameter in every band meets the test of JCGM 101 7.9.4 over
        the batches taken: twice the standard deviation of the batches'
        mean, combined standard uncertainty and interval's ends, each over the root of
        the number of batches, is at most the numerical tolerance of the combined
        standard uncertainty of all the draws; and so for the one-sided bounds, which
        are results as the interval's ends are. Else the one furthest from it: how many
        times the tolerance, its name and its band."""
        worst: tuple[float, str, int] | None = None
        for name in self._names:
            estimates = np.stack(self._estimates[name])  # (batches, 6, bands)
            count, size = estimates.shape[0], self.draws // estimates.shape[0]
            means, deviations = estimates[:, 0], estimates[:, 1]
            rays = np.nan_to_num(self._rays(name))
            # All the draws' standard deviation, from the batches' means and deviations,
            # and with the rays' the combined standard uncertainty: of all the draws,
            # and of each batch.
            squares = (size - 1) * deviations**2 + size * (means - means.mean(axis=0)) ** 2
            standard = np.hypot(np.sqrt(squares.sum(axis=0) / (self.draws - 1)), rays)
            estimates[:, 1] = np.hypot(deviations, rays)
            twice = 2.0 * estimates.std(axis=0, ddof=1) / math.sqrt(count)
            over = np.where(
                self._defined[name],
                (twice / numerical_tolerance(standard, ADAPTIVE_DIGITS)).max(axis=0),
                0.0,
            )
            band = int(np.argmax(over))
            if over[band] > 1.0 and (worst is None or over[band] > worst[0]):
                worst = (float(over[band]), name, band)
        return worst

    def spread(self, stable: bool | None) -> Spread:
        """The :class:`Spread` of all the draws taken."""
        parts: dict[str, dict[str, np.ndarray]] = {part: {} for part in PARTS}
        for name in self._names:
            values = np.concatenate(self._values[name])
            defined = np.all(np.isfinite(values), axis=0)
            safe = np.where(defined, values, 0.0)
            mean, absorption = safe.mean(axis=0), safe.std(axis=0, ddof=1)
            rays = self._rays(name)
            if self._errors is not None:
                safe = safe + np.concatenate(self._noise[name])
            low, high, at_least, at_most = self._ends(name, safe)
            for part, value in (
                ("mean", mean),
                ("standard", np.hypot(absorption, np.nan_to_num(rays))),
                ("low", low),
                ("high", high),
                ("at_least", at_least),
                ("at_most", at_most),
                ("absorption", absorption),
                ("rays", rays),
            ):
                parts[part][name] = np.where(defined, value, np.nan)
        each = {part: RoomParameters(**values) for part, values in parts.items()}
        return Spread(
            mean=each["mean"],
            standard=each["standard"],
            low=each["low"],
            high=each["high"],
            at_least=each["at_least"],
            at_most=each["at_most"],
            absorption=each["absorption"],
            rays=each["rays"],
            coverage=self._coverage,
            one_sided=self._one_sided,
            draws=self.draws,
            batches=self._batches,
            stable=stable,
        )


def material_spread(
    result,
    scene,
    face_group: np.ndarray,
    materials: Sequence[str],
    draws: int | str = DEFAULT_DRAWS,
    seed: int = 0,
    coverage: float = 0.95,
    rays: bool = True,
    progress=None,
    max_draws: int | None = None,
    one_sided: float = DEFAULT_ONE_SIDED,
) -> dict:
    """{(source, listener): :class:`Spread`} the spread of ``result``'s parameters that
    the uncertainty of its materials' absorption leads to, with the rays' random error
    (unless ``rays`` is False). ``result`` is ``scene`` simulated with ``face_group`` for
    ``materials`` (:func:`~acoustic_spaces.engine.reweight.groups_of` and
    ``simulate(groups=...)``). ``draws`` is their number, or ``ADAPTIVE``: batches of
    ``ADAPTIVE_BATCH``, ``ADAPTIVE_MIN_BATCHES`` or more, until every result of every
    response is stable by the test of JCGM 101 7.9 (:attr:`Spread.stable`),
    ``max_draws`` at most (default ``ADAPTIVE_MAX_DRAWS``). ``one_sided`` is the
    probability of the one-sided bounds, between 0.5 and 1. ``progress(message)``,
    if given, is called before each response's draws and every ``PROGRESS_DRAWS`` draws;
    an exception it raises (a cancel) stops the spread."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("the coverage probability lies between 0 and 1")
    if not 0.5 < one_sided < 1.0:
        raise ValueError("the one-sided bounds' probability lies between 0.5 and 1")
    if not has_uncertainty(scene, materials):
        raise ValueError("no material on the surfaces has an absorption uncertainty")
    adaptive = draws == ADAPTIVE
    limit = ADAPTIVE_MAX_DRAWS if max_draws is None else max_draws
    first = ADAPTIVE_MIN_BATCHES * ADAPTIVE_BATCH
    if adaptive and limit < first:
        raise ValueError(f"the adaptive spread needs {first} draws or more")
    if not adaptive and (isinstance(draws, str) or draws < 2):
        raise ValueError(f"the spread needs 2 draws or more, or {ADAPTIVE}")
    traced = np.array([scene.materials[n].coefficients("octave")[0] for n in materials])
    say = progress or (lambda _: None)
    tallies = {}
    for key, response in result.responses.items():
        batches = len(response.late_batches) if rays else 0
        errors = response.standard_errors() if batches >= MIN_BATCHES else None
        tallies[key] = _Tally(errors, 0 if errors is None else batches, coverage, one_sided)
    admittance_of: _Admittances | None = None
    steps = [
        ADMITTANCE_STEPS[getattr(uncertainty_of(scene, n), "distribution", "normal")]
        for n in materials
    ]
    limits = [_limits(scene.materials[n], own) for n, own in zip(materials, traced, strict=True)]
    stable: bool | None = None
    taken_so_far = 0
    while True:
        if adaptive:
            batch = taken_so_far // ADAPTIVE_BATCH
            # Each batch from streams of its own, so that the first ones are the same
            # however many follow.
            taken, deviates, widths = _draws(scene, materials, ADAPTIVE_BATCH, [seed, 2, batch])
            stream = [seed, 3, batch]
            if admittance_of is None:
                admittance_of = _Admittances(traced, widths, limits)
            admittance_of.along(taken, deviates, steps)
        else:
            taken, deviates, widths = _draws(scene, materials, draws, seed)
            # The rays' deviates from a stream of their own, so that the absorption's
            # draws stay those of ``seed``.
            stream = [seed, 1]
            admittance_of = _Admittances(traced, widths, limits)
            admittance_of.over(taken, deviates)
        ratios = [log_ratio(traced, drawn) for drawn in taken]
        of = "" if adaptive else f" of {draws}"
        for n, (key, response) in enumerate(result.responses.items(), start=1):
            found = []
            for k, (ratio, drawn) in enumerate(zip(ratios, taken, strict=True)):
                if k % PROGRESS_DRAWS == 0:
                    say(
                        f"the spread: {key[0]} -> {key[1]} ({n} of {len(result.responses)}), "
                        f"draw {taken_so_far + k + 1}{of}"
                    )
                found.append(
                    response.reweighted(
                        face_group, ratio, absorption=(traced, drawn), admittance_of=admittance_of
                    ).parameters()
                )
            # Each response from the start of the stream: its spread is the same
            # whichever other responses the run has.
            tallies[key].add(found, np.random.default_rng(stream))
        taken_so_far += len(taken)
        if not adaptive:
            break
        if taken_so_far >= first:
            stable = all(tally.unstable() is None for tally in tallies.values())
            if stable or taken_so_far + ADAPTIVE_BATCH > limit:
                break
    defaults, full = assumed(scene, materials), held(scene, materials)
    return {
        key: dataclasses.replace(
            tally.spread(stable),
            not_counted=edge_errors(result.responses[key], result.settings),
            assumed=defaults,
            held=full,
        )
        for key, tally in tallies.items()
    }


def edge_errors(response, settings) -> dict[str, np.ndarray]:
    """{name: (bands,) dB} the method's errors over edges (:data:`EDGE_ERRORS_DB`) that
    ``response`` has, simulated with ``settings`` (the run's, not the scene's where they
    differ), in each band as the band adds its early paths: none without paths over an
    edge; over two edges with ``second_order``, and with a path reflected on both sides
    of an edge left out with ``one_side``."""
    early = response.early
    if early.edge is None or not np.any(early.edge >= 0):
        return {}
    bands = early.energy.shape[1]
    if bands != len(EDGE_ERRORS_DB["one edge"]["pressure"]):
        return {}  # measured in octave bands only
    coherent = 0 if early.spectrum is None else early.spectrum.shape[1]
    settings = settings.early
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


def with_spread(
    result,
    scene,
    face_group,
    materials,
    draws: int | str = DEFAULT_DRAWS,
    seed: int = 0,
    progress=None,
    one_sided: float = DEFAULT_ONE_SIDED,
):
    """``result`` with its ``spreads`` (:func:`material_spread`, which ``progress`` and
    ``one_sided`` go to), or as it is when no material on the surfaces has an absorption uncertainty
    (:func:`has_uncertainty`)."""
    if not has_uncertainty(scene, materials):
        return result
    spreads = material_spread(
        result,
        scene,
        face_group,
        materials,
        draws=draws,
        seed=seed,
        progress=progress,
        one_sided=one_sided,
    )
    return dataclasses.replace(result, spreads=spreads)


__all__ = [
    "ADAPTIVE",
    "ADAPTIVE_BATCH",
    "ADAPTIVE_MAX_DRAWS",
    "ADAPTIVE_MIN_BATCHES",
    "DEFAULT_DRAWS",
    "DEFAULT_ONE_SIDED",
    "EDGE_ERRORS_DB",
    "MAX_ABSORPTION",
    "MIN_BATCHES",
    "Spread",
    "absorption_draws",
    "edge_errors",
    "grounds_left_out",
    "held",
    "material_spread",
    "numerical_tolerance",
    "spread_groups",
    "with_spread",
]
