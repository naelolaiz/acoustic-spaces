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
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import RoomParameters
from acoustic_spaces.scene.model import AbsorptionUncertainty

#: The highest absorption a draw takes: re-weighting needs some sound reflected.
MAX_ABSORPTION = 0.99
#: The draws :func:`material_spread` takes unless told otherwise.
DEFAULT_DRAWS = 400
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
PARTS = ("mean", "standard", "low", "high", "absorption", "rays")
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
    # Over two edges (second_order): blocks 0.3 to 3 m wide and two thin screens, and in
    # pressure a barrier 1 m thick on a reflecting ground, whose paths over both edges
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
    and the budget, ``absorption`` and ``rays``, the standard uncertainty from the
    materials' absorption alone and from the rays' random error alone (NaN where it is
    not counted; from ``batches`` batches, 0 for none). NaN where a parameter is
    undefined in some draw. ``not_counted``: the method's errors over edges that the
    response has (:data:`EDGE_ERRORS_DB`), each the largest difference of the level
    from the exact solution in each band (dB; NaN where not measured). ``assumed``: the
    built-in materials of unknown source that took ``simulation.uncertainty.presets``,
    with that choice (:func:`assumed`)."""

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
    assumed: dict[str, str] = dataclasses.field(default_factory=dict)

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
            "assumed": dict(self.assumed),
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
    ``simulation.uncertainty.presets``; else None."""
    material = scene.materials[name]
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


def has_uncertainty(scene, materials: Sequence[str] | None = None) -> bool:
    """Whether some material of ``materials`` (default all of ``scene``'s) has an
    absorption uncertainty a spread takes (:func:`uncertainty_of`)."""
    names = scene.materials if materials is None else materials
    return any(uncertainty_of(scene, name) is not None for name in names)


def absorption_draws(scene, materials: Sequence[str], draws: int, seed: int) -> np.ndarray:
    """(draws, materials, bands) the absorption of ``materials`` in each draw: each
    material's own, plus its uncertainty times one deviate per draw that its bands share
    (standard normal, or uniform on -1 to 1 for a tolerance); those without an
    uncertainty (:func:`uncertainty_of`) keep their own. Held to 0 to
    ``MAX_ABSORPTION``."""
    return _draws(scene, materials, draws, seed)[0]


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
        width = np.asarray(spec.absorption, dtype=float)
        if spec.distribution == "uniform":
            deviate = rng.uniform(-1.0, 1.0, draws)
        else:
            deviate = rng.standard_normal(draws)
        out.append(np.clip(own + deviate[:, None] * width, 0.0, MAX_ABSORPTION))
        deviates.append(deviate)
        widths.append(width)
    return np.stack(out, axis=1), np.stack(deviates, axis=1), np.stack(widths)


class _Admittances:
    """The admittance of each draw's absorption for the phases of the paths that add in
    pressure (:func:`~acoustic_spaces.engine.coherent.absorption_phases`): for a
    material's draws, fitted when first asked for at ``ADMITTANCE_NODES`` deviates over
    their range (:func:`~acoustic_spaces.engine.impedance.fit_admittance`, about a second
    each) and taken between the two nearest, as fitting each draw's would take that for
    every draw. Other rows (as traced) are fitted as the trace fits them."""

    def __init__(self, traced, taken, deviates, widths) -> None:
        self._traced, self._widths = traced, widths
        self._nodes: dict[int, np.ndarray] = {}
        self._fits: dict[int, list] = {}
        self._found: dict[bytes, tuple[int, int, float]] = {}
        for k in range(traced.shape[0]):
            if not np.any(widths[k]):
                continue
            nodes = np.linspace(deviates[:, k].min(), deviates[:, k].max(), ADMITTANCE_NODES)
            self._nodes[k] = nodes
            for drawn, deviate in zip(taken[:, k], deviates[:, k], strict=True):
                j = int(np.clip(np.searchsorted(nodes, deviate) - 1, 0, len(nodes) - 2))
                t = float((deviate - nodes[j]) / (nodes[j + 1] - nodes[j]))
                self._found[np.ascontiguousarray(drawn).tobytes()] = (k, j, t)

    def __call__(self, row: np.ndarray):
        from acoustic_spaces.engine.impedance import fit_admittance

        total = self._traced.shape[1]
        found = self._found.get(np.ascontiguousarray(row, dtype=float).tobytes())
        if found is None:
            return fit_admittance(row.tolist(), total)
        k, j, t = found
        if k not in self._fits:
            own, width = self._traced[k], self._widths[k]
            self._fits[k] = [
                fit_admittance(np.clip(own + node * width, 0.0, MAX_ABSORPTION), total)
                for node in self._nodes[k]
            ]
        low, high = self._fits[k][j], self._fits[k][j + 1]
        return lambda f: (1.0 - t) * low(f) + t * high(f)


def material_spread(
    result,
    scene,
    face_group: np.ndarray,
    materials: Sequence[str],
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    coverage: float = 0.95,
    rays: bool = True,
    progress=None,
) -> dict:
    """{(source, listener): :class:`Spread`} the spread of ``result``'s parameters that
    the uncertainty of its materials' absorption leads to, with the rays' random error
    (unless ``rays`` is False). ``result`` is ``scene`` simulated with ``face_group`` for
    ``materials`` (:func:`~acoustic_spaces.engine.reweight.groups_of` and
    ``simulate(groups=...)``). ``progress(message)``, if given, is called before each
    response's draws and every ``PROGRESS_DRAWS`` draws; an exception it raises (a
    cancel) stops the spread."""
    if not 0.0 < coverage < 1.0:
        raise ValueError("the coverage probability lies between 0 and 1")
    if not has_uncertainty(scene, materials):
        raise ValueError("no material on the surfaces has an absorption uncertainty")
    traced = np.array([scene.materials[n].coefficients("octave")[0] for n in materials])
    taken, deviates, widths = _draws(scene, materials, draws, seed)
    ratios = [log_ratio(traced, drawn) for drawn in taken]
    admittance_of = _Admittances(traced, taken, deviates, widths)
    # The rays' deviates from a stream of their own, so that the absorption's draws stay
    # those of ``seed``.
    rng = np.random.default_rng([seed, 1])
    defaults = assumed(scene, materials)
    say = progress or (lambda _: None)
    out = {}
    for n, (key, response) in enumerate(result.responses.items(), start=1):
        found = []
        for k, (ratio, drawn) in enumerate(zip(ratios, taken, strict=True)):
            if k % PROGRESS_DRAWS == 0:
                say(
                    f"the spread: {key[0]} -> {key[1]} ({n} of {len(result.responses)}), "
                    f"draw {k + 1} of {draws}"
                )
            found.append(
                response.reweighted(
                    face_group, ratio, absorption=(traced, drawn), admittance_of=admittance_of
                ).parameters()
            )
        batches = len(response.late_batches) if rays else 0
        errors = response.standard_errors() if batches >= MIN_BATCHES else None
        spread = _spread(found, coverage, errors, 0 if errors is None else batches, rng)
        out[key] = dataclasses.replace(
            spread, not_counted=edge_errors(response, scene), assumed=defaults
        )
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


def with_spread(
    result,
    scene,
    face_group,
    materials,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    progress=None,
):
    """``result`` with its ``spreads`` (:func:`material_spread`, which ``progress`` goes
    to), or as it is when no material on the surfaces has an absorption uncertainty
    (:func:`has_uncertainty`)."""
    if not has_uncertainty(scene, materials):
        return result
    spreads = material_spread(
        result, scene, face_group, materials, draws=draws, seed=seed, progress=progress
    )
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
    each = {part: RoomParameters(**values) for part, values in parts.items()}
    return Spread(
        mean=each["mean"],
        standard=each["standard"],
        low=each["low"],
        high=each["high"],
        absorption=each["absorption"],
        rays=each["rays"],
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
