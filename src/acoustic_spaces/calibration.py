"""Material calibration: absorption fitted to the parameters measured in a room.

A model of an existing room starts from absorption coefficients out of tables, which
seldom match the room as built. :func:`calibrate` adjusts the absorption of the
materials named, octave band by octave band, until the simulated parameters (T30, EDT
and C80 by default) at each measured source and listener pair come as close as they can
to those measured, in just noticeable differences (ISO 3382-1, annex A; see
:data:`~acoustic_spaces.compare.JND`).

It does not trace the room again for every trial. Each round traces it once, keeping
the reflections each ray had off each material
(:mod:`~acoustic_spaces.engine.reweight`); the responses for other absorption then come
from re-weighting those rays, so a least-squares fit per band
(:func:`scipy.optimize.least_squares`) costs little more than computing the parameters.
The next round traces again at the absorption found, until it moves by less than
``tolerance``; a last trace gives the parameters reported after the fit.

Measurements seldom pin down every material: what they fix best is the total absorption,
and two surfaces that the sound meets in much the same way (a ceiling and a floor of a
flat room, say) trade absorption freely. So the materials keep their proportions unless
the measurements ask otherwise: the fit is free to scale every material's absorption by
a common factor, but a material that moves by a factor more than the others costs as
much as missing a parameter, one JND for each factor of ``PRIOR_FACTOR``. Without
measurements that tell the materials apart this is the fit of a single factor per band,
as fitting a diffuse-field formula to the reverberation time does.
:attr:`Calibration.uncertainty` says, per material and band, how far its absorption
could move, the other materials following it, before the sum of the squared misses
grows by one JND squared (its standard error): what the measurements leave open.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from acoustic_spaces import parameters as iso3382
from acoustic_spaces.compare import JND
from acoustic_spaces.engine import reweight
from acoustic_spaces.engine.coherent import EarlyPressure, filtered_bands
from acoustic_spaces.engine.late import MAX_GROUPS
from acoustic_spaces.engine.response import Response
from acoustic_spaces.engine.simulate import _as_room, simulate
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import RoomParameters
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Scene

#: The parameters fitted unless others are named.
DEFAULT_PARAMETERS: tuple[str, ...] = ("t30", "edt", "c80")
#: A material whose absorption moves by this factor more than the others' does costs as
#: much as missing one parameter by one JND.
PRIOR_FACTOR = 2.0
#: The step (in absorption) of the finite differences behind the fit and its
#: uncertainty.
_STEP = 0.01

Pair = tuple[str, str]


@dataclass
class Calibration:
    """What :func:`calibrate` found: the absorption per material and octave band, and
    the parameters before and after, against those measured."""

    scene: Scene  # the scene as it was
    materials: list[str]
    parameters: tuple[str, ...]
    initial: np.ndarray  # (materials, bands)
    absorption: np.ndarray  # (materials, bands), fitted
    #: (materials, bands) the standard error of each material's absorption, for misses
    #: of one JND: how far it can move, the others following it, before the sum of the
    #: squared misses grows by one JND squared. It is what the measurements leave open.
    #: Above 0.1 or so they hardly constrain it.
    uncertainty: np.ndarray
    measured: dict[Pair, RoomParameters]
    before: dict[Pair, RoomParameters]
    after: dict[Pair, RoomParameters]
    rounds: int  # traces at new absorption, the last one included
    bands_hz: tuple[int, ...] = OCTAVE_BANDS_HZ
    details: dict[str, float] = field(default_factory=dict)

    def fitted_scene(self) -> Scene:
        """A copy of the scene with the fitted absorption."""
        return with_absorption(self.scene, self.materials, self.absorption)

    def errors(self, which: str = "after") -> dict[str, np.ndarray]:
        """Per fitted parameter, the mean absolute difference from the measurements in
        JNDs per band over the pairs (NaN where no pair has both values); ``which`` is
        "before" or "after"."""
        simulated = {"before": self.before, "after": self.after}[which]
        out = {}
        for name in self.parameters:
            rows = [
                np.abs(_jnd_differences(name, getattr(simulated[p], name), getattr(m, name)))
                for p, m in self.measured.items()
            ]
            with np.errstate(invalid="ignore"):
                stacked = np.array(rows)
                count = np.isfinite(stacked).sum(axis=0)
                total = np.nansum(stacked, axis=0)
                out[name] = np.where(count > 0, total / np.maximum(count, 1), math.nan)
        return out

    def table(self) -> str:
        """The fitted absorption, what it started from, its uncertainty and the errors
        before and after, as text."""
        bands = "".join(f"{iso3382._band_label(f):>8}" for f in self.bands_hz)
        width = max(12, *(len(m) for m in self.materials))
        lines = [f"{'absorption':<{width}}{bands}"]
        for k, name in enumerate(self.materials):
            lines.append(f"{name:<{width}}" + "".join(f"{v:8.2f}" for v in self.absorption[k]))
            was = "".join(f"{v:8.2f}" for v in self.initial[k])
            lines.append(f"{'  was':<{width}}{was}")
            marks = "".join(
                f"{v:8.2f}" if np.isfinite(v) and v < 10 else f"{'-':>8}"
                for v in self.uncertainty[k]
            )
            lines.append(f"{'  +-':<{width}}{marks}")
        lines.append("")
        lines.append(f"{'mean |error|':<{width}}{bands}   (JND)")
        before, after = self.errors("before"), self.errors("after")
        for name in self.parameters:
            for label, errors in (("before", before), ("after", after)):
                row = "".join(f"{v:8.2f}" if np.isfinite(v) else f"{'-':>8}" for v in errors[name])
                lines.append(f"{iso3382._symbol(name) + ' ' + label:<{width}}{row}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        def plain(a: np.ndarray):
            return [None if not math.isfinite(v) else round(float(v), 6) for v in a]

        return {
            "materials": {
                name: {
                    "absorption": plain(self.absorption[k]),
                    "initial": plain(self.initial[k]),
                    "uncertainty": plain(self.uncertainty[k]),
                }
                for k, name in enumerate(self.materials)
            },
            "parameters": list(self.parameters),
            "bands_hz": list(self.bands_hz),
            "errors_jnd": {
                which: {name: plain(v) for name, v in self.errors(which).items()}
                for which in ("before", "after")
            },
            "rounds": self.rounds,
            "pairs": {
                f"{source} {listener}": {
                    side: {
                        name: plain(getattr(values[source, listener], name))
                        for name in self.parameters
                    }
                    for side, values in (
                        ("measured", self.measured),
                        ("before", self.before),
                        ("after", self.after),
                    )
                }
                for source, listener in self.measured
            },
            "details": self.details,
        }


def with_absorption(scene: Scene, materials: Sequence[str], absorption: np.ndarray) -> Scene:
    """A copy of ``scene`` whose ``materials`` take ``absorption`` (materials, octave
    bands), their other properties kept. A material given in third octaves keeps its
    shape within each octave: its three values move together, by what the octave does."""
    out = scene.model_copy(deep=True)
    for name, values in zip(materials, np.asarray(absorption, dtype=float), strict=True):
        material = out.materials[name]
        old = np.asarray(material.absorption, dtype=float)
        if len(old) == len(THIRD_OCTAVE_BANDS_HZ):
            passing = material.transmission_coefficients("third_octave")
            least = 0.0 if passing is None else np.asarray(passing)
            shift = values - old.reshape(-1, 3).mean(axis=1)
            values = np.clip(old + np.repeat(shift, 3), least, 1.0)
        was = f"; was: {material.source}" if material.source else ""
        out.materials[name] = material.model_copy(
            update={
                "absorption": [round(float(v), 4) for v in values],
                "source": f"absorption calibrated to measurements{was}",
            }
        )
    return out


def _jnd_differences(name: str, simulated, measured) -> np.ndarray:
    """(simulated - measured) in JNDs, NaN where either is missing."""
    s, m = (np.asarray(v, dtype=float) for v in (simulated, measured))
    step, relative = JND[name]
    jnd = step * np.abs(m) if relative else np.full(m.shape, step)
    out = np.full(m.shape, math.nan)
    ok = np.isfinite(s) & np.isfinite(m) & (jnd > 0)
    out[ok] = (s[ok] - m[ok]) / jnd[ok]
    return out


@dataclass
class _Band:
    """One response's energy in one band, ready to re-weight."""

    slots: int
    time_step: float
    onset: float
    reference: float
    early_slot: np.ndarray  # (paths,)
    early_energy: np.ndarray  # (paths,)
    counts: np.ndarray  # (paths, groups)
    late: np.ndarray  # (slots,)
    mean: np.ndarray  # (groups, moment slots)
    second: np.ndarray  # (groups, groups, moment slots): E[n_i n_j]
    moment_slot: np.ndarray  # (slots,) the moment slot of each slot
    #: Where the early paths add in pressure (:mod:`~acoustic_spaces.engine.coherent`):
    #: the reflection counts (U, groups) the paths come in, the pressure of each set
    #: of paths alike through the band's filter (U, samples), the slot of each sample
    #: and the factor from squared pressure to energy; else None.
    pressure: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

    @classmethod
    def of(cls, response: Response, band: int, groups: int, face_group: np.ndarray) -> _Band:
        step = response.time_step
        slots = int(math.ceil(response.length_s / step - 1e-9))
        early = response.early
        index = (early.delay_s / step).astype(np.int64)
        keep = index < slots
        every = reweight.path_counts(early, face_group, groups)
        counts = every[keep]
        pressure = None
        if band < filtered_bands(early) and len(early):
            # Paths that reflect alike re-weight alike: their pressure is summed once.
            alike, label = np.unique(every, axis=0, return_inverse=True)
            weights = np.zeros((len(early), len(alike)))
            weights[np.arange(len(early)), label.reshape(-1)] = 1.0
            signals = EarlyPressure(early, weights).band(
                band, step, slots, start=response.direct_delay_s
            )
            pressure = (alike, *signals)
        moments = response.late_moments[:, band]  # (terms, moment slots)
        total = moments[0]
        safe = np.where(total > 0.0, total, 1.0)
        mean = np.where(total > 0.0, moments[1 : 1 + groups] / safe, 0.0)
        second = np.zeros((groups, groups, moments.shape[1]))
        t = 1 + groups
        for i in range(groups):
            for j in range(i, groups):
                second[i, j] = second[j, i] = np.where(total > 0.0, moments[t] / safe, 0.0)
                t += 1
        late = np.zeros(slots)
        n = min(slots, response.late.shape[1])
        late[:n] = response.late[band, :n]
        moment_slot = ((np.arange(slots) + 0.5) * step / response.moment_step).astype(np.int64)
        return cls(
            slots=slots,
            time_step=step,
            onset=response.direct_delay_s,
            reference=float(response._reference()[band]),
            early_slot=index[keep],
            early_energy=early.energy[keep, band],
            counts=counts,
            late=late,
            mean=mean,
            second=second,
            moment_slot=np.minimum(moment_slot, moments.shape[1] - 1),
            pressure=pressure,
        )

    def energy(self, ratio: np.ndarray) -> np.ndarray:
        """(slots,) the echogram for the change per reflection ``ratio`` (groups,)."""
        covariance = self.second - self.mean[:, None] * self.mean[None, :]
        exponent = ratio @ self.mean + 0.5 * np.einsum("i,ijs,j->s", ratio, covariance, ratio)
        out = self.late * np.exp(np.minimum(exponent, 50.0))[self.moment_slot]
        if self.pressure is not None:
            alike, pressure, slot, scale = self.pressure
            p = np.exp(0.5 * (alike @ ratio)) @ pressure
            return out + np.bincount(slot, weights=scale * p**2, minlength=self.slots)[: self.slots]
        early = self.early_energy * np.exp(self.counts @ ratio)
        return out + np.bincount(self.early_slot, weights=early, minlength=self.slots)

    def parameters(self, ratio: np.ndarray) -> RoomParameters:
        return iso3382.from_energy(
            self.energy(ratio)[None], self.time_step, self.onset, reference_energy=self.reference
        )


def _initial_absorption(room: Room, materials: Sequence[str]) -> np.ndarray:
    scene = room.scene
    return np.array([scene.materials[m].coefficients("octave")[0] for m in materials])


def _lowest(room: Room, materials: Sequence[str], low: float) -> np.ndarray:
    """(materials, bands) the least absorption each may take: ``low``, or what it lets
    through where that is more (in the third octave of the band that lets most through)."""
    out = np.full((len(materials), len(OCTAVE_BANDS_HZ)), low)
    for k, name in enumerate(materials):
        passing = room.scene.materials[name].transmission_coefficients("third_octave")
        if passing is not None:
            most = np.asarray(passing).reshape(-1, 3).max(axis=1)
            out[k] = np.maximum(out[k], most + 1e-6)
    return out


def _default_materials(room: Room) -> list[str]:
    used = sorted(set(np.asarray(room.face_materials)[np.asarray(room.face_materials) >= 0]))
    names = [room.material_names[k] for k in used]
    # Openings reflect nothing: there is nothing to fit; an impedance gives its own.
    materials = room.scene.materials
    return [
        n
        for n in names
        if min(materials[n].coefficients("octave")[0]) < 1.0 and materials[n].impedance is None
    ]


def jacobian_of(function, x: np.ndarray, lowest: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Central differences of ``function`` a step of absorption (``_STEP``) apart, kept
    within the bounds. The parameters come from decay curves sampled every millisecond,
    which jump by tiny amounts as slots cross the levels they are read at: smaller steps
    would see only those jumps."""
    columns = []
    for m in range(len(x)):
        up, down = x.copy(), x.copy()
        up[m] = min(x[m] + _STEP, upper[m])
        down[m] = max(x[m] - _STEP, lowest[m])
        columns.append((function(up) - function(down)) / (up[m] - down[m]))
    return np.array(columns).T


def _fit_band(
    prepared: dict[Pair, _Band],
    targets: dict[Pair, list[float]],
    parameters: tuple[str, ...],
    start: np.ndarray,
    prior: np.ndarray,
    lowest: np.ndarray,
    high: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """One band's absorption per material, re-weighting the rays traced at ``start``,
    and its uncertainty; None when nothing was measured in the band."""
    from scipy.optimize import least_squares

    def misses(x: np.ndarray) -> np.ndarray:
        ratio = np.log1p(-x) - np.log1p(-start)
        out = []
        for p, band in prepared.items():
            found = band.parameters(ratio)
            for name, target in zip(parameters, targets[p], strict=True):
                value = float(getattr(found, name)[0])
                step, relative = JND[name]
                jnd = step * abs(target) if relative else step
                if math.isfinite(target) and jnd > 0:
                    # A response too short or too flat for the parameter misses it by much.
                    out.append((value - target) / jnd if math.isfinite(value) else 10.0)
        return np.array(out)

    if len(misses(start)) == 0:
        return None

    reference = np.log(np.maximum(prior, lowest))

    def residuals(x: np.ndarray) -> np.ndarray:
        scaled = np.log(x) - reference
        return np.concatenate([misses(x), (scaled - scaled.mean()) / math.log(PRIOR_FACTOR)])

    upper = np.full(len(start), high)

    x0 = np.clip(start, lowest, upper)
    solution = least_squares(
        residuals,
        x0,
        jac=lambda x: jacobian_of(residuals, x, lowest, upper),
        bounds=(lowest, upper),
    )
    # The standard error of each material, for misses of one JND: how far it can move,
    # the others following, before the sum of the squared misses grows by one JND squared.
    data = jacobian_of(misses, solution.x, lowest, upper)
    try:
        spread = np.sqrt(np.maximum(np.diag(np.linalg.inv(data.T @ data)), 0.0))
    except np.linalg.LinAlgError:
        spread = np.full(len(start), math.inf)
    return solution.x, spread


def calibrate(
    room: Room | Scene | str | Path,
    measured: Mapping[Pair, RoomParameters],
    materials: Sequence[str] | None = None,
    *,
    parameters: Sequence[str] = DEFAULT_PARAMETERS,
    rounds: int = 4,
    bounds: tuple[float, float] = (0.01, 0.99),
    tolerance: float = 0.01,
    threads: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Calibration:
    """Fit the absorption of ``materials`` (default: every material on a surface but
    openings, at most 16) so that the simulated ``parameters`` (edt, t20, t30, c50, c80,
    d50, ts and g) match those ``measured`` at each (source, listener) pair of the scene,
    band by band, in JNDs.

    The scene's simulation settings (rays, response length, seed, and threads unless
    ``threads`` is given) are used, with stochastic ray tracing in octave bands and no
    wave solver. At most ``rounds`` traces at new absorption are made; it stops sooner
    once no value moves by more than ``tolerance``. Absorption stays within ``bounds``
    (the lower one above 0, as the prior takes the logarithm of the absorption), and
    above what a material lets through.
    """
    room = _as_room(room)
    scene = room.scene
    say = progress or (lambda _: None)
    sim = scene.simulation
    if sim.late.method != "stochastic_ray_tracing":
        raise ValueError("calibration re-weights traced rays: use stochastic ray tracing")
    if sim.bands != "octave":
        raise ValueError("calibration works in octave bands: set simulation bands to octave")
    if sim.wave.enabled:
        raise ValueError("calibration re-weights rays; switch the wave solver off for it")
    if room.outdoor is not None:
        raise ValueError("calibration fits a room's absorption to how it decays; not outdoors")
    # The re-weighted echogram has no directions: no LF, LFC, IACC or directional decay times.
    fitted = [n for n in JND if n in iso3382.PARAMETERS]
    unknown = [n for n in parameters if n not in fitted]
    if unknown:
        raise ValueError(f"cannot fit {', '.join(unknown)}; choose from {', '.join(fitted)}")
    parameters = tuple(parameters)
    pairs = list(measured)
    if not pairs:
        raise ValueError("no measurements to fit")
    for p in pairs:
        for name in parameters:
            if np.shape(getattr(measured[p], name)) != (len(OCTAVE_BANDS_HZ),):
                raise ValueError(
                    f"the measured {name} of {p[0]}/{p[1]} is not in octave bands "
                    f"({len(OCTAVE_BANDS_HZ)} values, 63 Hz to 8 kHz)"
                )
    sources = {s.name for s in scene.sources}
    listeners = {lis.name for lis in scene.listeners}
    missing = [f"{s}/{lis}" for s, lis in pairs if s not in sources or lis not in listeners]
    if missing:
        raise ValueError(f"the scene has no source and listener for {', '.join(missing)}")
    named = materials is not None
    materials = list(materials) if named else _default_materials(room)
    if not materials:
        raise ValueError("no material to fit")
    if len(materials) > MAX_GROUPS:
        where = "named" if named else "on the surfaces"
        raise ValueError(
            f"at most {MAX_GROUPS} materials at once ({len(materials)} {where}); "
            "name the ones to fit"
        )
    reweight.face_groups(room, materials)  # each known, and named once
    grounds = [m for m in materials if scene.materials[m].impedance is not None]
    if grounds:
        raise ValueError(
            f"{', '.join(grounds)} take their absorption from an impedance; calibration "
            "fits absorption coefficients"
        )
    on_faces = {room.material_names[k] for k in np.unique(room.face_materials) if k >= 0}
    unused = [m for m in materials if m not in on_faces]
    if unused:
        raise ValueError(f"no surface has the material {', '.join(unused)}")
    if rounds < 1:
        raise ValueError("calibration needs at least one round")
    low, high = bounds
    if not 0.0 < low < high < 1.0:
        raise ValueError("bounds must be 0 < low < high < 1")
    groups = len(materials)
    initial = _initial_absorption(room, materials)
    lowest = _lowest(room, materials, low)
    current = np.clip(initial, lowest, high)
    used_sources = sorted({s for s, _ in pairs})
    used_listeners = sorted({lis for _, lis in pairs})

    def trace(absorption: np.ndarray, label: str):
        """The responses at ``absorption``, with the absorption traced (rounded, and
        averaged from third octaves)."""
        say(f"Calibration: {label}")
        traced = Room.from_scene(with_absorption(scene, materials, absorption))
        face_group = reweight.face_groups(traced, materials)
        result = simulate(
            traced,
            sources=used_sources,
            listeners=used_listeners,
            threads=threads,
            groups=face_group,
        )
        actual = _initial_absorption(traced, materials)
        return face_group, {p: result[p] for p in pairs}, actual

    def fit(face_group, responses, around: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fitted = around.copy()
        spread = np.full(around.shape, math.nan)
        for b in range(len(OCTAVE_BANDS_HZ)):
            prepared = {p: _Band.of(r, b, groups, face_group) for p, r in responses.items()}
            targets = {
                p: [float(getattr(measured[p], name)[b]) for name in parameters] for p in pairs
            }
            found = _fit_band(
                prepared, targets, parameters, around[:, b], initial[:, b], lowest[:, b], high
            )
            if found is not None:
                fitted[:, b], spread[:, b] = found
        return fitted, spread

    face_group, responses, current = trace(current, "tracing at the starting absorption")
    before = {p: r.parameters() for p, r in responses.items()}
    done = 0
    change = math.nan
    uncertainty = np.full(current.shape, math.nan)
    for k in range(rounds):
        say(f"Calibration: fitting, round {k + 1}")
        fitted, uncertainty = fit(face_group, responses, current)
        change = float(np.max(np.abs(fitted - current)))
        current = fitted
        done += 1
        face_group, responses, current = trace(
            current, f"tracing at the fitted absorption ({done})"
        )
        if change < tolerance:
            break
    after = {p: r.parameters() for p, r in responses.items()}
    return Calibration(
        scene=scene,
        materials=materials,
        parameters=parameters,
        initial=initial,
        absorption=current,
        uncertainty=uncertainty,
        measured=dict(measured),
        before=before,
        after=after,
        rounds=done,
        details={"last_change": change},
    )


__all__ = [
    "DEFAULT_PARAMETERS",
    "PRIOR_FACTOR",
    "Calibration",
    "calibrate",
    "with_absorption",
]
