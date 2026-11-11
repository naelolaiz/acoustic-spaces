"""Re-weighting a traced response for new absorption, without tracing it again.

The energy of every arrival is what left the source that way times, at each reflection,
the share the surface reflects, 1 - a, and factors that do not depend on the absorption
(scattering, air, the source's directivity, what a partition lets through). With other
absorption a', the same paths bring each arrival times the product over its reflections
of (1 - a') / (1 - a). The paths themselves need not be traced again: where a ray goes
depends on the scattering, not on the absorption. (Russian roulette and splitting chose
rays by their energies, but those choices only weight the estimate; it stays unbiased.)
An early reflection off a ground with an impedance is not of this kind: it takes the
impedance's reflection factor in place of 1 - a, and that impedance cannot change
without tracing again (:func:`reweighted_result`). Nor quite one off a ground with
zones, which takes the factors of the grounds its Fresnel zone covers and is
re-weighted as off the ground it meets alone. With source and listener 0.5 and 1.5 m
high and 30 m apart, and a zone without an impedance taken from an absorption of 0.1 to
0.2, a reflection half a metre inside the zone's edge falls by 0.5 dB re-weighted and
by 0.13 to 0.20 dB traced again; half a metre outside it, by none and by 0.11 to
0.16 dB.

Early paths carry the faces they reflect off, so their energies re-weight exactly. Where
they add in pressure (:mod:`~acoustic_spaces.engine.coherent`), a reflection also
shifts the phase, by that of its reflection factor, which comes from the admittance
fitted to its face's absorption and the angle it meets the face at; the paths keep
those angles (:attr:`~acoustic_spaces.engine.early.EarlyReflections.cosines`), so their
phases turn to the new absorption's exactly too (:func:`turned_paths`). Of a diffracted
path only the reflections on its way count, not the faces of the edge it bends round,
nor how it interferes with the reflection it shades. The late part's arrivals are far
too many to keep, so the tracer sums, per listener and slot
of ``MOMENT_STEP``, the energy times the reflections each ray has had off each group of
faces and their products (:func:`~acoustic_spaces.engine.late.stochastic_energy` with
``groups``): the energy-weighted mean mu and covariance C of the counts n in the slot.
With l_i = ln((1 - a'_i) / (1 - a_i)) for group i, a slot's energy becomes E exp(K),
K = ln of the mean of exp(n . l) over its arrivals, which mu and C give to second order
in l as mu . l + 1/2 l' C l. That form alone holds near the traced absorption only: for
much more absorption its square term outgrows the first and the energy rises again,
without bound. :func:`moment_exponent` takes instead the form the counts of a diffuse
field lead to. There a ray's N reflections fall on the groups by chance (a multinomial),
and with u_i = exp(l_i) - 1, the change of what group i reflects,

    K = N ln(1 + sum_i mu_i u_i / N),

which is Eyring's decay for the new absorption over that for the old. N comes from the
covariance: 1 / N = -mu' F mu / (mu . mu)^2 with F = C - diag(mu), which is -mu mu' / N
for a multinomial. It is held within plus and minus one over the sum of mu: a ray has
had at least the reflections the groups count, and counts that spread wider than by
chance make it negative. What of the covariance it leaves, R = F + mu mu' / N, adds
1/2 w' R w, with w_i = u_i for more absorption and l_i for less. K is then the same to
second order in l as the form above, exact for the same absorption and for counts that
fall by chance, and held to what holds for any counts: at least mu . l (Jensen's
inequality), and no gain where no group reflects more. Further from the traced
absorption it stays an estimate from two moments. In a box of 8 by 6 by 3.5 m traced
with a ceiling that absorbs 0.5 (20 000 rays, at 1 kHz), the late part's decay time for
a ceiling of 0.9 is within 1 % of tracing again, for 0.99 within 5 %, and for 0.1 it is
6 % too short; with a ceiling of 0.85 taken to 0.2, 16 % too short. So
:func:`~acoustic_spaces.calibration.calibrate` traces again at the absorption it arrives
at. Its fit within a round keeps the early paths' phases as traced, which that next
trace turns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from acoustic_spaces.engine.coherent import SameDelays, absorption_phases, sample_frequencies
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.late import MAX_GROUPS, MOMENT_STEP
from acoustic_spaces.materials.bands import THIRD_OCTAVE_BANDS_HZ


def face_groups(room, materials: Sequence[str]) -> np.ndarray:
    """(faces,) the index in ``materials`` of each face's material, or -1 for faces of
    other materials."""
    names = list(room.material_names)
    unknown = [m for m in materials if m not in names]
    if unknown:
        raise ValueError(
            f"no surface has the material {', '.join(unknown)}; "
            f"the model's materials are: {', '.join(names)}"
        )
    if len(set(materials)) != len(materials):
        raise ValueError("each material may be named once")
    index = {name: k for k, name in enumerate(materials)}
    lookup = np.array([index.get(name, -1) for name in names] + [-1], dtype=np.int64)
    return lookup[np.asarray(room.face_materials)]


def groups_of(room, settings=None) -> tuple[np.ndarray, list[str]] | None:
    """Face groups to trace ``room`` with so that its absorption can change afterwards
    (:func:`reweighted_result`): the group of each face and the materials, those on
    surfaces that absorb less than everything in some band (a band in which one absorbs
    everything keeps that: no energy is left there to re-weight). None when its
    simulation settings (``settings``, default the scene's) do not keep what
    re-weighting needs (:func:`cannot_reweight`) or no material is left."""
    if cannot_reweight(room, settings) is not None:
        return None
    materials = reweighted_materials(room)
    return (face_groups(room, materials), materials) if materials else None


def reweighted_materials(room) -> list[str]:
    """The materials on ``room``'s surfaces that absorb less than everything in some
    band: those whose absorption a re-weighting can change."""
    used = np.unique(np.asarray(room.face_materials))
    names = [room.material_names[k] for k in used if k >= 0]
    return [n for n in names if min(room.scene.materials[n].coefficients("octave")[0]) < 1]


def cannot_reweight(room, settings=None) -> str | None:
    """Why a run of ``room`` with ``settings`` (default the scene's) cannot be
    re-weighted for other absorption, in words that follow "it needs", or None when it
    can: another late method, third-octave bands, the wave solver, or more than
    ``MAX_GROUPS`` materials on its surfaces."""
    sim = room.scene.simulation if settings is None else settings
    if sim.late.method != "stochastic_ray_tracing":
        return "stochastic ray tracing for the late part"
    if sim.bands != "octave":
        return "octave bands"
    if sim.wave.enabled:
        return "a run without the wave solver"
    count = len(reweighted_materials(room))
    if count > MAX_GROUPS:
        return f"at most {MAX_GROUPS} materials on the surfaces, and there are {count}"
    return None


def with_absorption(
    scene, materials: Sequence[str], absorption, source: str, digits: int | None = 4
):
    """A copy of ``scene`` whose ``materials`` take ``absorption`` (for each, its octave
    bands), their other properties kept. A material given in third octaves keeps its
    shape within each octave: its three values move together, by what the octave does;
    one that would pass 1, or fall below what the surface lets through, stays there, and
    the octave's others take up what it could not, so that the octave is what was asked.
    Given its third-octave values instead, it takes them as they are. Each one's source
    then reads ``source``, with what it was after it; the values are rounded to
    ``digits`` decimals (None: as given)."""
    out = scene.model_copy(deep=True)
    for name, values in zip(materials, absorption, strict=True):
        values = np.asarray(values, dtype=float)
        material = out.materials[name]
        old = np.asarray(material.absorption, dtype=float)
        if len(old) == len(THIRD_OCTAVE_BANDS_HZ) and len(values) != len(old):
            passing = material.transmission_coefficients("third_octave")
            least = np.zeros(len(old)) if passing is None else np.asarray(passing, dtype=float)
            values = _thirds_at(old.reshape(-1, 3), values, least.reshape(-1, 3)).ravel()
        was = f"; was: {material.source}" if material.source else ""
        out.materials[name] = material.model_copy(
            update={
                "absorption": [
                    float(v) if digits is None else round(float(v), digits) for v in values
                ],
                "source": f"{source}{was}",
            }
        )
    return out


def _thirds_at(thirds: np.ndarray, octaves: np.ndarray, least: np.ndarray) -> np.ndarray:
    """The third-octave values ``thirds`` (octaves, 3) moved together so that each
    octave's mean is that of ``octaves``, each held between its ``least`` and 1. Where
    one is held, the octave's others take up evenly what it could not, as far as their
    own limits let them."""
    shifted = thirds + (octaves - thirds.mean(axis=1))[:, None]
    moved = np.clip(shifted, least, 1.0)
    held = np.any(moved != shifted, axis=1)
    for _ in range(2):
        short = np.where(held, octaves - moved.mean(axis=1), 0.0)
        free = np.where(short[:, None] > 0.0, moved < 1.0, moved > least)
        count = free.sum(axis=1)
        each = np.divide(3.0 * short, count, out=np.zeros_like(short), where=count > 0)
        moved = np.clip(moved + np.where(free, each[:, None], 0.0), least, 1.0)
    return moved


def _without_absorption(scene) -> dict:
    data = scene.model_dump()
    for material in data["materials"].values():
        for key in ("absorption", "source", "preset", "file", "uncertainty"):
            material.pop(key, None)
    return data


def reweighted_result(result, traced, face_group: np.ndarray, materials: Sequence[str], scene):
    """``result``, simulated from the scene ``traced`` with ``face_group`` for
    ``materials`` (:func:`groups_of`), for the absorption of ``scene``: re-weighted,
    without tracing again. None when ``scene`` differs from ``traced`` in more than the
    absorption of its materials, or a material re-weighting covers changed."""
    if scene is traced:
        return result
    if _without_absorption(scene) != _without_absorption(traced):
        return None
    old = {n: m.coefficients("octave")[0] for n, m in traced.materials.items()}
    new = {n: m.coefficients("octave")[0] for n, m in scene.materials.items()}
    if any(old[n] != new[n] for n in old if n not in materials):
        return None
    fresh = np.array([new[n] for n in materials])
    traced_absorption = np.array([old[n] for n in materials])
    # A band that absorbs everything has no energy to re-weight, to it or from it.
    if np.any((fresh >= 1.0) != (traced_absorption >= 1.0)):
        return None
    ratio = log_ratio(traced_absorption, fresh)
    responses = {
        key: r.reweighted(face_group, ratio, absorption=(traced_absorption, fresh))
        for key, r in result.responses.items()
    }
    # The runs without the removed parts and in the weather classes are not re-weighted:
    # their insertion loss and the long-term levels go, and so does the spread drawn
    # around the absorption as traced.
    return dataclasses.replace(
        result, responses=responses, removed=(), insertion_loss={}, long_term=None, spreads={}
    )


def log_ratio(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """(groups, bands) ln((1 - new) / (1 - old)): the change in level per reflection;
    0 where both absorb everything, which leaves no energy to change."""
    old, new = (np.asarray(a, dtype=float) for a in (old, new))
    full = old >= 1.0
    if np.any(full != (new >= 1.0)):
        raise ValueError(
            "re-weighting needs absorption below 1, or 1 before and after: a band that "
            "absorbs everything has no energy to re-weight"
        )
    kept = np.where(full, 0.0, new), np.where(full, 0.0, old)
    return np.log1p(-kept[0]) - np.log1p(-kept[1])


def path_counts(paths: EarlyReflections, face_group: np.ndarray, groups: int) -> np.ndarray:
    """(paths, groups) how often each early path reflects off each group of faces."""
    counts = np.zeros((len(paths), groups))
    for k, faces in enumerate(paths.faces):
        found = face_group[np.asarray(faces, dtype=np.int64)]
        np.add.at(counts[k], found[found >= 0], 1.0)
    return counts


def moment_exponent(moments: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """(bands, slots) the exponent by which each slot of late energy changes, from the
    moments of the counts (terms, bands, slots) and ``log_ratio`` (groups, bands): the
    multinomial's form with the rest of the covariance to second order, within the
    bounds that hold for any counts (see the module's text)."""
    groups = ratio.shape[0]
    total = moments[0]
    safe = np.where(total > 0.0, total, 1.0)
    mean = moments[1 : 1 + groups] / safe  # (groups, bands, slots)
    change = np.expm1(ratio)  # u: what each group reflects, over what it did, less 1
    rest_by = np.where(ratio > 0.0, ratio, change)  # w
    shift = np.einsum("ibs,ib->bs", mean, ratio)  # mu . l
    first = np.einsum("ibs,ib->bs", mean, change)  # mu . u
    along = np.einsum("ibs,ib->bs", mean, rest_by)  # mu . w
    count = mean.sum(axis=0)
    square = np.einsum("ibs,ibs->bs", mean, mean)
    # mu' F mu and w' F w, F the covariance less the means on its diagonal.
    spread = np.zeros_like(total)
    rest = np.zeros_like(total)
    t = 1 + groups
    for i in range(groups):
        for j in range(i, groups):
            factorial = moments[t] / safe - mean[i] * mean[j]
            if i == j:
                factorial = factorial - mean[i]
            weight = 1.0 if i == j else 2.0  # F_ij and F_ji both
            spread += weight * factorial * mean[i] * mean[j]
            rest += weight * factorial * (rest_by[i] * rest_by[j])[:, None]
            t += 1
    some = (total > 0.0) & (count > 0.0)
    limit = 1.0 / np.where(some, count, 1.0)
    inverse = np.clip(-spread / np.where(some, square, 1.0) ** 2, -limit, limit)  # 1 / N
    x = inverse * first
    small = np.abs(x) < 1e-6
    exponent = np.where(
        small,
        first * (1.0 - 0.5 * x),
        np.log1p(np.maximum(x, -1.0 + 1e-12)) / np.where(small, 1.0, inverse),
    )
    # Counts spread wider than by chance, and less absorption: the form has a pole
    # there, so its first two terms stand for it.
    exponent = np.where((inverse < 0.0) & (first > 0.0), first * (1.0 - 0.5 * x), exponent)
    exponent += 0.5 * (rest + inverse * along * along)
    exponent = np.maximum(exponent, shift)
    no_gain = np.all(ratio <= 0.0, axis=0)[:, None]
    exponent = np.where(no_gain, np.minimum(exponent, 0.0), exponent)
    return np.where(some, exponent, 0.0)


def slot_factor(exponent: np.ndarray, slots: int, time_step: float, moment_step: float):
    """(bands, slots) the factor on each of ``slots`` slots of ``time_step``, taking that
    of the moment slot it lies in."""
    index = ((np.arange(slots) + 0.5) * time_step / moment_step).astype(np.int64)
    index = np.minimum(index, exponent.shape[1] - 1)
    return np.exp(np.minimum(exponent[:, index], 50.0))


def reweighted_paths(paths: EarlyReflections, counts: np.ndarray, ratio: np.ndarray):
    """``paths`` with their energies for the new absorption (``counts`` from
    :func:`path_counts`, ``ratio`` from :func:`log_ratio`)."""
    if len(paths) == 0:
        return paths
    return dataclasses.replace(paths, energy=paths.energy * np.exp(counts @ ratio))


def turned_paths(
    paths: EarlyReflections,
    face_group: np.ndarray,
    old: np.ndarray,
    new: np.ndarray,
    admittance_of=None,
) -> EarlyReflections:
    """``paths`` with the phase of each reflection off a group of faces (``face_group``)
    turned, in the bands where they add in pressure, from that of the group's absorption
    ``old`` (groups, bands) to that of ``new``: from the angle each meets its face at
    (:attr:`~acoustic_spaces.engine.early.EarlyReflections.cosines`). Reflections whose
    phase their face's absorption does not give keep theirs. ``admittance_of`` gives the
    admittance of an absorption row (:func:`~acoustic_spaces.engine.coherent.
    absorption_phases`; by default the one fitted to it)."""
    if paths.spectrum is None or len(paths) == 0 or not paths.cosines:
        return paths
    old, new = (np.asarray(a, dtype=float) for a in (old, new))
    rows, groups, cosines = _turned_reflections(paths, face_group, np.any(old != new, axis=1))
    if not rows:
        return paths
    f = sample_frequencies(old.shape[1], paths.spectrum.shape[1])
    turn = absorption_phases(new[groups], cosines, f, admittance_of) * np.conj(
        absorption_phases(old[groups], cosines, f, admittance_of)
    )
    return _turned(paths, rows, turn)


def _turned_reflections(
    paths: EarlyReflections, face_group: np.ndarray, changed: np.ndarray
) -> tuple[list[int], np.ndarray, list[float]]:
    """The reflections :func:`turned_paths` turns, those off a group ``changed`` (groups,)
    at a known angle: the path of each, its group and its incidence cosine."""
    rows, groups, cosines = [], [], []
    for k, (faces, angles) in enumerate(zip(paths.faces, paths.cosines, strict=True)):
        group = face_group[np.asarray(faces, dtype=np.int64)]
        mine = group >= 0
        mine[mine] = changed[group[mine]]
        mine &= np.isfinite(angles)
        rows.extend([k] * int(mine.sum()))
        groups.extend(group[mine])
        cosines.extend(angles[mine])
    return rows, np.asarray(groups, dtype=np.int64), cosines


def _turned(paths: EarlyReflections, rows: list[int], turn: np.ndarray) -> EarlyReflections:
    """``paths`` with the spectrum of path ``rows[i]`` times ``turn[i]``, in turn."""
    spectrum = paths.spectrum.copy()
    np.multiply.at(spectrum, np.asarray(rows), turn)
    return dataclasses.replace(paths, spectrum=spectrum)


class Redraws:
    """What re-weighting the early paths ``paths`` of one traced response
    (:meth:`~acoustic_spaces.engine.response.Response.reweighted`) for draw after draw of
    the absorption of ``groups`` groups of faces (``face_group``) takes that does not
    depend on the draw: how often each path reflects off each group
    (:func:`path_counts`); for each set of groups a draw changes, the reflections
    :func:`turned_paths` turns and the phases of the absorption ``old`` as traced that it
    turns them from, with ``admittance_of``; and what the coherent bands' echograms of
    the draws share (``delays``, a :class:`~acoustic_spaces.engine.coherent.SameDelays`).
    A draw comes out the same with it as without it: for other paths, groups, traced
    absorption or admittances than its own, each is worked out as without it. Made from
    its arguments alone, for one response's draws, and dropped after them."""

    def __init__(
        self,
        paths: EarlyReflections,
        face_group: np.ndarray,
        groups: int,
        old: np.ndarray,
        admittance_of=None,
    ) -> None:
        self._paths, self._groups = paths, groups
        self._face_group = np.array(face_group)
        self._old = np.array(old, dtype=float)
        self._admittance_of = admittance_of
        self._counts = path_counts(paths, self._face_group, groups)
        self._counts.flags.writeable = False
        self._turned: dict[bytes, tuple] = {}
        self.delays = SameDelays()

    def _mine(self, paths: EarlyReflections, face_group: np.ndarray) -> bool:
        return paths.faces is self._paths.faces and np.array_equal(face_group, self._face_group)

    def counts(self, paths: EarlyReflections, face_group: np.ndarray, groups: int) -> np.ndarray:
        """:func:`path_counts` of these arguments (read-only)."""
        if paths is self._paths and groups == self._groups and self._mine(paths, face_group):
            return self._counts
        return path_counts(paths, face_group, groups)

    def turned(
        self,
        paths: EarlyReflections,
        face_group: np.ndarray,
        old: np.ndarray,
        new: np.ndarray,
        admittance_of=None,
    ) -> EarlyReflections:
        """:func:`turned_paths` of these arguments, for ``paths`` that are this one's re-weighted
        (:func:`reweighted_paths`)."""
        old, new = (np.asarray(a, dtype=float) for a in (old, new))
        if not (
            paths.cosines is self._paths.cosines
            and paths.spectrum is self._paths.spectrum
            and admittance_of is self._admittance_of
            and old.shape == self._old.shape
            and np.array_equal(old, self._old)
            and self._mine(paths, face_group)
        ):
            return turned_paths(paths, face_group, old, new, admittance_of)
        if paths.spectrum is None or len(paths) == 0 or not paths.cosines:
            return paths
        changed = np.any(old != new, axis=1)
        held = self._turned.get(changed.tobytes())
        if held is None:
            rows, groups, cosines = _turned_reflections(paths, face_group, changed)
            back = None
            f = sample_frequencies(old.shape[1], paths.spectrum.shape[1])
            if rows:
                back = np.conj(absorption_phases(old[groups], cosines, f, admittance_of))
                back.flags.writeable = False
            held = self._turned[changed.tobytes()] = (rows, groups, cosines, f, back)
        rows, groups, cosines, f, back = held
        if not rows:
            return paths
        return _turned(
            paths, rows, absorption_phases(new[groups], cosines, f, admittance_of) * back
        )


__all__ = [
    "MOMENT_STEP",
    "Redraws",
    "cannot_reweight",
    "face_groups",
    "groups_of",
    "log_ratio",
    "moment_exponent",
    "path_counts",
    "reweighted_paths",
    "reweighted_materials",
    "reweighted_result",
    "slot_factor",
    "turned_paths",
]
