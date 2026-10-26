"""Re-weighting a traced response for new absorption, without tracing it again.

The energy of every arrival is what left the source that way times, at each reflection,
the share the surface reflects, 1 - a, and factors that do not depend on the absorption
(scattering, air, the source's directivity, what a partition lets through). With other
absorption a', the same paths bring each arrival times the product over its reflections
of (1 - a') / (1 - a). The paths themselves need not be traced again: where a ray goes
depends on the scattering, not on the absorption. (Russian roulette and splitting chose
rays by their energies, but those choices only weight the estimate; it stays unbiased.)

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
``groups``). With l_i = ln((1 - a'_i) / (1 - a_i)) for group i, a slot's energy becomes

    E exp(sum_i mu_i l_i + 1/2 sum_ij C_ij l_i l_j),

mu and C the energy-weighted mean and covariance of the counts in the slot: the
second-order cumulant expansion of the sum over arrivals of E_r exp(n_r . l). It is exact
for the same absorption and to second order around it; after a change of a few tenths
the tail errs by some per cent, and :func:`~acoustic_spaces.calibration.calibrate`
traces again at the absorption it arrives at. Its fit within a round keeps the early
paths' phases as traced, which that next trace turns.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from acoustic_spaces.engine.coherent import absorption_phases, sample_frequencies
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.late import MAX_GROUPS, MOMENT_STEP


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


def groups_of(room) -> tuple[np.ndarray, list[str]] | None:
    """Face groups to trace ``room`` with so that its absorption can change afterwards
    (:func:`reweighted_result`): the group of each face and the materials, those on
    surfaces that absorb less than everything in every band. None when its simulation
    settings do not keep what re-weighting needs (another late method, third-octave
    bands, the wave solver) or more than ``MAX_GROUPS`` materials are on its surfaces."""
    sim = room.scene.simulation
    if sim.late.method != "stochastic_ray_tracing" or sim.bands != "octave" or sim.wave.enabled:
        return None
    used = np.unique(np.asarray(room.face_materials))
    names = [room.material_names[k] for k in used if k >= 0]
    materials = [n for n in names if max(room.scene.materials[n].coefficients("octave")[0]) < 1]
    if not materials or len(materials) > MAX_GROUPS:
        return None
    return face_groups(room, materials), materials


def _without_absorption(scene) -> dict:
    data = scene.model_dump()
    for material in data["materials"].values():
        for key in ("absorption", "source", "preset", "file"):
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
    if np.any(fresh >= 1.0):
        return None
    traced_absorption = np.array([old[n] for n in materials])
    ratio = log_ratio(traced_absorption, fresh)
    responses = {
        key: r.reweighted(face_group, ratio, absorption=(traced_absorption, fresh))
        for key, r in result.responses.items()
    }
    # The runs without the removed parts and in the weather classes are not re-weighted:
    # their insertion loss and the long-term levels go.
    return dataclasses.replace(
        result, responses=responses, removed=(), insertion_loss={}, long_term=None
    )


def log_ratio(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    """(groups, bands) ln((1 - new) / (1 - old)): the change in level per reflection."""
    old, new = (np.asarray(a, dtype=float) for a in (old, new))
    if np.any(old >= 1.0) or np.any(new >= 1.0):
        raise ValueError("re-weighting needs absorption below 1")
    return np.log1p(-new) - np.log1p(-old)


def path_counts(paths: EarlyReflections, face_group: np.ndarray, groups: int) -> np.ndarray:
    """(paths, groups) how often each early path reflects off each group of faces."""
    counts = np.zeros((len(paths), groups))
    for k, faces in enumerate(paths.faces):
        found = face_group[np.asarray(faces, dtype=np.int64)]
        np.add.at(counts[k], found[found >= 0], 1.0)
    return counts


def moment_exponent(moments: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """(bands, slots) the exponent by which each slot of late energy changes, from the
    moments of the counts (terms, bands, slots) and ``log_ratio`` (groups, bands)."""
    groups = ratio.shape[0]
    total = moments[0]
    safe = np.where(total > 0.0, total, 1.0)
    mean = moments[1 : 1 + groups] / safe  # (groups, bands, slots)
    exponent = np.einsum("ibs,ib->bs", mean, ratio)
    t = 1 + groups
    for i in range(groups):
        for j in range(i, groups):
            covariance = moments[t] / safe - mean[i] * mean[j]
            weight = 0.5 if i == j else 1.0  # C_ij and C_ji both
            exponent += weight * covariance * (ratio[i] * ratio[j])[:, None]
            t += 1
    return np.where(total > 0.0, exponent, 0.0)


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
    paths: EarlyReflections, face_group: np.ndarray, old: np.ndarray, new: np.ndarray
) -> EarlyReflections:
    """``paths`` with the phase of each reflection off a group of faces (``face_group``)
    turned, in the bands where they add in pressure, from that of the group's absorption
    ``old`` (groups, bands) to that of ``new``: from the angle each meets its face at
    (:attr:`~acoustic_spaces.engine.early.EarlyReflections.cosines`). Reflections whose
    phase their face's absorption does not give keep theirs."""
    if paths.spectrum is None or len(paths) == 0 or not paths.cosines:
        return paths
    old, new = (np.asarray(a, dtype=float) for a in (old, new))
    changed = np.any(old != new, axis=1)
    rows, groups, cosines = [], [], []
    for k, (faces, angles) in enumerate(zip(paths.faces, paths.cosines, strict=True)):
        group = face_group[np.asarray(faces, dtype=np.int64)]
        mine = group >= 0
        mine[mine] = changed[group[mine]]
        mine &= np.isfinite(angles)
        rows.extend([k] * int(mine.sum()))
        groups.extend(group[mine])
        cosines.extend(angles[mine])
    if not rows:
        return paths
    groups = np.asarray(groups, dtype=np.int64)
    f = sample_frequencies(old.shape[1], paths.spectrum.shape[1])
    turn = absorption_phases(new[groups], cosines, f) * np.conj(
        absorption_phases(old[groups], cosines, f)
    )
    spectrum = paths.spectrum.copy()
    np.multiply.at(spectrum, np.asarray(rows), turn)
    return dataclasses.replace(paths, spectrum=spectrum)


__all__ = [
    "MOMENT_STEP",
    "face_groups",
    "groups_of",
    "log_ratio",
    "moment_exponent",
    "path_counts",
    "reweighted_paths",
    "reweighted_result",
    "slot_factor",
    "turned_paths",
]
