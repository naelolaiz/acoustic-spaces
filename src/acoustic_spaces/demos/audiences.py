"""A scene's audience, read from its materials (:func:`audiences`): people standing on a
zone of the ground or seated on the model, as many on each square metre as their
material's preset says, and seats no one sits in. The stills of a demo of any scene draw
it (:mod:`acoustic_spaces.demos.scene_pictures`); a demo asks whether its scene has one
(``--hide audience``) before it draws anything, so this module needs no Pillow."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence

import numpy as np

#: How far apart (m) the dots that stand for an audience are drawn where one person is
#: on each square metre; closer where more are, so each dot is some ten people.
DOT_SPACING_M = 3.2
#: An empty stand's rows of seats are this many times farther apart than the seats
#: along them are drawn.
ROW_RATIO = 1.6
#: People on each square metre of an audience whose material's preset does not say
#: (``audience_1.5_m2``).
AUDIENCE_PER_M2 = 2.0


def _inside(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Which of the points (n, 2) lie in any of the triangles (t, 3, 2)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1 = b - a, c - a
    det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    keep = np.abs(det) > 1e-9
    a, v0, v1, det = a[keep], v0[keep], v1[keep], det[keep]
    d = points[:, None, :] - a[None]
    u = (d[..., 0] * v1[:, 1] - d[..., 1] * v1[:, 0]) / det
    v = (v0[:, 0] * d[..., 1] - v0[:, 1] * d[..., 0]) / det
    return np.any((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9), axis=1)


@dataclasses.dataclass(frozen=True)
class Audience:
    """Where a scene's audience is, or seats no one sits in (:func:`audiences`):
    ``kind`` standing (on a zone of the ground), seated (on the model) or seats (no
    one in them), its material's name, how many people on each square metre, how much
    of the sound at 1 kHz the material takes up, its triangles on plan ((t, 3, 2), m),
    whether its rows run along x (the longer side of its plan), and how many people."""

    kind: str
    material: str
    per_m2: float
    absorbs: float
    triangles: np.ndarray
    along_x: bool
    people: int

    def dots(self, seed: int) -> np.ndarray:
        """Where its dots are drawn (n, 2): people every DOT_SPACING_M or closer, in a
        loose grid, or seats in rows."""
        corners = self.triangles.reshape(-1, 2)
        (x0, y0), (x1, y1) = corners.min(axis=0), corners.max(axis=0)
        per = DOT_SPACING_M**2 / max(self.per_m2, 1.0)  # m² per dot
        if self.kind == "standing":
            step = math.sqrt(per)
            xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
            ys = ys + (np.arange(ys.shape[1]) % 2)[None, :] * step / 2
            points = np.column_stack([xs.ravel(), ys.ravel()])
            points += np.random.default_rng(seed).uniform(-0.3, 0.3, points.shape) * step
        else:
            along = math.sqrt(per / ROW_RATIO)
            across = along * ROW_RATIO
            sx, sy = (along, across) if self.along_x else (across, along)
            xs, ys = np.meshgrid(np.arange(x0 + sx / 2, x1, sx), np.arange(y0 + sy / 2, y1, sy))
            points = np.column_stack([xs.ravel(), ys.ravel()])
        return points[_inside(points, self.triangles)] if len(points) else points


def per_m2_of(entry) -> float:
    """How many people (or seats) are on each square metre of a material of the
    database's ``entry`` (:class:`~acoustic_spaces.materials.database.MaterialEntry`),
    as its description says (``1.5 persons / m2``, ``1 per m2``, ``1.5 m2 per person``,
    chairs ``0,85 m × 0,63 m``), else its category (``2 persons per m2``, ``2 seats per
    m2``); AUDIENCE_PER_M2 where neither does."""
    words = entry.description.replace(",", ".")
    if found := re.search(r"([\d.]+)\s*m2 per person", words):
        return 1.0 / float(found.group(1))
    if found := re.search(r"([\d.]+)\s*(?:persons?\s*/\s*m2|per m2)", words):
        return float(found.group(1))
    if found := re.search(r"([\d.]+)\s*m\s*×\s*([\d.]+)\s*m", words):
        return 1.0 / (float(found.group(1)) * float(found.group(2)))
    if found := re.search(r"([\d.]+)\s*(?:persons|seats) per m2", entry.category):
        return float(found.group(1))
    return AUDIENCE_PER_M2


def audience_of(material) -> tuple[str, object] | None:
    """Whether ``material`` (a :class:`~acoustic_spaces.scene.Material`) is an audience:
    ("people", its database entry or None) where it is of the database's audiences
    (``audience_1.5_m2``, ``audience_upholstered_chairs_1``) or an audience of its own
    (``audience: true``, of no preset of seats), ("seats", its entry) where it is of the
    database's seating (``chairs_unoccupied_plastic``); None where it is neither."""
    from acoustic_spaces.materials.database import builtin_database

    try:
        entry = builtin_database().get(str(material.preset)) if material.preset else None
    except KeyError:
        entry = None
    category = entry.category if entry is not None else ""
    if category.startswith("Seating"):
        return ("seats", entry)
    if category.startswith("Audience") or material.audience:
        return ("people", entry)
    return None


def audiences(room) -> list[Audience]:
    """The audience of ``room``'s scene (a :class:`~acoustic_spaces.room.Room`), by the
    materials of its faces, each part labelled alike (a group of the model, a zone of
    the ground): people where the material is of the database's audiences
    (``audience_1.5_m2``, ``audience_upholstered_chairs_1``) or an audience of its own
    (``audience: true``, of no preset of seats), standing on a zone of the ground or
    seated on the model, as many on each square metre as the preset says
    (:func:`per_m2_of`); seats no one sits in where it is of the database's seating
    (``chairs_unoccupied_plastic``)."""
    mesh, scene = room.mesh, room.scene
    areas = np.asarray(mesh.face_areas())
    out = []
    for k, label in enumerate(mesh.labels):
        faces = np.flatnonzero(mesh.face_labels == k)
        if not len(faces):
            continue
        index = int(room.face_materials[faces[0]])
        if index < 0:
            continue
        name = room.material_names[index]
        material = scene.materials.get(name)
        if material is None:
            continue
        found = audience_of(material)
        if found is None:
            continue
        what, entry = found
        kind = "seats" if what == "seats" else "standing" if "zones" in label else "seated"
        per_m2 = per_m2_of(entry) if entry is not None else AUDIENCE_PER_M2
        triangles = np.asarray(mesh.vertices, dtype=float)[np.asarray(mesh.faces)[faces]][..., :2]
        (x0, y0), (x1, y1) = triangles.reshape(-1, 2).min(0), triangles.reshape(-1, 2).max(0)
        surface = float(areas[faces].sum())
        absorbs = float(material.coefficients("octave")[0][4])  # 1 kHz
        people = 0 if kind == "seats" else round(surface * per_m2)
        out.append(Audience(kind, name, per_m2, absorbs, triangles, x1 - x0 >= y1 - y0, people))
    return out


def about(people: int) -> str:
    """``people``, as many as a still says: to the hundred from a thousand, to ten from
    a hundred."""
    step = -2 if people >= 1000 else -1 if people >= 100 else 0
    return f"{round(people, step):,.0f}"


def audience_lines(found: Sequence[Audience]) -> list[str]:
    """What a still says of the audience ``found``: how many stand and sit and the
    seats no one sits in, then how much of the sound at 1 kHz each takes up; none where
    there is none."""
    words, takes = [], []
    for kind, called in (("standing", "standing"), ("seated", "seated"), ("seats", "empty seats")):
        of = [a for a in found if a.kind == kind]
        if not of:
            continue
        people = sum(a.people for a in of)
        words.append(called if kind == "seats" else f"{about(people)} {called}")
        share = "/".join(f"{a:.0%}" for a in sorted({round(a.absorbs, 2) for a in of}))
        takes.append(f"{called} {share}")
    if not words:
        return []
    return ["Audience: " + ", ".join(words), "Absorbing at 1 kHz: " + ", ".join(takes)]


__all__ = [
    "AUDIENCE_PER_M2",
    "Audience",
    "about",
    "audience_lines",
    "audience_of",
    "audiences",
    "per_m2_of",
]
