"""Convert a BRAS room model from SketchUp to OBJ, as the files in this folder were made.

    python validation/bras/convert_skp.py ".../CR2 small room (seminar room)/Geometry/CR2.skp" \
        validation/bras/CR2/CR2.obj

Needs openskp (MIT, the ``sketchup`` extra), which reads SketchUp files without
SketchUp. The simulator reads them with it too, but without the repairs below, which
CR3 needs. The room is written in metres with +z up, in the model's own coordinates
(those of the source and receiver positions in the BRAS SOFA files), one ``usemtl``
per BRAS material (``mat_CR2_plaster`` and so on). Along the way:

- the back faces of the model, which carry no material, are left out;
- corners closer than 12 mm are merged: CR3 has cracks under 1 mm and, on its attic
  floor, a strip 1 cm wide between two walls that leaves the room open; merging
  changes its surface area by 0.2 m2 and its volume by 0.15 m3;
- zero-area faces are dropped: SketchUp adds them where a face's edge runs past the
  corners of the faces on its other side, which the simulator splits instead;
- where three or more faces lie on top of each other, pairs facing opposite ways are
  taken out until one is left (a double-sided strip lying on a surface).

openskp 1.4.0 reads CR1 to CR3; it cannot read CR4 as it is, where each loudspeaker's
text label is anchored to the loudspeaker's component, and :func:`_anchored_texts` mends
that. CR1 with its door at 4.1 degrees reads without the laboratory's wall at
y = -3.65 m, which the model with the door at 30.4 degrees has; ``--fill-from`` takes the
triangles that close such a hole from another model of the same room:

    python validation/bras/convert_skp.py ".../rooms/CR1_DoorAngle1.skp" \
        validation/bras/CR1_DoorAngle1/CR1_DoorAngle1.obj \
        --fill-from validation/bras/CR1_DoorAngle3/CR1_DoorAngle3.obj
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

WELD_M = 0.012
MIN_AREA_M2 = 1e-10


def _anchored_texts() -> None:
    """Mend openskp 1.4.0's reading of a text label anchored to an entity.

    Such a label carries its anchor as a dimension's connection does: [entity ref]
    [entity ref][u32 n][n refs][u32 n][n refs], the second part the instances the
    anchor lives in, where an instance not written yet comes in full (MFC writes an
    object in full the first time anything points at it). openskp skips that part as
    bytes, so the instance's class and object get no store-map slots and every later
    reference to them, or past them, is off: CR4's loudspeaker labels each carry the
    loudspeaker's instance. Here the label reads that part as openskp reads a
    dimension's, and a definition's entity list, whose count also takes in each such
    anchor as an item never written, stops where the definition's tail begins (no
    relationships, then 22 bytes on the string marker of its name).
    """
    import struct

    from openskp import legacy

    marker = legacy._STR_MARKER
    anchored = b"\x02\x00\x00\x00\x04\x00\x00\x00"  # connection type 2, then 4

    def text_string(r, start: int) -> int:
        # the text string comes right after `01 00 00 00 ?? 00 03 00 00 00 01`
        while True:
            at = r.data.find(marker, start, r.pos + 512)
            if at < 0:
                raise legacy.LegacyParseError(f"text delimiter not found {r.ctx()}")
            block = r.data[at - 11 : at]
            if (
                block[:4] == b"\x01\x00\x00\x00"
                and struct.unpack_from("<I", block, 6)[0] <= 8
                and block[10] == 1
            ):
                return at
            start = at + 3

    def read_text(ar, r):
        legacy._preamble(ar, r)
        legacy._drawbase(ar, r)
        ar.read_object(r, expect="CSkFont")
        at = r.data.find(anchored, r.pos, text_string(r, r.pos))
        if at >= 0:
            r.raw(at + 8 + 24 - r.pos)  # the type, 4 and the (zeroed) point
            legacy._entity_ref(ar, r)
            legacy._connection_paths(ar, r)
        r.raw(text_string(r, r.pos) - r.pos)
        text = r.utf16()
        r.raw(5)
        while r.peek(2) == b"\xff\x7f" and not r.data[r.pos + 5] & 0x80:
            r.raw(6)  # leader attachments
        return {"k": "text", "text": text, "attach": [], "p": None, "lp": None}

    entity_list = legacy._read_entity_list_inner

    def read_entity_list(ar, r, count, owner, ents):
        if owner != "def":
            return entity_list(ar, r, count, owner, ents)
        while len(ents) < count:
            p = r.pos
            if struct.unpack_from("<I", r.data, p)[0] == 0 and r.data[p + 22 : p + 25] == marker:
                break
            before = len(ents)
            entity_list(ar, r, before + 1, owner, ents)
            if len(ents) == before:
                break
        return ents

    legacy._READERS["CText"] = legacy._read_text = read_text
    legacy._read_entity_list_inner = read_entity_list


def read_skp(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Corners (metres, +z up), triangles and the material name of each triangle."""
    from openskp import SkpFile

    _anchored_texts()
    skp = SkpFile.open(path)
    model = skp.parse()
    by_colour = {tuple(int(c) for c in m.color[:3]): m.name for m in model.materials}
    scene = skp.build_scene(include_curve_sets=False)
    corners, triangles, names = [], [], []
    for prim in scene.glb_primitives:
        material = scene.gltf_materials[prim.material_index]
        rgba = material.get("pbrMetallicRoughness", {}).get("baseColorFactor", [0, 0, 0])
        name = by_colour.get(tuple(round(c * 255) for c in rgba[:3]))
        if name is None:  # the back faces: SketchUp's default back colour, no material
            continue
        p = np.asarray(prim.positions, dtype=float).reshape(-1, 3)
        t = np.asarray(prim.indices, dtype=int).reshape(-1, 3)
        offset = sum(len(c) for c in corners)
        corners.append(np.column_stack([p[:, 0], -p[:, 2], p[:, 1]]))  # glTF is +y up
        triangles.append(t + offset)
        names += [name] * len(t)
    return np.vstack(corners), np.vstack(triangles), names


def weld(corners: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corners closer than ``WELD_M`` merged into the first of them."""
    keep = np.arange(len(corners))
    order = np.argsort(corners[:, 0], kind="stable")
    xs = corners[order, 0]
    for i in range(len(corners)):
        if keep[i] != i:
            continue
        lo, hi = np.searchsorted(xs, [corners[i, 0] - WELD_M, corners[i, 0] + WELD_M])
        near = order[lo:hi]
        near = near[np.linalg.norm(corners[near] - corners[i], axis=1) < WELD_M]
        near = near[keep[near] == near]
        keep[near] = i
    used, index = np.unique(keep, return_inverse=True)
    return corners[used], index[triangles]


def tidy(corners: np.ndarray, triangles: np.ndarray, names: list[str]):
    """Zero-area triangles dropped, and stacked ones cancelled in opposite pairs."""
    tri = corners[triangles]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    alive = [i for i in range(len(triangles)) if area[i] > MIN_AREA_M2]
    stacks = defaultdict(list)
    for i in alive:
        stacks[tuple(sorted(triangles[i].tolist()))].append(i)
    drop = set()
    for stack in stacks.values():
        if len(stack) < 3:
            continue
        # The same three corners in a cyclic order face one way; reversed, the other.
        a, b, c = triangles[stack[0]]
        same = {(a, b, c), (b, c, a), (c, a, b)}
        ways = {True: [], False: []}
        for i in stack:
            ways[tuple(triangles[i].tolist()) in same].append(i)
        for one, other in zip(ways[True], ways[False], strict=False):
            drop.update((one, other))
    kept = [i for i in alive if i not in drop]
    return triangles[kept], [names[i] for i in kept]


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Corners, triangles and material names of an OBJ file this script wrote."""
    corners, triangles, names, material = [], [], [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            corners.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("usemtl "):
            material = line.split()[1]
        elif line.startswith("f "):
            triangles.append([int(v) - 1 for v in line.split()[1:4]])
            names.append(material)
    return np.array(corners), np.array(triangles, dtype=int), names


def open_edges(triangles: np.ndarray) -> set[tuple[int, int]]:
    """The edges only one triangle has."""
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    return {(int(a), int(b)) for a, b in unique[count == 1]}


def fill_from(corners: np.ndarray, triangles: np.ndarray, names: list[str], other: Path):
    """Add the triangles of ``other``, a model of the same room, that close holes here:
    those with two of their three edges on open edges here."""
    edges = open_edges(triangles)
    there, their_triangles, their_names = read_obj(other)
    gap = np.linalg.norm(there[:, None, :] - corners[None, :, :], axis=2)
    nearest = np.where(gap.min(axis=1) < WELD_M, gap.argmin(axis=1), -1)
    have = {tuple(sorted(t)) for t in triangles.tolist()}
    added = []
    for t, name in zip(their_triangles, their_names, strict=True):
        here = nearest[t]
        if (here < 0).any() or len(set(here.tolist())) < 3 or tuple(sorted(here.tolist())) in have:
            continue
        sides = [tuple(sorted((int(here[i]), int(here[(i + 1) % 3])))) for i in range(3)]
        if sum(side in edges for side in sides) >= 2:
            added.append((here, name))
    if added:
        triangles = np.vstack([triangles, [t for t, _ in added]])
        names = names + [name for _, name in added]
    print(f"{len(added)} triangles from {other.name} close holes")
    return triangles, names


def write_obj(path: Path, source: str, corners: np.ndarray, triangles, names) -> None:
    used, index = np.unique(triangles, return_inverse=True)
    triangles = index.reshape(-1, 3) + 1
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"# {source} from the BRAS benchmark (Aspöck et al., 2020)\n")
        fh.write("# DOI 10.14279/depositonce-6726.3, CC BY-SA 4.0\n")
        fh.write("# converted by validation/bras/convert_skp.py: metres, +z up\n")
        for x, y, z in corners[used]:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n".replace("-0.000000", "0.000000"))
        for name in sorted(set(names)):
            fh.write(f"usemtl {name}\n")
            for t, n in zip(triangles, names, strict=True):
                if n == name:
                    fh.write(f"f {t[0]} {t[1]} {t[2]}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("skp", type=Path, help="the BRAS SketchUp model, such as CR2.skp")
    parser.add_argument("obj", type=Path, help="the OBJ file to write")
    parser.add_argument(
        "--fill-from", type=Path, help="an OBJ of the same room whose triangles close holes here"
    )
    args = parser.parse_args()
    corners, triangles, names = read_skp(args.skp)
    corners, triangles = weld(corners, triangles)
    triangles, names = tidy(corners, triangles, names)
    if args.fill_from:
        triangles, names = fill_from(corners, triangles, names, args.fill_from)
    write_obj(args.obj, args.skp.name, corners, triangles, names)
    tri = corners[triangles]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    print(f"wrote {args.obj}: {len(triangles)} triangles")
    for name in sorted(set(names)):
        total = sum(a for a, n in zip(area, names, strict=True) if n == name)
        print(f"  {name:32} {total:9.2f} m2")


if __name__ == "__main__":
    main()
