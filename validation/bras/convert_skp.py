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
that. In CR1 with its door at 30.4 degrees the door leaf runs into the wall at its hinge,
which leaves the model without an inside; ``--join-door-leaf`` joins the leaf to the
wall and the floor it stands against (:func:`join_door_leaf`, which needs this package
installed):

    python validation/bras/convert_skp.py ".../rooms/CR1_DoorAngle3.skp" \
        validation/bras/CR1_DoorAngle3/CR1_DoorAngle3.obj --join-door-leaf

CR1 with its door at 4.1 degrees reads without the laboratory's wall at y = -3.65 m,
which the model with the door at 30.4 degrees has; ``--fill-from`` takes the triangles
that close such a hole from another model of the same room:

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
#: The corners (metres) of the door leaf of CR1 with the door at 30.4 degrees, after
#: welding, and of the wall it hangs on, which :func:`join_door_leaf` joins. The leaf is
#: a slab 30 mm thick (paintedConcrete) from the reverberation chamber's floor (z 0.05 m)
#: up to z 2.22 m, with a box 75 mm thick (tablesEquipment) on its inner side up to
#: z 2.19 m. The chamber's face of the wall is the plane x = 0.570738 m.
DOOR_LEAF = {
    "jamb": (0.570738, -0.012, 0.05),  # the foot of the doorway's side on the hinge side
    "jamb top": (0.570738, -0.012, 2.195),  # its top, under the lintel
    "hinge": (0.570738, 0.042612, 0.05),  # where the slab's hinge end crosses the wall's face
    "slab in wall": (0.546766, 0.028547, 0.05),  # the slab's hinge corner, in the wall
    "slab in wall top": (0.546766, 0.028547, 2.22),
    "slab outer top": (0.572641, 0.043728, 2.22),  # its outer hinge corner, at the top
    "slab end": (1.367114, -1.310418, 0.05),  # the slab's free end, outer side
    "slab end inner": (1.341239, -1.325599, 0.05),
    "slab end inner top": (1.341239, -1.325599, 2.22),
    "box hinge": (0.583281, -0.033691, 0.05),  # the box's ends at the slab's inner face
    "box end": (1.306909, -1.267086, 0.05),
    "box hinge top": (0.58776, -0.031063, 2.19),  # the same at the top, inside the slab
    "box end top": (1.311389, -1.264458, 2.19),
    "box hinge outer": (0.523072, -0.069016, 0.05),  # the box's outer side
    "box end outer": (1.2467, -1.30241, 0.05),
}
#: Corners closer than this (metres) to those of :data:`DOOR_LEAF` are its corners.
DOOR_LEAF_TOLERANCE_M = 1e-5
#: Corners closer than this (metres) to a plane lie in it (openskp reads them in single
#: precision, to within 0.3 micrometres).
IN_PLANE_M = 1e-6


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


def join_door_leaf(corners: np.ndarray, triangles: np.ndarray, names: list[str]):
    """Join the door leaf of CR1 with the door at 30.4 degrees to the wall and the floor
    it stands against, so that the model has an inside.

    In BRAS's model the slab's hinge end runs 24 mm into the wall, its faces crossing
    the wall's face with no edge in common; the floor under the leaf is drawn; and the
    box stands 5.2 mm into the slab, with no face on that side. Here the leaf becomes
    part of the solid around the rooms, as a door leaf in its frame is, and the air
    around it stays as the model has it:

    - the slab ends at the wall's face, between the doorway's side and where its hinge
      end crosses the face, 55 mm along it: the part inside the wall goes, and so does
      the wall's face where the slab meets it. The slab's outer corner at the top, 1.9
      mm in front of the wall, moves 2.2 mm along the hinge end onto the face, as
      welding moved the one at its foot; its inner face meets the doorway's side, 0.3 mm
      from where it crosses the wall's face, as at its foot;
    - the floor under the leaf goes, and the slab's inner face where the box covers it;
    - the box's top corners on the slab's side move 5.2 mm onto the slab's face, as
      welding moved those at its foot, which takes off only what lay inside the slab.

    Returns the corners, triangles and names.
    """
    from acoustic_spaces.geometry.outdoor import in_polygon
    from acoustic_spaces.geometry.triangulate import triangulate_polygon

    def corner(name: str) -> int:
        gap = np.linalg.norm(corners - DOOR_LEAF[name], axis=1)
        if gap.min() > DOOR_LEAF_TOLERANCE_M:
            raise SystemExit(
                f"no corner at the door leaf's {name} {DOOR_LEAF[name]}: "
                "is this BRAS's CR1_DoorAngle3.skp?"
            )
        return int(gap.argmin())

    jamb, jamb_top, hinge = corner("jamb"), corner("jamb top"), corner("hinge")
    in_wall, in_wall_top = corner("slab in wall"), corner("slab in wall top")
    outer_top, end = corner("slab outer top"), corner("slab end")
    end_inner, end_inner_top = corner("slab end inner"), corner("slab end inner top")
    box_hinge, box_end = corner("box hinge"), corner("box end")
    box_hinge_top, box_end_top = corner("box hinge top"), corner("box end top")
    box_hinge_outer, box_end_outer = corner("box hinge outer"), corner("box end outer")
    wall_x, floor_z = corners[jamb, 0], corners[jamb, 2]
    a, d = corners[in_wall], corners[end_inner]
    crossing = a[1] + (wall_x - a[0]) / (d[0] - a[0]) * (d[1] - a[1])
    print(
        f"  the slab's hinge corner, {1000 * (wall_x - a[0]):.1f} mm behind the wall's face, "
        "goes; its inner face meets the doorway's side "
        f"{1000 * abs(corners[jamb, 1] - crossing):.1f} mm from where it crosses the face"
    )

    # The faces the joined leaf does not have: those with a corner inside the wall, the
    # slab's hinge end and its inner face (made anew below).
    inner = {jamb, in_wall, in_wall_top, end_inner, end_inner_top}
    inner_was = corners[
        next(t for t in triangles.tolist() if set(t) == {jamb, end_inner, end_inner_top})
    ]
    slab = names[next(f for f, t in enumerate(triangles.tolist()) if set(t) <= inner)]
    drop = np.array(
        [
            in_wall in t or {hinge, in_wall_top, outer_top} <= set(t) or set(t) <= inner
            for t in triangles.tolist()
        ]
    )
    corners = corners.copy()
    corners[in_wall_top, :2] = corners[jamb, :2]  # the slab's inner hinge corner at the top
    for moved, onto, what in (
        (outer_top, hinge, "the slab's outer hinge corner at the top"),
        (box_hinge_top, box_hinge, "the box's top corner at the slab's face, hinge end"),
        (box_end_top, box_end, "the box's top corner at the slab's face, free end"),
    ):
        before = corners[moved].copy()
        corners[moved, :2] = corners[onto, :2]
        print(f"  {what} moves {1000 * np.linalg.norm(corners[moved] - before):.1f} mm")
    jamb_up, hinge_up = in_wall_top, outer_top  # now above the jamb's foot and the hinge
    tri = corners[triangles]

    def in_plane(axis: int, value: float) -> np.ndarray:
        return np.all(np.abs(tri[:, :, axis] - value) < IN_PLANE_M, axis=1)

    # The floor under the leaf: the footprint of the slab and the box together.
    footprint = corners[
        [jamb, box_hinge, box_hinge_outer, box_end_outer, box_end, end_inner, end, hinge], :2
    ]
    under = in_plane(2, floor_z) & in_polygon(tri.mean(axis=1)[:, :2], footprint) & ~drop
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    x, y = footprint.T
    covered = 0.5 * abs(x @ np.roll(y, -1) - y @ np.roll(x, -1))
    if abs(area[under].sum() - covered) > 1e-3:
        raise SystemExit(
            f"the floor under the door leaf is {area[under].sum():.4f} m2, "
            f"its footprint {covered:.4f} m2"
        )
    drop |= under

    def facing(loop: list[int], like: np.ndarray) -> list[list[int]]:
        """The polygon ``loop`` in triangles that face as the triangle ``like`` does."""
        made = [[loop[i] for i in t] for t in triangulate_polygon(corners[loop])]
        p = corners[made[0]]
        if np.cross(p[1] - p[0], p[2] - p[0]) @ np.cross(like[1] - like[0], like[2] - like[0]) < 0:
            made = [t[::-1] for t in made]
        return made

    def holds(face: int, point: int) -> bool:
        """Whether the corner ``point`` lies inside the face, seen along x."""
        (ay, az), (by, bz), (cy, cz) = tri[face][:, 1:]
        py, pz = corners[point, 1:]
        turns = [
            (qy - oy) * (pz - oz) - (qz - oz) * (py - oy)
            for (oy, oz), (qy, qz) in (
                ((ay, az), (by, bz)),
                ((by, bz), (cy, cz)),
                ((cy, cz), (ay, az)),
            )
        ]
        return all(t > 0 for t in turns) or all(t < 0 for t in turns)

    # The wall's face round where the slab meets it, made anew: its outline, from the
    # doorway's side down to the floor and along it to the hinge, goes round the slab's
    # end instead.
    wall = [
        f
        for f in np.flatnonzero(in_plane(0, wall_x) & ~drop)
        if {jamb, jamb_top, hinge} & set(triangles[f].tolist())
        or holds(f, jamb_up)
        or holds(f, hinge_up)
    ]
    sides = defaultdict(list)
    for t in triangles[wall].tolist():
        for k in range(3):
            sides[tuple(sorted((t[k], t[k - 1])))].append(t)
    ends = defaultdict(list)
    for (p, q), faces in sides.items():
        if len(faces) == 1:
            ends[p].append(q)
            ends[q].append(p)
    outline = [jamb_top, jamb]
    while True:
        step = [q for q in ends[outline[-1]] if q != outline[-2]][0]
        if step == jamb_top:
            break
        outline.append(step)
    if outline[2] != hinge:
        raise SystemExit("the wall's face does not run from the doorway's side to the hinge")
    added = facing([jamb_top, jamb_up, hinge_up, *outline[2:]], tri[wall[0]])
    added_names = [names[wall[0]]] * len(added)
    drop[wall] = True
    # The slab's inner face, round the box.
    inner_face = [
        jamb, box_hinge, box_hinge_top, box_end_top, box_end,
        end_inner, end_inner_top, jamb_up, jamb_top,
    ]  # fmt: skip
    made = facing(inner_face, inner_was)
    added += made
    added_names += [slab] * len(made)
    kept = np.flatnonzero(~drop)
    triangles = np.vstack([triangles[kept], np.array(added, dtype=triangles.dtype)])
    names = [names[i] for i in kept] + added_names
    print(f"  {int(drop.sum())} triangles of the leaf, the floor and the wall become {len(added)}")
    return corners, triangles, names


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
    parser.add_argument(
        "--join-door-leaf",
        action="store_true",
        help="CR1_DoorAngle3.skp: join the door leaf to the wall and floor it stands against",
    )
    args = parser.parse_args()
    corners, triangles, names = read_skp(args.skp)
    corners, triangles = weld(corners, triangles)
    triangles, names = tidy(corners, triangles, names)
    if args.join_door_leaf:
        print("joining the door leaf to the wall and the floor:")
        corners, triangles, names = join_door_leaf(corners, triangles, names)
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
