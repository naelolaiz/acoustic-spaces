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

openskp 1.3.0 reads CR2 and CR3; it cannot read CR4, whose text labels are attached to
a component.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

WELD_M = 0.012
MIN_AREA_M2 = 1e-10


def read_skp(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Corners (metres, +z up), triangles and the material name of each triangle."""
    from openskp import SkpFile

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
    args = parser.parse_args()
    corners, triangles, names = read_skp(args.skp)
    corners, triangles = weld(corners, triangles)
    triangles, names = tidy(corners, triangles, names)
    write_obj(args.obj, args.skp.name, corners, triangles, names)
    tri = corners[triangles]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    print(f"wrote {args.obj}: {len(triangles)} triangles")
    for name in sorted(set(names)):
        total = sum(a for a, n in zip(area, names, strict=True) if n == name)
        print(f"  {name:32} {total:9.2f} m2")


if __name__ == "__main__":
    main()
