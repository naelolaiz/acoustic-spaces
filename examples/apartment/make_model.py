"""Write apartment.obj: a living room and a bedroom side by side, 2.6 m high, with a
door in the wall between them and a window in the living room.

    python make_model.py   # writes apartment.obj next to this script

The wall between the rooms is one layer of faces, which both rooms reflect off; its
material says how much sound passes through. Each surface is a group named for its
material assignment in scene.yaml.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

HEIGHT = 2.6
LIVING = (0.0, 5.0)  # x range of the living room
BEDROOM = (5.0, 9.0)  # x range of the bedroom
DEPTH = 4.0  # both rooms run from y = 0 to y = 4
DOOR = ((2.9, 3.8), (0.0, 2.0))  # in the wall between: y range, z range
WINDOW = ((1.0, 3.0), (0.9, 2.1))  # in the living room's wall at x = 0: y range, z range


def cells(u_range, v_range, holes=()):
    """The rectangle u_range x v_range cut into rectangles along the edges of ``holes``
    (each a (u range, v range) inside it): (u0, u1, v0, v1, inside a hole)."""
    us = sorted({*u_range, *(u for hole in holes for u in hole[0])})
    vs = sorted({*v_range, *(v for hole in holes for v in hole[1])})
    for u0, u1 in pairwise(us):
        for v0, v1 in pairwise(vs):
            inside = any(
                h[0][0] <= u0 and u1 <= h[0][1] and h[1][0] <= v0 and v1 <= h[1][1] for h in holes
            )
            yield u0, u1, v0, v1, inside


def main() -> None:
    groups: dict[str, list[list[tuple[float, float, float]]]] = {}

    def quad(label, corners):
        # The corners below go round anticlockwise seen from outside; reversed, every
        # face's normal points into its room.
        groups.setdefault(label, []).append(corners[::-1])

    for (x0, x1), floor in ((LIVING, "living_floor"), (BEDROOM, "bedroom_floor")):
        quad(floor, [(x0, 0, 0), (x0, DEPTH, 0), (x1, DEPTH, 0), (x1, 0, 0)])
        top = [(x0, 0, HEIGHT), (x1, 0, HEIGHT), (x1, DEPTH, HEIGHT), (x0, DEPTH, HEIGHT)]
        quad("ceiling", top)
        # The long walls, y = 0 and y = DEPTH, room by room.
        quad("walls", [(x0, 0, 0), (x1, 0, 0), (x1, 0, HEIGHT), (x0, 0, HEIGHT)])
        quad("walls", [(x1, DEPTH, 0), (x0, DEPTH, 0), (x0, DEPTH, HEIGHT), (x1, DEPTH, HEIGHT)])
    # The end walls: the living room's with its window, the bedroom's plain.
    for y0, y1, z0, z1, glass in cells((0.0, DEPTH), (0.0, HEIGHT), [WINDOW]):
        quad("window" if glass else "walls", [(0, y1, z0), (0, y0, z0), (0, y0, z1), (0, y1, z1)])
    x = BEDROOM[1]
    quad("walls", [(x, 0, 0), (x, DEPTH, 0), (x, DEPTH, HEIGHT), (x, 0, HEIGHT)])
    # The wall between, with its door.
    x = LIVING[1]
    for y0, y1, z0, z1, door in cells((0.0, DEPTH), (0.0, HEIGHT), [DOOR]):
        quad("door" if door else "partition", [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)])

    index: dict[tuple[float, float, float], int] = {}
    lines = ["# A living room and a bedroom joined by a door; see make_model.py.", ""]
    faces = []
    for label, quads in groups.items():
        faces.append(f"g {label}")
        for corners in quads:
            ids = []
            for corner in corners:
                key = tuple(round(float(c), 6) for c in corner)
                if key not in index:
                    index[key] = len(index) + 1
                    lines.append("v {:g} {:g} {:g}".format(*key))
                ids.append(str(index[key]))
            faces.append("f " + " ".join(ids))
    out = Path(__file__).with_name("apartment.obj")
    out.write_text("\n".join([*lines, *faces]) + "\n", encoding="utf-8")
    print(f"wrote {out}: {len(index)} vertices, {len(faces) - len(groups)} faces")


if __name__ == "__main__":
    main()
