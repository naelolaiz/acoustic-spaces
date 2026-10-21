"""Writes stadium.obj, the model of this example, and its loudspeaker arrays.

A football stadium seen as a bowl: a pitch of 105 x 68 m with 8 m of grass round it and
a wall 2 m high, then stands on all four sides that rise at 30 degrees, 38 m deep, to
the last row 24 m up, meeting at the corners. A back wall closes the bowl up to a roof
30 m above the pitch, which reaches 30 m in over the stands and leaves the pitch open to
the sky. At the west end a concert stage stands on the pitch: a deck 2 m high, with a
back wall and a roof. Metres, +z up, origin at the centre spot; face normals point to
the air the audience is in.

The arrays are written as array files (see docs/scenes.md, coherent arrays): the main
line array of 12 boxes hung in a J, curving down more towards its bottom so that its top
boxes throw to the far stand and its bottom ones cover the pitch in front of the stage;
a delay tower's line of 6 boxes; a fill of 4 boxes for the stands; and 8 cardioid subs in
a block, 4 wide and 2 high, under the middle of the stage front. A block rather than a
line across the whole front: a line 17 m long has not formed its far-field pattern,
which the scene takes, anywhere on the pitch, while a block 2 m wide has at the lowest
bands a few metres out. Each box is 0.42 m tall, and its own pattern is
box.csv, a generic box made up for this example, not any product's data: 100 degrees
wide from 1 kHz up, wider below, and as tall as its front, which its horn sounds as one
line (a measured balloon of a real box, in SOFA or CSV, would stand in for it).

The monitors on stage are a pair of wedges at the singer's feet, an array of two of
wedge.csv, a generic coaxial wedge whose horn covers a cone 60 to 90 degrees wide from
1 kHz up; the drum fill and the front fills on the stage's lip are one each of them,
and the side fills are fills of 4 boxes.

Run it from this folder: python make_model.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

#: Half the length (x) and width (y) of the grass, inside the wall round the pitch.
FIELD_X, FIELD_Y = 60.5, 42.0
#: Height of the wall round the pitch, where the first row starts.
PITCH_WALL = 2.0
#: Depth of the stands on plan, and the height of their last row.
DEPTH, TOP = 38.0, 24.0
#: Height of the roof, and how far it reaches in over the stands.
ROOF, ROOF_DEPTH = 30.0, 30.0
#: The stage: deck from x = STAGE_BACK to STAGE_FRONT and y = -STAGE_Y to STAGE_Y, the
#: deck's height, and the height and depth of its roof.
STAGE_BACK, STAGE_FRONT, STAGE_Y = -58.0, -42.0, 16.0
DECK, STAGE_ROOF, STAGE_ROOF_DEPTH = 2.0, 18.0, 14.0

#: Height of a line array's box, from one rigging point to the next (m).
BOX = 0.42
#: The box (box.csv) in the octave bands from 63 Hz to 8 kHz: how wide it covers across
#: (degrees between the directions 6 dB down), how far down it is behind and far off
#: its aim, and the height of the front its horn sounds as one line.
BOX_BANDS_HZ = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
BOX_WIDTH = [360, 300, 220, 160, 110, 100, 100, 90]
BOX_BACK_DB = [0, -6, -10, -15, -20, -20, -20, -20]
BOX_FRONT = 0.40
#: The wedge (wedge.csv) in the same bands: how wide its cone is (degrees between the
#: directions 6 dB down), and how far down it is behind and far off its aim.
WEDGE_WIDTH = [360, 300, 200, 130, 90, 70, 60, 60]
WEDGE_BACK_DB = [0, -6, -10, -15, -18, -20, -20, -20]
#: How far apart the singer's two wedges stand.
WEDGE_SPACING = 0.9
#: Each array: the aim of its top box (degrees above the array's own aim) and the angles
#: between one box and the next, down the array.
ARRAYS = {
    "main_array": (3.0, [0.5, 0.5, 1, 1, 1.5, 2, 3, 4, 5, 6, 7]),
    "tower_array": (0.0, [0.5, 1, 2, 3, 4]),
    "fill_array": (0.0, [8, 8, 8]),
}
#: The block of subs: how many side by side (along y) and stacked, and how far apart.
SUB_COLUMNS, SUB_ROWS, SUB_WIDTH, SUB_HEIGHT = 4, 2, 0.7, 0.6


def _ring(x: float, y: float, z: float) -> list[tuple[float, float, float]]:
    """A rectangle's corners, counter-clockwise from above, from its south-west corner."""
    return [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]


def _sides(ring_a, ring_b):
    """The four quadrilaterals between two rectangles of corners, south side first."""
    return [(ring_a[k], ring_a[(k + 1) % 4], ring_b[(k + 1) % 4], ring_b[k]) for k in range(4)]


def _surfaces():
    """Each surface's name and quadrilaterals, with a point its normal faces (or, with
    True after it, faces away from)."""
    outer_x, outer_y = FIELD_X + DEPTH, FIELD_Y + DEPTH
    inner_x, inner_y = outer_x - ROOF_DEPTH, outer_y - ROOF_DEPTH
    bowl = (0.0, 0.0, 15.0)
    south, east, north, west = _sides(
        _ring(FIELD_X, FIELD_Y, PITCH_WALL), _ring(outer_x, outer_y, TOP)
    )
    walls = _sides(_ring(FIELD_X, FIELD_Y, 0.0), _ring(FIELD_X, FIELD_Y, PITCH_WALL))
    back = _sides(_ring(outer_x, outer_y, TOP), _ring(outer_x, outer_y, ROOF))
    facade = _sides(_ring(outer_x, outer_y, 0.0), _ring(outer_x, outer_y, TOP))
    roof = _sides(_ring(outer_x, outer_y, ROOF), _ring(inner_x, inner_y, ROOF))
    deck = [(STAGE_BACK, -STAGE_Y), (STAGE_FRONT, -STAGE_Y), (STAGE_FRONT, STAGE_Y)]
    deck.append((STAGE_BACK, STAGE_Y))
    roof_front = STAGE_BACK + STAGE_ROOF_DEPTH
    in_front = (0.0, 0.0, 10.0)
    return [
        ("audience_south_stand", [south], bowl, False),
        ("audience_north_stand", [north], bowl, False),
        ("audience_east_stand", [east], bowl, False),
        # Behind the stage: no one sits there.
        ("west_stand", [west], bowl, False),
        ("pitch_wall", walls, bowl, False),
        ("back_wall", back, bowl, False),
        ("facade", facade, bowl, True),
        ("roof", roof, (0.0, 0.0, -1000.0), False),
        ("stage_floor", [tuple((x, y, DECK) for x, y in deck)], (0.0, 0.0, 1000.0), False),
        (
            "stage_deck",
            [
                ((STAGE_FRONT, -STAGE_Y, 0), (STAGE_FRONT, STAGE_Y, 0))
                + ((STAGE_FRONT, STAGE_Y, DECK), (STAGE_FRONT, -STAGE_Y, DECK)),
                ((STAGE_BACK, -STAGE_Y, 0), (STAGE_FRONT, -STAGE_Y, 0))
                + ((STAGE_FRONT, -STAGE_Y, DECK), (STAGE_BACK, -STAGE_Y, DECK)),
                ((STAGE_FRONT, STAGE_Y, 0), (STAGE_BACK, STAGE_Y, 0))
                + ((STAGE_BACK, STAGE_Y, DECK), (STAGE_FRONT, STAGE_Y, DECK)),
            ],
            (0.5 * (STAGE_BACK + STAGE_FRONT), 0.0, 1.0),
            True,
        ),
        (
            "stage_back",
            [
                ((STAGE_BACK, -STAGE_Y, 0), (STAGE_BACK, STAGE_Y, 0))
                + ((STAGE_BACK, STAGE_Y, STAGE_ROOF), (STAGE_BACK, -STAGE_Y, STAGE_ROOF))
            ],
            in_front,
            False,
        ),
        (
            "stage_roof",
            [
                ((STAGE_BACK, -STAGE_Y, STAGE_ROOF), (roof_front, -STAGE_Y, STAGE_ROOF))
                + ((roof_front, STAGE_Y, STAGE_ROOF), (STAGE_BACK, STAGE_Y, STAGE_ROOF))
            ],
            (-50.0, 0.0, 0.0),
            False,
        ),
    ]


def _facing(quad, toward, away):
    """The quadrilateral's corners in the order whose normal faces the point ``toward``,
    or faces away from it."""
    a, b, c = quad[0], quad[1], quad[2]
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    normal = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    side = sum(normal[k] * (toward[k] - a[k]) for k in range(3))
    return list(quad) if (side > 0) != away else list(reversed(quad))


def _line_array(top: float, splays: list[float]) -> dict:
    """A line array hung in a curve: each box BOX below the last along the last's front,
    turned down by the splay between them; positions from the boxes' centroid."""
    aims, points = [top], [(0.0, 0.0)]
    for splay in splays:
        a = math.radians(aims[-1])
        x, z = points[-1]
        # Down the front of the box above, then this box's centre half a box further.
        points.append((x + BOX * math.sin(a), z - BOX * math.cos(a)))
        aims.append(aims[-1] - splay)
    cx = sum(x for x, _ in points) / len(points)
    cz = sum(z for _, z in points) / len(points)
    elements = [
        {"position": [round(x - cx, 4), 0.0, round(z - cz, 4)], "orientation": {"pitch": aim}}
        for (x, z), aim in zip(points, aims, strict=True)
    ]
    return {"element": {"file": "box.csv"}, "elements": elements}


def _box_levels(azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """The box's level (dB) towards each direction (degrees), per octave band: across, a
    cosine power that is 6 dB down at half the band's width; up and down, a line as tall
    as its front, averaged over the band; neither below the band's floor. The across part
    fades towards straight up and down, where every azimuth is the same direction."""
    az, el = np.radians(azimuth)[:, None], np.radians(elevation)[:, None]
    levels = []
    for band, width, back in zip(BOX_BANDS_HZ, BOX_WIDTH, BOX_BACK_DB, strict=True):
        floor = 10.0 ** (back / 10.0)
        if width >= 360:
            across = np.ones_like(az)
        else:
            power = math.log(0.5) / math.log(0.5 * (1.0 + math.cos(math.radians(width / 2))))
            across = np.maximum((0.5 * (1.0 + np.cos(az))) ** (2.0 * power), floor)
        frequencies = band * 2.0 ** (np.linspace(-0.5, 0.5, 9)[None, :])
        x = np.pi * BOX_FRONT * frequencies * np.sin(el) / 343.0
        line = np.maximum(np.mean(np.sinc(x / np.pi) ** 2, axis=1, keepdims=True), floor)
        levels.append(10.0 * np.log10(across) * np.cos(el) ** 2 + 10.0 * np.log10(line))
    return np.hstack(levels)


def _wedge_levels(azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """The wedge's level (dB) towards each direction (degrees), per octave band: a cosine
    power of the angle off its aim that is 6 dB down at half the band's width, above the
    band's floor."""
    off = np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
    levels = []
    for width, back in zip(WEDGE_WIDTH, WEDGE_BACK_DB, strict=True):
        cone = np.ones_like(off)
        if width < 360:
            power = math.log(0.5) / math.log(0.5 * (1.0 + math.cos(math.radians(width / 2))))
            cone = (0.5 * (1.0 + off)) ** (2.0 * power)
        levels.append(10.0 * np.log10(np.maximum(cone, 10.0 ** (back / 10.0)))[:, None])
    return np.hstack(levels)


def _write_balloon(path: Path, what: str, pattern) -> None:
    azimuth, elevation = np.meshgrid(np.arange(-180, 181, 10), np.arange(-90, 90.1, 2.5))
    azimuth, elevation = azimuth.ravel(), elevation.ravel()
    levels = pattern(azimuth, elevation)
    lines = [
        f"# A generic {what}, written by make_model.py (not any product's data).",
        "azimuth, elevation, " + ", ".join(str(b) for b in BOX_BANDS_HZ),
    ]
    for a, e, row in zip(azimuth, elevation, levels, strict=True):
        lines.append(f"{a:g}, {e:g}, " + ", ".join(f"{v:.1f}" for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _subs() -> dict:
    """The block of subs, each a cardioid that sends little back onto the stage."""
    across = [(k - 0.5 * (SUB_COLUMNS - 1)) * SUB_WIDTH for k in range(SUB_COLUMNS)]
    up = [(k - 0.5 * (SUB_ROWS - 1)) * SUB_HEIGHT for k in range(SUB_ROWS)]
    elements = [{"position": [0.0, round(y, 4), round(z, 4)]} for z in up for y in across]
    return {"element": "cardioid", "elements": elements}


def _write_arrays() -> None:
    folder = Path(__file__).parent
    arrays = {name: _line_array(*layout) for name, layout in ARRAYS.items()}
    arrays["sub_array"] = _subs()
    wedges = [{"position": [0.0, side * WEDGE_SPACING / 2, 0.0]} for side in (1, -1)]
    arrays["monitor_wedges"] = {"element": {"file": "wedge.csv"}, "elements": wedges}
    _write_balloon(folder / "box.csv", "line-array box", _box_levels)
    _write_balloon(folder / "wedge.csv", "monitor wedge", _wedge_levels)
    for name, data in arrays.items():
        body = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=88)
        text = f"# Written by make_model.py: see there.\n{body}"
        (folder / f"{name}.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    lines = [
        "# A football stadium with a concert stage at its west end (make_model.py).",
        "# Metres, +z up, origin at the centre spot; normals face the audience's air.",
    ]
    vertices: list[tuple[float, float, float]] = []
    groups = []
    for name, quads, toward, away in _surfaces():
        faces = []
        for quad in quads:
            corners = _facing(quad, toward, away)
            first = len(vertices) + 1
            vertices.extend(corners)
            faces.append(f"f {first} {first + 1} {first + 2}")
            faces.append(f"f {first} {first + 2} {first + 3}")
        groups.append((name, faces))
    lines += [f"v {x:g} {y:g} {z:g}" for x, y, z in vertices]
    for name, faces in groups:
        lines += ["", f"g {name}", *faces]
    Path(__file__).with_name("stadium.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_arrays()


if __name__ == "__main__":
    main()
