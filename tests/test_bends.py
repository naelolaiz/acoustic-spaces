"""Reflections of a source or a listener near the line where two walls meet, in a model
far larger than its distance from it: the search's paths against the specular paths
worked out apart from it."""

import itertools
import math

import numpy as np
import pytest

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import early, early_paths
from acoustic_spaces.engine.model import TraceModel

# A model 10 km wide: the search's tolerance for planes is 1 mm, and the intersection
# test meets no face nearer than about 14 µm.
_WIDTH = 10000.0
_TOLERANCE = 1e-7 * _WIDTH
# The room the comparison leaves at every edge (m): seven times that least distance.
_ROOM = 1e-4
_ORDER = 3


class _Face:
    """A rectangle from ``origin`` along the unit vectors ``u`` and ``v``, ``lu`` by
    ``lv`` m: one that reflects on both sides, or on the side of ``u`` x ``v`` only."""

    def __init__(self, origin, u, v, lu, lv, two_sided):
        self.origin, self.u, self.v = (np.asarray(x, dtype=float) for x in (origin, u, v))
        self.lu, self.lv, self.two_sided = float(lu), float(lv), two_sided
        self.normal = np.cross(self.u, self.v)

    def height(self, point) -> float:
        return float(self.normal @ (point - self.origin))

    def mirror(self, point):
        return point - 2.0 * self.height(point) * self.normal

    def holds(self, point, room: float) -> bool:
        """Whether ``point`` (in the plane) lies on the rectangle grown by ``room``."""
        a, b = float(self.u @ (point - self.origin)), float(self.v @ (point - self.origin))
        return -room <= a <= self.lu + room and -room <= b <= self.lv + room


def _faces(walls, height: float) -> list[_Face]:
    """The ground (60 m wide: no path of these scenes reflects farther out) and one
    upright rectangle ``height`` high per stretch of the paths ``walls``."""
    out = [_Face([-30.0, -30.0, 0.0], [1, 0, 0], [0, 1, 0], 60.0, 60.0, False)]
    for path in walls:
        for p, q in itertools.pairwise(path):
            along = np.array([q[0] - p[0], q[1] - p[1], 0.0])
            length = float(np.linalg.norm(along))
            out.append(_Face([p[0], p[1], 0.0], along / length, [0, 0, 1], length, height, True))
    return out


def _specular(source, listener, faces: list[_Face], room: float) -> list[tuple[float, int]]:
    """(length, order) of the specular paths off up to ``_ORDER`` walls, with any bounces
    off the ground (``faces[0]``, whose bounces count for no order outdoors), by the
    image source method written out: every sequence of faces, the source mirrored in each in
    turn, the path traced back from the listener. A path holds where each leg's two ends
    lie on either side of the plane it reflects off, each reflection point on its face
    and no leg crosses another face. ``room`` (m) moves every limit of a face one way:
    negative, a reflection point must lie that far inside its face's edges and in front
    of the next plane, and a leg must pass that far outside another face's edges;
    positive, each may lie as far on the other side, and a reflection point within that
    of the next plane may be the next reflection's point too (a corner, whose two
    points the search does not tell apart). With -r and +r: the paths that must hold and
    those that may, for a search that tells places apart to r. Sequences of as many
    faces that make one mirror (the same faces in another order, where that changes
    nothing) are one path, as in the search."""
    source, listener = np.asarray(source, dtype=float), np.asarray(listener, dtype=float)
    marks = np.vstack([np.zeros(3), np.eye(3)])
    found, mirrors = [], []
    for order in range(2 * _ORDER + 2):
        for seq in itertools.product(range(len(faces)), repeat=order):
            if any(a == b for a, b in itertools.pairwise(seq)) or np.count_nonzero(seq) > _ORDER:
                continue
            images, moved = [source], marks
            for s in seq:
                images.append(faces[s].mirror(images[-1]))
                moved = np.array([faces[s].mirror(mark) for mark in moved])
            if not _traced(faces, seq, images, order - 1, [listener], room):
                continue
            if any(k == order and np.abs(m - moved).max() <= 1e-6 for k, m in mirrors):
                continue
            mirrors.append((order, moved))
            found.append((float(np.linalg.norm(images[-1] - listener)), order))
    return found


def _traced(faces: list[_Face], seq, images, j: int, points: list, room: float) -> bool:
    """Whether the path through ``points`` (from the listener back to reflection j + 1)
    goes on to hold off the faces ``seq[: j + 1]`` down to the source (see
    :func:`_specular`)."""
    if j < 0:
        return not _blocked(faces, [*points, images[0]], room)
    face, point = faces[seq[j]], points[-1]
    here, there = face.height(point), face.height(images[j + 1])
    if not face.two_sided and there > 0.0:
        return False
    # The listener's side of the plane is its own; a reflection point's, within the room.
    near = j < len(seq) - 1 and abs(here) <= abs(room)
    at = []
    if here * there < 0.0 and not (near and room < 0.0):
        at.append(point + here / (here - there) * (images[j + 1] - point))
    if near and room > 0.0:
        at.append(point - here * face.normal)  # a corner: at the last reflection's point
    return any(
        face.holds(p, room) and _traced(faces, seq, images, j - 1, [*points, p], room) for p in at
    )


def _blocked(faces: list[_Face], points: list, room: float) -> bool:
    """Whether a leg of the path through ``points`` crosses a face (shrunk by ``room``):
    one with an end on a face's plane, to rounding, reflects there."""
    for p, q in itertools.pairwise(points):
        for face in faces:
            hp, hq = face.height(p), face.height(q)
            if abs(hp) <= 1e-11 or abs(hq) <= 1e-11 or hp * hq > 0.0:
                continue
            if face.holds(p + hp / (hp - hq) * (q - p), -room):
                return True
    return False


def _without(need, have) -> list:
    """The (length, order) of ``need`` with no partner of their own in ``have`` (the
    same order, the length within 1 nm)."""
    left = list(have)
    missing = []
    for length, order in sorted(need):
        partner = next(
            (i for i, (x, k) in enumerate(left) if k == order and abs(x - length) <= 1e-9), None
        )
        if partner is None:
            missing.append((length, order))
        else:
            left.pop(partner)
    return missing


def _bend(angle: float, apex, turn: float) -> list[list[float]]:
    """The path of two walls 12 m long from ``apex`` at ``angle`` degrees to each other,
    the first at ``turn`` degrees from x."""
    a, b = math.radians(turn), math.radians(turn + angle)
    return [
        [apex[0] + 12.0 * math.cos(a), apex[1] + 12.0 * math.sin(a)],
        [apex[0], apex[1]],
        [apex[0] + 12.0 * math.cos(b), apex[1] + 12.0 * math.sin(b)],
    ]


def _room(near, far, walls, near_is: str) -> Room:
    """A hard ground ``_WIDTH`` wide under an open sky with the walls ``walls`` 4 m high;
    the point ``near`` as the source and the points ``far`` as listeners, or the other
    way round (``near_is`` "listener")."""
    half = _WIDTH / 2.0
    one = [{"name": "near", "position": [float(x) for x in near]}]
    many = [{"name": f"p{i}", "position": [float(x) for x in p]} for i, p in enumerate(far)]
    scene = {
        "geometry": {
            "ground": {"extent": [[-half, -half], [half, half]], "sky": 30.0},
            "barriers": [
                {"name": f"wall{i}", "path": path, "height": 4.0} for i, path in enumerate(walls)
            ],
        },
        "materials": {"hard": {"absorption": [0.02] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": one if near_is == "source" else many,
        "listeners": many if near_is == "source" else one,
        "simulation": {
            "early": {"method": "image_sources", "max_order": _ORDER, "diffraction": "none"}
        },
    }
    return Room.from_scene(Scene.model_validate(scene))


def _not_specular(
    angle: float, distance: float, share: float, near_is: str, up: float = 1.5, places: int = 16
) -> list[tuple[int, str]]:
    """Two walls meet at ``angle`` degrees in a model 10 km wide. A source (or a
    listener, ``near_is``) stands ``up`` m over the ground, ``distance`` times the
    search's tolerance for planes (1 mm here) from the line where they meet, at
    ``share`` of the angle from the first wall; ``places`` listeners (sources) stand 7 m
    from that line across the angle. Returned: the places where the search's paths up to
    three reflections are not the specular ones (:func:`_specular`), each with what is
    wrong: one that holds with 0.1 mm to spare at every edge is missing, or one found
    does not hold with 0.1 mm given. A path is known by its number of reflections and
    its length, to 1 nm; where it reflects is not compared (the search takes a
    reflection point within its tolerance of the next plane, from which the path goes on
    to a face of it, as that plane's too)."""
    apex, turn = (1.3, -2.1), 37.0
    walls = [_bend(angle, apex, turn)]
    way = math.radians(turn + share * angle)
    reach = distance * _TOLERANCE
    near = [apex[0] + reach * math.cos(way), apex[1] + reach * math.sin(way), up]
    far = []
    for i in range(places):
        way = math.radians(turn + angle * (i + 0.37) / places)
        far.append([apex[0] + 7.0 * math.cos(way), apex[1] + 7.0 * math.sin(way), 1.7])
    room = _room(near, far, walls, near_is)
    faces = _faces(walls, 4.0)
    wrong = []
    for i, there in enumerate(far):
        source, listener = (near, there) if near_is == "source" else (there, near)
        paths = early_paths(room, **{"listener" if near_is == "source" else "source": f"p{i}"})
        found = list(zip(paths.distance_m.tolist(), paths.order.tolist(), strict=True))
        if _without(_specular(source, listener, faces, -_ROOM), found):
            wrong.append((i, "missing"))
        if _without(found, _specular(source, listener, faces, _ROOM)):
            wrong.append((i, "extra"))
    return wrong


@pytest.mark.parametrize("near_is", ["source", "listener"])
@pytest.mark.parametrize("share", [0.2, 0.5])
@pytest.mark.parametrize("distance", [0.5, 4.0, 16.0])
@pytest.mark.parametrize("angle", [50.0, 60.0, 75.0, 90.0, 91.0, 120.0, 170.0])
def test_the_paths_of_a_point_near_a_bend_are_the_specular_ones(
    angle: float, distance: float, share: float, near_is: str
) -> None:
    """A source or a listener 1.5 m up, 0.5, 4 or 16 mm from the line where two walls
    meet at ``angle`` degrees in a model 10 km wide, has the specular paths to 16 places
    7 m away (:func:`_not_specular`).

    The search had other paths in most of these cases: a face reached past its edge by
    the tolerance times its triangles' shape, so that the reflection off one wall went
    on over the next; two paths whose images lay within the tolerance of each other
    counted as one; a reflection point within the tolerance of the next plane, off its
    faces, ended the path; a ray that met one wall beside the other's plane without
    crossing it reflected off both; and so did one that crossed that plane, but past
    the other wall's end (seen at bends of 50 to 110 degrees other than 60 and 90)."""
    assert not _not_specular(angle, distance, share, near_is)


@pytest.mark.parametrize("near_is", ["source", "listener"])
@pytest.mark.parametrize(
    ("angle", "up"), [(75.0, 0.2), (91.0, 0.5), (120.0, 0.2), (170.0, 0.2), (170.0, 0.5)]
)
def test_the_paths_of_a_point_at_the_foot_of_a_bend_are_the_specular_ones(
    angle: float, up: float, near_is: str
) -> None:
    """A source or a listener 0.5 mm from the line where two walls meet at ``angle``
    degrees and ``up`` mm over the ground, in a model 10 km wide, has the specular paths
    to 48 places 7 m away (:func:`_not_specular`). Besides the faults above, a
    reflection point within the tolerance of the next plane, over one of its faces,
    reflected off that plane there too wherever the path went on: with the image it
    heads for on the point's own side of the plane, or meeting the plane past its
    faces. That left reflections that do not exist at up to 44 of the 48 places."""
    wrong = _not_specular(angle, 0.5, 0.2, near_is, up=up * _TOLERANCE, places=48)
    assert not wrong


def test_a_face_ends_at_its_corners_as_at_its_edges() -> None:
    """A wall 1000 m long and 4 m high is two triangles with corners of 0.23 degrees.
    Past the wall's end along its top, as past its top midway, a point lies on the wall
    only within the least distance at which a ray meets a face (14 µm in this model),
    though the lines of the two edges that meet in such a corner stay within that
    distance of each other for 3.5 mm past it."""
    room = _room([10.0, 5.0, 1.5], [[20.0, 5.0, 1.7]], [[[0.0, 0.0], [1000.0, 0.0]]], "source")
    model = TraceModel.from_room(room)
    least = model.bvh.min_distance
    assert 1.3e-5 < least < 1.5e-5
    mesh = room.mesh
    corners = mesh.vertices[mesh.faces]
    wall = np.flatnonzero(
        (np.abs(corners[:, :, 1]) < 1e-9).all(axis=1) & (np.ptp(corners[:, :, 2], axis=1) > 3.0)
    )
    assert len(wall) >= 2

    def on_the_wall(x: float, z: float) -> bool:
        at = np.broadcast_to([x, 0.0, z], (len(wall), 3))
        return bool(early._over(model, wall, at).any())

    assert on_the_wall(1000.0 - 1e-3, 4.0 - 1e-3)
    for (x, z), (along, up) in (
        ((1000.0, 4.0), (1.0, 0.0)),  # past the corner, along the top's line
        ((500.0, 4.0), (0.0, 1.0)),  # over the top, midway
        ((1000.0, 2.0), (1.0, 0.0)),  # past the end, midway up
    ):
        assert on_the_wall(x + 0.5 * least * along, z + 0.5 * least * up)
        assert not on_the_wall(x + 3.0 * least * along, z + 3.0 * least * up)
    assert not on_the_wall(1000.0 + 1e-3, 4.0)
