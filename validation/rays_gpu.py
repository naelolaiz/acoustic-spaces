"""Time ray searches on a GPU against the CPU, and check that the two agree.

    python validation/rays_gpu.py [--scenes validation/bras/CR3/scene.yaml ...]
                                  [--rays 500000]

Needs the ``gpu`` extra (wgpu) and a GPU driver: Vulkan, Metal or Direct3D 12. The
late stage's ray tracer spends nearly all its time searching the model for the face a
ray meets next, and whether anything blocks the way from a reflection to a listener (its
diffuse rain). This times those two searches, the part a tracer on the GPU would speed
up, on rays like the tracer's: rays from the first source meet the model, and from
where they meet it, ``--rays`` reflections leave in Lambert directions and as many
shadow tests head for the listeners in turn. The CPU searches them on all its cores,
the GPU in one dispatch each (its hits read back included; the upload of the rays left
out, since a tracer on the GPU makes its rays there).

Prints the settings, then a Markdown table with aligned columns: the model's faces, the
searches per second of each kind on each, the speed-up for the tracer's mix (a
reflection and one shadow test per listener), and the share of rays for which the two
agree (the same face; both blocked or both clear).
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room
from acoustic_spaces.engine import gpu
from acoustic_spaces.engine.gpu_rays import GpuBvh
from acoustic_spaces.engine.model import TraceModel

ROOT = Path(__file__).resolve().parent.parent
SCENES = [
    ROOT / "validation" / "bras" / "CR2" / "scene.yaml",
    ROOT / "validation" / "bras" / "CR3" / "scene.yaml",
    ROOT / "examples" / "concert_hall" / "scene.yaml",
    ROOT / "examples" / "church" / "scene.yaml",
]
COLUMNS = (
    "Scene",
    "Faces",
    "CPU reflections (M/s)",
    "GPU reflections (M/s)",
    "CPU shadows (M/s)",
    "GPU shadows (M/s)",
    "Speed-up",
    "Same face",
    "Same shadow",
)
#: Timed runs of each search; the fastest counts.
REPEATS = 3


def _normals(bvh) -> np.ndarray:
    """Each face's unit normal (faces, 3)."""
    layout = bvh.layout()
    edges = layout["triangles"].reshape(-1, 3, 3)
    normal = np.cross(edges[:, 1], edges[:, 2])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    out = np.zeros((bvh.face_count, 3))
    out[layout["faces"]] = normal
    return out


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    d = rng.standard_normal((n, 3))
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _queries(model: TraceModel, room: Room, count: int, rng: np.random.Generator):
    """Reflections (origins, directions, faces they leave) and shadow tests (origins,
    directions, distances, faces they leave), ``count`` of each."""
    bvh = model.bvh
    source = np.array(room.scene.sources[0].position, float)
    listeners = np.array([listener.position for listener in room.scene.listeners], float)
    directions = _unit(rng, 2 * count)
    t, face = bvh.intersect(
        np.tile(source, (len(directions), 1)),
        directions,
        np.full(len(directions), np.inf),
        np.full(len(directions), -1),
    )
    hit = np.flatnonzero(face >= 0)
    if len(hit) == 0:
        raise SystemExit("no ray from the first source meets the model")
    hit = np.resize(hit, count)
    points = source + directions[hit] * t[hit, None]
    face = face[hit]
    normal = _normals(bvh)[face]
    # The side each ray came from, and a Lambert direction off it.
    normal *= -np.sign(np.einsum("ij,ij->i", normal, directions[hit]))[:, None]
    bounce = normal + _unit(rng, count)
    bounce /= np.linalg.norm(bounce, axis=1, keepdims=True)
    starts = points + normal * (1e-7 * bvh.scale)
    towards = listeners[np.arange(count) % len(listeners)] - starts
    distance = np.linalg.norm(towards, axis=1)
    return (points, bounce, face), (starts, towards / distance[:, None], distance, face)


def _fastest(search, *args, **kwargs) -> tuple[float, object]:
    """The fastest of REPEATS runs of search(*args, **kwargs) (s), and its result."""
    best, result = np.inf, None
    for _ in range(REPEATS):
        started = time.perf_counter()
        result = search(*args, **kwargs)
        best = min(best, time.perf_counter() - started)
    return best, result


def _settings(pairs: list[tuple[str, str]]) -> None:
    """Prints one setting per line, the values lined up."""
    width = max(len(label) for label, _ in pairs) + 1
    for label, value in pairs:
        print(f"{label + ':':<{width}} {value}")


def _table(rows: list[tuple[str, ...]]) -> str:
    """A Markdown table whose columns line up in a terminal too: text on the left,
    numbers on the right."""
    widths = [max(len(row[i]) for row in [COLUMNS, *rows]) for i in range(len(COLUMNS))]

    def line(cells) -> str:
        padded = [
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths, strict=True))
        ]
        return "| " + " | ".join(padded) + " |"

    rule = ["-" * widths[0], *("-" * (width - 1) + ":" for width in widths[1:])]
    return "\n".join([line(COLUMNS), line(rule), *(line(row) for row in rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenes", nargs="+", type=Path, default=SCENES)
    parser.add_argument("--rays", type=int, default=500_000, help="searches of each kind")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    name = gpu.gpu_name()
    if name is None:
        raise SystemExit('wgpu finds no GPU: pip install "acoustic-spaces[gpu]" and a driver')
    _settings(
        [
            ("Machine", f"{platform.machine()}, {platform.system()}"),
            ("CPU", f"{os.cpu_count()} threads, all used"),
            ("GPU", name),
            ("Searches", f"{args.rays:,} reflections and {args.rays:,} shadow tests per scene"),
            ("Timing", f"the fastest of {REPEATS} runs"),
        ]
    )
    print()
    gpu._device()  # start the GPU once, outside the timings
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in args.scenes:
        room = Room.from_yaml(path)
        model = TraceModel.from_room(room)
        bvh = model.bvh
        listeners = len(room.scene.listeners)
        (points, bounce, left), (starts, towards, distance, face) = _queries(
            model, room, args.rays, rng
        )
        no_limit = np.full(args.rays, np.inf)
        cpu_reflect, (_, cpu_face) = _fastest(bvh.intersect, points, bounce, no_limit, left)
        cpu_shadow, cpu_blocked = _fastest(bvh.occluded, starts, towards, distance, face)
        on_gpu = GpuBvh(bvh)
        reflections = on_gpu.upload(points, bounce, no_limit, left)
        shadows = on_gpu.upload(starts, towards, distance, face)
        on_gpu.search(reflections)  # compiles and warms up
        gpu_reflect, (_, gpu_face) = _fastest(on_gpu.search, reflections)
        gpu_shadow, (_, gpu_blocker) = _fastest(on_gpu.search, shadows, any_hit=True)
        mix_cpu = cpu_reflect + listeners * cpu_shadow
        mix_gpu = gpu_reflect + listeners * gpu_shadow
        million = args.rays / 1e6
        rows.append(
            (
                path.parent.name,
                f"{bvh.face_count:,}",
                f"{million / cpu_reflect:.1f}",
                f"{million / gpu_reflect:.1f}",
                f"{million / cpu_shadow:.1f}",
                f"{million / gpu_shadow:.1f}",
                f"{mix_cpu / mix_gpu:.2f}x",
                f"{np.mean(cpu_face == gpu_face):.4%}",
                f"{np.mean(cpu_blocked == (gpu_blocker >= 0)):.4%}",
            )
        )
        print(f"{path.parent.name} done", flush=True)
    print()
    print(_table(rows))


if __name__ == "__main__":
    main()
