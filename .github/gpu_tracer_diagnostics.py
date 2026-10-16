"""Temporary: the GPU late tracer and ray search on this runner's adapter against the
CPU, for variants of the WGSL ray search, each in its own process with a time limit."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from test_native import _unit_cube  # noqa: E402
from test_tracer_reference import REFERENCE, reference_arguments  # noqa: E402

from acoustic_spaces import _native  # noqa: E402
from acoustic_spaces.engine import gpu, gpu_rays, gpu_tracer  # noqa: E402

ROUNDS = 300
VARIANTS = ("new", "old", "new_private_stack", "old_search_new_triangle")
OLD = """// Moller-Trumbore, two-sided, the edges widened by EDGE: t, or a negative value.
fn hit_triangle(place: u32, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let v0 = bitcast<vec3<f32>>(triangles[3u * place].xyz);
    let e1 = bitcast<vec3<f32>>(triangles[3u * place + 1u].xyz) - v0;
    let e2 = bitcast<vec3<f32>>(triangles[3u * place + 2u].xyz) - v0;
    let p = cross(d, e2);
    let det = dot(e1, p);
    if (abs(det) < 1e-30) {
        return -1.0;
    }
    let inv_det = 1.0 / det;
    let s = o - v0;
    let u = dot(s, p) * inv_det;
    if (u < -EDGE || u > 1.0 + EDGE) {
        return -1.0;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv_det;
    if (v < -EDGE || u + v > 1.0 + EDGE) {
        return -1.0;
    }
    return dot(e2, q) * inv_det;
}

// The nearest face along o + t d (d of unit length) with t in (t_min, t_max), skipping
// face `skip`, or (t_max, NONE); with `any_hit`, the first face found, which is enough
// to know the way is blocked. Of two children the ray enters, the nearer goes first;
// the other waits and is dropped if a hit nearer than its box turns up meanwhile.
fn search_ray(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32, skip: u32,
              any_hit: bool) -> Hit {
    var best = Hit(t_max, NONE);
    let inv = reciprocal(d);
    if (enter(0u, o, inv, t_max) > t_max) {
        return best;
    }
    var waiting_node: array<u32, STACK>;
    var waiting_entry: array<f32, STACK>;
    var top = 0u;
    var node = 0u;
    loop {
        let low = nodes[2u * node];
        let count = nodes[2u * node + 1u].w;
        if (count > 0u) {
            for (var place = low.w; place < low.w + count; place++) {
                let face = triangles[3u * place].w;
                if (face == skip) {
                    continue;
                }
                let t = hit_triangle(place, o, d);
                if (t > t_min && t < best.t) {
                    best = Hit(t, face);
                    if (any_hit) {
                        return best;
                    }
                }
            }
        } else {
            let left = low.w;
            let right = left + 1u;
            let to_left = enter(left, o, inv, best.t);
            let to_right = enter(right, o, inv, best.t);
            let in_left = to_left <= best.t;
            let in_right = to_right <= best.t;
            if (in_left && in_right) {
                let left_first = to_left <= to_right;
                waiting_node[top] = select(left, right, left_first);
                waiting_entry[top] = select(to_left, to_right, left_first);
                top++;
                node = select(right, left, left_first);
                continue;
            }
            if (in_left || in_right) {
                node = select(right, left, in_left);
                continue;
            }
        }
        // The nearest waiting node the ray still reaches before its best hit, if any.
        var resumed = false;
        while (top > 0u) {
            top--;
            if (waiting_entry[top] <= best.t) {
                node = waiting_node[top];
                resumed = true;
                break;
            }
        }
        if (!resumed) {
            break;
        }
    }
    return best;
}
"""


def _variant(name: str) -> None:
    """Puts the variant's search into both shaders."""
    start = gpu_rays.BVH_WGSL.index("// Moller-Trumbore")
    new = gpu_rays.BVH_WGSL[start:]
    if name == "new":
        return
    if name == "old":
        text = OLD
    elif name == "new_private_stack":
        stack = (
            "    var waiting_node: array<u32, STACK>;\n    var waiting_entry: array<f32, STACK>;\n"
        )
        assert stack in new
        text = new.replace(stack, "").replace(
            "fn search_ray(",
            "var<private> waiting_node: array<u32, STACK>;\n"
            "var<private> waiting_entry: array<f32, STACK>;\n\nfn search_ray(",
        )
    else:
        cut_old = OLD.index("// The nearest face")
        cut_new = new.index("// The nearest face")
        text = new[:cut_new] + OLD[cut_old:]
    for module in (gpu_rays, gpu_tracer):
        assert module._SHADER.count(new) == 1
        module._SHADER = module._SHADER.replace(new, text)


class _Numpy:
    """numpy for gpu_tracer, printing each round's counters."""

    def __init__(self) -> None:
        self.rounds = 0
        self.started = time.perf_counter()

    def __getattr__(self, name):
        return getattr(np, name)

    def frombuffer(self, buffer, dtype=float, *args, **kwargs):
        out = np.frombuffer(buffer, dtype, *args, **kwargs)
        if len(out) == gpu_tracer._COUNTERS:
            self.rounds += 1
            if self.rounds <= 6 or self.rounds % 50 == 0:
                took = time.perf_counter() - self.started
                print(f"    round {self.rounds}: counters {out[:5].tolist()} at {took:.1f} s")
            if self.rounds >= ROUNDS:
                raise RuntimeError(f"still going after {ROUNDS} rounds")
        return out


def _trace(label: str, values: dict) -> None:
    print(f"  {label}:", flush=True)
    on_cpu = _native.trace_stochastic(**values)
    gpu_tracer.np = _Numpy()
    started = time.perf_counter()
    try:
        on_gpu = gpu_tracer.trace_stochastic(**values)
    finally:
        gpu_tracer.np = np
    print(f"    gpu {time.perf_counter() - started:.2f} s", flush=True)
    a, b = on_cpu[0], on_gpu[0]
    print(f"    energy per listener cpu {a.sum((1, 2))} gpu {b.sum((1, 2))}")
    for i, name in ((3, "escaped"), (6, "traced")):
        print(f"    {name} cpu {on_cpu[i]} gpu {on_gpu[i]}")
    print(f"    reflections cpu {np.sum(on_cpu[5]):.6g} gpu {np.sum(on_gpu[5]):.6g}", flush=True)


def _searches(vertices, faces) -> None:
    """The ray search shader's misses and other faces than the CPU's."""
    rng = np.random.default_rng(3)
    bvh = _native.Bvh(vertices, faces)
    on_gpu = gpu_rays.GpuBvh(bvh)
    n = 20000
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    origins = lo + rng.uniform(0.02, 0.98, (n, 3)) * (hi - lo)
    directions = rng.standard_normal((n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    none = np.full(n, -1)
    t, face = bvh.intersect(origins, directions, np.full(n, np.inf), none)
    t_gpu, face_gpu = on_gpu.intersect(origins, directions, np.inf, none)
    hit = face >= 0
    print(
        f"  first hits: cpu {hit.sum()}, gpu misses {(hit & (face_gpu < 0)).sum()}, "
        f"other faces {(hit & (face_gpu >= 0) & (face_gpu != face)).sum()}"
    )
    starts = (origins + directions * t[:, None])[hit]
    bounce = -directions[hit]
    t2, face2 = bvh.intersect(starts, bounce, np.full(len(starts), np.inf), face[hit])
    t2_gpu, face2_gpu = on_gpu.intersect(starts, bounce, np.inf, face[hit])
    hit2 = face2 >= 0
    print(
        f"  bounces skipping the face: cpu {hit2.sum()}, gpu misses "
        f"{(hit2 & (face2_gpu < 0)).sum()}, "
        f"other faces {(hit2 & (face2_gpu >= 0) & (face2_gpu != face2)).sum()}"
    )
    blocked = bvh.occluded(starts, bounce, np.full(len(starts), 1e30), face[hit])
    blocked_gpu = on_gpu.occluded(starts, bounce, 1e30, face[hit])
    print(
        f"  any hit: cpu {blocked.sum()}, gpu misses {(blocked & ~blocked_gpu).sum()}, "
        f"extra {(~blocked & blocked_gpu).sum()}",
        flush=True,
    )


def child(name: str) -> None:
    _variant(name)
    with np.load(REFERENCE) as stored:
        data = dict(stored)
    vertices, faces = _unit_cube()
    top = np.all(vertices[faces][:, :, 2] == 1.0, axis=1)
    open_vertices, open_faces = vertices * 20.0 - (4.0, 5.0, 0.0), faces[~top]
    print(" open box searches:")
    _searches(open_vertices, open_faces)
    print(" hall searches:")
    _searches(data["hall_vertices"], data["hall_faces"])

    values = reference_arguments(_native, data, "random_detector")
    values |= {
        "directions": data["directions"],
        "bvh": _native.Bvh(vertices * 120.0 - 55.0, faces),
        "absorption": np.full((len(faces), 3), 0.2),
        "scattering": np.full((len(faces), 3), 0.3),
        "receiver_radius": np.full(3, 1.0),
        "rays": 4000,
        "max_time": 0.12,
    }
    _trace("direct sound", values)
    values = reference_arguments(_native, data, "random_detector")
    walls = len(open_faces)
    values |= {
        "bvh": _native.Bvh(open_vertices, open_faces),
        "absorption": np.full((walls, 3), 0.1),
        "scattering": np.full((walls, 3), 0.5),
        "rays": 4000,
    }
    _trace("open box", values)
    for run in ("random_detector", "vector"):
        _trace(f"reference {run}", reference_arguments(_native, data, run))


def main() -> None:
    adapter = gpu._adapter()
    print("adapter:", None if adapter is None else adapter.info, flush=True)
    if adapter is None:
        return
    for name in VARIANTS:
        print(f"variant {name}:", flush=True)
        try:
            subprocess.run([sys.executable, "-u", __file__, name], timeout=150, check=False)
        except subprocess.TimeoutExpired:
            print(f"variant {name}: no answer within 150 s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        main()
