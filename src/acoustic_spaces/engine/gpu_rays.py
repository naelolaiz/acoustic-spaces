"""Ray queries against a room model on a GPU, through WebGPU.

:class:`GpuBvh` uploads the compiled core's bounding volume hierarchy (see
``acoustic_spaces._native.Bvh.layout``) and answers the CPU's batch queries,
``Bvh.intersect`` (the nearest face along each ray) and ``Bvh.occluded`` (whether any
face blocks it), with one WGSL compute shader on any GPU wgpu reaches (see
:mod:`acoustic_spaces.engine.gpu`). It needs the optional ``gpu`` extra. It is the ray
search a tracer on the GPU builds on; ``validation/rays_gpu.py`` times it against the
CPU's.

The GPU works in single precision, where the CPU's double precision needs next to no
slack. So the model is moved to centre its box on the origin (which keeps coordinates
small), every box of the tree is widened by :data:`BOX_PAD` of its own size, and every
triangle by :data:`EDGE` across each edge, so that a ray through an edge or a corner
two faces share hits one of them instead of slipping between. The nearest face along a
ray is then the CPU's, but for rays meeting two faces at nearly the same point, and the
distance agrees to single-precision rounding.
"""

from __future__ import annotations

import numpy as np

from acoustic_spaces.engine import gpu

#: Slack for single precision: each box of the tree grows by this share of its
#: diagonal, and of the model's (rounding of coordinates); each triangle by this share
#: of its own size across each edge (barycentric).
BOX_PAD = 2e-4
MODEL_PAD = 4e-7
EDGE = 1e-4
#: Closest hits count from this share of the model's size on, as on the CPU (1e-9 of it
#: there): a ray starting on a face skips that face, not its neighbours.
T_MIN = 1e-6
#: A face index meaning none: a miss, or no face to skip.
NONE = 0xFFFFFFFF
#: Limit of the rays that have none: far enough for any model, short of overflow.
_REACH = 1e30
#: Threads per workgroup; workgroups per dispatch dimension WebGPU always allows.
_GROUP = 64
_MAX_GROUPS = 65535

#: The tree and the search in WGSL, for any shader that binds ``nodes`` (two vec4<u32>
#: per node: its box's low corner and first child or triangle, then its high corner and
#: triangle count, zero for an inner node) and ``triangles`` (three vec4<u32> per place
#: in leaf order: the three corners, the first with its face) as read-only storage.
BVH_WGSL = """
const NONE: u32 = 0xffffffffu;
const FAR: f32 = 3.0e38;
const EDGE: f32 = __EDGE__;
const STACK: u32 = 64u;

struct Hit {
    t: f32,
    face: u32,
}

// 1 / d per axis, with axes the ray runs (nearly) parallel to kept finite.
fn reciprocal(d: vec3<f32>) -> vec3<f32> {
    let tiny = abs(d) < vec3<f32>(1e-20);
    let away = select(vec3<f32>(1e-20), vec3<f32>(-1e-20), d < vec3<f32>(0.0));
    return 1.0 / select(d, away, tiny);
}

// Where the ray enters the box of `node`, if it does before t_max (else FAR).
fn enter(node: u32, o: vec3<f32>, inv: vec3<f32>, t_max: f32) -> f32 {
    let lo = bitcast<vec3<f32>>(nodes[2u * node].xyz);
    let hi = bitcast<vec3<f32>>(nodes[2u * node + 1u].xyz);
    let ta = (lo - o) * inv;
    let tb = (hi - o) * inv;
    let t_in = min(ta, tb);
    let t_out = max(ta, tb);
    let t0 = max(max(t_in.x, t_in.y), max(t_in.z, 0.0));
    let t1 = min(min(t_out.x, t_out.y), min(t_out.z, t_max));
    return select(FAR, t0, t0 <= t1);
}

// Moller-Trumbore, two-sided, the edges widened by EDGE: t, or a negative value.
fn hit_triangle(place: u32, o: vec3<f32>, d: vec3<f32>) -> f32 {
    let v0 = bitcast<vec3<f32>>(triangles[3u * place].xyz);
    let e1 = bitcast<vec3<f32>>(triangles[3u * place + 1u].xyz) - v0;
    let e2 = bitcast<vec3<f32>>(triangles[3u * place + 2u].xyz) - v0;
    let p = cross(d, e2);
    let det = dot(e1, p);
    let flat = abs(det) < 1e-30;
    let inv_det = 1.0 / select(det, 1.0, flat);
    let s = o - v0;
    let u = dot(s, p) * inv_det;
    let q = cross(s, e1);
    let v = dot(d, q) * inv_det;
    let inside = !flat && u >= -EDGE && u <= 1.0 + EDGE && v >= -EDGE && u + v <= 1.0 + EDGE;
    return select(-1.0, dot(e2, q) * inv_det, inside);
}

// The nearest face along o + t d (d of unit length) with t in (t_min, t_max), skipping
// face `skip`, or (t_max, NONE); with `any_hit`, the first face found, which is enough
// to know the way is blocked. Of two children the ray enters, the nearer goes first;
// the other waits and is dropped if a hit nearer than its box turns up meanwhile.
// (No early exits from the loops: Direct3D's shader compiler got them wrong.)
fn search_ray(o: vec3<f32>, d: vec3<f32>, t_min: f32, t_max: f32, skip: u32,
              any_hit: bool) -> Hit {
    var best = Hit(t_max, NONE);
    let inv = reciprocal(d);
    var waiting_node: array<u32, STACK>;
    var waiting_entry: array<f32, STACK>;
    var top = 0u;
    // The node to look in next; NONE once there is none.
    var node = select(0u, NONE, enter(0u, o, inv, t_max) > t_max);
    while (node != NONE) {
        let low = nodes[2u * node];
        let count = nodes[2u * node + 1u].w;
        var next = NONE;
        if (count > 0u) {
            let end = low.w + count;
            for (var place = low.w; place < end && !(any_hit && best.face != NONE); place++) {
                let face = triangles[3u * place].w;
                if (face != skip) {
                    let t = hit_triangle(place, o, d);
                    if (t > t_min && t < best.t) {
                        best = Hit(t, face);
                    }
                }
            }
            if (any_hit && best.face != NONE) {
                top = 0u;
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
                next = select(right, left, left_first);
            } else if (in_left || in_right) {
                next = select(right, left, in_left);
            }
        }
        // Else the nearest waiting node the ray still reaches before its best hit, if any.
        while (next == NONE && top > 0u) {
            top--;
            if (waiting_entry[top] <= best.t) {
                next = waiting_node[top];
            }
        }
        node = next;
    }
    return best;
}
""".replace("__EDGE__", repr(EDGE))

_SHADER = (
    """
struct Search {
    count: u32,
    any_hit: u32,
    t_min: f32,
    unused: u32,
}

@group(0) @binding(0) var<uniform> search: Search;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
// Per ray: origin and t_max, then direction and the face to skip.
@group(0) @binding(3) var<storage, read> rays: array<vec4<u32>>;
// Per ray: the hit's t and face.
@group(0) @binding(4) var<storage, read_write> hits: array<vec2<u32>>;
"""
    + BVH_WGSL
    + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    let k = id.x + id.y * groups.x * 64u;
    if (k >= search.count) {
        return;
    }
    let a = rays[2u * k];
    let b = rays[2u * k + 1u];
    let hit = search_ray(bitcast<vec3<f32>>(a.xyz), bitcast<vec3<f32>>(b.xyz), search.t_min,
                         bitcast<f32>(a.w), b.w, search.any_hit != 0u);
    hits[k] = vec2<u32>(bitcast<u32>(hit.t), hit.face);
}
"""
)


def _words(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _groups(threads: int) -> tuple[int, int]:
    groups = max(1, -(-threads // _GROUP))
    return min(groups, _MAX_GROUPS), -(-groups // _MAX_GROUPS)


def pack_tree(layout: dict[str, np.ndarray], scale: float):
    """The tree of a ``Bvh.layout()`` as the shader reads it: the centre of the model's
    box (the shader's coordinates are relative to it), the nodes (nodes, 2, 4) and the
    triangles (places, 3, 4), as 32-bit words (see :data:`BVH_WGSL`)."""
    boxes = np.asarray(layout["boxes"], float).reshape(-1, 6)
    lo, hi = boxes[:, :3], boxes[:, 3:]
    centre = 0.5 * (lo[0] + hi[0])
    pad = (BOX_PAD * np.linalg.norm(hi - lo, axis=1) + MODEL_PAD * scale)[:, None]
    nodes = np.zeros((len(boxes), 2, 4), np.uint32)
    nodes[:, 0, :3] = _words(lo - centre - pad)
    nodes[:, 0, 3] = layout["first"]
    nodes[:, 1, :3] = _words(hi - centre + pad)
    nodes[:, 1, 3] = layout["count"]
    edges = np.asarray(layout["triangles"], float).reshape(-1, 3, 3)
    corners = np.stack([edges[:, 0], edges[:, 0] + edges[:, 1], edges[:, 0] + edges[:, 2]], 1)
    triangles = np.zeros((len(corners), 3, 4), np.uint32)
    triangles[:, :, :3] = _words(corners - centre)
    triangles[:, 0, 3] = layout["faces"]
    return centre, nodes, triangles


class RayBatch:
    """Rays uploaded to the GPU (see :meth:`GpuBvh.upload`), to search more than once."""

    def __init__(self, rays, hits, count: int, t_max: np.ndarray) -> None:
        self.rays, self.hits, self.count, self.t_max = rays, hits, count, t_max


class GpuBvh:
    """The tree of ``bvh`` (an :class:`acoustic_spaces._native.Bvh`) on the GPU, with
    the CPU's batch queries."""

    def __init__(self, bvh) -> None:
        wgpu = gpu._wgpu("ray tracing")
        self._device = device = gpu._device()
        #: Length of the model's box diagonal (m).
        self.scale = float(bvh.scale)
        self.faces = int(bvh.face_count)
        #: Where the model's box is centred: the GPU's coordinates are relative to it.
        self.centre, nodes, triangles = pack_tree(bvh.layout(), self.scale)
        usage = wgpu.BufferUsage
        self._nodes = device.create_buffer_with_data(data=nodes, usage=usage.STORAGE)
        self._triangles = device.create_buffer_with_data(data=triangles, usage=usage.STORAGE)
        self._uniforms = [
            device.create_buffer(size=16, usage=usage.UNIFORM | usage.COPY_DST) for _ in (0, 1)
        ]
        self._limit = int(device.limits["max-storage-buffer-binding-size"])

        def entry(binding, kind):
            return {
                "binding": binding,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {"type": kind},
            }

        kinds = wgpu.BufferBindingType
        self._layout = device.create_bind_group_layout(
            entries=[
                entry(0, kinds.uniform),
                entry(1, kinds.read_only_storage),
                entry(2, kinds.read_only_storage),
                entry(3, kinds.read_only_storage),
                entry(4, kinds.storage),
            ]
        )
        self._pipeline = device.create_compute_pipeline(
            layout=device.create_pipeline_layout(bind_group_layouts=[self._layout]),
            compute={"module": device.create_shader_module(code=_SHADER), "entry_point": "main"},
        )

    @property
    def most_rays(self) -> int:
        """The most rays one :class:`RayBatch` holds on this GPU."""
        return self._limit // 32

    def upload(self, origins, directions, t_max, ignore) -> RayBatch:
        """Puts rays on the GPU: ``origins`` and unit ``directions`` (n, 3), how far to
        search (n,; inf for no limit) and a face each one skips (n,; -1 for none)."""
        wgpu = gpu._wgpu("ray tracing")
        origins = np.asarray(origins, float).reshape(-1, 3)
        n = len(origins)
        if n > self.most_rays:
            raise ValueError(f"at most {self.most_rays:,} rays at once on this GPU, not {n:,}")
        t_max = np.minimum(np.broadcast_to(np.asarray(t_max, float), (n,)), _REACH)
        ignore = np.broadcast_to(np.asarray(ignore, np.int64), (n,))
        rays = np.zeros((max(n, 1), 2, 4), np.uint32)
        rays[:n, 0, :3] = _words(origins - self.centre)
        rays[:n, 0, 3] = _words(t_max)
        rays[:n, 1, :3] = _words(np.asarray(directions, float).reshape(-1, 3))
        rays[:n, 1, 3] = np.where(ignore < 0, NONE, ignore).astype(np.uint32)
        usage = wgpu.BufferUsage
        buffer = self._device.create_buffer_with_data(data=rays, usage=usage.STORAGE)
        hits = self._device.create_buffer(size=8 * max(n, 1), usage=usage.STORAGE | usage.COPY_SRC)
        return RayBatch(buffer, hits, n, t_max)

    def search(self, batch: RayBatch, any_hit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Searches the rays of ``batch``: per ray the distance and the face of the
        nearest hit, or with ``any_hit`` of a hit, (inf, -1) for none."""
        device = self._device
        n = batch.count
        if n == 0:
            return np.zeros(0), np.zeros(0, np.int64)
        uniform = self._uniforms[int(any_hit)]
        values = np.array([n, int(any_hit), 0, 0], np.uint32)
        values[2] = _words(np.array([T_MIN * self.scale]))[0]
        device.queue.write_buffer(uniform, 0, values)
        buffers = [uniform, self._nodes, self._triangles, batch.rays, batch.hits]
        group = device.create_bind_group(
            layout=self._layout,
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(buffers)
            ],
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self._pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(*_groups(n))
        compute.end()
        device.queue.submit([encoder.finish()])
        found = np.frombuffer(device.queue.read_buffer(batch.hits), np.uint32)
        found = found[: 2 * n].reshape(n, 2)
        face = found[:, 1].astype(np.int64)
        missed = found[:, 1] == NONE
        face[missed] = -1
        t = found[:, 0].view(np.float32).astype(float)
        t[missed] = np.inf
        return t, face

    def _batches(self, origins, directions, t_max, ignore, any_hit: bool):
        origins = np.asarray(origins, float).reshape(-1, 3)
        directions = np.asarray(directions, float).reshape(-1, 3)
        n = len(origins)
        t_max = np.broadcast_to(np.asarray(t_max, float), (n,))
        ignore = np.broadcast_to(np.asarray(ignore, np.int64), (n,))
        step = self.most_rays
        t, face = np.full(n, np.inf), np.full(n, -1, np.int64)
        for lo in range(0, n, step):
            part = slice(lo, lo + step)
            batch = self.upload(origins[part], directions[part], t_max[part], ignore[part])
            t[part], face[part] = self.search(batch, any_hit)
        return t, face

    def intersect(self, origins, directions, t_max, ignore) -> tuple[np.ndarray, np.ndarray]:
        """As ``Bvh.intersect``: per ray, the nearest hit's distance and face, (inf, -1)
        for a miss."""
        return self._batches(origins, directions, t_max, ignore, False)

    def occluded(self, origins, directions, t_max, ignore) -> np.ndarray:
        """As ``Bvh.occluded``: per ray, whether a face lies within its ``t_max``."""
        return self._batches(origins, directions, t_max, ignore, True)[1] >= 0


__all__ = ["BVH_WGSL", "GpuBvh", "RayBatch", "pack_tree"]
