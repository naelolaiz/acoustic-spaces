"""The late stage's stochastic ray tracer on a GPU, through WebGPU.

:func:`trace_stochastic` takes the arguments of the compiled core's
``acoustic_spaces._native.trace_stochastic`` and returns what it does, computed by one
WGSL compute shader on any GPU wgpu reaches (see :mod:`acoustic_spaces.engine.gpu`). It
traces what the CPU does: detector spheres and diffuse or specular rain, vector-based
or random scattering, the scattering of a surface's size, air absorption, the early
stage's paths left out, arrival directions and the weight window that splits rays and
thins them out. It leaves to the CPU (raising :class:`Unsupported`) sound through
surfaces into other rooms and the face groups of material calibration.

Rays start where the CPU's do (``_native.ray_starts``: the same directions and energies
for the same seed), then go their own way: the GPU draws its random numbers from
streams of its own (xoshiro128**) and works in single precision, so the two agree
statistically, not ray for ray. Each ray adds its energy to the listeners' time slots
with atomic additions of 64-bit fixed-point numbers (two 32-bit words with a carry,
rounded at random to the nearest step, so that sums are exact and unbiased; WGSL has no
atomic floating-point addition), scaled so that no sum can come near overflowing.

The rays advance in rounds of a few reflections each, so that no dispatch runs long
enough for the operating system to reset the GPU. Rays still going at the end of a
round, and the copies split off during it, are packed into the other half of a buffer
for the next round, and new rays from the source fill it up to half its size; the
other half takes the copies, and a split that finds no room left is not made (which
leaves every expectation as it was). A ray's copies so advance together, where the CPU
follows each to its end before it takes the next: a family that reaches its limit
(``max_family``) reaches it sooner here, its copies made earlier on their way. That
leaves the expected energies as they were too, but where the weight window asks for
more copies than a family may have, the GPU traces more reflections in the late tail
than the CPU (in the shoebox example, up to twice as many), and its tail varies less.
"""

from __future__ import annotations

import functools
import math
import threading

import numpy as np

from acoustic_spaces.engine import gpu
from acoustic_spaces.engine.gpu_rays import BVH_WGSL, T_MIN, pack_tree


class Unsupported(ValueError):
    """The GPU tracer does not do what a run asks for; the CPU's does."""


#: Threads per workgroup; workgroups per dispatch dimension WebGPU always allows.
_GROUP = 64
_MAX_GROUPS = 65535
#: Most rays in flight (each half of the state buffer), and ray searches per dispatch
#: (fewer reflections per round where many rays or listeners would take longer).
_MOST_RAYS = 1 << 18
_SEARCHES = 1 << 23
_MOST_LEGS = 32
#: Fixed point: the largest single addition the energy can make is 2**_FRACTION steps.
_FRACTION = 40
#: A ray starts its next leg this share of the model's size off the face it left (the
#: CPU's rain starts 1e-7 of it off): enough for single precision.
_OFFSET = 1e-5
#: Words of the counters at the start of the tallies, and per time window.
_COUNTERS = 8
_WINDOW = 4
#: Flags of the run.
_DIFFUSE_RAIN, _SPECULAR_RAIN, _VECTOR, _SIZES, _SPLITTING = 1, 2, 4, 8, 16

#: The run's settings, one 32-bit word each, in the order of ``Run`` in the shader.
_FIELDS = (
    ("kept", "u"),
    ("spawned", "u"),
    ("fresh", "u"),
    ("fresh_from", "u"),
    ("in_at", "u"),
    ("spawn_from", "u"),
    ("out_at", "u"),
    ("spawn_room", "u"),
    ("receivers", "u"),
    ("directions", "u"),
    ("windows", "u"),
    ("bins", "u"),
    ("dir_bins", "u"),
    ("legs", "u"),
    ("flags", "u"),
    ("covered_order", "i"),
    ("size_from_order", "i"),
    ("max_split", "u"),
    ("max_family", "u"),
    ("seed", "u"),
    ("face_at", "u"),
    ("receiver_at", "u"),
    ("direction_at", "u"),
    ("air_at", "u"),
    ("hz_at", "u"),
    ("target_at", "u"),
    ("targets", "u"),
    ("energy_at", "u"),
    ("directional_at", "u"),
    ("family_at", "u"),
    ("ray_seed", "u"),
    ("unused_a", "u"),
    ("max_length", "f"),
    ("c", "f"),
    ("time_step", "f"),
    ("dir_time_step", "f"),
    ("window_step", "f"),
    ("split_ratio", "f"),
    ("t_min", "f"),
    ("offset", "f"),
    ("scale", "f"),
    ("window_scale", "f"),
    ("unused_b", "f"),
    ("unused_c", "f"),
)
_TYPES = {"u": "u32", "i": "i32", "f": "f32"}

_SHADER = (
    """
const BANDS: u32 = __BANDS__u;
// Words per ray in flight and per ray from the source; words per face and per listener.
const RECORD: u32 = 16u + 3u * BANDS;
const START: u32 = 3u + BANDS;
const FACE: u32 = 5u + 2u * BANDS;
const LISTENER: u32 = 6u;
const PI: f32 = 3.14159265358979;
const COUNTERS: u32 = __COUNTERS__u;
const DIFFUSE_RAIN: u32 = 1u;
const SPECULAR_RAIN: u32 = 2u;
const VECTOR: u32 = 4u;
const SIZES: u32 = 8u;
const SPLITTING: u32 = 16u;

struct Run {
"""
    + "".join(f"    {name}: {_TYPES[kind]},\n" for name, kind in _FIELDS)
    + """}

@group(0) @binding(0) var<uniform> run: Run;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
// The source, then per face its normal, size, absorption and scattering; per listener
// its place, sphere radius and volume and how long the early stage covers; the arrival
// directions; air absorption and band centres; the weight window's expected energies.
@group(0) @binding(3) var<storage, read> scene: array<u32>;
// Per ray from the source: its direction and energy per band.
@group(0) @binding(4) var<storage, read> starts: array<u32>;
// Rays in flight: two halves, read from one and written to the other in turn.
@group(0) @binding(5) var<storage, read_write> states: array<u32>;
// Counters (rays kept and copies made this round, escaped, traced, overflow), then per
// window the rays' energy and reflections, the energy per listener, band and slot, per
// listener, direction, band and slot, and the copies made of each ray from the source.
@group(0) @binding(6) var<storage, read_write> tallies: array<atomic<u32>>;
"""
    + BVH_WGSL
    + """
struct Ray {
    pos: vec3<f32>,
    travelled: f32,
    dir: vec3<f32>,
    last: u32,
    rng: vec4<u32>,
    order: i32,
    root: u32,
    cutoff: f32,
    energy: array<f32, BANDS>,
    // The share of the energy the detector spheres count on this leg, and the share
    // that has only ever reflected specularly.
    seen: array<f32, BANDS>,
    alone: array<f32, BANDS>,
}

var<private> ray: Ray;
var<private> counted: array<f32, BANDS>;
var<private> reflected: array<f32, BANDS>;
var<private> scattered: array<f32, BANDS>;
var<private> scatter: array<f32, BANDS>;

fn num(at: u32) -> f32 {
    return bitcast<f32>(scene[at]);
}

fn point(at: u32) -> vec3<f32> {
    return vec3<f32>(num(at), num(at + 1u), num(at + 2u));
}

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

// xoshiro128**.
fn next_u32() -> u32 {
    let s = ray.rng;
    let result = rotl(s.y * 5u, 7u) * 9u;
    var n = s;
    n.z ^= n.x;
    n.w ^= n.y;
    n.y ^= n.z;
    n.x ^= n.w;
    n.z ^= s.y << 9u;
    n.w = rotl(n.w, 11u);
    ray.rng = n;
    return result;
}

// Uniform in [0, 1).
fn random() -> f32 {
    return f32(next_u32() >> 8u) * (1.0 / 16777216.0);
}

fn mix32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn seeded(a: u32, b: u32, c: u32) -> vec4<u32> {
    let h0 = mix32(a ^ mix32(b ^ mix32(c)));
    let h1 = mix32(h0 ^ 0x9e3779b9u);
    let h2 = mix32(h1 ^ 0x9e3779b9u);
    let h3 = mix32(h2 ^ 0x9e3779b9u);
    return vec4<u32>(h0, h1, h2, h3 | 1u);
}

// A new, independent stream for a copy split off the ray.
fn split_rng() -> vec4<u32> {
    let a = next_u32();
    let b = next_u32();
    return seeded(a, b, 0x632be5abu);
}

// Lambertian (cosine-weighted) direction on the hemisphere round unit normal n.
fn lambert(n: vec3<f32>) -> vec3<f32> {
    let u1 = random();
    let phi = 2.0 * PI * random();
    let r = sqrt(u1);
    let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
    let t = normalize(cross(n, helper));
    let b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - u1)));
}

// Adds `value` (scaled) to the 64-bit fixed-point number at words at and at + 1, rounded
// at random to a whole step.
fn add_fixed(at: u32, value: f32) {
    if (!(value > 0.0)) {
        return;
    }
    var v = value;
    if (v < 16777216.0) {
        v = floor(v + random());
        if (v == 0.0) {
            return;
        }
    }
    if (v >= 9.2e18) {
        atomicStore(&tallies[4], 1u);
        return;
    }
    let high = floor(v * (1.0 / 4294967296.0));
    let low = u32(v - high * 4294967296.0);
    let old = atomicAdd(&tallies[at], low);
    let up = u32(high) + select(0u, 1u, old > 0xffffffffu - low);
    if (up != 0u) {
        if (atomicAdd(&tallies[at + 1u], up) >= 0x80000000u) {
            atomicStore(&tallies[4], 1u);
        }
    }
}

fn add_one(at: u32) {
    if (atomicAdd(&tallies[at], 1u) == 0xffffffffu) {
        atomicAdd(&tallies[at + 1u], 1u);
    }
}

// Adds counted[b] * factor * air(travel) arriving from `arrival` `travel` metres from
// the source to listener k.
fn deposit(k: u32, factor: f32, travel: f32, arrival: vec3<f32>) {
    let time = travel / run.c;
    let bin = u32(max(time, 0.0) / run.time_step);
    if (bin >= run.bins) {
        return;
    }
    var drow = NONE;
    if (run.directions > 0u) {
        var best = 0u;
        var best_dot = -2.0;
        for (var d = 0u; d < run.directions; d++) {
            let v = dot(arrival, point(run.direction_at + 3u * d));
            if (v > best_dot) {
                best_dot = v;
                best = d;
            }
        }
        let dbin = min(run.dir_bins - 1u, u32(max(time, 0.0) / run.dir_time_step));
        let cell = (k * run.directions + best) * BANDS * run.dir_bins + dbin;
        drow = run.directional_at + 2u * cell;
    }
    let row = run.energy_at + 2u * (k * BANDS * run.bins + bin);
    for (var b = 0u; b < BANDS; b++) {
        let e = counted[b] * factor * exp(-num(run.air_at + b) * travel) * run.scale;
        add_fixed(row + 2u * b * run.bins, e);
        if (drow != NONE) {
            add_fixed(drow + 2u * b * run.dir_bins, e);
        }
    }
}

// The detector spheres along the leg from ray.pos for `segment` metres.
fn detectors(segment: f32) {
    for (var k = 0u; k < run.receivers; k++) {
        let at = run.receiver_at + LISTENER * k;
        let rel = point(at) - ray.pos;
        let s = dot(rel, ray.dir);
        let d2 = dot(rel, rel) - s * s;
        let r = num(at + 3u);
        if (d2 >= r * r) {
            continue;
        }
        let chord = sqrt(r * r - d2);
        let t0 = max(s - chord, 0.0);
        let t1 = min(s + chord, segment);
        if (t1 <= t0) {
            continue;
        }
        let travel = ray.travelled + clamp(s, t0, t1);
        let covered = ray.order <= run.covered_order && travel / run.c <= num(at + 5u);
        var any = 0.0;
        for (var b = 0u; b < BANDS; b++) {
            counted[b] = ray.energy[b] * max(ray.seen[b] - select(0.0, ray.alone[b], covered), 0.0);
            any += counted[b];
        }
        if (any > 0.0) {
            deposit(k, (t1 - t0) / num(at + 4u), travel, -ray.dir);
        }
    }
}

// The rain from the reflection at ray.pos (normal n towards the ray's side) to every
// listener that sees it.
fn rain(n: vec3<f32>) {
    let specular = (run.flags & SPECULAR_RAIN) != 0u;
    let origin = ray.pos + n * run.offset;
    for (var k = 0u; k < run.receivers; k++) {
        let at = run.receiver_at + LISTENER * k;
        let v = point(at) - origin;
        let gap = length(v);
        let dist = max(gap, num(at + 3u));
        let cos_theta = dot(n, v) / dist;
        if (cos_theta <= 0.0 || gap <= 0.0) {
            continue;
        }
        let toward = v / gap;
        if (search_ray(origin, toward, run.t_min, gap, ray.last, true).face != NONE) {
            continue;
        }
        let covered = specular && ray.order <= run.covered_order &&
                      (ray.travelled + dist) / run.c <= num(at + 5u);
        for (var b = 0u; b < BANDS; b++) {
            if (specular) {
                counted[b] = reflected[b] * select(1.0, 1.0 - ray.alone[b], covered);
            } else {
                counted[b] = scattered[b];
            }
        }
        deposit(k, cos_theta / (PI * dist * dist), ray.travelled + dist, -toward);
    }
}

fn load(at: u32) {
    ray.pos = bitcast<vec3<f32>>(vec3<u32>(states[at], states[at + 1u], states[at + 2u]));
    ray.travelled = bitcast<f32>(states[at + 3u]);
    ray.dir = bitcast<vec3<f32>>(vec3<u32>(states[at + 4u], states[at + 5u], states[at + 6u]));
    ray.last = states[at + 7u];
    ray.rng = vec4<u32>(states[at + 8u], states[at + 9u], states[at + 10u], states[at + 11u]);
    ray.order = bitcast<i32>(states[at + 12u]);
    ray.root = states[at + 13u];
    ray.cutoff = bitcast<f32>(states[at + 14u]);
    for (var b = 0u; b < BANDS; b++) {
        ray.energy[b] = bitcast<f32>(states[at + 16u + b]);
        ray.seen[b] = bitcast<f32>(states[at + 16u + BANDS + b]);
        ray.alone[b] = bitcast<f32>(states[at + 16u + 2u * BANDS + b]);
    }
}

fn store(at: u32) {
    let pos = bitcast<vec3<u32>>(ray.pos);
    let dir = bitcast<vec3<u32>>(ray.dir);
    states[at] = pos.x;
    states[at + 1u] = pos.y;
    states[at + 2u] = pos.z;
    states[at + 3u] = bitcast<u32>(ray.travelled);
    states[at + 4u] = dir.x;
    states[at + 5u] = dir.y;
    states[at + 6u] = dir.z;
    states[at + 7u] = ray.last;
    states[at + 8u] = ray.rng.x;
    states[at + 9u] = ray.rng.y;
    states[at + 10u] = ray.rng.z;
    states[at + 11u] = ray.rng.w;
    states[at + 12u] = bitcast<u32>(ray.order);
    states[at + 13u] = ray.root;
    states[at + 14u] = bitcast<u32>(ray.cutoff);
    states[at + 15u] = 0u;
    for (var b = 0u; b < BANDS; b++) {
        states[at + 16u + b] = bitcast<u32>(ray.energy[b]);
        states[at + 16u + BANDS + b] = bitcast<u32>(ray.seen[b]);
        states[at + 16u + 2u * BANDS + b] = bitcast<u32>(ray.alone[b]);
    }
}

// Ray `k` from the source; false if the source sends nothing its way.
fn start(k: u32) -> bool {
    let at = k * START;
    var loudest = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        let e = bitcast<f32>(starts[at + 3u + b]);
        ray.energy[b] = e;
        ray.seen[b] = 1.0;
        ray.alone[b] = 1.0;
        loudest = max(loudest, e);
    }
    if (!(loudest > 0.0)) {
        return false;
    }
    ray.pos = point(0u);
    ray.travelled = 0.0;
    ray.dir = bitcast<vec3<f32>>(vec3<u32>(starts[at], starts[at + 1u], starts[at + 2u]));
    ray.last = NONE;
    ray.rng = seeded(run.seed, run.ray_seed + k, 0x85ebca6bu);
    ray.order = 0;
    ray.root = k;
    ray.cutoff = loudest * 1e-14;
    atomicAdd(&tallies[3], 1u);
    return true;
}

// The ray's next leg and reflection; false once it is done. `n_in` is the number of
// rays this round, where the copies' place starts.
fn advance(n_in: u32) -> bool {
    if (ray.travelled >= run.max_length) {
        return false;
    }
    let reach = run.max_length - ray.travelled;
    let hit = search_ray(ray.pos, ray.dir, run.t_min, reach, ray.last, false);
    let found = hit.face != NONE;
    let segment = select(reach, hit.t, found);
    if ((run.flags & SPECULAR_RAIN) == 0u) {
        detectors(segment);
    }
    if (!found) {
        return false;
    }
    ray.travelled += hit.t;
    ray.pos += ray.dir * hit.t;
    let face = run.face_at + FACE * hit.face;
    var n = point(face);
    if (dot(n, ray.dir) > 0.0) {
        n = -n;
    }
    ray.last = hit.face;
    ray.order += 1;
    let sizes = (run.flags & SIZES) != 0u && ray.order >= run.size_from_order;
    var f_width = 0.0;
    var f_length = 0.0;
    if (sizes) {
        // Christensen and Rindel's scattering of a surface's limited size.
        let across = max(num(face + 3u) * max(-dot(n, ray.dir), 1e-3), 1e-9);
        let along = max(num(face + 4u), 1e-9);
        f_width = run.c * hit.t / (2.0 * across * across);
        f_length = run.c * hit.t / (2.0 * along * along);
    }
    var peak = 0.0;
    var mean = 0.0;
    var p = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        let surface = num(face + 5u + BANDS + b);
        if (sizes) {
            let hz = num(run.hz_at + b);
            let specular = min(1.0, hz / f_width) * min(1.0, hz / f_length);
            scatter[b] = 1.0 - (1.0 - surface) * specular;
        } else {
            scatter[b] = surface;
        }
        mean += ray.energy[b];
        reflected[b] = ray.energy[b] * (1.0 - num(face + 5u + b));
        scattered[b] = reflected[b] * scatter[b];
        peak = max(peak, reflected[b]);
        p += scatter[b];
        ray.alone[b] *= 1.0 - scatter[b];
    }
    p /= f32(BANDS);
    let window = min(run.windows - 1u, u32(ray.travelled / run.c / run.window_step));
    add_fixed(COUNTERS + 4u * window, mean / f32(BANDS) * run.window_scale);
    add_one(COUNTERS + 4u * window + 2u);
    if ((run.flags & (DIFFUSE_RAIN | SPECULAR_RAIN)) != 0u) {
        rain(n);
    }
    if (peak < ray.cutoff) {
        return false;
    }
    let diffuse_rain = (run.flags & DIFFUSE_RAIN) != 0u;
    if ((run.flags & VECTOR) != 0u) {
        // One ray for the mixture: all the reflected energy, in between the specular and
        // a Lambert direction.
        let mixed = reflect(ray.dir, n) * (1.0 - p) + lambert(n) * p;
        let size = length(mixed);
        ray.dir = select(n, mixed / size, size > 1e-12);
        for (var b = 0u; b < BANDS; b++) {
            ray.energy[b] = reflected[b];
            ray.seen[b] = select(1.0, 1.0 - scatter[b], diffuse_rain);
        }
    } else if (random() < p) {
        for (var b = 0u; b < BANDS; b++) {
            ray.energy[b] = scattered[b] / p;
            ray.seen[b] = select(1.0, 0.0, diffuse_rain);
            ray.alone[b] = 0.0;
        }
        ray.dir = lambert(n);
    } else {
        for (var b = 0u; b < BANDS; b++) {
            ray.energy[b] = (reflected[b] - scattered[b]) / (1.0 - p);
            ray.seen[b] = 1.0;
            ray.alone[b] = select(0.0, 1.0, ray.alone[b] > 0.0);
        }
        ray.dir = reflect(ray.dir, n);
    }
    // The next leg starts just off the face.
    ray.pos += n * run.offset;
    if ((run.flags & SPLITTING) == 0u) {
        return true;
    }
    let expected = num(run.target_at + min(window, run.targets - 1u));
    if (!(expected > 0.0)) {
        return true;
    }
    var now = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        now += ray.energy[b];
    }
    now /= f32(BANDS);
    if (now > run.split_ratio * expected) {
        let want = u32(min(ceil(now / expected), f32(run.max_split)));
        if (want < 2u) {
            return true;
        }
        // The ray and its copies number at most max_family.
        let family = 1u + atomicAdd(&tallies[run.family_at + ray.root], want - 1u);
        var extra = 0u;
        if (family < run.max_family) {
            extra = min(want - 1u, run.max_family - family);
        }
        if (extra == 0u) {
            return true;
        }
        let slot = atomicAdd(&tallies[1], extra);
        if (slot >= run.spawn_room) {
            return true;
        }
        extra = min(extra, run.spawn_room - slot);
        for (var b = 0u; b < BANDS; b++) {
            ray.energy[b] /= f32(extra + 1u);
        }
        for (var i = 0u; i < extra; i++) {
            let child = split_rng();
            let mine = ray.rng;
            ray.rng = child;
            store(run.out_at + (n_in + slot + i) * RECORD);
            ray.rng = mine;
        }
        atomicAdd(&tallies[3], extra);
    } else if (now * run.split_ratio < expected) {
        let survive = now / expected;
        if (random() >= survive) {
            return false;
        }
        for (var b = 0u; b < BANDS; b++) {
            ray.energy[b] /= survive;
        }
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * 64u;
    let n_in = run.kept + run.spawned + run.fresh;
    if (i >= n_in) {
        return;
    }
    if (i < run.kept) {
        load(run.in_at + i * RECORD);
    } else if (i < run.kept + run.spawned) {
        load(run.in_at + (run.spawn_from + i - run.kept) * RECORD);
    } else if (!start(run.fresh_from + i - run.kept - run.spawned)) {
        return;
    }
    for (var leg = 0u; leg < run.legs; leg++) {
        if (!advance(n_in)) {
            return;
        }
    }
    store(run.out_at + atomicAdd(&tallies[0], 1u) * RECORD);
}
"""
).replace("__COUNTERS__", str(_COUNTERS))


def _words(values) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)


def _groups(threads: int) -> tuple[int, int]:
    groups = max(1, -(-threads // _GROUP))
    return min(groups, _MAX_GROUPS), -(-groups // _MAX_GROUPS)


def _uniform(values: dict) -> np.ndarray:
    out = np.zeros(len(_FIELDS), np.uint32)
    for i, (name, kind) in enumerate(_FIELDS):
        value = values.get(name, 0)
        if kind == "f":
            out[i] = _words(np.array([value]))[0]
        elif kind == "i":
            out[i] = np.array([value], np.int32).view(np.uint32)[0]
        else:
            out[i] = int(value)
    return out


def _fixed(words: np.ndarray) -> np.ndarray:
    """64-bit fixed-point numbers (pairs of 32-bit words, low first) as floats."""
    pairs = words.reshape(-1, 2).astype(np.float64)
    return pairs[:, 0] + pairs[:, 1] * 4294967296.0


@functools.lru_cache(maxsize=1)
def _bindings():
    """The layout of the shader's buffers."""
    wgpu = gpu._wgpu("ray tracing")
    kinds = wgpu.BufferBindingType

    def entry(binding, kind):
        return {
            "binding": binding,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": kind},
        }

    return gpu._device().create_bind_group_layout(
        entries=[
            entry(0, kinds.uniform),
            *(entry(i, kinds.read_only_storage) for i in (1, 2, 3, 4)),
            entry(5, kinds.storage),
            entry(6, kinds.storage),
        ]
    )


@functools.lru_cache(maxsize=8)
def _pipeline(bands: int):
    """The shader for rays of ``bands`` bands, compiled once per session."""
    device = gpu._device()
    code = _SHADER.replace("__BANDS__", str(bands))
    return device.create_compute_pipeline(
        layout=device.create_pipeline_layout(bind_group_layouts=[_bindings()]),
        compute={"module": device.create_shader_module(code=code), "entry_point": "main"},
    )


class GpuTracer:
    """The tree of ``bvh`` (an :class:`acoustic_spaces._native.Bvh`) and its faces'
    normals on the GPU, to trace rays in."""

    def __init__(self, bvh) -> None:
        wgpu = gpu._wgpu("ray tracing")
        self._device = device = gpu._device()
        self.scale = float(bvh.scale)
        self.faces = int(bvh.face_count)
        layout = bvh.layout()
        self.centre, nodes, triangles = pack_tree(layout, self.scale)
        edges = np.asarray(layout["triangles"], float).reshape(-1, 3, 3)
        normal = np.cross(edges[:, 1], edges[:, 2])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
        self._normals = np.zeros((self.faces, 3))
        self._normals[layout["faces"]] = normal
        usage = wgpu.BufferUsage
        self._nodes = device.create_buffer_with_data(data=nodes, usage=usage.STORAGE)
        self._triangles = device.create_buffer_with_data(data=triangles, usage=usage.STORAGE)
        self._uniform = device.create_buffer(
            size=4 * len(_FIELDS), usage=usage.UNIFORM | usage.COPY_DST
        )
        self._counters = device.create_buffer(
            size=4 * _COUNTERS, usage=usage.MAP_READ | usage.COPY_DST
        )
        self._limit = int(device.limits["max-storage-buffer-binding-size"])
        # One run at a time: the uniforms and counters are shared.
        self._busy = threading.Lock()

    def trace(self, **arguments) -> tuple:
        """Traces rays (see :meth:`_trace` for the arguments), one run at a time."""
        with self._busy:
            return self._trace(**arguments)

    def _trace(
        self,
        *,
        source,
        starts: tuple[np.ndarray, np.ndarray],
        total: int,
        absorption,
        scattering,
        air,
        surface_size,
        band_hz,
        vector_scattering: bool,
        size_from_order: int,
        receivers,
        directions,
        max_time: float,
        time_step: float,
        dir_time_step: float,
        speed_of_sound: float,
        receiver_radius,
        diffuse_rain: bool,
        specular_rain: bool,
        covered_order: int,
        covered_time,
        seed: int,
        window_time_step: float,
        first_ray: int,
        split_target,
        split_ratio: float,
        max_split: int,
        max_family: int,
    ) -> tuple:
        """Traces the rays ``starts`` (directions and the source's energy weights, as
        ``_native.ray_starts`` gives them for the streams from ``first_ray`` on), each
        carrying 1 / ``total`` of the source's energy; the rest as for
        ``_native.trace_stochastic``. Returns (energy, directional, escaped rays,
        ray_energy, reflections, traced rays)."""
        wgpu = gpu._wgpu("ray tracing")
        device = self._device
        dirs, weights = (np.asarray(a, float) for a in starts)
        rays, bands = weights.shape
        absorption = np.asarray(absorption, float).reshape(self.faces, bands)
        scattering = np.asarray(scattering, float).reshape(self.faces, bands)
        receivers = np.asarray(receivers, float).reshape(-1, 3)
        directions = np.asarray(directions, float).reshape(-1, 3)
        radius = np.broadcast_to(np.asarray(receiver_radius, float), (len(receivers),))
        n_rec, n_dirs = len(receivers), len(directions)
        bins = math.ceil(max_time / time_step)
        dir_bins = math.ceil(max_time / dir_time_step)
        windows = math.ceil(max_time / window_time_step)
        energy_out = np.zeros((n_rec, bands, bins))
        directional_out = np.zeros((n_rec, n_dirs, bands, dir_bins))
        energies = weights / total
        loudest = float(energies.max(initial=0.0))
        if rays == 0 or loudest <= 0.0:
            return energy_out, directional_out, 0, np.zeros(windows), np.zeros(windows), 0

        # The scene: source, faces, listeners, directions, air, band centres, targets.
        sizes = np.asarray(surface_size, float).reshape(-1, 2)
        with_sizes = len(sizes) > 0
        if not with_sizes:
            sizes = np.zeros((self.faces, 2))
        faces = np.hstack([self._normals, sizes, absorption, scattering])
        covered = np.asarray(covered_time, float)
        if covered.size == 0:
            covered = np.full(n_rec, np.inf)
        listeners = np.column_stack(
            [receivers - self.centre, radius, 4.0 / 3.0 * math.pi * radius**3, covered]
        )
        target = np.asarray(split_target, float)
        hz = np.asarray(band_hz, float) if with_sizes else np.zeros(bands)
        parts = [
            np.asarray(source, float) - self.centre,
            faces.ravel(),
            listeners.ravel(),
            directions.ravel(),
            np.asarray(air, float),
            hz,
            target,
        ]
        offsets = np.cumsum([0, *(len(p) for p in parts)])
        values = np.concatenate(parts)
        values = np.minimum(values, np.finfo(np.float32).max)  # no time limit: the largest
        scene = _words(values)

        # The tallies: counters, windows, energy, directional, families.
        energy_at = _COUNTERS + _WINDOW * windows
        directional_at = energy_at + 2 * n_rec * bands * bins
        family_at = directional_at + 2 * n_rec * n_dirs * bands * dir_bins
        tally_words = family_at + rays
        record = 16 + 3 * bands
        most = min(_MOST_RAYS, self._limit // (8 * record)) // _GROUP * _GROUP
        if 4 * tally_words > self._limit or most < _GROUP:
            raise Unsupported("the GPU's buffers cannot hold this run's results")

        usage = wgpu.BufferUsage
        start_words = np.hstack([dirs, energies]).astype(np.float32).view(np.uint32)
        buffers = [
            self._uniform,
            self._nodes,
            self._triangles,
            device.create_buffer_with_data(data=scene, usage=usage.STORAGE),
            device.create_buffer_with_data(data=start_words, usage=usage.STORAGE),
            device.create_buffer(size=8 * most * record, usage=usage.STORAGE),
            device.create_buffer(
                size=4 * tally_words, usage=usage.STORAGE | usage.COPY_SRC | usage.COPY_DST
            ),
        ]
        tallies = buffers[-1]
        group = device.create_bind_group(
            layout=_bindings(),
            entries=[
                {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                for i, b in enumerate(buffers)
            ],
        )
        pipeline = _pipeline(bands)
        r_min = float(radius.min(initial=1.0))
        settings = {
            "receivers": n_rec,
            "directions": n_dirs,
            "windows": windows,
            "bins": bins,
            "dir_bins": dir_bins,
            "flags": (_DIFFUSE_RAIN if diffuse_rain else 0)
            | (_SPECULAR_RAIN if specular_rain else 0)
            | (_VECTOR if vector_scattering else 0)
            | (_SIZES if with_sizes else 0)
            | (_SPLITTING if len(target) else 0),
            "covered_order": covered_order,
            "size_from_order": size_from_order,
            "max_split": max_split,
            "max_family": max_family,
            "seed": (seed ^ (seed >> 32)) & 0xFFFFFFFF,
            "face_at": offsets[1],
            "receiver_at": offsets[2],
            "direction_at": offsets[3],
            "air_at": offsets[4],
            "hz_at": offsets[5],
            "target_at": offsets[6],
            "targets": len(target),
            "energy_at": energy_at,
            "directional_at": directional_at,
            "family_at": family_at,
            "ray_seed": first_ray & 0xFFFFFFFF,
            "max_length": speed_of_sound * max_time,
            "c": speed_of_sound,
            "time_step": time_step,
            "dir_time_step": dir_time_step,
            "window_step": window_time_step,
            "split_ratio": split_ratio,
            "t_min": T_MIN * self.scale,
            "offset": _OFFSET * self.scale,
            # The most one addition can bring (a ray at a listener's sphere) is 2**40.
            "scale": 2.0**_FRACTION * math.pi * r_min**2 / (1.5 * loudest),
            "window_scale": 2.0**_FRACTION / loudest,
        }
        halves = (0, most * record)
        kept = spawned = spawn_from = fresh_from = rounds = 0
        counters = np.zeros(_COUNTERS, np.uint32)
        while True:
            fresh = min(rays - fresh_from, max(0, most // 2 - kept - spawned))
            n = kept + spawned + fresh
            if n == 0:
                break
            legs = max(1, min(_MOST_LEGS, _SEARCHES // (n * (1 + n_rec))))
            values = settings | {
                "kept": kept,
                "spawned": spawned,
                "fresh": fresh,
                "fresh_from": fresh_from,
                "in_at": halves[rounds % 2],
                "spawn_from": spawn_from,
                "out_at": halves[(rounds + 1) % 2],
                "spawn_room": most - n,
                "legs": legs,
            }
            device.queue.write_buffer(self._uniform, 0, _uniform(values))
            encoder = device.create_command_encoder()
            encoder.clear_buffer(tallies, 0, 8)
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(pipeline)
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(*_groups(n))
            compute.end()
            encoder.copy_buffer_to_buffer(tallies, 0, self._counters, 0, 4 * _COUNTERS)
            device.queue.submit([encoder.finish()])
            self._counters.map_sync(wgpu.MapMode.READ)
            counters = np.frombuffer(self._counters.read_mapped(), np.uint32).copy()
            self._counters.unmap()
            kept, spawned = int(counters[0]), min(int(counters[1]), most - n)
            spawn_from, fresh_from, rounds = n, fresh_from + fresh, rounds + 1
        if counters[4]:
            raise Unsupported("the GPU's sums overflowed")
        words = np.frombuffer(device.queue.read_buffer(tallies, 0, 4 * family_at), np.uint32)
        window = words[_COUNTERS:energy_at].reshape(windows, 2, 2)
        ray_energy = _fixed(window[:, 0]) / settings["window_scale"]
        reflections = _fixed(window[:, 1])
        energy_out[:] = (_fixed(words[energy_at:directional_at]) / settings["scale"]).reshape(
            energy_out.shape
        )
        directional_out[:] = (_fixed(words[directional_at:family_at]) / settings["scale"]).reshape(
            directional_out.shape
        )
        return (
            energy_out,
            directional_out,
            int(counters[2]),
            ray_energy,
            reflections,
            int(counters[3]),
        )


@functools.lru_cache(maxsize=2)
def tracer(bvh) -> GpuTracer:
    """The GPU tracer of ``bvh``, made once per model."""
    return GpuTracer(bvh)


def trace_stochastic(
    bvh,
    source,
    directivity_a,
    orientation,
    directivity_table,
    absorption,
    scattering,
    air,
    surface_size,
    band_hz,
    vector_scattering,
    size_from_order,
    transmission,
    face_room,
    source_room,
    receivers,
    directions,
    rays,
    max_time,
    time_step,
    dir_time_step,
    speed_of_sound,
    receiver_radius,
    diffuse_rain,
    specular_rain,
    covered_order,
    covered_time,
    seed,
    threads,
    window_time_step,
    first_ray,
    split_target,
    split_ratio,
    max_split,
    max_family,
    room_level,
    face_group,
    groups,
    moment_time_step,
    *,
    total: int | None = None,
) -> tuple:
    """As ``acoustic_spaces._native.trace_stochastic``, on the GPU. ``total`` (default
    ``rays``) is the number of rays the source's energy is shared between, for tracing
    part of a run's rays here and the rest elsewhere. Raises :class:`Unsupported` for
    what only the CPU traces: transmission and face groups."""
    from acoustic_spaces import _native

    del threads, face_room, source_room, room_level, face_group, moment_time_step
    if np.asarray(transmission).size:
        raise Unsupported("the GPU tracer does not let sound through surfaces")
    if groups:
        raise Unsupported("the GPU tracer does not count reflections per face group")
    bands = len(np.asarray(air))
    starts = _native.ray_starts(
        np.ascontiguousarray(directivity_a, dtype=np.float64),
        np.ascontiguousarray(orientation, dtype=np.float64),
        np.ascontiguousarray(directivity_table, dtype=np.float64),
        bands,
        int(rays),
        int(first_ray),
        int(seed),
    )
    energy, directional, escaped, ray_energy, reflections, traced = tracer(bvh).trace(
        source=source,
        starts=starts,
        total=int(rays) if total is None else int(total),
        absorption=absorption,
        scattering=scattering,
        air=air,
        surface_size=surface_size,
        band_hz=band_hz,
        vector_scattering=bool(vector_scattering),
        size_from_order=int(size_from_order),
        receivers=receivers,
        directions=directions,
        max_time=float(max_time),
        time_step=float(time_step),
        dir_time_step=float(dir_time_step),
        speed_of_sound=float(speed_of_sound),
        receiver_radius=receiver_radius,
        diffuse_rain=bool(diffuse_rain),
        specular_rain=bool(specular_rain),
        covered_order=int(covered_order),
        covered_time=covered_time,
        seed=int(seed),
        window_time_step=float(window_time_step),
        first_ray=int(first_ray),
        split_target=split_target,
        split_ratio=float(split_ratio),
        max_split=int(max_split),
        max_family=int(max_family),
    )
    moments = np.zeros((len(np.atleast_2d(receivers)), 0, bands, 0))
    return energy, directional, moments, escaped, ray_energy, reflections, traced


__all__ = ["GpuTracer", "Unsupported", "trace_stochastic", "tracer"]
