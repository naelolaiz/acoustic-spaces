"""The late stage's stochastic ray tracer on a GPU, through WebGPU.

:func:`trace_stochastic` takes the arguments of the compiled core's
``acoustic_spaces._native.trace_stochastic`` and returns what it does, computed by one
WGSL compute shader on any GPU wgpu reaches (see :mod:`acoustic_spaces.engine.gpu`). It
traces what the CPU does: detector spheres and diffuse or specular rain, vector-based
or random scattering, the scattering of a surface's size, air absorption, the early
stage's paths left out, arrival directions and the weight window that splits rays and
thins them out; outdoors, the rays bent by the atmosphere, followed in chords as the
CPU follows them (``cpp/include/acoustic_spaces/atmosphere.hpp``), and the plants that
absorb and scatter them (``vegetation.hpp``), with the stems' and the foliage's rain. It
leaves to the CPU (raising :class:`Unsupported`) sound through surfaces into other
rooms, the face groups of material calibration, runs whose results its buffers cannot
hold, and vegetation of more than :data:`MAX_MEDIA` media or so many volumes along one
ray that its fixed tables run out of room.

Rays start where the CPU's do (``_native.ray_starts``: the same directions and energies
for the same seed), then go their own way: the GPU draws its random numbers from
streams of its own (xoshiro128**) and works in single precision, so the two agree
statistically, not ray for ray. Each ray adds its energy to the listeners' time slots
with atomic additions of 64-bit fixed-point numbers (two 32-bit words with a carry,
rounded at random to the nearest step, so that sums are exact and unbiased; WGSL has no
atomic floating-point addition), scaled so that a window's sums stay under 2**40 times
the reflections per ray, where the listeners' sums grow with the ray count (and the GPU
gives up a run whose sums overflow).

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
Which of a family's copies split before it is full depends on the order the threads
come in, so such runs differ from one another within the noise of the ray count. Bent
rays take many chords to a leg, so their rounds hold fewer rays.
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
_ATMOSPHERE, _PLANTS = 32, 64
#: Buckets of the wind's cosine the atmosphere's tables are kept for (as on the CPU).
_WIND_BUCKETS = 401
#: Words per vegetation volume in the scene, and per ray in the chord test.
_SHAPE = 17
_CHORD_IN, _CHORD_OUT = 10, 9
#: Most media of vegetation (the shader keeps a scattering rate per medium), and most
#: volumes along one piece of a ray (more are dropped, and the run declined).
MAX_MEDIA = 16
MAX_SPANS = 32

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
    ("box_at", "u"),
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
    ("ground", "f"),
    ("sagitta", "f"),
    ("atm_nodes", "u"),
    ("atm_buckets", "u"),
    ("atm_at", "u"),
    ("atm_reach_at", "u"),
    ("atm_slow_at", "u"),
    ("shapes", "u"),
    ("shape_at", "u"),
    ("outline_at", "u"),
    ("media", "u"),
    ("medium_at", "u"),
    ("wind_x", "f"),
    ("wind_y", "f"),
)
_TYPES = {"u": "u32", "i": "i32", "f": "f32"}

_SHADER = (
    (
        """
const BANDS: u32 = __BANDS__u;
// Words per ray in flight and per ray from the source; words per face, per listener,
// per vegetation volume and per medium.
const RECORD: u32 = 20u + 3u * BANDS;
const START: u32 = 3u + BANDS;
const FACE: u32 = 5u + 2u * BANDS;
const LISTENER: u32 = 6u;
const SHAPE: u32 = __SHAPE__u;
const MEDIUM: u32 = 3u * BANDS + 1u;
const PI: f32 = 3.14159265358979;
const COUNTERS: u32 = __COUNTERS__u;
const DIFFUSE_RAIN: u32 = 1u;
const SPECULAR_RAIN: u32 = 2u;
const VECTOR: u32 = 4u;
const SIZES: u32 = 8u;
const SPLITTING: u32 = 16u;
// Zero in a shader built without them (__FEATURES__ is the flags it has), which leaves
// out their code: Direct3D 12's default compiler (FXC) takes minutes over all of it.
const ATMOSPHERE: u32 = 32u & __FEATURES__u;
const PLANTS: u32 = 64u & __FEATURES__u;
// Most pieces of arc in one chord, cuts of a segment through one prism's outline,
// media and volumes along one piece of a ray.
const MAX_PIECES: u32 = 1024u;
const MAX_CUTS: u32 = 64u;
const MAX_MEDIA: u32 = __MAX_MEDIA__u;
const MAX_SPANS: u32 = __MAX_SPANS__u;
const MAX_EDGES: u32 = 2u * MAX_SPANS + 2u;
// Turns over which the stems' rain averages the leg they send through a sphere.
const STEM_TURNS: u32 = 12u;

struct Run {
"""
        + "".join(f"    {name}: {_TYPES[kind]},\n" for name, kind in _FIELDS)
        + """}

@group(0) @binding(0) var<uniform> run: Run;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> triangles: array<vec4<u32>>;
// The source, then per face its normal, size, absorption and scattering; per listener
// its place, sphere radius and volume and how long the early stage covers; the arrival
// directions; air absorption and band centres; the weight window's expected energies;
// the model's box; the atmosphere's nodes (heights, still speeds, wind speeds) and its
// tables of reach and slowness per bucket; the vegetation's volumes, their outlines'
// corners and their media.
@group(0) @binding(3) var<storage, read> scene: array<u32>;
// Per ray from the source: its direction and energy per band.
@group(0) @binding(4) var<storage, read> starts: array<u32>;
// Rays in flight: two halves, read from one and written to the other in turn.
@group(0) @binding(5) var<storage, read_write> states: array<u32>;
// Counters (rays kept and copies made this round, escaped, traced, overflow, plant
// volumes dropped), then per window the rays' energy and reflections, the energy per
// listener, band and slot, per listener, direction, band and slot, and the copies made
// of each ray from the source.
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
    // In an atmosphere, while `leaving`: the normal of the face the ray left (pointing
    // the way it went), which it cannot meet again until it turns back.
    away: vec3<f32>,
    leaving: u32,
}

var<private> ray: Ray;
// In an atmosphere: metres along the ray so far (travelled is then the speed of sound
// times the time). (Not in Ray: the virtual GPU of Intel Macs in CI lost what it held
// there by the time the ray met a face, and the rain from it overflowed.)
var<private> ray_path: f32;
// The ray's energy per band, the share of it the detector spheres count on this leg and
// the share that has only ever reflected specularly. (Not in Ray: Direct3D's shader
// compiler cannot write to an array in a struct at a varying index, and gives up.)
var<private> energy: array<f32, BANDS>;
var<private> seen: array<f32, BANDS>;
var<private> alone: array<f32, BANDS>;
var<private> counted: array<f32, BANDS>;
var<private> reflected: array<f32, BANDS>;
var<private> scattered: array<f32, BANDS>;
var<private> scatter: array<f32, BANDS>;
// Through plants: each band's weight at a point of a piece of the ray, and what the
// plants take along a straight line (nepers of energy).
var<private> weights: array<f32, BANDS>;
var<private> tau: array<f32, BANDS>;
// What a leg ended with: the face it met and the metres it went to it, or where along
// its last piece it scatters in the plants.
var<private> leg_hit: Hit;
var<private> leg_metres: f32;
var<private> leg_scatter_at: f32;

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

// Whether p lies in the model's box, grown by the offset on every side.
fn encloses(p: vec3<f32>) -> bool {
    let lo = point(run.box_at) - vec3<f32>(run.offset);
    let hi = point(run.box_at + 3u) + vec3<f32>(run.offset);
    return all(p >= lo) && all(p <= hi);
}

// -- the atmosphere (see atmosphere.hpp) ---------------------------------------------

fn atm_height(i: u32) -> f32 {
    return num(run.atm_at + i);
}

// The effective speed at node i along a direction at `along` to the wind.
fn atm_speed(i: u32, along: f32) -> f32 {
    return num(run.atm_at + run.atm_nodes + i) + num(run.atm_at + 2u * run.atm_nodes + i) * along;
}

// The layer z (m above the ground, between the first and the last node) lies in: the
// last node at or below it.
fn atm_layer(z: f32) -> u32 {
    var i = 0u;
    for (var k = 1u; k < run.atm_nodes; k++) {
        if (atm_height(k) <= z) {
            i = k;
        } else {
            break;
        }
    }
    return i;
}

// The effective speed at z (m above the ground).
fn atm_speed_at(z: f32, along: f32) -> f32 {
    let n = run.atm_nodes;
    if (!(z > 0.0)) {
        return atm_speed(0u, along);
    }
    if (z >= atm_height(n - 1u)) {
        return atm_speed(n - 1u, along);
    }
    let i = atm_layer(z);
    let t = (z - atm_height(i)) / (atm_height(i + 1u) - atm_height(i));
    return atm_speed(i, along) + t * (atm_speed(i + 1u, along) - atm_speed(i, along));
}

// The cosine of the wind to the horizontal direction (x, y) (unit, or zero).
fn atm_along(x: f32, y: f32) -> f32 {
    return x * run.wind_x + y * run.wind_y;
}

// The bucket of the wind's cosine `along`, and the cosine of bucket m.
fn atm_bucket(along: f32) -> u32 {
    if (run.atm_buckets < 2u) {
        return 0u;
    }
    let t = 0.5 * (clamp(along, -1.0, 1.0) + 1.0);
    return u32(round(t * f32(run.atm_buckets - 1u)));
}

fn atm_cosine(m: u32) -> f32 {
    if (run.atm_buckets < 2u) {
        return 0.0;
    }
    return -1.0 + 2.0 * f32(m) / f32(run.atm_buckets - 1u);
}

// log(1 + x) / x, 1 at 0.
fn log_ratio(x: f32) -> f32 {
    if (abs(x) < 1e-4) {
        return 1.0 - 0.5 * x;
    }
    return log(1.0 + x) / x;
}

// The integral of dz / c from the ground up to z (negative below it), in bucket m.
fn atm_slowness_at(m: u32, z: f32) -> f32 {
    let n = run.atm_nodes;
    let along = atm_cosine(m);
    if (!(z > 0.0)) {
        return z / atm_speed(0u, along);
    }
    let time_at = run.atm_slow_at + m * n;
    if (z >= atm_height(n - 1u)) {
        return num(time_at + n - 1u) + (z - atm_height(n - 1u)) / atm_speed(n - 1u, along);
    }
    let i = atm_layer(z);
    let ca = atm_speed(i, along);
    let cz = ca + (atm_speed(i + 1u, along) - ca) * (z - atm_height(i)) /
                  (atm_height(i + 1u) - atm_height(i));
    return num(time_at + i) + (z - atm_height(i)) * log_ratio((cz - ca) / ca) / ca;
}

// The horizontal distance from where the ray that grazes the ground touches it to
// height z, in bucket m; FAR where that ray turns back down first.
fn atm_reach_at(m: u32, z: f32) -> f32 {
    let n = run.atm_nodes;
    let row = run.atm_reach_at + m * n;
    if (!(z > 0.0)) {
        return 0.0;
    }
    if (z >= atm_height(n - 1u)) {
        return num(row + n - 1u);
    }
    let i = atm_layer(z);
    let above = num(row + i + 1u);
    if (above >= FAR) {
        return above;
    }
    let t = (z - atm_height(i)) / (atm_height(i + 1u) - atm_height(i));
    return num(row + i) + t * (above - num(row + i));
}

// The time (s) sound takes along the straight line from a to b.
fn atm_straight_time(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let v = b - a;
    let len = length(v);
    if (!(len > 0.0)) {
        return 0.0;
    }
    let flat = length(v.xy);
    var along = 0.0;
    if (flat > 0.0) {
        along = atm_along(v.x / flat, v.y / flat);
    }
    let m = atm_bucket(along);
    let za = a.z - run.ground;
    let zb = b.z - run.ground;
    // Nearly level: the speed at the middle (the speed is linear in height).
    if (abs(zb - za) <= 1e-2 * len) {
        return len / atm_speed_at(0.5 * (za + zb), atm_cosine(m));
    }
    return len * (atm_slowness_at(m, zb) - atm_slowness_at(m, za)) / (zb - za);
}

// Whether b lies in the shadow of upward refraction as seen from a: past where the ray
// that grazes the ground could take sound from one to the other.
fn atm_shadowed(a: vec3<f32>, b: vec3<f32>) -> bool {
    let d = b.xy - a.xy;
    let flat = length(d);
    if (!(flat > 0.0)) {
        return false;
    }
    let m = atm_bucket(atm_along(d.x / flat, d.y / flat));
    return flat > atm_reach_at(m, a.z - run.ground) + atm_reach_at(m, b.z - run.ground);
}

// A ray bent by the atmosphere: where it is, which way it goes, and its arc through the
// layers (BentRay in atmosphere.hpp). Its elevation is kept as a cosine and a sine, and
// turned by series in the small angles a chord turns through (at most a few
// hundredths of a radian), so that nothing here leans on the GPU's sine, cosine or
// arc tangent, which may be off by many steps of single precision.
struct Bent {
    pos: vec3<f32>,
    ex: f32,
    ey: f32,
    along: f32,
    ct: f32,
    st: f32,
    p: f32,
    z: f32,
    // The layer the ray is in: i between nodes i and i + 1, -1 below the ground, nodes - 1
    // above the last node; its bottom and top, and the speed there as c + g (z - base).
    layer: i32,
    lo: f32,
    hi: f32,
    base: f32,
    c: f32,
    g: f32,
}

struct Piece {
    length: f32,
    time: f32,
    turn: f32,
    boundary: bool,
    turned: bool,
}

struct Chord {
    start: vec3<f32>,
    end: vec3<f32>,
    arc: f32,
    time: f32,
    // The chord stops where the ray turns back towards the plane it left.
    turned: bool,
}

var<private> bent: Bent;

// cos x, sin x and sin x / x by series (|x| up to 0.5: within 1e-8).
fn cos_small(x: f32) -> f32 {
    if (abs(x) > 0.5) {
        return cos(x);
    }
    let x2 = x * x;
    return 1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 * (1.0 / 720.0 - x2 * (1.0 / 40320.0))));
}

fn sin_small(x: f32) -> f32 {
    if (abs(x) > 0.5) {
        return sin(x);
    }
    let x2 = x * x;
    let tail = 1.0 / 120.0 - x2 * (1.0 / 5040.0 - x2 * (1.0 / 362880.0));
    return x * (1.0 - x2 * (1.0 / 6.0 - x2 * tail));
}

fn sinc_small(x: f32) -> f32 {
    if (abs(x) > 0.5) {
        return sin(x) / x;
    }
    let x2 = x * x;
    let tail = 1.0 / 120.0 - x2 * (1.0 / 5040.0 - x2 * (1.0 / 362880.0));
    return 1.0 - x2 * (1.0 / 6.0 - x2 * tail);
}

// atan x for |x| up to 0.7, by halving the angle twice and a series (within 1e-9).
fn atan_small(x: f32) -> f32 {
    var y = x / (1.0 + sqrt(1.0 + x * x));
    y = y / (1.0 + sqrt(1.0 + y * y));
    let y2 = y * y;
    return 4.0 * y * (1.0 - y2 * (1.0 / 3.0 - y2 * (0.2 - y2 * (1.0 / 7.0 - y2 * (1.0 / 9.0)))));
}

// The angle from elevation (ca, sa) to elevation (cb, sb), a - b: accurate where it is
// small, which is wherever it decides a piece's length.
fn turn_between(ca: f32, sa: f32, cb: f32, sb: f32) -> f32 {
    let s = sa * cb - ca * sb;
    let c = ca * cb + sa * sb;
    if (c > 0.0 && abs(s) < 0.7 * c) {
        return atan_small(s / c);
    }
    return atan2(s, c);
}

// The cosine and sine of the elevation (ct, st) turned down by phi.
fn turned_down(ct: f32, st: f32, phi: f32) -> vec2<f32> {
    let c = cos_small(phi);
    let s = sin_small(phi);
    return vec2<f32>(ct * c + st * s, st * c - ct * s);
}

fn bent_enter(i: i32) {
    let n = i32(run.atm_nodes);
    bent.layer = i;
    bent.g = 0.0;
    if (i < 0) {
        bent.lo = -FAR;
        bent.hi = 0.0;
        bent.base = 0.0;
        bent.c = atm_speed(0u, bent.along);
    } else if (i + 1 >= n) {
        bent.lo = atm_height(u32(n - 1));
        bent.base = bent.lo;
        bent.hi = FAR;
        bent.c = atm_speed(u32(n - 1), bent.along);
    } else {
        let k = u32(i);
        bent.lo = atm_height(k);
        bent.base = bent.lo;
        bent.hi = atm_height(k + 1u);
        bent.c = atm_speed(k, bent.along);
        bent.g = (atm_speed(k + 1u, bent.along) - bent.c) / (bent.hi - bent.lo);
    }
}

fn bent_speed_in(z: f32) -> f32 {
    return bent.c + bent.g * (z - bent.base);
}

fn bent_init(pos: vec3<f32>, dir: vec3<f32>) {
    bent.pos = pos;
    let flat = length(dir.xy);
    bent.ex = 0.0;
    bent.ey = 0.0;
    if (flat > 0.0) {
        bent.ex = dir.x / flat;
        bent.ey = dir.y / flat;
    }
    bent.along = atm_along(bent.ex, bent.ey);
    bent.ct = flat;
    bent.st = dir.z;
    bent.z = pos.z - run.ground;
    bent.p = flat / atm_speed_at(bent.z, bent.along);
    let n = i32(run.atm_nodes);
    var layer = 0;
    if (bent.z < 0.0) {
        layer = -1;
    } else if (bent.z >= atm_height(u32(n - 1))) {
        layer = n - 1;
    } else {
        layer = i32(atm_layer(bent.z));
    }
    if (layer >= 0 && bent.z == atm_height(u32(layer)) && bent.st < 0.0) {
        layer -= 1;
    }
    bent_enter(layer);
}

fn bent_direction() -> vec3<f32> {
    return vec3<f32>(bent.ct * bent.ex, bent.ct * bent.ey, bent.st);
}

// 1 - q for q = cos(elevation) c_at / c(z), the ray parameter times the speed c_at
// (q < 1 where the ray can reach that speed): accurate near grazing, where 1 - q is
// tiny, from the sine (1 - cos = sin^2 / (1 + cos)).
fn one_minus_q(c_at: f32) -> f32 {
    let c_here = bent_speed_in(bent.z);
    let one_minus_ct = bent.st * bent.st / (1.0 + bent.ct);
    return (one_minus_ct * c_here - bent.ct * (c_at - c_here)) / c_here;
}

// One piece of arc in the current layer, at most `limit` metres, within the chord's
// tolerance after `arc` metres and `turn` radians of it. With `use_away`, it stops where
// the ray turns back towards the plane of ray.away.
fn bent_step(limit: f32, arc: f32, turn: f32, use_away: bool) -> Piece {
    var kappa = 0.0;
    var len = FAR;
    var ct_end = bent.ct;
    var st_end = bent.st;
    var next = bent.layer;
    // Where the ray leaves the layer. A ray that only just touches a node's height goes
    // on in the layer it heads into; one that neither layer takes (level on the axis of
    // a duct) goes on level.
    for (var attempt = 0; attempt < 3; attempt++) {
        kappa = bent.g * bent.p;
        if (abs(kappa) * (bent.hi - bent.lo) < 1e-12) {
            kappa = 0.0;
        }
        next = bent.layer;
        ct_end = bent.ct;
        st_end = bent.st;
        len = FAR;
        if (kappa == 0.0) {
            if (bent.st > 0.0 && bent.hi < FAR) {
                len = (bent.hi - bent.z) / bent.st;
                next = bent.layer + 1;
            } else if (bent.st < 0.0 && bent.lo > -FAR) {
                len = (bent.lo - bent.z) / bent.st;
                next = bent.layer - 1;
            }
        } else {
            let up = bent.st > 0.0 || (bent.st == 0.0 && kappa < 0.0);
            let gap_hi = one_minus_q(bent_speed_in(bent.hi));
            let gap_lo = one_minus_q(bent_speed_in(bent.lo));
            if (up && gap_hi > 0.0) {
                // On up into the next layer.
                ct_end = 1.0 - gap_hi;
                st_end = sqrt(gap_hi * (2.0 - gap_hi));
                next = bent.layer + 1;
            } else if (up) {
                // Turns back down.
                let gap = max(gap_lo, 0.0);
                ct_end = 1.0 - gap;
                st_end = -sqrt(gap * (2.0 - gap));
                next = bent.layer - 1;
            } else if (gap_lo > 0.0) {
                // On down into the layer below.
                ct_end = 1.0 - gap_lo;
                st_end = -sqrt(gap_lo * (2.0 - gap_lo));
                next = bent.layer - 1;
            } else {
                // Turns back up.
                let gap = max(gap_hi, 0.0);
                ct_end = 1.0 - gap;
                st_end = sqrt(gap * (2.0 - gap));
                next = bent.layer + 1;
            }
            len = turn_between(bent.ct, bent.st, ct_end, st_end) / kappa;
        }
        if (len > 0.0) {
            break;
        }
        bent_enter(next);
        if (attempt == 2) {
            kappa = 0.0;
            len = FAR;
            ct_end = bent.ct;
            st_end = bent.st;
            next = bent.layer;
        }
    }

    // The chord's tolerance: (arc + s) (turn + |kappa| s) <= 4 sagitta.
    let room = max(4.0 * run.sagitta - arc * turn, 0.0);
    let k = abs(kappa);
    var tolerance = FAR;
    if (k > 0.0) {
        let b = turn + k * arc;
        tolerance = 2.0 * room / (b + sqrt(b * b + 4.0 * k * room));
    } else if (turn > 0.0) {
        tolerance = room / turn;
    }
    var piece: Piece;
    piece.length = 0.0;
    piece.time = 0.0;
    piece.turn = 0.0;
    piece.turned = false;
    var s = min(min(len, limit), tolerance);
    piece.boundary = s == len && len < FAR;
    // Where the ray turns back towards the plane it left.
    if (use_away && kappa != 0.0 && ray.away.z != 0.0) {
        let a = ray.away.x * bent.ex + ray.away.y * bent.ey;
        let nz = ray.away.z;
        var at_end = vec2<f32>(ct_end, st_end);
        if (!piece.boundary) {
            at_end = turned_down(bent.ct, bent.st, kappa * s);
        }
        let facing_now = a * bent.ct + nz * bent.st;
        let facing_end = a * at_end.x + nz * at_end.y;
        if (facing_now > 0.0 && facing_end <= 0.0) {
            // The elevation at which the ray runs along the plane: tan = -a / nz, and
            // its cosine positive.
            let h = sqrt(a * a + nz * nz);
            let ct_t = abs(nz) / h;
            let st_t = -sign(nz) * a / h;
            let at = turn_between(bent.ct, bent.st, ct_t, st_t) / kappa;
            if (at <= s) {
                s = max(at, 0.0);
                piece.boundary = false;
                piece.turned = true;
            }
        }
    }
    // (One return, at the end: after an early one, Direct3D 12's software GPU, WARP,
    // gave every chord of a bent ray no travel time.)
    if (s > 0.0) {
        var phi = kappa * s;
        var at_new = turned_down(bent.ct, bent.st, phi);
        if (piece.boundary) {
            phi = turn_between(bent.ct, bent.st, ct_end, st_end);
            at_new = vec2<f32>(ct_end, st_end);
        }
        let chord = s * sinc_small(0.5 * phi);
        let mean = turned_down(bent.ct, bent.st, 0.5 * phi);
        let flat = chord * mean.x;
        let rise = chord * mean.y;
        var z_end = bent.z + rise;
        if (piece.boundary) {
            z_end = select(bent.lo, bent.hi, next > bent.layer);
        }
        // The travel time by Simpson's rule (the speed is linear in height in a layer).
        var mid = 0.5 * (bent.z + z_end);
        if (kappa != 0.0) {
            let quarter = turned_down(bent.ct, bent.st, 0.25 * phi);
            mid = bent.z + 0.5 * s * sinc_small(0.25 * phi) * quarter.y;
        }
        piece.length = s;
        piece.time = s / 6.0 * (1.0 / bent_speed_in(bent.z) + 4.0 / bent_speed_in(mid) +
                                1.0 / bent_speed_in(z_end));
        piece.turn = abs(phi);
        bent.pos += vec3<f32>(flat * bent.ex, flat * bent.ey, rise);
        bent.ct = at_new.x;
        bent.st = at_new.y;
        bent.z = z_end;
        if (piece.boundary) {
            bent_enter(next);
        }
    }
    return piece;
}

// Follows the arc for at most `budget` metres, as far as one chord may go.
fn bent_chord(budget: f32, use_away: bool) -> Chord {
    var out: Chord;
    out.start = bent.pos;
    out.arc = 0.0;
    out.time = 0.0;
    out.turned = false;
    // The elevation's cosine and sine, kept unit through the turns.
    let h = sqrt(bent.ct * bent.ct + bent.st * bent.st);
    if (h > 0.0) {
        bent.ct /= h;
        bent.st /= h;
    }
    var turn = 0.0;
    for (var n = 0u; n < MAX_PIECES && out.arc < budget; n++) {
        let piece = bent_step(budget - out.arc, out.arc, turn, use_away);
        out.arc += piece.length;
        out.time += piece.time;
        turn += piece.turn;
        if (piece.turned) {
            out.turned = true;
            break;
        }
        if (!(piece.length > 0.0) || !piece.boundary) {
            break;
        }
    }
    out.end = bent.pos;
    return out;
}

// -- vegetation (see vegetation.hpp) ---------------------------------------------------

fn shape_word(s: u32, w: u32) -> u32 {
    return scene[run.shape_at + SHAPE * s + w];
}

fn shape_num(s: u32, w: u32) -> f32 {
    return num(run.shape_at + SHAPE * s + w);
}

fn corner(k: u32) -> vec2<f32> {
    return vec2<f32>(num(run.outline_at + 2u * k), num(run.outline_at + 2u * k + 1u));
}

fn medium_at(m: u32) -> u32 {
    return run.medium_at + MEDIUM * m;
}

fn upright(m: u32) -> bool {
    return scene[medium_at(m) + 3u * BANDS] != 0u;
}

// How much of a ray going along unit d a medium meets per metre: its horizontal share
// for stems, all of it otherwise.
fn share(m: u32, d: vec3<f32>) -> f32 {
    return select(1.0, length(d.xy), upright(m));
}

// Whether (x, y) lies inside the polygon of prism s (crossing number).
fn inside(s: u32, x: f32, y: f32) -> bool {
    let first = shape_word(s, 4u);
    let count = shape_word(s, 5u);
    var within = false;
    var j = count - 1u;
    for (var i = 0u; i < count; i++) {
        let a = corner(first + i);
        let b = corner(first + j);
        if ((a.y > y) != (b.y > y) && x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x) {
            within = !within;
        }
        j = i;
    }
    return within;
}

// The parts of the segment from p along unit d, t in [0, len], inside volume s: pieces
// (piece_lo, piece_hi), parts of one prism that meet joined.
var<private> cut: array<f32, MAX_CUTS>;
var<private> piece_lo: array<f32, MAX_CUTS>;
var<private> piece_hi: array<f32, MAX_CUTS>;
var<private> n_pieces: u32;

fn shape_pieces(s: u32, p: vec3<f32>, d: vec3<f32>, len: f32) {
    n_pieces = 0u;
    let lo = vec3<f32>(shape_num(s, 11u), shape_num(s, 12u), shape_num(s, 13u));
    let hi = vec3<f32>(shape_num(s, 14u), shape_num(s, 15u), shape_num(s, 16u));
    let q = p + d * len;
    if (any(max(p, q) < lo) || any(min(p, q) > hi)) {
        return;
    }
    if (shape_word(s, 0u) == 1u) {
        // The spheroid.
        let centre = vec3<f32>(shape_num(s, 6u), shape_num(s, 7u), shape_num(s, 8u));
        let axes = vec3<f32>(shape_num(s, 9u), shape_num(s, 9u), shape_num(s, 10u));
        let a = (p - centre) / axes;
        let e = d / axes;
        let aa = dot(e, e);
        let b = dot(a, e);
        let c = dot(a, a) - 1.0;
        let disc = b * b - aa * c;
        if (!(disc > 0.0)) {
            return;
        }
        let root = sqrt(disc);
        let t0 = max((-b - root) / aa, 0.0);
        let t1 = min((-b + root) / aa, len);
        if (t1 > t0) {
            piece_lo[0] = t0;
            piece_hi[0] = t1;
            n_pieces = 1u;
        }
        return;
    }
    // The prism: the slab between its heights, then the polygon's crossings.
    var ta = 0.0;
    var tb = len;
    let z0 = shape_num(s, 2u);
    let z1 = shape_num(s, 3u);
    if (abs(d.z) < 1e-12) {
        if (!(p.z > z0 && p.z < z1)) {
            return;
        }
    } else {
        var a = (z0 - p.z) / d.z;
        var b = (z1 - p.z) / d.z;
        if (a > b) {
            let swap = a;
            a = b;
            b = swap;
        }
        ta = max(ta, a);
        tb = min(tb, b);
    }
    if (!(tb > ta)) {
        return;
    }
    var n = 1u;
    var lost = false;
    cut[0] = ta;
    let first = shape_word(s, 4u);
    let count = shape_word(s, 5u);
    for (var i = 0u; i < count; i++) {
        let a = corner(first + i);
        let e = corner(first + (i + 1u) % count) - a;
        let denom = d.x * e.y - d.y * e.x;
        if (abs(denom) < 1e-12) {
            continue;
        }
        let w = a - p.xy;
        let t = (w.x * e.y - w.y * e.x) / denom;
        let u = (w.x * d.y - w.y * d.x) / denom;
        if (u >= 0.0 && u < 1.0 && t > ta && t < tb) {
            if (n + 1u < MAX_CUTS) {
                cut[n] = t;
                n++;
            } else {
                lost = true;
            }
        }
    }
    // Counted out of the loop: FXC fails on an atomic in that branch ("flattened side
    // effect").
    if (lost) {
        atomicStore(&tallies[5], 1u);
    }
    cut[n] = tb;
    n++;
    // The crossings in order (insertion sort; the ends stay put).
    for (var i = 2u; i + 1u < n; i++) {
        let v = cut[i];
        var j = i;
        while (j > 1u && cut[j - 1u] > v) {
            cut[j] = cut[j - 1u];
            j--;
        }
        cut[j] = v;
    }
    for (var i = 0u; i + 1u < n; i++) {
        let t0 = cut[i];
        let t1 = cut[i + 1u];
        if (!(t1 > t0)) {
            continue;
        }
        let middle = 0.5 * (t0 + t1);
        if (!inside(s, p.x + d.x * middle, p.y + d.y * middle)) {
            continue;
        }
        // Join it to the part before it when they meet.
        if (n_pieces > 0u && piece_hi[n_pieces - 1u] == t0) {
            piece_hi[n_pieces - 1u] = t1;
        } else {
            piece_lo[n_pieces] = t0;
            piece_hi[n_pieces] = t1;
            n_pieces++;
        }
    }
}

// Per band, the absorption and scattering together met along the segment from p along
// unit d for `len` metres (nepers of energy), into tau.
fn depth_tau(p: vec3<f32>, d: vec3<f32>, len: f32) {
    for (var b = 0u; b < BANDS; b++) {
        tau[b] = 0.0;
    }
    for (var s = 0u; s < run.shapes; s++) {
        shape_pieces(s, p, d, len);
        if (n_pieces == 0u) {
            continue;
        }
        let m = shape_word(s, 1u);
        let at = medium_at(m);
        let sh = share(m, d);
        for (var i = 0u; i < n_pieces; i++) {
            let metres = (piece_hi[i] - piece_lo[i]) * sh;
            for (var b = 0u; b < BANDS; b++) {
                tau[b] += (num(at + b) + num(at + BANDS + b)) * metres;
            }
        }
    }
}

// A straight piece of a ray through plants (Track in ray_tracer.hpp): the parts of it
// inside each volume, each medium's scattering rate averaged over the bands weighted by
// the ray's energy, and each part's chance of scattering per metre (its medium's rate
// times its share). The bands are then weighted by what they meet against that rate
// (w_b(s) = exp(S(s) - tau_b(s)), S the drawn rate and tau_b the band's absorption and
// scattering, both integrated to s), so each keeps its expected energy.
var<private> span_t0: array<f32, MAX_SPANS>;
var<private> span_t1: array<f32, MAX_SPANS>;
var<private> span_medium: array<u32, MAX_SPANS>;
var<private> span_chance: array<f32, MAX_SPANS>;
var<private> n_spans: u32;
var<private> track_d: vec3<f32>;
var<private> track_len: f32;
var<private> rate: array<f32, MAX_MEDIA>;
var<private> edge: array<f32, MAX_EDGES>;
// The piece between edges the ray last scattered in.
var<private> scatter_lo: f32;
var<private> scatter_hi: f32;

// The piece from p along unit d for `len` metres; false where it meets no plants.
fn track_build(p: vec3<f32>, d: vec3<f32>, len: f32) -> bool {
    n_spans = 0u;
    track_d = d;
    track_len = len;
    var lost = false;
    for (var s = 0u; s < run.shapes; s++) {
        shape_pieces(s, p, d, len);
        let m = shape_word(s, 1u);
        for (var i = 0u; i < n_pieces; i++) {
            if (n_spans < MAX_SPANS) {
                span_t0[n_spans] = piece_lo[i];
                span_t1[n_spans] = piece_hi[i];
                span_medium[n_spans] = m;
                n_spans++;
            } else {
                lost = true;
            }
        }
    }
    if (lost) {
        atomicStore(&tallies[5], 1u);  // out of the loop, as in shape_pieces
    }
    if (n_spans == 0u) {
        return false;
    }
    var total = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        total += energy[b];
    }
    for (var m = 0u; m < run.media; m++) {
        rate[m] = 0.0;
        if (total > 0.0) {
            let at = medium_at(m);
            for (var b = 0u; b < BANDS; b++) {
                rate[m] += energy[b] * num(at + BANDS + b);
            }
            rate[m] /= total;
        }
    }
    for (var i = 0u; i < n_spans; i++) {
        span_chance[i] = rate[span_medium[i]] * share(span_medium[i], d);
    }
    return true;
}

// The drawn rate integrated to s metres along the piece.
fn track_drawn(s: f32) -> f32 {
    var drawn = 0.0;
    for (var i = 0u; i < n_spans; i++) {
        drawn += span_chance[i] * max(min(s, span_t1[i]) - span_t0[i], 0.0);
    }
    return drawn;
}

// The weight w_b of each band at s metres along the piece, into weights.
fn track_weight(s: f32) {
    let drawn = track_drawn(s);
    for (var b = 0u; b < BANDS; b++) {
        tau[b] = 0.0;
    }
    for (var i = 0u; i < n_spans; i++) {
        let overlap = max(min(s, span_t1[i]) - span_t0[i], 0.0);
        if (!(overlap > 0.0)) {
            continue;
        }
        let metres = overlap * share(span_medium[i], track_d);
        let at = medium_at(span_medium[i]);
        for (var b = 0u; b < BANDS; b++) {
            tau[b] += (num(at + b) + num(at + BANDS + b)) * metres;
        }
    }
    for (var b = 0u; b < BANDS; b++) {
        weights[b] = exp(drawn - tau[b]);
    }
}

// Where along the piece the ray scatters, or FAR where it goes through. Draws a random
// number only where it may scatter.
fn track_collide() -> f32 {
    let total = track_drawn(track_len);
    if (!(total > 0.0)) {
        return FAR;
    }
    let aim = -log(1.0 - random());
    if (aim >= total) {
        return FAR;
    }
    // The edges where the rate changes, in order.
    var n = 2u;
    edge[0] = 0.0;
    edge[1] = track_len;
    for (var i = 0u; i < n_spans; i++) {
        edge[n] = span_t0[i];
        edge[n + 1u] = span_t1[i];
        n += 2u;
    }
    for (var i = 1u; i < n; i++) {
        let v = edge[i];
        var j = i;
        while (j > 0u && edge[j - 1u] > v) {
            edge[j] = edge[j - 1u];
            j--;
        }
        edge[j] = v;
    }
    var drawn = 0.0;
    for (var e = 0u; e + 1u < n; e++) {
        let t0 = edge[e];
        let t1 = edge[e + 1u];
        if (!(t1 > t0)) {
            continue;
        }
        let middle = 0.5 * (t0 + t1);
        var slope = 0.0;
        for (var i = 0u; i < n_spans; i++) {
            if (span_t0[i] <= middle && middle <= span_t1[i]) {
                slope += span_chance[i];
            }
        }
        let next = drawn + slope * (t1 - t0);
        if (next > aim) {
            scatter_lo = t0;
            scatter_hi = t1;
            let f = (aim - drawn) / (next - drawn);
            return min(t0 + f * (t1 - t0), t1);
        }
        drawn = next;
    }
    return FAR;
}

// The medium the ray scatters in where track_collide had it scatter: one of those
// there, at random in proportion to their chances.
fn track_medium() -> u32 {
    let middle = 0.5 * (scatter_lo + scatter_hi);
    var total = 0.0;
    var only = 0u;
    var count = 0u;
    for (var i = 0u; i < n_spans; i++) {
        if (span_t0[i] <= middle && middle <= span_t1[i] && span_chance[i] > 0.0) {
            total += span_chance[i];
            only = i;
            count++;
        }
    }
    if (count <= 1u) {
        return span_medium[only];
    }
    var pick = random() * total;
    for (var i = 0u; i < n_spans; i++) {
        if (!(span_t0[i] <= middle && middle <= span_t1[i] && span_chance[i] > 0.0)) {
            continue;
        }
        only = i;
        pick -= span_chance[i];
        if (pick < 0.0) {
            break;
        }
    }
    return span_medium[only];
}

// Henyey and Greenstein's phase function: the share per steradian of what foliage
// scatters that goes off at cos(angle) mu to the way the sound went, mean cosine g.
fn henyey_greenstein(mu: f32, g: f32) -> f32 {
    let q = 1.0 + g * g - 2.0 * g * mu;
    return (1.0 - g * g) / (4.0 * PI * q * sqrt(q));
}

// The cosine of an angle drawn from henyey_greenstein with mean cosine g.
fn henyey_greenstein_cosine(g: f32) -> f32 {
    let u = random();
    if (abs(g) < 1e-6) {
        return 1.0 - 2.0 * u;
    }
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u);
    return clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
}

// The wrapped Cauchy distribution's density at angle phi, mean cosine rho: how stems
// share out the angles through which they turn sound sideways.
fn wrapped_cauchy(phi: f32, rho: f32) -> f32 {
    return (1.0 - rho * rho) / (2.0 * PI * (1.0 + rho * rho - 2.0 * rho * cos(phi)));
}

// An angle drawn from wrapped_cauchy with mean cosine rho.
fn wrapped_cauchy_angle(rho: f32) -> f32 {
    let u = random();
    return 2.0 * atan((1.0 - rho) / (1.0 + rho) * tan(PI * (u - 0.5)));
}

// -- what reaches the listeners --------------------------------------------------------

// Adds counted[b] * factor * air(absorbing) arriving from `arrival` `travel` metres from
// the source to listener k.
fn deposit(k: u32, factor: f32, travel: f32, absorbing: f32, arrival: vec3<f32>) {
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
        let e = counted[b] * factor * exp(-num(run.air_at + b) * absorbing) * run.scale;
        add_fixed(row + 2u * b * run.bins, e);
        if (drow != NONE) {
            add_fixed(drow + 2u * b * run.dir_bins, e);
        }
    }
}

// The detector spheres along a straight piece of the ray: `segment` metres from `from`
// along `along` (unit). On a chord of a bent ray, travelled grows by `rate` per metre
// and the air absorbs from `path` on, `stretch` metres per metre (path negative: as
// travelled). Through plants (`tracked`), the piece's track weights the bands.
fn detectors(origin: vec3<f32>, along: vec3<f32>, segment: f32, rate: f32, path: f32,
             stretch: f32, tracked: bool) {
    for (var k = 0u; k < run.receivers; k++) {
        let at = run.receiver_at + LISTENER * k;
        let rel = point(at) - origin;
        let s = dot(rel, along);
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
        let nearest = clamp(s, t0, t1);
        let travel = ray.travelled + rate * nearest;
        let covered = ray.order <= run.covered_order && travel / run.c <= num(at + 5u);
        if (tracked) {
            track_weight(nearest);
        }
        var any = 0.0;
        for (var b = 0u; b < BANDS; b++) {
            counted[b] = energy[b] * max(seen[b] - select(0.0, alone[b], covered), 0.0);
            if (tracked) {
                counted[b] *= weights[b];
            }
            any += counted[b];
        }
        if (any > 0.0) {
            let absorbing = select(travel, path + stretch * nearest, path >= 0.0);
            deposit(k, (t1 - t0) / num(at + 4u), travel, absorbing, -along);
        }
    }
}

// The rain from the reflection at ray.pos (normal n towards the ray's side) to every
// listener that sees it.
fn rain(n: vec3<f32>) {
    let specular = (run.flags & SPECULAR_RAIN) != 0u;
    let bending = (run.flags & ATMOSPHERE) != 0u;
    let origin = ray.pos + n * run.offset;
    for (var k = 0u; k < run.receivers; k++) {
        let at = run.receiver_at + LISTENER * k;
        let listener = point(at);
        let v = listener - origin;
        let gap = length(v);
        let dist = max(gap, num(at + 3u));
        let cos_theta = dot(n, v) / dist;
        if (cos_theta <= 0.0 || gap <= 0.0) {
            continue;
        }
        if (bending && atm_shadowed(origin, listener)) {
            continue;
        }
        let toward = v / gap;
        if (search_ray(origin, toward, run.t_min, gap, ray.last, true).face != NONE) {
            continue;
        }
        // In an atmosphere: the time the straight line takes.
        var reach = dist;
        var absorbing = ray.travelled + dist;
        if (bending) {
            reach = run.c * atm_straight_time(origin, listener);
            absorbing = ray_path + gap;
        }
        let covered = specular && ray.order <= run.covered_order &&
                      (ray.travelled + reach) / run.c <= num(at + 5u);
        let planted = (run.flags & PLANTS) != 0u;
        if (planted) {
            depth_tau(origin, toward, gap);
        }
        for (var b = 0u; b < BANDS; b++) {
            if (specular) {
                counted[b] = reflected[b] * select(1.0, 1.0 - alone[b], covered);
            } else {
                counted[b] = scattered[b];
            }
            if (planted) {
                counted[b] *= exp(-tau[b]);
            }
        }
        deposit(k, cos_theta / (PI * dist * dist), ray.travelled + reach, absorbing, -toward);
    }
}

// What foliage at ray.pos sends each listener straight, as the phase function has it
// (mean cosines g at `at`, the medium's asymmetries), the plants and faces on the way
// taking their share.
fn foliage_rain(at: u32) {
    let bending = (run.flags & ATMOSPHERE) != 0u;
    for (var k = 0u; k < run.receivers; k++) {
        let rec = run.receiver_at + LISTENER * k;
        let listener = point(rec);
        let v = listener - ray.pos;
        let gap = length(v);
        if (!(gap > 0.0)) {
            continue;
        }
        if (bending && atm_shadowed(ray.pos, listener)) {
            continue;
        }
        let toward = v / gap;
        if (search_ray(ray.pos, toward, run.t_min, gap, NONE, true).face != NONE) {
            continue;
        }
        depth_tau(ray.pos, toward, gap);
        let cosine = dot(ray.dir, toward);
        for (var b = 0u; b < BANDS; b++) {
            counted[b] = energy[b] * henyey_greenstein(cosine, num(at + 2u * BANDS + b)) *
                         exp(-tau[b]);
        }
        let dist = max(gap, num(rec + 3u));
        var reach = dist;
        var absorbing = ray.travelled + dist;
        if (bending) {
            reach = run.c * atm_straight_time(ray.pos, listener);
            absorbing = ray_path + gap;
        }
        deposit(k, 1.0 / (dist * dist), ray.travelled + reach, absorbing, -toward);
    }
}

// What stems at ray.pos send each listener as rain: the detector sphere's count of the
// leg the ray goes on along after they turn it, on average over the turns (mean cosine
// per band at `at`) instead of for the one turn drawn. The leg keeps the slope of
// ray.dir, so it crosses a sphere for the turns in one range, by quadrature over the
// sine of the angle from its middle (the chord grows from nothing at the ends). Each
// turn's leg ends at the first face in the sphere, and the plants on the way take
// their share.
fn stem_rain(at: u32) {
    let bending = (run.flags & ATMOSPHERE) != 0u;
    let p = ray.pos;
    let d = ray.dir;
    let flat = length(d.xy);
    if (!(flat > 0.0)) {
        return;
    }
    let heading = atan2(d.y, d.x);
    for (var k = 0u; k < run.receivers; k++) {
        let rec = run.receiver_at + LISTENER * k;
        let r = num(rec + 3u);
        let v = point(rec) - p;
        let vv = dot(v, v);
        let across = length(v.xy);
        // A leg heading a meets the sphere ahead where v . leg > sqrt(|v|^2 - r^2),
        // v . leg = flat * across * cos(a - bearing) + d.z * v.z: every heading from
        // inside, none or a range round the bearing from outside.
        var middle = 0.0;
        var half = PI;
        if (vv > r * r) {
            let need = sqrt(vv - r * r) - d.z * v.z;
            let most = flat * across;
            if (need >= most) {
                continue;
            }
            if (need > -most) {
                middle = atan2(v.y, v.x);
                half = acos(need / most);
            }
        }
        let all_round = half >= PI;
        for (var j = 0u; j < STEM_TURNS; j++) {
            let x = (f32(j) + 0.5) / f32(STEM_TURNS);
            var heading_j = 0.0;
            var width = 0.0;
            if (all_round) {
                heading_j = middle + PI * (2.0 * x - 1.0);
                width = 2.0 * PI / f32(STEM_TURNS);
            } else {
                let psi = PI * (x - 0.5);
                heading_j = middle + half * sin(psi);
                width = half * cos(psi) * PI / f32(STEM_TURNS);
            }
            let leg = vec3<f32>(flat * cos(heading_j), flat * sin(heading_j), d.z);
            let along = dot(v, leg);
            let d2 = vv - along * along;
            if (d2 >= r * r) {
                continue;
            }
            let chord = sqrt(r * r - d2);
            let t0 = max(along - chord, 0.0);
            var t1 = along + chord;
            if (t1 <= t0) {
                continue;
            }
            if (t0 > 0.0 && search_ray(p, leg, run.t_min, t0, NONE, true).face != NONE) {
                continue;
            }
            let wall = search_ray(p + leg * t0, leg, run.t_min, t1 - t0, NONE, false);
            if (wall.face != NONE) {
                t1 = t0 + wall.t;
            }
            let nearest = clamp(along, t0, t1);
            let there = p + leg * nearest;
            if (bending && atm_shadowed(p, there)) {
                continue;
            }
            depth_tau(p, leg, nearest);
            let turn = heading_j - heading;
            var any = 0.0;
            for (var b = 0u; b < BANDS; b++) {
                counted[b] = energy[b] * wrapped_cauchy(turn, num(at + 2u * BANDS + b)) * width *
                             exp(-tau[b]);
                any += counted[b];
            }
            if (!(any > 0.0)) {
                continue;
            }
            var reach = nearest;
            var absorbing = ray.travelled + nearest;
            if (bending) {
                reach = run.c * atm_straight_time(p, there);
                absorbing = ray_path + nearest;
            }
            deposit(k, (t1 - t0) / num(rec + 4u), ray.travelled + reach, absorbing, -leg);
        }
    }
}

// -- the rays ----------------------------------------------------------------------------

fn load(at: u32) {
    ray.pos = bitcast<vec3<f32>>(vec3<u32>(states[at], states[at + 1u], states[at + 2u]));
    ray.travelled = bitcast<f32>(states[at + 3u]);
    ray.dir = bitcast<vec3<f32>>(vec3<u32>(states[at + 4u], states[at + 5u], states[at + 6u]));
    ray.last = states[at + 7u];
    ray.rng = vec4<u32>(states[at + 8u], states[at + 9u], states[at + 10u], states[at + 11u]);
    ray.order = bitcast<i32>(states[at + 12u]);
    ray.root = states[at + 13u];
    ray.cutoff = bitcast<f32>(states[at + 14u]);
    ray_path = bitcast<f32>(states[at + 15u]);
    ray.away = bitcast<vec3<f32>>(vec3<u32>(states[at + 16u], states[at + 17u], states[at + 18u]));
    ray.leaving = states[at + 19u];
    for (var b = 0u; b < BANDS; b++) {
        energy[b] = bitcast<f32>(states[at + 20u + b]);
        seen[b] = bitcast<f32>(states[at + 20u + BANDS + b]);
        alone[b] = bitcast<f32>(states[at + 20u + 2u * BANDS + b]);
    }
}

fn store(at: u32) {
    let pos = bitcast<vec3<u32>>(ray.pos);
    let dir = bitcast<vec3<u32>>(ray.dir);
    let away = bitcast<vec3<u32>>(ray.away);
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
    states[at + 15u] = bitcast<u32>(ray_path);
    states[at + 16u] = away.x;
    states[at + 17u] = away.y;
    states[at + 18u] = away.z;
    states[at + 19u] = ray.leaving;
    for (var b = 0u; b < BANDS; b++) {
        states[at + 20u + b] = bitcast<u32>(energy[b]);
        states[at + 20u + BANDS + b] = bitcast<u32>(seen[b]);
        states[at + 20u + 2u * BANDS + b] = bitcast<u32>(alone[b]);
    }
}

// Ray `k` from the source; false if the source sends nothing its way.
fn start(k: u32) -> bool {
    let at = k * START;
    var loudest = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        let e = bitcast<f32>(starts[at + 3u + b]);
        energy[b] = e;
        seen[b] = 1.0;
        alone[b] = 1.0;
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
    ray_path = 0.0;
    ray.away = vec3<f32>(0.0);
    ray.leaving = 0u;
    atomicAdd(&tallies[3], 1u);
    return true;
}

// Counts the ray as escaped when no face lies anywhere ahead of it.
fn escaped() {
    let beyond = search_ray(ray.pos, ray.dir, 0.0, 1e30, ray.last, true);
    if (beyond.face == NONE) {
        atomicAdd(&tallies[2], 1u);
    }
}

// A straight leg of the ray, to the face it meets (1), to where it scatters in the
// plants (2), or to its end (0: out of time, or out of the model).
fn straight_leg() -> u32 {
    let reach = run.max_length - ray.travelled;
    let hit = search_ray(ray.pos, ray.dir, run.t_min, reach, ray.last, false);
    let found = hit.face != NONE;
    var segment = select(reach, hit.t, found);
    var scatter_at = FAR;
    var tracked = false;
    if ((run.flags & PLANTS) != 0u) {
        tracked = track_build(ray.pos, ray.dir, segment);
        if (tracked) {
            scatter_at = track_collide();
            if (found && scatter_at > hit.t - run.offset) {
                scatter_at = FAR;
            }
            segment = min(segment, scatter_at);
        }
    }
    if ((run.flags & SPECULAR_RAIN) == 0u) {
        detectors(ray.pos, ray.dir, segment, 1.0, -1.0, 1.0, tracked);
    }
    if (scatter_at < FAR) {
        ray.travelled += segment;
        ray.pos += ray.dir * segment;
        leg_scatter_at = scatter_at;
        return 2u;
    }
    if (!found) {
        // Out of time, or out of the model: then no face lies anywhere ahead. (The
        // search above only looks as far as the ray can go.)
        escaped();
        return 0u;
    }
    ray.travelled += hit.t;
    ray.pos += ray.dir * hit.t;
    leg_hit = hit;
    leg_metres = hit.t;
    if (tracked) {
        track_weight(hit.t);
        for (var b = 0u; b < BANDS; b++) {
            energy[b] *= weights[b];
        }
    }
    return 1u;
}

// A leg of the ray bent by the atmosphere, chord by chord, to the first face it meets
// (1), to where it scatters in the plants (2), or to its end (0). (One way out of the
// loop, and what the leg ends at done after it: Direct3D 12's software GPU, WARP, ran
// none of a branch that returned from inside the loop, though the leg returned its
// number.)
fn bent_leg() -> u32 {
    bent_init(ray.pos, ray.dir);
    leg_metres = 0.0;
    var went = 0u;
    // The chord the leg ends on, from where it starts, and how far along it.
    var end_start = vec3<f32>(0.0);
    var end_along = vec3<f32>(0.0);
    var end_t = 0.0;
    var end_rate = 0.0;
    var end_stretch = 0.0;
    var end_arc = 0.0;
    var end_bent = bent;
    var end_away = false;
    while (ray.travelled < run.max_length) {
        let before = bent;
        let use_away = ray.leaving != 0u;
        let budget = 1.05 * (run.max_length - ray.travelled) / run.c * bent_speed_in(bent.z) +
                     run.offset;
        let chord = bent_chord(budget, use_away);
        let span = chord.end - chord.start;
        let len = length(span);
        if (len > 0.0) {
            let along = span / len;
            // A face the chord stops just short of (nearer than the search would look
            // from its end) is met now, not passed through on the next chord: looking
            // twice that far on leaves room for single precision's rounding between the
            // two searches (a ray stopping 0.4 mm short of the sky left through it). A
            // chord from the open air (after a turn or a scattering) meets a face however
            // near.
            let nearest = select(0.0, run.t_min, ray.last != NONE);
            let reach = len + 2.0 * run.t_min;
            let met = search_ray(chord.start, along, nearest, reach, ray.last, false);
            let found = met.face != NONE;
            var segment = select(len, met.t, found);
            let rate = run.c * chord.time / len;
            let stretch = chord.arc / len;
            var scatter_at = FAR;
            var tracked = false;
            if ((run.flags & PLANTS) != 0u) {
                tracked = track_build(chord.start, along, segment);
                if (tracked) {
                    scatter_at = track_collide();
                    if (found && scatter_at > met.t - run.offset) {
                        scatter_at = FAR;
                    }
                    segment = min(segment, scatter_at);
                }
            }
            if ((run.flags & SPECULAR_RAIN) == 0u) {
                detectors(chord.start, along, segment, rate, ray_path, stretch, tracked);
            }
            if (scatter_at < FAR) {
                went = 2u;
                end_t = segment;
                leg_scatter_at = scatter_at;
            } else {
                if (tracked) {
                    track_weight(segment);
                    for (var b = 0u; b < BANDS; b++) {
                        energy[b] *= weights[b];
                    }
                }
                if (found) {
                    went = 1u;
                    end_t = met.t;
                    leg_hit = met;
                }
            }
            if (went != 0u) {
                end_start = chord.start;
                end_along = along;
                end_rate = rate;
                end_stretch = stretch;
                end_arc = chord.arc * end_t / len;
                end_bent = before;
                end_away = use_away;
                break;
            }
        }
        ray.travelled += run.c * chord.time;
        ray_path += chord.arc;
        leg_metres += chord.arc;
        ray.pos = chord.end;
        ray.dir = bent_direction();
        // Out of the model: it left the scene, as through the open sky.
        if (!encloses(ray.pos)) {
            break;
        }
        if (chord.turned) {
            ray.leaving = 0u;  // back towards the face it left
            ray.last = NONE;
        } else if (!(len > 0.0)) {
            break;
        }
    }
    if (went == 0u) {
        escaped();
        return 0u;
    }
    // Where it meets the face, or scatters in the plants, going the ray's way there.
    ray.travelled += end_rate * end_t;
    ray_path += end_stretch * end_t;
    if (went == 1u) {
        leg_metres += end_stretch * end_t;
    }
    ray.pos = end_start + end_along * end_t;
    bent = end_bent;
    bent_chord(end_arc, end_away);
    ray.dir = bent_direction();
    return went;
}

// The weight window after a reflection or a scattering in window `window`: splits the
// ray (its copies go to the other half of the state buffer, after the `n_in` rays of
// this round) or plays roulette with it; false when it dies.
fn weigh(n_in: u32, window: u32) -> bool {
    if ((run.flags & SPLITTING) == 0u) {
        return true;
    }
    let expected = num(run.target_at + min(window, run.targets - 1u));
    if (!(expected > 0.0)) {
        return true;
    }
    var now = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        now += energy[b];
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
            energy[b] /= f32(extra + 1u);
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
            energy[b] /= survive;
        }
    }
    return true;
}

// The ray scatters in the plants at ray.pos, leg_scatter_at metres along its last
// piece: in medium m, its bands weighted by what they met on the way and by m's
// scattering against the rate the scattering was drawn at; false once it is done.
fn scatter_in_plants(n_in: u32) -> bool {
    let m = track_medium();
    let at = medium_at(m);
    track_weight(leg_scatter_at);
    var mean = 0.0;
    var kept = 0.0;
    var mean_g = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        energy[b] *= weights[b];
        mean += energy[b];
        energy[b] *= num(at + BANDS + b) / rate[m];
        kept += energy[b];
        mean_g += energy[b] * num(at + 2u * BANDS + b);
        alone[b] = 0.0;
    }
    let window = min(run.windows - 1u, u32(ray.travelled / run.c / run.window_step));
    add_fixed(COUNTERS + 4u * window, mean / f32(BANDS) * run.window_scale);
    add_one(COUNTERS + 4u * window + 2u);
    if (!(kept > 0.0)) {
        return false;
    }
    mean_g /= kept;
    ray.last = NONE;
    ray.leaving = 0u;
    let rains = (run.flags & (DIFFUSE_RAIN | SPECULAR_RAIN)) != 0u;
    if (upright(m)) {
        // Stems turn it sideways, keeping its slope, and send each listener what its
        // next leg carries through the listener's sphere, on average over the turns.
        if (rains) {
            stem_rain(at);
        }
        let turn = wrapped_cauchy_angle(mean_g);
        let cos_turn = cos(turn);
        let sin_turn = sin(turn);
        let d = ray.dir;
        ray.dir = normalize(vec3<f32>(d.x * cos_turn - d.y * sin_turn,
                                      d.x * sin_turn + d.y * cos_turn, d.z));
        let drawn = wrapped_cauchy(turn, mean_g);
        for (var b = 0u; b < BANDS; b++) {
            energy[b] *= wrapped_cauchy(turn, num(at + 2u * BANDS + b)) / drawn;
            seen[b] = select(1.0, 0.0, rains);  // the rain covered it
        }
    } else {
        // Foliage sends its share straight to each listener as the phase function has
        // it, and the ray goes on in a direction drawn from it.
        if (rains) {
            foliage_rain(at);
        }
        let cosine = henyey_greenstein_cosine(mean_g);
        let azimuth = 2.0 * PI * random();
        let d = ray.dir;
        let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.x) < 0.9);
        let t = normalize(cross(d, helper));
        let u = cross(d, t);
        let sine = sqrt(max(0.0, 1.0 - cosine * cosine));
        ray.dir = normalize(d * cosine + (t * cos(azimuth) + u * sin(azimuth)) * sine);
        let drawn = henyey_greenstein(cosine, mean_g);
        for (var b = 0u; b < BANDS; b++) {
            energy[b] *= henyey_greenstein(cosine, num(at + 2u * BANDS + b)) / drawn;
            seen[b] = select(1.0, 0.0, rains);  // the rain covered it
        }
    }
    var peak = 0.0;
    for (var b = 0u; b < BANDS; b++) {
        peak = max(peak, energy[b]);
    }
    if (peak < ray.cutoff) {
        return false;
    }
    return weigh(n_in, window);
}

// The ray's next leg and reflection; false once it is done. `n_in` is the number of
// rays this round, where the copies' place starts.
fn advance(n_in: u32) -> bool {
    if (ray.travelled >= run.max_length) {
        return false;
    }
    var went = 0u;
    if ((run.flags & ATMOSPHERE) != 0u) {
        went = bent_leg();
    } else {
        went = straight_leg();
    }
    if (went == 0u) {
        return false;
    }
    // PLANTS first: a shader built without plants leaves out their code.
    if (PLANTS != 0u && went == 2u) {
        return scatter_in_plants(n_in);
    }
    let hit = leg_hit;
    let face = run.face_at + FACE * hit.face;
    var n = point(face);
    if (dot(n, ray.dir) > 0.0) {
        n = -n;
    }
    ray.last = hit.face;
    // The ray leaves the face on the side it came from (when it reflects).
    ray.away = n;
    ray.leaving = 1u;
    ray.order += 1;
    let sizes = (run.flags & SIZES) != 0u && ray.order >= run.size_from_order;
    var f_width = 0.0;
    var f_length = 0.0;
    if (sizes) {
        // Christensen and Rindel's scattering of a surface's limited size.
        let across = max(num(face + 3u) * max(-dot(n, ray.dir), 1e-3), 1e-9);
        let along = max(num(face + 4u), 1e-9);
        f_width = run.c * leg_metres / (2.0 * across * across);
        f_length = run.c * leg_metres / (2.0 * along * along);
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
        mean += energy[b];
        reflected[b] = energy[b] * (1.0 - num(face + 5u + b));
        scattered[b] = reflected[b] * scatter[b];
        peak = max(peak, reflected[b]);
        p += scatter[b];
        alone[b] *= 1.0 - scatter[b];
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
            energy[b] = reflected[b];
            seen[b] = select(1.0, 1.0 - scatter[b], diffuse_rain);
        }
    } else if (random() < p) {
        for (var b = 0u; b < BANDS; b++) {
            energy[b] = scattered[b] / p;
            seen[b] = select(1.0, 0.0, diffuse_rain);
            alone[b] = 0.0;
        }
        ray.dir = lambert(n);
    } else {
        for (var b = 0u; b < BANDS; b++) {
            energy[b] = (reflected[b] - scattered[b]) / (1.0 - p);
            seen[b] = 1.0;
            alone[b] = select(0.0, 1.0, alone[b] > 0.0);
        }
        ray.dir = reflect(ray.dir, n);
    }
    // The next leg starts just off the face.
    ray.pos += n * run.offset;
    return weigh(n_in, window);
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

// One chord of each of run.fresh rays bent by the atmosphere (as _native.bent_chords):
// from `starts` (per ray its origin, direction, budget and the normal of the plane it
// leaves, zero for none) into `states` (per ray the chord's end, the direction there,
// its metres of arc, its seconds and whether it turned back), for the tests.
@compute @workgroup_size(64)
fn chords(@builtin(global_invocation_id) id: vec3<u32>,
          @builtin(num_workgroups) groups: vec3<u32>) {
    let i = id.x + id.y * groups.x * 64u;
    if (i >= run.fresh) {
        return;
    }
    let at = __CHORD_IN__u * i;
    let origin = bitcast<vec3<f32>>(vec3<u32>(starts[at], starts[at + 1u], starts[at + 2u]));
    let dir = bitcast<vec3<f32>>(vec3<u32>(starts[at + 3u], starts[at + 4u], starts[at + 5u]));
    let budget = bitcast<f32>(starts[at + 6u]);
    ray.away = bitcast<vec3<f32>>(vec3<u32>(starts[at + 7u], starts[at + 8u], starts[at + 9u]));
    let use_away = any(ray.away != vec3<f32>(0.0));
    bent_init(origin, dir);
    let chord = bent_chord(budget, use_away);
    // Metal packs a struct's vec3 that a scalar follows, and cannot cast it whole.
    let end = vec3<u32>(
        bitcast<u32>(chord.end.x), bitcast<u32>(chord.end.y), bitcast<u32>(chord.end.z)
    );
    let there = bitcast<vec3<u32>>(bent_direction());
    let out = __CHORD_OUT__u * i;
    states[out] = end.x;
    states[out + 1u] = end.y;
    states[out + 2u] = end.z;
    states[out + 3u] = there.x;
    states[out + 4u] = there.y;
    states[out + 5u] = there.z;
    states[out + 6u] = bitcast<u32>(chord.arc);
    states[out + 7u] = bitcast<u32>(chord.time);
    states[out + 8u] = select(0u, 1u, chord.turned);
}
"""
    )
    .replace("__COUNTERS__", str(_COUNTERS))
    .replace("__SHAPE__", str(_SHAPE))
    .replace("__MAX_MEDIA__", str(MAX_MEDIA))
    .replace("__MAX_SPANS__", str(MAX_SPANS))
    .replace("__CHORD_IN__", str(_CHORD_IN))
    .replace("__CHORD_OUT__", str(_CHORD_OUT))
)


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


def _atmosphere_rows(atmosphere, wind_direction) -> tuple[np.ndarray, np.ndarray] | None:
    """The atmosphere's rows (height, speed in still air, wind speed) and the way the
    wind blows, checked as the CPU checks them; None where the rays go straight."""
    if atmosphere is None:
        return None
    rows = np.asarray(atmosphere, float).reshape(-1, 3)
    if len(rows) == 0:
        return None
    wind = np.asarray(wind_direction, float).reshape(-1)
    z, still, speed = rows.T
    if not (
        len(rows) >= 2
        and z[0] == 0.0
        and np.all(np.diff(z) > 0.0)
        and np.all(still > 0.0)
        and np.all(np.isfinite(speed))
        and np.all(still - np.abs(speed) > 0.0)
        and len(wind) == 2
        and abs(np.hypot(*wind) - 1.0) < 1e-6
    ):
        raise ValueError(
            "trace_stochastic: atmosphere rows must start at height 0 and rise, with speeds "
            "above the wind's, and wind_direction a unit vector"
        )
    return rows, wind


def _atmosphere_tables(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per bucket of the wind's cosine to a direction (one without wind), at each node:
    the horizontal distance from where the ray that grazes the ground touches it to the
    node's height, infinite above where that ray turns back down (or where the speed
    never falls below the ground's: no shadow); and the integral of dz / c from the
    ground up (``Atmosphere::prepare`` in atmosphere.hpp)."""
    z, still, wind = rows.T
    n = len(z)
    windy = bool(np.any(wind != 0.0))
    along = np.linspace(-1.0, 1.0, _WIND_BUCKETS) if windy else np.zeros(1)
    c = still[None, :] + wind[None, :] * along[:, None]
    dz = np.diff(z)[None, :]
    ca, cb = c[:, :-1], c[:, 1:]
    x = (cb - ca) / ca
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(np.abs(x) < 1e-8, 1.0 - 0.5 * x, np.log1p(x) / x)
    slow = np.zeros((len(along), n))
    slow[:, 1:] = np.cumsum(dz * log_ratio / ca, axis=1)
    c0 = c[:, :1]
    p = 1.0 / c0
    # The grazing ray goes on up through node i + 1 while cb < c0, from the ground on.
    rising = np.logical_and.accumulate(cb < c0, axis=1)
    sa = np.sqrt(np.maximum(1.0 - (p * ca) ** 2, 0.0))
    sb = np.sqrt(np.maximum(1.0 - (p * cb) ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(rising, p * (ca + cb) * dz / (sa + sb), 0.0)
    reach = np.full((len(along), n), np.inf)
    reach[:, 0] = 0.0
    reach[:, 1:] = np.where(rising, np.cumsum(step, axis=1), np.inf)
    return reach, slow


def _plant_tables(plants, outline, absorption, scattering, asymmetry, upright, bands, centre):
    """The vegetation's volumes (17 words each), its outlines' corners and its media
    (absorption, scattering and mean cosine per band, then whether it stands upright),
    checked as the CPU checks them, in the shader's coordinates; None without plants."""
    if plants is None:
        return None
    rows = np.asarray(plants, float)
    if rows.size == 0:
        return None
    if any(a is None for a in (outline, absorption, scattering, asymmetry, upright)):
        raise ValueError(
            "trace_stochastic: plants need plant_outline, plant_absorption, plant_scattering, "
            "plant_asymmetry and plant_upright"
        )
    outline = np.asarray(outline, float).reshape(-1, 2)
    absorption, scattering, asymmetry = (
        np.atleast_2d(np.asarray(a, float)) for a in (absorption, scattering, asymmetry)
    )
    upright = np.asarray(upright).reshape(-1)
    media = len(upright)
    if (
        rows.ndim != 2
        or rows.shape[1] != 8
        or absorption.shape != (media, bands)
        or scattering.shape != (media, bands)
        or asymmetry.shape != (media, bands)
    ):
        raise ValueError(
            "trace_stochastic: plants must have rows of 8, plant_outline rows of (x, y), and "
            "one row of plant_absorption, plant_scattering and plant_asymmetry per medium, "
            "with one value per band"
        )
    if media > MAX_MEDIA:
        raise Unsupported(f"the GPU tracer takes at most {MAX_MEDIA} media of vegetation")
    bad = ValueError(
        "trace_stochastic: plants must be prisms of 3 or more corners between rising heights "
        "or spheroids of positive size, each with a medium, absorption and scattering not "
        "negative and mean cosines between -1 and 1"
    )
    if (
        media == 0
        or not np.all(absorption >= 0.0)
        or not np.all(scattering >= 0.0)
        or not np.all(np.abs(asymmetry) < 1.0)
    ):
        raise bad
    shapes = np.zeros((len(rows), _SHAPE), np.uint32)
    corners = outline - centre[:2]
    for i, row in enumerate(rows):
        kind, medium = int(row[0]), int(row[1])
        if kind not in (0, 1) or not 0 <= medium < media or row[0] != kind or row[1] != medium:
            raise bad
        shapes[i, 0], shapes[i, 1] = kind, medium
        if kind == 0:
            z0, z1 = row[2] - centre[2], row[3] - centre[2]
            first, count = int(row[4]), int(row[5])
            if count < 3 or first + count > len(corners) or not z1 > z0:
                raise bad
            shapes[i, 2:4] = _words([z0, z1])
            shapes[i, 4:6] = (first, count)
            part = corners[first : first + count]
            lo = (*part.min(0), z0)
            hi = (*part.max(0), z1)
        else:
            x, y, z, radius, half = row[2:7]
            if not (radius > 0.0 and half > 0.0):
                raise bad
            at = np.array([x, y, z]) - centre
            shapes[i, 6:9] = _words(at)
            shapes[i, 9:11] = _words([radius, half])
            lo = at - (radius, radius, half)
            hi = at + (radius, radius, half)
        shapes[i, 11:14] = _words(lo)
        shapes[i, 14:17] = _words(hi)
    table = np.zeros((media, 3 * bands + 1), np.uint32)
    table[:, : 3 * bands] = _words(np.hstack([absorption, scattering, asymmetry]))
    table[:, 3 * bands] = (upright != 0).astype(np.uint32)
    return shapes.reshape(-1), _words(corners.reshape(-1)), table.reshape(-1)


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
def _module(bands: int, features: int):
    """The shader for rays of ``bands`` bands, with the code for ``features`` (of
    ``_ATMOSPHERE`` and ``_PLANTS``) only."""
    code = _SHADER.replace("__BANDS__", str(bands)).replace("__FEATURES__", str(features))
    return gpu._device().create_shader_module(code=code)


@functools.lru_cache(maxsize=8)
def _pipeline(entry: str, bands: int, features: int):
    """The shader's kernel ``entry`` (the tracer, "main", or the chord test, "chords")
    as :func:`_module` builds it, compiled once per session."""
    device = gpu._device()
    layout = device.create_pipeline_layout(bind_group_layouts=[_bindings()])
    return device.create_compute_pipeline(
        layout=layout, compute={"module": _module(bands, features), "entry_point": entry}
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
        boxes = np.asarray(layout["boxes"], float).reshape(-1, 6)
        #: The model's box (its lowest and highest corner) in the shader's coordinates.
        self.box = boxes[0, :3] - self.centre, boxes[0, 3:] - self.centre
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

    def _atmosphere(self, bending, ground: float, sagitta: float) -> tuple[list, dict]:
        """The atmosphere's words for the scene (nodes, then its tables) and the run's
        settings for it; none without one."""
        if bending is None:
            return [], {}
        rows, wind = bending
        reach, slow = _atmosphere_tables(rows)
        return (
            [rows.T.reshape(-1), reach.reshape(-1), slow.reshape(-1)],
            {
                "atm_nodes": len(rows),
                "atm_buckets": len(reach),
                "ground": ground - self.centre[2],
                "sagitta": sagitta,
                "wind_x": wind[0],
                "wind_y": wind[1],
            },
        )

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
        bending=None,
        ground: float = 0.0,
        chord_sagitta: float = 0.02,
        plants=None,
    ) -> tuple:
        """Traces the rays ``starts`` (directions and the source's energy weights, as
        ``_native.ray_starts`` gives them for the streams from ``first_ray`` on), each
        carrying 1 / ``total`` of the source's energy; the rest as for
        ``_native.trace_stochastic``, with ``bending`` the atmosphere's rows and wind
        direction (see :func:`_atmosphere_rows`) and ``plants`` the vegetation's
        tables (see :func:`_plant_tables`), or None. Returns (energy, directional,
        escaped rays, ray_energy, reflections, traced rays)."""
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

        # The scene: source, faces, listeners, directions, air, band centres, targets,
        # the model's box, the atmosphere and the plants.
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
        atmosphere, in_air = self._atmosphere(bending, ground, chord_sagitta)
        parts = [
            np.asarray(source, float) - self.centre,
            faces.ravel(),
            listeners.ravel(),
            directions.ravel(),
            np.asarray(air, float),
            hz,
            target,
            np.concatenate(self.box),
            *atmosphere,
        ]
        offsets = np.cumsum([0, *(len(p) for p in parts)])
        values = np.concatenate(parts)
        values = np.minimum(values, np.finfo(np.float32).max)  # no time limit: the largest
        scene_words = [_words(values)]
        greenery = {"shapes": 0, "shape_at": 0, "outline_at": 0, "media": 0, "medium_at": 0}
        if plants is not None:
            shapes, corners, media = plants
            at = len(values)
            greenery = {
                "shapes": len(shapes) // _SHAPE,
                "shape_at": at,
                "outline_at": at + len(shapes),
                "media": len(media) // (3 * bands + 1),
                "medium_at": at + len(shapes) + len(corners),
            }
            scene_words += [shapes, corners, media]
        scene = np.concatenate(scene_words)

        # The tallies: counters, windows, energy, directional, families.
        energy_at = _COUNTERS + _WINDOW * windows
        directional_at = energy_at + 2 * n_rec * bands * bins
        family_at = directional_at + 2 * n_rec * n_dirs * bands * dir_bins
        tally_words = family_at + rays
        record = 20 + 3 * bands
        most = min(_MOST_RAYS, self._limit // (8 * record))
        # Bent rays take many chords, and so many searches, to a leg: fewer at a time.
        chords = 1
        if bending is not None:
            rows = bending[0]
            steep = np.abs(np.diff(rows[:, 1]) + np.abs(np.diff(rows[:, 2]))) / np.diff(rows[:, 0])
            kappa = float(steep.max(initial=0.0)) / float(rows[:, 1].min())
            chord = math.sqrt(4.0 * chord_sagitta / kappa) if kappa > 0.0 else math.inf
            chords = max(1, math.ceil(speed_of_sound * max_time / chord))
            most = min(most, max(_GROUP, _SEARCHES // (chords * (1 + n_rec))))
        most = most // _GROUP * _GROUP
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
            | (_SPLITTING if len(target) else 0)
            | (_ATMOSPHERE if bending is not None else 0)
            | (_PLANTS if plants is not None else 0),
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
            "box_at": offsets[7],
            "atm_at": offsets[8] if atmosphere else 0,
            "atm_reach_at": offsets[9] if atmosphere else 0,
            "atm_slow_at": offsets[10] if atmosphere else 0,
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
            # A window takes an addition at every reflection of every ray: 2**40 all
            # together for each reflection per ray.
            "window_scale": 2.0**_FRACTION / (loudest * rays),
            **in_air,
            **greenery,
        }
        pipeline = _pipeline("main", bands, settings["flags"] & (_ATMOSPHERE | _PLANTS))
        halves = (0, most * record)
        kept = spawned = spawn_from = fresh_from = rounds = 0
        counters = np.zeros(_COUNTERS, np.uint32)
        while True:
            fresh = min(rays - fresh_from, max(0, most // 2 - kept - spawned))
            n = kept + spawned + fresh
            if n == 0:
                break
            legs = max(1, min(_MOST_LEGS, _SEARCHES // (n * chords * (1 + n_rec))))
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
        if counters[5]:
            raise Unsupported(
                f"a ray met more than {MAX_SPANS} volumes of vegetation at once on the GPU"
            )
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

    def bent_chords(
        self, origins, directions, budget, away, atmosphere, wind_direction, ground, chord_sagitta
    ) -> tuple:
        """As ``_native.bent_chords``, on the GPU: one chord of each ray bent by the
        atmosphere (see ``atmosphere.hpp``), from ``origins`` along ``directions`` (unit),
        at most ``budget`` metres of arc, stopping where a ray turns back towards the
        plane whose normal is its row of ``away`` (zeros: none). Returns (ends,
        directions at the ends, arc lengths, times, turned), for the tests."""
        wgpu = gpu._wgpu("ray tracing")
        device = self._device
        bending = _atmosphere_rows(atmosphere, wind_direction)
        if bending is None:
            raise ValueError("bent_chords: no atmosphere")
        origins = np.asarray(origins, float).reshape(-1, 3) - self.centre
        n = len(origins)
        rays = np.hstack(
            [
                origins,
                np.asarray(directions, float).reshape(n, 3),
                np.asarray(budget, float).reshape(n, 1),
                np.asarray(away, float).reshape(n, 3),
            ]
        )
        atmosphere_words, in_air = self._atmosphere(bending, float(ground), float(chord_sagitta))
        parts = [np.concatenate(self.box), *atmosphere_words]
        offsets = np.cumsum([0, *(len(p) for p in parts)])
        values = np.minimum(np.concatenate(parts), np.finfo(np.float32).max)
        with self._busy:
            usage = wgpu.BufferUsage
            out = device.create_buffer(
                size=4 * _CHORD_OUT * max(n, 1), usage=usage.STORAGE | usage.COPY_SRC
            )
            buffers = [
                self._uniform,
                self._nodes,
                self._triangles,
                device.create_buffer_with_data(data=_words(values), usage=usage.STORAGE),
                device.create_buffer_with_data(
                    data=_words(rays.reshape(-1)) if n else np.zeros(4, np.uint32),
                    usage=usage.STORAGE,
                ),
                out,
                device.create_buffer(size=4 * _COUNTERS, usage=usage.STORAGE),
            ]
            group = device.create_bind_group(
                layout=_bindings(),
                entries=[
                    {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
                    for i, b in enumerate(buffers)
                ],
            )
            settings = {
                "fresh": n,
                "box_at": offsets[0],
                "atm_at": offsets[1],
                "atm_reach_at": offsets[2],
                "atm_slow_at": offsets[3],
                "offset": _OFFSET * self.scale,
                **in_air,
            }
            device.queue.write_buffer(self._uniform, 0, _uniform(settings))
            encoder = device.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(_pipeline("chords", 1, _ATMOSPHERE))
            compute.set_bind_group(0, group)
            compute.dispatch_workgroups(*_groups(n))
            compute.end()
            device.queue.submit([encoder.finish()])
            words = np.frombuffer(device.queue.read_buffer(out), np.uint32)[: _CHORD_OUT * n]
        found = words.reshape(n, _CHORD_OUT)
        floats = found[:, :8].copy().view(np.float32).astype(np.float64)
        return (
            floats[:, :3] + self.centre,
            floats[:, 3:6],
            floats[:, 6],
            floats[:, 7],
            found[:, 8].astype(np.uint8),
        )


def outdoors_declined(bending: bool, plants: bool) -> str:
    """Why this computer's GPU leaves a run's rays bent by the air (``bending``) through
    vegetation (``plants``) to the CPU, or ''. Direct3D 12's shader compiler (FXC) takes
    minutes over the tracer with either, but more than an hour with both (25 minutes
    with its optimizer, which miscompiles the vegetation: see gpu._primary_backends), so
    the CPU traces those there."""
    adapter = gpu._adapter() if bending and plants else None
    if adapter is not None and adapter.info["backend_type"] == "D3D12":
        return "rays bent by the air through vegetation are traced on the CPU on Direct3D 12"
    return ""


def passes_sound(transmission, face_room, source_room) -> bool:
    """Whether a run's ``transmission`` lets sound through some face into a room, which
    only the CPU tracer follows. The open sky of an outdoor scene lets all sound out of
    the scene, into no room: to the tracers that is a surface that keeps none of it, and
    the GPU traces such a scene as the CPU does."""
    through = np.asarray(transmission, float)
    if not through.size:
        return False
    rooms = np.asarray(face_room, dtype=np.int64).reshape(-1, 2)
    if len(rooms) != len(through) or int(source_room) < 0:
        return True
    return bool(((rooms >= 0).sum(axis=1) > 1)[through.any(axis=1)].any())


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
    atmosphere=None,
    wind_direction=None,
    ground=0.0,
    chord_sagitta=0.02,
    plants=None,
    plant_outline=None,
    plant_absorption=None,
    plant_scattering=None,
    plant_asymmetry=None,
    plant_upright=None,
    *,
    total: int | None = None,
) -> tuple:
    """As ``acoustic_spaces._native.trace_stochastic``, on the GPU. ``total`` (default
    ``rays``) is the number of rays the source's energy is shared between, for tracing
    part of a run's rays here and the rest elsewhere. Raises :class:`Unsupported` for
    what only the CPU traces: transmission into a room (see :func:`passes_sound`), face
    groups, and vegetation of more media than :data:`MAX_MEDIA`."""
    from acoustic_spaces import _native

    del threads, room_level, face_group, moment_time_step
    if passes_sound(transmission, face_room, source_room):
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
    bending = _atmosphere_rows(atmosphere, wind_direction)
    declined = outdoors_declined(bending is not None, plants is not None)
    if declined:
        raise Unsupported(declined)
    on_gpu = tracer(bvh)
    energy, directional, escaped, ray_energy, reflections, traced = on_gpu.trace(
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
        bending=bending,
        ground=float(ground),
        chord_sagitta=float(chord_sagitta),
        plants=_plant_tables(
            plants,
            plant_outline,
            plant_absorption,
            plant_scattering,
            plant_asymmetry,
            plant_upright,
            bands,
            on_gpu.centre,
        ),
    )
    moments = np.zeros((len(np.atleast_2d(receivers)), 0, bands, 0))
    return energy, directional, moments, escaped, ray_energy, reflections, traced


__all__ = ["MAX_MEDIA", "MAX_SPANS", "GpuTracer", "Unsupported", "trace_stochastic", "tracer"]
