"""Check the late rays through vegetation against an independent particle simulation.

    python validation/vegetation.py [--photons N] [--runs N] [--json results.json]

A source in the middle of a box 200 m across whose walls absorb everything, and a slab
of plants 30 m thick across it, from x = 10 to 40 m. Three receivers, spheres 10 m in
radius 50 m from the source: through the slab (+x), beside it (+y) and behind the
source (-x). The reference follows sound particles one by one in numpy, written apart
from the engine: each goes straight to its next scattering, drawn at the medium's
scattering rate, losing the absorbed share on the way, and leaves in a direction drawn
from the medium's phase function. It counts the energy in each sphere by the length of
track through it, and also as the engine's rain does: the particle's track before it
first scatters, and from every scattering in foliage what the phase function sends
straight to each receiver's centre (a next-event estimate); stems turn sound keeping
its slope, so they send a receiver nothing straight, and from every scattering off
them it counts the track that the next leg would have through each sphere, averaged
over 16 turns drawn at random. The engine's compiled tracer runs the same with its
detector spheres and with its rain (for stems, a quadrature over the turns that cross
the sphere), for five media:

- foliage that scatters every way alike, absorbing nothing: 0.03 per metre;
- foliage that scatters forwards (mean cosine 0.6), 0.02 per metre, and absorbs 0.01;
- stems, which scatter sideways keeping the slope: 0.03 per metre of horizontal travel,
  the turn's mean cosine -0.3 (rigid cylinders at low frequencies scatter backwards);
- leaves of 0.25 m² per cubic metre, 5 cm wide, as the engine works them out in the
  4 kHz band (:func:`~acoustic_spaces.engine.vegetation.leaves_medium`): 0.0535 per
  metre scattered with a mean cosine of 0.14, 0.00075 absorbed;
- the first two in two bands at once, which the engine traces with one set of rays,
  weighting each band by what it meets (the reference runs each band on its own).

Energies are over the free field's at 50 m, the mean of the runs with its standard
error. Both run for 0.6 s, so the longest paths drop out of each alike. Each run also
sends the same particles through empty air, and the table shows its energy less that
run's plus the exact free field's: the direct sound's own randomness, which both have
alike in every medium (the particles set off the same ways), drops out. The first row
shows those runs in empty air against the exact value.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

HALF = 100.0  # the box reaches from -HALF to HALF in x, y and z
SLAB = (10.0, 40.0)  # the plants, in x
RADIUS = 10.0
RECEIVERS = np.array([[50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [-50.0, 0.0, 0.0]])
NAMES = ("through", "beside", "behind")
MAX_TIME = 0.6
SPEED_OF_SOUND = 343.0
CAP = MAX_TIME * SPEED_OF_SOUND  # metres a particle may go
FREE = 1.0 / (4.0 * math.pi * 50.0**2)
TURNS = 16  # turns drawn at each scattering off stems, for their rain
EMPTY = {"absorption": 0.0, "scattering": 0.0, "asymmetry": 0.0, "upright": False}

MEDIA = {
    "isotropic": {"absorption": 0.0, "scattering": 0.03, "asymmetry": 0.0, "upright": False},
    "forward": {"absorption": 0.01, "scattering": 0.02, "asymmetry": 0.6, "upright": False},
    "stems": {"absorption": 0.0, "scattering": 0.03, "asymmetry": -0.3, "upright": True},
    # Leaves of 0.25 m² per m³, 5 cm wide, in the 4 kHz band (engine.vegetation.leaves_medium).
    "leaves": {"absorption": 0.00075, "scattering": 0.0535, "asymmetry": 0.14, "upright": False},
}


def _inside(points: np.ndarray) -> np.ndarray:
    return (points[:, 0] > SLAB[0]) & (points[:, 0] < SLAB[1])


def _slab_length(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Metres of each segment from p along v (not unit) inside the slab."""
    x0, x1 = p[:, 0], p[:, 0] + v[:, 0]
    lo, hi = np.minimum(x0, x1), np.maximum(x0, x1)
    overlap = np.clip(np.minimum(hi, SLAB[1]) - np.maximum(lo, SLAB[0]), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(hi > lo, overlap / (hi - lo), 0.0)
    return share * np.linalg.norm(v, axis=1)


def _hg(mu, g):
    q = 1.0 + g * g - 2.0 * g * mu
    return (1.0 - g * g) / (4.0 * math.pi * q * np.sqrt(q))


def particles(medium: dict, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The reference: (energy in each sphere by track length, energy at each receiver as
    the rain counts it), over the free field's at 50 m."""
    rng = np.random.default_rng(seed)
    mu_a, mu_s, g = medium["absorption"], medium["scattering"], medium["asymmetry"]
    upright = medium["upright"]
    pos = np.zeros((count, 3))
    way = rng.normal(size=(count, 3))
    way /= np.linalg.norm(way, axis=1)[:, None]
    weight, gone = np.ones(count), np.zeros(count)
    fresh = np.ones(count, bool)  # not scattered yet
    alive = np.ones(count, bool)
    spheres, first, events = np.zeros(3), np.zeros(3), np.zeros(3)
    volume = 4.0 / 3.0 * math.pi * RADIUS**3
    while alive.any():
        idx = np.flatnonzero(alive)
        p, q, w, s, new = pos[idx], way[idx], weight[idx], gone[idx], fresh[idx]
        share = np.hypot(q[:, 0], q[:, 1]) if upright else np.ones(len(idx))
        inside = _inside(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            wall = np.min(
                np.where(q > 0, (HALF - p) / q, np.where(q < 0, (-HALF - p) / q, np.inf)), axis=1
            )
            crossings = [(x - p[:, 0]) / q[:, 0] for x in SLAB]
        edge = np.full(len(idx), np.inf)
        for t in crossings:
            edge = np.where((t > 1e-9) & (t < edge), t, edge)
        bound = np.minimum(np.minimum(wall, edge), CAP - s)
        rate = mu_s * share
        with np.errstate(divide="ignore"):
            free = np.where(inside & (rate > 0), rng.exponential(1.0, len(idx)) / rate, np.inf)
        step = np.minimum(free, bound)
        loss = np.where(inside, mu_a * share, 0.0)
        for k, centre in enumerate(RECEIVERS):
            rel = centre - p
            along = np.einsum("ij,ij->i", rel, q)
            d2 = np.einsum("ij,ij->i", rel, rel) - along**2
            half = np.sqrt(np.clip(RADIUS**2 - d2, 0.0, None))
            t0, t1 = np.maximum(along - half, 0.0), np.minimum(along + half, step)
            track = np.where(d2 < RADIUS**2, np.clip(t1 - t0, 0.0, None), 0.0)
            counted = w * np.exp(-loss * 0.5 * (t0 + t1)) * track
            spheres[k] += counted.sum()
            first[k] += counted[new].sum()
        w = w * np.exp(-loss * step)
        p = p + q * step[:, None]
        s = s + step
        scatter = free < bound
        dead = ~scatter & ((wall <= edge) | (s >= CAP - 1e-9))
        hit = np.flatnonzero(scatter)
        if len(hit):
            a = q[hit]
            if upright:
                # What the next leg would carry through each sphere, over many turns.
                u = rng.random((len(hit), TURNS))
                turn = 2.0 * np.arctan((1.0 - g) / (1.0 + g) * np.tan(math.pi * (u - 0.5)))
                c, n = np.cos(turn), np.sin(turn)
                legs = np.stack(
                    [
                        a[:, :1] * c - a[:, 1:2] * n,
                        a[:, :1] * n + a[:, 1:2] * c,
                        np.broadcast_to(a[:, 2:], c.shape),
                    ],
                    axis=-1,
                )
                for k, centre in enumerate(RECEIVERS):
                    rel = centre - p[hit]
                    along = np.einsum("hj,htj->ht", rel, legs)
                    d2 = np.einsum("hj,hj->h", rel, rel)[:, None] - along**2
                    half = np.sqrt(np.clip(RADIUS**2 - d2, 0.0, None))
                    t0, t1 = np.maximum(along - half, 0.0), along + half
                    chord = np.where(d2 < RADIUS**2, np.clip(t1 - t0, 0.0, None), 0.0)
                    near = np.clip(along, t0, t1)
                    reach = (legs * near[..., None]).reshape(-1, 3)
                    inside = _slab_length(np.repeat(p[hit], TURNS, axis=0), reach).reshape(
                        near.shape
                    )
                    depth = (mu_a + mu_s) * share[hit][:, None] * inside
                    late = s[hit][:, None] + near < CAP
                    sent = w[hit][:, None] * np.exp(-depth) * chord * late
                    events[k] += sent.mean(axis=1).sum() / volume
                u = rng.random(len(hit))
                turn = 2.0 * np.arctan((1.0 - g) / (1.0 + g) * np.tan(math.pi * (u - 0.5)))
                c, n = np.cos(turn), np.sin(turn)
                q[hit] = np.column_stack(
                    [a[:, 0] * c - a[:, 1] * n, a[:, 0] * n + a[:, 1] * c, a[:, 2]]
                )
            else:
                for k, centre in enumerate(RECEIVERS):  # what the rain sends each receiver
                    v = centre - p[hit]
                    dist = np.linalg.norm(v, axis=1)
                    cos = np.einsum("ij,ij->i", a, v) / dist
                    depth = (mu_a + mu_s) * _slab_length(p[hit], v)
                    late = s[hit] + np.maximum(dist, RADIUS) < CAP
                    sent = w[hit] * _hg(cos, g) * np.exp(-depth) / np.maximum(dist, RADIUS) ** 2
                    events[k] += sent[late].sum()
                u = rng.random(len(hit))
                if abs(g) < 1e-9:
                    cos = 1.0 - 2.0 * u
                else:
                    t = (1.0 - g * g) / (1.0 - g + 2.0 * g * u)
                    cos = np.clip((1.0 + g * g - t * t) / (2.0 * g), -1.0, 1.0)
                phi = 2.0 * math.pi * rng.random(len(hit))
                helper = np.where(np.abs(a[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
                t1 = np.cross(a, helper)
                t1 /= np.linalg.norm(t1, axis=1)[:, None]
                t2 = np.cross(a, t1)
                sin = np.sqrt(np.clip(1.0 - cos**2, 0.0, None))
                q[hit] = (
                    a * cos[:, None]
                    + (t1 * np.cos(phi)[:, None] + t2 * np.sin(phi)[:, None]) * sin[:, None]
                )
            new[hit] = False
        p = p + q * 1e-9  # off the slab's faces
        pos[idx], way[idx], weight[idx], gone[idx], fresh[idx] = p, q, w, s, new
        alive[idx[dead]] = False
    spheres /= count * volume * FREE
    rain = (first / volume + events) / (count * FREE)
    return spheres, rain


def _tracer_arguments(media: list[dict], rays: int, seed: int, rain: bool) -> dict:
    from acoustic_spaces import _native

    bands = len(media)
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    upright = media[0]["upright"]
    return dict(
        bvh=_native.Bvh(corners * 2 * HALF - HALF, faces),
        source=np.zeros(3),
        directivity_a=np.zeros(0),
        orientation=np.eye(3),
        directivity_table=np.zeros((0, 0, bands)),
        absorption=np.ones((12, bands)),
        scattering=np.zeros((12, bands)),
        air=np.zeros(bands),
        surface_size=np.zeros((0, 2)),
        band_hz=np.zeros(0),
        vector_scattering=False,
        size_from_order=1,
        transmission=np.zeros((0, 0)),
        face_room=np.zeros((0, 2), dtype=np.int64),
        source_room=-1,
        receivers=RECEIVERS,
        directions=np.zeros((0, 3)),
        rays=rays,
        max_time=MAX_TIME,
        time_step=1e-3,
        dir_time_step=1e-2,
        speed_of_sound=SPEED_OF_SOUND,
        receiver_radius=np.full(3, RADIUS),
        diffuse_rain=rain,
        specular_rain=False,
        covered_order=-1,
        covered_time=np.zeros(0),
        seed=seed,
        threads=0,
        window_time_step=0.05,
        first_ray=0,
        split_target=np.zeros(0),
        split_ratio=4.0,
        max_split=8,
        max_family=256,
        room_level=np.zeros(0),
        face_group=np.zeros(0, dtype=np.int64),
        groups=0,
        moment_time_step=0.01,
        plants=np.array([[0, 0, -HALF, HALF, 0, 4, 0, 0]], float),
        plant_outline=np.array(
            [[SLAB[0], -HALF], [SLAB[1], -HALF], [SLAB[1], HALF], [SLAB[0], HALF]]
        ),
        plant_absorption=np.array([[m["absorption"] for m in media]]),
        plant_scattering=np.array([[m["scattering"] for m in media]]),
        plant_asymmetry=np.array([[m["asymmetry"] for m in media]]),
        plant_upright=np.array([upright], dtype=np.uint8),
    )


def tracer(media: list[dict], rays: int, seed: int, rain: bool) -> np.ndarray:
    """The engine: (receivers, bands) energy over the free field's at 50 m."""
    from acoustic_spaces import _native

    energy = _native.trace_stochastic(**_tracer_arguments(media, rays, seed, rain))[0]
    return energy.sum(axis=2) / FREE


def free_field() -> float:
    """The free field's mean over a receiver's sphere, over its value at the centre."""
    d, a = 50.0, RADIUS
    inner = a - (d * d - a * a) / (2.0 * d) * math.log((d + a) / (d - a))
    return 3.0 / (4.0 * math.pi * a**3) * 2.0 * math.pi * inner * d * d


def _stats(runs) -> tuple[np.ndarray, np.ndarray]:
    runs = np.asarray(runs)
    return runs.mean(axis=0), runs.std(axis=0, ddof=1) / math.sqrt(len(runs))


def check(photons: int, runs: int) -> dict:
    out = {}
    exact = free_field()
    cases = [(name, [MEDIA[name]]) for name in MEDIA]
    cases.append(("two bands", [MEDIA["isotropic"], MEDIA["forward"]]))
    print("energy over the free field's at 50 m (mean ± standard error)")
    heads = ("reference, spheres", "engine, spheres", "reference, rain", "engine, rain")
    print("| medium | receiver | " + " | ".join(heads) + " |")
    print("|---|---|---|---|---|---|")
    ref_empty = _stats([particles(EMPTY, photons, seed)[0] for seed in range(runs)])
    eng_empty = _stats([tracer([EMPTY], photons, seed, False)[:, 0] for seed in range(runs)])
    out["none"] = {"exact": exact}
    for k, receiver in enumerate(NAMES):
        out["none"][receiver] = {
            "reference_spheres": [ref_empty[0][k], ref_empty[1][k]],
            "engine_spheres": [eng_empty[0][k], eng_empty[1][k]],
        }
        cells = [f"{mean[k]:.3f} ± {error[k]:.3f}" for mean, error in (ref_empty, eng_empty)]
        print(f"| none (exact {exact:.3f}) | {receiver} | " + " | ".join(cells) + " | - | - |")
    for case, (name, media) in enumerate(cases, start=1):
        # Seeds of their own for each case, so that the cases do not share their luck.
        seeds = [100 * case + run for run in range(runs)]
        eng_free = [tracer([EMPTY], photons, seed, False)[:, 0] for seed in seeds]
        eng_spheres = [tracer(media, photons, seed, False) for seed in seeds]
        eng_rain = [tracer(media, photons, seed, True) for seed in seeds]
        for band, medium in enumerate(media):
            ref_seeds = [1000 * band + seed for seed in seeds]
            ref_free = [particles(EMPTY, photons, seed)[0] for seed in ref_seeds]
            ref = [particles(medium, photons, seed) for seed in ref_seeds]
            columns = {
                "reference_spheres": [r[0] - f + exact for r, f in zip(ref, ref_free, strict=True)],
                "engine_spheres": [
                    e[:, band] - f + exact for e, f in zip(eng_spheres, eng_free, strict=True)
                ],
                "reference_rain": [r[1] - f + exact for r, f in zip(ref, ref_free, strict=True)],
                "engine_rain": [
                    e[:, band] - f + exact for e, f in zip(eng_rain, eng_free, strict=True)
                ],
            }
            stats = {key: _stats(values) for key, values in columns.items()}
            label = name if len(media) == 1 else f"{name}, band {band + 1}"
            rows = []
            for k, receiver in enumerate(NAMES):
                rows.append({key: [mean[k], error[k]] for key, (mean, error) in stats.items()})
                cells = [f"{mean[k]:.3f} ± {error[k]:.3f}" for mean, error in stats.values()]
                print(f"| {label} | {receiver} | " + " | ".join(cells) + " |")
            out[label] = dict(zip(NAMES, rows, strict=True))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--photons", type=int, default=1_000_000, help="particles per run")
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    start = time.time()
    results = check(args.photons, args.runs)
    print(f"({time.time() - start:.0f} s)")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
