"""Cross-check against pyroomacoustics: the same rooms simulated by both programs.

    python validation/pyroomacoustics_crosscheck.py [--rays 100000] [--json results.json]
                                                    [--plots DIR] [--cases specular,diffuse]

Needs ``pip install pyroomacoustics`` (and matplotlib for ``--plots``). Both programs
get the same geometry, absorption and scattering per octave band, the same air
absorption (ISO 9613-1, from this package) and an image-source order of 3 before ray
tracing takes over. Both scatter vector-based; pyroomacoustics has no scattering from the
size of a surface, so this package leaves it out for the comparison and adds a run with
its default, which has it. The impulse responses of both are analysed alike, with
:func:`acoustic_spaces.parameters.from_impulse_response`, and EDT, T30 and C80 are
averaged over the listeners of each room. The script prints a Markdown table per room.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import band_decay, decay_curve, from_impulse_response

SAMPLE_RATE = 48_000
#: Bands compared. pyroomacoustics has no 63 Hz band, and at 8 kHz its filter bank
#: runs into its 16 kHz band.
COMPARED = [1, 2, 3, 4, 5, 6]  # 125 Hz to 4 kHz
PARAMETERS = ("edt", "t30", "c80")

# Coefficients from this package's database (see examples/): painted concrete, carpet,
# an absorbing tile ceiling; for the hall plasterboard, wooden lining and an audience.
CONCRETE = [0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.02, 0.02]
CARPET = [0.03, 0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44]
TILES = [0.49, 0.49, 0.53, 0.53, 0.75, 0.92, 0.99, 0.99]
PLASTER = [0.2, 0.2, 0.15, 0.1, 0.08, 0.04, 0.02, 0.02]
WOOD = [0.27, 0.27, 0.23, 0.22, 0.15, 0.1, 0.07, 0.06]
AUDIENCE = [0.16, 0.16, 0.24, 0.56, 0.69, 0.81, 0.78, 0.78]

BOX = (8.0, 6.0, 3.0)
BOX_SOURCE = [2.0, 3.0, 1.5]
BOX_LISTENERS = [[6.0, 3.5, 1.2], [5.0, 1.5, 1.2], [3.5, 4.8, 1.2], [7.0, 2.0, 1.6]]
HALL_PLAN = [[0.0, -4.0], [0.0, 4.0], [14.0, 8.0], [14.0, -8.0]]
HALL_SOURCE = [1.5, 0.0, 1.6]
HALL_LISTENERS = [[4.0, 1.0, 1.2], [12.0, -3.0, 1.2], [8.0, 4.0, 1.2], [10.0, -1.0, 1.2]]


@dataclass
class Case:
    name: str
    description: str
    #: Surface name -> (absorption per band, scattering).
    surfaces: dict[str, tuple[list[float], float]]
    box: tuple[float, float, float] | None = None
    plan: list[list[float]] | None = None
    height: float = 0.0
    source: list[float] = field(default_factory=lambda: BOX_SOURCE)
    listeners: list[list[float]] = field(default_factory=lambda: BOX_LISTENERS)
    ir_length_s: float = 3.0
    #: A diffuse sound field, where Barron's revised theory applies.
    diffuse: bool = False


def _box_case(
    name: str, description: str, scattering: tuple[float, float, float], length: float = 3.0
) -> Case:
    walls, floor, ceiling = scattering
    return Case(
        name,
        description,
        {"walls": (CONCRETE, walls), "floor": (CARPET, floor), "ceiling": (TILES, ceiling)},
        box=BOX,
        ir_length_s=length,
        diffuse=min(scattering) == 1.0,
    )


CASES = [
    _box_case(
        "specular",
        "8 x 6 x 3 m box, concrete walls, carpet, absorbing ceiling; no scattering",
        (0.0, 0.0, 0.0),
        length=10.0,  # rays parallel to the floor keep going for seconds
    ),
    _box_case("diffuse", "the same box with every surface fully diffusing", (1.0, 1.0, 1.0)),
    _box_case(
        "example",
        "the same box with the example scene's scattering (walls 0.05, floor 0.1, ceiling 0.2)",
        (0.05, 0.1, 0.2),
    ),
    Case(
        "hall",
        "fan-shaped lecture hall (examples/lecture_hall), 1008 m3: plasterboard ceiling, "
        "wooden stage wall, concrete side and back walls, audience floor",
        {
            "floor": (AUDIENCE, 0.7),
            "ceiling": (PLASTER, 0.1),
            "wall_1": (WOOD, 0.15),
            "walls": (CONCRETE, 0.1),
        },
        plan=HALL_PLAN,
        height=6.0,
        source=HALL_SOURCE,
        listeners=HALL_LISTENERS,
        ir_length_s=4.0,
    ),
]


def _scene(case: Case, rays: int, seed: int, from_size: bool = True) -> Scene:
    geometry = (
        {"box": list(case.box)} if case.box else {"floor_plan": case.plan, "height": case.height}
    )
    return Scene.model_validate(
        {
            "name": case.name,
            "geometry": geometry,
            "materials": {
                name: {"absorption": alpha, "scattering": s}
                for name, (alpha, s) in case.surfaces.items()
            },
            "assignments": {name: name for name in case.surfaces},
            "sources": [{"name": "source", "position": case.source}],
            "listeners": [{"name": f"l{k}", "position": p} for k, p in enumerate(case.listeners)],
            "simulation": {
                "early": {"max_order": 3},
                "late": {"rays": rays},
                "scattering": {"from_size": from_size},
                "ir_length_s": case.ir_length_s,
                "seed": seed,
            },
        }
    )


def run_ours(
    case: Case, rays: int, seed: int, from_size: bool = True
) -> tuple[list[np.ndarray], Room, float]:
    room = Room.from_scene(_scene(case, rays, seed, from_size))
    start = time.perf_counter()
    result = simulate(room)
    irs = [
        result["source", f"l{k}"].impulse_response(SAMPLE_RATE, seed=seed + k)
        for k in range(len(case.listeners))
    ]
    return irs, room, time.perf_counter() - start


def _pra_material(alpha: list[float], scattering: float):
    import pyroomacoustics as pra

    bands = list(OCTAVE_BANDS_HZ[1:])
    return pra.Material(
        energy_absorption={"coeffs": alpha[1:], "center_freqs": bands},
        scattering={"coeffs": [scattering] * len(bands), "center_freqs": bands},
    )


def run_pyroomacoustics(case: Case, rays: int, air: np.ndarray) -> tuple[list[np.ndarray], float]:
    import pyroomacoustics as pra

    materials = {name: _pra_material(*value) for name, value in case.surfaces.items()}
    common = {"fs": SAMPLE_RATE, "max_order": 3, "ray_tracing": True, "air_absorption": True}
    if case.box:
        # pyroomacoustics names the walls of a box by compass direction.
        walls = materials["walls"]
        room = pra.ShoeBox(
            list(case.box),
            materials={
                "east": walls,
                "west": walls,
                "north": walls,
                "south": walls,
                "floor": materials["floor"],
                "ceiling": materials["ceiling"],
            },
            **common,
        )
    else:
        # The plan's corners run clockwise; pyroomacoustics wants them anticlockwise.
        # Wall k of the scene is the edge from corner k to corner k + 1.
        n = len(case.plan)
        walls = [materials.get(f"wall_{k + 1}", materials["walls"]) for k in range(n)]
        corners = np.array(case.plan[::-1]).T
        walls = walls[::-1][1:] + walls[::-1][:1]  # edge k of the reversed plan
        room = pra.Room.from_corners(corners, materials=walls, **common)
        room.extrude(
            case.height, materials={"floor": materials["floor"], "ceiling": materials["ceiling"]}
        )
    room.set_air_absorption(list(air[1:]), list(OCTAVE_BANDS_HZ[1:]))
    room.set_ray_tracing(
        n_rays=rays, receiver_radius=0.5, time_thres=case.ir_length_s, energy_thres=1e-9
    )
    room.add_source(case.source)
    room.add_microphone_array(np.array(case.listeners).T)
    start = time.perf_counter()
    room.compute_rir()
    return [np.asarray(room.rir[k][0]) for k in range(len(case.listeners))], (
        time.perf_counter() - start
    )


def _lambert(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u, v = rng.random(len(normals)), rng.random(len(normals))
    r, phi = np.sqrt(u), 2.0 * np.pi * v
    helper = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = np.cross(normals, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    return (
        (r * np.cos(phi))[:, None] * t1
        + (r * np.sin(phi))[:, None] * t2
        + np.sqrt(1.0 - u)[:, None] * normals
    )


def global_decay(
    room: Room, scattering: str, rays: int = 20_000, duration: float = 6.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Decay of the total sound energy in the room after an impulse, per band: times (s)
    and levels (dB re the start), in 1 ms steps, backward-integrated like the Schroeder
    curve of an impulse response, so its T30 compares with a listener's.

    Rays reflect with one of three scattering models, independent of any receiver:

    - ``random``: specularly or, with the mean of the bands' scattering coefficients as
      probability, in a Lambert direction, each band weighted by its own coefficient, so
      its scattered share leaves diffusely (this package's tracer up to version 2);
    - ``vector``: always in the direction of the scattering-weighted sum of the
      specular and a Lambert direction (vector-based scattering, as in pyroomacoustics
      0.10 and ODEON), each band with rays of its own where the bands' coefficients
      differ (pyroomacoustics takes their mean);
    - ``size``: vector-based, with the scattering a surface's limited size adds
      (:func:`~acoustic_spaces.engine.model.size_scattering`), from the order the
      room's diffraction setting leaves it to, each band with rays of its own: this
      package's default since version 3.
    """
    from acoustic_spaces.engine.model import TraceModel

    model = TraceModel.from_room(room)
    sizes = scattering == "size" and model.surface_size is not None
    uniform = np.all(model.scattering == model.scattering[:, :1])
    if scattering == "random" or (uniform and not sizes):
        return _decay(room, model, scattering, None, rays, duration, seed)
    levels = []
    for band in range(model.absorption.shape[1]):
        t, level = _decay(room, model, scattering, band, rays, duration, seed)
        levels.append(level[:, band])
    return t, np.stack(levels, axis=1)


def _decay(room: Room, model, scattering: str, band: int | None, rays, duration, seed):
    """:func:`global_decay` with rays turned by the scattering of ``band``, or by the
    bands' mean (None)."""
    from acoustic_spaces.engine.model import size_scattering
    from acoustic_spaces.spatial.directions import fibonacci_sphere

    rng = np.random.default_rng(seed)
    c, air = room.speed_of_sound(), room.air_attenuation()
    step = 0.001
    slots = int(duration / step)
    bands = model.absorption.shape[1]
    change = np.zeros((slots + 1, bands))  # energy gained (lost) in each time slot
    change[0] = 1.0
    position = np.repeat(np.asarray(room.scene.sources[0].position, float)[None], rays, axis=0)
    direction = fibonacci_sphere(rays)
    energy = np.full((rays, bands), 1.0 / rays)
    time_s = np.zeros(rays)
    face = np.full(rays, -1, dtype=np.int64)
    order = np.zeros(rays, dtype=np.int64)
    sizes = scattering == "size" and model.surface_size is not None
    while len(time_s):
        distance, hit = model.intersect(position, direction, np.full(len(time_s), np.inf), face)
        escaped = hit < 0  # through a gap in the model: lost at once
        slot = np.minimum((time_s / step).astype(int), slots)
        np.add.at(change, slot, -energy * escaped[:, None])
        keep = ~escaped
        position, direction, energy = position[keep], direction[keep], energy[keep]
        time_s, distance, face = time_s[keep], distance[keep], hit[keep]
        order = order[keep] + 1
        time_s = time_s + distance / c
        position = position + distance[:, None] * direction
        kept = energy * (1.0 - model.absorption[face])
        np.add.at(change, np.minimum((time_s / step).astype(int), slots), kept - energy)
        energy = kept
        normal = model.plane_normal[model.face_plane[face]]
        normal = np.where((np.einsum("ij,ij->i", normal, direction) > 0)[:, None], -normal, normal)
        specular = direction - 2.0 * np.einsum("ij,ij->i", direction, normal)[:, None] * normal
        diffuse = _lambert(normal, rng)
        s = model.scattering[face]
        if sizes:
            width, length = model.surface_size[face].T
            cosine = np.abs(np.einsum("ij,ij->i", normal, direction))
            sized = size_scattering(width, length, distance, cosine, c, s)
            s = np.where((order >= model.size_from_order)[:, None], sized, s)
        if scattering == "random":
            # Each band keeps its own expected diffuse share: weighted by s / p or
            # (1 - s) / (1 - p), as the tracer does.
            p = s.mean(axis=1, keepdims=True)
            scattered = rng.random(len(p)) < p[:, 0]
            weight = np.where(
                scattered[:, None],
                np.divide(s, p, out=np.ones_like(s), where=p > 0),
                np.divide(1.0 - s, 1.0 - p, out=np.ones_like(s), where=p < 1),
            )
            slot = np.minimum((time_s / step).astype(int), slots)
            np.add.at(change, slot, energy * (weight - 1.0))
            energy = energy * weight
            direction = np.where(scattered[:, None], diffuse, specular)
        else:
            s = s.mean(axis=1) if band is None else s[:, band]
            mixed = s[:, None] * diffuse + (1.0 - s[:, None]) * specular
            direction = mixed / np.linalg.norm(mixed, axis=1, keepdims=True)
        live = time_s < duration
        position, direction, energy = position[live], direction[live], energy[live]
        time_s, face, order = time_s[live], face[live], order[live]
    t = np.arange(slots) * step
    total = np.cumsum(change, axis=0)[:slots] * np.exp(-np.outer(c * t, air))
    remaining = np.cumsum(total[::-1], axis=0)[::-1]
    return t, 10.0 * np.log10(np.maximum(remaining / remaining[0], 1e-30))


def decay_time(t: np.ndarray, level: np.ndarray, start: float, stop: float) -> np.ndarray:
    """-60 dB over the slope of each band's level (columns) fitted from ``start`` to
    ``stop`` seconds."""
    inside = (t >= start) & (t <= stop)
    return -60.0 / np.polyfit(t[inside], level[inside], 1)[0]


def global_t30(level_t: np.ndarray, level: np.ndarray) -> np.ndarray:
    """T30 of a global decay: the fit from where it is 5 dB down to where it is 35 dB down."""
    t30 = np.full(level.shape[1], np.nan)
    for b in range(level.shape[1]):
        below = np.flatnonzero(level[:, b] <= -5.0), np.flatnonzero(level[:, b] <= -35.0)
        if len(below[1]):
            t30[b] = decay_time(level_t, level[:, b], level_t[below[0][0]], level_t[below[1][0]])
    return t30


def _mean_parameters(irs: list[np.ndarray]) -> dict[str, np.ndarray]:
    found = [from_impulse_response(ir, SAMPLE_RATE, noise=False) for ir in irs]
    return {p: np.nanmean([getattr(f, p) for f in found], axis=0) for p in PARAMETERS}


def _t30_windows(irs: list[np.ndarray]) -> np.ndarray:
    """Per listener and band, the times (s from emission) where the Schroeder curve of
    the impulse response is 5 and 35 dB down: the stretch T30 is fitted to."""
    windows = np.full((len(irs), len(OCTAVE_BANDS_HZ), 2), np.nan)
    for k, ir in enumerate(irs):
        decay = band_decay(ir, SAMPLE_RATE, noise=False)
        for b, energy in enumerate(decay.energy):
            level = decay_curve(energy, 1.0 / SAMPLE_RATE)
            if level[-1] <= -35.0:
                found = np.argmax(level <= -5.0), np.argmax(level <= -35.0)
                windows[k, b] = np.array(found) / SAMPLE_RATE + decay.onset_s
    return windows


def _global_t30(room: Room, scattering: str, windows: np.ndarray, duration: float) -> np.ndarray:
    """T30 of the room's total energy with one scattering model, fitted over the same
    stretch of the decay as each listener's T30 and averaged like those."""
    t, level = global_decay(room, scattering, duration=duration)
    found = np.full(windows.shape[:2], np.nan)
    for k, b in np.ndindex(*found.shape):
        if np.all(np.isfinite(windows[k, b])):
            found[k, b] = decay_time(t, level[:, b], *windows[k, b])
    return np.nanmean(found, axis=0)


def barron_c80(case: Case, volume: float, t: np.ndarray) -> np.ndarray:
    """C80 per band (dB) from Barron's revised theory for a diffuse field with decay
    time ``t`` per band (Barron and Lee, JASA 84, 1988), averaged over the listeners like
    the simulated values."""
    r = np.linalg.norm(np.array(case.listeners) - np.array(case.source), axis=1)[:, None]
    t = np.asarray(t, dtype=float)[None, :]
    reflected = 31200.0 * t / volume * np.exp(-0.04 * r / t)
    early = 100.0 / r**2 + reflected * (1.0 - np.exp(-1.11 / t))
    late = reflected * np.exp(-1.11 / t)
    return np.mean(10.0 * np.log10(early / late), axis=0)


def compare(case: Case, rays: int, seed: int) -> dict:
    # pyroomacoustics has no scattering from the size of a surface: the comparison
    # leaves it out here too, and this package's default adds it.
    ours, room, ours_s = run_ours(case, rays, seed, from_size=False)
    default, default_room, _ = run_ours(case, rays, seed)
    theirs, theirs_s = run_pyroomacoustics(case, rays, room.air_attenuation())
    a, b, d = _mean_parameters(ours), _mean_parameters(theirs), _mean_parameters(default)
    duration = 1.5 * case.ir_length_s
    global_t30 = {
        "vector": _global_t30(room, "vector", _t30_windows(theirs), duration),
        "size": _global_t30(default_room, "size", _t30_windows(default), duration),
    }
    result = {
        "name": case.name,
        "description": case.description,
        "bands_hz": [OCTAVE_BANDS_HZ[k] for k in COMPARED],
        "eyring_s": [float(v) for v in room.reverberation_time("eyring", True)[COMPARED]],
        "seconds": {"acoustic_spaces": ours_s, "pyroomacoustics": theirs_s},
        **{
            p: {
                "acoustic_spaces": [float(v) for v in a[p][COMPARED]],
                "pyroomacoustics": [float(v) for v in b[p][COMPARED]],
                "default": [float(v) for v in d[p][COMPARED]],
            }
            for p in PARAMETERS
        },
        "t30_room_energy": {m: [float(v) for v in t[COMPARED]] for m, t in global_t30.items()},
    }
    if case.diffuse:
        c80 = barron_c80(case, room.volume_m3, global_t30["vector"][COMPARED])
        result["c80_barron"] = [float(v) for v in c80]
    return result


def _band(f: int) -> str:
    return f"{f // 1000} kHz" if f >= 1000 else f"{f} Hz"


def table(result: dict) -> str:
    bands = result["bands_hz"]
    lines = [
        f"**{result['name']}**: {result['description']}.",
        "",
        "| | " + " | ".join(_band(f) for f in bands) + " |",
        "|---|" + "---:|" * len(bands),
    ]
    units = {"edt": "s", "t30": "s", "c80": "dB"}
    for p in PARAMETERS:
        ours = np.array(result[p]["acoustic_spaces"])
        theirs = np.array(result[p]["pyroomacoustics"])
        fmt = "{:.1f}" if p == "c80" else "{:.2f}"
        label = f"{p.upper()} ({units[p]})"
        lines.append(f"| {label}, acoustic_spaces | " + " | ".join(map(fmt.format, ours)) + " |")
        lines.append(f"| {label}, pyroomacoustics | " + " | ".join(map(fmt.format, theirs)) + " |")
        if p == "c80":
            diff = [f"{round(d, 1) + 0.0:+.1f} dB" for d in theirs - ours]
        else:
            diff = [f"{round(d):+d} %" for d in 100.0 * (theirs / ours - 1.0)]
        lines.append(f"| {p.upper()} difference | " + " | ".join(diff) + " |")
        if p == "c80" and "c80_barron" in result:
            values = " | ".join(f"{v:.1f}" for v in result["c80_barron"])
            lines.append(f"| C80 (dB), Barron's revised theory | {values} |")
    values = " | ".join(f"{v:.2f}" for v in result["t30"]["default"])
    lines.append(f"| T30 (s), acoustic_spaces with surface size (default) | {values} |")
    rows = {
        "vector": "T30 (s) of the room's energy, vector-based scattering",
        "size": "T30 (s) of the room's energy, with surface size",
    }
    for model, label in rows.items():
        values = result["t30_room_energy"][model]
        lines.append(f"| {label} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    eyring = result["eyring_s"]
    lines.append("| Eyring (s) | " + " | ".join(f"{v:.2f}" for v in eyring) + " |")
    return "\n".join(lines)


def plot(result: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bands = result["bands_hz"]
    x = np.arange(len(bands))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    titles = {"edt": "EDT (s)", "t30": "T30 (s)", "c80": "C80 (dB)"}
    for ax, p in zip(axes, PARAMETERS, strict=True):
        ax.plot(x, result[p]["acoustic_spaces"], "o-", color="C0", label="acoustic_spaces")
        ax.plot(x, result[p]["pyroomacoustics"], "s-", color="C1", label="pyroomacoustics")
        if p == "t30":
            energy = result["t30_room_energy"]
            ax.plot(x, result[p]["default"], "^-", color="C2", label="acoustic_spaces, size")
            ax.plot(x, energy["vector"], ":", color="C1", label="room energy")
            ax.plot(x, energy["size"], ":", color="C2", label="room energy, size")
            ax.plot(x, result["eyring_s"], "--", color="gray", label="Eyring")
        if p == "c80" and "c80_barron" in result:
            ax.plot(x, result["c80_barron"], "--", color="gray", label="Barron")
        ax.set_xticks(x, [_band(f) for f in bands], fontsize=8)
        ax.set_title(titles[p])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(result["description"], fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rays", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cases", default=",".join(c.name for c in CASES))
    parser.add_argument("--json", help="write every number to this file")
    parser.add_argument("--plots", help="draw a PNG per room into this folder")
    parser.add_argument("--tables", help="only print the tables of a file from --json")
    args = parser.parse_args()

    if args.tables:
        with open(args.tables, encoding="utf-8") as f:
            saved = json.load(f)
        version, results, rays = saved["pyroomacoustics"], saved["results"], saved["rays"]
    else:
        import pyroomacoustics

        version, results, rays = pyroomacoustics.__version__, [], args.rays
    print(f"pyroomacoustics {version}, {rays} rays each\n")
    if not args.tables:
        chosen = set(args.cases.split(","))
        for case in CASES:
            if case.name in chosen:
                results.append(compare(case, args.rays, args.seed))
                print(table(results[-1]) + "\n", flush=True)
    else:
        for result in results:
            print(table(result) + "\n")
    if args.plots:
        from pathlib import Path

        Path(args.plots).mkdir(parents=True, exist_ok=True)
        for result in results:
            plot(result, str(Path(args.plots) / f"crosscheck_{result['name']}.png"))
    if args.json and not args.tables:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"pyroomacoustics": version, "results": results, "rays": args.rays}, f)


if __name__ == "__main__":
    main()
