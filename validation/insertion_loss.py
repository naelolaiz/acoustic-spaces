"""Check the outdoor results, the level relative to free field and the insertion loss,
against exact solutions.

    python validation/insertion_loss.py [--json results.json]

The results an outdoor run reports (``acoustic-spaces simulate`` and ``map``, the
editor's Results panel) are the level relative to free field, the energy a response
holds over what the same source would bring there in free field, and the insertion
loss of what ``simulation.insertion_loss_of`` names, the level without it less the level
with it, from a second run of the scene without it. The script runs whole scenes
through those (the early paths added in pressure, the late rays, the two runs) and
compares:

- a thin barrier 3 m high over a ground that absorbs everything: its insertion loss at
  listeners 2 to 90 m behind it against the exact Biot-Tolstoy-Medwin solution for a
  half-plane (:mod:`edge_diffraction`), and Maekawa's chart for reference; without the
  barrier the level is that of free field, so the insertion loss is the level relative
  to free field with its sign turned;
- the same barrier on a rigid ground: the insertion loss against the four paths over
  the top (the source and its image in the ground, to the listener and its image), each
  by the exact solution, and without the barrier the direct sound and its reflection.
  The engine's ``first_order`` diffraction takes one reflection, before or after the
  edge, so it leaves out the path reflected both before and after the barrier, a
  quarter of the sound over it; the table shows what that costs. The sound that goes
  over the top, down to the ground and over the top again is left out of both;
- a source 1 m over grass and nothing else: the level relative to free field at 10, 50
  and 200 m against the exact solution for a locally reacting ground
  (:mod:`ground_reflection`), averaged over each band as the engine reads it. In the
  bands up to 500 Hz the engine reads the early paths through band filters, whose
  skirts fill a sharp dip by up to about 1 dB; from 1 kHz up it averages each pair of
  paths' interference over the band exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from edge_diffraction import C, Edge, btm_first  # noqa: E402
from ground_reflection import SLICES, exact_reflection, field  # noqa: E402

from acoustic_spaces import Room, Scene  # noqa: E402
from acoustic_spaces.engine import insertion  # noqa: E402
from acoustic_spaces.engine.diffraction import band_frequencies  # noqa: E402
from acoustic_spaces.materials import OCTAVE_BANDS_HZ  # noqa: E402
from acoustic_spaces.materials.ground import PRESETS, surface_impedance  # noqa: E402

#: The barrier: along y at this x, this high; the source stands at the origin, 1 m up.
BARRIER_X = 10.0
BARRIER_HEIGHT = 3.0
SOURCE = np.array([0.0, 0.0, 1.0])
#: Listeners behind the barrier: x along the ground, 1.5 m up.
BEHIND = (12.0, 20.0, 50.0, 100.0)
LISTENER_HEIGHT = 1.5
#: Over grass: distance along the ground, source and listener heights (m).
GRASS = ((10.0, 1.0, 1.5), (50.0, 1.0, 1.5), (200.0, 1.0, 1.5))
#: Frequencies the exact solutions are averaged over in each octave band, (bands, per
#: band): dense enough for the ripples of four paths that differ by up to 0.6 m.
FREQUENCIES, _ = band_frequencies(96)
K = 2.0 * np.pi * FREQUENCIES.ravel() / C


def barrier_scene(ground: dict, listeners) -> Scene:
    """A barrier across a ground of this material, long and wide enough for its ends
    and the ground's edge to add nothing."""
    far = max(x for x, _, _ in listeners) + 100.0
    return Scene.model_validate(
        {
            "geometry": {
                "ground": {"extent": [[-100, -200], [far, 200]], "sky": 40},
                "barriers": [
                    {
                        "name": "barrier",
                        "path": [[BARRIER_X, -190], [BARRIER_X, 190]],
                        "height": BARRIER_HEIGHT,
                    }
                ],
            },
            "materials": {"ground": ground, "rigid": {"absorption": [0.0] * 8, "scattering": 0.0}},
            "default_material": "ground",
            "assignments": {"barrier": "rigid"},
            "sources": [{"name": "s", "position": SOURCE.tolist()}],
            "listeners": [{"name": f"l{k}", "position": list(p)} for k, p in enumerate(listeners)],
            "simulation": {
                "ir_length_s": 0.4,
                "seed": 1,
                "early": {"method": "image_sources", "max_order": 2, "diffraction": "first_order"},
                "late": {"rays": 2000},
            },
        }
    )


def _engine(scene: Scene) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The engine's level relative to free field and insertion loss of the barrier at
    every listener, per octave band."""
    result = insertion.simulate(Room.from_scene(scene), without=["barrier"])
    names = [listener.name for listener in scene.listeners]
    relative = [result[("s", name)].relative_to_free_field() for name in names]
    loss = [result.insertion_loss[("s", name)] for name in names]
    return relative, loss


def _level(pressure: np.ndarray, distance: float) -> np.ndarray:
    """Octave band energies of ``pressure`` (per wave number of ``K``) in dB re the
    direct sound over ``distance`` in free field."""
    energy = (np.abs(pressure) ** 2).reshape(FREQUENCIES.shape).mean(axis=1)
    return 10.0 * np.log10(energy * distance**2)


def _btm(listener, ground: str) -> np.ndarray:
    """The sound over the barrier at ``listener`` by the exact solution, per wave number,
    as the direct sound in free field is e^(-jkr) / r: over an absorbing ground the one
    path, over a rigid one the four."""
    edge = Edge([BARRIER_X, 0.0, BARRIER_HEIGHT], [0, 1, 0], [0, 0, -1], [1, 0, 0], 2.0)
    listener = np.asarray(listener, dtype=float)
    pressure = btm_first(edge, SOURCE, listener, K)
    if ground == "rigid":
        mirror = np.array([1.0, 1.0, -1.0])
        pressure = pressure + btm_first(edge, SOURCE * mirror, listener, K)
        pressure += btm_first(edge, SOURCE, listener * mirror, K)
        pressure += btm_first(edge, SOURCE * mirror, listener * mirror, K)
    return pressure


def _maekawa(listener) -> np.ndarray:
    """Maekawa's insertion loss (dB) of a thin screen, per octave band."""
    listener = np.asarray(listener, dtype=float)
    top = np.array([BARRIER_X, 0.0, BARRIER_HEIGHT])
    direct = float(np.linalg.norm(listener - SOURCE))
    detour = float(np.linalg.norm(top - SOURCE) + np.linalg.norm(listener - top)) - direct
    fresnel = 2.0 * detour * np.array(OCTAVE_BANDS_HZ) / C
    return 10.0 * np.log10(3.0 + 20.0 * fresnel)


def _bare_level(listener) -> np.ndarray:
    """The level relative to free field without the barrier over a rigid ground, per
    octave band: the direct sound and its reflection, averaged over each band."""
    listener = np.asarray(listener, dtype=float)
    r1 = float(np.linalg.norm(listener - SOURCE))
    r2 = float(np.linalg.norm(listener * [1, 1, -1] - SOURCE))
    pressure = (np.exp(-1j * K * r1) / r1 + np.exp(-1j * K * r2) / r2) * r1
    return _level(pressure, 1.0)


def check_barrier(ground: str) -> list[dict]:
    """The barrier over an absorbing or a rigid ground: the engine against the exact
    solution at every listener."""
    material = (
        {"absorption": [1.0] * 8, "scattering": 0.0}
        if ground == "absorbing"
        else {"absorption": [0.0] * 8, "scattering": 0.0}
    )
    listeners = [(x, 0.0, LISTENER_HEIGHT) for x in BEHIND]
    started = time.perf_counter()
    relative, loss = _engine(barrier_scene(material, listeners))
    seconds = time.perf_counter() - started
    print(f"A barrier {BARRIER_HEIGHT:g} m high, {BARRIER_X:g} m from the source, over")
    print(f"{'an absorbing' if ground == 'absorbing' else 'a rigid'} ground ({seconds:.0f} s)")
    print(f"{'':30}" + "".join(f"{f:>7g}" for f in OCTAVE_BANDS_HZ))
    rows = []
    for listener, engine_relative, engine_loss in zip(listeners, relative, loss, strict=True):
        distance = float(np.linalg.norm(np.array(listener) - SOURCE))
        exact_relative = _level(_btm(listener, ground), distance)
        exact_loss = (0.0 if ground == "absorbing" else _bare_level(listener)) - exact_relative
        name = f"{listener[0] - BARRIER_X:g} m behind it"
        print(f"{name:30}" + "".join(f"{v:7.2f}" for v in exact_loss) + "  exact insertion loss")
        print(
            f"{'':30}"
            + "".join(f"{v:+7.2f}" for v in engine_loss - exact_loss)
            + "  engine - exact"
        )
        row = {
            "listener": list(listener),
            "exact_re_free_field_db": exact_relative.tolist(),
            "engine_re_free_field_db": engine_relative.tolist(),
            "exact_insertion_loss_db": exact_loss.tolist(),
            "engine_insertion_loss_db": engine_loss.tolist(),
        }
        if ground == "absorbing":
            maekawa = _maekawa(listener)
            row["maekawa_insertion_loss_db"] = maekawa.tolist()
            print(
                f"{'':30}"
                + "".join(f"{v:+7.2f}" for v in maekawa - exact_loss)
                + "  Maekawa - exact"
            )
        else:
            bare = _bare_level(listener)
            print(
                f"{'':30}"
                + "".join(f"{v:+7.2f}" for v in engine_relative + engine_loss - bare)
                + "  engine - exact, without the barrier"
            )
        rows.append(row)
    print()
    return rows


def check_grass() -> list[dict]:
    """A source over grass: the engine's level relative to free field against the exact
    solution, per octave band."""
    sigma = PRESETS["grass"][0]
    print("A source over grass: the level relative to free field (dB)")
    print(f"{'':30}" + "".join(f"{f:>7g}" for f in OCTAVE_BANDS_HZ))
    rows = []
    for r, zs, zr in GRASS:
        scene = Scene.model_validate(
            {
                "geometry": {"ground": {"extent": [[-20, -20], [r + 20, 20]], "sky": 20}},
                "materials": {"field": {"impedance": "grass", "scattering": 0.0}},
                "default_material": "field",
                "sources": [{"name": "s", "position": [0, 0, zs]}],
                "listeners": [{"name": "l", "position": [r, 0, zr]}],
                "simulation": {
                    "ir_length_s": 0.2,
                    "seed": 1,
                    "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
                    "late": {"rays": 1000},
                },
            }
        )
        room = Room.from_scene(scene)
        engine = insertion.simulate(room)[("s", "l")].relative_to_free_field()
        c = room.speed_of_sound()
        r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
        exact = []
        for centre in OCTAVE_BANDS_HZ:
            low = centre / math.sqrt(2.0)
            f = low + low * (np.arange(SLICES) + 0.5) / SLICES
            k = 2.0 * math.pi * f / c
            beta = 1.0 / surface_impedance(f, sigma, speed_of_sound=c)
            q = np.array([exact_reflection(b, x, r, zs, zr) for b, x in zip(beta, k, strict=True)])
            exact.append(10.0 * math.log10(np.mean(np.abs(field(q, k, r1, r2)) ** 2)))
        exact = np.array(exact)
        name = f"{r:g} m, heights {zs:g} and {zr:g} m"
        print(f"{name:30}" + "".join(f"{v:7.2f}" for v in exact) + "  exact")
        print(f"{'':30}" + "".join(f"{v:+7.2f}" for v in engine - exact) + "  engine - exact")
        rows.append({"distance_m": r, "exact_db": exact.tolist(), "engine_db": engine.tolist()})
    print()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    results = {
        "barrier_absorbing_ground": check_barrier("absorbing"),
        "barrier_rigid_ground": check_barrier("rigid"),
        "grass": check_grass(),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
