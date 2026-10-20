"""Check what foliage given as leaves (their area per cubic metre and their width) takes
from sound passing through it, against the limits a thin disc's scattering has in
closed form and against ISO 9613-2's dense-foliage table.

    python validation/leaves.py [--rays N] [--json results.json]

``foliage: {leaf_area_per_m3: F, leaf_width: a}`` makes the engine work out what the
leaves scatter and absorb in each band from thin rigid discs of that size, turned
every way alike (:func:`acoustic_spaces.engine.vegetation.disc_scattering`,
:func:`~acoustic_spaces.engine.vegetation.leaf_absorption`). The script:

- checks the disc's scattering cross-section against its closed forms: small against
  the wavelength, the dipole of the air the disc holds back, 16 (kr)^4 / (81 pi^2) of
  its area turned every way alike (from the added mass of a disc, 8 rho r^3 / 3); large,
  twice its mean projected area, pi r^2 / 2. In between the engine bridges the two, as
  the exact cross-section of a disc of a size near the wavelength is not in closed
  form; the table shows where the bridge puts the rise;
- checks the leaves' absorption against the boundary-layer loss of a rigid wall in
  closed form (Morse and Ingard 1968, 6.4), per unit area and band;
- runs sound through a wood of leaves 50 m deep over a ground that absorbs everything,
  the early paths and the late rays, and compares the wood's insertion loss (the level
  without it less the level with it, from two runs) at listeners 5, 20 and 50 m behind
  it with ISO 9613-2's dense-foliage attenuation for 50 m of foliage (Annex A: its
  rates per metre for 20 to 200 m of foliage), for leaves of several densities and
  widths. The standard's table is empirical, for the foliage of whole woods, so this
  is a comparison of magnitudes, not a check against an exact answer: it shows which
  leaves give about what the standard gives, and how the loss grows with their area
  and width. What the leaves scatter comes back as late sound, so the loss is far less
  than what the early paths lose (their extinction, which the table shows too).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from acoustic_spaces import Room, Scene
from acoustic_spaces.engine import insertion
from acoustic_spaces.engine.vegetation import (
    AIR_GAMMA,
    AIR_PRANDTL,
    AIR_VISCOSITY_M2_S,
    DB_PER_NEPER,
    DISC_RAYLEIGH,
    SHADOW_SHARE,
    disc_scattering,
    leaf_absorption,
    leaves_medium,
)
from acoustic_spaces.materials import OCTAVE_BANDS_HZ
from acoustic_spaces.scene.model import DENSE_FOLIAGE_DB_PER_M

C = 343.0
#: The wood: from x = WOOD_FROM to WOOD_TO, wide and high enough to shade the listeners.
WOOD_FROM, WOOD_TO = 10.0, 60.0
#: Listeners this far behind the wood, 1.5 m up; the source stands at the origin, 1.5 m up.
BEHIND = (5.0, 20.0, 50.0)
HEIGHT = 1.5
#: The leaves compared: (area per cubic metre, width).
LEAVES = ((0.1, 0.05), (0.25, 0.05), (0.5, 0.05), (1.0, 0.05), (0.25, 0.02), (0.25, 0.1))


def check_disc() -> dict:
    """The disc's scattering against its closed forms at the two ends, and the bridge
    between them."""
    kr = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])
    efficiency, cosine = disc_scattering(kr)
    small = DISC_RAYLEIGH * kr**4
    print("A thin rigid disc of radius r turned every way alike: its scattering cross-section")
    print("over its area, and the mean cosine of the angle it scatters through")
    print(f"{'kr':>8}{'engine':>12}{'dipole':>12}{'large':>8}{'cosine':>10}")
    for x, e, s, g in zip(kr, efficiency, small, cosine, strict=True):
        print(f"{x:8g}{e:12.4g}{s:12.4g}{1.0:8g}{g:10.3f}")
    assert np.allclose(efficiency[:3], small[:3], rtol=1e-3)
    assert efficiency[-1] > 0.999 and abs(cosine[-1] - SHADOW_SHARE) < 1e-3
    print()
    return {"kr": kr.tolist(), "efficiency": efficiency.tolist(), "cosine": cosine.tolist()}


def check_absorption() -> dict:
    """The leaves' absorption per unit area against the boundary-layer loss of a rigid
    wall in closed form, at the band centres."""
    f = np.array(OCTAVE_BANDS_HZ, dtype=float)
    k = 2.0 * math.pi * f / C
    omega = 2.0 * math.pi * f
    delta_v = np.sqrt(2.0 * AIR_VISCOSITY_M2_S / omega)
    delta_t = delta_v / math.sqrt(AIR_PRANDTL)
    # Two faces, the tangential velocity's square 2/3 of the wave's on average.
    exact = 2.0 * (k / 2.0) * (delta_v * 2.0 / 3.0 + (AIR_GAMMA - 1.0) * delta_t)
    engine = leaf_absorption(k, C)
    print("A leaf's absorption cross-section over its area (both faces, every way alike)")
    print(f"{'':10}" + "".join(f"{v:>9g}" for v in f))
    print(f"{'engine':10}" + "".join(f"{v:9.2e}" for v in engine))
    print(f"{'exact':10}" + "".join(f"{v:9.2e}" for v in exact))
    assert np.allclose(engine, exact, rtol=1e-12)
    print()
    return {"band_hz": f.tolist(), "engine": engine.tolist(), "exact": exact.tolist()}


def wood_scene(area: float | None, width: float, rays: int) -> Scene:
    """A wood of these leaves across the ground, or without any (None)."""
    vegetation = []
    if area is not None:
        vegetation.append(
            {
                "name": "wood",
                "footprint": [[WOOD_FROM, -150], [WOOD_TO, -150], [WOOD_TO, 150], [WOOD_FROM, 150]],
                "height": 20,
                "foliage": {"leaf_area_per_m3": area, "leaf_width": width},
            }
        )
    return Scene.model_validate(
        {
            "geometry": {
                "ground": {"extent": [[-50, -200], [WOOD_TO + 100, 200]], "sky": 40},
                "vegetation": vegetation,
            },
            "materials": {"soil": {"absorption": [1.0] * 8, "scattering": 0.0}},
            "default_material": "soil",
            "sources": [{"name": "s", "position": [0, 0, HEIGHT]}],
            "listeners": [
                {"name": f"l{k}", "position": [WOOD_TO + x, 0, HEIGHT]}
                for k, x in enumerate(BEHIND)
            ],
            "simulation": {
                "ir_length_s": 1.0,
                "seed": 1,
                "early": {"method": "image_sources", "max_order": 1, "diffraction": "none"},
                "late": {"rays": rays},
            },
        }
    )


def check_wood(rays: int) -> list[dict]:
    """The wood's insertion loss against ISO 9613-2's table for 50 m of dense foliage."""
    depth = WOOD_TO - WOOD_FROM
    iso = depth * np.array(DENSE_FOLIAGE_DB_PER_M)
    print(f"A wood {depth:g} m deep over an absorbing ground: its insertion loss (dB)")
    print(f"{'':34}" + "".join(f"{f:>7g}" for f in OCTAVE_BANDS_HZ))
    print(f"{'ISO 9613-2, dense foliage':34}" + "".join(f"{v:7.2f}" for v in iso))
    rows = []
    for area, width in LEAVES:
        medium = leaves_medium(area, width, "octave", C)
        extinction = depth * medium.extinction * DB_PER_NEPER
        started = time.perf_counter()
        scene = wood_scene(area, width, rays)
        result = insertion.simulate(Room.from_scene(scene), without=["wood"])
        seconds = time.perf_counter() - started
        name = f"{area:g} m²/m³, {100 * width:g} cm wide ({seconds:.0f} s)"
        print(f"{name:34}" + "".join(f"{v:7.2f}" for v in extinction) + "  early paths' loss")
        row = {"leaf_area_per_m3": area, "leaf_width": width, "extinction_db": extinction.tolist()}
        for k, x in enumerate(BEHIND):
            loss = np.asarray(result.insertion_loss[("s", f"l{k}")])
            print(f"{f'  {x:g} m behind it':34}" + "".join(f"{v:7.2f}" for v in loss))
            row[f"insertion_loss_db_{x:g}_m"] = loss.tolist()
        rows.append(row)
    print()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rays", type=int, default=100_000, help="late rays per band")
    parser.add_argument("--json", help="write the results to this JSON file")
    args = parser.parse_args(argv)
    results = {
        "disc": check_disc(),
        "absorption": check_absorption(),
        "wood": check_wood(args.rays),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
