"""Fitted walls: the fit gives the same bits on every platform.

Eleven small cases (tests/reference/fit_golden.npz: a sphere on 19.8 cm cells, a fold, a far
origin, a rounded room, a screen of two materials, a tee refused with code 6, a slab refused with
code 5, a strip, a room with a slab in which 11 regions are owned by a tie of the owner rule
(tie_room), a room of 336 faces whose rows change when the area of an offer of the owner rule
changes (offer_room), and a room far from the origin with open area between two rows in the
diagonal direction (-1, -1, 0) (diagonal_far)) are fitted, and every result is hashed field by
field: every array of cut_cells(), cut_air() and cut_widths(), the cells and weights of
cut_points() on fixed probes, and every entry of cut_stats but the timings. The expected hashes
(tests/reference/fit_golden.json) are those of the Linux x86-64 build; every other platform must
give the same bytes. Set ACOUSTIC_SPACES_WRITE_FIT_GOLDEN=1 to write the expected hashes from this
build instead.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import _native

REFERENCE = Path(__file__).parent / "reference"
DATA = np.load(REFERENCE / "fit_golden.npz")
NAMES = [str(n) for n in DATA["names"]]
EXPECTED_FILE = REFERENCE / "fit_golden.json"
REFUSED = {"tee": 6, "slab": 5}
# Probe offsets from the source in cells: dyadic, so that every probe is the same double everywhere.
OFFSETS = [
    (0.0, 0.0, 0.0),
    (0.5, 0.25, -0.75),
    (-1.5, 1.0, 0.125),
    (2.0, -2.5, 1.5),
    (0.375, 3.0, -1.0),
]


def digest(value):
    """The SHA-256 of a value's bytes: arrays as float64 or int64, strings as UTF-8."""
    if isinstance(value, str):
        data = value.encode()
    elif isinstance(value, (bool, int, np.integer)) and not isinstance(value, float):
        data = np.int64(int(value)).tobytes()
    elif isinstance(value, (float, np.floating)):
        data = np.float64(value).tobytes()
    else:
        a = np.asarray(value)
        a = a.astype(np.float64 if a.dtype.kind == "f" else np.int64)
        data = np.ascontiguousarray(a).tobytes() + str(a.shape).encode()
    return hashlib.sha256(data).hexdigest()[:24]


def fields(name):
    """Each hashed field of case `name`, by its key."""
    v = DATA[name + "/vertices"]
    f = DATA[name + "/faces"]
    mat = DATA[name + "/materials"]
    g = DATA[name + "/grid"]
    nx, ny, nz, materials = (int(x) for x in DATA[name + "/shape"])
    cut = _native.CutFit(v, f, mat, materials, g[0:3], float(g[3]), nx, ny, nz, g[4:7], 0, 2, False)
    out = {}
    stats = cut.cut_stats
    for key in sorted(stats):
        if key not in ("seconds", "stage_seconds"):
            out["stats/" + key] = digest(stats[key])
    if stats["refused"]:
        return out
    for key, value in sorted(cut.cut_cells().items()):
        out["cells/" + key] = digest(value)
    out["air"] = digest(cut.cut_air())
    for a, w in enumerate(cut.cut_widths()):
        out[f"widths/{a}"] = digest(w)
    source = g[4:7]
    probes = np.array([source + float(g[3]) * np.array(o) for o in OFFSETS])
    pts = cut.cut_points(source.reshape(1, 3), probes, 8)
    lists = [pts["source"]] + list(pts["probes"])
    out["points/cells"] = digest([len(x) for x in lists] + [c for x in lists for c, _ in x])
    out["points/weights"] = digest(np.array([w for x in lists for _, w in x], np.float64))
    out["points/flags"] = digest(
        [int(b) for b in pts["probe_found"]]
        + [int(b) for b in pts["probe_in_sight"]]
        + [int(pts["source_on_face"])]
    )
    return out


@pytest.fixture(scope="module")
def expected():
    if os.environ.get("ACOUSTIC_SPACES_WRITE_FIT_GOLDEN"):
        EXPECTED_FILE.write_text(
            json.dumps({n: fields(n) for n in NAMES}, indent=1, sort_keys=True)
        )
    return json.loads(EXPECTED_FILE.read_text())


@pytest.mark.parametrize("name", NAMES)
def test_every_field_has_the_expected_bytes(name, expected):
    got = fields(name)
    stats_refused = got["stats/refused"]
    assert stats_refused == digest(REFUSED.get(name, 0))
    differ = sorted(
        k for k in set(got) | set(expected[name]) if got.get(k) != expected[name].get(k)
    )
    assert not differ, differ
