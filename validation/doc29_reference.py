"""Check :mod:`acoustic_spaces.doc29` against the reference cases of ECAC Doc 29 Volume 3
(5th edition, 2026): Part 1's three hypothetical aircraft arriving and departing on
straight and curved routes, twelve cases, and Part 2's flight profiles, 26 cases.

    python validation/doc29_reference.py [--workbook doc29-volume3-part1.xlsx]

- **The sound exposure levels at the receptors** (Volume 3's table B-1, 99 levels,
  which the package holds): each level against the reference, to its 0.01 dB.
- **Every segment** (worksheet B-2_Segment_Results of Volume 3's Part 1 workbook, which
  ECAC publishes beside the volume; ``--workbook`` gives it): the number of segments of
  each case, and for each term of each segment at each receptor, 3516 rows, the
  largest difference from the reference.
- **The grids** (worksheet B-3_Grid_Results, 37458 points of 80 dB SEL or more): the rms
  difference of each case, δRMS, which Volume 3 aims at 0.01 dB or less.
- **Volume 3's own placement of the cuts by speed** of a climbing or descending
  segment: the segment results lay out the lengths of eq. 3-3i, which are along the
  segment, along the ground track, which puts the cuts a few metres farther on. Done
  their way, the segments' places and terms match the reference closer, and the
  levels change this little.

- **The flight profiles** of Part 2 (tables D-1 and D-2, which the package holds): each
  case's fixed-point profile from its procedural steps by Volume 2 Appendix B, nine
  arrivals and seventeen departures, 314 points, against Volume 3's, which round to
  0.1 ft, kt and lb.

What is not checked: the maximum levels (Volume 3 gives none), the noise of Part 2's
profiles (Volume 3 gives none either), bank angles in turns (its cases fly them level)
and the idle steps of an approach away from sea level in the standard atmosphere
(Appendix B7.1.2 and B7.1.4 adjust them; all of Volume 3's arrivals are at sea level in
it).
"""

from __future__ import annotations

import argparse
import collections
import math
import re
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from acoustic_spaces.doc29 import path as doc29_path
from acoustic_spaces.doc29 import volume3

FOOT_M = 0.3048
#: The columns of the segment results compared, with the attribute of
#: :class:`acoustic_spaces.doc29.SegmentLevels` and the unit's scale.
TERMS = {
    "slant_distance(ft)": ("slant", 1 / FOOT_M),
    "distance_d1(ft)": ("d1", 1 / FOOT_M),
    "distance_d2(ft)": ("d2", 1 / FOOT_M),
    "distance_q(ft)": ("q", 1 / FOOT_M),
    "lateral_displacement(ft)": ("lateral", 1 / FOOT_M),
    "NPD_interpolation_thrust(lb/e)": ("power", 1.0),
    "angle_beta(°)": ("elevation", 1.0),
    "engine_Install_correction(dB)": ("installation", 1.0),
    "lateral_attenuation(dB)": ("lateral_attenuation", 1.0),
    "baseline_SEL(dB)": ("baseline", 1.0),
    "speed_corr(dB)": ("duration", 1.0),
    "noise_fraction": ("fraction", 1.0),
    "start_of_roll_correction(dB)": ("start_of_roll", 1.0),
    "segment_SEL(dB)": ("exposure", 1.0),
}
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def sheet(workbook: Path, name: str) -> list[dict[str, str]]:
    """The rows of the worksheet ``name`` of ``workbook``, by its header row."""
    with zipfile.ZipFile(workbook) as z:
        shared = [
            "".join(t.text or "" for t in si.iter(f"{NS}t"))
            for si in ElementTree.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{NS}si")
        ]
        book = ElementTree.fromstring(z.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        found = next(s for s in book.iter(f"{NS}sheet") if s.get("name") == name)
        target = str(targets[found.get(f"{REL}id")]).lstrip("/")
        root = ElementTree.fromstring(
            z.read(target if target.startswith("xl/") else f"xl/{target}")
        )
    rows = []
    for row in root.iter(f"{NS}row"):
        cells = {}
        for c in row.iter(f"{NS}c"):
            value = c.find(f"{NS}v")
            if value is None or value.text is None:
                continue
            letters = re.match(r"[A-Z]+", str(c.get("r"))).group(0)  # type: ignore[union-attr]
            column = 0
            for ch in letters:
                column = column * 26 + ord(ch) - 64
            text = shared[int(value.text)] if c.get("t") == "s" else value.text
            cells[column - 1] = text.strip()
        rows.append([cells.get(i, "") for i in range(max(cells) + 1)] if cells else [])
    header = rows[0]
    return [dict(zip(header, r, strict=False)) for r in rows[1:] if r]


def receptor_levels() -> float:
    """Table B-1: each case's levels at its receptors against Volume 3's."""
    reference = volume3.sel_results()
    receptors = volume3.receptors()
    print("Sound exposure levels at the receptors against table B-1 (dB):")
    worst = 0.0
    for case in volume3.CASES:
        names = [r for (c, r) in reference if c == case.name]
        levels = case.event(np.array([receptors[r] for r in names])).sel
        diffs = levels - np.array([reference[(case.name, r)] for r in names])
        worst = max(worst, float(np.max(np.abs(diffs))))
        print(
            f"  {case.name}: {len(names):2d} receptors, largest difference "
            f"{float(np.max(np.abs(diffs))):.4f}"
        )
    print(f"  all {len(reference)}: within {worst:.4f} dB (the table gives 0.01 dB)")
    return worst


def profiles() -> float:
    """Tables D-1 and D-2: each case's fixed-point profile against Volume 3's."""
    reference = volume3.reference_profiles()
    print("Flight profiles from the procedures against tables D-1 and D-2 (ft, ft, kt, lb):")
    worst = np.zeros(4)
    for case in volume3.profile_cases():
        mine = case.points()
        ref = reference[case.number]
        if len(mine) != len(ref):
            print(f"  {case.number}: {len(mine)} points, Volume 3 {len(ref)}")
            worst[:] = np.inf
            continue
        d = np.abs(
            [
                (
                    a.distance_ft - b.distance_ft,
                    a.height_ft - b.height_ft,
                    a.speed_kt - b.speed_kt,
                    a.thrust_lb - b.thrust_lb,
                )
                for a, b in zip(mine, ref, strict=True)
            ]
        ).max(axis=0)
        worst = np.maximum(worst, d)
        print(
            f"  {case.number:>3} {case.aircraft} {case.procedure:<27} {len(ref):2d} points,"
            f" largest differences {d[0]:.2f} {d[1]:.2f} {d[2]:.2f} {d[3]:.2f}"
        )
    count = sum(len(p) for p in reference.values())
    print(
        f"  all {count} points: distance within {worst[0]:.2f} ft, height {worst[1]:.2f} ft,"
        f" speed {worst[2]:.2f} kt, thrust {worst[3]:.2f} lb (the tables give 0.1)"
    )
    return float(worst.max())


def segment_terms(rows: list[dict[str, str]]) -> None:
    """Worksheet B-2: each term of each segment against Volume 3's."""
    by_case: dict[tuple[str, str], list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_case[(row["case_ID"], row["receptor_ID"])].append(row)
    receptors = volume3.receptors()
    worst = dict.fromkeys(TERMS, 0.0)
    place = 0.0
    counts = []
    for case in volume3.CASES:
        segments = case.segments()
        mine = {r: case.event(np.array([receptors[r]])) for (c, r) in by_case if c == case.name}
        counts.append(
            f"{case.name} {len(segments)}"
            + ("" if all(len(by_case[(case.name, r)]) == len(segments) for r in mine) else " (!)")
        )
        for receptor, levels in mine.items():
            for ref, segment, part in zip(
                by_case[(case.name, receptor)], segments, levels.segments, strict=True
            ):
                start = [float(ref[f"segment_start_{a}(ft)"]) * FOOT_M for a in "xyz"]
                place = max(place, float(np.max(np.abs(np.subtract(segment.start, start)))))
                for column, (attribute, scale) in TERMS.items():
                    value = float(np.asarray(getattr(part, attribute))[0]) * scale
                    worst[column] = max(worst[column], abs(value - float(ref[column])))
    print("  segments:", ", ".join(counts))
    print(f"  largest difference of a segment's start: {place:.2f} m")
    for column, value in worst.items():
        print(f"  {column:<34}{value:10.4f}")


def grids(rows: list[dict[str, str]]) -> None:
    """Worksheet B-3: each case's grid against Volume 3's."""
    by_case: dict[str, list[tuple[float, float, float]]] = collections.defaultdict(list)
    for row in rows:
        by_case[row["case_ID"]].append(
            (float(row["grid_x(m)"]), float(row["grid_y(m)"]), float(row["SEL(dB)"]))
        )
    for case in volume3.CASES:
        g = np.array(by_case[case.name])
        points = np.column_stack([g[:, 0], g[:, 1], np.zeros(len(g))])
        d = case.event(points).sel - g[:, 2]
        print(
            f"  {case.name}: {len(g):5d} points, rms {math.sqrt(float(np.mean(d**2))):.4f}, "
            f"largest {float(np.max(np.abs(d))):.4f}"
        )


@contextmanager
def their_speed_cuts() -> Iterator[None]:
    """Cut segments by speed as Volume 3's segment results do: the lengths of eq. 3-3i,
    along the segment, laid out along the ground track."""
    original = doc29_path._by_speed

    def cuts(a: doc29_path.Node, b: doc29_path.Node, ground: bool) -> list[doc29_path.Node]:
        n = int(1 + abs(b.speed - a.speed) / doc29_path.SPEED_STEP_M_S)
        if n < 2:
            return []
        across = b.s - a.s
        length = math.hypot(across, b.height - a.height)
        dv, dp = (b.speed - a.speed) / n, (b.power - a.power) / n
        found, s = [], a.s
        for k in range(1, n):
            s += (a.speed + dv * (k - 0.5)) * 2.0 * length / ((a.speed + b.speed) * n)
            f = (s - a.s) / across
            found.append(
                doc29_path.Node(
                    s=s,
                    height=a.height + f * (b.height - a.height),
                    speed=a.speed + k * dv,
                    power=a.power + k * dp,
                    bank=a.bank + f * (b.bank - a.bank),
                    ground=ground,
                )
            )
        return found

    doc29_path._by_speed = cuts
    try:
        yield
    finally:
        doc29_path._by_speed = original


def all_levels() -> dict[str, np.ndarray]:
    reference = volume3.sel_results()
    receptors = volume3.receptors()
    return {
        case.name: case.event(
            np.array([receptors[r] for (c, r) in reference if c == case.name])
        ).sel
        for case in volume3.CASES
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", type=Path, help="Volume 3 Part 1's workbook (xlsx)")
    args = parser.parse_args()
    receptor_levels()
    print()
    profiles()
    if args.workbook is not None:
        segments = sheet(args.workbook, "B-2_Segment_Results")
        print(f"\nEvery segment against worksheet B-2 ({len(segments)} rows):")
        segment_terms(segments)
        print("\nThe grids against worksheet B-3:")
        grids(sheet(args.workbook, "B-3_Grid_Results"))
        print("\nWith Volume 3's own placement of the cuts by speed:")
        before = all_levels()
        with their_speed_cuts():
            segment_terms(segments)
            after = all_levels()
        change = max(float(np.max(np.abs(after[c] - before[c]))) for c in before)
        print(f"  the levels at the receptors change by at most {change:.4f} dB")
    else:
        print("\n(--workbook: Volume 3 Part 1's workbook, for every segment and the grids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
