# WIP: turning rays (work in progress, not for merging yet)

This branch fixes the level spikes of curved early paths that turn at, or very near, the source's or the
listener's height (the aircraft landing spike traced on 2026-11-03), and makes the eigenray search in
`engine/refraction.py` find rays near level and near profile corners without accepting false ones.

## State (2026-11-03, evening)

- Committed: `2795571` (the first fix: the sine of a ray at a turning end is exactly 0) and the merge of main `b23f98a`.
- This commit snapshots the uncommitted work of three fix-and-review rounds plus the fourth round in progress:
  closed-form singular parts of the range near level rays (`_singular`), every table row and level-layer edge
  as a corner (`Profile(rows=...)`), the straight direct ray through level air, source folds by the gradient on
  each side, the `_graze` acceptance bound, the ground-row sine in `gpu_tracer.py` and
  `cpp/include/acoustic_spaces/atmosphere.hpp`, tests in `tests/test_refraction.py`, docs.
- Round 3 review REJECTED it for one blocker: a layer whose speed changes only very slightly (table rows
  1e-12 to 1e-6 m/s apart, e.g. a row stored as float32) gave false eigenrays and up to 30 dB level errors,
  because `_graze` and the 1/(p g) singular parts scale as 1/|g|. Round 4 (in progress here) decides "level"
  by what the arithmetic can resolve, bounds `_graze`, falls back to the raw range where a singular part is
  ill-conditioned, and adds regression tests for the float32 row, dw = 1e-8 and a 1-ulp step.
- Minors open from round 3: rays with both ends 1 mm from a level-layer edge at 400 m (0.3 dB), equal-height
  preset rays at 10-80 m (window slide, up to 0.16 dB), the 600 m caustic carve-out and "within 0.74 dB" in
  `docs/outdoor.md`, version history in new test docstrings.

## What it moves (checked by the reviewers)

- `docs/validation.md`: long-term table and C_met fit, the 1995 benchmark's 1 kHz column (within 0.55 dB).
- `docs/outdoor.md`: ISO 9613-2 C_met agreement 0.74 dB; preset accuracy bounds at 150, 300 and 600 m.
- `docs/demos.md`: lakeside garden 4.9 dB.

## To finish

1. A fresh review of the round-4 state must accept it (no false eigenray in any scan family, never worse than
   `b23f98a` by more than 0.05 dB on rays above -20 dB re free field).
2. Delete this file, merge main, run tests/test_refraction.py, test_refracted, test_weather, test_outdoor,
   the GPU tracer reference and native tests, lint and the catalog check; open the PR on main.
