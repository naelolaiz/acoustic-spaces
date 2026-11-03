# WIP: barriers in wind in the parabolic equation (work in progress)

Replaces draft PR #180's ray method. The owner chose on 2026-11-03 to compute barriers in wind with the
engine's parabolic equation (the barrier inside the PE), with rays only for the top bands.

## State (2026-11-03, evening)

- Design stage: the engine's PE and the barrier code are being mapped, reference methods surveyed
  (Salomons' screens in the GFPE, hybrid BEM-PE, Harmonoise, Nord2000, CNOSSOS-EU/ISO 9613-2), and two
  designs merged into one. No engine code yet; this branch is main plus this note.

## Next

1. Design document with the band crossover, continuity rules, run-time budget and validation plan.
2. First version: the screen inside the PE for the bands the PE covers, rays above, checked against
   independent PE probes and published cases.
3. Review rounds, then a PR that replaces #180.

The ray method after five rounds is kept, not for merging, on `claude/project-thread-jd2ksn-barrier-rays-wip`.
