# WIP: barriers in wind in the parabolic equation (work in progress)

Replaces draft PR #180's ray method. The owner chose on 2026-11-03 to compute barriers in wind with the
engine's parabolic equation (the barrier inside the PE), with rays only where the design says.

## State (2026-11-03, 22:00 Madrid)

- Design done: `wip/barrier-pe-design.md`. In short: behind a barrier or building in wind, each path in the
  source-listener plane keeps main's still-air level and is multiplied per band by G = |p_wind|^2 / |p_still|^2
  from two PE runs with the obstacle inside (zero-field screens; rigid roof floors for buildings). Measured
  against a full-wave reference: within 0.57 dB at 5 m/s and 0.74 dB at 10 m/s (63-500 Hz); against a PE started
  from the exact rigid-screen field: -0.59 to +0.32 dB (125 Hz-2 kHz). Main is off by -12.7 to +17.5 dB there.
  Still air and rooms stay bit-identical.
- Prototypes the design builds on: `wip/prototypes/` (screen patch for both PE solvers, rigid-roof prototype,
  full-wave mode-matching reference, started-PE reference).
- Owner questions open: PE in all eight bands for maps (recommended) or capped at 2 kHz; the barrier's own wind
  field as a documented limit in v1 (recommended); the still-air two-edge fix after v1 (recommended).
- No engine code yet.

## Next

The one-day plan in section 12 of the design: screens in the PE, roof floor, barrier cut, requests and runs,
placement, validation script, docs. Then independent review rounds, then a PR that replaces #180. Delete
`WIP_STATUS.md` and `wip/` before that PR.

The ray method after five rounds is kept, not for merging, on `claude/project-thread-jd2ksn-barrier-rays-wip`.
