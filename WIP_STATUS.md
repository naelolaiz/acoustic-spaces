# WIP: barriers in wind in the parabolic equation (work in progress)

Replaces draft PR #180's ray method. The owner chose on 2026-11-03 to compute barriers in wind with the
engine's parabolic equation (the barrier inside the PE). Owner answers: the PE runs all eight bands for
points and maps (Q1); the barrier's own wind field is a documented limit in v1 (Q2); the still-air two-edge
fix comes after v1 as its own PR (Q3).

## State

Building v1 from `wip/barrier-pe-design.md`, following its one-day plan (section 12).

- Step 1 done: thin screens and rigid roofs in the narrow-angle PE (`parabolic.march(obstacles=...)`, obstacles
  as (near, far, top)), screens (and buildings as two screens) in the wide-angle PE, `refracting=` to force the
  steep split, `steps_per_wavelength`. Holds: with no obstacles (or one beyond the farthest listener) both
  solvers are bit-identical to main, independent and not; screens equal design B's patch exactly (it differs
  only for a listener exactly at a screen's range in an independent run, which now gets the screened field in
  both modes); screens and roofs equal design A's prototype exactly; roof wind effect within 0.15 dB of the
  full-wave solution (125 Hz); 0.006 dB/mm level and 0.002 dB/mm in the wind effect across grid nodes (500 Hz);
  quarter-wavelength grid within 0.1 dB of a tenth at 1 kHz. Tests: `tests/test_parabolic_obstacles.py`
  (13, all fail on main).
- Step 2 (roof floor) done with step 1.
- Steps 3-5 coded: `Layout.barriers` (each barrier's top line) and `Layout.obstacles_along` (what a line meets,
  seen from above); in `engine/refracted.py` the barrier request (rules a-g of the design), the cut along the
  listener's one-degree direction, runs in the weather and in still air with the cut (shared by profile and
  cut, and by cut and absorbing layer), the open-ground pair for paths round ends, 16 samples at 4-8 kHz and a
  quarter-wavelength grid from 1 kHz; the paths in the plane scale by the weather's effect (plus the eddies'
  share in a refraction shadow) and take one shared delay; paths round ends and over other edges take the
  open ground's effect on both sides of an end (for listeners in the open, from a still-air twin of the open
  ground's run on the same steps). The engine's weather effect equals design A's prototype on the same cut to
  0.01 dB in all eight bands. Tests: `tests/test_barriers_in_wind.py` (5).

## Open

Validation script (`validation/barrier_wind.py`), docs and CHANGELOG; V1-V6. Test suites of the neighbours
(refraction, outdoor, long-term, insertion loss, turbulence, fast field program) running.

## Next

Port the full-wave mode matching and the started PE into `validation/barrier_wind.py` and run them.
