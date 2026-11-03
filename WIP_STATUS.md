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

- Step 6 coded: `validation/barrier_wind.py` ports the full-wave mode matching (Evans 1983) and the started
  PE (from the exact four-path field behind a rigid screen), plus the engine end to end against the full-wave
  effect, with an acceptance table. The full-wave solution reproduces design A's numbers to 0.01 dB (near
  screen, rigid ground, 125 Hz); the engine's error there is 0.03-0.08 dB (A: 0.06-0.09). Full runs going.

## Open

V3 runs (full-wave cases, started PE 125 Hz-8 kHz, then the engine end to end); V1 (still air bit-identical:
edge diffraction JSON identical so far, others running); V2; V4 convergence; V5 continuity; V6 map time;
docs and CHANGELOG. Found while measuring V5: in still air main's level steps by up to 4.2 dB (63 Hz) where a
listener crosses the shadow boundary of a barrier's end, because the paths over the top vanish there.

## Next

Collect V3, run V4-V6, write the docs.
