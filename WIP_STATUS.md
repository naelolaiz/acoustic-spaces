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

- Changed from the design (D7): the paths round ends keep their phases in still air in the open ground's
  bands, delayed by one shared shift (the first of them's), times the open ground's effect; their legs' own
  times and shadows apply only above those bands. With each leg's own time kept there (the design), the
  weather changed how the pair round an end (off the ground and not) adds up a second time, on top of the
  open ground's effect, which already holds it: at a barrier end's shadow boundary 120 m downwind the level
  stepped by 13.7 dB at 500 Hz per 0.1 m; now 5.8 dB (main: 16.8; still air: 0.8).
- Barrier-end continuity (V5) misses its 1 dB target: per 0.1 m across an end's shadow boundary, 60 and 120 m
  behind a 4 m barrier, up to 5.8 dB downwind and 24 dB upwind (2 kHz, deep in the shadow) in the open
  ground's bands, and up to 21 dB above them, where the paths round the end keep main's straight legs (main:
  up to 69 dB). Still air (main's, unchanged): up to 4.5 dB at 63 Hz, where the paths over the top end with
  the barrier.

- Measured so far (V4-V6): a quarter-wavelength grid from 1 kHz is within 0.3 dB of a tenth's at 1-2 kHz
  and 0.2 dB at 4-8 kHz, but 2.7 dB at 4-8 kHz 11 m behind a top 60 degrees up; 16 samples an octave at 4-8 kHz
  are off by up to 1.8 dB against 32 over rigid ground (6.4 dB where the ground's fringes fall in step with the
  frequencies). Continuity: a breath of wind 0.024 dB at 0.01 m/s (linear in the wind); top height 0.007 dB/mm
  up to 4 kHz, 0.09 dB/mm at 8 kHz (a smooth slope); line of sight 0.077 dB/cm (still air 0.070); a degree's
  boundary behind a slanted barrier up to 0.7 dB; no step at the parabolic equation's minimum range. Coverage
  maps behind two barriers or a building take 10 to 16 times as long as on main (target 2).
- Found on main, left for their own PRs: coverage maps drop early paths that arrive before the straight
  line's time where no path is clear (bands added in energy), so a map behind barriers in wind misses most of
  the weather's effect from 1 kHz up; a leg from a ground reflection a rounding error (1e-16 m) above the ground
  gets no refraction shadow (shadow_start returns 1e284 m), which steps the level by up to 65 dB at 8 kHz.

- Changed from the design (D8): the 8 kHz band behind barriers takes 32 frequencies an octave (4 kHz keeps 16).
  With 16 the weather's effect was off by up to 1.8 dB over rigid ground and 6.4 dB where the ground's fringes
  fell in step with the frequencies, and stepped smoothly by 0.09 dB/mm as a barrier's top rose (aliasing: 32 and
  64 agree there within 0.05 dB); 32 are within 0.18 dB of 64 but 0.51 dB behind a top 60 degrees up. It costs
  about 1.4 times more per map.
- V3 done: full wave within 0.62 dB in 5 m/s (limit 0.6) and 0.74 in 10 m/s behind one barrier or building, 0.98
  behind a top 45 degrees up (limit 0.6), 1.68/1.63 behind two barriers (limits 1.7 at 125 Hz / 1.0); started PE
  within 0.47 dB (limit 0.6); the engine end to end within 0.08 dB of its PE's error (0.18 at a 60-degree top),
  where main's straight legs were 1 to 33 dB off.
- V5 roof: as a roof rises 1 cm the level steps by up to 0.95 dB from 2 kHz up (the roof sits on the PE's grid;
  still air 0.02): the design's guess that the ratio cancels the snap is wrong.

- V5/V6 with 32 at 8 kHz: barrier top 0.020 dB/mm (aim 0.01); line of sight 0.077 dB/cm (still 0.070, aim
  0.05); a degree's boundary 0.66 dB; roof 1.16 dB/cm (aim 0.1 a cell); barrier end 9.7 dB at 60 m and 24 dB at
  120 m per 10 cm (aim 1; still air 4.5). Maps behind barriers take 15 to 22 times main's CPU (aim 2), and from
  1 kHz up show none of the effect until main's coverage onset bug is fixed.
- Docs, CHANGELOG and roadmap written; tests: barrier and neighbouring outdoor files pass (184).

## Review round 1 fixes (in progress)

Two reviews rejected v1. Fixes so far (each with a test that fails before it):
- The cut behind barriers: runs along whole degrees, the nearest alone over the middle half of the step
  between two, both weighed in dB over the half round the halfway direction; a direction counts for nothing
  where more or fewer obstacles stand short of the listener than on its own way or the listener would be
  under a roof, and less where the last one stands nearer than half as far as on its own way; the
  listener's own way takes the weight lost. A zero still-air energy raises. Behind a 45-degree wall 0.7 m
  behind it the level stepped by 25 to 72 dB within 1.5 m; now within the still air's own changes (2 dB).
  At a whole degree's halfway direction behind a barrier at 18 degrees the step was 2.35 dB downwind and 3
  dB upwind; now 0.03 dB per 0.01 degree. Against the listener's own way: within 0.9 dB (120 m behind the
  18-degree barrier), 1.8 dB within 3 m behind a 45-degree wall.
- Paths round ends: only those round barriers' ends and buildings' corners alone, off nothing but the
  ground, with a partner off the ground, take the open ground's effect, now along their own way round the
  edges (its length seen from above); others keep their straight legs. Corner rows behind touching buildings
  upwind: +6.8 dB at 1 kHz before, now -0.3 (straight legs +0.1); a crate of the model file: as main again.
- Maps: the direct sound's time behind barriers is the earliest path's where it comes sooner than the straight
  line's (no clear path), so maps keep the paths over a barrier that the wind brings sooner. 130 m behind two
  barriers in 8 m/s of wind the map read 1 to 8 kHz 17.7 to 22.2 dB below the point; now +0.6 to +13.5 dB, the
  same as in still air (+0.6, +1.3, +3.8, +13.5 at 1, 2, 4, 8 kHz: main's own map against point behind
  barriers), and the weather's effect is the same in the map and at the point.
- A line touching a footprint's corner from outside no longer makes a zero-length building.
- A degree's boundary behind a slanted barrier (60 and 120 m): 0.66 dB per 0.01 degree before, now 0.017
  (still air 0.018).
- Validation script (uncommitted runs going): a plane (line-source, straight screens) full-wave reference next
  to the ring; case two40 (3 m at 20 m, 5 m at 60 m) on both; case near_up (60 m domain, 5 and 10 m/s
  upwind to 200 m); a domain self-check (30 m domain: 0.00 dB at 120 m, 10.5 dB at 200 m in 5 m/s upwind;
  60 m within 0.02 dB of 90 m but 1.24 dB at 200 m in 10 m/s); `--band`, `--wind`, `--reference`,
  `--only convergence`; started 8 kHz rows rerun with 32 frequencies. near_up 250 Hz rigid in 10 m/s upwind:
  -0.71, -1.05 dB at 60 and 120 m (the narrow-angle bias the review found).

## Open

Nothing for v1 beyond review. V1 and V2 hold: the still-air scripts (edge diffraction, insertion loss, terrain,
edge fade at 5 cm steps) and the wind scripts (ffp, zones, long term, turbulence, outdoor propagation, ground
reflection) write the same JSON as main but for timings; the examples' early paths are identical to main's, their
late parts differing as main's own runs do.

## Next (follow-ups, not v1)

A blend across barrier ends; a roof between the grid's cells; main's coverage onset and leg-shadow rounding bugs
(own PRs); cheaper 4-8 kHz runs; the still-air two-edge fix (Q3).
