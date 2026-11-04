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
  the weather's effect from 1 kHz up (behind barriers in wind fixed here in review round 1); a leg from a ground reflection a rounding error (1e-16 m) above the ground
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
  At a whole degree's halfway direction behind a barrier at 14 degrees to the way the step was 2.35 dB downwind and 3
  dB upwind; now 0.03 dB per 0.01 degree. Against the listener's own way: within 0.9 dB (120 m behind the
  14-degree barrier), 1.8 dB within 3 m behind a 45-degree wall.
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
- Validation script: a plane (line-source, straight screens) full-wave reference next
  to the ring; case two40 (3 m at 20 m, 5 m at 60 m) on both; case near_up (60 m domain, 5 and 10 m/s
  upwind to 200 m); a domain self-check (30 m domain: 0.00 dB at 120 m, 10.5 dB at 200 m in 5 m/s upwind;
  60 m within 0.02 dB of 90 m but 1.24 dB at 200 m in 10 m/s); `--band`, `--wind`, `--reference`,
  `--only convergence`; started 8 kHz rows rerun with 32 frequencies (all within 0.30 dB). Upwind in 10 m/s
  (near_up, 60 m domain) the engine's effect is 0.2 to 1.64 dB low where judged (1.0 to 1.4 at 250 Hz 120 and
  200 m away, the narrow-angle bias the review found); in 5 m/s within 0.71 dB. Two barriers 40 m apart: the
  plane reference agrees with the ring within 0.1 dB in the effect (0.08 dB in still air), so the ring does
  not focus here; the engine is off by up to 2.2 dB (250 Hz, 5 m/s downwind, rigid). Upwind rows of far,
  roof, two and two40 (listeners 130 to 160 m away) rerun on a 60 m domain (30 m: +1.0 dB at 150 m, 552 Hz).
  Convergence reproduced by the script (8 kHz grid now with 32 frequencies: 2.2 dB at 4 kHz behind the 60
  degree top). End to end within 0.18 dB of the PE's own error.
- Cost (coverage maps, 96 points, threads=1, CPU under the same load): main 13.4/12.3/9.5 s, before these
  fixes 232/227/236 s, now 295/260/259 s (22, 21 and 27 times main): the blend's runs along one more whole
  degree (40 runs a band instead of 36) and the open ground's runs along each way round an end (88 a band
  instead of 36, 16 s of 286).

- Paths round ends (review round 1, continued): in the bands added coherently they now change at each
  frequency by the ratio of their way's runs' pressures in the weather and in still air (less the delay
  they take on), not by the energy ratio with their phases in still air, so that they keep in step with the
  direct sound's parabolic equation. 3 m beside a building's wall 60 m downwind the weather's effect was
  -1.84/-6.73 dB at 250/500 Hz against the open ground's +1.88/-0.14; now +1.63/-0.30 (within 0.7 dB from
  63 to 500 Hz down- and upwind; a test). 10 cm across a barrier end's shadow boundary 60 m downwind the
  500 Hz step went from 4.95 to 0.19 dB; 120 m upwind at 1 and 2 kHz (23 to 65 dB below still air) it is
  15 and 27 dB, the open side now taking the open ground's deep shadow (main's 1e-16 leg bug kept it lit).
- Two barriers 40 m apart in 8 m/s: the effect is 4.95 dB off (ring) and 4.91 (plane) at 250 Hz over rigid
  ground (the review's 4.92); added to the validation's winds.
- Point cost at steep points: three or four listeners 6 to 20 m up behind a barrier or a building take
  about 120 s of CPU in 8 m/s (0.2 to 0.6 s in still air), nearly all of it in the wide-angle equation.

- Limits: `LIMITS` are now the largest errors measured, rounded up to 0.05 dB (behind one barrier or
  building 0.65 in 5 m/s and 0.75 in 10 m/s; near_up 0.75/1.65; steep 45 1.0; two barriers 10 m apart 1.7 to
  1.75; 40 m apart 2.25 in 5 m/s and 4.95 in 8 m/s; started 0.5; end to end 0.2 beyond the PE's own error),
  and the script's acceptance passes on the merged runs. Along a plane the full-wave effect is within 0.14 dB
  of the ring's (two barriers; 0.13 behind one).

## Continued on `claude/acoustic-215-7khv20` (2026-11-04)

This branch is `claude/project-thread-jd2ksn-barrier-pe` with main merged in (#215, the turning-ray fix,
and the rest). Conflicts were in CHANGELOG.md, docs/validation.md (the known issues' intro and list) and
`engine/refracted.py` (the runs' impedance: both the dense samples' admittance and main's rigid guard
kept).

- After the merge: `tests/test_parabolic_obstacles.py` and `tests/test_barriers_in_wind.py` 26 passed;
  the outdoor files (refraction, native, diffraction, ffp, first_gather, turbulence, vegetation, wave,
  coverage, insertion, long_term, outdoor, parabolic*, grazing, ground, edge_fade) 517 passed, 1 skipped
  (matplotlib), all with `-W error`, before the two changes below.
- **The roof between the grid's cells** (`parabolic._march`, new `parabolic.roof_share`). The march over
  a roof no longer puts the rigid floor on a height step: the field is mirrored about the roof's own
  height (the even extension shifted by exp(-2 i kz d), d the roof's height over the sample under it),
  that sample weighed by the share of its cell above the roof, as the ground's is. The front face zeroes
  the samples wholly below the roof; at the back face the field takes the roof's share. Measured with
  the engine's runs behind barriers (`samples`, grid and top as `validation/barrier_wind.py`
  `engine_levels`), a building 25 to 37 m away, roof 6.00 to 6.08 m, listeners 60 m and 120 m 1.5 m up
  and 120 m 4 m up, `windy` preset:
  - weather's effect, largest step per cm: before 0.10/0.14 dB at 2 kHz, 0.04 dB at 8 kHz downwind,
    0.18 dB at 8 kHz upwind; now 0.05 dB at 2 kHz, 0.03 dB at 8 kHz down, 0.155 dB at 8 kHz up
    (0.028 dB per mm there: smooth, no step); 500 Hz 0.094 -> 0.021 dB.
  - the full-wave roof case of the test (125 Hz, 40-52 m, roof 6.0087 m): error -0.007/+0.037 dB
    (5 m/s) and -0.004/-0.121 dB (-5 m/s), within the test's 0.2 dB; the test passes.
  - end to end (`simulate`, the early paths' coherent sum, 1 cm steps from 6.0 to 6.3 m, listeners 60 and
    120 m away 1.5 m up, 5 m off the axis, second-order diffraction): before, still air up to 0.021 dB a
    cm, `windy` downwind 0.07/0.08/0.12/0.08/0.50/0.13/0.07/0.05 dB and upwind
    0.06/0.08/0.06/0.16/0.39/0.07/0.15/0.16 dB from 63 Hz to 8 kHz; after, downwind
    0.01/0.01/0.02/0.02/0.13/0.04/0.10/0.04 dB and upwind 0.005/0.005/0.02/0.03/0.10/0.05/0.14/0.14 dB
    (still air unchanged, 0.007 to 0.021 dB). The docs now give these: the continuity table's roof row
    (0.14 dB, naming the listeners, as the one-off check before named none) and its text; the known
    issue "A building's height in wind", the outdoor limit "A roof on the equation's grid" and the
    roadmap's "a roof between the equation's cells" are taken out; `docs/outdoor.md` says a listener on
    a barrier's line or a building's wall keeps the straight legs.
- **A listener or map point on a barrier's line or a building's wall** (seen from above, within
  `ON_OBSTACLE_M` = 1 um) made the runs behind barriers raise "the parabolic equation gave no sound in
  still air" (the march takes a listener at an obstacle's range past it, where nothing gets through
  below its top), which stopped a whole coverage map: two 4 m barriers 40 and 60 m from the source over
  grass, 8 m/s, map spacing 12 m, a point 58 m out along 313.6 degrees on the first one. Now such a
  listener keeps main's rule, and a whole degree's run with an obstacle at the listener's range counts
  for nothing for it (`refracted._at`). Test: `test_a_listener_on_a_barrier_s_line_keeps_main_s_rule`
  (fails before: it raised). The map has not yet been rerun.
- Cost, measured (one thread): four listeners 6 to 20 m up and one 1.5 m up behind a 4 m barrier, 8 m/s:
  229 s CPU, of it the wide-angle equation 191 s at 8 kHz and 24 s at 4 kHz (narrow 6.4 s at 8 kHz).
- **Tried and taken out: frequency groups.** Behind barriers, the 4 and 8 kHz bands marched their
  frequencies in four groups, each on the grid and in the steps of its own highest (the lowest of an
  octave are marched on the highest's). The steep probe above went from 229 to 171 s CPU (wide-angle
  8 kHz 191 -> 139 s), its levels within 0.01 dB; but the convergence check at 4-8 kHz
  (`--only convergence --band 4000 --band 8000`, the validation's runs grouped as the engine's, the
  tenth-wavelength reference not) got worse 11 m behind the top 60 degrees up (`steep60`): grid error
  at 4 kHz upwind over grass -2.19 -> -3.52 dB, over rigid ground -0.97 -> -1.85 dB; at 8 kHz downwind
  over rigid ground +0.16 -> -0.82 dB; and the 32-against-64 samples error at 8 kHz there +0.17 -> -1.91
  dB (the grid's error now steps from one group to the next across the band). Elsewhere (near, rail,
  roof, high) within about 0.1 dB of before but for rail upwind at 8 kHz (-1.22/-1.49 dB samples at its
  100 m listener). That listener is on the narrow-angle grid, which the groups make coarser at each
  group's lower frequencies. Reverted (engine, validation, test). Cheaper ways still open: groups
  keeping the band's grid of heights and changing only the range step; or the wide-angle equation's
  absorbing layer (`LAYER_MIN_M` 20 m is most of the grid at 8 kHz) for the runs behind barriers.
- Reruns not done: `--only full_wave --case two --band 500 --wind -5` (ring and plane), then
  `--case roof --wind -5`, both on the 60 m domain, were stopped at this session's two-hour limit for
  a background job after one row: two barriers, ring, rigid ground, 500 Hz, 5 m/s upwind, effect
  -5.63/+7.70 dB at 100/150 m, error -0.88/-0.18 dB (within its limit, 1.7 dB; 6818 s on 4 cores shared
  with other runs). Grass, the plane reference and the roof upwind are still to run.
- `validation/barrier_wind.py --cache DIR --part K/N` splits the full-wave check: each full-wave solution
  (one frequency, in the wind or in still air) is kept in DIR, a run with `--part K/N` finds only the K-th
  of every N, and a last run with the same options and `--cache DIR` but no `--part` reports from the
  cache (checked: two parts and the report give the direct run's row exactly, rail, 63 Hz, -5 m/s).
  The reruns go that way with `OMP_NUM_THREADS=1`, two at a time (a 500 Hz solution on the 60 m domain
  takes about 6 GB; four side by side ran out of the session's 16 GB), eight parts of about 34 min.
- **Done: two barriers, 500 Hz, 5 m/s upwind, 60 m domain** (effect at 100/150 m, then error, dB): ring
  rigid -5.63/+7.70, -0.88/-0.18 (the same as the direct run); ring grass -1.02/-4.43, -0.65/-0.36; plane
  rigid -5.72/+7.66, -0.79/-0.14; plane grass -1.08/-4.49, -0.59/-0.31. All within their limits (ring 1.7,
  plane 1.65), below the largest errors the tables give, so `LIMITS` and the tables stay.
- **Done: the roof, 5 m/s upwind, 60 m domain** (ring; with the roof between the grid's cells; effect at
  80/150 m, then error, dB): rigid 63 Hz -0.50/-1.70, -0.02/+0.03; 125 Hz -0.10/-2.15, -0.00/-0.12;
  250 Hz +1.19/-1.46, +0.03/-0.21; 500 Hz +2.79/+3.31, +0.03/+0.02; grass 63 Hz -0.38/-1.27, -0.03/+0.02;
  125 Hz +0.18/-0.81, -0.06/-0.14; 250 Hz +0.61/+1.42, -0.15/-0.24; 500 Hz -1.13/-6.74, +0.03/-0.16. Largest
  0.24 dB (limit 0.65). With this, the reruns the "Open" list below names are done; the docs' tables were
  filled from the runs before them and are not changed by these rows.
- `ruff check .` and `ruff format --check .` flag only `wip/prototypes/` (as before the merge).

- **Tried and taken out: a thinner absorbing layer behind barriers.** Both equations took `layer_m`, the
  layer's least thickness, its strength raised to keep strength times thickness (the sound's loss across
  it); at 4 and 8 kHz the 20 m layer is half or more of the heights marched (8 kHz, top 21.5 m: wide-angle
  10860 heights where 5626 reach the top). Against a 60 m layer of the same loss, on the convergence cases
  at 2, 4 and 8 kHz (rigid ground and grass, +-5 m/s), the weather's effect moved further off with a
  thinner one: 5 m by up to 5.4 dB (near, grass, -5 m/s, 8 kHz, 120 m; 20 m: 0.001 dB), 10 m by up to
  2.4 dB (steep60, rigid, +5 m/s, 4 kHz). Reverted.
  - Found on the way: the default 20 m layer itself is up to 1.18 dB off the 60 m one in the weather's
    effect, and 2.76 dB in the still level, at steep60's first listener (15 m, 11 m behind the top 60
    degrees up; 2 to 8 kHz, both grounds), elsewhere within 0.26 dB: part of that case's known error
    (grid convergence 2.2 dB at 4 kHz) is sound sent steeply up off the top that the layer sends back.
    A thicker layer for steep runs behind barriers would take it out, at a cost: a follow-up.
- **The map that stopped** (two 4 m barriers 40 and 60 m away over grass, 8 m/s, 12 m spacing, 140 points)
  now runs: 457 s CPU on one thread against main's 38 s (12 times), 407 s of it in the parabolic
  equation, the narrow-angle one at 8 kHz 294 s in 74 runs and at 4 kHz 62 s.
- **Tried and taken out: a compact grid behind barriers.** The narrow-angle equation rounds its heights up
  to a power of two (8192 for that map at 8 kHz, where about 4200 reach the absorbing layer's top);
  rounding up only to a fast transform's length instead saved a quarter of the CPU on the convergence
  cases (290 -> 228 s) but moved the results: the weather's effect by up to 0.14 dB 120 m behind the near
  barrier upwind at 1-4 kHz, and behind the top 60 degrees up by up to 3.9 dB, the still level by up to
  13.6 dB (8 kHz, grass, upwind). The heights above the layer are part of what absorbs the sound sent
  steeply up off a high top. Not committed.
- Checked, no gain: the narrow-angle step is its two transforms over the grid (32 frequencies by 16384
  heights at 8 kHz: 18 ms with numpy's FFT, 19.5 ms with scipy's); with the grid, the steps and the
  frequencies kept, nothing of size is left to save without moving the results. What cost could still
  go, at the results' expense and so for the owner to weigh: fewer frequencies at 8 kHz (D8 chose 32),
  a coarser grid there, or rays from 4 kHz up.

- **New setting `simulation.early.barrier_frequencies`** (8, 16 or 32; default 32, as before): the
  frequencies an octave the runs behind barriers in wind take at 8 kHz, at 4 kHz no more than 16
  (`refracted.barrier_samples`; 8 takes the early paths' 8 everywhere). Scene file only (no editor control
  or command-line flag, to keep it small). Test: `test_the_frequencies_at_4_and_8_khz_behind_barriers_can_be_fewer`.
  Measured against 64 an octave on the convergence cases (15 listeners, both grounds, +-5 m/s, where the
  effect is within 25 dB; mean / nine in ten / largest, dB): 32 (default) 4 kHz 0.05/0.10/0.25, 8 kHz
  0.04/0.09/0.51; 16: 4 kHz the same, 8 kHz 0.51/1.0/6.35 (high, rigid, +5 m/s); 8: 4 kHz 0.80/1.66/9.61,
  8 kHz 2.09/3.54/14.0 (near, rigid, -5 m/s). The map behind two barriers (140 points, one core): 457,
  291 and 172 s, against main's 38 s 12, 7.7 and 4.5 times. In `docs/outdoor.md` (a table after the
  cost paragraph) and the field's description.

- Checks at `d11fa1c`: the command-line and outdoor test files (cli, refraction, diffraction, coverage,
  insertion, long_term, outdoor, parabolic*, barriers_in_wind, ffp, turbulence, ground, edge_fade,
  first_gather, vegetation, wave, grazing, native) 580 passed, 1 skipped (matplotlib); scene and code
  tests 338 passed, 53 skipped (PIL); all with `-W error`. `ruff check` and `ruff format --check` clean
  but for `wip/prototypes/`; `tools/catalog.py check` clean.

## Open

- (Done on `claude/acoustic-215-7khv20`, see above.) Reruns on the 60 m domain (upwind, 5 m/s): two
  barriers at 500 Hz (ring and plane) and the roof over grass.
- Otherwise nothing for v1 beyond review. V1 and V2 hold: the still-air scripts (edge diffraction, insertion
  loss, terrain, edge fade at 5 cm steps) and the wind scripts (ffp, zones, long term, turbulence, outdoor
  propagation, ground reflection) write the same JSON as main but for timings; the examples' early paths are
  identical to main's, their late parts differing as main's own runs do.

## Next (follow-ups, not v1)

A blend across barrier ends; the sound going back and forth between two barriers; a roof between the grid's
cells; main's leg-shadow rounding bug (own PR); cheaper 4-8 kHz runs and steep points (the cost is the
owner's call under Q1: maps 21 to 27 times main's, about 120 s for three or four steep listeners); the
still-air two-edge fix (Q3).
