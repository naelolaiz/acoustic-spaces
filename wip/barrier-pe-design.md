# Barriers and buildings in wind with the engine's parabolic equation: design for v1

2026-11-03. Base: main at 3fc4e7d. This merges design A (accuracy first) and design B (minimal change) and checks their claims against the code and the probes. Code paths are under `src/acoustic_spaces/` unless they start with `docs/`, `validation/` or `tests/`.

Where the inputs are:
- **Design A:** `pe-design/design-accuracy/`. Holds `final_design.txt`, `NOTES.md`, the prototype `pe_obst.py` (the engine's narrow-angle solver with screens and rigid roofs) and the full-wave reference `mm_axi.py`.
- **Design B:** `pe-design/design-minimal/`. Holds `NOTES.md`, the solver patch `patch/pe_obstacles.patch` (161 lines, applies to 3fc4e7d) and the started-PE reference `proto/p2_reference.py`.
- **This synthesis:** `pe-design/synthesis/`. Holds `NOTES.md` and the probes S1 to S4 with their outputs.

The prototypes and references named here are kept in `wip/prototypes/`; the probes and their outputs stay with the review notes.

**Status of work in progress.** Nothing is committed yet. The solver patch, the roof prototype, both references and all probes sit in the directories above, ready to go into a work-in-progress branch.

## 1. Decisions

**D1. Weather ratio on the still-air rays (A and B agree).**
- Each in-plane path keeps main's still-air energy, spectrum and delay. In each octave band b, its energy is multiplied by G_b = mean|p_wind|^2 / mean|p_still|^2.
- G_b comes from two runs of the engine's parabolic equation (PE) with the same obstacles, grid, steps, samples and solver. The first run is in the weather; the second has a uniform speed equal to the wind run's speed at the ground, c0.
- Why:
  - Inside the PE a barrier is a zero-field (Kirchhoff) screen, as in Salomons (1994, JASA 95, 3109, "Diffraction by a screen in downwind sound propagation: a parabolic-equation approach"). As an absolute level it is too low against a rigid barrier:
    - 1-2 dB low 40-100 m behind, 3-7 dB low 2-12 m behind (A p1, mm_check1);
    - up to 11 dB low close behind and steeply below the top (B P2).
  - That bias depends on geometry, not weather, so it cancels in G. Against independent references the ratio holds:
    - **Full-wave mode matching**, 63-500 Hz, wind everywhere, rigid ground and grass: within 0.57 dB at 5 m/s and 0.74 dB at 10 m/s (A p2, recomputed in S4).
    - **PE started from the exact rigid-screen field**, 125 Hz-2 kHz, wind behind the screen: within -0.59 to +0.32 dB wherever the wind changes the level by 25 dB or less. B's text says -0.49; S4 finds -0.59 at 500 Hz, 400 m downwind.
  - Main today is off by -12.7 to +17.5 dB in the same cases (S4 on B's P2).
  - The reviewers' probes (`verify-two-edge/vpe.py`, zero field below the top at :87-88) match Fresnel to 0.05 dB. That checks the Kirchhoff screen only, not a rigid one (literature `kirchhoff_check.out`: 0.84-1.6 dB below rigid at 0.7-10 degrees below the edge).
- The absolute level stays the still-air result, which is already checked against exact solutions (`docs/validation.md:996-1097`, `:1943-2050`).

**D2. No starting field behind the barrier.**
- The reference models start the PE behind the barrier from a BEM or GTD field: Premat et al. 2002-03; Harmonoise, Defrance et al. 2007, Acta Acust. 93, 213.
- That starter leaves out the refraction between the source and the barrier. It is 0.4-1.6 dB off with the barrier 15 m out, and 2.6-11.5 dB off with the barrier 100 m out (A p2 "start", S4). Rejected.

**D3. Obstacles inside the march.**
- **Thin screens:** B's patch.
  - New argument `obstacles` in `parabolic.march` (:78) and `wide_angle.march` (:66). It needs a new name because `screen` is the turbulence phase screen (`parabolic.py:97-100`).
  - The march stops exactly at each obstacle range, as it does at listener ranges (`parabolic.py:268-283`, `wide_angle.py:177-191`).
  - There, `psi` is multiplied by `covered(z, top)`, the share of each cell above the top. The result is continuous in the barrier's height: 0.002 dB per mm (B P1).
  - The wide-angle self-starter (:119-131) starts no further out than the first obstacle.
  - With `obstacles=()` both solvers are bit-identical to main (B P7).
- **Buildings in the narrow solver: a rigid roof floor** (A's `pe_obst.py`).
  - The front face zeroes the field below the roof.
  - Over the roof, each step runs on the sub-grid z >= roof, with its own rigid `_Ground` (`parabolic.py:311-362`).
  - At the rear face the field below the roof starts from zero.
  - The ratio error against mode matching is 0.14 dB or less (A p5).
  - Why not B's two screens: in S1 their wind effect drifts from the roof's as the building gets deeper and the wind stronger. The difference is up to 0.28/0.94 dB for a 12 m deep building at 125/500 Hz, 0.46/1.06 dB at 30 m and 1.07/1.37 dB at 60 m (`s1_pe_125.out`, `s1_pe_500.out`).
  - Do not zero the whole block: that gives a pressure-release roof, 5-11 dB low (literature `thick_check.out`).
- **Buildings in the wide solver: two screens in v1** (front and rear face at roof height). It is opt-in, and it also takes the steep points. Its documented error is S1's. The rigid floor there is a follow-up (`_operators` on z[j0:], cache `made` keyed by (j0, dr), `wide_angle.py:133-154, 252-280`).

**D4. The PE gives every band of the in-plane family.**
- No rule extrapolates G above the PE's top band. S2 tried B's rule, min(G_1k, G_2k), on A's 30 cases: it is off by -12.8 to +1.5 dB (B measured -4.3 to +1.4 on its own 6). Holding G_2k is off by -11.4 to +8.0 dB, and main's straight legs (G = 1) by up to -16.2 dB (`s2_toprule.out`).
- So barrier runs cover 63 Hz-8 kHz. The settings:
  - **Grid:** lambda/10 to 500 Hz, as main (`parabolic.py:44`); lambda/4 from 1 kHz. That changes the still level and G by 0.03 dB or less over grass and rigid ground at 2-4 kHz (A p6), and costs 2-5 times less.
  - **Samples:** 8 per band to 2 kHz, as main (`diffraction.py:85`); 16 at 4 and 8 kHz. Against 32 samples, 8 are within 0.11 dB to 2 kHz but off by up to 0.85 dB at 4-8 kHz behind a 6 m screen. 16 are within 0.06 dB in G (S3, S3b).
  - Range step: 5 wavelengths, as main (`parabolic.py:51`).
- Owner question Q1 settles whether maps also run every band.

**D5. Request rule (B's, with two changes).**
A listener gets a barrier request when all of these hold. None of them depends on the weather.
- (a) The air is not still. In still air `refraction_for` returns None (`refracted.py:1069-1076`), so none of this runs.
- (b) Its own straight line, in plan, crosses a barrier or a building footprint strictly between the source and the listener.
- (c) Nothing else blocks the straight 3D line: no object from a model file. Otherwise main's rule applies (`refracted.py:353-357`).
- (d) It is at least `WAVE_MIN_RANGE` (zs + zr) away (`refracted.py:75, 356`).
- (e) `max_order >= 1` (`refracted.py:358`).
- (f) Its in-plane family (D7) is not empty. With `diffraction: none` behind a wall there is nothing to scale, and main's result stays.
- (g) Neither the source nor the listener stands over a building footprint.

What it does not depend on:
- **Whether the top blocks the line of sight.** A barrier below the line of sight is in the PE too, so nothing switches as a top crosses it. A's Fresnel-zone gate is dropped.
- **B's 3.73 (zs + zr) rule.** In its place, both runs take the same route. A new `march(..., refracting=True)` makes the still run hand its steep points to the wide solver, as the wind run does (`parabolic.py:118-119`). Narrow against wide G differs by 0.14 dB or less (B P8).
- **A's fallback to rays above 35 degrees.** The starting field's error at steep angles is a still-air error, the same in both runs (inference). Validation measures the error at 45 and 60 degrees (V3).

**D6. Runs, keys and the cut.**
- **Cut.** Add `Layout.barriers` (name, path, top z) beside `Layout.buildings` (`geometry/outdoor.py:76-78`), filled where the layout is built (:964, :1043). The cut follows the listener's azimuth rounded to `AZIMUTH_STEP_DEG`, as weather zones round it (`refracted.py:383-385`), out to the scene's extent. It gives screens (range, top above the ground) and roofs (entry, exit, roof).
  - A one-way march is not changed by obstacles beyond a range, so every listener in a 1 degree bin shares one cut and one run. That bounds a map to about 360 cuts.
  - Near an end, a listener's own line can cross the barrier while its bin's centre line misses it. The bin's empty cut then makes G the open-ground ratio, a soft change within the bin.
  - B's per-listener cut, with ranges rounded to 0.25 m, could need hundreds of runs behind a long oblique barrier (inference).
- **Wind run key:** (profile key, azimuth when there are ground or weather zones, cut). Sources share runs as `_sharing` allows (`refracted.py:285-301`), since the cut is in the key.
- **Still run key:** (cut, c0, azimuth when there are ground zones). It is shared across wind directions and across the weather classes of a long-term level. Same frequencies, `top`, steps and route as the wind run.
- **Absorbing layer:** `top` is main's rule (`refracted.py:377-381`), raised to at least 1.5 times the highest top in the cut.
- **Open pair for g_open (D7):** the wind run is main's open-ground run for the listener's key; the still run is one homogeneous run per source height and ground. Both cover only main's open-ground bands (`refracted.py:360-363`).
- `_admittance` (`refracted.py:702-717`) takes the sample set as an argument, since barrier runs use 16 samples at 4-8 kHz.

**D7. Placement (`_place_barrier`, beside `_place` at `refracted.py:737-816`).**
- **In-plane family:** rows of the still-air `paths` (`simulate.py:319-345`) of two kinds:
  - specular rows that reflect off the ground only;
  - edge rows whose every edge is not vertical (`Edges.direction`, `diffraction.py:296`) and crosses the plane between source and listener, reflecting off the ground only. This includes two-edge rows, which store the first edge (`diffraction.py:1325`); the second is checked through its diffraction point in `points`.
- **What each in-plane row gets:**
  - In band b, energy times (G_b + s_b). Here s_b is main's eddy share where the plane's refraction shadow starts before the listener, as `_leg` adds it (`refracted.py:681-691`); otherwise s_b = 0. B left this out below 2 kHz; main's open-ground shadow gets it in every band (:755-762).
  - Its delay plus one shared shift: the `_leg` shift of the earliest in-plane row (`refracted.py:663-698`).
  - Its still-air spectrum. All rows scale by sqrt(G) together and their phases relative to each other stay the still-air ones, so their coherent band energy (`coherent.py:590-619`) scales by G exactly (B).
  - These rows skip `_legs`. Main's curved open-ground family (`_open_ground`, `refracted.py:344-402`) is not built for this listener.
- **Rows over vertical edges** (barrier ends, building corners), and other edge rows outside the plane:
  - `_legs` as main, except in the bands where main runs the open-ground PE for this listener (:360-363). There, the leg's gain is 1 and the energy is multiplied by g_open = E_PE(wind, open) / E_PE(still, open). The time shift is kept.
  - This holds **on both sides of an end**, including listeners in the open with an open-ground request.
  - Why both sides: at an end's shadow boundary, UTD's continuity needs the end row and the direct sound to take the same weather factor. For g_open = +9.5 dB the step is about 14 dB with main's legs on both sides and 4.4 dB with g_open on the dark side only. With g_open on both sides it vanishes (my inference from UTD's half amplitude at the boundary).
  - Design B applied g_open on the dark side only, and on top of `_legs`' creeping-wave gain, which counts the upwind shadow twice.
- **Wall reflections** (`_off_walls`, `refracted.py:404-433`): unchanged.

**D8. Still air keeps main's UTD.** Main's still air is within 0.07 dB of the exact half-plane over absorbing ground and 0.15 dB over rigid ground with `both_sides` (`docs/validation.md:1943-2050`). The zero-field PE is 1-7 dB low (D1). Nothing physically requires a change, so none is made.

## 2. File-level changes (v1)

| File | Change |
|---|---|
| `engine/parabolic.py` | `obstacles` (screens and roofs), stops at their ranges, `covered()` (B's patch); a rigid roof floor on the sub-grid with a `_Ground` per (j0, impedance), ported from A's `pe_obst.py`; `refracting=None` to force the steep split; `steps_per_wavelength=STEPS_PER_WAVELENGTH`. The docstring cites Salomons 1994. |
| `engine/wide_angle.py` | `obstacles` (B's patch); roofs as front and rear screens in v1; `steps_per_wavelength`. |
| `geometry/outdoor.py` | `Layout.barriers`; filled at :964 and :1043. |
| `engine/refracted.py` | Constants `BARRIER_FINE_BAND_HZ = 1000` (lambda/4 from this band) and `BARRIER_SAMPLES_HIGH = 16` (from 4 kHz); `_cut(azimuth)`; barrier request in `_listener` (:305-342) before `_open_ground`; `_Wave` gets `kind` and `cut`; `solve` (:863-959) makes wind, still and still-open jobs and keeps the memory-bounded pool (:936-946); `_place_barrier`; g_open on vertical-edge rows; module docstring (:22-31). |
| `engine/simulate.py`, `coverage.py`, `walkthrough.py`, `insertion.py`, `long_term.py` | None expected: they go through `Refraction.plan`/`solve`/`finish`. Check. |
| `validation/barrier_wind.py` (new) | Two references and the engine end to end (V3). |
| `tests/` | Section 7. |
| `docs/`, `CHANGELOG.md` | Section 8. |

## 3. What stays bit-identical, and what changes on purpose

- **Bit-identical by construction.** Everything in still air, since `refraction_for` returns None (`refracted.py:1074-1075`). That covers rooms, all still-air tables in `docs/validation.md`, and the `railway` and `street` demos, which have no atmosphere.
- **Bit-identical in wind.** Listeners whose own line crosses no barrier or building and that have no rows over vertical edges. The march with `obstacles=()` is bit-identical (B P7). This covers:
  - the fast field program (FFP), wide-angle, weather-zone, long-term and turbulence validations (`docs/validation.md:2052-2545, 2748, 2856-2912`);
  - `open_air_stage` and `helicopter_flyover`;
  - `lakeside_concert`. It runs with `diffraction: none`, so it has no edge rows. Its listeners' lines cross no house, and its map points behind the houses have empty in-plane families, so rule (f) leaves them on main.
- **Changes on purpose; each must be checked:**
  - in wind, every listener with a barrier request;
  - in wind, listeners with rows round barrier ends or building corners (g_open);
  - insertion loss and long-term levels with barriers: `tests/test_long_term.py:294-312` asserts IL_A > 10 dB, so re-check it;
  - `docs/outdoor.md:527-532` and `:1428-1434`, which say the weather does not change how much a barrier screens.

## 4. Bands, crossover and continuity

| Event | Rule | Measured | Limit in CI |
|---|---|---|---|
| Wind -> 0 | G, g_open -> 1 linearly; s = 0 (no shadow); still air exact | 0.014 dB per 0.01 m/s at 500 Hz (A p4c, B P8) | wind 0.1 m/s: \|dL\| <= 0.2 dB |
| Top height across grid nodes | smooth mask | 0.002 dB/mm (B P1) | 0.01 dB/mm |
| Top through the line of sight | no switch; UTD continuous (main checks it in 1 mm steps, `validation/edge_diffraction.py:389-420`) | G 0.023 dB per 2 cm (A p4b) | 0.05 dB per cm on the level |
| Roof height | snapped to a node; the ratio cancels the snap to first order (inference) | not measured | 0.1 dB per node, to measure |
| Band crossover | none for the in-plane family; end rows switch at main's open-ground top band (500 Hz or 2 kHz), as today | n/a | n/a |
| Barrier end (plane leaves the obstacle) | end rows take g_open on both sides; the remaining step comes from in-plane top rows, which take G_cut on the dark side only | not measured; A's G_cut - G_open is 5.7 dB at 500 Hz and 8-16 dB at 1 kHz downwind (p3 ends), damped by the top rows' share (inference) | report; target 1 dB per 0.1 m |
| 1 degree bin boundary | the cut follows the bin's centre | not measured | report on a map |
| Request range WAVE_MIN_RANGE (zs + zr) | as main's open-ground switch | not measured | report |

If the barrier-end step exceeds 1 dB, the first follow-up is A's blend: G = w G_cut + (1 - w) G_open, with w from the end's lateral Fresnel parameter.

## 5. Run-time budget (CPU seconds, one thread, narrow solver)

Per run, from A's p8 (prototype, 8 samples). The 4-8 kHz column doubles A's time for 16 samples (inference):

| Reach | 63-500 Hz | 1-2 kHz (lambda/4) | 4-8 kHz (lambda/4, 16 samples) | All bands |
|---|---|---|---|---|
| 120 m | 0.24 | 0.34 | 3.6 | 4.1 |
| 400 m | 1.24 | 2.5 | 26.6 | 30 |

- **One barrier plane** is 2 runs, plus the shared open still run. Bands run in parallel in the existing pool (`refracted.py:936-946`).
- **Point result:** about 8 s of CPU at 120 m and 60 s at 400 m.
- **Reviewers' map A:** two barriers, 117 points at 100-160 m, grass, 8 m/s; main takes 17-20 s, `wf5-two-verify-code-4.txt:27-29`. It has about 11-20 one-degree planes (B P4 counts 11 wind keys). Estimates (inference, untimed, because the shared native build is stale):
  - all bands: about 6-12 times main;
  - PE capped at 2 kHz: about 2-2.6 times main.
- **Reviewers' target for maps behind barriers** was 2 times main (wf5-two-verify-code-4.txt:27-29).
- **Cost cuts, measured first, then applied (follow-ups):**
  - 10-wavelength range steps from 1 kHz: +0.07 dB or less (A p6);
  - a lower absorbing-layer start at high bands, where `WAVE_TOP_SLOPE * reach` and `LAYER_MIN_M` (`parabolic.py:74`) dominate;
  - 2 degree planes above 2 kHz.

## 6. Validation plan and acceptance

- **V1, still air bit-identical.** All still-air tests, the still-air validation scripts' JSON (diffed) and the indoor result keys the reviewers used. Must be exact.
- **V2, wind with nothing in the plane bit-identical.** The section 3 list: `tests/test_ffp.py`, `test_wide_angle.py`, `test_refraction.py`, the zone, long-term and turbulence scripts, and the presets. Must be exact. A unit test checks that `march(obstacles=())` returns the same array for narrow and wide, with `independent` true and false.
- **V3, `validation/barrier_wind.py`.**
  - **(a) Full-wave mode matching**, ported from A's `mm_axi.py`: stepwise coupled modes after Evans (1983, JASA 74, 188, "A coupled mode solution for acoustic propagation in a waveguide with stepwise depth variations of a penetrable bottom"), with a perfectly matched layer.
    - Self-checks: 0.01 dB of the exact two-path field with no obstacle (measured 0.003); ring screen within 0.5 dB of Biot-Tolstoy-Medwin (BTM) for a straight barrier (-0.42 to +0.39); open ground in wind within 0.2 dB per sample of the engine's PE.
    - Cases: near (15 m / 4 m), far (100 m / 4 m), rail (0.5 m source, 3 m top at 4 m, i.e. 32 degrees), new steep tops at 45 and 60 degrees, roof (block 40-52 m, 6 m), two screens (8 m at 45 and 55 m).
    - Conditions: rigid ground and grass; +-5 and +10 m/s; 63-500 Hz.
  - **(b) Started PE**, ported from B's `p2_reference.py`. It is an independent PE over rigid ground, started 1 m behind a rigid screen from the exact four-path field (starter check 0.14 dB or less), with wind behind the screen. It covers 125 Hz-2 kHz (to extend to 8 kHz); `windy` up- and downwind, `clear_night`, `sunny_day`; screens 3 m and 6 m at 20 and 50 m.
  - **(c) The engine end to end** (`simulate`, rigid ground, `both_sides`) against (a) and (b).
  - **Acceptance**, per band, listeners 10 m or more behind, where the reference is within 25 dB of still air. Each limit is the measured maximum rounded up:

    | Case | Limit (dB) | Measured (dB) |
    |---|---|---|
    | Single screen, rail, roof (narrow solver), 5 m/s | 0.6 | 0.57 |
    | Same, 10 m/s | 0.8 | 0.74 |
    | (b) 125 Hz-2 kHz | 0.6 | -0.59..+0.32 |
    | Two screens, 250 Hz and up | 1.0 | 0.9 |
    | Two screens, 125 Hz | 1.7, documented | 1.7 |
    | Steep tops, 45 and 60 degrees | 0.6, else document | new |
    | (c) | (a) or (b) plus 0.15 (the still-air engine error) | new |

  - Deeper shadows are reported, not gated (B's P2: errors of 0.06-10 dB at 25-75 dB below still air, S4).
- **V4, convergence.** lambda/4 against lambda/10 on G at 1-8 kHz in the reference cases: 0.1 dB or less (measured 0.03). Samples on G: 8 against 32 to 2 kHz, 0.15 dB or less (measured 0.11); 16 against 32 at 4-8 kHz, 0.1 dB or less (measured 0.06).
- **V5, continuity in the engine:** the section 4 table.
- **V6, run time:** the reviewers' `maptime.py` maps A and P with `threads=1`; report the factor against main.
- **V7, follow-ups:**
  - Nord2000's seven screen cases: Plovsing 2006, DELTA AV 1117/06, "Nord2000. Validation of the propagation model", Table 4; 3 m/s downwind, z0 = 0.05 m. Run against (a), and report the engine against Nord2000 next to the published PE-against-Nord2000 figures: mean 1.4 dB, s.d. 1.3 dB, maximum 6.5 dB.
  - The engine's wave solver with moving air (`docs/validation.md:1825`) with a rigid barrier, 63-125 Hz, which also tests finite barrier ends in 3D.
  - Wind-tunnel data, digitised: Rasmussen & Arranz 1998, JASA 104, 2692; Salomons 1999, JASA 105, 2287.

## 7. Test plan

- `tests/test_parabolic_obstacles.py` (new):
  - bit-identity with no obstacles (narrow, wide, `independent` true and false);
  - a screen at a listener's range, and one before the wide starter's first step;
  - two screens in any order;
  - mask continuity: 1 mm steps across two nodes, at most 0.01 dB/mm in G;
  - roof floor against A's p5 numbers;
  - `refracting=True` routes steep points to the wide solver in still air.
- `tests/test_refraction.py`:
  - request rules (a)-(g), each on and off;
  - keys: the still run is shared across mirror wind directions and across sources at one height; one run per 1 degree bin;
  - in-plane coherent energy scales by G exactly;
  - one shared delay shift;
  - g_open on end rows on both sides of an end, with the leg's gain at 1 in the open bands;
  - wind 0.01 m/s against still air: 0.02 dB or less;
  - an object from a model file blocking the line leaves main's result.
- `tests/test_long_term.py:294-312`: re-check; update only with a stated reason.
- Lint, format and the message catalog as for every PR.

## 8. Docs to update

- `docs/outdoor.md`:
  - :527-532 (over an edge): in-plane paths in wind take the PE's weather factor;
  - :1428-1434 (limits): replace with section 9;
  - :720-721: the eddies' share now reaches the in-plane barrier paths as it does edge legs; a barrier's own shadow still takes nothing more;
  - the cost notes.
- `docs/validation.md`: a new section "Barriers in wind against a full-wave reference", with V3-V5 tables; mention it at :4005-4015 (known issues).
- `engine/refracted.py` module docstring (:22-31); `CHANGELOG.md`.

## 9. Limits to document, with sizes

- **The barrier's own wind field is not modelled.** It is 3 dB or more in the literature: Aballéa et al. 2007, Noise Control Eng. J. 55(1); Salomons 1999, JASA 105, 2287. Nord2000 differs by up to 6.5 dB from a PE without it (Plovsing 2006). See Q2.
- **One-way PE:** no reflections between parallel screens (up to 1.7 dB at 125 Hz, A p7), and none back towards the source (those stay rays).
- **Two dimensions:**
  - a ring cut round the source;
  - obliquity, 0.4-0.9 dB for 0-45 degrees (2D against 2.5D BEM, Aballéa et al. 2007), mostly cancelled in G (inference);
  - end paths take the open-ground factor g_open, not their own.
- **Buildings in the wide-angle solver:** two screens, up to 1.4 dB in the wind effect for a 60 m deep building at 500 Hz in 10 m/s (S1).
- **Objects from model files** are not in the cut; where one blocks the line, main's rays apply.
- **Shadows deeper than 25 dB below still air:** up to 10 dB (B P2, S4).
- **Still-air limits carry over**, for example buildings on hard ground 1.7-5.7 dB low (`docs/validation.md:4010-4015`). See Q3.
- **Effective sound speed at steep angles:** Ostashev, Wilson & Muhlestein 2020, JASA 147, 3969; Colas et al. 2023, arXiv 2309.09669.
- **Standards.** ISO 9613-2 (K_met, D_z), CNOSSOS-EU (curved paths) and Nord2000 (refraction-modified wedges) are engineering heuristics. The method here is a PE reference approach, as in the Harmonoise reference model (BEM plus PE; Defrance et al. 2007). v1 offers no conformance mode for ISO/TR 17534.

## 10. Follow-ups (not v1)

1. A blend across barrier ends, if V5 shows more than 1 dB.
2. Cost cuts (section 5), then reconsider the map bands.
3. Rigid roof floors in the wide solver, and roofs with their own material.
4. Objects from model files: a mesh cut.
5. The barrier's own wind field: the analytic field of Salomons & Rasmussen (2000, Appl. Acoust. 60, 327) through `speed_at`.
6. Reflections between parallel screens: complementary Kirchhoff (Aballéa & Defrance, CFA 2004).
7. Eddies scattering into the barrier's own shadow (`docs/outdoor.md:720-721`).
8. A rigid-correct starting field, for an absolute PE level (B F2).
9. The ray branch's independent fixes, each its own PR (`branch `claude/project-thread-jd2ksn-barrier-rays-wip` against 73a4fa0`):
   - `shadow.py` heights under 1 um; `_start` caches by `round(z, 3)` but passes the raw height (`refracted.py:694-698`);
   - the turning-ray fix 2795571;
   - the still-air two-edge ground bounces.
10. V7.

## 11. Questions only the owner can answer

- **Q1. Should maps behind barriers in wind run the PE in all eight bands?**
  - Point results and walks always do.
  - All bands: maps behind barriers about 6-12 times main (estimate).
  - Capping maps at 2 kHz and holding G_2k above: about 2-2.6 times main, but 4-8 kHz off by -11.4 to +8.0 dB (S2).
  - **Recommended:** all bands everywhere, so maps and points agree; add a draft cap only if V6's measured time is unacceptable.
- **Q2. Is the barrier's own wind field acceptable as a documented limit in v1?** The literature puts it at 3 dB or more downwind.
  - **Recommended:** yes. Document it with that size; add the analytic field (follow-up 5) next.
- **Q3. Should the still-air two-edge fix land before v1, or after?** It changes still-air outdoor results: thick barriers on hard ground go from 1.7-5.7 dB low to within 0.30 dB of BTM with images. v1 multiplies main's still-air base, so buildings on hard ground inherit that error.
  - **Recommended:** after v1, as its own PR with its own still-air checks. v1 does not depend on it, and a still-air change needs separate review.

## 12. One day (v1)

1. **1.5 h.** Screens: B's patch, plus `refracting=` and `steps_per_wavelength`; tests.
2. **1.5 h.** Roof floor in the narrow solver; tests.
3. **1 h.** `Layout.barriers` and `_cut`; tests.
4. **2 h.** Request, keys, wind, still and still-open jobs, 16 samples at 4-8 kHz; tests.
5. **1.5 h.** `_place_barrier`, g_open on end rows; continuity tests.
6. **1.5 h.** `validation/barrier_wind.py` (port `mm_axi.py` and `p2_reference.py`); its runs go in the background.
7. **1 h.** Docs, CHANGELOG, V1, V2 and V6 runs, lint, catalog.

**Environment.** The worktree's `_native` symlink points to a build without `Bvh.min_distance`, so a full `simulate()` fails (map of the PE, section 7). Rebuild the native module before V1, V2, V3(c) and V6.

## Sources

- Salomons 1994, JASA 95, 3109, "Diffraction by a screen in downwind sound propagation: a parabolic-equation approach".
- Salomons 1996, Internoise 96, 813-816 (zeroed barrier; BEM or GTD starters).
- Salomons 1999, JASA 105, 2287 (screen-induced wind gradients).
- Salomons & Rasmussen 2000, Appl. Acoust. 60, 327.
- Gilbert & Di 1993 (GFPE); Collins 1993, JASA 93, 1736 (split-step Padé); Evans 1983, JASA 74, 188 (coupled modes).
- Premat et al. 2002-03 and Aballéa et al. 2003, Mécanique & Industries 4, 411 (BEM starter).
- Defrance et al. 2007, Acta Acust. 93, 213 (Harmonoise reference model).
- Aballéa et al. 2007, Noise Control Eng. J. 55(1).
- Plovsing 2006, DELTA AV 1117/06 (Nord2000 validation).
- Rasmussen & Arranz 1998, JASA 104, 2692.
- Van Renterghem, Salomons & Botteldooren 2005, Acta Acust. 91, 671.
- Hornikx, Waxler & Forssén 2010, JASA 128, 1632.
- Ostashev, Wilson & Muhlestein 2020, JASA 147, 3969; Colas et al. 2023, arXiv 2309.09669.

The literature survey (`pe-design/literature/NOTES.md`) lists the web sources. It also notes what it could not check: Defrance 2007's full text, and Daigle 1982 and ISO 9613-2 Table 5, both quoted from memory.
