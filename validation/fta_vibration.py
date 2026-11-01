"""Check the ground-borne vibration of :mod:`acoustic_spaces.vibration` against the FTA's
manual it comes from (Transit Noise and Vibration Impact Assessment Manual, FTA Report
No. 0123, 2018).

    python validation/fta_vibration.py

- **The base curves against figure 6-4.** The equations of table 6-10 at 10, 100 and
  300 ft against the curves the manual draws, read off the figure by eye (to about
  half a decibel).
- **The worked example.** Example 6-1, light rail at 40 mph 125 ft from wood-frame
  houses on soil that propagates efficiently: each sum of the example, from its own
  reading of the base curve (65 VdB), and the whole from the curve unrounded.
- **The speed rows.** Table 6-11's adjustments for 20 to 60 mph against eq. 6-4.
- **The plant's and the damage criteria's levels.** Each row of tables 7-4 and 7-5:
  its rms level against its peak particle velocity 12 dB down (a crest factor of 4).
- **The manual's words.** Its text's typical levels against its curves.

What is not checked: no measurement is compared here. The curves are North American
measurements' upper range; the method has no spectra.
"""

from __future__ import annotations

from acoustic_spaces import vibration as v

#: Figure 6-4 read by eye at 10, 100 and 300 ft (the bus at 200 ft, where it leaves the
#: figure's 50 VdB): (curve, feet, VdB).
FIGURE = (
    ("locomotive", 10, 94.5),
    ("locomotive", 100, 78.5),
    ("locomotive", 300, 67.0),
    ("rapid_transit", 10, 81.5),
    ("rapid_transit", 100, 67.5),
    ("rapid_transit", 300, 56.0),
    ("rubber_tyred", 10, 75.0),
    ("rubber_tyred", 100, 57.0),
    ("rubber_tyred", 200, 50.5),
)
MPH = v.MILE_KM


def main() -> int:
    print("Base curves against figure 6-4 (VdB):")
    for curve, feet, read in FIGURE:
        level = v.CURVES[curve].level(feet * v.FOOT_M)
        print(f"  {curve:<14}{feet:4d} ft  figure {read:5.1f}  equation {level:5.1f}")

    d = 125 * v.FOOT_M
    base = v.CURVES["rapid_transit"].level(d)
    speed = v.speed_db(40 * MPH, 50 * MPH)
    soil = v.Line(ground="efficient").path_db(d)
    coupling = v.COUPLINGS["wood_frame"]
    floors = v.floors_db(1) + v.FLOOR_AMPLIFICATION_DB
    print("\nExample 6-1 (VdB), from its reading of the curve, 65:")
    print(f"  the curve at 125 ft  {base:.2f} (the example reads 65)")
    print(f"  speed {speed:+.1f}: {65 + speed:.1f} (example 63.1)")
    print(
        f"  soil {soil:+.0f}, coupling {coupling:+.0f}: {65 + speed + soil + coupling:.1f} (68.1)"
    )
    total = 65 + speed + soil + coupling + floors
    print(f"  first floor {floors:+.0f}: {total:.1f} (72.1)")
    whole = v.ground_level("rapid_transit", d, 40 * MPH, v.Line(ground="efficient"))
    print(f"  from the curve unrounded: {whole + v.indoor_db('wood_frame', 1):.1f}")

    print("\nTable 6-11's speeds against eq. 6-4 (dB):")
    rows = {60: (1.6, 6.0), 50: (0.0, 4.4), 40: (-1.9, 2.5), 30: (-4.4, 0.0), 20: (-8.0, -3.5)}
    for mph, (rail_db, bus_db) in rows.items():
        r, b = v.speed_db(mph * MPH, 50 * MPH), v.speed_db(mph * MPH, 30 * MPH)
        print(f"  {mph} mph  table {rail_db:+.1f} {bus_db:+.1f}  eq. 6-4 {r:+.2f} {b:+.2f}")

    print("\nTables 7-4 and 7-5: rms level against PPV - 12 dB (VdB):")
    for key, row in v.PLANT.items():
        print(f"  {key:<26}{row.level_vdb:6.0f}{v.level_of(row.ppv_in_s):8.1f}")
    for key, damage in v.DAMAGE.items():
        print(f"  damage {key:<19}{damage.level_vdb:6.0f}{v.level_of(damage.ppv_in_s):8.1f}")

    loco = [v.CURVES["locomotive"].level(ft * v.FOOT_M) for ft in (25, 50, 100)]
    rt = [v.CURVES["rapid_transit"].level(ft * v.FOOT_M) for ft in (25, 50, 100)]
    bus = v.CURVES["rubber_tyred"].level(25 * v.FOOT_M)
    gaps = ", ".join(f"{a - b:.1f}" for a, b in zip(loco, rt, strict=True))
    print("\nThe manual's words against its curves:")
    print(f"  light rail '70 VdB or more near their tracks': {rt[1]:.1f} VdB 50 ft out")
    print(f"  buses and trucks 'rarely' over 70 VdB: a bus {bus:.1f} VdB 25 ft out")
    print(f"  heavy locomotives '5 to 10 dB' over rail transit: {gaps} dB at 25, 50, 100 ft")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
