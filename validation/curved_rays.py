"""Check the levels of the curved paths outdoors against the smooth profiles they are
sampled from.

    python validation/curved_rays.py [--listeners 1300] [--seed 1] [--weathers ...]
        [--json results.json]

The engine traces the direct sound and its reflections off the open ground through a
profile of the speed of sound that is linear between nodes, sampled from the smooth
profile of the weather (:mod:`acoustic_spaces.engine.refraction`). Here the same paths
are worked out on the smooth profile itself (the atmosphere's ``sound_speed`` written
out, and checked against it): the range of a ray of parameter p = cos(elevation) / c,
the integral of p c / sqrt(1 - p²c²) dz, and its length, of dz / sqrt(1 - p²c²), by
adaptive quadrature with the square root at the turning height as the quadrature's
weight; the ray through the listener by a root of the range; the slope of the range by
central differences in p; and the level over free field at the path's length L, 10
lg(p c_s² L² / (r |dr/dp| sin_s sin_r)), sin_s and sin_r the ray's sines at the source
and the listener (0 dB on a straight path). Nothing comes from the engine's ray code
but which paths it found. Two families, at 20 °C in the weathers of ``WEATHERS``: the
``windy`` preset downwind and the ``clear_night`` preset, where the sound bends down,
and the ``windy`` preset upwind and the ``sunny_day`` preset, where it bends up:

- **Paths to listeners about.** From sources 1, 4 and 5 m up to listeners at random 0.5
  to 12 m up and 100 to 1000 m away (20 to 400 m where the sound bends up; seeded),
  straight or off the ground up to three times: each path's level against the smooth
  profile's, those 8 dB or more over a straight path's (nearer a caustic, where the
  engine caps the level at 10 dB) and those the engine caps counted apart; and the
  energy sum of the paths at each listener none of whose paths are near a caustic,
  capped, at the shadow's edge (``Smooth.grazes``) or without a smooth counterpart.
- **Ends at one height.** The direct path between a source and a listener at the same
  height, 1 to 25 m up and 10 to 600 m apart, which turns just above them (just below
  them where the sound bends up), those near a caustic, capped or at the shadow's edge
  counted apart.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from acoustic_spaces.engine.refraction import Profile, Rays
from acoustic_spaces.scene import Atmosphere
from acoustic_spaces.scene.model import SOUND_SPEED_0C

GROUND_C = 20.0
SOURCES = (1.0, 4.0, 5.0)
#: The weathers by name: the preset, the way the sound goes over the ground (None in
#: still air), and how far away the listeners are (m): nearer where the sound bends up,
#: within the shadow's edge.
WEATHERS = {
    "windy_downwind": ("windy", np.array([1.0, 0.0]), (100.0, 1000.0)),
    "clear_night": ("clear_night", None, (100.0, 1000.0)),
    "windy_upwind": ("windy", np.array([-1.0, 0.0]), (20.0, 400.0)),
    "sunny_day": ("sunny_day", None, (20.0, 400.0)),
}
MAX_BOUNCES = 3
#: Paths this many dB or more over a straight path's are nearer a caustic.
CAUSTIC_DB = 8.0
ONE_HEIGHT = np.arange(1.0, 25.01, 0.5)
ONE_HEIGHT_RANGES = (10.0, 30.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0)
#: Rays that turn higher than this, or not at all, are not followed back down (the
#: engine's profile stops changing 1000 m up): the search for a ray through the
#: listener passes them by.
HIGHEST_TURN_M = 1e4
#: Rays this close to running level on the ground (1 - p c(0)) where the profile falls
#: are not checked: there the range here and the ray equations shot and differenced can
#: part (by 13 dB 1.2e-7 from it), where 2.8e-6 to 4.3e-5 from it they agree to 0.001 dB.
GRAZING = 1e-6
_TOLERANCE = {"epsabs": 1e-10, "epsrel": 1e-11, "limit": 200}


class Smooth:
    """Rays in a smooth profile that only rises or only falls with height: ``speed(z)``
    (m/s), and ``rate(z, dz)``, the change of speed from z to z + dz over dz worked out
    without cancellation (the slope at dz = 0). Where it rises (``rising``), every ray
    turns back down somewhere above the ground; where it falls, none does, and a ray
    shallow enough turns back up in the air before it reaches the ground."""

    def __init__(self, speed, rate, rising: bool) -> None:
        self.speed, self.rate, self.rising = speed, rate, rising

    def turn(self, p: float) -> float:
        """The height where the ray of parameter p runs level going up (inf above
        ``HIGHEST_TURN_M``, or where the profile falls)."""
        if not self.rising:
            return math.inf
        top = 1.0
        while self.speed(top) * p < 1.0:
            if top > HIGHEST_TURN_M:
                return math.inf
            top *= 2.0
        return brentq(lambda z: self.speed(z) * p - 1.0, 0.0, top, xtol=1e-14, rtol=1e-15)

    def low(self, p: float) -> float:
        """The lowest height the ray of parameter p reaches: where it runs level going
        down, or the ground."""
        if self.rising or self.speed(0.0) * p <= 1.0:
            return 0.0
        top = 1.0
        while self.speed(top) * p >= 1.0:
            top *= 2.0
        return brentq(lambda z: self.speed(z) * p - 1.0, 0.0, top, xtol=1e-14, rtol=1e-15)

    def grazes(self, p: float, ground: bool) -> bool:
        """Whether the ray of parameter p is at the edge of the rays that reach the ground
        where the profile falls, the shadow's edge, where the level of a ray off the
        ground changes without bound: within ``GRAZING`` of running level on it, or
        reaching the ground where the engine's does not (``ground``) or the other way
        round, as the edge lies at a slightly different p in the sampled profile."""
        if self.rising:
            return False
        return abs(1.0 - self.speed(0.0) * p) < GRAZING or (self.low(p) == 0.0) != ground

    def _weighted(self, p: float, zt: float, length: bool):
        """The integrand times sqrt|zt - h|, smooth up to the turn at zt: 1 - p²c² =
        |zt - h| q(h), q = p (1 + p c) |c(zt) - c(h)| / |zt - h|."""

        def weighted(h: float) -> float:
            c = self.speed(h)
            q = p * (1.0 + p * c) * abs(self.rate(min(h, zt), abs(zt - h)))
            return (1.0 if length else p * c) / math.sqrt(q)

        return weighted

    def _part(self, lo: float, hi: float, p: float, zt: float, length: bool) -> float:
        """The range (or the length) of the ray of parameter p, turning at zt, from height
        lo up to hi (at most zt): plainly up to half the turn's height, and above that with
        the turn's square root as the quadrature's weight."""

        def plain(h: float) -> float:
            c = self.speed(h)
            return (1.0 if length else p * c) / math.sqrt(1.0 - (p * c) ** 2)

        weighted = self._weighted(p, zt, length)

        def to_turn(z: float) -> float:
            if zt - z <= 1e-9 * zt:
                # Too short to subdivide: the weight's integral, 2 sqrt(zt - z), times
                # the rest.
                return 2.0 * math.sqrt(max(zt - z, 0.0)) * weighted(0.5 * (z + zt))
            return quad(weighted, z, zt, weight="alg", wvar=(0.0, -0.5), **_TOLERANCE)[0]

        split = 0.5 * zt
        total = 0.0
        if lo < min(hi, split):
            total += quad(plain, lo, min(hi, split), **_TOLERANCE)[0]
        if hi > split:
            total += to_turn(max(lo, split)) - to_turn(hi)
        return total

    def _from_low(self, zl: float, hi: float, p: float, length: bool) -> float:
        """The range (or the length) of the ray of parameter p from the height zl where it
        turns going down (above the ground) up to hi, with the turn's square root as the
        quadrature's weight."""
        weighted = self._weighted(p, zl, length)
        if hi - zl <= 1e-9 * zl:
            return 2.0 * math.sqrt(max(hi - zl, 0.0)) * weighted(0.5 * (zl + hi))
        return quad(weighted, zl, hi, weight="alg", wvar=(-0.5, 0.0), **_TOLERANCE)[0]

    def _combine(self, zs, zr, p, branch, length) -> float:
        up, arrives_up, cycle = branch
        turns = 2 * cycle + 2 * (up and not arrives_up)
        if not self.rising:
            # From the lowest height reached: the ground, or a turn in the air.
            zl = self.low(p)
            if turns or zs < zl or zr < zl or (up and arrives_up and zr < zs):
                return math.nan
            if zl == 0.0:
                a = self._part(0.0, zs, p, math.inf, length)
                b = self._part(0.0, zr, p, math.inf, length)
            else:
                a, b = self._from_low(zl, zs, p, length), self._from_low(zl, zr, p, length)
            return (-a if up else a) + (b if arrives_up else -b)
        zt = self.turn(p)
        a = self._part(0.0, zs, p, zt, length)  # from the ground up to the source
        b = self._part(0.0, zr, p, zt, length)  # from the ground up to the listener
        if turns and zt > HIGHEST_TURN_M:
            return math.nan
        h = self._part(0.0, zt, p, zt, length) if turns else 0.0  # up to the turn
        return (-a if up else a) + (b if arrives_up else -b) + turns * h

    def range(self, zs, zr, p, branch) -> float:
        """The range (m) of the ray of parameter p along ``branch`` (up, arrives_up,
        cycle) from zs to zr."""
        return self._combine(zs, zr, p, branch, False)

    def length(self, zs, zr, p, branch) -> float:
        """Its length (m)."""
        return self._combine(zs, zr, p, branch, True)

    def eigenray(self, zs, zr, distance, branch, guess) -> float | None:
        """The p of the ray along ``branch`` that reaches the listener, nearest ``guess``."""
        top = (1.0 - 1e-15) / max(self.speed(zs), self.speed(zr))

        def miss(q):
            return self.range(zs, zr, q, branch) - distance

        width = 3e-6 * guess
        for _ in range(40):
            lo, hi = max(guess - width, 0.5 * guess), min(guess + width, top)
            if miss(lo) * miss(hi) < 0.0:
                return brentq(miss, lo, hi, xtol=1e-22, rtol=1e-15)
            width *= 1.5
        return None

    def level(self, zs, zr, p, branch) -> float:
        """The path's level over free field at its length (dB)."""
        cs, cr = self.speed(zs), self.speed(zr)
        # Central differences, well inside the way to the rays that run level at an end,
        # or, where the profile falls, on the ground (those that only graze it).
        step = min(1e-3 * min(1.0 / max(cs, cr) - p, p), 1e-7 * p)
        if not self.rising:
            step = min(step, 1e-3 * abs(1.0 / self.speed(0.0) - p))
        ahead = self.range(zs, zr, p + step, branch)
        slope = (ahead - self.range(zs, zr, p - step, branch)) / (2.0 * step)
        r, length = self.range(zs, zr, p, branch), self.length(zs, zr, p, branch)
        sines = math.sqrt(1.0 - (p * cs) ** 2) * math.sqrt(1.0 - (p * cr) ** 2)
        return 10.0 * math.log10(p * cs * cs * length**2 / (r * abs(slope) * sines))


def smooth_of(atmosphere: Atmosphere, ground_c: float, towards) -> Smooth:
    """The smooth profile of ``atmosphere`` (a log wind profile, temperatures linear and
    logarithmic in height) with the air at ``ground_c`` °C on the ground, along
    ``towards`` (None: still air), checked against its ``sound_speed``."""
    z0 = atmosphere.roughness_m
    wind = atmosphere.wind
    along = 0.0 if towards is None else float(np.dot(atmosphere.to_wind(), towards))
    rise = 0.0 if wind is None else along * wind.speed_m_s / math.log1p(wind.height_m / z0)
    gradient, log_k = atmosphere.temperature_gradient_k_m, atmosphere.temperature_log_k

    def kelvin(z: float) -> float:
        return 1.0 + (ground_c + gradient * z + log_k * math.log1p(z / z0)) / 273.15

    def speed(z: float) -> float:
        z = max(z, 0.0)
        return SOUND_SPEED_0C * math.sqrt(kelvin(z)) + rise * math.log1p(z / z0)

    def rate(z: float, dz: float) -> float:
        # log(1 + (z + dz) / z0) - log(1 + z / z0) = log(1 + dz / (z0 + z)), over dz.
        logs = math.log1p(dz / (z0 + z)) / dz if dz > 0.0 else 1.0 / (z0 + z)
        warming = (gradient + log_k * logs) / 273.15
        roots = math.sqrt(kelvin(z + dz)) + math.sqrt(kelvin(z))
        return SOUND_SPEED_0C * warming / roots + rise * logs

    heights = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 200)])
    given = atmosphere.sound_speed(heights, ground_c, towards)
    assert np.allclose([speed(z) for z in heights], given, rtol=0.0, atol=1e-9)
    ahead = atmosphere.sound_speed(heights + 0.01, ground_c, towards)
    slopes = [rate(z, 0.01) for z in heights]
    assert np.allclose(slopes, (ahead - given) / 0.01, rtol=1e-6)
    rising = all(s > 0.0 for s in slopes)
    if not rising and not all(s < 0.0 for s in slopes):
        raise ValueError("the profile neither only rises nor only falls with height")
    return Smooth(speed, rate, rising)


def _weather(name: str):
    preset, towards, _ = WEATHERS[name]
    atmosphere = Atmosphere.model_validate({"preset": preset})
    return Profile.of(atmosphere, GROUND_C, towards), smooth_of(atmosphere, GROUND_C, towards)


def _level(found, i: int) -> float:
    return 10.0 * math.log10(found.spreading[i] * found.length[i] ** 2)


def places(weather: str, count: int, seed: int) -> list[tuple[float, float]]:
    """``count`` listeners at random (distance, height) for ``weather``, as far away as
    it gives and 0.5 to 12 m up, seeded."""
    near, far = WEATHERS[weather][2]
    rng = np.random.default_rng(seed)
    return [(float(rng.uniform(near, far)), float(rng.uniform(0.5, 12.0))) for _ in range(count)]


def _counterparts(smooth: Smooth, zs, zr, distance, found) -> list[float | None]:
    """The p of the smooth profile's path that matches each of the engine's paths
    ``found`` (``Smooth.eigenray`` from its p), or None where there is none: where two
    of the engine's paths along a branch come to the same smooth path, it matches the
    one whose p is nearer, as the sampled profile can have a path the smooth one lacks
    (such as one that leaves the source almost level)."""
    out: list[float | None] = []
    for i in range(len(found)):
        b = found.branch[i]
        branch = (b.up, b.arrives_up, b.cycle)
        out.append(smooth.eigenray(zs, zr, distance, branch, float(found.p[i])))
    for i, j in itertools.combinations(range(len(found)), 2):
        pi, pj = out[i], out[j]
        same = found.branch[i] == found.branch[j]
        if same and pi is not None and pj is not None and abs(pi - pj) <= 1e-12 * pi:
            far = i if abs(found.p[i] - pi) > abs(found.p[j] - pj) else j
            out[far] = None
    return out


def family(weather: str, zs: float, listeners: list[tuple[float, float]]) -> dict:
    """The engine's paths against the smooth profile's for one weather and source."""
    profile, smooth = _weather(weather)
    rays = Rays(profile, zs)
    off, near, totals, unmatched, grazing = [], [], [], 0, 0
    worst: tuple[float, object] = (0.0, None)
    for distance, zr in listeners:
        found = rays.eigenrays(zr, distance, MAX_BOUNCES)
        engine_sum = smooth_sum = 0.0
        clean = len(found) > 0
        matches = _counterparts(smooth, zs, zr, distance, found)
        for i, p in enumerate(matches):
            b = found.branch[i]
            branch = (b.up, b.arrives_up, b.cycle)
            if p is None:
                unmatched += 1
                clean = False
                continue
            if smooth.grazes(p, bool(found.ground[i])):
                grazing += 1
                clean = False
                continue
            reference = smooth.level(zs, zr, p, branch)
            if found.capped[i] or reference >= CAUSTIC_DB:
                near.append(_level(found, i) - reference)
                clean = False
                continue
            off.append(_level(found, i) - reference)
            if abs(off[-1]) > abs(worst[0]):
                worst = (off[-1], (distance, zr, branch))
            engine_sum += found.spreading[i]
            smooth_sum += 10.0 ** (reference / 10.0) / smooth.length(zs, zr, p, branch) ** 2
        if clean:
            totals.append(10.0 * math.log10(engine_sum / smooth_sum))
    off_a, near_a, totals_a = (np.abs(np.array(x, dtype=float)) for x in (off, near, totals))
    return {
        "weather": weather,
        "source_m": zs,
        "paths": len(off_a),
        "within_0.1_db": float(np.mean(off_a <= 0.1)) if len(off_a) else None,
        "over_1_db": int(np.sum(off_a > 1.0)),
        "max_db": float(off_a.max(initial=0.0)),
        "worst": {"off_db": worst[0], "where": worst[1]},
        "near_a_caustic": len(near_a),
        "near_a_caustic_max_db": float(near_a.max(initial=0.0)),
        "unmatched": unmatched,
        "grazing": grazing,
        "listeners_summed": len(totals_a),
        "sum_over_0.5_db": int(np.sum(totals_a > 0.5)),
        "sum_max_db": float(totals_a.max(initial=0.0)),
    }


def one_height(weather: str) -> dict:
    """The direct path between ends at one height against the smooth profile's: the one
    that turns once in the air, above the ends where the profile rises, below them
    where it falls."""
    profile, smooth = _weather(weather)
    turning = (True, False, 0) if smooth.rising else (False, True, 0)
    off, near, grazing = [], [], 0
    worst: tuple[float, object] = (0.0, None)
    for height in ONE_HEIGHT:
        rays = Rays(profile, float(height))
        for distance in ONE_HEIGHT_RANGES:
            found = rays.eigenrays(float(height), distance, 0)
            matches = _counterparts(smooth, height, height, distance, found)
            for i, p in enumerate(matches):
                branch = (found.branch[i].up, found.branch[i].arrives_up, found.branch[i].cycle)
                if branch != turning:
                    continue
                if p is None:
                    continue
                if smooth.grazes(p, bool(found.ground[i])):
                    grazing += 1
                    continue
                reference = smooth.level(height, height, p, branch)
                if found.capped[i] or reference >= CAUSTIC_DB:
                    near.append(_level(found, i) - reference)
                    continue
                off.append(_level(found, i) - reference)
                if abs(off[-1]) > abs(worst[0]):
                    worst = (off[-1], (float(height), distance))
    off_a = np.abs(np.array(off))
    return {
        "weather": weather,
        "paths": len(off_a),
        "within_0.1_db": float(np.mean(off_a <= 0.1)) if len(off_a) else None,
        "max_db": float(off_a.max(initial=0.0)),
        "worst": {"off_db": worst[0], "where": worst[1]},
        "near_a_caustic": len(near),
        "near_a_caustic_max_db": float(np.abs(np.array(near)).max(initial=0.0)),
        "grazing": grazing,
    }


def _share(value) -> str:
    return "-" if value is None else f"{100 * value:.0f} %"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--listeners", type=int, default=1300, help="per weather and source")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--weathers", nargs="+", default=list(WEATHERS), choices=list(WEATHERS))
    parser.add_argument("--json", type=Path, help="write the numbers to this file")
    args = parser.parse_args(argv)

    results: dict[str, list] = {"about": [], "one_height": []}
    print(
        f"Paths to {args.listeners} listeners 0.5 to 12 m up (seed {args.seed}), up to "
        f"{MAX_BOUNCES} bounces, {GROUND_C:g} C, engine less smooth:\n"
    )
    print(
        "| weather | listeners | source | paths | within 0.1 dB | over 1 dB | most | where "
        "(distance, height, branch) | near a caustic (most) | grazing | sums over 0.5 dB | "
        "most |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for weather in args.weathers:
        listeners = places(weather, args.listeners, args.seed)
        near, far = WEATHERS[weather][2]
        for zs in SOURCES:
            started = time.perf_counter()
            r = family(weather, zs, listeners)
            r["seconds"] = time.perf_counter() - started
            results["about"].append(r)
            print(
                f"| `{weather}` | {near:g} to {far:g} m | {zs:g} m | {r['paths']} | "
                f"{_share(r['within_0.1_db'])} | {r['over_1_db']} | {r['max_db']:.2f} dB | "
                f"{r['worst']['where']} | "
                f"{r['near_a_caustic']} ({r['near_a_caustic_max_db']:.2f} dB) | {r['grazing']} | "
                f"{r['sum_over_0.5_db']} of {r['listeners_summed']} | {r['sum_max_db']:.2f} dB |",
                flush=True,
            )
    print(
        f"\nThe direct path between ends at one height, {ONE_HEIGHT[0]:g} to "
        f"{ONE_HEIGHT[-1]:g} m up, {ONE_HEIGHT_RANGES[0]:g} to {ONE_HEIGHT_RANGES[-1]:g} m "
        "apart:\n"
    )
    print(
        "| weather | paths | within 0.1 dB | most | where (height, distance) | near a "
        "caustic (most) | grazing |"
    )
    print("|---|---|---|---|---|---|---|")
    for weather in args.weathers:
        r = one_height(weather)
        results["one_height"].append(r)
        print(
            f"| `{weather}` | {r['paths']} | {_share(r['within_0.1_db'])} | "
            f"{r['max_db']:.2f} dB | {r['worst']['where']} | {r['near_a_caustic']} "
            f"({r['near_a_caustic_max_db']:.2f} dB) | {r['grazing']} |",
            flush=True,
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
