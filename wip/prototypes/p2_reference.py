"""P2: the weather ratio against an independent rigid-barrier reference.

Reference: rigid ground, rigid thin screen. A split-step PE of my own (exact square-root
propagator, GFPE phase screen for c(z), even field about the rigid ground) started 1 m
behind the screen from the exact still-air field there (UTD half-plane, four paths
S/S' -> top -> R/R', literature/kirchhoff_check.utd_halfplane), then marched in wind.
The wind blows only behind the screen in both the reference and the candidates, which
isolates what the barrier's model does to the weather effect.

Candidates at each receiver (dB re free field, band = energy mean of the engine's 8 freqs):
  rays   : still-air 4-path UTD (stands in for the engine's validated still-air rays;
           main gives exactly this downwind behind a barrier)
  kirch  : the engine's narrow PE with the screen zeroed (prototype), in wind, absolute
  ratio  : rays + 10 lg(E_pe,wind / E_pe,still), both PE runs with the zeroed screen
Usage: p2_reference.py band cosine h [x_b]"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kirchhoff_check import utd_halfplane  # noqa: E402

from common import CENTRES, F, P, atmosphere, still  # noqa: E402

b, cos, h = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
xb = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
preset = sys.argv[5] if len(sys.argv) > 5 else "windy"
zs = 1.0
x0 = xb + 1.0
RECS = [(50.0, 1.5), (50.0, 4.0), (100.0, 1.5), (100.0, 4.0), (200.0, 1.5), (200.0, 4.0), (400.0, 1.5)]
if len(sys.argv) > 6:
    RECS = [tuple(map(float, s.split(":"))) for s in sys.argv[6].split(",")]
recs = [r for r in RECS if r[0] > x0 + 1.0]
wind = atmosphere(preset, cos)
c0 = float(wind(np.array([0.0]))[0])
calm = still(c0)


def four(k, x, z):
    tot = 0.0
    for s in ((0.0, zs), (0.0, -zs)):
        for r in ((x, z), (x, -z)):
            u, _, _ = utd_halfplane(k, s, (xb, h), r, True)
            tot = tot + u
    return tot


def reference(f, speed):
    k = 2 * math.pi * f / c0
    lam = c0 / f
    zmax = max(h, 4.0) + 25.0
    layer = max(30.0, 20 * lam)
    Z = zmax + layer
    dz = lam / 16
    N = 1 << int(math.ceil(math.log2(2 * Z / dz)))
    dz = 2 * Z / N
    z = -Z + dz * (np.arange(N) + 0.5)  # symmetric, no sample at 0
    pos = z > 0
    p0 = np.zeros(N, dtype=complex)
    zp = z[pos]
    p0[pos] = [four(k, x0, zz) for zz in zp]
    p0[~pos] = p0[pos][::-1]
    psi = p0 * math.sqrt(x0) * np.exp(-1j * k * x0)
    kz = 2 * math.pi * np.fft.fftfreq(N, d=dz)
    kx = np.sqrt((k * k - kz * kz).astype(complex))
    kx = np.where(kx.imag < 0, np.conj(kx), kx)
    dk = 2 * math.pi * f / speed(np.abs(z)) - k
    sponge = np.ones(N)
    out = np.abs(z) > zmax
    sponge[out] = np.exp(-(((np.abs(z[out]) - zmax) / layer) ** 2) * 3.0)
    dr0 = min(0.5, lam)
    r = x0
    res = {}
    for x, zr in sorted(recs):
        while r < x - 1e-9:
            dr = min(dr0, x - r)
            half = np.exp(0.5j * dr * dk)
            psi = sponge * half * np.fft.ifft(np.fft.fft(half * psi) * np.exp(1j * dr * (kx - k)))
            r += dr
        res[(x, zr)] = (np.interp(zr, z, psi.real) + 1j * np.interp(zr, z, psi.imag)) * np.exp(1j * k * x) / math.sqrt(x)
    return res


def main():
    f = F[b]
    k = 2 * math.pi * f / c0
    E = {key: {"ref_wind": [], "ref_still": [], "rays": []} for key in recs}
    for fi in f:
        rw = reference(fi, wind)
        rs = reference(fi, calm)
        for key in recs:
            d = math.hypot(key[0], key[1] - zs)
            E[key]["ref_wind"].append(abs(rw[key] * d) ** 2)
            E[key]["ref_still"].append(abs(rs[key] * d) ** 2)
            E[key]["rays"].append(abs(four(2 * math.pi * fi / c0, *key) * d) ** 2)
    # engine PE with zeroed screen (prototype), wind only behind the screen
    behind = lambda r: wind if r >= xb else calm  # noqa: E731
    xs = np.array([x for x, _ in recs]); zr = np.array([z for _, z in recs])
    top = max(h, 4.0) + 20.0
    pw = P._march(f, calm, lambda r: None, zs, xs, zr, top, None, behind, False, ((xb, h),))
    ps = P._march(f, calm, lambda r: None, zs, xs, zr, top, None, None, False, ((xb, h),))
    out = []
    for j, key in enumerate(recs):
        d = math.hypot(key[0], key[1] - zs)
        L = {n: 10 * math.log10(np.mean(v)) for n, v in E[key].items()}
        ew = np.mean(np.abs(pw[:, j]) ** 2) * d * d
        es = np.mean(np.abs(ps[:, j]) ** 2) * d * d
        g = 10 * math.log10(ew / es)
        row = dict(band=CENTRES[b], cos=cos, h=h, xb=xb, x=key[0], z=key[1],
                   ref_wind=L["ref_wind"], ref_still=L["ref_still"], rays_still=L["rays"],
                   ref_effect=L["ref_wind"] - L["ref_still"], pe_g=g,
                   kirch_wind=10 * math.log10(ew), kirch_still=10 * math.log10(es))
        row["err_ratio"] = L["rays"] + g - L["ref_wind"]
        row["err_kirch"] = row["kirch_wind"] - L["ref_wind"]
        row["err_main"] = L["rays"] - L["ref_wind"]
        row["starter_check"] = L["ref_still"] - L["rays"]
        out.append(row)
        print(json.dumps({k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "p2.jsonl"), "a") as fh:
        for row in out:
            fh.write(json.dumps(row) + "\n")


main()
