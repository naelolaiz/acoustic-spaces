"""How far is the PE's thin screen (field zeroed below the top, vpe.py) from the exact
rigid and soft half-plane, deep in the shadow?  Free space, still air, point source.

References (all in dB re the free field at the receiver):
  pe     : vpe.march (exact sqrt propagator, field set to zero below the top at the screen)
  fresnel: Fresnel knife-edge, v = 2 sqrt(delta/lambda) (the reviewers' reference)
  hard / soft : UTD half-plane (Kouyoumjian & Pathak 1974), point source, straight edge,
           normal incidence; asymptotically exact for k L >> 1 (Macdonald's solution).
  ring   : 10 lg((s + s') / r_h): what the PE's axisymmetry (a ring screen round the source)
           adds over a straight screen, my inference (spreading over the horizontal range
           instead of the path length).
Usage: python kirchhoff_check.py [freqs...]
"""

import math
import os
import sys

import numpy as np
from scipy.special import fresnel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vpe import march  # noqa: E402

C0 = 340.0


def tail(a):
    """int_a^inf exp(i t^2) dt."""
    u = a * math.sqrt(2 / math.pi)
    S, C = fresnel(u)
    return math.sqrt(math.pi / 2) * ((0.5 - C) + 1j * (0.5 - S))


def Fstar(X):
    """Kouyoumjian-Pathak transition function, conjugated for exp(-i w t)."""
    if X < 1e-12:
        return 0.0
    a = math.sqrt(X)
    return -2j * a * np.exp(-1j * X) * tail(a)


def utd_halfplane(k, src, edge, rec, hard):
    """Total field (exp(-iwt), source exp(ikR)/R) behind a half-plane whose edge is at
    ``edge`` and which hangs down from it (vertical, in x = edge x)."""
    def ang(p):
        dx, dz = p[0] - edge[0], p[1] - edge[1]
        a = math.atan2(-dx, -dz)
        return a % (2 * math.pi)
    phi_s, phi = ang(src), ang(rec)
    s1 = math.hypot(src[0] - edge[0], src[1] - edge[1])
    s = math.hypot(rec[0] - edge[0], rec[1] - edge[1])
    L = s * s1 / (s + s1)
    D = 0.0
    for beta, sign in ((phi - phi_s, 1.0), (phi + phi_s, (1.0 if hard else -1.0))):
        D += sign * Fstar(k * L * 2 * math.cos(beta / 2) ** 2) / math.cos(beta / 2)
    D *= -np.exp(1j * math.pi / 4) / (2 * math.sqrt(2 * math.pi * k))
    ud = np.exp(1j * k * s1) / s1 * D * math.sqrt(s1 / (s * (s + s1))) * np.exp(1j * k * s)
    lit = phi < phi_s + math.pi
    R = math.hypot(rec[0] - src[0], rec[1] - src[1])
    ui = np.exp(1j * k * R) / R if lit else 0.0
    return ui + ud, R, s + s1


def fresnel_ke(lam, src, edge, rec):
    R = math.hypot(rec[0] - src[0], rec[1] - src[1])
    d = math.hypot(edge[0] - src[0], edge[1] - src[1]) + math.hypot(rec[0] - edge[0], rec[1] - edge[1])
    # edge above the line of sight: shadow
    zline = src[1] + (rec[1] - src[1]) * (edge[0] - src[0]) / (rec[0] - src[0])
    v = math.copysign(2 * math.sqrt((d - R) / lam), edge[1] - zline)
    S, C = fresnel(v)
    return 20 * math.log10(math.hypot(0.5 - C, 0.5 - S) / math.sqrt(2.0))


def main():
    freqs = [float(f) for f in sys.argv[1:]] or [250.0, 1000.0]
    zs, xb, hb = 1.0, 20.0, 5.0
    src, edge = (0.0, zs), (xb, hb)
    recs = [(21.0, 1.5), (22.0, 0.5), (22.0, 3.0), (25.0, 0.5), (25.0, 1.5), (30.0, 1.5),
            (40.0, 1.5), (60.0, 1.5), (100.0, 1.5), (100.0, 4.0), (200.0, 1.5)]
    print("screen at x=20 m, top 5 m, source 1 m, free space, still air")
    print(f"{'f':>6} {'x':>5} {'z':>4} {'angle':>6} {'pe':>7} {'fresnel':>7} {'hard':>7} {'soft':>7}"
          f" {'pe-hard':>7} {'pe-soft':>7} {'pe-mean':>7} {'ring':>5}")
    for f in freqs:
        lam, k = C0 / f, 2 * math.pi * f / C0
        p = march(f, C0, 0.0, zs, recs, screens=[(xb, hb)], zmin=-40.0, zmax=45.0)
        p0 = march(f, C0, 0.0, zs, recs, zmin=-40.0, zmax=45.0)
        pe = 20 * np.log10(np.abs(p) / np.abs(p0))
        for (x, z), L in zip(recs, pe):
            uh, R, path = utd_halfplane(k, src, edge, (x, z), True)
            us, _, _ = utd_halfplane(k, src, edge, (x, z), False)
            hard = 20 * math.log10(abs(uh) * R)
            soft = 20 * math.log10(abs(us) * R)
            mean = 10 * math.log10(0.5 * (abs(uh) ** 2 + abs(us) ** 2) * R * R)
            fk = fresnel_ke(lam, src, edge, (x, z))
            angle = math.degrees(math.atan2(hb - z, x - xb))
            ring = 10 * math.log10(path / x)
            print(f"{f:6.0f} {x:5.0f} {z:4.1f} {angle:6.1f} {L:7.2f} {fk:7.2f} {hard:7.2f} {soft:7.2f}"
                  f" {L - hard:7.2f} {L - soft:7.2f} {L - mean:7.2f} {ring:5.2f}")


if __name__ == "__main__":
    main()
