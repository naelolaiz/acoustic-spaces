"""Full-wave reference (probe only), independent of the PE: axisymmetric Helmholtz equation
in a medium whose (effective) speed of sound depends on height, with thin rigid ring screens
and rigid flat-topped ring blocks, by mode matching (stepwise coupled modes, as in Evans 1983,
JASA 74, 188, "COUPLE"): exact in range (Hankel and Bessel functions), finite volumes in height
(cell centres z_j = (j + 1/2) h) with a PML at the top.

    (1/r) d/dr (r dp/dr) + d2p/dz2 + k(z)^2 p = -delta(point source on the axis at zs)

p -> exp(ikR) / (4 pi R) in free field (time exp(-i w t)). Ground at z = 0: rigid, or a
locally reacting normalised impedance Z (dp/dz = -i k p / Z). A thin screen at range b blocks
the radial flux for the cells below its top: dp/dr = 0 on both faces there; above, p and dp/dr
are continuous (pointwise at the cell centres). A block (b0, b1, top): between b0 and b1 the
field lives on the cells above the top, over a rigid floor (its own modes); its faces are
rigid. Tops are snapped to cell faces (multiples of h).

No approximation besides the height discretisation (second order, h small) and the PML.
The sound a screen reflects back towards the source travels in as H0^(2) and is not sent out
again from the axis (an axisymmetric ring would focus it on the axis and send it back out,
3-8 dB too loud behind a ring at 63-250 Hz; a straight barrier sends it away): between the
source and the first wall the field is the source's H0^(1) plus an incoming H0^(2); between
two walls H0^(1) (normalised at the left wall) plus H0^(2) (at the right wall).
"""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg as la
from scipy import special as sps


class Modes:
    def __init__(self, f, speed, h, Nz, j0=0, ground_Z=None, zmax=None, L=None, c_ref=None,
                 layer=False):
        """Modes on cells j0..Nz-1 (floor at face j0: the ground if j0 == 0 with ground_Z,
        else rigid)."""
        w = 2 * math.pi * f
        z = (np.arange(Nz) + 0.5) * h
        zf = np.arange(Nz + 1) * h
        smax = 20.7 * c_ref / L

        def sig(x):
            d = np.clip((x - zmax) / L, 0.0, None)
            return smax * d**2

        sc = 1 + 1j * sig(z) / w
        sf = 1 + 1j * sig(zf) / w
        k2 = (w / speed(z)) ** 2
        if layer:
            # an absorbing layer instead of the PML: k^2 (1 + i eta), eta quadratic over L,
            # k eta_max L / 3 = 15 (two passes: -130 dB); T stays real symmetric.
            sc = np.ones_like(sc)
            sf = np.ones_like(sf)
            kref = w / c_ref
            eta = 45.0 / (kref * L) * np.clip((z - zmax) / L, 0.0, None) ** 2
            k2 = k2 * (1 + 1j * eta)
        g = 1.0 / (h * sf)  # face conductance / (1/h)
        n = Nz - j0
        T = np.zeros((n, n), dtype=complex)
        idx = np.arange(n)
        jj = idx + j0
        up = g[jj + 1] / h
        down = g[jj] / h
        down[0] = 0.0  # floor
        T[idx, idx] = -(up + down)
        T[idx[:-1], idx[:-1] + 1] = up[:-1]
        T[idx[:-1] + 1, idx[:-1]] = up[:-1]
        if j0 == 0 and ground_Z is not None:
            k0 = w / float(speed(np.array([0.0]))[0])
            T[0, 0] += 1j * k0 / (ground_Z * h)
        S = sc[jj]
        A = T + np.diag(S * k2[jj])
        Wh = 1.0 / np.sqrt(S)
        At = Wh[:, None] * A * Wh[None, :]
        mu, psi = la.eig(At)
        phi = Wh[:, None] * psi
        N = np.sum(psi * psi, axis=0)  # phi^T S phi
        kap = np.sqrt(mu.astype(complex))
        # outgoing: Im kappa >= 0; a trapped mode (downwind duct) has a real kappa whose
        # imaginary part is round-off of either sign: keep Re kappa > 0 there.
        kap = np.where(kap.imag < -1e-9 * np.abs(kap), -kap, kap)
        self.kap, self.phi, self.N, self.S, self.j0, self.z = kap, phi, N, S, j0, z[jj]

    def ratio_J(self, b):  # incoming wave H0^(2): -kappa H1^(2)/H0^(2) at b
        x = self.kap * b
        return -self.kap * sps.hankel2e(1, x) / sps.hankel2e(0, x)

    def ratio_H(self, b):
        x = self.kap * b
        return -self.kap * sps.hankel1e(1, x) / sps.hankel1e(0, x)

    def Hrel(self, r, b):  # H0(k r) / H0(k b)
        return sps.hankel1e(0, self.kap * r) / sps.hankel1e(0, self.kap * b) * np.exp(
            1j * self.kap * (r - b))

    def Jrel(self, r, b):  # H0^(2)(k r) / H0^(2)(k b): the wave going in, towards the axis
        return sps.hankel2e(0, self.kap * r) / sps.hankel2e(0, self.kap * b) * np.exp(
            -1j * self.kap * (r - b))


def solve(f, speed, zs, receivers, zmax, ppw=30, screens=(), blocks=(), ground_Z=None,
          pml_wavelengths=3.0, c_ref=None, h=None, want_line=False, layer=False):
    """Complex pressure at receivers [(r, z)], convention exp(ikR)/(4 pi R).
    screens [(b, top)], blocks [(b0, b1, top)] (non-overlapping)."""
    if c_ref is None:
        c_ref = float(np.min(speed(np.linspace(0, zmax, 400))))
    lam = c_ref / f
    if h is None:
        h = lam / ppw
    L = (12.0 if layer else pml_wavelengths) * lam
    Nz = int(math.ceil((zmax + L) / h))
    common = dict(ground_Z=ground_Z, zmax=zmax, L=L, c_ref=c_ref, layer=layer)
    base = Modes(f, speed, h, Nz, 0, **common)
    # interfaces in range order: (b, kind, jt) ; regions between them
    walls = []
    for b, top in screens:
        walls.append((b, "screen", int(round(top / h))))
    for b0, b1, top in blocks:
        walls.append((b0, "front", int(round(top / h))))
        walls.append((b1, "back", int(round(top / h))))
    walls.sort()
    # regions: list of Modes; region i spans walls[i-1].b .. walls[i].b
    regions = [base]
    for b, kind, jt in walls:
        if kind == "front":
            regions.append(Modes(f, speed, h, Nz, jt, **common))
        else:
            regions.append(base)
    R = len(regions)
    # unknowns: region 0: J coefficients (n0) ; inner region i: H (at left) + J (at right) ;
    # last: H.
    sizes = []
    for i, m in enumerate(regions):
        n = len(m.kap)
        if i == 0:
            sizes.append(("J", n))
        elif i == R - 1:
            sizes.append(("H", n))
        else:
            sizes.append(("HJ", n))
    offs = []
    o = 0
    for kind, n in sizes:
        offs.append(o)
        o += n if kind != "HJ" else 2 * n
    total = o
    # source expansion in region 0
    t = zs / h - 0.5
    js = int(math.floor(t))
    wgt = t - js
    d = np.zeros(Nz, dtype=complex)
    d[js] = (1 - wgt) / h
    d[js + 1] = wgt / h
    alpha = (base.phi.T @ (base.S * d)) / base.N  # coefficient of (i/4) H0
    rows = []
    rhs = []

    def field_terms(i, b, deriv):
        """For region i at range b: list of (column offset, coefficient vector per mode) and
        the known part (vector over modes, region 0 incident only)."""
        m = regions[i]
        kind, n = sizes[i]
        terms = []
        if kind == "J":
            terms.append((offs[i], m.ratio_J(b) if deriv else np.ones(n, dtype=complex)))
        elif kind == "H":
            terms.append((offs[i], m.ratio_H(b) if deriv else np.ones(n, dtype=complex)))
        else:
            bl = walls[i - 1][0]
            br = walls[i][0]
            # H normalised at the left wall, J at the right wall
            Hr = m.Hrel(b, bl)
            Jr = m.Jrel(b, br)
            if deriv:
                terms.append((offs[i], Hr * m.ratio_H(b)))
                terms.append((offs[i] + n, Jr * m.ratio_J(b)))
            else:
                terms.append((offs[i], Hr))
                terms.append((offs[i] + n, Jr))
        known = None
        if i == 0:
            x = m.kap * b
            H0 = sps.hankel1(0, x)
            H1 = sps.hankel1(1, x)
            known = 0.25j * alpha * (-m.kap * H1 if deriv else H0)
        return terms, known

    for w, (b, kind, jt) in enumerate(walls):
        left, right = w, w + 1
        ml, mr = regions[left], regions[right]
        # nodes of the full grid
        for side_eq in ("value", "deriv"):
            deriv = side_eq == "deriv"
            tl, kl = field_terms(left, b, deriv)
            tr, kr = field_terms(right, b, deriv)
            for j in range(Nz):
                open_ = j >= jt
                inl = j >= ml.j0
                inr = j >= mr.j0
                row = np.zeros(total, dtype=complex)
                val = 0.0
                if open_ and inl and inr:
                    # continuity: left - right = 0
                    for off, coef in tl:
                        row[off:off + len(coef)] += ml.phi[j - ml.j0] * coef
                    for off, coef in tr:
                        row[off:off + len(coef)] -= mr.phi[j - mr.j0] * coef
                    if kl is not None:
                        val -= ml.phi[j - ml.j0] @ kl
                    rows.append(row); rhs.append(val)
                elif deriv:
                    # rigid face on each side where that side has the node
                    if inl:
                        row = np.zeros(total, dtype=complex)
                        val = 0.0
                        for off, coef in tl:
                            row[off:off + len(coef)] += ml.phi[j - ml.j0] * coef
                        if kl is not None:
                            val -= ml.phi[j - ml.j0] @ kl
                        rows.append(row); rhs.append(val)
                    if inr:
                        row = np.zeros(total, dtype=complex)
                        for off, coef in tr:
                            row[off:off + len(coef)] += mr.phi[j - mr.j0] * coef
                        rows.append(row); rhs.append(0.0)
    if walls:
        M = np.array(rows)
        y = np.array(rhs, dtype=complex)
        if M.shape[0] != M.shape[1]:
            raise RuntimeError(f"system {M.shape}")
        X = la.solve(M, y)
    else:
        X = np.zeros(total, dtype=complex)
    bounds = [0.0] + [w[0] for w in walls] + [np.inf]

    def coef_at(r):
        i = int(np.searchsorted(bounds, r, side="right") - 1)
        m = regions[i]
        kind, n = sizes[i]
        if kind == "J":
            coef = X[offs[i]:offs[i] + n] * m.Jrel(r, bounds[1]) if walls else 0.0
            coef = coef + 0.25j * alpha * sps.hankel1(0, m.kap * r)
        elif kind == "H":
            coef = X[offs[i]:offs[i] + n] * m.Hrel(r, bounds[i])
        else:
            coef = (X[offs[i]:offs[i] + n] * m.Hrel(r, bounds[i])
                    + X[offs[i] + n:offs[i] + 2 * n] * m.Jrel(r, bounds[i + 1]))
        return m, coef

    def line(r, zz):
        """p at range r and heights zz (array)."""
        m, coef = coef_at(r)
        zz = np.asarray(zz, dtype=float)
        vals = np.empty((len(zz), len(m.kap)), dtype=complex)
        for q in range(len(m.kap)):
            vals[:, q] = np.interp(zz, m.z, m.phi[:, q].real) + 1j * np.interp(zz, m.z, m.phi[:, q].imag)
        return vals @ coef

    out = [line(r, [zr])[0] for r, zr in receivers]
    if want_line:
        return np.array(out), dict(h=h, Nz=Nz, line=line)
    return np.array(out), dict(h=h, Nz=Nz, screens=[(b, jt * h) for b, k, jt in walls if k == "screen"],
                               blocks=[(w[0], w[2] * h) for w in walls if w[1] != "screen"])
