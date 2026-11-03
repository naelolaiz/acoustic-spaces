"""My own split-step Fourier PE (exact free-space square-root propagator, GFPE-style phase screen
for refraction), free space (no ground), linear effective sound speed c(z) = c0 + g z.
Thin screens: field set to zero below the top at the screen's range (Kirchhoff-like screen).
Rigid flat roofs: Neumann boundary at the roof height over the roof's range (method of images);
the front and rear faces cut the field below the roof.

p(r, z) = psi exp(i ka r) / sqrt(r);  time exp(-i w t). Levels are taken re the PE's own
free field (g = 0, no obstacles) at the same receiver, which removes the starter's normalisation.
"""

import math

import numpy as np


def _greene(z, zs, ka):
    u = ka * (z - zs)
    return math.sqrt(ka) * (1.4467 - 0.4201 * u * u) * np.exp(-u * u / 3.0512)


def march(f, c0, g, zs, receivers, screens=(), roofs=(), zmin=-30.0, zmax=45.0,
          dz_per_lambda=8, dr_max=0.25, dr_per_lambda=1.0, layer_lambda=40, layer_min=30.0):
    lam = c0 / f
    ka = 2 * math.pi * f / c0
    dz = lam / dz_per_lambda
    layer = max(layer_min, layer_lambda * lam)
    zlo, zhi = zmin - layer, zmax + layer
    N = 1 << int(math.ceil(math.log2((zhi - zlo) / dz)))
    dz = (zhi - zlo) / N
    z = zlo + dz * np.arange(N)
    kz = 2 * math.pi * np.fft.fftfreq(N, d=dz)
    kx = np.sqrt((ka * ka - kz * kz).astype(complex))
    kx = np.where(kx.imag < 0, np.conj(kx), kx)  # evanescent decays
    dk = 2 * math.pi * f / (c0 + g * z) - ka
    # sponge layers
    sponge = np.ones(N)
    top = z > zmax
    bot = z < zmin
    sponge[top] = np.exp(-((z[top] - zmax) / layer) ** 2 * 3.0)
    sponge[bot] = np.exp(-((zmin - z[bot]) / layer) ** 2 * 3.0)

    events = []
    for x, h in screens:
        events.append((x, "screen", h))
    for x0, x1, h in roofs:
        events.append((x0, "roof_start", h))
        events.append((x1, "roof_end", h))
    for k, (x, zr) in enumerate(receivers):
        events.append((x, "rec", (k, zr)))
    events.sort(key=lambda e: (e[0], {"screen": 0, "roof_start": 0, "roof_end": 1, "rec": 2}[e[1]]))

    psi = _greene(z, zs, ka).astype(complex)
    r = 0.0
    roof_h = None
    out = np.zeros(len(receivers), dtype=complex)
    dr_nom = min(dr_max, dr_per_lambda * lam)

    def step(psi, dr, roof_h):
        if roof_h is None:
            half = np.exp(0.5j * dr * dk)
            psi = half * psi
            psi = np.fft.ifft(np.fft.fft(psi) * np.exp(1j * dr * (kx - ka)))
            return sponge * half * psi
        # Neumann at roof_h by the method of images: even extension about roof_h.
        j0 = int(np.searchsorted(z, roof_h))
        zz = z[j0:]
        zim = 2 * roof_h - zz
        dk_up = dk[j0:]
        half = np.exp(0.5j * dr * dk_up)
        up = half * psi[j0:]
        full = np.zeros(N, dtype=complex)
        full[j0:] = up
        # image values on the grid below roof_h
        below = z < roof_h
        mirror = 2 * roof_h - z[below]
        full[below] = np.interp(mirror, zz, up.real) + 1j * np.interp(mirror, zz, up.imag)
        full = np.fft.ifft(np.fft.fft(full) * np.exp(1j * dr * (kx - ka)))
        res = np.zeros(N, dtype=complex)
        res[j0:] = sponge[j0:] * half * full[j0:]
        return res

    for x, kind, arg in events:
        while r < x - 1e-12:
            dr = min(dr_nom, x - r)
            psi = step(psi, dr, roof_h)
            r += dr
        if kind == "screen":
            psi[z < arg] = 0.0
        elif kind == "roof_start":
            psi[z < arg] = 0.0
            roof_h = arg
        elif kind == "roof_end":
            psi[z < arg] = 0.0
            roof_h = None
        else:
            k, zr = arg
            out[k] = (np.interp(zr, z, psi.real) + 1j * np.interp(zr, z, psi.imag)) * np.exp(
                1j * ka * x) / math.sqrt(x)
    return out


def level_re_free(f, c0, g, zs, receivers, **kw):
    """dB re the PE free field (g=0, no obstacles) at each receiver."""
    p = march(f, c0, g, zs, receivers, **kw)
    p0 = march(f, c0, 0.0, zs, receivers)
    return 20 * np.log10(np.abs(p) / np.abs(p0))
