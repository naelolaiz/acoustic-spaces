"""Ground impedance: the acoustic impedance of porous grounds (grass, soil, snow, gravel,
asphalt) and the reflection of a point source's sound off a ground that has one.

A ground's surface impedance follows from its flow resistivity sigma (kPa s/m², how hard
it is to push air through it) by an empirical model of porous media:

- ``miki`` (Miki 1990, the default): Delany and Bazley's model refitted so that it stays
  passive and causal at every frequency, as the wave solver needs;
- ``delany_bazley`` (Delany and Bazley 1970), as most of the outdoor literature quotes it;
- ``two_parameter`` (Attenborough 1985): a porosity that falls off with depth, from an
  effective flow resistivity and the rate at which the porosity changes (1/m), which
  fits measured grass and soil better than one parameter does.

Miki's and Delany and Bazley's models give a ground as deep as it likes; with a
``thickness`` d, a layer of it on a hard backing (snow on frozen ground, a thin soil over
rock), whose impedance is Zc coth(j kc d). :data:`PRESETS` are the eight ground classes
of Nord2000, A to H, with their flow resistivities, for Miki's model.

Those are the impedances of a locally reacting ground, met at normal incidence. In a
porous ground sound also travels sideways (extended reaction): a wave meeting the ground
at an angle theta from the normal goes on into it at the angle Snell's law gives, sin
theta_t = sin theta / n with n = kc / k the ground's complex index, and the surface
impedance it meets is Zc / cos theta_t = Zc n / sqrt(n² - sin² theta), for a layer (Zc
n / sqrt(n² - sin² theta)) coth(j k d sqrt(n² - sin² theta)) (Attenborough 1985; Li,
Attenborough and Heap 1998). With a large index (grass and harder grounds) that is the
normal incidence's at every angle; it tells for soft snow, moss and forest floors, whose
index is a few, and for thin layers. :func:`surface_impedance` gives it at an incidence
``cosine``; the two-parameter model, with no index, reacts locally.

Impedances here are normalised by the air's (rho c) and in the engine's sign convention,
e^(jωt), a delay of tau being e^(-j 2π f tau); the outdoor literature mostly writes
e^(-iωt), whose impedances are the complex conjugates of these.

A point source's spherical wave does not reflect off such a ground as a plane wave does,
R = (cos θ - β) / (cos θ + β), with β the admittance (1 / impedance) and θ the angle of
incidence: near grazing incidence a ground wave adds to it. :func:`spherical_reflection`
gives the Weyl-van der Pol coefficient Q = R + (1 - R) F(w), F(w) = 1 + i √π w e^(-w²)
erfc(-iw), with the numerical distance w² = (i k r / 2) (cos θ + β)² over the path's
length r (in e^(-iωt)): the image source's pressure times Q is the reflected sound. It
is the asymptotic form of the exact (Sommerfeld) solution for a locally reacting plane,
good to a fraction of a decibel once k r is more than a few.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

#: The ground classes of Nord2000 (A to H): flow resistivity (kPa s/m²) and what each
#: stands for. Representative values; real grounds spread over a factor of two or more.
PRESETS: dict[str, tuple[float, str]] = {
    "snow": (12.5, "very soft ground: new snow, moss (Nord2000 class A)"),
    "forest_floor": (31.5, "soft forest floor: short dense heather, thick moss (class B)"),
    "loose_soil": (80.0, "uncompacted loose ground: turf, loose soil (class C)"),
    "grass": (200.0, "normal uncompacted ground: pasture, lawns, forest floors (class D)"),
    "compacted_soil": (500.0, "compacted field and gravel: trodden lawns, parks (class E)"),
    "gravel": (2000.0, "compacted dense ground: gravel roads, car parks (class F)"),
    "asphalt": (20000.0, "hard surfaces: most asphalt and concrete (class G)"),
    "water": (200000.0, "very hard and dense surfaces: dense asphalt, concrete, water (class H)"),
}

#: The impedance models.
MODELS = ("miki", "delany_bazley", "two_parameter")


def characteristic(
    f: np.ndarray, flow_resistivity: float, model: str = "miki"
) -> tuple[np.ndarray, np.ndarray]:
    """The characteristic impedance (normalised) and the wavenumber over the air's of a
    porous medium of ``flow_resistivity`` (kPa s/m²) at frequencies ``f`` (Hz), by
    Miki's or Delany and Bazley's model."""
    x = np.asarray(f, dtype=float) / float(flow_resistivity)
    if model == "miki":
        z = 1.0 + 5.50 * x**-0.632 - 8.43j * x**-0.632
        k = 1.0 + 7.81 * x**-0.618 - 11.41j * x**-0.618
    elif model == "delany_bazley":
        z = 1.0 + 9.08 * x**-0.75 - 11.9j * x**-0.73
        k = 1.0 + 10.8 * x**-0.70 - 10.3j * x**-0.59
    else:
        raise ValueError(f"no characteristic impedance in the {model!r} model")
    return z, k


def surface_impedance(
    f: np.ndarray,
    flow_resistivity: float,
    model: str = "miki",
    porosity_rate: float = 0.0,
    thickness: float | None = None,
    speed_of_sound: float = 343.0,
    cosine=None,
) -> np.ndarray:
    """The normalised surface impedance of a ground at frequencies ``f`` (Hz): of
    ``flow_resistivity`` (kPa s/m²) by ``model``; with ``two_parameter``, that of an
    effective flow resistivity and a rate of change of porosity ``porosity_rate`` (1/m);
    with a ``thickness`` (m, not for ``two_parameter``), a layer on a hard backing. At
    normal incidence, a locally reacting ground's, unless the incidence ``cosine`` is
    given (broadcasting with ``f``; not for ``two_parameter``): then the extended
    reaction's at that angle, Zc n / sqrt(n² - sin² theta) and the layer's with it (see
    the module)."""
    f = np.asarray(f, dtype=float)
    if model == "two_parameter":
        if thickness is not None:
            raise ValueError(
                "the two-parameter model has no thickness: its porosity falls with depth"
            )
        if cosine is not None:
            raise ValueError("the two-parameter model reacts locally: it has no index")
        sigma = 1000.0 * float(flow_resistivity)  # Pa s/m²
        return 0.436 * (1.0 - 1.0j) * np.sqrt(sigma / f) - 19.48j * float(porosity_rate) / f
    z, k = characteristic(f, flow_resistivity, model)
    if cosine is not None:
        # Into the ground at Snell's angle: n cos theta_t = sqrt(n² - sin² theta), the
        # root that decays downwards (Im < 0, as k's), the principal one.
        sine2 = 1.0 - np.asarray(cosine, dtype=complex) ** 2
        down = np.sqrt(k * k - sine2)
        z = z * k / down
        k = down
    if thickness is None:
        return z
    wavenumber = 2.0 * np.pi * f / float(speed_of_sound) * k
    return z / np.tanh(1.0j * wavenumber * float(thickness))


def spherical_reflection(
    admittance: np.ndarray, cosine: np.ndarray, wavenumber: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    """The Weyl-van der Pol reflection factor Q of a point source's sound off a locally
    reacting ground of normalised ``admittance`` (the engine's convention), met at
    incidence cosine ``cosine`` by a path ``distance`` (m) long from the image source,
    at ``wavenumber`` (rad/m); the arguments broadcast. A hard ground (admittance 0)
    gives 1."""
    from scipy.special import wofz

    beta = np.conj(np.asarray(admittance, dtype=complex))  # e^(-iωt), as written above
    cos = np.clip(np.asarray(cosine, dtype=float), 0.0, 1.0)
    total = cos + beta
    hard = np.abs(total) < 1e-12  # a hard ground met at grazing incidence
    plane = np.where(hard, 1.0, (cos - beta) / np.where(hard, 1.0, total))
    w = 0.5 * (1.0 + 1.0j) * np.sqrt(np.asarray(wavenumber) * np.asarray(distance)) * total
    boundary = 1.0 + 1.0j * math.sqrt(math.pi) * w * wofz(w)
    return np.conj(plane + (1.0 - plane) * boundary)


def plane_reflection(admittance: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    """The plane wave's reflection factor R = (cos θ - β) / (cos θ + β) off a locally
    reacting surface of normalised ``admittance``, met at incidence cosine ``cosine``;
    1 for a hard surface met at grazing incidence."""
    beta = np.asarray(admittance, dtype=complex)
    cos = np.clip(np.asarray(cosine, dtype=float), 0.0, 1.0)
    total = cos + beta
    hard = np.abs(total) < 1e-12
    return np.where(hard, 1.0, (cos - beta) / np.where(hard, 1.0, total))


def band_absorption(admittance: Callable[[np.ndarray], np.ndarray], bands: int) -> list[float]:
    """The random-incidence absorption (Paris's formula) of a surface whose admittance at
    frequencies f is ``admittance(f)``, in each of ``bands`` (8 octave or 24
    third-octave) bands, through the analysis's band filters."""
    from acoustic_spaces.engine.impedance import (
        _frequencies,
        band_weights,
        random_incidence_absorption,
    )

    f = _frequencies(bands, bands)
    alpha = band_weights(f, bands, bands) @ random_incidence_absorption(admittance(f))
    return [float(a) for a in np.clip(alpha, 0.0, 1.0)]


__all__ = [
    "MODELS",
    "PRESETS",
    "band_absorption",
    "characteristic",
    "plane_reflection",
    "spherical_reflection",
    "surface_impedance",
]
