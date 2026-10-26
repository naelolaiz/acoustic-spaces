"""Sound over the crests of a terrain.

A terrain (``geometry.ground.terrain``) is a grid of heights cut into triangles, and a
hill in it bends only a degree or so at each grid line: no edge of the mesh diffracts
the sound over it, so a listener behind the hill would hear nothing of the direct
sound. Outdoor standards (ISO 9613-2, CNOSSOS-EU, Nord2000) take the ground's profile
in the vertical plane between the source and the listener instead, and bend the sound
at the crests the straight line between them cuts through. This module does the same,
with the wedge diffraction the engine uses for buildings
(:mod:`acoustic_spaces.engine.diffraction`):

- the line between the source and the listener is drawn over the terrain's profile
  (:meth:`~acoustic_spaces.geometry.outdoor.TerrainSurface.profile`), and the string
  pulled taut over the profile from one to the other (the upper convex hull of the
  profile in that plane) bends at the crests in the way;
- each crest is a wedge whose faces run from it down to the ground under its
  neighbours on the string: the source's foot, the listener's, or the next crest. A
  ridge of two slopes is then the wedge it is; a rounded hill is the wedge of its
  chords, as Nord2000 has it. The sound over the crests is the uniform theory of
  diffraction's, wedge after wedge along the string: the direct wave over each edge,
  and the ground's reflections before and after it. On each side the neighbour's
  images in the planes of the profile's segments between it and the crest carry
  those, each image weighted by the share of its Fresnel zone the segment holds
  (Nord2000's Fresnel-zone weights: at low frequencies the zone spans the whole side
  and the images add up to the one reflection off its mean plane, at high ones the
  segments that mirror the sound to the edge carry it all), with the ground's
  reflection factor (the spherical wave's over a ground with an impedance, with the
  ground wave that makes the ground dip) and the extra path of the reflection over
  the straight line to the edge. An image's wave is the wedge's term for the
  reflection off its face, with the reflection factor as Luebbers has it, which is the
  exact solution's where the reflection reaches the edge along the face, and grows
  the other term of a wave of its own, and the double reflection off both sides, as
  the uniform theory's transition over that extra path lets it, so that far from the
  crest the ground's reflections are the separate paths the four-path model of a
  barrier takes them for. Where the string runs along the ground from one crest to the
  next (a flat top, a rounded crest) the wedges share that face, and the wave along it is
  taken once (half the coefficient at the second), as the engine takes it over a thick
  wall;
- the path over the crests arrives at the listener from the last crest, after the
  string's length, with the air's attenuation over it. A crest the string runs along
  screens the sound more the wider it is, by ISO 9613-2's factor for a thick screen:
  as much as a thin one at the edge of its shadow, up to three times (4.8 dB) more
  deep in it;
- where the listener sees the source, the crest the ground comes nearest the line of
  sight between them still diffracts: its wedge's waves are added to the direct sound
  and to the reflection off the slope before the crest where the listener sees it,
  with the uniform theory's transition at each boundary, so that the level is
  continuous across the edge of the shadow (as it is over a building's edge,
  :class:`~acoustic_spaces.engine.diffraction.EdgeDiffraction`). The reflection off
  the mean plane of the whole profile (:mod:`acoustic_spaces.engine.terrain`), the
  standards' ground reflection in sight, takes over from the slope's as the crest
  drops below the line of sight, by half the longest wavelength of path difference;
- ground that comes within the first Fresnel zone of a straight stretch of the string
  between two crests (N above -0.2, between the stretch's ends) screens the sound a
  little, by Kurze and Anderson's fit to the knife edge's solution.

A ridge agrees with the exact solution for the wedge within a fraction of a decibel
(see ``validation/terrain.py``); a smooth crest, which the sound creeps over, has no
exact solution and is an engineering model here as in the standards. Buildings,
barriers or model objects standing on the string stop the path: the sound bent over a
hill and a building in turn is not modelled. Nothing of this touches a scene without a
terrain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from acoustic_spaces.directivity import Directivity
from acoustic_spaces.engine.coherent import ground_factors, reflection_phases
from acoustic_spaces.engine.diffraction import (
    _ON_BOUNDARY,
    _clear,
    _share,
    band_frequencies,
    boundary_angles,
    transition,
    wedge_prefactor,
    wedge_terms,
)
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.model import TraceModel, _ground_faces, source_weights
from acoustic_spaces.engine.terrain import (
    GroundReflection,
    ground_face_under,
    own_cosine,
    reflection_factor,
)

#: Fresnel number below which a crest in the lit zone no longer screens the sound.
LIT_LIMIT = -0.2
#: Marks a path that bends over the terrain's crests in ``EarlyReflections.edge``.
CREST = -2
#: Ground within this (m) of the string between two of its corners lies on it.
_LIFT = 0.01
#: The longest wavelength in the lowest band (m, at 20 Hz): a crest farther below the
#: line of sight than half of it in path difference bends nothing in any band, and
#: the mean plane's reflection stands in for the slope's from there.
LARGEST_WAVELENGTH = 17.5
#: A point this close (rad) to a wedge's face, seen from its edge, lies on it.
_GRAZING = 1e-7
_TINY = 1e-9


def knife_edge_loss(fresnel: np.ndarray) -> np.ndarray:
    """Kurze and Anderson's insertion loss (dB) of a thin screen for Fresnel numbers
    ``fresnel`` (2 δ / λ, with the path difference δ negative in the lit zone): 5 + 20
    log(x / tanh x) with x = sqrt(2 π N) in the shadow, 5 + 20 log(x / tan x) with
    x = sqrt(-2 π N) in the lit zone down to :data:`LIT_LIMIT`, and nothing below."""
    n = np.asarray(fresnel, dtype=float)
    x = np.sqrt(2.0 * np.pi * np.abs(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        shadow = np.where(x > _TINY, x / np.tanh(np.maximum(x, _TINY)), 1.0)
        lit = np.where(x > _TINY, x / np.tan(np.minimum(x, 1.5)), 1.0)
    ratio = np.where(n >= 0.0, shadow, lit)
    loss = 5.0 + 20.0 * np.log10(np.maximum(ratio, 1e-12))
    return np.where(n > LIT_LIMIT, np.maximum(loss, 0.0), 0.0)


def taut_string(s: np.ndarray, z: np.ndarray) -> list[int]:
    """The indices of the corners of the upper convex hull of the profile ``(s, z)``
    (``s`` increasing), its ends included: a string pulled taut over the profile from
    its first point to its last bends at the corners between them."""
    hull: list[int] = []
    for i in range(len(s)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Pop b when it lies on or below the line from a to i.
            cross = (s[b] - s[a]) * (z[i] - z[a]) - (z[b] - z[a]) * (s[i] - s[a])
            if cross >= -_TINY * max(1.0, abs(s[i] - s[a])):
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


@dataclass
class CrestDiffraction:
    """The sound of one source over the crests of the terrain, for any number of
    listeners: :meth:`apply` adds it to the paths of one listener. ``air`` is the
    energy attenuation per metre per band; ``directivity`` and ``orientation`` the
    source's, as for :func:`~acoustic_spaces.engine.early.find_paths`; in the lowest
    ``coherent`` bands the path carries its pressure spectrum."""

    model: TraceModel
    source: np.ndarray
    speed_of_sound: float
    air: np.ndarray
    directivity: Directivity | None = None
    orientation: np.ndarray | None = None
    coherent: int = 0

    @classmethod
    def of(cls, model: TraceModel, source, **settings) -> CrestDiffraction | None:
        """The crest diffraction of ``model``, or None when it has no terrain."""
        layout = model.room.outdoor
        if layout is None or layout.terrain is None:
            return None
        return cls(model, np.asarray(source, dtype=float), **settings)

    def __post_init__(self) -> None:
        self.source = np.asarray(self.source, dtype=float)
        self.air = np.asarray(self.air, dtype=float)
        self.frequencies, self.slice_hz = band_frequencies(bands=self.model.bands)  # (B, M)
        self.wavelength = self.speed_of_sound / self.frequencies
        self.wavenumber = 2.0 * np.pi / self.wavelength
        self.ground = _ground_faces(self.model.room)

    def loss(self, path_difference: float) -> np.ndarray:
        """(bands, M) the loss (dB) at each frequency of a thin screen with this path
        difference (m; negative when it lies below the line of sight)."""
        return knife_edge_loss(2.0 * path_difference / self.wavelength)

    def apply(
        self, listener: np.ndarray, paths: EarlyReflections, max_delay_s: float
    ) -> EarlyReflections:
        """``paths`` (to ``listener``) with the sound over the terrain's crests: the
        path over them added when they hide the source, or the wave of the crest
        nearest the line of sight when they do not."""
        listener = np.asarray(listener, dtype=float)
        surface = self.model.room.outdoor.terrain
        source = self.source
        if float(np.linalg.norm(listener[:2] - source[:2])) <= _TINY:
            return paths
        profile = surface.profile(source[:2], listener[:2])
        if len(profile) < 3:
            return paths
        s = np.linalg.norm(profile[:, :2] - source[:2], axis=1)
        ground_z = profile[:, 2]
        z = ground_z.copy()
        # The ends of the string are the source and the listener themselves.
        z[0], z[-1] = source[2], listener[2]
        points = profile.copy()
        points[0], points[-1] = source, listener
        corners = taut_string(s, z)
        ground = self.ground
        # The direct sound and the ground's reflection, not the buildings'.
        grounds = [
            k
            for k in range(len(paths))
            if paths.edge[k] == -1 and ground[np.asarray(paths.faces[k], np.int64)].all()
        ]
        straight = [k for k in grounds if paths.order[k] == 0]
        if len(corners) == 2 and not straight:
            # The string clears the ground but the direct sound did not get through:
            # the line of sight grazes a crest. Bend the sound there.
            line = z[0] + (z[-1] - z[0]) * s / max(s[-1], _TINY)
            nearest = int(np.argmin(line - ground_z))
            if 0 < nearest < len(s) - 1:
                corners = [0, nearest, len(s) - 1]
        if len(corners) == 2:
            found = self._near(points, ground_z, s, 0, len(s) - 1)
            if found is None or not straight:
                return paths
            return self._lit(listener, points, ground_z, s, found, paths, straight[0], grounds)
        # The ground that comes near the string between the crests screens the sound a
        # little.
        near = np.zeros_like(self.wavelength)
        for i, j in zip(corners, corners[1:], strict=False):
            if _on_ground(s, z, i, j):
                continue
            found = self._near(points, ground_z, s, i, j)
            if found is not None:
                near += self.loss(found[1])
        bands = self.model.absorption.shape[1]
        over = self._over(listener, points, ground_z, s, z, corners, near, max_delay_s)
        parts = [paths, *over]
        return EarlyReflections.concatenate(parts, bands)

    def _lit(self, listener, points, ground_z, s, found, paths, straight: int, grounds):
        """``paths`` with the wave of the crest ``found`` (its index in the profile
        ``points`` over ``ground_z``, and the path difference over it, negative) when
        the listener sees the source: the wedge's waves with the transition at each
        boundary, the reflections off the ground on either side of the crest where the
        listener sees the source's image in it, and their cross terms with the paths
        they continue (``straight`` the direct sound's row; ``grounds`` the rows of the
        direct sound and the ground's reflection). The reflection off the mean plane
        fades in as the crest drops below the line of sight."""
        model, source = self.model, self.source
        bands = model.absorption.shape[1]
        nearest, delta = found
        ends = [0, nearest, len(points) - 1]
        chain = points[ends]
        legs = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        distance = float(legs.sum())
        if np.any(legs <= _TINY):
            return paths
        # How much of the wedge counts: all of it at the edge of the shadow, nothing
        # once the crest lies half the longest wavelength of path difference below the
        # line of sight, smoothly in between.
        t = min(-delta / (0.5 * LARGEST_WAVELENGTH), 1.0)
        weight = 1.0 - t * t * (3.0 - 2.0 * t)
        if weight <= 0.0:
            return paths
        along = (listener[:2] - source[:2]) / max(float(s[-1]), _TINY)
        chain_s = s[ends]
        w = self._wedge(chain, ground_z[ends], chain_s, 1, along)
        k = self.wavenumber  # (B, M)
        kl = k * w["m"] * w["l"] / (w["m"] + w["l"])
        angle = boundary_angles(w["n"], w["phi"], w["phi_s"])  # (3,)
        # The paths the waves continue: the direct sound, and the reflections off the
        # ground on each side where the listener sees the source's image in it. The
        # mean plane's reflection fades out as the wedge's reflections fade in.
        energy = paths.energy.copy()
        for row in grounds:
            if paths.order[row] > 0:
                energy[row] *= 1.0 - weight
        paths = replace(paths, energy=energy)
        # The waves: the direct one, then one per image of the source in the ground
        # before the crest and of the listener in the ground after it, each with the
        # angle of its boundary and the path it continues, where there is one.
        images = [self._images(w, chain, chain_s, s, ground_z, 1, side, along) for side in (0, 1)]
        sides = [side for side in (0, 1) for _ in images[side]]
        found_images = images[0] + images[1]
        extra = [None]
        for image in found_images:
            path = self._image_path(image, source, listener, s, ground_z, along)
            extra.append(None if path is None else replace(path, energy=path.energy * weight))
        pairs = list(zip(found_images, sides, strict=True))
        doubles = [(a, b) for a in images[0] for b in images[1]]
        bounds = [angle[0]] + [self._bound(w, image, side) for image, side in pairs]
        bounds += [self._bound(w, a, 0, b) for a, b in doubles]
        bounds = np.array(bounds)
        extra += [None] * len(doubles)
        has = np.array([True] + [path is not None for path in extra[1:]])
        lit = (bounds > _ON_BOUNDARY) | ((np.abs(bounds) <= _ON_BOUNDARY) & has)
        counted = lit == has
        rows = [self._direct_wave(w, bool(lit[0]), kl)]
        for r, (image, side) in enumerate(pairs, start=1):
            rows.append(
                self._image_wave(w, image, side, bool(lit[r]), w["m"], w["l"], legs, distance)
            )
        for r, (a, b) in enumerate(doubles, start=len(rows)):
            rows.append(self._double_wave(w, a, b, bool(lit[r]), w["m"], w["l"], legs, distance))
        factors = [np.ones_like(k, dtype=complex)] + [image["factor"] for image in found_images]
        factors += [a["factor"] * b["factor"] for a, b in doubles]
        detours = [0.0] + [image["extra"] for image, _ in pairs]
        detours += [a["extra"] + b["extra"] for a, b in doubles]
        waves = self._merged(w, np.stack(rows), len(images[0])) * counted[:, None, None]
        departure = (chain[1] - chain[0]) / legs[0]
        strength = source_weights(
            self.directivity, self.orientation, departure, bands, distances=legs[:1]
        )[0]
        amplitude = np.sqrt(weight * strength * np.exp(-self.air * distance) / (4.0 * np.pi))
        amplitude /= math.sqrt(float(np.prod(legs)) * distance)
        waves = waves * (wedge_prefactor(w["n"], k, 1.0) * amplitude[:, None])[None]
        coherence = self._coherence(self._shares(kl, bounds), detours)
        energy = np.einsum("ijbm,ibm,jbm->bm", coherence, waves, np.conj(waves)).real.mean(axis=1)
        # Each wave interferes with the path it continues, which arrives at almost the
        # same time: the path takes the cross term with the wave, as over a building's
        # edge; in the coherent bands the two add in pressure instead. Where a cross
        # term takes more than its path has, the rest comes off the wave.
        for i in range(len(rows)):
            if not (lit[i] and has[i]):
                continue
            path = paths if i == 0 else extra[i]
            row = straight if i == 0 else 0
            reach = float(path.distance_m[row])
            phase = np.exp(-1j * k * (reach - distance))
            size = np.abs(factors[i])  # the path's pressure has its reflection's phase
            shape = np.where(size > 0.0, factors[i] / np.where(size > 0.0, size, 1.0), 1.0)
            specular = np.sqrt(path.energy[row])[:, None] * shape * phase
            fade = np.sinc((reach - distance) / self.speed_of_sound * self.slice_hz)
            cross = 2.0 * fade * np.mean(np.real(specular * np.conj(waves[i])), axis=1)
            cross[: self.coherent] = 0.0
            total = path.energy[row] + cross
            energy = np.maximum(energy + np.minimum(total, 0.0), 0.0)
            taken = path.energy.copy()
            taken[row] = np.maximum(total, 0.0)
            if i == 0:
                paths = replace(paths, energy=taken)
            else:
                extra[i] = replace(path, energy=taken)
        spectrum = None
        if self.coherent:
            low = waves.sum(axis=0)[: self.coherent]
            level = np.sqrt(np.mean(np.abs(low) ** 2, axis=1))[:, None]
            spectrum = np.where(level > 0.0, low / np.where(level > 0.0, level, 1.0), 1.0)[None]
        arrival = (chain[1] - listener) / legs[1]
        wave = EarlyReflections(
            order=np.zeros(1, dtype=np.int64),
            delay_s=np.array([distance / self.speed_of_sound]),
            distance_m=np.array([distance]),
            direction=arrival[None],
            energy=energy[None],
            points=[chain],
            faces=[np.zeros(0, dtype=np.int64)],
            edge=np.array([CREST], dtype=np.int64),
            spectrum=spectrum,
        )
        parts = [paths, *[p for p in extra if p is not None], wave]
        return EarlyReflections.concatenate(parts, bands)

    def _over(self, listener, points, ground_z, s, z, corners, near, max_delay_s):
        """The path over the crests of the string with ``corners`` (indices into the
        profile ``points``, over ``ground_z``, the string at heights ``z``; the source
        first, the listener last): the wedges' waves in turn, as the module notes have
        it; and the reflections off the ground after a lone crest that the source sees
        over it. Nothing when something stands in the way or the path comes too late.
        ``near`` is the loss (bands, M) the ground near the straight stretches adds."""
        model, source = self.model, self.source
        bands = model.absorption.shape[1]
        string = points[corners]
        distance = float(np.linalg.norm(np.diff(string, axis=0), axis=1).sum())
        if distance / self.speed_of_sound > max_delay_s:
            return []
        if not _clear(model, string[:-1], string[1:], through=self.ground).all():
            return []
        # One wedge per crest: a lone corner's, or, where the string lies along a flat
        # or rounded top from one corner to another, the wedge of the string's two
        # legs onto it extended to where they meet above it (the equivalent edge of
        # a thick screen), which moves smoothly as the string lifts off a corner; the
        # crest is as wide as the ground within an eighth of a wavelength of the string.
        groups = _crests(corners, s, z)
        gap = np.maximum(np.interp(s, s[corners], z[corners]) - z, 0.0)
        chain_s, chain_z, tops = [s[0]], [z[0]], []
        for a, b in groups:
            first, last = corners[a], corners[b]
            apex_s, apex_z = _apex(
                (s[corners[a - 1]], z[corners[a - 1]]),
                (s[first], z[first]),
                (s[last], z[last]),
                (s[corners[b + 1]], z[corners[b + 1]]),
            )
            chain_s.append(apex_s)
            chain_z.append(apex_z)
            bends = [_bend(string[p - 1], string[p], string[p + 1]) for p in range(a, b + 1)]
            tops.append(corners[a + int(np.argmax(bends))])
        chain_s.append(s[-1])
        chain_z.append(z[-1])
        chain_s = np.asarray(chain_s)
        along = (listener[:2] - source[:2]) / max(float(s[-1]), _TINY)
        chain = np.column_stack([source[:2] + np.outer(chain_s, along), chain_z])
        chain[0], chain[-1] = source, listener
        feet = np.interp(chain_s, s, ground_z)
        # How fast the ground of a sharp wedge with the chain's legs would fall away
        # from the string on each side of each crest: the leg's end's height over the
        # ground, over the leg's run.
        lifts = np.maximum(np.asarray(chain_z) - feet, 0.0)
        runs = np.maximum(np.diff(chain_s), _TINY)
        tops = [
            self._top_width(gap, s, c, (lifts[j] / runs[j], lifts[j + 2] / runs[j + 1]))
            for j, c in enumerate(tops)
        ]
        legs = np.linalg.norm(np.diff(chain, axis=0), axis=1)
        if np.any(legs <= _TINY):
            return []
        k = self.wavenumber  # (B, M)
        count = len(groups)
        wedges = [self._wedge(chain, feet, chain_s, j, along) for j in range(1, count + 1)]
        angles = [boundary_angles(w["n"], w["phi"], w["phi_s"]) for w in wedges]
        # How near each wedge's wave is to a boundary, 0 on one and 1 well off it, by
        # its distance parameter from the legs alone.
        nearness = []
        for w, angle in zip(wedges, angles, strict=True):
            kl = k * w["m"] * w["l"] / (w["m"] + w["l"])
            nearness.append(self._shares(kl, angle).min(axis=0))
        before, after = self._radii(legs, nearness)
        pressure = np.ones_like(k, dtype=complex)
        strength = np.ones_like(k)
        extra = []
        for j, (w, angle) in enumerate(zip(wedges, angles, strict=True)):
            kl = k * before[j] * after[j] / (before[j] + after[j])
            # The waves: the direct one, then one per image of the point before the
            # crest in the ground before it and of the point after in the ground after.
            images = [
                self._images(w, chain, chain_s, s, ground_z, j + 1, side, along) for side in (0, 1)
            ]
            rows = [self._direct_wave(w, bool(angle[0] > _ON_BOUNDARY), kl)]
            bounds, detours = [float(angle[0])], [0.0]
            for side in (0, 1):
                for image in images[side]:
                    bound = self._bound(w, image, side)
                    detours.append(image["extra"])
                    lit = bound > _ON_BOUNDARY
                    wave = self._image_wave(
                        w, image, side, lit, before[j], after[j], legs[j : j + 2], distance
                    )
                    if lit and side == 1:
                        # The source sees the ground after the crest over it: its
                        # reflection reaches the listener as a path of its own, which
                        # the wedge's wave continues.
                        path = None
                        if count == 1:
                            path = self._image_path(image, source, listener, s, ground_z, along)
                        if path is None:
                            wave = np.zeros_like(wave)
                        else:
                            extra.append(path)
                    rows.append(wave)
                    bounds.append(bound)
            for a in images[0]:
                for b in images[1]:
                    bound = self._bound(w, a, 0, b)
                    detours.append(a["extra"] + b["extra"])
                    rows.append(
                        self._double_wave(
                            w,
                            a,
                            b,
                            bound > _ON_BOUNDARY,
                            before[j],
                            after[j],
                            legs[j : j + 2],
                            distance,
                        )
                    )
                    bounds.append(bound)
            waves = self._merged(w, np.stack(rows), len(images[0]))
            coherence = self._coherence(self._shares(kl, np.array(bounds)), detours)
            prefactor = wedge_prefactor(w["n"], k, 1.0)
            thin = self._thin(tops[j], chain, j + 1)
            pressure *= waves.sum(axis=0) * prefactor * np.sqrt(thin)
            strength *= (
                np.einsum("ijbm,ibm,jbm->bm", coherence, waves, np.conj(waves)).real
                * np.abs(prefactor) ** 2
                * thin
            )
        departure = (chain[1] - chain[0]) / legs[0]
        weight = source_weights(
            self.directivity, self.orientation, departure, bands, distances=legs[:1]
        )[0]
        amplitude = np.sqrt(weight * np.exp(-self.air * distance) / (4.0 * np.pi))
        amplitude /= math.sqrt(float(np.prod(legs)) * float(legs.sum()))
        strength *= 10.0 ** (-0.1 * near)
        energy = amplitude**2 * strength.mean(axis=1)
        spectrum = None
        if self.coherent:
            low = pressure[: self.coherent] * 10.0 ** (-0.05 * near[: self.coherent])
            level = np.sqrt(np.mean(np.abs(low) ** 2, axis=1))[:, None]
            spectrum = np.where(level > 0.0, low / np.where(level > 0.0, level, 1.0), 1.0)[None]
        arrival = (chain[-2] - listener) / legs[-1]
        over = EarlyReflections(
            order=np.zeros(1, dtype=np.int64),
            delay_s=np.array([distance / self.speed_of_sound]),
            distance_m=np.array([distance]),
            direction=arrival[None],
            energy=energy[None],
            points=[string],
            faces=[np.zeros(0, dtype=np.int64)],
            edge=np.array([CREST], dtype=np.int64),
            spectrum=spectrum,
        )
        return [over, *extra]

    @staticmethod
    def _bound(w: dict, image: dict, side: int, second: dict | None = None) -> float:
        """The angle of the listener from the boundary of the wedge ``w``'s wave of
        ``image`` (on ``side`` 0 before the crest or 1 after it), or of the wave from
        ``image`` (before it) to ``second`` (after it): positive where the one sees
        the other over the crest."""
        if second is not None:
            return math.pi - abs(w["n"] * math.pi - second["angle"] - image["angle"])
        if side == 0:
            return math.pi - abs(w["phi"] - image["angle"])
        return math.pi - abs(w["n"] * math.pi - image["angle"] - w["phi_s"])

    def _image_wave(
        self, w: dict, image: dict, side: int, lit: bool, before, after, legs, distance: float
    ) -> np.ndarray:
        """(B, M) the wedge ``w``'s wave of ``image`` (one of :meth:`_images`, on
        ``side`` 0 before the crest or 1 after it; ``lit`` on the lit side of its
        boundary): the uniform theory's terms for a source at the image's own angle
        and distance (its distance parameters ``before`` and ``after`` the crest,
        (B, M) or the legs ``legs`` (2,) themselves, the image's in place of its leg),
        times the ground's reflection factor, the image's farther spreading over the
        path ``distance`` m long and its later arrival. The term with the image's
        boundary counts in full; the other, which is the far face's in the exact
        solution for a wedge, by how far the reflection lies from the edge
        (:meth:`_apart`), so that the exact solution's four terms stand where the
        ground reflects at the edge, and the image's whole wave where it reflects far
        from it, as off the ground before a barrier."""
        k = self.wavenumber
        n, leg, far = w["n"], float(legs[side]), image["distance"]
        if side == 0:
            phi, phi_s = w["phi"], image["angle"]
            before = before + (far - leg)
        else:
            phi, phi_s = n * math.pi - image["angle"], w["phi_s"]
            after = after + (far - leg)
        kl = k * before * after / (before + after)
        terms = wedge_terms(n, phi, phi_s, kl, np.array([lit, True, True]))
        term = (terms[..., 1] + self._apart(image) * terms[..., 0]) * image["weight"]
        spread = math.sqrt(leg * distance / max(far * (distance + far - leg), _TINY))
        return term * image["factor"] * (spread * np.exp(-1j * k * (far - leg)))

    def _double_wave(
        self, w: dict, first: dict, second: dict, lit: bool, before, after, legs, distance: float
    ) -> np.ndarray:
        """(B, M) the wedge ``w``'s wave from the image ``first`` (before the crest) to
        the image ``second`` (after it): the sound reflected off the ground on both
        sides, the uniform theory's two terms for the two images, as far as both
        reflections lie from the edge (nothing where either reflects at it, where the
        exact solution's four terms hold it)."""
        k = self.wavenumber
        n = w["n"]
        phi, phi_s = n * math.pi - second["angle"], first["angle"]
        leg_0, leg_n = float(legs[0]), float(legs[1])
        far_0, far_n = first["distance"], second["distance"]
        before = before + (far_0 - leg_0)
        after = after + (far_n - leg_n)
        kl = k * before * after / (before + after)
        terms = wedge_terms(n, phi, phi_s, kl, np.array([lit, True, True]))
        term = (terms[..., 0] + terms[..., 1]) * self._apart(first) * self._apart(second)
        term = term * first["weight"] * second["weight"]
        longer = far_0 - leg_0 + far_n - leg_n
        spread = math.sqrt(
            leg_0 * leg_n * distance / max(far_0 * far_n * (distance + longer), _TINY)
        )
        factor = first["factor"] * second["factor"]
        return term * factor * (spread * np.exp(-1j * k * longer))

    def _apart(self, image: dict) -> np.ndarray:
        """(B, M) how far the reflection of ``image`` lies from the edge, 0 at it and
        1 many wavelengths away: |F(k δ)| of the uniform theory's transition function,
        for the extra path δ the sound goes by way of the reflection."""
        if image["extra"] <= 0.0:
            return np.zeros_like(self.wavenumber)
        return np.abs(transition(self.wavenumber * image["extra"]))

    def _coherence(self, share: np.ndarray, detours: list[float]) -> np.ndarray:
        """(R, R, B, M) how much of the interference of each pair of a wedge's R waves
        counts at each frequency: the product of the two waves' shares (``share``,
        (R, B, M); 1 for a wave with itself), and, for two that arrive apart by their
        reflections' extra paths (``detours``, m), what is left of it once averaged over
        the slice of the band the frequency stands for, as the engine averages it
        between paths (:func:`~acoustic_spaces.engine.coherent.total_energy`)."""
        rows = len(detours)
        coherence = share[:, None] * share[None, :]
        coherence[np.arange(rows), np.arange(rows)] = 1.0
        apart = (np.array(detours)[:, None] - np.array(detours)[None, :]) / self.speed_of_sound
        return coherence * np.sinc(apart[:, :, None] * self.slice_hz[None, None, :])[..., None]

    @staticmethod
    def _merged(w: dict, waves: np.ndarray, first: int) -> np.ndarray:
        """``waves`` (R, B, M; the direct wave first, then ``first`` images before the
        crest and the rest after it) with the wave along a face the chain runs along
        folded into the direct wave: along face n where the next point lies on it, and
        along face 0, taken once (half of the two), where the point before does."""
        if w["leave"]:  # the next point on face n: one wave along it
            waves[0] += waves[1 + first :].sum(axis=0)
            waves[1 + first :] = 0.0
        if w["graze"]:  # the sound comes along face 0, from the crest before
            waves[0] += waves[1 : 1 + first].sum(axis=0)
            waves[1 : 1 + first] = 0.0
            waves = waves * 0.5
        return waves

    def _top_width(self, gap: np.ndarray, s: np.ndarray, c: int, rates) -> np.ndarray:
        """(B, M) how wide the crest at profile point ``c`` is at each frequency: the
        span of the ground within an eighth of the wavelength of the string on either
        side of it (``gap`` the string's height over the profile's points, ``s`` their
        distance along it, the gap linear between them), less the span a sharp wedge
        would have, whose ground falls away from the string at ``rates`` (m per m, on
        each side): nothing for a ridge, the whole top of a flat crest, and a few
        metres of a rounded one at the highest bands."""
        tolerance = self.wavelength / 8.0  # (B, M)
        width = np.zeros_like(tolerance)
        for step in (-1, 1):
            # The points outwards from the crest, the ends (the source and the
            # listener, on the string) left out; the gap never narrows outwards.
            index = np.arange(c + step, 0 if step < 0 else len(s) - 1, step)
            if len(index) == 0:
                continue
            rising = np.maximum.accumulate(gap[index])
            count = (rising[None, None] <= tolerance[..., None]).sum(axis=-1)  # (B, M)
            inside = np.where(count > 0, s[index[np.maximum(count - 1, 0)]], s[c])
            gap_in = np.where(count > 0, rising[np.maximum(count - 1, 0)], 0.0)
            beyond = np.minimum(count, len(index) - 1)
            gap_out = rising[beyond]
            part = np.where(
                count < len(index),
                (tolerance - gap_in) / np.maximum(gap_out - gap_in, _TINY),
                0.0,
            )
            width += np.abs(inside - s[c]) + part * np.abs(s[index[beyond]] - inside)
            rate = rates[0 if step < 0 else 1]
            width -= tolerance / rate if rate > _TINY else width
        return np.maximum(width, 0.0)

    def _thin(self, top: np.ndarray, chain: np.ndarray, j: int) -> np.ndarray:
        """(B, M) the share of the energy of the wave over crest ``j`` of the chain
        that a crest ``top`` (B, M) m wide lets through, against a thin one: ISO
        9613-2's 10 lg(3 + C3 X) against the thin screen's 10 lg(3 + X), with X = 20 N
        for the path difference over the crest, so that a wide crest screens as a thin
        one at the edge of its shadow and up to three times more deep in it."""
        delta = (
            float(np.linalg.norm(chain[j] - chain[j - 1]))
            + float(np.linalg.norm(chain[j + 1] - chain[j]))
            - float(np.linalg.norm(chain[j + 1] - chain[j - 1]))
        )
        x = 40.0 * max(delta, 0.0) / self.wavelength
        return (3.0 + x) / (3.0 + _thickness(top, self.wavelength) * x)

    def _shares(self, kl: np.ndarray, angle: np.ndarray) -> np.ndarray:
        """(3, B, M) |F(X)| of a wedge's waves' boundaries ``angle`` away, for wave
        numbers times its distance parameter ``kl`` (B, M); 1 in the coherent bands."""
        share = _share(kl[None], angle[None])[0]
        share[:, : self.coherent] = 1.0
        return share

    @staticmethod
    def _radii(legs: np.ndarray, nearness: list[np.ndarray]) -> tuple[list, list]:
        """For each wedge of a chain with ``legs`` (one more than wedges), the distances
        (B, M) its distance parameter takes before and after it. Each is its leg, but
        near a neighbour's boundary the neighbour's wave is still curved like the wave
        that reaches it, not like a wave from its edge, and the distance grows towards
        the neighbour's own: by 1 - |F(X)| of the neighbour's boundary, shared between
        the two by how much nearer each is to its boundary (see
        :meth:`acoustic_spaces.engine.diffraction.EdgeDiffraction._double`)."""
        count = len(nearness)
        before = [np.full_like(nearness[0], legs[0])]
        for j in range(1, count):
            flat = 1.0 / legs[j]
            bend = (1.0 - nearness[j - 1]) * (flat - 1.0 / (before[j - 1] + legs[j]))
            total = nearness[j - 1] + nearness[j]
            part = np.where(total > 0.0, nearness[j] / np.where(total > 0.0, total, 1.0), 0.5)
            before.append(1.0 / (flat - part * bend))
        after = [np.full_like(nearness[0], legs[count])]
        for j in range(count - 2, -1, -1):
            flat = 1.0 / legs[j + 1]
            bend = (1.0 - nearness[j + 1]) * (flat - 1.0 / (after[0] + legs[j + 1]))
            total = nearness[j] + nearness[j + 1]
            part = np.where(total > 0.0, nearness[j] / np.where(total > 0.0, total, 1.0), 0.5)
            after.insert(0, 1.0 / (flat - part * bend))
        return before, after

    def _wedge(self, chain, feet, s, j: int, along) -> dict:
        """The wedge at crest ``j`` of the chain: its open angle over pi, the angles of
        its neighbours round the edge from face 0 (and the next one's from face n), the
        legs to them, each face's frame in the vertical plane (along it from the edge,
        and its normal into the air), and for each face the ground face it stands for,
        the cosine the wave meets it at, the leg of the reflection and the points of
        its Fresnel zone, for the wave along the face where a neighbour lies on it."""
        previous, edge, following = chain[j - 1], chain[j], chain[j + 1]
        # In the vertical plane: s along the profile, z up; face 0 runs from the edge
        # down to the ground under the point before, face n to that under the next.
        alpha_0 = math.atan2(edge[2] - feet[j - 1], s[j] - s[j - 1])
        alpha_n = math.atan2(edge[2] - feet[j + 1], s[j + 1] - s[j])
        n = (math.pi + alpha_0 + alpha_n) / math.pi
        u_0 = np.array([-math.cos(alpha_0), -math.sin(alpha_0)])
        n_0 = np.array([-math.sin(alpha_0), math.cos(alpha_0)])  # into the air
        u_n = np.array([math.cos(alpha_n), -math.sin(alpha_n)])
        n_n = np.array([math.sin(alpha_n), math.cos(alpha_n)])
        to_previous = np.array([s[j - 1] - s[j], previous[2] - edge[2]])
        to_next = np.array([s[j + 1] - s[j], following[2] - edge[2]])
        phi_s = float(np.clip(math.atan2(to_previous @ n_0, to_previous @ u_0), 0.0, n * math.pi))
        psi = float(np.clip(math.atan2(to_next @ n_n, to_next @ u_n), 0.0, n * math.pi))
        m = float(np.linalg.norm(previous - edge))
        length = float(np.linalg.norm(following - edge))

        def foot(index: int, towards: int) -> np.ndarray:
            """The ground under chain point ``index``, a little towards ``towards``."""
            at = s[index] + 1e-3 * np.sign(s[towards] - s[index])
            return np.array([*(self.source[:2] + at * along), feet[index]])

        feet_3d = foot(j - 1, j), foot(j + 1, j)
        faces = [ground_face_under(self.model, p, self.ground) for p in feet_3d]
        return {
            "n": n,
            "phi_s": phi_s,
            "psi": psi,
            "phi": n * math.pi - psi,
            "m": m,
            "l": length,
            "graze": phi_s < _GRAZING,
            "leave": psi < _GRAZING,
            "frame": ((u_0, n_0), (u_n, n_n)),
            "face": faces,
            "cosine": (math.sin(phi_s), math.sin(psi)),
            "leg": (m, length),
            "ends": ((previous, feet_3d[0], edge), (edge, feet_3d[1], following)),
        }

    @staticmethod
    def _direct_wave(w: dict, lit: bool, kl: np.ndarray) -> np.ndarray:
        """(B, M) the two terms of the wedge ``w``'s coefficient for the wave straight
        over its edge, ``lit`` on the lit side of its shadow boundary."""
        terms = wedge_terms(w["n"], w["phi"], w["phi_s"], kl, np.array([lit, True, True]))
        return terms[..., 0] + terms[..., 1]

    def _face_factor(self, face: int, cosine: float, leg: float, ends) -> np.ndarray:
        """(B, M) the reflection factor of a wedge's face that stands for ground face
        ``face``, for a wave that meets it at ``cosine`` after a leg ``leg`` m long:
        its specular share, the reflection's phase in the coherent bands, and the
        spherical wave's factor over a ground with an impedance (the zones its Fresnel
        zone from ``ends`` covers)."""
        model = self.model
        if face < 0:
            return np.ones_like(self.wavenumber, dtype=complex)
        index = np.array([face])
        kept = (1.0 - model.absorption[face]) * (1.0 - model.scattering[face])
        factor = model.factor_faces(index)[0]
        if factor:
            kept = np.ones_like(kept)
        out = np.sqrt(kept)[:, None] * np.ones(self.wavenumber.shape[1], dtype=complex)
        cosines = np.array([cosine])
        if self.coherent:
            out[: self.coherent] *= reflection_phases(model, index, cosines, self.coherent)[0]
        if factor:
            points = tuple(np.asarray(p, dtype=float)[None] for p in ends)
            out *= ground_factors(model, index, cosines, np.array([leg]), ends=points)[0]
        return out

    def _images(self, w, chain, chain_s, s, ground_z, j: int, side: int, along) -> list[dict]:
        """The images of the chain point before crest ``j`` (``side`` 0) or after it
        (1) in the ground on that side of the crest, whose waves the wedge ``w``
        bends over it: one for each reflection of the point towards the edge off the
        profile's segments between them (:meth:`_facets`), or the image in the
        wedge's own face when the point lies on it (the wave along the face). Each has
        its angle from the face, as the wedge measures it (negative below the face),
        its distance to the edge, the ground's reflection factor (B, M), how much
        farther the sound goes by way of the reflection than straight to the edge (m),
        its weight (B, M; the share of its Fresnel zone the segment holds, 1 along the
        face), the plane it reflects in (a point of it and its normal, into the air;
        None for the face) and the span along the profile of the ground the plane
        stands for."""
        edge = chain[j]
        other = j - 1 if side == 0 else j + 1
        end = chain[other]
        u, normal = w["frame"][side]
        if w["graze" if side == 0 else "leave"]:
            factor = self._face_factor(
                w["face"][side], w["cosine"][side], w["leg"][side], w["ends"][side]
            )
            angle = -(w["phi_s"] if side == 0 else w["psi"])
            return [
                {
                    "angle": angle,
                    "distance": w["leg"][side],
                    "factor": factor,
                    "extra": 0.0,
                    "weight": np.ones_like(self.wavenumber),
                    "plane": None,
                }
            ]
        out = []
        for found in self._facets(end, chain_s[other], edge, chain_s[j], s, ground_z, along):
            to_image = found["image"] - np.array([chain_s[j], edge[2]])
            factor = self._face_factor(
                found["face"], found["cosine"], found["leg"], (end, found["at"], edge)
            )
            out.append(
                {
                    "angle": math.atan2(to_image @ normal, to_image @ u),
                    "distance": float(np.linalg.norm(to_image)),
                    "factor": factor,
                    "extra": found["extra"],
                    "weight": found["weight"],
                    "plane": (found["at"], found["normal"]),
                    "span": found["span"],
                }
            )
        return out

    def _facets(self, end, s_end: float, apex, s_apex: float, s, ground_z, along) -> list[dict]:
        """The reflections of the point ``end`` (``s_end`` along the profile) towards
        ``apex`` (``s_apex``) off the segments of the profile (``s``, ``ground_z``)
        between them, each weighted by the share (B, M) of the first Fresnel zone of the
        reflection off its plane that the segment holds, of the zone's share within the
        side (:data:`~acoustic_spaces.geometry.outdoor.FRESNEL_PATH_DIFFERENCE`
        wavelengths of extra path, as the ground's zones are weighted): at high
        frequencies the segment the sound reflects off, at low ones every segment
        round it with its own plane, as Nord2000 weighs terrain segments, and never
        more than one reflection in all: a plane slope's segments add up to its one,
        and so do neighbours whose zones overlap. For each: the image of ``end``
        in the segment's plane (s, z), the reflection point (in space, on the segment),
        the plane's normal (into the air), the cosine the sound meets it at, the
        path's length by way of it and how much longer that is than straight to the
        edge, the ground face under it, the segment's span along the profile and the
        weight. Only reflections off planes both points lie over, whose leg to the
        reflection clears the ground, and that nothing in the model stands in the way
        of; a segment that passes through the point itself gives none."""
        from acoustic_spaces.geometry.outdoor import FRESNEL_PATH_DIFFERENCE

        model = self.model
        lo, hi = min(s_end, s_apex), max(s_end, s_apex)
        first = max(int(np.searchsorted(s, lo, side="right")) - 1, 0)
        last = min(int(np.searchsorted(s, hi, side="left")), len(s) - 1)
        if last - first < 1:
            return []
        s_a, s_b = s[first:last], s[first + 1 : last + 1]
        z_a, z_b = ground_z[first:last], ground_z[first + 1 : last + 1]
        run = np.hypot(s_b - s_a, z_b - z_a)
        keep = run > _TINY
        t_s, t_z = (s_b - s_a) / np.maximum(run, _TINY), (z_b - z_a) / np.maximum(run, _TINY)
        n_s, n_z = -t_z, t_s  # into the air
        e, a = np.array([s_end, end[2]]), np.array([s_apex, apex[2]])
        h_e = (e[0] - s_a) * n_s + (e[1] - z_a) * n_z
        h_a = (a[0] - s_a) * n_s + (a[1] - z_a) * n_z
        keep &= (h_e > _TINY) & (h_a >= -_TINY)
        if not keep.any():
            return []
        image_s, image_z = e[0] - 2.0 * h_e * n_s, e[1] - 2.0 * h_e * n_z
        # The reflection point on each plane, along it from the segment's start, and
        # the first Fresnel zone of the reflection along the plane: where the plane
        # cuts the ellipse round the image and the edge.
        share = h_e / np.maximum(h_e + h_a, _TINY)
        p_s = image_s + (a[0] - image_s) * share
        p_z = image_z + (a[1] - image_z) * share
        tau_p = (p_s - s_a) * t_s + (p_z - z_a) * t_z
        axis_s, axis_z = a[0] - image_s, a[1] - image_z
        length = np.maximum(np.hypot(axis_s, axis_z), _TINY)
        e_s, e_z = axis_s / length, axis_z / length
        delta = FRESNEL_PATH_DIFFERENCE * self.wavelength  # (B, M)
        half = 0.5 * (length[:, None, None] + delta)  # (K, B, M) the semi-axes
        across = 0.5 * np.sqrt(delta * (2.0 * length[:, None, None] + delta))
        c_s, c_z = 0.5 * (image_s + a[0]), 0.5 * (image_z + a[1])
        v_s, v_z = s_a - c_s, z_a - c_z
        alpha = (v_s * e_s + v_z * e_z)[:, None, None]
        beta = (t_s * e_s + t_z * e_z)[:, None, None]
        gamma = (v_s * -e_z + v_z * e_s)[:, None, None]
        eta = (t_s * -e_z + t_z * e_s)[:, None, None]
        qa = beta**2 / half**2 + eta**2 / across**2
        qb = alpha * beta / half**2 + gamma * eta / across**2
        qc = alpha**2 / half**2 + gamma**2 / across**2 - 1.0
        root = np.sqrt(np.maximum(qb**2 - qa * qc, 0.0))
        zone_lo, zone_hi = (-qb - root) / qa, (-qb + root) / qa  # (K, B, M) along the plane
        # The side's span along each plane, and the segment's within it.
        side_lo = np.minimum((lo - s_a) / np.maximum(t_s, _TINY), run)
        side_hi = np.maximum((hi - s_a) / np.maximum(t_s, _TINY), 0.0)
        seg_lo, seg_hi = np.maximum(side_lo, 0.0), np.minimum(side_hi, run)
        within_side = np.maximum(
            np.minimum(zone_hi, side_hi[:, None, None])
            - np.maximum(zone_lo, side_lo[:, None, None]),
            0.0,
        )
        within_seg = np.maximum(
            np.minimum(zone_hi, seg_hi[:, None, None]) - np.maximum(zone_lo, seg_lo[:, None, None]),
            0.0,
        )
        weight = np.where(within_side > _TINY, within_seg / np.maximum(within_side, _TINY), 0.0)
        # The ground reflects the sound once at most: where the zones of neighbouring
        # segments overlap (a bend near the reflection point, a hollow) they share the
        # one reflection between them, as the segments of a plane do.
        weight = weight / np.maximum(weight.sum(axis=0), 1.0)

        keep &= weight.reshape(len(run), -1).max(axis=1) > 1e-6
        out = []
        for i in np.flatnonzero(keep):
            # The reflection point on the segment itself: the nearest of it to the
            # plane's, for the ground under it and the leg up to it.
            tau = float(np.clip(tau_p[i], 0.0, run[i]))
            at_s, at_z = s_a[i] + tau * t_s[i], z_a[i] + tau * t_z[i]
            if not self._above(s, ground_z, e, (at_s, at_z)):
                continue
            at = np.array([*(self.source[:2] + at_s * along), at_z])
            towards = np.sign(s_end - at_s) if abs(s_end - at_s) > _TINY else 1.0
            under = np.array([*(self.source[:2] + (at_s + 1e-3 * towards) * along), at_z])
            face = ground_face_under(model, under, self.ground)
            if face < 0:
                continue
            corners = np.vstack([end, at, apex])
            steps = np.linalg.norm(np.diff(corners, axis=0), axis=1) > _TINY
            if not _clear(
                model, corners[:-1][steps], corners[1:][steps], through=self.ground
            ).all():
                continue
            leg_e = float(np.hypot(p_s[i] - e[0], p_z[i] - e[1]))
            leg = leg_e + float(np.hypot(p_s[i] - a[0], p_z[i] - a[1]))
            out.append(
                {
                    "image": np.array([image_s[i], image_z[i]]),
                    "at": at,
                    "normal": np.array([*(n_s[i] * along), n_z[i]]),
                    "cosine": float(h_e[i] / max(leg_e, _TINY)),
                    "leg": leg,
                    "extra": max(leg - float(np.linalg.norm(e - a)), 0.0),
                    "face": int(face),
                    "span": (float(s_a[i]), float(s_b[i])),
                    "weight": weight[i],
                }
            )
        return out

    @staticmethod
    def _above(s, ground_z, p, q) -> bool:
        """Whether the straight line from ``p`` to ``q`` ((s, z) in the profile's
        plane) stays over the ground at the profile's points between them."""
        lo, hi = (p, q) if p[0] <= q[0] else (q, p)
        inside = np.flatnonzero((s > lo[0] + _TINY) & (s < hi[0] - _TINY))
        if len(inside) == 0:
            return True
        line = lo[1] + (hi[1] - lo[1]) * (s[inside] - lo[0]) / max(hi[0] - lo[0], _TINY)
        return bool(np.all(line >= ground_z[inside] - 1e-6))

    def _image_path(
        self, image: dict, source, listener, s, ground_z, along
    ) -> EarlyReflections | None:
        """The reflection from the source to the listener off the plane of ``image``
        (one of :meth:`_images`): where the path meets the plane within the ground it
        stands for, when nothing stands in its way (None for the wave along a face)."""
        if image["plane"] is None:
            return None
        model = self.model
        point, normal = image["plane"]
        height_s = float((source - point) @ normal)
        height_l = float((listener - point) @ normal)
        if height_s <= _TINY or height_l <= _TINY:
            return None
        mirror = source - 2.0 * height_s * normal
        at = mirror + (listener - mirror) * height_s / (height_s + height_l)
        s_at = float((at[:2] - source[:2]) @ along)
        lo, hi = image["span"]
        slack = 1e-6 * max(hi - lo, 1.0)
        if not lo - slack <= s_at <= hi + slack:
            return None
        s_source, s_listener = 0.0, float((listener[:2] - source[:2]) @ along)
        ends = (s_source, source[2]), (s_at, at[2]), (s_listener, listener[2])
        if not (self._above(s, ground_z, *ends[:2]) and self._above(s, ground_z, *ends[1:])):
            return None
        face = ground_face_under(model, at, self.ground)
        if face < 0:
            return None
        corners = np.vstack([source, at, listener])
        if not _clear(model, corners[:-1], corners[1:], through=self.ground).all():
            return None
        steps = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        distance = float(steps.sum())
        found = GroundReflection(at, int(face), source, listener, height_s, height_l, normal)
        share, spectrum = reflection_factor(model, found, distance, source, listener, self.coherent)
        bands = model.absorption.shape[1]
        energy = share * np.exp(-self.air * distance) / (4.0 * np.pi * distance**2)
        departure = (at - source) / steps[0]
        aimed = source_weights(
            self.directivity, self.orientation, departure, bands, distances=[distance]
        )
        energy = energy * aimed[0]
        energy = energy * np.mean(image["weight"] ** 2, axis=1)
        return EarlyReflections(
            order=np.ones(1, dtype=np.int64),
            delay_s=np.array([distance / self.speed_of_sound]),
            distance_m=np.array([distance]),
            direction=((at - listener) / steps[1])[None],
            energy=energy[None],
            points=[corners],
            faces=[np.array([face], dtype=np.int64)],
            spectrum=spectrum[None] if self.coherent else None,
            cosines=[own_cosine(model, found)] if self.coherent else [],
        )

    @staticmethod
    def _near(points, ground, s, i: int, j: int) -> tuple[int, float] | None:
        """The crest between points ``i`` and ``j`` of the profile (``points``, over
        ``ground``) that comes nearest the straight string between them: a point the
        ground rises to from both sides (over a plane the ground only comes nearer
        towards an end), the one the path over it is the least longer than the
        string; its index and that path difference (negative). None when there is no
        crest, or it lies too far below the string to matter."""
        a, b = points[i], points[j]
        line = a[2] + (b[2] - a[2]) * (s[i : j + 1] - s[i]) / max(s[j] - s[i], _TINY)
        gap = line - ground[i : j + 1]
        if len(gap) < 3:
            return None
        # A crest: the ground comes no nearer the string next to it, and falls away
        # from it on both sides somewhere (not along a plane parallel to the string).
        lift = _LIFT + 1e-3 * float(s[j] - s[i])
        left = np.maximum.accumulate(gap)[:-2]  # the widest gap before each point
        right = np.maximum.accumulate(gap[::-1])[::-1][2:]  # and after it
        inner = gap[1:-1]
        lowest = (inner <= gap[:-2]) & (inner <= gap[2:])
        crests = np.flatnonzero(lowest & (left > inner + lift) & (right > inner + lift))
        if len(crests) == 0:
            return None
        crests = crests + 1
        tops = points[i + crests]
        over = np.linalg.norm(tops - a, axis=1) + np.linalg.norm(b - tops, axis=1)
        delta = float(np.linalg.norm(b - a)) - over
        best = int(np.argmax(delta))
        if delta[best] <= -0.5 * LARGEST_WAVELENGTH:
            return None
        return i + int(crests[best]), float(delta[best])


def _apex(p, a, b, q) -> tuple[float, float]:
    """Where the line through ``p`` and ``a`` meets the line through ``b`` and ``q``
    (points (s, z) along the string, in order): the equivalent edge of the crest the
    string runs along from ``a`` to ``b``; ``a`` itself when the two are one point,
    and the middle of the two when the lines run parallel."""
    if a == b:
        return a
    u = (a[0] - p[0], a[1] - p[1])
    v = (q[0] - b[0], q[1] - b[1])
    cross = u[0] * v[1] - u[1] * v[0]
    if abs(cross) <= _TINY * max(1.0, abs(u[0] * v[0])):
        return (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    t = ((b[0] - a[0]) * v[1] - (b[1] - a[1]) * v[0]) / cross
    return (a[0] + t * u[0], a[1] + t * u[1])


def _bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """The angle (rad) the string turns by at ``b`` on its way from ``a`` to ``c``."""
    u, v = b - a, c - b
    cosine = float(u @ v) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-300)
    return math.acos(min(1.0, max(-1.0, cosine)))


def _thickness(top: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """ISO 9613-2's factor C3 for a screen ``top`` m thick (bands, M, or one width), 1
    for a thin one and 3 for one much thicker than the ``wavelength`` (bands, M):
    (1 + (5 λ / e)²) / (1/3 + (5 λ / e)²)."""
    ratio = (5.0 * wavelength / np.maximum(np.asarray(top, dtype=float), _TINY)) ** 2
    return (1.0 + ratio) / (1.0 / 3.0 + ratio)


def _crests(corners: list[int], s: np.ndarray, z: np.ndarray) -> list[tuple[int, int]]:
    """The crests of the string with these corners (indices into the profile, the
    ends first and last): runs of corners between which the string lies on the ground,
    as (first, last) positions in ``corners``."""
    groups: list[list[int]] = []
    for k in range(1, len(corners) - 1):
        p, q = corners[k - 1], corners[k]
        if groups and groups[-1][1] == k - 1 and _on_ground(s, z, p, q):
            groups[-1][1] = k
        else:
            groups.append([k, k])
    return [(a, b) for a, b in groups]


def _on_ground(s: np.ndarray, z: np.ndarray, p: int, q: int) -> bool:
    """Whether the string from profile point ``p`` to ``q`` runs along the ground: no
    point between them lies more than a little below it."""
    if q - p < 2:
        return True
    line = z[p] + (z[q] - z[p]) * (s[p + 1 : q] - s[p]) / max(s[q] - s[p], _TINY)
    return bool(np.all(line - z[p + 1 : q] <= _LIFT + 1e-3 * (s[q] - s[p])))


__all__ = [
    "CREST",
    "LIT_LIMIT",
    "CrestDiffraction",
    "knife_edge_loss",
    "taut_string",
]
