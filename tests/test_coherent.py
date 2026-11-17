"""Early paths that add in pressure in the lowest bands (engine.coherent)."""

import dataclasses
import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import (
    arriving_with,
    band_count,
    ground_wave_factors,
    own_phases,
    reflection_phases,
    sample_frequencies,
    total_energy,
    with_own_phases,
)
from acoustic_spaces.engine.diffraction import (
    EdgeDiffraction,
    band_frequencies,
    wedge_prefactor,
    wedge_terms,
)
from acoustic_spaces.engine.impedance import fit_admittance
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.response import Response
from acoustic_spaces.engine.simulate import coherent_bands
from acoustic_spaces.materials.ground import band_absorption, plane_reflection, surface_impedance
from acoustic_spaces.parameters import from_impulse_response, octave_bands
from acoustic_spaces.scene.model import EarlyStage
from acoustic_spaces.synthesis import REFERENCE_ENERGY_10M


def _box(coherent, **simulation) -> Room:
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": [7.3, 5.1, 3.0]},
                "materials": {"m": {"absorption": [0.15] * 8, "scattering": 0.0}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [1.7, 2.2, 1.6]}],
                "listeners": [{"name": "l", "position": [5.3, 3.4, 1.2]}],
                "simulation": {
                    "seed": 3,
                    "ir_length_s": 0.6,
                    "early": {"diffraction": "none", "coherent_max_band_hz": coherent},
                    "late": {"rays": 500},
                    "scattering": {"from_size": False},
                    **simulation,
                },
            }
        )
    )


def _floor(tmp_path: Path, absorption: float | list[float]) -> Room:
    """A floor 100 m wide in free field, a source and a listener 0.3 m above it."""
    if isinstance(absorption, float):
        absorption = [absorption] * 8
    obj = tmp_path / "floor.obj"
    obj.write_text("v -50 -50 0\nv 50 -50 0\nv 50 50 0\nv -50 50 0\ng floor\nf 1 2 3 4\n")
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"floor": {"absorption": absorption, "scattering": 0.0}},
                "default_material": "floor",
                "sources": [{"name": "s", "position": [0.0, 0.0, 0.3]}],
                "listeners": [{"name": "l", "position": [10.0, 0.0, 0.3]}],
                "simulation": {"early": {"max_order": 1}, "late": {"rays": 100}},
            }
        )
    )


def test_settings_pick_the_bands() -> None:
    assert EarlyStage().coherent_max_band_hz == 125
    assert band_count(125, 8) == 2 and band_count(125, 24) == 6 and band_count(None, 8) == 0
    assert band_count(250, 8, wave_max_band_hz=125) == 3
    assert band_count(125, 8, wave_max_band_hz=125) == 0  # the wave solver's bands
    assert sample_frequencies(8, 2).shape == (2, 8)
    with pytest.raises(ValueError):
        EarlyStage(coherent_max_band_hz=100)


def test_impulse_response_and_echogram_agree_where_paths_add_in_pressure() -> None:
    def difference(coherent):
        response = simulate(_box(coherent))["s", "l"]
        early = dataclasses.replace(response, late=np.zeros_like(response.late))
        ir = early.impulse_response(48_000)
        measured = from_impulse_response(ir, 48_000, REFERENCE_ENERGY_10M, noise=False)
        simulated = early.parameters()
        return {
            name: getattr(measured, name)[:2] - getattr(simulated, name)[:2]
            for name in ("g", "c50", "ts")
        }

    coherent = difference(125)
    np.testing.assert_allclose(coherent["g"], 0.0, atol=0.02)
    np.testing.assert_allclose(coherent["c50"], 0.0, atol=0.05)
    np.testing.assert_allclose(coherent["ts"], 0.0, atol=3e-4)
    # In energy the paths' interference within the low bands is missed.
    energy = difference(None)
    assert np.abs(energy["g"]).max() > 1.0 or np.abs(energy["c50"]).max() > 3.0


def test_a_coloured_sound_in_the_first_milliseconds() -> None:
    # The bands are put together with zero-phase filters, which spread an arrival both
    # ways in time; what falls before the response's first sample is lost. The
    # echogram is not affected; the G an impulse response gives is, a little, for a
    # sound whose level changes from band to band, 2.9 ms in against 202.9 ms.
    def strength(delay, levels):
        paths = early_stage.EarlyReflections(
            order=np.zeros(1, dtype=np.int64),
            delay_s=np.array([delay]),
            distance_m=np.array([delay * 343.0]),
            direction=np.array([[1.0, 0.0, 0.0]]),
            energy=1e-2 * 10 ** (np.asarray(levels, dtype=float)[None, :] / 10),
        )
        response = Response(
            source="s", listener="l", early=paths, late=np.zeros((8, 500)), time_step=1e-3,
            length_s=0.5, direct_delay_s=delay, transition_s=0.3, volume_m3=None,
            speed_of_sound=343.0, seed=0,
        )  # fmt: skip
        ir = response.impulse_response(48_000)
        return response.energy().sum(axis=1), from_impulse_response(ir, 48_000, 1.0).g

    for levels, bound in (([0.0] * 8, 1e-3), ([0, -1.5, -4.7, -10, -15, -20, -20, -20], 0.15)):
        early, g_early = strength(0.0029, levels)
        late, g_late = strength(0.2029, levels)
        np.testing.assert_allclose(early, late, rtol=1e-9)
        assert np.abs(g_early - g_late).max() < bound


def test_reflection_phases_of_hard_and_absorbing_surfaces(tmp_path: Path) -> None:
    # At 3 degrees above the surface; at exactly grazing incidence every surface that
    # absorbs at all reflects half a turn out of phase.
    for absorption, grazing in ((0.01, 0.0), (0.8, math.pi)):
        model = TraceModel.from_room(_floor(tmp_path, absorption))
        faces = np.zeros(3, dtype=np.int64)
        phase = np.angle(reflection_phases(model, faces, np.array([1.0, 0.5, 0.05]), 2))
        np.testing.assert_allclose(phase[:2], 0.0, atol=1e-6)  # constant: a real admittance
        np.testing.assert_allclose(np.abs(phase[2]), grazing, atol=1e-6)


@pytest.mark.parametrize(("flow", "thickness"), [(10.0, 0.05), (20.0, 0.1), (40.0, 0.025)])
def test_reflection_phases_come_from_every_band(
    tmp_path: Path, flow: float, thickness: float
) -> None:
    # A porous layer on a hard backing (Miki's model) absorbs more with frequency. The
    # phase it gives at 63 and 125 Hz is matched more closely by an admittance fitted
    # to its absorption in every band than by one fitted to those two bands alone, and
    # it does not depend on how many bands add in pressure.
    def layer(f):
        return 1.0 / surface_impedance(f, flow, thickness=thickness)

    absorption = band_absorption(layer, 8)
    model = TraceModel.from_room(_floor(tmp_path, absorption))
    cosines = np.array([1.0, 0.7, 0.4, 0.15])
    faces = np.zeros(len(cosines), dtype=np.int64)
    phase = reflection_phases(model, faces, cosines, 2)
    np.testing.assert_allclose(phase, reflection_phases(model, faces, cosines, 8)[:, :2])
    f = sample_frequencies(8, 2)
    true = plane_reflection(layer(f)[None], cosines[:, None, None])
    two = plane_reflection(fit_admittance(absorption[:2], 8)(f)[None], cosines[:, None, None])
    error = np.abs(np.angle(phase / true)).max()
    assert error < 0.5 * np.abs(np.angle(two / true)).max()


@pytest.mark.parametrize(("absorption", "low"), [(0.01, 5.5), (0.8, -4.0)])
def test_a_floor_reflection_adds_to_the_direct_sound_in_pressure(
    tmp_path: Path, absorption: float, low: float
) -> None:
    # The reflection travels 2 cm further at grazing incidence: off a hard floor it
    # doubles the pressure at low frequencies (+6 dB), off an absorbing one it arrives
    # with its phase turned and takes from it, as outdoors over grass.
    room = _floor(tmp_path, absorption)
    paths = early_paths(room)
    assert len(paths) == 2
    free = 1.0 / (4 * math.pi * 100.0)
    level = 10 * np.log10(total_energy(paths) / free)
    in_energy = 10 * np.log10(paths.energy.sum(axis=0) / free)
    assert np.all(level[:2] > low) if low > 0 else np.all(level[:2] < low)
    np.testing.assert_allclose(level[2:], in_energy[2:])
    response = simulate(room)["s", "l"]
    early = dataclasses.replace(response, late=np.zeros_like(response.late))
    g = early.parameters().g[:2]
    np.testing.assert_allclose(g, level[:2], atol=0.3)


def test_sound_over_an_edge_adds_all_its_terms_where_coherent(tmp_path: Path) -> None:
    # In front of a thin screen, lit by the direct sound and its reflection off it:
    # in the coherent bands the diffracted wave is all four terms in phase.
    obj = tmp_path / "screen.obj"
    obj.write_text("v 0 -300 -300\nv 0 300 -300\nv 0 300 0\nv 0 -300 0\ng screen\nf 1 2 3 4\n")
    source, listener = np.array([-2.0, 0.0, -1.0]), np.array([-1.0, 0.0, -3.0])
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": list(source)}],
                "listeners": [{"name": "l", "position": list(listener)}],
            }
        )
    )
    model = TraceModel.from_room(room)
    c, air = room.speed_of_sound(), np.zeros(8)
    specular = early_stage.find_paths(
        model,
        source,
        listener,
        [np.array([[0]])],
        speed_of_sound=c,
        air=air,
        max_delay_s=1.0,
        coherent=2,
    )
    diffraction = EdgeDiffraction(
        model, source, speed_of_sound=c, air=air, mode="first_order", coherent=2
    )
    paths = diffraction.apply(listener, specular, 1.0)
    bent = np.flatnonzero(paths.diffracted)
    assert len(bent) == 1
    # The specular paths keep their own energy: they interfere in the response.
    np.testing.assert_allclose(paths.energy[~paths.diffracted][:, :2], specular.energy[:, :2])
    frequencies, _ = band_frequencies()
    k = 2 * math.pi * frequencies / c
    r1, r2 = np.linalg.norm(source), np.linalg.norm(listener)
    terms = wedge_terms(2.0, math.atan2(1, 3), math.atan2(2, 1), k * r1 * r2 / (r1 + r2))
    pressure = wedge_prefactor(2.0, k, 1.0) * terms.sum(axis=-1)
    expected = np.mean(np.abs(pressure) ** 2, axis=1) / (4 * math.pi * r1 * r2 * (r1 + r2))
    np.testing.assert_allclose(paths.energy[bent[0], :2], expected[:2], rtol=1e-9)
    # Its spectrum is the coefficient's, relative to its level and delay.
    shape = pressure[:2] / np.sqrt(np.mean(np.abs(pressure[:2]) ** 2, axis=1, keepdims=True))
    np.testing.assert_allclose(paths.spectrum[bent[0]], shape, rtol=1e-9, atol=1e-12)


def test_without_coherent_bands_paths_add_in_energy() -> None:
    response = simulate(_box(None))["s", "l"]
    assert response.early.spectrum is None and response.coherent_bands == 0
    wave = _box(125, wave={"enabled": True, "max_band_hz": 125}).scene.simulation
    assert coherent_bands(wave, 8) == 0  # the wave solver's bands take their place
    # The points of a line source are uncorrelated.
    data = _box(125).scene.model_dump(mode="json")
    data["sources"][0].update(line_m=2.0, spacing_m=1.0)
    line = simulate(Room.from_scene(Scene.model_validate(data)))["s", "l"]
    assert line.coherent_bands == 0 and line.early.spectrum is None


def test_each_band_keeps_its_own_shape_into_the_crossovers() -> None:
    """A path whose spectrum stands for different sound in two neighbouring bands (as
    the parabolic equation's bands do outdoors): in the crossover between them each
    band's amplitude takes its own band's shape, not a blend of the two."""
    from acoustic_spaces import synthesis
    from acoustic_spaces.engine.coherent import shaped

    n_fft, rate = 1 << 12, 8000.0
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gains = synthesis.band_gains(n_fft, rate, 8)[:2]
    spectrum = np.ones((1, 2, 8), dtype=complex)
    spectrum[0, 0] = 2.0j  # the lower band: twice as loud, a quarter turn ahead
    amplitude = np.array([[3.0, 5.0]])
    out = shaped(spectrum, 8, f, amplitude, gains)[0]
    expected = amplitude[0, 0] * gains[0] * 2.0j + amplitude[0, 1] * gains[1]
    np.testing.assert_allclose(out, expected, atol=1e-12)
    # Both bands are at work in the crossover around their edge, 88 Hz.
    middle = np.argmin(np.abs(f - 88.0))
    assert gains[0, middle] > 0.2 and gains[1, middle] > 0.2
    # A smooth spectrum across the bands is interpolated within each band as before.
    smooth = np.exp(1j * np.linspace(0.0, 0.5, 16)).reshape(1, 2, 8)
    out = shaped(smooth, 8, f, np.ones((1, 2)), gains)[0]
    samples = sample_frequencies(8, 2)
    inside = (f > samples[0, 0]) & (f < samples[0, -1]) & (gains[1] == 0.0)
    phase = np.interp(np.log2(f[inside]), np.log2(samples[0]), np.linspace(0.0, 0.5, 16)[:8])
    np.testing.assert_allclose(np.angle(out[inside]), phase, atol=1e-12)


def _shaped_bin_by_bin(spectrum, bands, f, amplitude, gains) -> np.ndarray:
    """:func:`~acoustic_spaces.engine.coherent.shaped` with the shape worked out in each
    bin, those beyond the band's samples too."""
    n, count, per_band = spectrum.shape
    samples = np.log2(sample_frequencies(bands, count))
    x = np.log2(np.maximum(np.asarray(f, dtype=float), 1e-6))
    out = np.zeros((n, len(f)), dtype=complex)
    for b in range(count):
        support = np.flatnonzero(gains[b] > 0.0)
        lo, hi = int(support[0]), int(support[-1]) + 1
        own = spectrum[:, b]
        xs = np.clip(x[lo:hi], samples[b, 0], samples[b, -1])
        j = np.clip(np.searchsorted(samples[b], xs, side="right") - 1, 0, per_band - 2)
        w = (xs - samples[b, j]) / (samples[b, j + 1] - samples[b, j])
        magnitude = np.abs(own)
        phase = np.unwrap(np.angle(own), axis=1)
        level = magnitude[:, j] * (1.0 - w) + magnitude[:, j + 1] * w
        angle = phase[:, j] * (1.0 - w) + phase[:, j + 1] * w
        out[:, lo:hi] += (amplitude[:, b, None] * gains[b, lo:hi]) * (level * np.exp(1j * angle))
    return out


def test_the_shape_beyond_a_band_s_samples_is_held_bit_for_bit() -> None:
    """Below a band's first sample and above its last its shape is held, worked out
    once for those bins: the same to the last bit as bin by bin, at the echogram's
    rates and a rendered response's, for two coherent bands and for all eight."""
    from acoustic_spaces import synthesis
    from acoustic_spaces.engine.coherent import shaped

    rng = np.random.default_rng(4)
    for count, n_fft, rate in ((2, 1 << 12, 4000.0), (8, 1 << 14, 16000.0), (2, 1 << 15, 48000.0)):
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        gains = synthesis.band_gains(n_fft, rate, 8)[:count]
        size = (30, count, 8)
        spectrum = rng.uniform(0.2, 1.5, size) * np.exp(1j * rng.uniform(-np.pi, np.pi, size))
        amplitude = rng.uniform(0.0, 2.0, (30, count))
        np.testing.assert_array_equal(
            shaped(spectrum, 8, f, amplitude, gains),
            _shaped_bin_by_bin(spectrum, 8, f, amplitude, gains),
        )


def test_the_echograms_of_draws_read_what_they_share_as_their_own() -> None:
    """Paths that differ in their energies and phases only (draws of the absorption)
    share their delay phases and band grids (:class:`SameDelays`), and a draw's
    echograms each band's filtered pressure: the echograms are bit for bit those read
    without, and so when nothing may be kept."""
    from acoustic_spaces.engine.coherent import EarlyPressure, SameDelays, with_mean_phases

    delay = np.concatenate([0.01 + np.linspace(0.0, 0.002, 6), [0.03, 0.0301, 0.05]])
    paths = _arriving(delay, 1e-3, edge=True, together=3)
    paths.edge[[6, 7]] = -1  # two reflections among the paths over edges
    rng = np.random.default_rng(2)
    kept, none = SameDelays(), SameDelays(limit=0)
    for _ in range(3):
        drawn = dataclasses.replace(
            paths,
            energy=paths.energy * rng.uniform(0.5, 1.5, paths.energy.shape),
            spectrum=np.exp(1j * rng.uniform(-1.0, 1.0, paths.spectrum.shape)),
        )
        plain = EarlyPressure(with_mean_phases(drawn))
        for store in (kept, none):
            shared = EarlyPressure(with_mean_phases(drawn, store), kept=store)
            for axis, exponent in ((None, 2), ([0.6, 0.8, 0.0], 2), ([0.0, 1.0, 0.0], 1)):
                for _again in range(2):
                    np.testing.assert_array_equal(
                        shared.energy(1e-3, 120, axis, exponent, 0.0, 0.01),
                        plain.energy(1e-3, 120, axis, exponent, 0.0, 0.01),
                    )
    assert kept._phases and kept._grids and 0 < kept.size <= kept.limit
    assert not any(a.flags.writeable for a in kept._phases.values())
    assert none.size == 0 and not none._phases and not none._grids
    # A band's filtered pressure is read once per draw, and is read-only.
    first = shared.band(0, 1e-3, 120, 0.0, 0.01)
    assert shared.band(0, 1e-3, 120, 0.0, 0.01) is first and not first[0].flags.writeable
    assert plain.band(0, 1e-3, 120, 0.0, 0.01)[0] is not plain.band(0, 1e-3, 120, 0.0, 0.01)[0]


def _arriving(delay, energy: float, edge: bool, together: int = 1) -> early_stage.EarlyReflections:
    """Paths straight ahead at ``delay`` (s), over edges of their own or reflections, with
    ``energy`` in all in each band. In the two coherent bands they are in phase, each
    with its share of the pressure, so that there too they come to ``energy`` when they
    arrive ``together`` (their count) or apart (1)."""
    delay = np.atleast_1d(np.asarray(delay, dtype=float))
    n = len(delay)
    energies = np.full((n, 8), energy / n)
    energies[:, :2] = energy / (n * together)
    return early_stage.EarlyReflections(
        order=np.ones(n, dtype=np.int64),
        delay_s=delay,
        distance_m=delay * 343.0,
        direction=np.tile([1.0, 0.0, 0.0], (n, 1)),
        energy=energies,
        edge=np.arange(n) if edge else None,
        spectrum=np.ones((n, 2, 8), dtype=complex),
    )


def _early(paths: early_stage.EarlyReflections, seed: int = 0) -> Response:
    """A response of ``paths`` alone."""
    return Response(
        source="s",
        listener="l",
        early=paths,
        late=np.zeros((8, 100)),
        time_step=1e-3,
        length_s=0.1,
        direct_delay_s=0.04,
        transition_s=0.1,
        volume_m3=None,
        speed_of_sound=343.0,
        seed=seed,
    )


def _octaves(paths: early_stage.EarlyReflections, seed: int = 0) -> np.ndarray:
    """The energy of each octave band of the mono impulse response of ``paths``."""
    ir = _early(paths, seed).impulse_response(48_000)
    return (octave_bands(ir, 48_000) ** 2).sum(axis=1)


def test_paths_over_edges_that_arrive_together_render_their_energy() -> None:
    # Forty pieces of sound bent round edges within a millisecond, as the facets of a
    # curved surface give: each one's energy holds its share of their interference
    # already (engine.clusters). Each takes a phase of its own, drawn from the seed, so
    # that they add in energy on average, as one path of all their energy renders.
    delay = 0.05 + np.linspace(0.0, 0.001, 40)
    alone = _octaves(_arriving(0.0505, 1e-3, edge=True))
    pieces = _arriving(delay, 1e-3, edge=True, together=40)
    rendered = np.array([_octaves(pieces, seed) for seed in range(100)]) / alone
    np.testing.assert_allclose(rendered.mean(axis=0)[2:], 1.0, atol=0.2)
    # One response scatters about that, as in a room from seat to seat.
    assert rendered[:, 2:].std(axis=0).min() > 0.2
    # Reflections of one phase arriving so close add in pressure, at a point of a room
    # as here: many times their energy where the band's period is longer than they are
    # apart.
    reflections = _octaves(_arriving(delay, 1e-3, edge=False, together=40)) / alone
    assert reflections[2] > 20.0 and reflections[3] > 10.0


def test_the_echogram_reads_paths_over_edges_that_arrive_together_once() -> None:
    # The 63 and 125 Hz bands are read through filters that let in a little of the
    # 250 Hz band. There the pieces' energies hold their interference already: they add
    # nothing more in pressure, so how much energy they carry there leaves the low bands
    # as they are, their energy in them.
    delay = 0.05 + np.linspace(0.0, 0.001, 40)

    def low(edge: bool, louder: float) -> np.ndarray:
        paths = _arriving(delay, 1e-3, edge=edge, together=40)
        paths.energy[:, 2:] *= louder
        return _early(paths).energy().sum(axis=1)[:2]

    np.testing.assert_allclose(low(True, 10.0), low(True, 1.0), rtol=1e-9)
    np.testing.assert_allclose(low(True, 1.0), 1e-3, rtol=0.1)


def test_a_path_that_arrives_alone_keeps_its_phase() -> None:
    # Over an edge, 40 ms after two reflections: farther from them than the band filters
    # ring, it keeps the phase the coherent bands give it, which carries its bands into
    # theirs.
    paths = early_stage.EarlyReflections.concatenate(
        [_arriving([0.01, 0.0102], 1e-3, edge=False), _arriving(0.05, 1e-3, edge=True)], 8
    )
    paths.edge[2] = 5
    assert arriving_with(paths)[2] < 0.01
    np.testing.assert_allclose(own_phases(paths, 0), 1.0, atol=0.01)
    assert np.all(own_phases(paths, 0)[:2] == 1.0)  # reflections are never turned
    # Two over edges at one delay: each hears the other as loud as itself.
    two = _arriving([0.05, 0.05], 1e-3, edge=True, together=2)
    np.testing.assert_allclose(arriving_with(two), 1.0, rtol=1e-6)
    # Paths that add in energy in every band, or without paths over edges, stay as they are.
    without = _arriving([0.05, 0.05], 1e-3, edge=False)
    assert with_own_phases(without, 0) is without
    coherent = dataclasses.replace(two, spectrum=np.ones((2, 8, 8), dtype=complex))
    assert with_own_phases(coherent, 0) is coherent


def test_each_path_over_an_edge_takes_the_same_phase_wherever_it_is() -> None:
    delay = 0.05 + np.linspace(0.0, 0.001, 10)
    paths = _arriving(delay, 1e-3, edge=True, together=10)
    phases = own_phases(paths, 7)
    # Whatever order the paths come in, and wherever the source and listener stand.
    np.testing.assert_array_equal(own_phases(paths[::-1], 7), phases[::-1])
    later = dataclasses.replace(paths, delay_s=paths.delay_s + 0.0123)
    np.testing.assert_allclose(own_phases(later, 7), phases, atol=1e-12)
    # Another seed draws other phases; the same one, the same response.
    assert np.abs(np.angle(own_phases(paths, 8) / phases)).max() > 0.5
    np.testing.assert_array_equal(_octaves(paths, 3), _octaves(paths, 3))


def test_sound_along_a_ground_zone_without_an_impedance_takes_its_fitted_admittance() -> None:
    # Along a zone of absorbing ground (no impedance of its own) the wave at grazing
    # incidence is the Weyl-van der Pol field 2 F(w) of the admittance fitted to its
    # absorption, as along any face of that absorption; not the plane wave's factor,
    # -1 at grazing incidence, which left 2 F(w) = 1 + Q near 0.
    from scipy.optimize import brentq
    from scipy.special import wofz

    zone = [[-5, -5], [5, -5], [5, 5], [-5, 5]]
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": {
                        "extent": [[-20, -20], [20, 20]],
                        "sky": 20,
                        "zones": [{"name": "bed", "polygon": zone}],
                    }
                },
                "materials": {
                    "grass": {"impedance": "grass"},
                    "felt": {"absorption": [0.02] * 8, "scattering": 0.0},
                },
                "assignments": {"ground": "grass", "bed": "felt"},
                "sources": [{"name": "s", "position": [0.0, 0.0, 1.0]}],
                "listeners": [{"name": "l", "position": [3.0, 0.0, 1.0]}],
            }
        )
    )
    model = TraceModel.from_room(room)
    faces = np.arange(len(model.absorption))
    bed = faces[model.factor_faces(faces) & ~model.impedance_faces(faces)]
    assert len(bed) and np.all(model.absorption[bed] == 0.02)

    def paris(z):  # random-incidence absorption of a real normalised impedance z
        return 8.0 / z * (1.0 + 1.0 / (1.0 + z) - 2.0 / z * math.log(1.0 + z))

    beta = 1.0 / brentq(lambda z: paris(z) - 0.02, 1.0, 1e6)
    w = 0.5 * (1 + 1j) * np.sqrt(2 * np.pi * np.array(model.band_hz) * 10.0 / 343.0) * beta
    hand = 20 * np.log10(np.abs(1 + 1j * math.sqrt(math.pi) * w * wofz(w)))  # 10 m along it
    q = ground_wave_factors(model, bed[:1], np.array([10.0]))[0]
    level = 10 * np.log10(np.mean(np.abs(0.5 * (1.0 + q)) ** 2, axis=1))
    np.testing.assert_allclose(level, hand, atol=0.05)


# -- the parabolic equation's field at fine frequencies -----------------------------------


def _two_arrivals(f, delay: float = 0.29e-3, reflected: float = -0.8):
    """The pressure relative to the first arrival of two, the second ``delay`` s later
    (as the ground's reflection 100 m downwind in the benchmark's case 2)."""
    return 1.0 + reflected * np.exp(-2j * np.pi * np.asarray(f) * delay)


def test_a_band_s_shape_through_the_equation_s_samples_meets_its_bound() -> None:
    """A field of two arrivals 0.29 ms apart, sampled as the parabolic equation's runs
    take it in the 1 and 2 kHz bands (the 2 kHz band through its upper crossover):
    across both crossovers the shape is within the bound of linear interpolation of the
    complex pressure, h^2 / 8 times the sum of |a_i| (2 pi dtau_i)^2 (h = 58.9 Hz, the
    2 kHz band's spacing), and the estimate read from the samples gives the error
    found within 1 %. Held past the band's own eight samples the shape is off by more
    than 20 times the bound."""
    from acoustic_spaces.engine import fine as fine_stage
    from acoustic_spaces.engine.coherent import _band_shape, shaped

    bands = count = 8
    runs = {
        4: [
            (at, _two_arrivals(at)) for at in [fine_stage.band_fine_frequencies(8, 4, True, False)]
        ],
        5: [(at, _two_arrivals(at)) for at in [fine_stage.band_fine_frequencies(8, 5, True)]],
    }
    fine = fine_stage.placed(bands, count, runs, 0.0)[None]
    delay = np.array([fine_stage.reference_delay(fine[0], bands)])
    f = np.arange(1000.0, 3364.0, 1.0)
    gains = np.array([_band_shape(f, b, bands) for b in range(bands)])
    amplitude = np.zeros((1, bands))
    for band in (4, 5):
        amplitude[0, band] = np.sqrt(np.mean(np.abs(_two_arrivals(runs[band][0][0][:24])) ** 2))
    out = shaped(np.ones((1, count, 8), dtype=complex), bands, f, amplitude, gains, fine, delay)[0]
    expected = _two_arrivals(f) * (gains[4] + gains[5])
    # Relative to the reference delay (between 0 and 0.29 ms) each arrival's dtau is at
    # most 0.29 ms.
    bound = 58.93**2 / 8.0 * (1.0 + 0.8) * (2.0 * np.pi * 0.29e-3) ** 2
    error = np.abs(out - expected)
    assert error.max() <= bound, (error.max(), bound)
    estimate = fine_stage.interpolation_bound(fine, delay, bands)[0, 4:6] * amplitude[0, 4:6]
    assert np.all(estimate <= bound) and np.isclose(estimate.max(), error.max(), rtol=0.01)
    # The same field held past the band's own eight samples, as without fine samples.
    at = sample_frequencies(bands, count)
    spectrum = np.ones((1, count, 8), dtype=complex)
    spectrum[0, 4:6] = _two_arrivals(at[4:6]) / amplitude[0, 4:6, None]
    held = shaped(spectrum, bands, f, amplitude, gains)[0]
    assert np.abs(held - expected).max() > 20 * bound


def test_the_layout_holds_every_run_s_frequencies() -> None:
    """In octave and third-octave bands the fine layout holds the frequencies of the
    open ground's runs at 24 an octave from 1 kHz and at the early paths' 8 (their
    skirts too, each at its own spacing, the last at or past the top of the band's upper
    crossover), the early paths' own among them; a frequency off it is an error."""
    from acoustic_spaces import synthesis
    from acoustic_spaces.engine import fine as fine_stage

    for bands in (8, 24):
        layout = fine_stage.fine_frequencies(bands, bands)
        early = sample_frequencies(bands, bands)
        top = synthesis.crossover_frequencies(bands) * 2.0 ** (synthesis.crossover_width(bands) / 2)
        for band in range(bands):
            row = layout[band][np.isfinite(layout[band])]
            for dense in (True, False):
                at = fine_stage.band_fine_frequencies(bands, band, dense, band + 1 < bands)
                assert np.all(np.isclose(row[np.searchsorted(row, at - 1e-6)], at, rtol=1e-9))
                if band + 1 < bands:
                    assert at[-1] >= top[band] and at[-2] < top[band]
            plain = fine_stage.band_fine_frequencies(bands, band, False, skirt=False)
            np.testing.assert_allclose(plain, early[band], rtol=1e-12)
    with pytest.raises(ValueError, match="not on its layout"):
        fine_stage.placed(8, 8, {5: [(np.array([2000.5]), np.ones(1))]}, 0.0)


def test_paths_without_the_equation_s_samples_are_shaped_as_before() -> None:
    """Rows without fine samples take the band's own spectrum exactly as before (bit for
    bit), whatever the other rows carry; a room's paths carry none."""
    from acoustic_spaces.engine import fine as fine_stage
    from acoustic_spaces.engine.coherent import _band_shape, shaped

    rng = np.random.default_rng(1)
    spectrum = rng.normal(size=(3, 8, 8)) + 1j * rng.normal(size=(3, 8, 8))
    amplitude = rng.uniform(0.5, 1.0, size=(3, 8))
    f = np.arange(20.0, 8000.0, 3.0)
    gains = np.array([_band_shape(f, b, 8) for b in range(8)])
    plain = shaped(spectrum, 8, f, amplitude, gains)
    layout = fine_stage.fine_frequencies(8, 8)
    fine = np.full((3, 8, layout.shape[1]), np.nan, dtype=complex)
    np.testing.assert_array_equal(
        shaped(spectrum, 8, f, amplitude, gains, fine, np.zeros(3)), plain
    )
    fine[1] = fine_stage.placed(8, 8, {5: [(layout[5][:24], np.ones(24))]}, 0.0)
    mixed = shaped(spectrum, 8, f, amplitude, gains, fine, np.zeros(3))
    np.testing.assert_array_equal(mixed[[0, 2]], plain[[0, 2]])
    assert not np.allclose(mixed[1], plain[1])
    response = simulate(_box(500), threads=1)[("s", "l")]
    assert response.early.fine is None and response.early.fine_delay is None


def _digest(values) -> list[float]:
    """Three sums of an array that any change in it moves: of the values, of their
    squares and of the values times their index."""
    x = np.asarray(values, dtype=float).ravel()
    return [float(np.sum(x)), float(np.sum(x * x)), float(np.sum(x * np.arange(len(x))))]


def test_a_room_s_early_signal_and_echogram_are_those_without_the_equation() -> None:
    """An indoor box with first-order diffraction: the early paths' coherent signal
    (8192 samples at 16 kHz) and their echogram (1 ms bins) are, to 1e-12, those of the
    main checkout at ae79163, which has no fine samples (``main_values.py box``,
    /mnt/project-files/wind-fixes-2026-11-10/crossover/build-scripts/)."""
    from acoustic_spaces.engine import coherent

    room = _box(500, early={"diffraction": "first_order", "coherent_max_band_hz": 500})
    early = simulate(room, threads=1)[("s", "l")].early
    signal = coherent.signal(early, 8192, 16000.0)[:, 0]
    echogram = coherent.EarlyPressure(early).energy(0.001, 200)
    np.testing.assert_allclose(
        _digest(signal), [5.665402896963265, 0.12028315532403483, 2405.64534965662], rtol=1e-12
    )
    np.testing.assert_allclose(
        _digest(echogram),
        [0.20203985979353015, 0.0007957309819662612, 60.02199481586753],
        rtol=1e-12,
    )


def test_the_equation_s_samples_go_with_their_path() -> None:
    """A path's fine samples and their delay survive picking, sorting, joining with paths
    that have none (NaN and 0 for those) and replacing other fields; re-weighting for
    another absorption turns none of them, as the carrier's reflections have no phase of
    the faces' to turn (their cosines NaN)."""
    from acoustic_spaces.engine.early import EarlyReflections
    from acoustic_spaces.engine.reweight import turned_paths

    def paths(n, fine):
        return EarlyReflections(
            order=np.arange(n),
            delay_s=np.linspace(0.1, 0.2, n),
            distance_m=np.full(n, 34.0),
            direction=np.tile([1.0, 0.0, 0.0], (n, 1)),
            energy=np.ones((n, 8)),
            faces=[np.zeros(k, dtype=np.int64) for k in range(n)],
            spectrum=np.ones((n, 8, 8), dtype=complex),
            cosines=[np.full(k, np.nan) for k in range(n)],
            fine=None if fine is None else np.full((n, 8, 5), fine, dtype=complex),
            fine_delay=None if fine is None else np.full(n, 1e-4),
        )

    carrier = paths(2, 2.0 + 1.0j)
    joined = EarlyReflections.concatenate([paths(1, None), carrier], 8)
    assert joined.fine is not None and joined.fine_delay is not None
    np.testing.assert_array_equal(np.isnan(joined.fine.real).all(axis=(1, 2)), [True, False, False])
    np.testing.assert_array_equal(joined.fine_delay, [0.0, 1e-4, 1e-4])
    picked = joined[[2]]
    np.testing.assert_array_equal(picked.fine, carrier.fine[[1]])
    kept = dataclasses.replace(picked, energy=2.0 * picked.energy)
    assert kept.fine is picked.fine and kept.fine_delay is picked.fine_delay
    turned = turned_paths(
        carrier, np.zeros(1, dtype=np.int64), np.full((1, 8), 0.1), np.full((1, 8), 0.5)
    )
    np.testing.assert_array_equal(turned.spectrum, carrier.spectrum)
    np.testing.assert_array_equal(turned.fine, carrier.fine)
    assert paths(1, None).fine is None


_PRESSURE_DIGEST = """
import hashlib
import numpy as np
from acoustic_spaces.engine import coherent
from acoustic_spaces.engine.early import EarlyReflections

rng = np.random.default_rng(5)
n = 203
delay = np.sort(rng.uniform(0.005, 0.3, n))
energy = rng.uniform(0.1, 1.0, (n, 8)) * 1e-3
energy[:2, 1] = 0.0
paths = EarlyReflections(
    order=np.ones(n, dtype=np.int64),
    delay_s=delay,
    distance_m=delay * 343.0,
    direction=rng.normal(size=(n, 3)),
    energy=energy,
    edge=None,
    spectrum=np.exp(1j * rng.uniform(-3.0, 3.0, (n, 2, 8))),
)
pressure = coherent.EarlyPressure(paths)
digest = hashlib.sha256()
for x in (pressure.spectra, *pressure._spectra_of.values(), coherent.signal(paths, 4096, 16000.0)):
    digest.update(np.ascontiguousarray(x).tobytes())
print(digest.hexdigest())
"""


def test_the_early_pressure_has_the_same_bits_on_any_number_of_library_threads() -> None:
    """203 paths, two of them silent in the second band: the early part's spectra, those
    of the 201 paths heard in that band, and the coherent signal are bit for bit the same
    in fresh processes on one and on four OpenBLAS threads."""
    digests = set()
    for threads in ("1", "4"):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=threads)
        run = subprocess.run(
            [sys.executable, "-c", _PRESSURE_DIGEST],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(run.stdout)
    assert len(digests) == 1


def _scattered(n: int, seed: int, wander: bool = False) -> early_stage.EarlyReflections:
    """``n`` paths from all directions, with random delays, energies and phases, every
    seventh silent in the second band; with ``wander``, apart by random delay variances."""
    from acoustic_spaces.engine.turbulence import Wander

    rng = np.random.default_rng(seed)
    delay = np.sort(rng.uniform(0.005, 0.3, n))
    energy = rng.uniform(0.1, 1.0, (n, 8)) * 1e-3
    energy[::7, 1] = 0.0
    variance = rng.uniform(0.0, 1e-7, (n, n))
    variance = variance + variance.T
    np.fill_diagonal(variance, 0.0)
    return early_stage.EarlyReflections(
        order=np.ones(n, dtype=np.int64),
        delay_s=delay,
        distance_m=delay * 343.0,
        direction=rng.normal(size=(n, 3)),
        energy=energy,
        edge=None,
        spectrum=np.exp(1j * rng.uniform(-3.0, 3.0, (n, 2, 8))),
        wander=Wander(variance, np.zeros((n, n)), 343.0) if wander else None,
    )


def test_the_early_pressure_and_the_signal_add_every_path_of_every_chunk() -> None:
    """600 paths, three chunks of the loop: the early pressure's spectra, those of the
    paths heard in the second band, and the coherent signal of three weightings are the
    weighted sums over all the paths' delayed spectra at once, to rounding."""
    from acoustic_spaces import synthesis
    from acoustic_spaces.engine import coherent

    paths = _scattered(600, 5)
    pressure = coherent.EarlyPressure(paths)
    f = np.fft.rfftfreq(pressure.n_fft, 1.0 / pressure.rate)
    gains = synthesis.band_gains(pressure.n_fft, pressure.rate, 8)
    shift = paths.delay_s + pressure.lead - 0.5 / pressure.rate
    spectra = coherent.amplitude_spectra(paths, f, gains) * np.exp(-2j * np.pi * np.outer(shift, f))
    weights = np.column_stack([np.ones(len(paths)), paths.direction])
    np.testing.assert_allclose(pressure.spectra, weights.T @ spectra, rtol=0, atol=1e-12)
    heard = paths.energy[:, 1] > 0.0
    assert list(pressure._spectra_of) == [1]
    np.testing.assert_allclose(
        pressure._spectra_of[1], weights[heard].T @ spectra[heard], rtol=0, atol=1e-12
    )

    n, rate = 4096, 16000.0
    weights = np.random.default_rng(1).normal(size=(len(paths), 3))
    n_fft = synthesis.fft_size(n, rate)
    gains = synthesis.band_gains(n_fft, rate, 8)
    top = int(np.flatnonzero(gains[:2].sum(axis=0) > 0.0).max()) + 1
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)[:top]
    spectra = coherent.amplitude_spectra(paths, f, gains[:, :top], only_coherent=True)
    spectra *= np.exp(-2j * np.pi * np.outer(paths.delay_s, f))
    expected = np.zeros((n_fft // 2 + 1, 3), dtype=complex)
    expected[:top] = spectra.T @ weights
    expected = np.fft.irfft(expected, n_fft, axis=0)[:n]
    np.testing.assert_allclose(
        coherent.signal(paths, n, rate, weights),
        expected,
        rtol=0,
        atol=1e-13 * np.abs(expected).max(),
    )


def test_the_early_figure_of_eight_is_the_pressure_weighted_by_its_cosines() -> None:
    """Facing an axis of unequal components, the early pressure's figure of eight is the
    energy of the paths each weighted by the cosine of its direction to the axis."""
    from acoustic_spaces.engine import coherent

    paths = _scattered(300, 2)
    axis = np.array([0.6, 0.0, 0.8])
    facing = coherent.EarlyPressure(paths).energy(0.001, 300, axis=axis)
    weighted = coherent.EarlyPressure(paths, weights=(paths.direction @ axis)[:, None])
    expected = weighted.energy(0.001, 300)
    np.testing.assert_allclose(facing, expected, rtol=0, atol=1e-12 * expected.max())


def test_the_turbulent_figure_of_eight_takes_each_two_paths_coherence() -> None:
    """In turbulence, a figure of eight facing an axis with exponent 1 adds, for each two
    paths, the first's cosine times its band signal, their coherence factor, and the
    second's pressure: the einsum of these, through the band filters, to rounding."""
    from acoustic_spaces.engine import coherent
    from acoustic_spaces.materials.bands import resolution_of

    paths = _scattered(150, 3, wander=True)
    pressure = coherent.EarlyPressure(paths)
    axis = np.array([0.6, 0.0, 0.8])
    facing = pressure.energy(0.001, 300, axis=axis, exponent=1)
    side = paths.direction @ axis
    for b in range(pressure.count):
        gain, keep, slot, scale = pressure._filter(b, 0.001, 300, 0.0, None)
        signals = np.fft.irfft(gain * pressure._each, pressure.n_fft, axis=1)[:, keep]
        signals[paths.energy[:, b] <= 0.0] = 0.0
        f = band_frequencies(bands=resolution_of(8))[0][b]
        together = paths.wander.coherence(f).mean(axis=2)
        power = np.abs(np.einsum("it,ij,jt->t", side[:, None] * signals, together, signals))
        expected = np.bincount(slot, power * scale, minlength=300)[:300]
        np.testing.assert_allclose(facing[b], expected, rtol=0, atol=1e-12 * expected.max())


def _through_a_corner(paths, corner, within: float) -> list[int]:
    """The paths that reflect three times within ``within`` of ``corner``."""
    return [
        i
        for i in range(len(paths))
        if paths.edge[i] < 0
        and paths.order[i] == 3
        and all(math.dist(point, corner) <= within for point in paths.points[i][1:4])
    ]


def _beside(scene: dict, shift):
    """The early paths of the scene's listener moved by ``shift``."""
    moved = (np.asarray(scene["listeners"][0]["position"]) + shift).tolist()
    listeners = [{"name": "l", "position": moved}]
    return early_paths(Room.from_scene(Scene.model_validate({**scene, "listeners": listeners})))


#: A room with a source 1.9 m from a corner and a listener on the line from that corner
#: through the source, twice as far: the corner's three walls send sound back along
#: the line it came on, so that the path between the two that reflects off all three
#: does so at the corner point itself.
ROOM_CORNER = {
    "geometry": {"box": [7.3, 5.1, 3.0]},
    "materials": {"m": {"absorption": [0.15] * 8, "scattering": 0.0}},
    "default_material": "m",
    "sources": [{"name": "s", "position": [1.0, 1.4, 0.8]}],
    "listeners": [{"name": "l", "position": [2.0, 2.8, 1.6]}],
}
#: Two barriers that meet at a right angle, on a ground, with a source and a listener on
#: one line from the foot of their corner (see :data:`ROOM_CORNER`).
BARRIERS_CORNER = {
    "geometry": {
        "ground": {"extent": [[-30, -30], [30, 30]], "sky": 30},
        "barriers": [{"name": "b", "path": [[0, 12], [0, 0], [12, 0]], "height": 5}],
    },
    "materials": {
        "hard": {"absorption": [0.2] * 8, "scattering": 0.0},
        "grass": {"impedance": "grass"},
    },
    "default_material": "hard",
    "sources": [{"name": "s", "position": [1.5, 2.0, 1.2]}],
    "listeners": [{"name": "l", "position": [3.0, 4.0, 2.4]}],
}


@pytest.mark.parametrize(
    ("scene", "early"),
    [
        (ROOM_CORNER, {}),
        (ROOM_CORNER, {"method": "image_sources", "coherent_max_band_hz": 500}),
        (BARRIERS_CORNER, {}),
        (BARRIERS_CORNER, {"method": "image_sources", "diffraction": "none"}),
        ({**BARRIERS_CORNER, "assignments": {"ground": "grass"}}, {}),
    ],
    ids=["a room", "a room, image sources", "barriers", "barriers, no edges", "on grass"],
)
def test_a_path_that_reflects_three_times_at_one_point_meets_each_wall_at_its_angle(
    scene: dict, early: dict
) -> None:
    # The path that reflects off three walls right in their corner meets each of them at
    # the angle of the line it comes and leaves on: also the middle one of the three,
    # which it reaches and leaves by no leg of any length. Its pressure is then that of
    # the path that reflects off the three one after the other, as it does for a
    # listener 0.1 mm beside that line, and the two listeners' early levels agree.
    # (Taken to graze the middle wall, the path had its pressure turned over where the
    # bands add in pressure: the level in line was 0.7 dB off at 63 Hz in the room, 1.6
    # dB between the barriers on the hard ground and 2.3 dB on grass, with the default
    # settings.)
    scene = {**scene, "simulation": {"early": early}}
    exact = _beside(scene, np.zeros(3))
    (corner,) = _through_a_corner(exact, np.zeros(3), 1e-9)
    towards = -np.asarray(scene["sources"][0]["position"])
    towards /= math.sqrt(float(towards @ towards))
    if "assignments" not in scene:  # (a ground with an impedance keeps no cosine)
        np.testing.assert_allclose(
            np.sort(exact.cosines[corner]), np.sort(np.abs(towards)), atol=1e-9
        )
    level = 10 * np.log10(total_energy(exact))
    for shift in ((0, 0, 1e-4), (1e-4, 0, 0), (0, -1e-4, 0)):
        beside = _beside(scene, shift)
        assert len(beside) == len(exact)
        (other,) = _through_a_corner(beside, np.zeros(3), 1e-3)
        assert not _through_a_corner(beside, np.zeros(3), 1e-9)
        np.testing.assert_allclose(exact.spectrum[corner], beside.spectrum[other], atol=1e-4)
        np.testing.assert_allclose(level, 10 * np.log10(total_energy(beside)), atol=0.01)

    # The three surfaces are at right angles, so every order of them gives this path: its
    # pressure and its energy are the same whichever it is put together in, without the
    # scattering of a surface's size, which only the surface met first takes. (With a
    # ground that has an impedance in the middle, the angle of 0 had also given the path
    # another energy: 1.3 dB more at 63 Hz and 6.1 dB more at 250 Hz on the grass.)
    scene["simulation"]["scattering"] = {"from_size": False}
    room = Room.from_scene(Scene.model_validate(scene))
    model = TraceModel.from_room(room)
    source, listener = (
        np.asarray(scene[end][0]["position"], dtype=float) for end in ("sources", "listeners")
    )
    bands = exact.energy.shape[1]
    # (With no band that adds in pressure, a ground's factor still takes its angle.)
    for coherent in (exact.coherent_bands, 0):
        orders = [
            early_stage.assemble_paths(
                model,
                source,
                listener,
                [np.zeros((1, 3))] * 3 + [listener[None]],
                np.array([faces]),
                exact.distance_m[[corner]],
                np.ones((1, bands)),
                np.zeros(1, dtype=np.int64),
                speed_of_sound=room.speed_of_sound(),
                air=np.zeros(bands),
                coherent=coherent,
            )
            for faces in itertools.permutations(exact.faces[corner])
        ]
        assert len(orders) == 6
        for order in orders:
            np.testing.assert_allclose(order.energy[0], orders[0].energy[0], rtol=1e-9)
            if coherent:
                np.testing.assert_allclose(order.spectrum[0], exact.spectrum[corner], atol=1e-6)


def test_planes_at_one_point_turn_a_path_from_one_to_the_next() -> None:
    # A roof, the wall of a taller building along its edge and a barrier that stands on
    # the roof at 80 degrees to that wall: three planes at one point that are not all at
    # right angles, so that the angle a path meets the wall at there depends on how the
    # barrier turned it. For a listener in line with that point and the image of the
    # source in the three, the path that reflects off all three there meets each at the
    # angle the path 0.1 mm beside does (0.37, 0.60 and 0.80 as cosines), and the early
    # levels agree (the level in line had been 1.1 dB off at 63 Hz).
    corner = np.array([10.0, 5.0, 6.0])
    along = np.radians(170.0)
    end = [10 + 8 * math.cos(along), 5 + 8 * math.sin(along)]
    normals = np.array([[-math.sin(along), math.cos(along), 0.0], [1.0, 0, 0], [0, 0, 1.0]])
    source = corner + (2.5 * math.cos(np.radians(130.0)), 2.5 * math.sin(np.radians(130.0)), 1)
    image = source
    for normal in normals:
        image = image - 2.0 * ((image - corner) @ normal) * normal
    scene = {
        "geometry": {
            "ground": {"margin": 60},
            "buildings": [
                {"name": "low", "footprint": [[0, 0], [10, 0], [10, 20], [0, 20]], "height": 6},
                {"name": "tall", "footprint": [[10, 0], [14, 0], [14, 20], [10, 20]], "height": 12},
            ],
            "barriers": [{"name": "b", "path": [[10, 5], end], "height": 9}],
        },
        "materials": {"hard": {"absorption": [0.2] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": source.tolist()}],
        "listeners": [{"name": "l", "position": (corner + 1.5 * (corner - image)).tolist()}],
        "simulation": {"early": {"method": "image_sources", "max_order": 3, "diffraction": "none"}},
    }
    exact, beside = _beside(scene, np.zeros(3)), _beside(scene, (0, 0, 1e-4))
    (at,) = _through_a_corner(exact, corner, 1e-9)
    (near,) = _through_a_corner(beside, corner, 1e-3)
    assert len(exact) == len(beside) and not _through_a_corner(beside, corner, 1e-9)
    np.testing.assert_allclose(exact.cosines[at], beside.cosines[near], atol=1e-4)
    np.testing.assert_allclose(exact.cosines[at], [0.3714, 0.5968, 0.8041], atol=1e-4)
    np.testing.assert_allclose(
        10 * np.log10(total_energy(exact)), 10 * np.log10(total_energy(beside)), atol=0.01
    )

    # The roof is at right angles to the other two, so the three orders that have the
    # barrier before the wall give the same image: whichever of them the path is put
    # together in, it meets the planes at those three angles, also where the wall is the
    # middle one, met as the barrier turned the path.
    model = TraceModel.from_room(Room.from_scene(Scene.model_validate(scene)))
    own = model.plane_normal[model.face_plane[exact.faces[at]]]
    faces = [
        int(exact.faces[at][np.flatnonzero(np.abs(own @ normal) > 1.0 - 1e-9)[0]])
        for normal in normals
    ]
    assert len(set(faces)) == 3
    bands = model.absorption.shape[1]

    def cosines(planes, source, listener, points=None, tolerance=0.0) -> np.ndarray:
        """The cosines of a path off ``planes`` (the barrier 0, the wall 1, the roof 2),
        all at the corner or at ``points``."""
        points = [corner] * len(planes) if points is None else points
        found = early_stage.assemble_paths(
            model,
            source,
            listener,
            [point[None] for point in (*points, listener)],
            np.array([[faces[plane] for plane in planes]]),
            np.array([math.dist(source, corner) + math.dist(corner, listener)]),
            np.ones((1, bands)),
            np.zeros(1, dtype=np.int64),
            speed_of_sound=343.0,
            air=np.zeros(bands),
            coherent=2,
            tolerance=tolerance,
        )
        return found.cosines[0]

    def mirrored(start, planes) -> list[float]:
        """The cosines at which a path from ``start`` meets ``planes`` at the corner, one
        after the other, each turning it for the next."""
        heading = (corner - start) / math.dist(corner, start)
        out = []
        for normal in normals[list(planes)]:
            out.append(abs(float(normal @ heading)))
            heading = heading - 2.0 * (normal @ heading) * normal
        return out

    in_line = corner + 1.5 * (corner - image)
    for planes in ((2, 0, 1), (0, 2, 1), (0, 1, 2)):
        np.testing.assert_allclose(
            np.sort(cosines(planes, source, in_line)), [0.3714, 0.5968, 0.8041], atol=1e-4
        )
    np.testing.assert_allclose(cosines((0, 1, 2), source, in_line)[1], 0.8041, atol=1e-4)

    # Four reflections at the point, off the barrier, the wall, the barrier and the
    # roof, to a listener in line with the source's image in the four: the angles are
    # those the source's leg gives, turned from plane to plane. The path is given them
    # from the leg that leaves the corner, turned back; from the leg into it where none
    # leaves (a listener at the corner); and from the leg that leaves alone for a source
    # at the corner. Legs within the tolerance count as none.
    planes = (0, 1, 0, 2)
    image = source
    for normal in normals[list(planes)]:
        image = image - 2.0 * ((image - corner) @ normal) * normal
    in_line = corner + 1.5 * (corner - image)
    expected = mirrored(source, planes)
    assert abs(expected[2] - expected[0]) > 0.1 and abs(expected[1] - 0.8041) < 1e-4
    np.testing.assert_allclose(cosines(planes, source, in_line), expected, atol=1e-9)
    np.testing.assert_allclose(cosines(planes, source, corner), expected, atol=1e-9)
    np.testing.assert_allclose(
        cosines(planes, corner, in_line), mirrored(in_line, planes[::-1])[::-1], atol=1e-9
    )
    hair = [corner, corner + (1e-12, 0, 2e-12), corner + (0, -1e-9, 1e-9), corner]
    np.testing.assert_allclose(
        cosines(planes, source, in_line, hair, tolerance=1e-6), expected, atol=1e-6
    )
    assert np.abs(cosines(planes, source, in_line, hair) - expected).max() > 0.1


def test_a_source_in_the_edge_of_a_room_meets_both_walls_at_their_angles() -> None:
    # A source 10 nm from the floor and from a wall, in their edge for the search: the
    # path that reflects off the two right there has no leg before either reflection,
    # and takes the angles of the leg that leaves the edge for the listener (its
    # pressure had been turned over by an angle of 0 at the first).
    scene = {
        **ROOM_CORNER,
        "sources": [{"name": "s", "position": [4.0, 1e-8, 1e-8]}],
        "listeners": [{"name": "l", "position": [4.1, 2.2, 1.3]}],
        "simulation": {"early": {"method": "image_sources", "diffraction": "none"}},
    }
    paths = _beside(scene, np.zeros(3))
    (both,) = [
        i
        for i in range(len(paths))
        if paths.order[i] == 2
        and all(math.dist(point, (4.0, 0.0, 0.0)) < 1e-6 for point in paths.points[i][1:3])
    ]
    leaves = np.array([0.1, 2.2, 1.3]) / math.sqrt(0.1**2 + 2.2**2 + 1.3**2)
    np.testing.assert_allclose(np.sort(paths.cosines[both]), np.sort(leaves[1:]), atol=1e-6)
    np.testing.assert_allclose(paths.spectrum[both], 1.0, atol=1e-9)
