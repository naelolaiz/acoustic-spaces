"""Third-octave bands: the band tables, coefficients and directivity in either kind of
band, the filter banks, and simulations in third-octave bands against octave ones."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate, synthesis
from acoustic_spaces.air import energy_attenuation_per_m
from acoustic_spaces.directivity import Balloon, FirstOrder
from acoustic_spaces.engine import coverage_map, early_paths
from acoustic_spaces.materials import OCTAVE_BANDS_HZ, THIRD_OCTAVE_BANDS_HZ
from acoustic_spaces.materials.bands import (
    band_centres,
    convert,
    crossover_frequencies,
    octave_of,
    resolution_of,
    to_bands,
    to_octaves,
)
from acoustic_spaces.parameters import band_filter_gains, from_impulse_response, octave_filter_gains
from acoustic_spaces.scene import Coverage, Material
from acoustic_spaces.spatial.directions import fibonacci_sphere

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
FS = 48_000


def test_third_octaves_sit_three_to_an_octave() -> None:
    assert len(THIRD_OCTAVE_BANDS_HZ) == 3 * len(OCTAVE_BANDS_HZ)
    assert THIRD_OCTAVE_BANDS_HZ[1::3] == OCTAVE_BANDS_HZ  # an octave's centre in the middle
    assert band_centres("third_octave") == THIRD_OCTAVE_BANDS_HZ
    np.testing.assert_array_equal(octave_of("third_octave"), np.repeat(np.arange(8), 3))
    edges = crossover_frequencies("third_octave")
    np.testing.assert_allclose(np.diff(np.log2(edges)), 1 / 3)
    centres = np.array(THIRD_OCTAVE_BANDS_HZ, dtype=float)
    assert np.all((edges > centres[:-1]) & (edges < centres[1:]))
    assert resolution_of(8) == "octave" and resolution_of(24) == "third_octave"
    with pytest.raises(ValueError, match="third-octave"):
        resolution_of(10)
    with pytest.raises(ValueError, match="band resolution"):
        band_centres("sixth_octave")


def test_values_convert_between_octaves_and_third_octaves() -> None:
    octave = [0.1 * k for k in range(1, 9)]
    third = convert(octave, "third_octave")
    assert third == pytest.approx(np.repeat(octave, 3))
    assert convert(third, "octave") == pytest.approx(octave)
    assert convert(octave, "octave") == pytest.approx(octave)
    # Energies are spectral levels: an octave's is the mean of its three thirds'.
    energy = np.arange(48.0).reshape(24, 2)
    np.testing.assert_allclose(to_octaves(energy), energy.reshape(8, 3, 2).mean(axis=1))
    np.testing.assert_allclose(to_octaves(energy.T, axis=1), to_octaves(energy).T)
    np.testing.assert_array_equal(to_octaves(energy[:8]), energy[:8])


def test_data_maps_onto_third_octave_bands() -> None:
    freqs = [100, 125, 160, 200, 250, 315]
    values, extrapolated = to_bands(freqs, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], "third_octave")
    assert values[THIRD_OCTAVE_BANDS_HZ.index(160)] == pytest.approx(0.3)
    assert values[0] == pytest.approx(0.1)  # 50 Hz copied from 100 Hz
    assert values[-1] == pytest.approx(0.6)
    assert extrapolated == [50, 63, 80, *THIRD_OCTAVE_BANDS_HZ[9:]]


def test_materials_hold_octave_or_third_octave_values(tmp_path: Path) -> None:
    third = [0.01 * k for k in range(24)]
    material = Material(absorption=third, scattering=0.2)
    assert material.absorption == third
    absorption, scattering = material.coefficients("octave")
    assert absorption == pytest.approx(np.reshape(third, (8, 3)).mean(axis=1))
    assert scattering == pytest.approx([0.2] * 8)
    assert material.coefficients("third_octave")[1] == pytest.approx([0.2] * 24)
    with pytest.raises(ValidationError, match="or 24 .third-octave bands., got 10"):
        Material(absorption=[0.1] * 10)
    # Data at octave centres stays in octave bands; more detail stays in third octaves.
    path = tmp_path / "octaves.csv"
    path.write_text("125,250,500,1000\n0.1,0.2,0.3,0.4\n", encoding="utf-8")
    assert len(Material(file=path).absorption) == 8
    path.write_text("100,125,160,1000\n0.1,0.2,0.3,0.4\n", encoding="utf-8")
    material = Material(file=path)
    assert len(material.absorption) == 24 and "copied from the nearest band" in material.source


def test_directivity_in_third_octave_bands() -> None:
    pattern = FirstOrder((1.0, 1.0, 0.7, 0.5, 0.5, 0.37, 0.25, 0.25))
    third = pattern.in_bands("third_octave")
    assert third.a == tuple(np.repeat(pattern.a, 3))
    assert third.in_bands("octave").a == pytest.approx(pattern.a)
    assert pattern.in_bands("octave") is pattern
    directions = fibonacci_sphere(200)
    np.testing.assert_allclose(third.energy(directions)[:, 1::3], pattern.energy(directions))
    measured = np.column_stack([1.0 + 0.5 * k * (1.0 + directions[:, 0]) for k in range(8)])
    balloon = Balloon.from_measurements(directions, measured)
    thirds = balloon.in_bands("third_octave")
    assert thirds.native()[1].shape[-1] == 24 and len(thirds.native()[0]) == 24
    np.testing.assert_allclose(thirds.energy(directions)[:, 1::3], balloon.energy(directions))
    np.testing.assert_allclose(thirds.in_bands("octave").table, balloon.table)


def test_air_absorption_per_third_octave() -> None:
    octave = energy_attenuation_per_m(20.0, 50.0)
    third = energy_attenuation_per_m(20.0, 50.0, bands="third_octave")
    assert third.shape == (24,)
    np.testing.assert_allclose(third[1::3], octave)  # the same exact mid-band frequencies
    assert np.all(np.diff(third) > 0)


def test_third_octave_filter_banks() -> None:
    n_fft = 1 << 14
    gains = synthesis.band_gains(n_fft, FS, 24)
    np.testing.assert_allclose(gains.sum(axis=0), 1.0, atol=1e-12)
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    for k, centre in enumerate(THIRD_OCTAVE_BANDS_HZ):
        assert gains[k, np.argmin(np.abs(f - centre))] == pytest.approx(1.0)
    x = np.random.default_rng(1).standard_normal(3000)
    bands = synthesis.split_bands(x, FS, 24)
    assert bands.shape == (24, 3000)
    np.testing.assert_allclose(bands.sum(axis=0), x, atol=1e-12)
    np.testing.assert_allclose(synthesis.combine_bands(np.tile(x, (24, 1)), FS), x, atol=1e-12)
    # The analysis filters: 1 at mid-band and -3 dB at the band edges, G^(+-1/6).
    np.testing.assert_allclose(band_filter_gains(n_fft, FS), octave_filter_gains(n_fft, FS))
    assert band_filter_gains(n_fft, FS, 24).shape == (24, n_fft // 2 + 1)
    edge = 10.0 ** (0.3 / 6)
    probe = band_filter_gains(1 << 20, FS, 24)[THIRD_OCTAVE_BANDS_HZ.index(1000)]
    bins = np.fft.rfftfreq(1 << 20, 1.0 / FS)
    assert probe[np.argmin(np.abs(bins - 1000.0))] == pytest.approx(1.0, abs=1e-6)
    for at in (1000.0 / edge, 1000.0 * edge):
        assert probe[np.argmin(np.abs(bins - at))] == pytest.approx(0.5**0.5, abs=1e-3)


def _scene(**simulation) -> Scene:
    scene = Scene.from_yaml(SHOEBOX)
    return scene.model_copy(update={"simulation": scene.simulation.model_copy(update=simulation)})


def _runs(method: str) -> dict:
    runs = {}
    for bands in ("octave", "third_octave"):
        scene = _scene(bands=bands, seed=3, ir_length_s=1.5)
        scene.simulation.late.method = method
        scene.simulation.late.rays = 5000
        runs[bands] = next(iter(simulate(Room.from_scene(scene))))
    return runs


def test_third_octave_responses_fold_into_octaves() -> None:
    runs = _runs("statistical")
    octave, third = runs["octave"], runs["third_octave"]
    assert third.bands == 24 and third.band_hz == THIRD_OCTAVE_BANDS_HZ
    assert third.energy().shape[0] == 24 and third.octave_energy().shape[0] == 8
    assert octave.band_hz == OCTAVE_BANDS_HZ
    np.testing.assert_array_equal(octave.octave_energy(), octave.energy())
    a, b = octave.parameters(), third.parameters()
    # Where the early paths add in pressure (up to 500 Hz), three third-octave filters
    # weigh their interference across an octave a little differently from one.
    low, high = slice(0, octave.coherent_bands), slice(octave.coherent_bands, None)
    assert octave.coherent_bands == 4 and third.coherent_bands == 12
    for name in ("t30", "edt", "ts"):
        np.testing.assert_allclose(getattr(b, name)[high], getattr(a, name)[high], rtol=0.01)
        np.testing.assert_allclose(getattr(b, name)[low], getattr(a, name)[low], rtol=0.05)
    for name, near in (("g", 0.4), ("c80", 0.4), ("d50", 0.03)):
        b_, a_ = getattr(b, name), getattr(a, name)
        np.testing.assert_allclose(b_[high], a_[high], rtol=0.01, atol=0.1)
        np.testing.assert_allclose(b_[low], a_[low], atol=near)
    # Across the 8 kHz octave the air absorbs about twice as much at 10 kHz as at
    # 6.3 kHz: three decays decay together slower than their middle one alone.
    assert b.t30[-1] >= a.t30[-1] * 0.99
    np.testing.assert_allclose(third.speech().sti, octave.speech().sti, atol=0.01)
    # The impulse response synthesized in third octaves measures as the octave one.
    measured = [from_impulse_response(r.impulse_response(FS), FS) for r in (octave, third)]
    np.testing.assert_allclose(measured[1].t30, measured[0].t30, rtol=0.03)


def test_stochastic_third_octave_runs_give_the_octave_parameters() -> None:
    # Within the tracer's noise: 5000 rays per band leave the decay times of single
    # octave bands several per cent apart from one seed to the next.
    runs = _runs("stochastic_ray_tracing")
    a, b = runs["octave"].parameters(), runs["third_octave"].parameters()
    middle = slice(1, 7)  # 125 Hz to 4 kHz
    np.testing.assert_allclose(b.g, a.g, atol=0.5)
    np.testing.assert_allclose(b.c80[middle], a.c80[middle], atol=1.0)
    np.testing.assert_allclose(b.edt[middle], a.edt[middle], rtol=0.15)
    np.testing.assert_allclose(b.t30[4:6], a.t30[4:6], rtol=0.1)  # 1 and 2 kHz


def test_third_octave_absorption_shows_in_its_band() -> None:
    # A panel that absorbs in the 100 Hz third octave alone: in third-octave bands that
    # band decays at its own Eyring rate, between neighbours that do not.
    absorption = [0.05] * 24
    absorption[THIRD_OCTAVE_BANDS_HZ.index(100)] = 0.6
    scene = _scene(bands="third_octave", seed=1, ir_length_s=1.5)
    scene.simulation.late.method = "statistical"
    scene.materials = {
        name: Material(absorption=absorption, scattering=0.5) for name in scene.materials
    }
    room = Room.from_scene(scene)
    eyring = room.reverberation_time("eyring", air_absorption=True, bands="third_octave")
    k = THIRD_OCTAVE_BANDS_HZ.index(100)
    assert eyring[k] < 0.3 * eyring[k - 1] and eyring[k] < 0.3 * eyring[k + 1]
    response = next(iter(simulate(room)))
    t = np.arange(response.late.shape[1]) * response.time_step
    window = (t > response.transition_s + 0.02) & (t < response.transition_s + 0.15)
    for band in (k - 1, k, k + 1):
        slope = np.polyfit(t[window], 10 * np.log10(response.late[band, window]), 1)[0]
        assert -60.0 / slope == pytest.approx(eyring[band], rel=0.03)


def test_wave_solver_in_third_octave_bands() -> None:
    scene = _scene(bands="third_octave", seed=1, ir_length_s=0.5)
    scene.simulation.late.rays = 500
    scene.simulation.wave.enabled = True
    scene.simulation.wave.max_band_hz = 63
    result = simulate(Room.from_scene(scene))
    response = next(iter(result))
    assert response.wave.bands == 3 and response.wave.total_bands == 24
    energy = response.energy()
    assert energy.shape[0] == 24 and np.all(np.isfinite(energy)) and energy[:3].sum() > 0
    ir = response.impulse_response(FS)
    assert np.all(np.isfinite(ir))
    assert np.isfinite(response.parameters().g[0])


def test_diffraction_in_third_octave_bands(tmp_path: Path) -> None:
    obj = tmp_path / "screen.obj"
    obj.write_text("v 0 -300 -300\nv 0 300 -300\nv 0 300 0\nv 0 -300 0\ng screen\nf 1 2 3 4\n")
    paths = {}
    for bands in ("octave", "third_octave"):
        scene = Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": [-4.0, 0.0, -1.0]}],
                "listeners": [{"name": "l", "position": [4.0, 0.0, -1.5]}],
                "simulation": {"bands": bands},
            }
        )
        paths[bands] = early_paths(Room.from_scene(scene), diffraction="direct")
    octave, third = paths["octave"], paths["third_octave"]
    assert octave.diffracted.sum() == third.diffracted.sum() == 1
    a = octave.energy[octave.diffracted][0]
    b = to_octaves(third.energy[third.diffracted], axis=1)[0]
    np.testing.assert_allclose(10 * np.log10(b), 10 * np.log10(a), atol=0.5)


def test_coverage_maps_in_third_octave_bands() -> None:
    values = {}
    for bands in ("octave", "third_octave"):
        scene = _scene(bands=bands, seed=1, ir_length_s=1.0)
        scene.simulation.late.method = "statistical"
        scene.simulation.early.max_order = 1
        values[bands] = coverage_map(Room.from_scene(scene), settings=Coverage(spacing=2.0))
    a, b = (values[k].get().values for k in ("octave", "third_octave"))
    assert b["g"].shape == a["g"].shape and a["g"].shape[1] == 8
    np.testing.assert_allclose(b["g"][:, 4:], a["g"][:, 4:], atol=0.1)
    np.testing.assert_allclose(b["g"][:, :4], a["g"][:, :4], atol=0.4)  # in pressure
    np.testing.assert_allclose(b["sti"], a["sti"], atol=0.01)
    assert math.isfinite(float(np.nanmean(b["t30"])))
