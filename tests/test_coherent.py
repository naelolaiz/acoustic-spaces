"""Early paths that add in pressure in the lowest bands (engine.coherent)."""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early as early_stage
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coherent import (
    band_count,
    reflection_phases,
    sample_frequencies,
    total_energy,
)
from acoustic_spaces.engine.diffraction import (
    EdgeDiffraction,
    band_frequencies,
    wedge_prefactor,
    wedge_terms,
)
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.simulate import coherent_bands
from acoustic_spaces.parameters import from_impulse_response
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


def _floor(tmp_path: Path, absorption: float) -> Room:
    """A floor 100 m wide in free field, a source and a listener 0.3 m above it."""
    obj = tmp_path / "floor.obj"
    obj.write_text("v -50 -50 0\nv 50 -50 0\nv 50 50 0\nv -50 50 0\ng floor\nf 1 2 3 4\n")
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"floor": {"absorption": [absorption] * 8, "scattering": 0.0}},
                "default_material": "floor",
                "sources": [{"name": "s", "position": [0.0, 0.0, 0.3]}],
                "listeners": [{"name": "l", "position": [10.0, 0.0, 0.3]}],
                "simulation": {"early": {"max_order": 1}, "late": {"rays": 100}},
            }
        )
    )


def test_settings_pick_the_bands() -> None:
    assert EarlyStage().coherent_max_band_hz == 500
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


def test_reflection_phases_of_hard_and_absorbing_surfaces(tmp_path: Path) -> None:
    # At 3 degrees above the surface; at exactly grazing incidence every surface that
    # absorbs at all reflects half a turn out of phase.
    for absorption, grazing in ((0.01, 0.0), (0.8, math.pi)):
        model = TraceModel.from_room(_floor(tmp_path, absorption))
        faces = np.zeros(3, dtype=np.int64)
        phase = np.angle(reflection_phases(model, faces, np.array([1.0, 0.5, 0.05]), 2))
        np.testing.assert_allclose(phase[:2], 0.0, atol=1e-6)  # constant: a real admittance
        np.testing.assert_allclose(np.abs(phase[2]), grazing, atol=1e-6)


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
    assert np.all(level[:4] > low) if low > 0 else np.all(level[:4] < low)
    np.testing.assert_allclose(level[4:], in_energy[4:])
    response = simulate(room)["s", "l"]
    early = dataclasses.replace(response, late=np.zeros_like(response.late))
    g = early.parameters().g[:4]
    np.testing.assert_allclose(g, level[:4], atol=0.3)


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
