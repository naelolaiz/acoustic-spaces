import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, _native, simulate, synthesis
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.impedance import (
    Admittance,
    band_absorption,
    fit_admittance,
    random_incidence_absorption,
)
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.response import Response
from acoustic_spaces.engine.wave import (
    COURANT,
    LAYER_CELLS,
    LAYER_DAMPING,
    MAX_ABSORPTION,
    WaveResponse,
    _first_order,
    absorption_of_impedance,
    admittance,
    band_grid,
    band_top,
    source_signal,
    wave_gains,
    wave_responses,
)
from acoustic_spaces.geometry import box_mesh
from acoustic_spaces.spatial import Ambisonics, Binaural, Loudspeakers, Mono

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
C = 343.0
FS = 48_000


def _solver(size, admittances, source, probes, cell=0.1, mesh=None, branches=None):
    """A wave solver in a box from (0, 0, 0) to ``size`` (or in ``mesh``, within it), laid
    out as wave_responses does, and its sample rate. ``admittances(mesh)`` gives each
    face's constant admittance, each face its own material; ``branches``, a list of
    (face, Admittance), adds the branches of those admittances."""
    mesh = box_mesh(size) if mesh is None else mesh
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((np.array(size) - origin) / cell).astype(int) + 2
    rate = C / (COURANT * cell)
    beta = np.array(admittances(mesh), dtype=float)
    rows, owners = [np.zeros((0, 3))], [np.zeros(0, np.int64)]
    for face, fit in branches or []:
        beta[face] += fit.beta
        rows.append(fit.branches(rate))
        owners.append(np.full(len(fit.conductance), face, np.int64))
    solver = _native.WaveSolver(
        bvh,
        np.arange(len(mesh.faces), dtype=np.int64),
        beta,
        np.ascontiguousarray(np.vstack(rows)),
        np.concatenate(owners),
        origin,
        cell,
        *map(int, shape),
        np.asarray(source, dtype=float),
        np.asarray(probes, dtype=float).reshape(-1, 3),
        threads=1,
    )
    return solver, rate


def _drive(rate: float, cell: float, steps: int) -> np.ndarray:
    signal, _ = source_signal(rate, cell)
    out = np.zeros(steps, np.float32)
    out[: len(signal)] = signal[:steps]
    return out


def _free_room(distances, max_band_hz=63, box=(16.0, 12.0, 12.0)) -> Room:
    """A large, very absorbing room, with listeners along +x from the source."""
    source = [box[0] / 2 - 2.03, box[1] / 2 + 0.07, box[2] / 2 + 0.02]
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": list(box)},
                "materials": {"a": {"absorption": [0.95] * 8, "scattering": 0.0}},
                "default_material": "a",
                "sources": [{"name": "s", "position": source}],
                "listeners": [
                    {"name": f"r{d}", "position": [source[0] + d, source[1], source[2]]}
                    for d in distances
                ],
                "simulation": {
                    "ir_length_s": 0.2,
                    "seed": 1,
                    "late": {"rays": 500},
                    "early": {"max_order": 0},
                    "wave": {"enabled": True, "max_band_hz": max_band_hz},
                },
            }
        )
    )


def test_admittance_inverts_paris_formula() -> None:
    zeta = np.geomspace(0.5, 1e4, 400)
    alpha = absorption_of_impedance(zeta)
    assert alpha.max() == pytest.approx(MAX_ABSORPTION, abs=1e-3)
    wanted = np.array([0.01, 0.05, 0.2, 0.5, 0.8, 0.95])
    np.testing.assert_allclose(absorption_of_impedance(1.0 / admittance(wanted)), wanted, rtol=1e-6)
    # Hard surfaces stay hard; anything beyond the reachable maximum gets the best match.
    assert admittance(np.array([0.0]))[0] == 0.0
    best = admittance(np.array([0.99, 1.0]))
    np.testing.assert_allclose(absorption_of_impedance(1.0 / best), MAX_ABSORPTION, atol=1e-3)


@pytest.mark.parametrize("band", [0, 1, 2, 3])
def test_band_grid_resolves_the_band_and_divides_the_output_rate(band: int) -> None:
    grid = band_grid(band, FS, C, points_per_wavelength=8.0)
    assert FS / grid.rate == pytest.approx(round(FS / grid.rate))
    assert C / grid.rate == pytest.approx(COURANT * grid.cell)  # the scheme's time step
    assert C / (band_top(band) * grid.cell) >= 8.0
    # About one octave apart, and no finer than needed.
    assert C / (band_top(band) * grid.cell) < 8.0 * 1.2


def test_band_grid_runs_faster_than_a_low_output_rate() -> None:
    # 8 kHz output and the 500 Hz band: the grid needs more than 8000 steps a second, and
    # takes twice that rather than coarser cells.
    grid = band_grid(3, 8000, C, points_per_wavelength=8.0)
    assert grid.rate == pytest.approx(16000)
    assert C / grid.rate == pytest.approx(COURANT * grid.cell)
    assert C / (band_top(3) * grid.cell) >= 8.0


def test_source_signal_adds_no_air() -> None:
    rate = band_grid(0, FS, C).rate
    signal, delay = source_signal(rate, 0.4)
    assert abs(signal.sum()) < 1e-9 * np.abs(signal).max()
    # A closed room's total pressure has the source's term as its second difference: it
    # comes back to rest once the source's first moment is zero too.
    moment = np.arange(len(signal)) * signal
    assert abs(moment.sum()) < 1e-9 * np.abs(moment).max()
    assert int(np.argmax(signal)) == delay
    spectrum = np.abs(np.fft.rfft(signal, 1 << 16))
    f = np.fft.rfftfreq(1 << 16, 1.0 / rate)
    flat = spectrum[(f > 40) & (f < 180)]
    assert flat.max() / flat.min() < 1.05


def test_wave_gains_start_below_the_lowest_band() -> None:
    n_fft = 1 << 18
    gains = wave_gains(n_fft, FS, 2)
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    total = gains.sum(axis=0)
    assert total[f < 18.5].max() < 1e-12
    np.testing.assert_allclose(total[(f > 27) & (f < 140)], 1.0, atol=1e-9)
    assert total[f > 300].max() < 1e-12
    # The source is flat where the wave solver's part starts.
    rate = band_grid(1, FS, C).rate
    signal, _ = source_signal(rate, 0.2)
    spectrum = np.abs(np.fft.rfft(signal, n_fft))
    g = np.fft.rfftfreq(n_fft, 1.0 / rate)
    flat = spectrum[(g > 18.5) & (g < 180)]
    assert 20 * np.log10(flat.max() / flat.min()) < 0.6


def test_rigid_box_rings_at_its_modes() -> None:
    size = (2.0, 1.5, 1.0)
    solver, rate = _solver(
        size, lambda m: np.zeros(len(m.faces)), (0.23, 0.17, 0.13), (1.77, 1.31, 0.87)
    )
    assert solver.air_cells == 20 * 15 * 10
    assert not solver.open
    steps = int(4 * rate)
    pressure = solver.run(_drive(rate, 0.1, steps))[0]
    n_fft = 1 << 18
    spectrum = np.abs(np.fft.rfft(pressure, n_fft))
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    for mode in ((1, 0, 0), (0, 1, 0), (1, 1, 0)):
        expected = C / 2 * math.hypot(*(m / s for m, s in zip(mode, size, strict=True)))
        near = (f > expected - 4) & (f < expected + 4)
        peak = f[near][np.argmax(spectrum[near])]
        assert peak == pytest.approx(expected, abs=0.6)


@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 1.0])
def test_duct_end_reflects_as_its_admittance_says(beta: float) -> None:
    # A plane wave down a duct with an absorbing near end, at the far end's admittance:
    # the reflection is (1 - beta) / (1 + beta) at normal incidence.
    size = (80.0, 0.6, 0.6)

    def admittances(mesh):
        x = mesh.vertices[mesh.faces][:, :, 0]
        far = np.all(np.isclose(x, size[0]), axis=1)
        near = np.all(np.isclose(x, 0.0), axis=1)
        return np.where(far, beta, np.where(near, 1.0, 0.0))

    solver, rate = _solver(size, admittances, (10.0, 0.3, 0.3), (30.0, 0.3, 0.3))
    steps = int(0.45 * rate)
    signal, delay = source_signal(rate, 0.1)
    pressure = solver.run(_drive(rate, 0.1, steps))[0]
    # 40 to 200 Hz, then the incident and reflected waves gated with Hann windows.
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
    pressure = np.fft.irfft(np.fft.rfft(pressure, n_fft) * gain, n_fft)[:steps]
    t = (np.arange(steps) - delay) / rate

    def gated(path: float) -> float:
        x = t - path / C
        window = np.where(np.abs(x) < 0.04, 0.5 + 0.5 * np.cos(np.pi * x / 0.04), 0.0)
        return float(np.sum((pressure * window) ** 2))

    reflection = math.sqrt(gated(120.0) / gated(20.0))
    assert reflection == pytest.approx((1 - beta) / (1 + beta), abs=0.005 if beta < 1 else 0.05)


def test_air_meets_the_inner_face_of_a_thin_wall() -> None:
    # The duct above the other way round, its near end a wall thinner than a cell:
    # absorbing inside (x = 0.03), rigid outside (x = 0). The air meets the inside.
    size, inner, beta = (80.0, 0.6, 0.6), 0.03, 0.5
    box = box_mesh(size)
    lid = [
        [inner, 0.0, 0.0],
        [inner, size[1], 0.0],
        [inner, size[1], size[2]],
        [inner, 0.0, size[2]],
    ]
    n = len(box.vertices)
    mesh = dataclasses.replace(
        box,
        vertices=np.vstack([box.vertices, lid]),
        faces=np.vstack([box.faces, [[n, n + 1, n + 2], [n, n + 2, n + 3]]]),
        face_labels=np.append(box.face_labels, [0, 0]),
    )

    def admittances(mesh):
        x = mesh.vertices[mesh.faces][:, :, 0]
        far = np.all(np.isclose(x, size[0]), axis=1)
        return np.where(far, 1.0, np.where(np.all(np.isclose(x, inner), axis=1), beta, 0.0))

    solver, rate = _solver(size, admittances, (70.0, 0.3, 0.3), (50.0, 0.3, 0.3), mesh=mesh)
    steps = int(0.45 * rate)
    _, delay = source_signal(rate, 0.1)
    pressure = solver.run(_drive(rate, 0.1, steps))[0]
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gain = np.clip((f - 40) / 20, 0, 1) * np.clip((200 - f) / 50, 0, 1)
    pressure = np.fft.irfft(np.fft.rfft(pressure, n_fft) * gain, n_fft)[:steps]
    t = (np.arange(steps) - delay) / rate

    def gated(path: float) -> float:
        x = t - path / C
        window = np.where(np.abs(x) < 0.04, 0.5 + 0.5 * np.cos(np.pi * x / 0.04), 0.0)
        return float(np.sum((pressure * window) ** 2))

    reflection = math.sqrt(gated(120.0) / gated(20.0))
    assert reflection == pytest.approx((1 - beta) / (1 + beta), abs=0.005)


def test_free_field_level_matches_the_rays() -> None:
    room = _free_room((1.0, 2.0, 4.0))
    result = simulate(room)
    assert result.details["s"]["wave_cells"] > 0
    for (_, name), response in result.responses.items():
        assert response.wave is not None and response.wave.bands == 1
        d = float(name[1:])
        energy = response.energy()
        t = np.arange(energy.shape[1]) * response.time_step
        direct = energy[:, t < (d + 6.5) / C].sum(axis=1) * 4 * math.pi * d * d
        assert 10 * math.log10(direct[0]) == pytest.approx(0.0, abs=1.0)
        assert 10 * math.log10(direct[1]) == pytest.approx(0.0, abs=0.1)  # rays


def test_direct_sound_arrives_on_time() -> None:
    # 4 m from the source, 25 to 100 Hz, against the source's own pulse: the wave bands
    # arrive r / c after emission, as the rays do, within a fraction of a step (0.67 ms).
    room = _free_room((4.0,))
    c = room.speed_of_sound()
    source = np.array(room.scene.sources[0].position)
    responses, _ = wave_responses(
        TraceModel.from_room(room),
        source,
        [source + [4.0, 0.0, 0.0]],
        ["r"],
        max_band_hz=63,
        length_s=0.1,
        sample_rate=FS,
        speed_of_sound=c,
    )
    wave = responses[0]
    signal, delay = source_signal(wave.rate, wave.cell)
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / wave.rate)
    band = np.clip((f - 25) / 10, 0, 1) * np.clip((100 - f) / 20, 0, 1)
    spectrum = np.fft.rfft(wave.signals[:, 0], n_fft) * np.conj(np.fft.rfft(signal, n_fft))
    correlation = np.fft.irfft(spectrum * band**2, n_fft)
    lag = int(np.argmax(correlation[: n_fft // 2]))
    y0, y1, y2 = correlation[lag - 1 : lag + 2]
    arrival = delay + lag + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    assert arrival == pytest.approx(4.0 / c * wave.rate, abs=0.25)


def test_velocity_is_in_step_with_the_pressure() -> None:
    # A plane wave from -x crosses the probe and its neighbours one cell away (1 / COURANT
    # steps): X = -W sample by sample, not half a step ahead of it.
    t = np.arange(400.0)
    shift = 1.0 / COURANT
    probes = np.zeros((7, len(t)))
    for k, delay in ((0, 0.0), (1, shift), (2, -shift)):
        probes[k] = np.exp(-(((t - 200.0 - delay) / 16.0) ** 2))
    field = _first_order(probes, [True] * 7, "l")
    np.testing.assert_allclose(field[:, 1], -field[:, 0], atol=0.01)
    assert not np.any(field[:, 2:])


def test_first_order_field_points_to_the_source() -> None:
    room = _free_room((3.0,), max_band_hz=125)
    response = next(iter(simulate(room).responses.values()))
    # From 60 Hz up, well away from the source (kr > 3), where the velocity is in phase
    # with the pressure; and only the direct sound.
    field = response.wave.field(FS, int(0.2 * FS))
    f = np.fft.rfftfreq(len(field), 1.0 / FS)
    gain = np.clip((f - 50) / 10, 0, 1)[:, None]
    field = np.fft.irfft(np.fft.rfft(field, axis=0) * gain, len(field), axis=0)
    n = int(0.03 * FS)
    w, x, y, z = field[:n].T
    assert np.dot(w, x) / math.sqrt(np.dot(w, w) * np.dot(x, x)) < -0.95  # from -x
    assert np.dot(y, y) + np.dot(z, z) < 0.01 * np.dot(x, x)
    # So the first-order Ambisonics of a listener facing +x has it behind.
    ambisonics = response.spatial_impulse_response(Ambisonics(1), FS)
    low = response.wave.field(FS, len(ambisonics))
    assert np.dot(ambisonics[:n, 3], low[:n, 0]) < 0


def test_hybrid_tail_is_flat_across_the_crossover() -> None:
    # The rays' late noise and a wave band tail of the same, flat level: they do not
    # interfere, so they add in energy, and the sum stays flat in 1/6 octaves round 177
    # Hz, where one hands over to the other (it dipped by 3 dB). The tolerance is what
    # eight renderings of the noise leave (0.7 dB; 0.2 dB with 32).
    length, step, energy = 2.0, 1e-3, 1e-4
    n, slots = int(length * FS), int(length / step)
    spectrum = np.exp(2j * np.pi * np.random.default_rng(0).random(n // 2 + 1))
    spectrum[0] = spectrum[-1] = 1.0
    signals = np.zeros((n, 4))
    signals[:, 0] = np.fft.irfft(spectrum, n)
    signals *= math.sqrt(synthesis.ENERGY_TO_IR * energy * slots / np.sum(signals**2))
    response = Response(
        source="s",
        listener="l",
        early=EarlyReflections.empty(8),
        late=np.full((8, slots), energy),
        time_step=step,
        length_s=length,
        direct_delay_s=0.0,
        transition_s=0.0,
        volume_m3=200.0,
        speed_of_sound=C,
        wave=WaveResponse(FS, signals, 2, 0.2),
    )
    f = np.fft.rfftfreq(n, 1.0 / FS)
    centres = 177.0 * 2.0 ** (np.arange(-6, 7) / 6)

    def levels(render, seeds: int) -> np.ndarray:
        power = np.zeros(len(centres))
        for seed in range(seeds):
            s = np.abs(np.fft.rfft(render(seed))) ** 2
            power += [s[(f > c * 2 ** (-1 / 12)) & (f < c * 2 ** (1 / 12))].mean() for c in centres]
        return 10 * np.log10(power)

    mono = levels(lambda seed: response.impulse_response(FS, seed=seed), 8)
    assert np.ptp(mono) < 1.0
    w = levels(lambda seed: response.spatial_impulse_response(Ambisonics(1), FS, seed)[:, 0], 4)
    assert np.ptp(w) < 1.0


def test_simulate_with_the_wave_solver() -> None:
    room = Room.from_yaml(SHOEBOX)
    settings = room.scene.simulation.model_copy(deep=True)
    # Long enough for the 250 Hz decay to pass 20 dB, which every parameter needs: this
    # room's EDT there is up to 1.7 s with 2000 rays.
    settings.ir_length_s = 0.8
    settings.late.rays = 2000
    settings.seed = 1
    settings.wave.enabled = True
    settings.wave.max_band_hz = 125
    result = simulate(room, settings=settings)
    info = next(iter(result.details.values()))
    assert info["wave_cells"] > 0 and info["wave_cell_m"] == pytest.approx(0.2, rel=0.05)
    assert "wave_open" not in info and info["wave_s"] > 0
    response = next(iter(result.responses.values()))
    assert response.wave.bands == 2
    parameters = response.parameters()
    for values in (parameters.edt, parameters.c80, parameters.d50, parameters.g):
        assert np.all(np.isfinite(values))
    n = int(math.ceil(settings.ir_length_s * FS))
    ir = response.impulse_response(FS, seed=1)
    assert ir.shape == (n,) and np.all(np.isfinite(ir))
    # Only the low bands change: the rays' high bands stay as they were.
    rays = dataclasses.replace(response, wave=None).impulse_response(FS, seed=1)
    n_fft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfftfreq(n_fft, 1.0 / FS)
    with_wave, without = np.fft.rfft(ir, n_fft), np.fft.rfft(rays, n_fft)
    high = f > 400
    assert np.abs(with_wave[high] - without[high]).max() < 0.02 * np.abs(without[high]).max()
    low = (f > 50) & (f < 150)
    assert np.abs(with_wave[low] - without[low]).max() > 0.1 * np.abs(without[low]).max()
    for fmt in (Mono(), Ambisonics(2), Binaural(), Loudspeakers.layout("5.1")):
        out = response.spatial_impulse_response(fmt, FS, seed=1)
        assert out.shape == (n, fmt.channels) and np.all(np.isfinite(out))


def test_wave_responses_refuse_listeners_outside_and_huge_grids() -> None:
    room = Room.from_yaml(SHOEBOX)
    model = TraceModel.from_room(room)
    source = np.array(room.scene.sources[0].position)
    common = {"length_s": 0.1, "sample_rate": FS, "speed_of_sound": C}
    with pytest.raises(ValueError, match="'out' is not in the air"):
        wave_responses(model, source, [[50.0, 50.0, 50.0]], ["out"], max_band_hz=63, **common)
    with pytest.raises(ValueError, match="cells"):
        wave_responses(
            model, source, [source], ["l"], max_band_hz=500, points_per_wavelength=200, **common
        )
    with pytest.raises(ValueError, match="max_band_hz"):
        wave_responses(model, source, [source], ["l"], max_band_hz=100, **common)


def test_wave_responses_refuse_before_running(tmp_path: Path) -> None:
    said: list[str] = []
    common = {"max_band_hz": 63, "length_s": 0.1, "sample_rate": FS, "speed_of_sound": C}
    common["say"] = said.append
    room = Room.from_yaml(SHOEBOX)
    source = np.array(room.scene.sources[0].position)
    with pytest.raises(ValueError, match="'out' is not in the air"):
        wave_responses(TraceModel.from_room(room), source, [[50.0, 50.0, 50.0]], ["out"], **common)
    assert said == []  # not a step taken
    # A source shut in a box inside the room leaves every listener out of its air.
    vertices, faces = [], []
    for low, size in (((0, 0, 0), (4.0, 3.0, 2.5)), ((1, 1, 1), (0.6, 0.6, 0.6))):
        mesh = box_mesh(size)
        faces += [f"f {' '.join(str(len(vertices) + 1 + i) for i in f)}" for f in mesh.faces]
        vertices += [f"v {x} {y} {z}" for x, y, z in mesh.vertices + low]
    obj = tmp_path / "boxed.obj"
    obj.write_text("\n".join([*vertices, "g walls", *faces]))
    boxed = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"m": {"absorption": [0.2] * 8, "scattering": 0.1}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [1.3, 1.3, 1.3]}],
                "listeners": [{"name": "l", "position": [3.0, 2.0, 1.2]}],
            }
        )
    )
    with pytest.raises(ValueError, match="source 's' is closed in"):
        wave_responses(
            TraceModel.from_room(boxed),
            [1.3, 1.3, 1.3],
            [[3.0, 2.0, 1.2]],
            ["l"],
            label="s",
            **common,
        )


def test_open_models_absorb_at_the_grid_edge(tmp_path: Path) -> None:
    obj = tmp_path / "floor.obj"
    obj.write_text("v 0 0 0\nv 8 0 0\nv 8 6 3\nv 0 6 3\ng floor\nf 1 2 3 4\n")
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"file": str(obj)},
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": [2.0, 2.0, 2.5]}],
                "listeners": [{"name": "l", "position": [6.0, 3.0, 3.0]}],
            }
        )
    )
    model = TraceModel.from_room(room)
    responses, info = wave_responses(
        model,
        np.array([2.0, 2.0, 2.5]),
        [[6.0, 3.0, 3.0]],
        ["l"],
        max_band_hz=63,
        length_s=0.3,
        sample_rate=FS,
        speed_of_sound=C,
    )
    assert info["wave_open"] == 1.0
    pressure = responses[0].signals[:, 0]
    rate = responses[0].rate
    # The sound leaves: little is left after 150 ms.
    late = pressure[int(0.15 * rate) :]
    assert np.sum(late**2) < 1e-2 * np.sum(pressure**2)


def test_bands_with_their_own_absorption_share_one_run() -> None:
    def room(low: float) -> Room:
        return Room.from_scene(
            Scene.model_validate(
                {
                    "geometry": {"box": [3.0, 2.5, 2.0]},
                    "materials": {"m": {"absorption": [low] + [0.3] * 7, "scattering": 0.1}},
                    "default_material": "m",
                    "sources": [{"name": "s", "position": [0.7, 0.6, 1.1]}],
                    "listeners": [{"name": "l", "position": [2.3, 1.8, 1.3]}],
                    "simulation": {
                        "ir_length_s": 1.0,
                        "seed": 1,
                        "late": {"rays": 500},
                        "wave": {"enabled": True, "max_band_hz": 125},
                    },
                }
            )
        )

    for low in (0.3, 0.1):
        result = simulate(room(low))
        assert result.details["s"]["wave_absorption_miss"] < 0.01
        wave = result.responses[("s", "l")].wave
        assert wave.bands == 2 and wave.signals.shape[1] == 4
        # The less absorbing 63 Hz band decays more slowly: more of its sound is left
        # after 300 ms for what it has between 100 and 300 ms. (Its share of all the sound
        # is no measure: with the same absorption it has twice the other band's after
        # 100 ms.)
        energy = wave.energy(0.01, 100)
        late = energy[:, 30:].sum(axis=1) / energy[:, 10:30].sum(axis=1)
        assert (late[0] > 1.5 * late[1]) == (low < 0.3)


def test_random_incidence_absorption_of_a_complex_admittance() -> None:
    mu = np.linspace(0.0, 1.0, 20001)
    for y in (0.3 + 0.2j, 0.05 - 0.4j, 1.2 + 0.9j, 0.01 + 0.0j, 0.0 + 0.5j):
        # 1 - |R|^2 at incidence cos(theta) = mu, weighted by sin(2 theta).
        absorbed = 4 * y.real * mu / np.abs(mu + y) ** 2
        expected = np.trapezoid(absorbed * 2 * mu, mu)
        assert random_incidence_absorption(np.array([y]))[0] == pytest.approx(expected, abs=1e-8)
    beta = np.array([0.01, 0.2, 0.638, 3.0])
    np.testing.assert_allclose(random_incidence_absorption(beta), absorption_of_impedance(1 / beta))


@pytest.mark.parametrize(("total", "bands"), [(8, 4), (8, 2), (24, 12)])
def test_fitted_admittances_absorb_as_their_materials(total: int, bands: int) -> None:
    rate = band_grid(bands - 1, FS, C, total=total).rate
    shapes = {
        "porous": [0.05, 0.15, 0.4, 0.7],
        "panel": [0.35, 0.3, 0.15, 0.08],
        "hard": [0.01, 0.01, 0.015, 0.02],
        "seats": [0.5, 0.6, 0.75, 0.85],
    }
    f = np.geomspace(5.0, 0.4 * rate, 2000)
    for name, values in shapes.items():
        target = np.interp(np.linspace(0, 3, bands), np.arange(4), values)
        fit = fit_admittance(target, total, rate)
        # Steep changes between neighbouring octaves (porous, two bands) are met roughly.
        tolerance = 0.1 if (name == "porous" and bands == 2) else 0.01
        np.testing.assert_allclose(fit.achieved, target, atol=tolerance)
        np.testing.assert_allclose(band_absorption(fit, bands, total, rate), fit.achieved)
        # Passive: every branch resistive, the admittance's real part never negative.
        assert fit.beta >= 0 and np.all(fit.conductance > 0) and np.all(fit.quality > 0)
        assert np.all(fit(f).real >= 0) and np.all(fit(f, rate).real >= 0)
        np.testing.assert_allclose(fit.branches(rate)[:, 1], 1 / fit.conductance)
    # A constant needs no branches; nothing absorbs past what a surface can.
    flat = fit_admittance([0.3] * bands, total, rate)
    assert len(flat.conductance) == 0
    assert flat.beta == pytest.approx(admittance(np.array([0.3]))[0])
    assert fit_admittance([0.0] * bands, total, rate).beta == 0.0
    np.testing.assert_allclose(fit_admittance([0.99] * bands, total, rate).achieved, MAX_ABSORPTION)


def _discrete_reflection(f: np.ndarray, rate: float, y: np.ndarray) -> np.ndarray:
    """Reflection at normal incidence off a wall of admittance y (the scheme's, at that
    rate) in the grid: a plane wave along an axis, the wall half a cell past the last
    centre. With time going as e^(-i w t), so y is the conjugate of Y(i w)."""
    phi = 2 * np.pi * f / rate
    shift = np.exp(2j * np.arcsin(np.minimum(np.sin(phi / 2) / COURANT, 1.0)))  # one cell
    c = 1j * np.sin(phi) * y / COURANT
    return (c - shift + 1) / (1 / shift - 1 - c)


def test_duct_end_reflects_as_its_frequency_dependent_admittance_says() -> None:
    # A plane wave down a duct to an end whose admittance resonates at 100 Hz: the
    # reflected wave over the incident one (a run in a longer duct gives that), against
    # a rigid end's, which the grid reflects whole, at every frequency.
    resonant = Admittance(0.05, np.array([0.6]), np.array([2.0]), np.array([100.0]))

    def run(length: float, end: Admittance | None) -> tuple[np.ndarray, float]:
        size = (length, 0.3, 0.3)

        def admittances(mesh):
            x = mesh.vertices[mesh.faces][:, :, 0]
            return np.where(np.all(np.isclose(x, 0.0), axis=1), 1.0, 0.0)

        far = []
        if end is not None:
            mesh = box_mesh(size)
            x = mesh.vertices[mesh.faces][:, :, 0]
            far = [(f, end) for f in np.flatnonzero(np.all(np.isclose(x, length), axis=1))]
        solver, rate = _solver(
            size, admittances, (60.0, 0.15, 0.15), (110.0, 0.15, 0.15), branches=far
        )
        return solver.run(_drive(rate, 0.1, int(0.6 * rate)))[0], rate

    reference, rate = run(200.0, None)
    incident = np.where(np.arange(len(reference)) / rate < 0.48, reference, 0.0)
    # (before the sound off the near end comes by)
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    band = (f > 30) & (f < 250)
    f = f[band]

    def reflection(end: Admittance) -> np.ndarray:
        reflected = run(130.0, end)[0] - reference
        return np.fft.rfft(reflected, n_fft)[band] / np.fft.rfft(incident, n_fft)[band]

    measured = np.abs(reflection(resonant) / reflection(Admittance(0.0)))
    expected = np.abs(_discrete_reflection(f, rate, np.conj(resonant(f, rate))))
    np.testing.assert_allclose(measured, expected, atol=0.012)
    # It absorbs most at resonance, as it should: little comes back at 100 Hz.
    assert measured[np.argmin(np.abs(f - 100))] < 0.25 < measured[np.argmin(np.abs(f - 40))]


def _plate(h: float, pad: int, layer: int, seconds: float, damping: float = LAYER_DAMPING):
    """The pressure at three points round a rigid 8 x 6 m plate in the open, with
    ``pad`` more cells round the grid than wave_responses gives it and ``layer`` cells
    of matched layer."""
    vertices = np.array([[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0]], float)
    bvh = _native.Bvh(vertices, np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    source = np.array([2.0, 2.0, 1.5])
    listeners = np.array([[6.0, 4.0, 1.2], [4.0, 3.0, 3.0], [7.5, 1.0, 0.5]])
    points = np.vstack([source, listeners])
    low = np.minimum(vertices.min(0), points.min(0) - 2 * h)
    high = np.maximum(vertices.max(0), points.max(0) + 2 * h)
    origin = low - h * (2.0 + pad + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((high - origin) / h).astype(int) + 2 + pad
    solver = _native.WaveSolver(
        bvh,
        np.zeros(2, np.int64),
        np.zeros(1),
        np.zeros((0, 3)),
        np.zeros(0, np.int64),
        origin,
        h,
        *map(int, shape),
        source,
        listeners,
        layer=layer,
        layer_damping=damping,
    )
    rate = C / (COURANT * h)
    return solver.run(_drive(rate, h, int(seconds * rate))), solver, rate


def test_matched_layer_takes_in_the_sound_leaving_an_open_model() -> None:
    # Against a grid so large that nothing comes back from its edge in time: the edge
    # alone sends back 7 to 14 dB below the sound, the matched layer 40 dB and more.
    h, seconds = 0.3, 0.07
    reference, _, rate = _plate(h, 45, 0, seconds)
    n_fft = 1 << 12
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    gain = np.clip((f - 25) / 10, 0, 1) * np.clip((150 - f) / 30, 0, 1)

    def band(x):
        return np.fft.irfft(np.fft.rfft(x, n_fft, axis=1) * gain, n_fft, axis=1)[:, : x.shape[1]]

    def error_db(pressure):
        miss = band(pressure) - band(reference)
        return 10 * np.log10(np.sum(miss**2, axis=1) / np.sum(band(reference) ** 2, axis=1))

    edge, solver, _ = _plate(h, 0, 0, seconds)
    assert solver.open and solver.layer_cells == 0
    assert np.all(error_db(edge) > -20)
    layered, solver, _ = _plate(h, LAYER_CELLS, LAYER_CELLS, seconds)
    assert solver.layer_cells > 0
    assert np.all(error_db(layered) < -40)


def test_matched_layer_stays_stable() -> None:
    pressure, _, rate = _plate(0.5, LAYER_CELLS, LAYER_CELLS, 8.0)
    tenth = pressure.shape[1] // 10
    levels = [np.mean(pressure[:, k * tenth : (k + 1) * tenth] ** 2) for k in range(10)]
    # Down to the rounding of single precision, and no growth over 9,500 steps.
    assert max(levels[6:]) < 1e-11 * levels[0]


def test_wave_solver_checks_its_boundaries() -> None:
    mesh = box_mesh((2.0, 2.0, 2.0))
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    faces = len(mesh.faces)
    common = (np.array([-0.2, -0.2, -0.2]), 0.1, 26, 26, 26, np.array([1.0, 1.0, 1.0]))
    probes = np.array([[1.2, 1.1, 0.9]])

    def make(face_material=None, beta=(0.0,), branches=(), owners=(), **options):
        material = np.zeros(faces, np.int64) if face_material is None else face_material
        return _native.WaveSolver(
            bvh,
            material,
            np.array(beta, float),
            np.array(branches, float).reshape(-1, 3),
            np.array(owners, np.int64),
            *common,
            probes,
            **options,
        )

    assert make(branches=[[1.0, 2.0, 0.5]], owners=[0]).branch_states > 0
    with pytest.raises(ValueError, match="no material"):
        make(face_material=np.full(faces, 1, np.int64))
    with pytest.raises(ValueError, match="negative admittance"):
        make(beta=(-0.1,))
    with pytest.raises(ValueError, match="stiffness >= 0"):
        make(branches=[[1.0, -2.0, 0.5]], owners=[0])
    with pytest.raises(ValueError, match="branch has no material"):
        make(branches=[[1.0, 2.0, 0.5]], owners=[3])
    # A closed room's grid has room for a layer, which the air never reaches...
    assert make(layer=1).layer_cells == 0
    # ...but a model through the layer is refused.
    with pytest.raises(ValueError, match="reaches into the matched layer"):
        _native.WaveSolver(
            bvh,
            np.zeros(faces, np.int64),
            np.zeros(1),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            np.array([0.05, 0.05, 0.05]),
            0.1,
            24,
            24,
            30,
            np.array([1.0, 1.0, 1.0]),
            probes,
            layer=3,
        )
