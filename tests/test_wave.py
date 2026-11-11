import dataclasses
import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import Room, Scene, _native, simulate, synthesis
from acoustic_spaces.engine import wave as wave_stage
from acoustic_spaces.engine.early import EarlyReflections
from acoustic_spaces.engine.impedance import (
    Admittance,
    band_absorption,
    fit_admittance,
    random_incidence_absorption,
)
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.refraction import Profile, Rays
from acoustic_spaces.engine.response import Response
from acoustic_spaces.engine.wave import (
    COURANT,
    ISOTROPIC_COURANT,
    LAYER_CELLS,
    LAYER_DAMPING,
    LAYER_ROOM,
    MAX_ABSORPTION,
    WAVE_BANDS_HZ,
    WaveResponse,
    _first_order,
    _off_the_source,
    _open_sides,
    _probe_points,
    _spans_across,
    absorption_of_impedance,
    admittance,
    band_grid,
    band_top,
    slow_by,
    source_signal,
    wave_gains,
    wave_responses,
)
from acoustic_spaces.geometry import box_mesh
from acoustic_spaces.scene.model import WaveStage
from acoustic_spaces.spatial import Ambisonics, Binaural, Loudspeakers, Mono

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
C = 343.0
FS = 48_000


STENCILS = pytest.mark.parametrize("stencil", [7, 19])


def _solver(
    size,
    admittances,
    source,
    probes,
    cell=0.1,
    mesh=None,
    branches=None,
    stencil=7,
    speed=None,
    flow=None,
):
    """A wave solver in a box from (0, 0, 0) to ``size`` (or in ``mesh``, within it), laid
    out as wave_responses does, and its sample rate. ``admittances(mesh)`` gives each
    face's constant admittance, each face its own material; ``branches``, a list of
    (face, Admittance), adds the branches of those admittances; ``speed(centres)`` gives
    the cells' speed of sound over the grid's, from their centres (n, 3), and
    ``flow(centres)`` the air's motion along x and y (cells per step)."""
    mesh = box_mesh(size) if mesh is None else mesh
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((np.array(size) - origin) / cell).astype(int) + 2
    rate = C / (_courant(stencil) * cell)
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
        stencil=stencil,
        speed=_speeds(speed, origin, cell, shape),
        **_flows(flow, origin, cell, shape),
    )
    return solver, rate


def _centres(origin, cell, shape) -> np.ndarray:
    """The centres (n, 3) of a grid's cells, x fastest."""
    x, y, z = (origin[a] + (np.arange(shape[a]) + 0.5) * cell for a in range(3))
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)])


def _speeds(speed, origin, cell, shape):
    """The ratios ``speed(centres)`` of the cells of a grid, as the solver takes them."""
    if speed is None:
        return None
    return np.ascontiguousarray(speed(_centres(origin, cell, shape)), dtype=np.float32)


def _flows(flow, origin, cell, shape) -> dict:
    """The solver's arguments for the air's motion ``flow(centres)`` (cells per step
    along x and y), none for still air."""
    if flow is None:
        return {}
    fx, fy = flow(_centres(origin, cell, shape))
    return {
        "flow_x": np.ascontiguousarray(fx, dtype=np.float32),
        "flow_y": np.ascontiguousarray(fy, dtype=np.float32),
    }


def _courant(stencil: int) -> float:
    return COURANT if stencil == 7 else ISOTROPIC_COURANT


def _drive(rate: float, cell: float, steps: int, stencil: int = 7) -> np.ndarray:
    """The source's term on each of ``steps`` steps of the ``stencil``-point grid, as
    wave_responses gives it."""
    signal, _ = source_signal(rate, cell, _courant(stencil))
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


@STENCILS
def test_source_is_flat_to_the_top_of_the_highest_band(stencil: int) -> None:
    """On the coarsest grid the schema allows (5 points per wavelength), the source's
    low-pass leaves the highest wave band whole with either stencil: from half the band's
    top up to it, within 0.05 dB (0.023 dB at most). The 19-point grid's time step is 1.5
    times the 7-point one's for the same cell, so the band's top lies 1.5 times higher in
    its sample rate, and the low-pass's cutoff with it; with the 7-point stencil's cutoff
    it fell 0.6 to 1.6 dB by the top. (The 19-point grid's signal as its grid sends it
    out: with the grid's own gain on a point source's sound, which the signal undoes.)"""
    courant = _courant(stencil)
    for total in (8, 24):
        per_octave = total // 8
        for band in range(per_octave - 1, 4 * per_octave, per_octave):  # each max_band_hz
            grid = band_grid(band, FS, C, 5.0, total, courant)
            signal, _ = source_signal(grid.rate, grid.cell, courant)
            n_fft = 1 << 18
            f = np.fft.rfftfreq(n_fft, 1.0 / grid.rate)
            spectrum = np.abs(np.fft.rfft(signal, n_fft))
            if stencil == 19:
                spectrum *= 9.0 / (5.0 + 4.0 * np.cos(2.0 * np.pi * f / grid.rate))
            top = band_top(band, total)
            level = 20 * np.log10(spectrum[(f >= top / 2) & (f <= top)] / spectrum[f >= top / 2][0])
            assert np.abs(level).max() < 0.05, (total, band)


@STENCILS
def test_rigid_box_rings_at_its_modes(stencil: int) -> None:
    size = (2.0, 1.5, 1.0)
    solver, rate = _solver(
        size,
        lambda m: np.zeros(len(m.faces)),
        (0.23, 0.17, 0.13),
        (1.77, 1.31, 0.87),
        stencil=stencil,
    )
    assert solver.air_cells == 20 * 15 * 10
    assert not solver.open
    steps = int(4 * rate)
    pressure = solver.run(_drive(rate, 0.1, steps, stencil))[0]
    n_fft = 1 << 18
    spectrum = np.abs(np.fft.rfft(pressure, n_fft))
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    for mode in ((1, 0, 0), (0, 1, 0), (1, 1, 0)):
        expected = C / 2 * math.hypot(*(m / s for m, s in zip(mode, size, strict=True)))
        near = (f > expected - 4) & (f < expected + 4)
        peak = f[near][np.argmax(spectrum[near])]
        assert peak == pytest.approx(expected, abs=0.6)


def test_isotropic_stencil_disperses_less() -> None:
    # The axial modes of a 6 m tube (rigid, 10 cm cells) at 12, 8 and 5 cells per
    # wavelength: each stencil's frequencies fall short of the tube's as much as
    # slow_by says its waves along an axis are slow, and the 19-point stencil's by less
    # than half as much as the 7-point one's. (The source 0.17 m from the tube's end, off
    # the node that the mode of 5 cells per wavelength has 0.125 m from it.)
    size, h = (6.0, 0.3, 0.2), 0.1
    spacing = C / (2 * size[0])  # between the tube's modes
    error = {}
    for stencil in (7, 19):
        solver, rate = _solver(
            size,
            lambda m: np.zeros(len(m.faces)),
            (0.17, 0.12, 0.07),
            (5.84, 0.17, 0.12),
            stencil=stencil,
        )
        pressure = solver.run(_drive(rate, h, int(3 * rate), stencil))[0]
        n_fft = 1 << 20
        spectrum = np.abs(np.fft.rfft(pressure * np.hanning(len(pressure)), n_fft))
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        for per_wavelength in (12, 8, 5):
            m = round(2 * size[0] / (per_wavelength * h))
            exact = m * spacing
            grid = exact * (1 - slow_by(stencil, per_wavelength))
            near = np.abs(f - grid) < 0.4 * spacing
            peak = f[near][np.argmax(spectrum[near])]
            assert peak == pytest.approx(grid, rel=2e-5)
            error[stencil, per_wavelength] = peak / exact - 1
    assert error[7, 8] < -0.015
    assert abs(error[19, 8]) < 0.5 * abs(error[7, 8])


def test_a_grid_coarser_than_the_default_is_noted(tmp_path: Path) -> None:
    # Below the default 8 cells per wavelength the warnings say how slow the top band's
    # waves travel with the stencil the run takes: in a closed room the isotropic one,
    # round an open model (auto) or asked for, the standard one.
    def notes(room: Room, **wave) -> list[str]:
        room.scene.simulation.wave = WaveStage(enabled=True, **wave)
        return [w for w in room.warnings() if "points_per_wavelength" in w]

    closed = _free_room([1.0])
    assert notes(closed) == [] and notes(closed, points_per_wavelength=7.9)
    closed.scene.simulation.wave = WaveStage(points_per_wavelength=5)  # not enabled
    assert not [w for w in closed.warnings() if "points_per_wavelength" in w]
    (note,) = notes(closed, points_per_wavelength=5)
    assert "is 5 (default 8): with the isotropic (19-point) stencil" in note
    assert "up to 1.85 % slow in still air (0.67 % at 8)" in note
    (note,) = notes(closed, points_per_wavelength=6, scheme="standard")
    assert "with the standard (7-point) stencil" in note
    assert "up to 3.13 % slow in still air (1.74 % at 8)" in note
    obj = tmp_path / "floor.obj"
    obj.write_text("v 0 0 0\nv 8 0 0\nv 8 6 3\nv 0 6 3\ng floor\nf 1 2 3 4\n")
    plate = Room.from_scene(
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
    (note,) = notes(plate, points_per_wavelength=5)
    assert "with an open model's standard (7-point) stencil" in note
    assert "up to 4.56 % slow" in note


def test_isotropic_stencil_stays_stable_in_a_slanted_room() -> None:
    # A rigid room turned off the grid's axes, with a slanted screen and a cube in it,
    # whose staircase walls fold the edge links every which way: over 30,000 steps its
    # pressure neither grows nor dies away (it keeps its energy).
    def turned(mesh, angles, shift):
        (a, b) = angles
        rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
        return (mesh.vertices - 0.5 * mesh.vertices.max(axis=0)) @ (rz @ rx).T + shift

    room, cube = box_mesh((1.6, 1.3, 1.1)), box_mesh((0.35, 0.3, 0.3))
    screen = np.array([[-0.4, -0.3, -0.5], [0.3, 0.2, -0.5], [0.3, 0.2, 0.3], [-0.4, -0.3, 0.3]])
    centre = np.array([1.0, 1.0, 1.0])
    vertices = np.vstack(
        [
            turned(room, (0.35, 0.44), centre),
            turned(cube, (0.6, 0.2), centre + [0.35, -0.25, 0.1]),
            screen + centre + [-0.1, 0.25, 0.0],
        ]
    )
    n, m = len(room.vertices), len(room.vertices) + len(cube.vertices)
    faces = np.vstack([room.faces, cube.faces + n, [[m, m + 1, m + 2], [m, m + 2, m + 3]]])
    mesh = dataclasses.replace(
        room,
        vertices=vertices,
        faces=faces,
        face_labels=np.zeros(len(faces), dtype=room.face_labels.dtype),
    )
    solver, rate = _solver(
        (2.0, 2.0, 2.0),
        lambda m: np.zeros(len(m.faces)),
        centre + [-0.3, -0.35, -0.2],
        [centre + [0.25, 0.3, -0.25], centre + [-0.35, 0.1, 0.3]],
        mesh=mesh,
        stencil=19,
    )
    assert 2000 < solver.air_cells < 1.6 * 1.3 * 1.1 / 0.1**3  # the room's, not the cube's
    pressure = solver.run(_drive(rate, 0.1, 30_000, 19))
    tenth = pressure.shape[1] // 10
    levels = [np.mean(pressure[:, k * tenth : (k + 1) * tenth] ** 2) for k in range(10)]
    assert np.all(np.isfinite(pressure))
    assert max(levels[1:]) < 3 * min(levels[1:])


@STENCILS
def test_the_update_keeps_its_highest_modes_inside_the_stability_limit(stencil: int) -> None:
    # On a mode alternating along some axes and constant along the rest, signs c = +-1,
    # the update is p+ = A p - p-, A = centre + 2 axial sum_a c_a + 4 diagonal sum_a<b
    # c_a c_b: the extremes of its factors. At A = -2 a kick grows linearly (the 7-point
    # checkerboard, the 19-point's modes alternating along two axes): the Laplacian
    # times 1 - 2^-18 keeps them 2^-16 inside, sound 2^-19 slower, and the uniform
    # pressure at A = 2, as a closed room's.
    solver, _ = _solver(
        (0.4, 0.4, 0.4), lambda m: np.zeros(len(m.faces)), [0.2] * 3, [[0.2] * 3], stencil=stencil
    )
    layout = solver.layout()
    axial = float(layout["weights"][0])
    diagonal = float(layout["weights"][6]) if stencil == 19 else 0.0
    centre = float(layout["centre"][0])
    margin = 2.0**-18
    exact = (1 / 3, 0.0) if stencil == 7 else (0.25, 0.125)
    np.testing.assert_allclose([axial, diagonal], np.multiply(exact, 1 - margin), rtol=2**-22)
    c = np.array(list(itertools.product([1.0, -1.0], repeat=3)))
    pairs = c[:, 0] * c[:, 1] + c[:, 0] * c[:, 2] + c[:, 1] * c[:, 2]
    factors = centre + 2 * axial * c.sum(axis=1) + 4 * diagonal * pairs
    assert factors[0] == 2.0
    assert factors.min() >= -2 + 4 * margin
    # In single precision, rounded as update_interior rounds (fdtd.hpp), the checkerboard
    # and the mode (0, pi, pi), kicked, swing within 1 / sin of their angle a step,
    # 1 / (2 sqrt(margin)) = 256 times the kick, rather than growing.
    f32 = np.float32
    kicks = np.random.default_rng(1).uniform(0.5, 2.0, 32).astype(f32)
    previous, current, peak = np.zeros_like(kicks), kicks.copy(), np.zeros_like(kicks)
    a, d, centre32 = f32(axial), f32(diagonal), f32(centre)
    for _ in range(20_000):
        p, q = current, -current
        if stencil == 7:
            following = centre32 * p + a * (q + q + q + q + q + q) - previous
        else:
            edges = q + q + q + q + q + q + q + q + p + p + p + p
            following = centre32 * p + a * (p + p + q + q + q + q) + d * edges - previous
        previous, current = current, following
        np.maximum(peak, np.abs(current), out=peak)
    assert np.max(peak / kicks) < 1.1 * 256


@STENCILS
@pytest.mark.parametrize("beta", [0.0, 0.1, 0.5, 1.0])
def test_duct_end_reflects_as_its_admittance_says(beta: float, stencil: int) -> None:
    # A plane wave down a duct with an absorbing near end, at the far end's admittance:
    # the reflection is (1 - beta) / (1 + beta) at normal incidence.
    size = (80.0, 0.6, 0.6)

    def admittances(mesh):
        x = mesh.vertices[mesh.faces][:, :, 0]
        far = np.all(np.isclose(x, size[0]), axis=1)
        near = np.all(np.isclose(x, 0.0), axis=1)
        return np.where(far, beta, np.where(near, 1.0, 0.0))

    solver, rate = _solver(size, admittances, (10.0, 0.3, 0.3), (30.0, 0.3, 0.3), stencil=stencil)
    steps = int(0.45 * rate)
    signal, delay = source_signal(rate, 0.1)
    pressure = solver.run(_drive(rate, 0.1, steps, stencil))[0]
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


@STENCILS
def test_air_meets_the_inner_face_of_a_thin_wall(stencil: int) -> None:
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

    solver, rate = _solver(
        size, admittances, (70.0, 0.3, 0.3), (50.0, 0.3, 0.3), mesh=mesh, stencil=stencil
    )
    steps = int(0.45 * rate)
    _, delay = source_signal(rate, 0.1)
    pressure = solver.run(_drive(rate, 0.1, steps, stencil))[0]
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
        # The wave solver: -0.39, -0.05 and -0.01 dB at 1, 2 and 4 m, which are 2.5, 4.9
        # and 9.9 cells from the source. Within three cells of it a listener reads two
        # cells a side, and the level changes with the source's place in its cell and
        # the direction: -0.59 to +0.61 dB two cells away and -0.13 to +0.31 dB three
        # cells away (the table of places in docs/validation.md, Free field).
        assert 10 * math.log10(direct[0]) == pytest.approx(0.0, abs=0.5 if d < 2 else 0.1)
        assert 10 * math.log10(direct[1]) == pytest.approx(0.0, abs=0.1)  # rays


def _readings(solver, cell: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Of a solver laid out by :func:`_solver`: the centres (n, 3) and the weights (n,)
    of the cells each probe reads, then of the cells the source feeds."""
    layout = solver.layout()
    nx, ny, _ = (int(n) for n in layout["shape"])
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))

    def centres(cells) -> np.ndarray:
        cells = np.asarray(cells, dtype=np.int64)
        index = np.column_stack([cells % nx, cells // nx % ny, cells // (nx * ny)])
        return origin + (index + 0.5) * cell

    start = np.asarray(layout["probe_start"], dtype=np.int64)
    cells = np.asarray(layout["probe_cells"])
    weights = np.asarray(layout["probe_weights"], dtype=float)
    out = [(centres(cells[a:b]), weights[a:b]) for a, b in itertools.pairwise(start)]
    fed = np.flatnonzero(np.asarray(layout["source"]))
    source = np.asarray(layout["source"], dtype=float)[fed]
    return [*out, (centres(np.asarray(layout["special"])[fed]), source)]


def _polynomial(degree: int | tuple[int, int, int], seed: int = 3):
    """A polynomial of ``degree`` along each of x, y and z (or of one degree for each),
    of points (n, 3)."""
    degrees = (degree,) * 3 if isinstance(degree, int) else degree
    c = np.random.default_rng(seed).normal(size=tuple(d + 1 for d in degrees))
    return lambda p: np.polynomial.polynomial.polyval3d(p[..., 0], p[..., 1], p[..., 2], c)


@STENCILS
def test_a_point_between_cells_takes_a_cubic_through_four_cells_a_side(stencil: int) -> None:
    # The source and a probe 15 cells from it, neither on a cell's centre: 4 x 4 x 4
    # cells each, whose weights give any cubic's value at the point from its values at
    # the cells' centres (and sum to one). A probe 1.7 cells from the source: the 8 cells
    # round it, weighted trilinearly.
    source = np.array([1.03, 1.52, 1.47])
    far, near = np.array([2.51, 1.38, 1.61]), np.array([1.20, 1.50, 1.45])
    solver, _ = _solver((4.0, 3.0, 3.0), lambda mesh: [0.0] * 12, source, [far, near])
    cubic, line = _polynomial(3), _polynomial(1)
    (far_at, far_w), (near_at, near_w), (source_at, source_w) = _readings(solver, 0.1)
    assert len(far_w) == 64 and far_w.sum() == pytest.approx(1.0, abs=1e-12)
    assert far_w @ cubic(far_at) == pytest.approx(cubic(far), rel=1e-10)
    assert np.abs(far_at - far).max() < 0.2 + 1e-9
    assert len(source_w) == 64 and source_w.sum() == pytest.approx(1.0, abs=1e-6)
    assert source_w @ cubic(source_at) == pytest.approx(cubic(source), rel=1e-5)  # float32
    assert len(near_w) == 8 and near_w.min() > 0.0 and near_w.sum() == pytest.approx(1.0)
    assert near_w @ line(near_at) == pytest.approx(line(near), rel=1e-10)
    assert not np.isclose(near_w @ cubic(near_at), cubic(near), rtol=1e-6)


def test_a_reading_keeps_to_its_side_of_a_surface() -> None:
    # A screen across the box at x = 2 m (y 0.5 to 2.5, floor to 2 m) and another 0.25 m
    # behind it. The four cells a side round a probe 1.2 cells from the box's wall, or
    # 0.7 cells in front of the first screen, hold a layer behind that surface: without
    # it, the two cells round the probe along x, weighted linearly, and four by four
    # along y and z, weighted by the cubic. A probe 0.4 cells in front of the screen,
    # the next centre across it: the one layer on its side. One in the slot between the
    # screens, 2.5 cells wide: the two cells round it. One by the screen's upper edge,
    # which sees some of the cells across it: the cells round it that it sees, weighted
    # trilinearly. No reading reaches past the two cells either side of its probe.
    box = box_mesh((4.0, 3.0, 3.0))
    screens = [[[x, 0.5, 0.0], [x, 2.5, 0.0], [x, 2.5, 2.0], [x, 0.5, 2.0]] for x in (2.0, 2.25)]
    n = len(box.vertices)
    plates = [[k, k + 1, k + 2] for k in (n, n + 4)] + [[k, k + 2, k + 3] for k in (n, n + 4)]
    mesh = dataclasses.replace(
        box,
        vertices=np.vstack([box.vertices, *screens]),
        faces=np.vstack([box.faces, plates]),
        face_labels=np.append(box.face_labels, [0, 0, 0, 0]),
    )
    probes = np.array(
        [
            [0.12, 1.52, 1.47],
            [1.93, 1.38, 1.01],
            [1.96, 1.38, 1.01],
            [2.13, 1.55, 0.93],
            [1.96, 1.38, 1.97],
        ]
    )
    wall, front, close, slot, edge = probes
    solver, _ = _solver(
        (4.0, 3.0, 3.0), lambda mesh: [0.0] * len(mesh.faces), (1.0, 1.5, 1.5), probes, mesh=mesh
    )
    across, along = _polynomial((1, 3, 3)), _polynomial((0, 3, 3))
    readings = _readings(solver, 0.1)[:-1]
    for probe, (at, weights) in zip(probes, readings, strict=True):
        assert weights.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.abs(at - probe).max() < 0.2 + 1e-9
    (wall_at, wall_w), (front_at, front_w), (close_at, close_w), (slot_at, slot_w), _ = readings
    for probe, at, weights, low, high in (
        (wall, wall_at, wall_w, 0.0, 0.2),
        (front, front_at, front_w, 1.8, 2.0),
        (slot, slot_at, slot_w, 2.0, 2.2),
    ):
        assert len(weights) == 32 and len(np.unique(np.round(at[:, 0], 6))) == 2
        assert low < at[:, 0].min() < probe[0] < at[:, 0].max() < high
        assert weights @ across(at) == pytest.approx(across(probe), rel=1e-10)
    assert len(close_w) == 16 and np.ptp(close_at[:, 0]) < 1e-9 and 1.9 < close_at[0, 0] < 2.0
    assert close_w @ along(close_at) == pytest.approx(along(close), rel=1e-10)
    edge_at, edge_w = readings[4]
    assert 4 < len(edge_w) < 8 and edge_w.min() > 0.0
    assert np.abs(edge_at - edge).max() < 0.1 and np.any(edge_at[:, 0] > 2.0)


def test_the_source_next_to_a_wall_keeps_its_cells_in_the_room() -> None:
    # The source 0.3 cells from the wall x = 0, nearer it than the first centres of cells
    # in the room: that layer of cells alone along x, four by four along y and z, whose
    # weights give a cubic's value at the source's place in that layer (as a listener
    # there takes them). 1.2 cells from the wall: the first two layers, weighted linearly
    # along x.
    for x, cells, polynomial in (
        (0.03, 16, _polynomial((0, 3, 3))),
        (0.12, 32, _polynomial((1, 3, 3))),
    ):
        source = np.array([x, 1.52, 1.47])
        solver, _ = _solver((4.0, 3.0, 3.0), lambda mesh: [0.0] * 12, source, [[2.0, 1.5, 1.5]])
        _, (at, weights) = _readings(solver, 0.1)
        assert len(weights) == cells and weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < at[:, 0].min() < 0.1
        assert at[:, 0].max() == pytest.approx(at[:, 0].min() + (0.0 if cells == 16 else 0.1))
        assert weights @ polynomial(at) == pytest.approx(polynomial(source), rel=1e-5)  # float32


@STENCILS
def test_a_wall_behind_a_listener_does_not_turn_its_reading_to_the_source(stencil: int) -> None:
    # A listener 0.6 cells in front of a wall along the grid, and the source 2.99 and
    # 3.01 cells from it, on the other side: from three cells on the listener takes the
    # cubic along the wall and the two cells round it across the wall, none nearer the
    # source than the cells round the listener. The direct sound and the wall's, in the
    # octaves up to 8 and 16 cells per wavelength, change by under 0.2 dB between the two
    # (0.09 to 0.11 dB; with four cells pushed to the listener's side of the wall, which
    # reach to within a cell of the source, 0.24 and 0.36 dB in the lower octave).
    cell, size = 0.1, (4.0, 4.0, 4.0)
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    listener = origin + (np.array([20.37, 20.16, 21.29]) + 0.5) * cell
    wall_y = origin[1] + (20.16 - 0.61 + 0.5) * cell
    box = box_mesh(size)
    far = 10.0
    wall = [[-far, wall_y, -far], [far, wall_y, -far], [far, wall_y, far], [-far, wall_y, far]]
    n = len(box.vertices)
    mesh = dataclasses.replace(
        box,
        vertices=np.vstack([box.vertices, wall]),
        faces=np.vstack([box.faces, [[n, n + 1, n + 2], [n, n + 2, n + 3]]]),
        face_labels=np.append(box.face_labels, [0, 0]),
    )
    levels = []
    for away in (2.99, 3.01):
        source = listener + cell * np.array([0.05, away, -0.04])
        solver, rate = _solver(
            size,
            lambda mesh: [0.0] * len(mesh.faces),
            source,
            [listener],
            mesh=mesh,
            stencil=stencil,
        )
        (at, weights), _ = _readings(solver, cell)
        assert np.abs(at - listener).max() < (1.0 if away < 3.0 else 2.0) * cell + 1e-9
        assert np.all(at[:, 1] > wall_y) and len(weights) == (8 if away < 3.0 else 32)
        signal, delay = source_signal(rate, cell, _courant(stencil))
        steps = 2 * delay + round(0.006 * rate)  # before the box's walls' sound
        heard = solver.run(_drive(rate, cell, steps, stencil))[0]
        f = np.fft.rfftfreq(1 << 12, 1.0 / rate)
        spectrum = np.abs(np.fft.rfft(heard, 1 << 12)) ** 2
        levels.append(
            [10 * np.log10(spectrum[(f >= low) & (f <= 2 * low)].sum()) for low in (215.0, 107.5)]
        )
    assert np.abs(np.diff(levels, axis=0)).max() < 0.2


@STENCILS
def test_a_reading_next_to_a_slanted_surface_does_not_jump_along_it(stencil: int) -> None:
    # A wall at 45 degrees to the grid across the box, and listeners along it 0.3 cells
    # from it, 0.05 cells apart. Of the four cells a side round each, every layer across
    # the wall holds cells on the listener's side too: each reads the cells round it that
    # it sees, weighted trilinearly (three of their four columns, or one), none across
    # the wall. The direct sound's level in the octave up to 8 cells per wavelength
    # changes by 0.24 dB at most from one listener to the next, as it does with every
    # point read trilinearly.
    cell, size = 0.1, (4.0, 4.0, 2.4)
    across, along = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0), np.array([-1.0, 1.0, 0.0])
    along /= math.sqrt(2.0)
    on_wall, far = np.array([1.5, 2.0, 1.2]), np.array([0.0, 0.0, 10.0])
    box = box_mesh(size)
    wall = [on_wall + a * 10.0 * along + b * far for a, b in ((1, -1), (-1, -1), (-1, 1), (1, 1))]
    n = len(box.vertices)
    mesh = dataclasses.replace(
        box,
        vertices=np.vstack([box.vertices, wall]),
        faces=np.vstack([box.faces, [[n, n + 1, n + 2], [n, n + 2, n + 3]]]),
        face_labels=np.append(box.face_labels, [0, 0]),
    )
    source = on_wall + cell * (2.3 * across + 0.37 * along + [0.0, 0.0, 0.21])
    probes = np.array(
        [
            on_wall + cell * (0.3 * across + k * along + [0.0, 0.0, 0.13])
            for k in np.arange(4.0, 10.0, 0.05)
        ]
    )
    solver, rate = _solver(
        size, lambda mesh: [0.0] * len(mesh.faces), source, probes, mesh=mesh, stencil=stencil
    )
    for probe, (at, weights) in zip(probes, _readings(solver, cell)[:-1], strict=True):
        assert np.all((at - on_wall) @ across > 0.0) and len(weights) <= 8
        assert np.abs(at[:, :2] - probe[:2]).max() < cell and weights.sum() == pytest.approx(1.0)
    signal, delay = source_signal(rate, cell, _courant(stencil))
    farthest = np.linalg.norm(probes - source, axis=1).max()
    steps = 2 * delay + round((farthest / C + 0.006) * rate)  # before the box's walls' sound
    heard = solver.run(_drive(rate, cell, steps, stencil))
    n_fft = 1 << 12
    f = np.fft.rfftfreq(n_fft, 1.0 / rate)
    energy = np.sum(
        np.abs(np.fft.rfft(heard, n_fft, axis=1)[:, (f >= 215.0) & (f <= 430.0)]) ** 2, 1
    )
    assert np.abs(np.diff(10 * np.log10(energy))).max() < 0.4


@STENCILS
def test_the_level_does_not_change_with_the_place_within_a_cell(stencil: int) -> None:
    # The source and a listener 13 cells from it, moved together across a cell: the
    # direct sound's energy from 215 to 430 Hz (16 down to 8 cells per wavelength) stays
    # within 0.06 dB of what it is with both on cells' centres (0.03 to 0.04 dB under
    # it; read from the eight cells round each point, 0.67 to 0.81 dB under it).
    cell, size = 0.1, (6.0, 5.0, 5.0)
    origin = -cell * (2.0 + np.array([0.0137, 0.0219, 0.0291]))
    source = origin + (np.array([22, 25, 25]) + 0.5) * cell
    listener = source + cell * np.array([12.0, 4.0, 3.0])

    def level(shift) -> float:
        solver, rate = _solver(
            size,
            lambda mesh: [0.0] * 12,
            source + cell * np.array(shift),
            [listener + cell * np.array(shift)],
            stencil=stencil,
        )
        _, delay = source_signal(rate, cell, _courant(stencil))
        arrival = delay + round(1.3 * rate / C)
        count, taper = round(0.006 * rate), round(0.002 * rate)
        heard = solver.run(_drive(rate, cell, arrival + count, stencil))[0]
        heard[-taper:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
        n_fft = 1 << 12
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        return float(np.sum(np.abs(np.fft.rfft(heard, n_fft)[(f >= 215.0) & (f <= 430.0)]) ** 2))

    centred = level((0.0, 0.0, 0.0))
    for shift in ((0.5, 0.5, 0.5), (0.5, 0.0, 0.0), (0.3, 0.7, 0.45)):
        assert 10 * math.log10(level(shift) / centred) == pytest.approx(0.0, abs=0.06), shift


def test_the_19_point_grids_source_undoes_the_grids_own_gain() -> None:
    # The 19-point grid's signal is the one without them (level=False) through three
    # taps, a step later; with a slower speed of sound at the source, the side taps over
    # its square. The 7-point grid's has none either way.
    rate, cell = 2000.0, 0.2
    plain, delay = source_signal(rate, cell, ISOTROPIC_COURANT, level=False)
    signal, later = source_signal(rate, cell, ISOTROPIC_COURANT)
    assert later == delay + 1
    assert signal == pytest.approx(np.convolve(plain, [2 / 9, 5 / 9, 2 / 9]), abs=1e-12)
    slow, _ = source_signal(rate, cell, ISOTROPIC_COURANT, 0.9)
    side = 2 / (9 * 0.9**2)
    assert slow == pytest.approx(np.convolve(plain, [side, 1 - 2 * side, side]), abs=1e-12)
    # (Two calls, so within 1e-12 and not to the bit: np.convolve's sums are BLAS dot
    # products, which Accelerate on an Intel Mac rounds by where their vectors lie in
    # memory. Taps here would show as two samples more, or as 0.2 in the samples.)
    seven, start = source_signal(rate, cell)
    for other, at in (
        source_signal(rate, cell, COURANT, 0.9),
        source_signal(rate, cell, level=False),
    ):
        assert at == start and other == pytest.approx(seven, abs=1e-12)


@pytest.mark.parametrize("ratio", [1.0, 0.8])
def test_the_19_point_grid_gives_a_point_sources_level(ratio: float) -> None:
    # The direct sound 12 cells from the source along the cube's diagonal, in the octave
    # up to 8 cells per wavelength, against the exact field of a point source, the
    # signal without its taps delayed and over the distance: within 0.05 dB (-0.02 dB),
    # also where the speed of sound is 0.8 times the grid's everywhere and the taps
    # follow it. Without the taps the grid's own gain shows: 0.5 dB over it.
    cell, r = 0.1, 1.2
    source = np.array([1.83, 1.86, 1.81])
    probe = source + r / math.sqrt(3.0)
    n_fft = 1 << 13

    def level(taps: bool) -> float:
        solver, rate = _solver(
            (4.4, 4.4, 4.4),
            lambda mesh: [0.0] * 12,
            source,
            [probe],
            stencil=19,
            speed=None if ratio == 1.0 else lambda centres: np.full(len(centres), ratio),
        )
        plain, start = source_signal(rate, cell, ISOTROPIC_COURANT, level=False)
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        band = (f >= 215.0 * ratio) & (f <= 430.0 * ratio)
        signal, delay = source_signal(rate, cell, ISOTROPIC_COURANT, ratio, level=taps)
        steps = delay + round((r / (C * ratio) + 0.005) * rate)  # the walls' sound is later
        drive = np.zeros(steps, np.float32)
        drive[: len(signal)] = signal[:steps]
        heard = solver.run(drive)[0]
        late = np.exp(-2j * np.pi * f * (r / (C * ratio) + (delay - start) / rate))
        exact = np.fft.irfft(np.fft.rfft(plain, n_fft) * late, n_fft)[:steps]
        exact /= r * 4.0 * math.pi * ISOTROPIC_COURANT**2 / cell
        window = np.ones(steps)
        taper = round(0.002 * rate)
        window[-taper:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
        energy = [np.sum(np.abs(np.fft.rfft(x * window, n_fft)[band]) ** 2) for x in (heard, exact)]
        return 10 * math.log10(energy[0] / energy[1])

    assert level(True) == pytest.approx(0.0, abs=0.05)
    assert level(False) == pytest.approx(0.5, abs=0.05)


def test_points_among_the_sources_cells_are_left_out_of_a_gradient() -> None:
    # By the listener's place in cells from the source, the points round it that do not
    # count (1 to 6: a cell along +x, -x, +y, -y, +z, -z; 7 to 12: two cells): on the
    # 19-point grid those nearer the source than 1.5 cells and nearer it than the
    # listener; on the 7-point grid none.
    cell, source = 0.4, np.array([5.0, 4.0, 3.0])

    def left_out(offset, missing=(), stencil=19) -> list[int]:
        probes = _probe_points(source + cell * np.array(offset, dtype=float), cell)
        found = np.ones(13, dtype=bool)
        found[list(missing)] = False
        kept = _off_the_source(found, probes, source, cell, stencil)
        return [k for k, ok in enumerate(kept) if not ok]

    assert left_out((3.0, 0.0, 0.0)) == [8]
    assert left_out((2.0, 0.0, 0.0)) == [2, 8]
    assert left_out((1.5, 0.0, 0.0)) == [2, 8]
    assert left_out((1.2, 1.2, 0.0)) == [2, 4, 8, 10]
    assert left_out((2.0, 0.8, 0.5)) == [2, 8]
    assert left_out((0.0, -2.5, 0.0)) == [9]  # the point 1.5 cells from the source stays
    assert left_out((0.0, 0.0, 3.4)) == [12]
    assert left_out((0.0, 0.0, 3.5)) == []
    # A listener among the source's cells itself keeps the points that lie no nearer the
    # source than it does: 1.49 cells away the two towards the source go, as from 1.5
    # cells on; (0.9, 0.3, 0.1) loses the one 0.33 cells from the source and keeps those
    # 1.14 to 1.45 cells from it (itself 0.95); (0.95, 0.9, 0), 1.31 cells away, loses
    # those 0.90 and 0.96 cells from the source and keeps the two 1.38 and 1.45 cells
    # from it; within half a cell of the source along every axis, every point lies
    # further from it than the listener, or as far.
    assert left_out((1.49, 0.0, 0.0)) == [2, 8]
    assert left_out((0.9, 0.3, 0.1)) == [2]
    assert left_out((0.95, 0.9, 0.0)) == [2, 4]
    assert left_out((0.4, 0.3, 0.2)) == []
    assert left_out((0.5, 0.0, 0.0)) == []
    assert left_out((2.0, 0.0, 0.0), stencil=7) == []
    assert left_out((0.9, 0.3, 0.1), stencil=7) == []
    # A wall behind the listener, which hides the points away from the source: the axis
    # keeps the points towards it, or it would have none.
    assert left_out((2.0, 0.0, 0.0), missing=(1, 7)) == [1, 7]
    assert left_out((2.0, 0.8, 0.5), missing=(1, 7)) == [1, 7]
    assert left_out((0.9, 0.3, 0.1), missing=(1, 7)) == [1, 7]
    assert left_out((0.0, -2.5, 0.0), missing=(4, 10)) == [4, 9, 10]


def test_a_listener_next_to_the_source_leaves_the_sources_cells_out_of_its_gradient(
    monkeypatch,
) -> None:
    # Listeners off the grid's axes, 2.2, 0.95 and 0.58 cells from the source. On the
    # 19-point grid the first goes without the two points towards the source along x
    # (1.4 and 0.9 cells from it), the second without the one a cell towards it (0.33
    # cells from it), and the third, nearer the source than any of its points, keeps
    # them all; on the 7-point grid all three keep theirs. The gradient's main axis lies
    # within 8 degrees of the line to the source (0.8, 0.6 and 2.8 degrees on the
    # 19-point grid, 5.3, 3.6 and 6.1 on the 7-point one), and its part in step with the
    # pressure points to the source.
    room = _free_room((1.0,))
    cell = band_grid(0, FS, room.speed_of_sound(), 8.0, 8, ISOTROPIC_COURANT).cell
    source = np.array(room.scene.sources[0].position, dtype=float)
    offsets = np.array([[2.0, 0.8, 0.5], [0.9, 0.3, 0.1], [0.45, 0.3, 0.2]])
    used = []
    real = wave_stage._first_order

    def spy(probes, found, *args):
        used.append([k for k, ok in enumerate(found) if not ok])
        return real(probes, found, *args)

    monkeypatch.setattr(wave_stage, "_first_order", spy)
    for scheme, without in (("isotropic", [[2, 8], [2], []]), ("standard", [[], [], []])):
        used.clear()
        responses, _ = wave_responses(
            TraceModel.from_room(room),
            source,
            source + cell * offsets,
            ["far", "near", "nearest"],
            max_band_hz=63,
            length_s=0.05,
            sample_rate=FS,
            speed_of_sound=room.speed_of_sound(),
            scheme=scheme,
        )
        assert used == without, scheme
        for response, offset in zip(responses, offsets, strict=True):
            line = offset / np.linalg.norm(offset)
            gradient = response.signals[:, 1:]
            main = np.linalg.eigh(gradient.T @ gradient)[1][:, -1]
            angle = math.degrees(math.acos(min(1.0, abs(float(main @ line)))))
            assert angle < 8.0, (scheme, offset)
            assert (response.signals[:, :1] * gradient).sum(axis=0) @ line < 0.0


def test_a_long_run_records_its_tenths_as_one_run_does(monkeypatch) -> None:
    """A long run steps in tenths, reporting each, into one array of the probes'
    pressures: the responses are those of one run, to the bit."""
    from acoustic_spaces.engine import wave as wave_module

    room = _free_room((4.0,))
    source = np.array(room.scene.sources[0].position)

    def run():
        said: list[str] = []
        responses, _ = wave_responses(
            TraceModel.from_room(room),
            source,
            [source + [4.0, 0.0, 0.0]],
            ["r"],
            max_band_hz=63,
            length_s=0.1,
            sample_rate=FS,
            speed_of_sound=room.speed_of_sound(),
            say=said.append,
        )
        return responses[0].signals, [s for s in said if s.endswith(" %")]

    once, reported = run()
    assert reported == []
    monkeypatch.setattr(wave_module, "_LONG_RUN", 0)
    tenths, reported = run()
    assert [int(s.split(", ")[-1][:-2]) for s in reported] == list(range(10, 100, 10))
    np.testing.assert_array_equal(tenths, once)


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
    signal, delay = source_signal(wave.rate, wave.cell, ISOTROPIC_COURANT)
    lead = round(wave.lead_s * wave.rate)  # the run starts that many steps before emission
    n_fft = 1 << 14
    f = np.fft.rfftfreq(n_fft, 1.0 / wave.rate)
    band = np.clip((f - 25) / 10, 0, 1) * np.clip((100 - f) / 20, 0, 1)
    spectrum = np.fft.rfft(wave.signals[:, 0], n_fft) * np.conj(np.fft.rfft(signal, n_fft))
    correlation = np.roll(np.fft.irfft(spectrum * band**2, n_fft), n_fft // 2)
    peak = int(np.argmax(correlation[: n_fft // 2 + 200]))
    y0, y1, y2 = correlation[peak - 1 : peak + 2]
    arrival = delay - lead + peak - n_fft // 2 + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    assert arrival == pytest.approx(4.0 / c * wave.rate, abs=0.25)


def test_a_listener_near_the_source_hears_all_of_its_pulse(monkeypatch) -> None:
    """The source's pulse is centred on emission and starts 16 steps before it (3.7 m of
    travel on the 7-point grid of the 63 Hz band), so a listener nearer than that hears
    part of it before emission: the run keeps that part. The direct sound's energy in the
    band, the pressure's and the figure of eight's facing the source, then does not
    depend on the source's low-pass, which shapes only frequencies far above the band:
    its cutoff from 0.2 to 0.3 of the grid's sample rate moves them by less than 0.01 dB
    (0.004 dB at most) at 0.5 and 1 m. Cut at emission, they moved by 0.04 to 0.22 dB
    and 0.5 to 0.7 dB."""
    from acoustic_spaces.engine import wave as wave_module

    distances = (0.5, 1.0)
    room = _free_room(distances)
    c = room.speed_of_sound()
    source = np.array(room.scene.sources[0].position)
    step, slots = 1e-3, 40
    levels = []
    for cutoff in (0.2, 0.3):
        monkeypatch.setattr(wave_module, "_LOW_PASS_CUTOFF", cutoff)
        responses, _ = wave_responses(
            TraceModel.from_room(room),
            source,
            [source + [d, 0.0, 0.0] for d in distances],
            [f"r{d}" for d in distances],
            max_band_hz=63,
            length_s=0.04,
            sample_rate=FS,
            speed_of_sound=c,
            scheme="standard",
        )
        for d, wave in zip(distances, responses, strict=True):
            direct = np.arange(slots) * step < (d + 6.5) / c  # before the first reflection
            omni = wave.energy(step, slots)[0, direct].sum()
            eight = wave.energy(step, slots, axis=np.array([-1.0, 0.0, 0.0]))[0, direct].sum()
            levels.append(10 * np.log10([omni, eight]))
    np.testing.assert_allclose(levels[:2], levels[2:], atol=0.01)


def test_a_run_from_before_emission_lines_up_with_a_delayed_one() -> None:
    """A wave response whose run starts before emission (its lead) is one whose run starts
    at emission, earlier by the lead: fed a delay, it is that response fed the delay
    less the lead, in its field and in its energy per band and slot. Without a feed, its
    field from emission is what the bands' filters make of the whole run, the part
    before emission too: the field of the run starting at emission, from the lead on."""
    rate, lead, n = 2000.0, 0.008, 9000
    signals = np.random.default_rng(1).standard_normal((300, 4))
    gain = np.ones(2)
    early = WaveResponse(rate, signals, 2, 0.2, lead_s=lead)
    late = WaveResponse(rate, signals, 2, 0.2)
    a, b = early.fed(gain, 0.05), late.fed(gain, 0.05 - lead)
    np.testing.assert_allclose(a.field(FS, n), b.field(FS, n), rtol=0, atol=1e-12)
    step = 1e-3 * math.sqrt(2)  # no slot's edge on a sample
    for axis in (None, np.array([0.6, 0.0, 0.8])):
        np.testing.assert_allclose(a.energy(step, 60, axis), b.energy(step, 60, axis), rtol=1e-9)
    shift = round(lead * FS)
    np.testing.assert_allclose(early.field(FS, n), late.field(FS, n + shift)[shift:], atol=1e-12)


@pytest.mark.parametrize("delay", [192.55 / FS, 1200.52 / FS])
def test_a_run_from_before_emission_lands_within_half_a_sample_at_another_rate(
    delay: float,
) -> None:
    """At a rate where the lead is not a whole number of samples (16 steps at 44100 / 22
    Hz, the 125 Hz band's grid of a 44.1 kHz scene, are 383.13 samples at 48 kHz), a run
    fed a delay shorter or longer than its lead lands where one that starts at emission,
    fed the delay less the lead, does: within half a sample of its time (0.42 and 0.39
    samples early here; with the delay and the lead rounded apart, 0.58 and 0.61 samples
    late)."""
    rate = 44100 / 22
    lead, n, whole = 16 / rate, 24_000, 9600
    signals = np.random.default_rng(1).standard_normal((300, 4))
    gain = np.ones(2)
    field = WaveResponse(rate, signals, 2, 0.2, lead_s=lead).fed(gain, delay).field(FS, n)
    later = WaveResponse(rate, signals, 2, 0.2).fed(gain, delay - lead)
    np.testing.assert_allclose(field, later.field(FS, n), rtol=0, atol=1e-12)
    # The run placed exactly: fed a whole number of samples, then moved by the rest.
    placed = WaveResponse(rate, signals, 2, 0.2).fed(gain, whole / FS).field(FS, 4 * n)
    f = np.fft.rfftfreq(4 * n)  # cycles per sample
    turn = np.exp(-2j * np.pi * f * ((delay - lead) * FS - whole))
    spectrum = np.fft.rfft(placed, axis=0) * turn[:, None]
    exact = np.fft.irfft(spectrum, 4 * n, axis=0)[:n]
    slope = np.fft.irfft(spectrum * (2j * np.pi * f)[:, None], 4 * n, axis=0)[:n]
    # field(t) = exact(t - late) ~ exact(t) - late * slope(t), in samples.
    late = -np.sum((field - exact) * slope) / np.sum(slope**2)
    assert abs(late) < 0.5


def test_velocity_is_in_step_with_the_pressure() -> None:
    # A plane wave from -x crosses the probe and its neighbours one and two cells away
    # (1 / COURANT steps a cell): X = -W sample by sample, not half a step ahead of it.
    t = np.arange(400.0)
    shift = 1.0 / COURANT
    probes = np.zeros((13, len(t)))
    for k, cells in ((0, 0), (1, 1), (2, -1), (7, 2), (8, -2)):
        probes[k] = np.exp(-(((t - 200.0 - cells * shift) / 16.0) ** 2))
    field = _first_order(probes, [True] * 13, "l")
    np.testing.assert_allclose(field[:, 1], -field[:, 0], atol=0.01)
    assert not np.any(field[:, 2:])


def test_velocity_falls_back_to_the_points_it_has() -> None:
    """Along an axis the gradient is the fourth-order central difference over two cells
    either side, the second-order one over one cell where a point two cells away is not
    found, and the one-sided difference to the listener where a point one cell away is
    not; a point not found never counts (its row is noise here). With no gain divided out
    (cutoff 0), X is c times the trapezoidal rule's integral of that difference, exactly;
    and Y and Z likewise, from their own points."""
    t = np.arange(400.0)
    shift = 1.0 / COURANT
    wave_at = {
        cells: np.exp(-(((t - 200.0 - cells * shift) / 16.0) ** 2)) for cells in range(-2, 3)
    }
    fourth = (8 * (wave_at[1] - wave_at[-1]) - (wave_at[2] - wave_at[-2])) / 12
    second = 0.5 * (wave_at[1] - wave_at[-1])
    ahead, behind = wave_at[1] - wave_at[0], wave_at[0] - wave_at[-1]
    # By the points missing, as they are numbered along x: 1 and 2 a cell either side,
    # 7 and 8 two cells either side.
    cases = {
        (): fourth,
        (7,): second,
        (8,): second,
        (7, 8): second,
        (1,): behind,
        (1, 7): behind,
        (1, 8): behind,
        (1, 7, 8): behind,
        (2,): ahead,
        (2, 8): ahead,
        (2, 7): ahead,
        (2, 7, 8): ahead,
        (1, 2): np.zeros(len(t)),
        (1, 2, 7, 8): np.zeros(len(t)),
    }
    noise = np.random.default_rng(2).standard_normal(len(t))
    for axis in range(3):
        p = np.zeros((13, len(t)))
        p[0] = wave_at[0]
        for k, cells in ((1, 1), (2, -1), (7, 2), (8, -2)):
            p[k + 2 * axis] = wave_at[cells]
        for missing, difference in cases.items():
            probes, found = p.copy(), [True] * 13
            for k in missing:
                probes[k + 2 * axis], found[k + 2 * axis] = noise, False
            field = _first_order(probes, found, "l", cutoff=0.0)
            expected = COURANT * (np.cumsum(difference) - 0.5 * difference)
            np.testing.assert_allclose(
                field[:, 1 + axis], expected, rtol=0, atol=1e-12, err_msg=f"{axis} {missing}"
            )
            assert not np.any(np.delete(field[:, 1:], axis, axis=1))


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


@pytest.mark.parametrize("scheme", ["isotropic", "standard"])
def test_first_order_field_matches_a_point_source(scheme: str) -> None:
    """The direct sound's figure of eight facing the source over its pressure, in energy,
    per wave band, 4 m from the source along a grid axis, a face diagonal and a body
    diagonal: a point source's particle velocity is p / (rho c) times 1 + 1 / (j k r)
    (the spherical wave; Kinsler et al., Fundamentals of Acoustics), so 10 log10(1 + 1 /
    (k r)^2) at the band's centre. Within 0.06 dB (0.040 dB at most here). Second-order
    differences fall 0.35 dB (19-point) and 0.31 dB (7-point) short along the axis in the
    125 Hz band; with the trapezoidal rule's gain left in, the fourth-order ones fall up
    to 0.15 and 0.075 dB short in that band over the three directions. A Courant number
    1.5 times off is 3.5 dB."""
    room = _free_room((4.0,), max_band_hz=125, box=(24.0, 20.0, 20.0))
    c = room.speed_of_sound()
    source = np.array(room.scene.sources[0].position)
    directions = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    responses, info = wave_responses(
        TraceModel.from_room(room),
        source,
        source + 4.0 * directions,
        ["axis", "face", "body"],
        max_band_hz=125,
        length_s=0.05,
        sample_rate=FS,
        speed_of_sound=c,
        scheme=scheme,
    )
    assert info["wave_stencil"] == (19 if scheme == "isotropic" else 7)
    step, slots = 1e-3, 50
    direct = np.arange(slots) * step < 0.045  # before the first reflection (at 52 ms)
    for towards, wave in zip(directions, responses, strict=True):
        omni = wave.energy(step, slots)[:, direct].sum(axis=1)
        eight = wave.energy(step, slots, axis=-towards)[:, direct].sum(axis=1)
        for band, centre in enumerate((63, 125)):
            kr = 2 * math.pi * centre * 4.0 / c
            expected = 10 * math.log10(1 + 1 / kr**2)
            assert 10 * math.log10(eight[band] / omni[band]) == pytest.approx(expected, abs=0.06)


@pytest.mark.parametrize("scheme", ["standard", "isotropic"])
def test_velocity_takes_no_difference_across_a_thin_screen(tmp_path: Path, scheme: str) -> None:
    """A listener 0.1 m in front of a rigid screen with air behind it: the point a cell
    from the listener towards the screen lies across it, in air of its own, so the
    gradient there is a one-sided difference, as in front of a block with the same front
    face (rigid, so the velocity normal to it falls as the listener nears it). X / W in
    the 63 Hz band within 1 dB of the block's; taken across the screen, it was 6 dB
    (7-point) and 15 dB (19-point) above it. A listener 1.5 cells in front of the screen
    has the point a cell towards it in front and the point two cells towards it across,
    so its gradient is of second order there, as in front of the block: within 0.5 dB of
    the block's (0.2 dB at most here); taken across the screen to fourth order, it was
    2.2 dB (7-point) and 2.3 dB (19-point) above it."""
    room_mesh, block = box_mesh((10.0, 8.0, 4.0)), box_mesh((1.0, 6.0, 3.0))
    screens = {
        "thin": (
            [[5.0, 1.0, 0.0], [5.0, 7.0, 0.0], [5.0, 7.0, 3.0], [5.0, 1.0, 3.0]],
            [[0, 1, 2, 3]],
        ),
        "block": (block.vertices + [5.0, 1.0, 0.0], block.faces),
    }
    source, listener = np.array([2.0, 4.03, 1.51]), np.array([4.9, 4.03, 1.51])
    ratios: dict[str, list[float]] = {}
    for name, (corners, faces) in screens.items():
        n = len(room_mesh.vertices)
        lines = [f"v {x} {y} {z}" for x, y, z in [*room_mesh.vertices, *corners]]
        lines += ["g room", *(f"f {' '.join(str(i + 1) for i in f)}" for f in room_mesh.faces)]
        lines += ["g screen", *(f"f {' '.join(str(n + i + 1) for i in f)}" for f in faces)]
        obj = tmp_path / f"{name}.obj"
        obj.write_text("\n".join(lines) + "\n")
        room = Room.from_scene(
            Scene.model_validate(
                {
                    "geometry": {"file": str(obj)},
                    "materials": {
                        "a": {"absorption": [0.6] * 8, "scattering": 0.0},
                        "hard": {"absorption": [0.02] * 8, "scattering": 0.0},
                    },
                    "assignments": {"room": "a", "screen": "hard"},
                    "sources": [{"name": "s", "position": source.tolist()}],
                    "listeners": [{"name": "l", "position": listener.tolist()}],
                }
            )
        )
        c = room.speed_of_sound()
        grid = band_grid(0, FS, c, courant=COURANT if scheme == "standard" else ISOTROPIC_COURANT)
        farther = np.array([5.0 - 1.5 * grid.cell, 4.03, 1.51])
        waves, info = wave_responses(
            TraceModel.from_room(room),
            source,
            [listener, farther],
            ["l", "farther"],
            max_band_hz=63,
            length_s=0.15,
            sample_rate=FS,
            speed_of_sound=c,
            scheme=scheme,
        )
        h = info["wave_cell_m"]
        assert listener[0] + h > 5.0  # the point a cell towards the screen is past it
        assert farther[0] + h < 5.0 < farther[0] + 2 * h  # in front of it; two cells, past
        ratios[name] = []
        for wave in waves:
            x = wave.energy(1e-3, 150, axis=np.array([1.0, 0.0, 0.0]))[0].sum()
            ratios[name].append(10 * math.log10(x / wave.energy(1e-3, 150)[0].sum()))
    assert ratios["thin"][0] == pytest.approx(ratios["block"][0], abs=1.0)
    assert ratios["thin"][1] == pytest.approx(ratios["block"][1], abs=0.5)


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
    # At least 8 cells per wavelength at the top of the 125 Hz band, and the waves there
    # as slow along the axes as the 19-point stencil makes them.
    per_wavelength = room.speed_of_sound() / (band_top(1) * info["wave_cell_m"])
    assert 8.0 <= per_wavelength < 8.5
    assert info["wave_slow_percent"] == pytest.approx(100 * slow_by(19, per_wavelength))
    assert info["wave_slow_percent"] <= 100 * slow_by(19, 8.0)
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
    # The 8 kHz band has no top edge to make a grid for; the schema stops at 500 Hz.
    for band in (100, 1000, 8000):
        with pytest.raises(ValueError, match="max_band_hz must be one of"):
            wave_responses(model, source, [source], ["l"], max_band_hz=band, **common)


def test_wave_responses_check_what_they_are_given() -> None:
    # The scene's schema checks these. Called directly, a mistyped scheme ran the
    # isotropic stencil and a mistyped wind the effective one, a length of 0 ran with no
    # samples, and a name short of the listeners left one out.
    room = Room.from_yaml(SHOEBOX)
    model = TraceModel.from_room(room)
    source = np.array(room.scene.sources[0].position)
    common = {"max_band_hz": 63, "length_s": 0.1, "sample_rate": FS, "speed_of_sound": C}
    for more, message in (
        ({"scheme": "isotropc"}, "unknown scheme 'isotropc'"),
        ({"wind": "moved"}, "unknown wind 'moved'"),
        ({"length_s": 0.0}, "length_s must be a positive number, not 0.0"),
        ({"sample_rate": -48000.0}, "sample_rate must be a positive number"),
        ({"points_per_wavelength": math.nan}, "points_per_wavelength must be a positive"),
    ):
        with pytest.raises(ValueError, match=message):
            wave_responses(model, source, [source], ["l"], **{**common, **more})
    with pytest.raises(ValueError, match="1 names for 2 listeners"):
        wave_responses(model, source, [source, source], ["l"], **common)
    # Every face an opening: no wall for the grid, which failed with "Bvh: the mesh has
    # no faces".
    openings = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"box": [4.0, 3.0, 2.5]},
                "materials": {"gap": {"absorption": [1.0] * 8, "transmission": [1.0] * 8}},
                "default_material": "gap",
                "sources": [{"name": "s", "position": [1.0, 1.0, 1.0]}],
                "listeners": [{"name": "l", "position": [3.0, 2.0, 1.2]}],
            }
        )
    )
    with pytest.raises(ValueError, match="needs a surface that reflects"):
        wave_responses(
            TraceModel.from_room(openings), [1.0, 1.0, 1.0], [[3.0, 2.0, 1.2]], ["l"], **common
        )


def test_the_wave_bands_match_the_schema() -> None:
    from typing import get_args

    from acoustic_spaces import cli
    from acoustic_spaces.scene.model import WaveStage

    assert WAVE_BANDS_HZ == cli.WAVE_BANDS_HZ == (63, 125, 250, 500)
    assert get_args(WaveStage.model_fields["max_band_hz"].annotation) == WAVE_BANDS_HZ


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


def test_open_models_absorb_at_the_grid_edge(tmp_path: Path, monkeypatch) -> None:
    # An open model's first solver (no layer) goes before the one with the layer is made.
    alive, most = [0], [0]

    class Counted(_native.WaveSolver):
        def __init__(self, *args) -> None:
            super().__init__(*args)
            alive[0] += 1
            most[0] = max(most[0], alive[0])

        def __del__(self) -> None:
            alive[0] -= 1

    monkeypatch.setattr(_native, "WaveSolver", Counted)
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
    assert info["wave_stencil"] == 7  # the matched layer needs the standard stencil
    per_wavelength = C / (band_top(0) * info["wave_cell_m"])
    assert info["wave_slow_percent"] == pytest.approx(100 * slow_by(7, per_wavelength))
    assert most[0] == 1
    with pytest.raises(ValueError, match="no matched layer"):
        wave_responses(
            model,
            np.array([2.0, 2.0, 2.5]),
            [[6.0, 3.0, 3.0]],
            ["l"],
            max_band_hz=63,
            length_s=0.3,
            sample_rate=FS,
            speed_of_sound=C,
            scheme="isotropic",
        )
    pressure = responses[0].signals[:, 0]
    rate = responses[0].rate
    # The sound leaves: little is left 150 ms after emission (the run starts lead_s
    # before it).
    late = pressure[round((0.15 + responses[0].lead_s) * rate) :]
    assert np.sum(late**2) < 1e-2 * np.sum(pressure**2)


def test_outdoor_models_make_only_the_grid_with_the_layer(monkeypatch) -> None:
    """Outdoors the sky is an opening, so the grid is open unless the source is shut in:
    auto and standard make the standard stencil's grid with the matched layer and no
    other (a grid without it, made to find the model open, went unused). Only where
    that grid is closed (a source in a hut) do they make the one without it, as in a
    closed room: auto the isotropic stencil's. The isotropic stencil is refused
    outdoors before any grid is made, for a source shut in too."""
    made: list[tuple[int, int]] = []

    class Counted(_native.WaveSolver):
        def __init__(self, *args) -> None:
            super().__init__(*args)
            made.append((args[16], args[13]))  # stencil, layer

    monkeypatch.setattr(_native, "WaveSolver", Counted)
    room = _windy_ground(0.0, "effective")
    for scheme in ("auto", "standard"):
        made.clear()
        _, info, _ = _windy_run(room, "effective", scheme=scheme)
        assert info["wave_open"] == 1.0 and made == [(7, LAYER_CELLS)]
    hut = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {
                    "ground": {"extent": [[-10, -6], [10, 6]], "sky": 5},
                    "buildings": [
                        {
                            "name": "hut",
                            "footprint": [[-3, -3], [3, -3], [3, 3], [-3, 3]],
                            "height": 4,
                        }
                    ],
                },
                "materials": {"m": {"absorption": [0.2] * 8, "scattering": 0.0}},
                "default_material": "m",
                "sources": [{"name": "s", "position": [0, 0, 1.5]}],
                "listeners": [{"name": "in", "position": [1.2, 0.7, 1.4]}],
            }
        )
    )
    for scheme, stencil in (("auto", 19), ("standard", 7)):
        made.clear()
        _, info, _ = _windy_run(hut, "effective", scheme=scheme)
        assert "wave_open" not in info and info["wave_stencil"] == stencil
        assert made == [(7, LAYER_CELLS), (stencil, 0)]
    for outdoors in (room, hut):
        made.clear()
        with pytest.raises(ValueError, match="outdoors the isotropic stencil is not available"):
            _windy_run(outdoors, "effective", scheme="isotropic")
        assert made == []


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


def _discrete_reflection(
    f: np.ndarray, rate: float, y: np.ndarray, courant: float = COURANT
) -> np.ndarray:
    """Reflection at normal incidence off a wall of admittance y (the scheme's, at that
    rate) in the grid: a plane wave along an axis, the wall half a cell past the last
    centre. With time going as e^(-i w t), so y is the conjugate of Y(i w). Either
    stencil reduces to the one-dimensional scheme at its Courant number there."""
    phi = 2 * np.pi * f / rate
    shift = np.exp(2j * np.arcsin(np.minimum(np.sin(phi / 2) / courant, 1.0)))  # one cell
    c = 1j * np.sin(phi) * y / courant
    return (c - shift + 1) / (1 / shift - 1 - c)


@STENCILS
def test_duct_end_reflects_as_its_frequency_dependent_admittance_says(stencil: int) -> None:
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
            size,
            admittances,
            (60.0, 0.15, 0.15),
            (110.0, 0.15, 0.15),
            branches=far,
            stencil=stencil,
        )
        return solver.run(_drive(rate, 0.1, int(0.6 * rate), stencil))[0], rate

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
    expected = np.abs(_discrete_reflection(f, rate, np.conj(resonant(f, rate)), _courant(stencil)))
    np.testing.assert_allclose(measured, expected, atol=0.012)
    # It absorbs most at resonance, as it should: little comes back at 100 Hz.
    assert measured[np.argmin(np.abs(f - 100))] < 0.25 < measured[np.argmin(np.abs(f - 40))]


def _plate(
    h: float,
    pad: int,
    layer: int,
    seconds: float,
    damping: float = LAYER_DAMPING,
    speed=None,
    flow=None,
    block: bool = False,
):
    """The pressure at three points round a rigid 8 x 6 m plate in the open, on a grid
    that reaches two cells past the plate, the source and the points (with none of the
    room :data:`LAYER_ROOM` leaves), then ``pad`` more cells and ``layer`` cells of
    matched layer; ``speed`` and ``flow`` as for :func:`_solver`; ``block`` stands a
    rigid block 0.8 x 3 x 2 m on the plate, across x from 4.6 m."""
    vertices = np.array([[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    if block:
        solid = box_mesh((0.8, 3.0, 2.0))
        faces = np.vstack([faces, solid.faces + len(vertices)])
        vertices = np.vstack([vertices, solid.vertices + [4.6, 1.5, 0.0]])
    bvh = _native.Bvh(vertices, faces)
    source = np.array([2.0, 2.0, 1.5])
    listeners = np.array([[6.0, 4.0, 1.2], [4.0, 3.0, 3.0], [7.5, 1.0, 0.5]])
    points = np.vstack([source, listeners])
    low = np.minimum(vertices.min(0), points.min(0) - 2 * h)
    high = np.maximum(vertices.max(0), points.max(0) + 2 * h)
    origin = low - h * (2.0 + pad + np.array([0.0137, 0.0219, 0.0291]))
    shape = np.ceil((high - origin) / h).astype(int) + 2 + pad
    solver = _native.WaveSolver(
        bvh,
        np.zeros(len(faces), np.int64),
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
        speed=_speeds(speed, origin, h, shape),
        **_flows(flow, origin, h, shape),
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


def _along_the_layer(geometry: dict, source, listeners, distance: float) -> tuple:
    """The 63 Hz wave solver's response, at the first of ``listeners``, to ``source`` in
    a rigid open model of ``geometry``, ``distance`` away: from emission until just
    before what the model's far edges send; with the response's rate and its grid's cell."""
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": geometry,
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": source}],
                "listeners": [
                    {"name": f"l{i}", "position": point} for i, point in enumerate(listeners)
                ],
            }
        )
    )
    c = room.speed_of_sound()
    (response, *_), info = wave_responses(
        TraceModel.from_room(room),
        np.array(source, dtype=float),
        np.array(listeners, dtype=float),
        [f"l{i}" for i in range(len(listeners))],
        max_band_hz=63,
        length_s=(distance + 8.0) / c,
        sample_rate=FS,
        speed_of_sound=c,
    )
    assert info["wave_open"] == 1.0
    lead = round(response.lead_s * response.rate)
    n = round(response.rate * (distance + 7.0) / c)
    taper = round(response.rate * 0.008)
    window = np.ones(n)
    window[-taper:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
    return response.signals[lead : lead + n, 0] * window, response.rate, response.cell


def test_the_matched_layer_lies_clear_of_sound_that_runs_along_it() -> None:
    # Sound runs 60 m over rigid ground, 1.5 m up, under a sky 3 m up. With the matched
    # layer a few cells over the sky the sound met it near grazing, and it sent back
    # enough to put the 63 Hz band 0.80 dB below the same run under a sky 18 m up. The
    # grid now reaches LAYER_ROOM of the 60 m above the source and the listener: within
    # 0.01 dB.
    distance, height = 60.0, 1.5
    half_width = math.sqrt((distance / 2 + 4.0) ** 2 - (distance / 2) ** 2) + 0.5
    runs = [
        _along_the_layer(
            {"ground": {"extent": [[-4.0, -half_width], [distance + 4.0, half_width]], "sky": sky}},
            [0.0, 0.0, height],
            [[distance, 0.0, height]],
            distance,
        )
        for sky in (3.0, 18.0)
    ]
    (low, rate, _), (high, _, _) = runs
    assert _band_db(low, high, rate, band=(40.0, 100.0)) == pytest.approx(0.0, abs=0.05)


def test_the_matched_layer_lies_clear_of_sound_along_the_sides() -> None:
    # Sound runs 40 m along a strip of rigid ground 3 m wide, 1.5 m up, under a sky 8 m
    # up: the grid's matched layer started two cells past the strip's edges, and the
    # 63 Hz band read 0.87 dB below the same run on a grid widened
    # by two more listeners 16.6 m off to either side, at whole cells from the strip's
    # edges so that both grids keep their place on it. Now within 0.01 dB.
    distance, half_width = 40.0, 1.5
    geometry = {
        "ground": {"extent": [[-4.0, -half_width], [distance + 4.0, half_width]], "sky": 8.0}
    }
    narrow, rate, h = _along_the_layer(geometry, [0.0, 0.0, 1.5], [[distance, 0.0, 1.5]], distance)
    side = half_width + 38 * h
    wide, _, _ = _along_the_layer(
        geometry,
        [0.0, 0.0, 1.5],
        [[distance, 0.0, 1.5], [distance / 2, side, 1.5], [distance / 2, -side, 1.5]],
        distance,
    )
    assert _band_db(narrow, wide, rate, band=(40.0, 100.0)) == pytest.approx(0.0, abs=0.05)


def test_the_matched_layer_lies_clear_of_sound_below_an_open_model(tmp_path: Path) -> None:
    # Sound runs 60 m under a rigid plate (a model file, nothing below it), 1.5 m under
    # it: the grid's matched layer started four cells under the source and the
    # listener, and the 63 Hz band read 1.67 dB below the same run on a
    # grid deepened by one more listener 11 m further down, at whole cells, so that both
    # grids keep their place on the plate. Now within 0.01 dB.
    distance = 60.0
    half_width = math.sqrt((distance / 2 + 4.0) ** 2 - (distance / 2) ** 2) + 0.5
    obj = tmp_path / "plate.obj"
    obj.write_text(
        f"v -4 {-half_width} 0\nv {distance + 4} {-half_width} 0\n"
        f"v {distance + 4} {half_width} 0\nv -4 {half_width} 0\ng plate\nf 1 2 3 4\n"
    )
    geometry = {"file": str(obj)}
    near, rate, h = _along_the_layer(geometry, [0.0, 0.0, -1.5], [[distance, 0.0, -1.5]], distance)
    deep, _, _ = _along_the_layer(
        geometry,
        [0.0, 0.0, -1.5],
        [[distance, 0.0, -1.5], [distance / 2, 0.0, -1.5 - 28 * h]],
        distance,
    )
    assert _band_db(near, deep, rate, band=(40.0, 100.0)) == pytest.approx(0.0, abs=0.05)


def test_spans_across_takes_the_farthest_listener_in_the_plane_of_the_other_axes() -> None:
    # By hand: across x the farthest is (0, 0, 12) at 12 m in the y-z plane, across y the
    # same, across z (3, 4, 0) at 5 m in the x-y plane.
    source = np.array([1.0, 1.0, 1.0])
    listeners = source + np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 12.0]])
    assert _spans_across(source, listeners) == pytest.approx([12.0, 12.0, 5.0])
    assert _spans_across(source, np.zeros((0, 3))) == pytest.approx([0.0, 0.0, 0.0])


class _Laid(Exception):
    """Where the grid with the matched layer starts, its cell and its shape."""


def _laid_grid(monkeypatch, geometry: dict, source, listener) -> tuple:
    """The origin, cell and shape of the 63 Hz grid with the matched layer round
    ``geometry``, from ``source`` to ``listener``, before any solver is made for it."""
    real = _native.WaveSolver

    def laid(*args):
        if args[13]:  # the matched layer's cells
            raise _Laid(np.asarray(args[5]), float(args[6]), np.array(args[7:10]))
        return real(*args)

    monkeypatch.setattr(_native, "WaveSolver", laid)
    room = Room.from_scene(
        Scene.model_validate(
            {
                "geometry": geometry,
                "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
                "default_material": "hard",
                "sources": [{"name": "s", "position": source}],
                "listeners": [{"name": "l", "position": listener}],
            }
        )
    )
    with pytest.raises(_Laid) as laid_out:
        wave_responses(
            TraceModel.from_room(room),
            np.array(source, dtype=float),
            np.array([listener], dtype=float),
            ["l"],
            max_band_hz=63,
            length_s=0.1,
            sample_rate=FS,
            speed_of_sound=room.speed_of_sound(),
        )
    return laid_out.value.args


def test_the_room_for_the_layer_keeps_the_grid_on_the_model(tmp_path: Path, monkeypatch) -> None:
    # Where the model reaches past the source, the listeners and their room, the grid
    # starts at the model's lowest corner, less two cells of margin, the layer and a
    # fraction of a cell off the model's planes; elsewhere whole cells further out, at
    # least the room past the source and the listeners; under an outdoor ground, at the
    # ground.
    margin = 2.0 + LAYER_CELLS + np.array([0.0137, 0.0219, 0.0291])
    distance = 60.0
    room = LAYER_ROOM * distance
    # Outdoors along a strip 3 m wide: nothing under the ground; along x the ground
    # reaches 4 m past the source and the listener (no room across x); across y whole
    # cells past the strip's edge, at least the room past the source and the listener;
    # above, the room over them.
    origin, h, shape = _laid_grid(
        monkeypatch,
        {"ground": {"extent": [[-4.0, -1.5], [distance + 4.0, 1.5]], "sky": 3.0}},
        [0.0, 0.0, 1.5],
        [distance, 0.0, 1.5],
    )
    model = np.array([-4.0, -1.5, 0.0]) - h * margin
    assert origin[[0, 2]] == pytest.approx(model[[0, 2]])
    cells = (model[1] - origin[1]) / h
    assert cells == pytest.approx(round(cells)) and cells >= 1
    assert origin[1] + h * margin[1] <= -room + 1e-9
    assert origin[2] + h * (shape[2] - 2 - LAYER_CELLS) >= 1.5 + room
    # Under a rigid plate, the model file's only face: below the source and the
    # listener 1.5 m under it the grid starts whole cells under where it did, at least
    # the room under them.
    obj = tmp_path / "plate.obj"
    obj.write_text(
        f"v -4 -20 0\nv {distance + 4} -20 0\nv {distance + 4} 20 0\nv -4 20 0\nf 1 2 3 4\n"
    )
    origin, h, _ = _laid_grid(
        monkeypatch, {"file": str(obj)}, [0.0, 0.0, -1.5], [distance, 0.0, -1.5]
    )
    before = -1.5 - 2.0 * h - h * margin[2]
    cells = (before - origin[2]) / h
    assert cells == pytest.approx(round(cells)) and cells >= 1
    assert origin[2] + h * margin[2] <= -1.5 - room + 1e-9
    assert origin[0] == pytest.approx(-4.0 - h * margin[0])


def _hall(path: Path) -> str:
    """A model file of a hall 40 x 20 x 4 m of single faces, open at its end x = 40 m:
    floor, roof, both long walls and the wall at x = 0."""
    path.write_text(
        "v 0 0 0\nv 40 0 0\nv 40 20 0\nv 0 20 0\nv 0 0 4\nv 40 0 4\nv 40 20 4\nv 0 20 4\n"
        "f 1 2 3 4\nf 5 6 7 8\nf 1 2 6 5\nf 4 3 7 8\nf 1 4 8 5\n"
    )
    return str(path)


def test_the_layer_gets_no_room_where_the_models_surfaces_close_the_side(
    tmp_path: Path, monkeypatch
) -> None:
    # In a hall open at one end, the source and a listener 33 m apart inside it: the
    # floor and the roof lie between them and the layer under and over the hall, so the
    # grid stays on the model there (5 m of room each way otherwise: 1.5 times the
    # cells). With the listener 4 m outside the open end, the air round it reaches the
    # layer both ways, and the grid leaves the room under and over the two.
    obj = _hall(tmp_path / "hall.obj")
    margin = 2.0 + LAYER_CELLS + np.array([0.0137, 0.0219, 0.0291])
    origin, h, shape = _laid_grid(monkeypatch, {"file": obj}, [3.0, 7.0, 1.5], [36.0, 9.0, 1.2])
    assert origin == pytest.approx(-h * margin)
    assert shape[2] == math.ceil((4.0 - origin[2]) / h) + 2 + LAYER_CELLS
    origin, h, shape = _laid_grid(monkeypatch, {"file": obj}, [3.0, 7.0, 1.5], [44.0, 9.0, 1.2])
    room = LAYER_ROOM * math.hypot(41.0, 2.0)
    assert origin[2] + h * margin[2] <= 1.2 - room + 1e-9
    assert origin[2] + h * (shape[2] - 2 - LAYER_CELLS) >= 1.5 + room


def test_open_sides_floods_the_air_of_every_point() -> None:
    # The grid round the hall of :func:`_hall`, as wave_responses lays it before any
    # room. From points inside, the air reaches no layer under or over the hall within
    # the box the room spans (along x and y that box ends inside the hall, where the air
    # reaches its edge).
    # A point outside the hall's closed end, whose air the source's flood does not reach
    # within the box, is flooded on its own, and reaches the layer under, over and
    # behind it. A point in no air (in a closed box smaller than a cell, standing in the
    # hall) is passed over: the sides are those without it.
    # Under a canopy 26 m over the hall the grid reaches far above the box: its top ends
    # short of the layer and counts as open (its room lies inside the grid), though the
    # roof keeps the air from it.
    hall = np.array(list(itertools.product((0.0, 40.0), (0.0, 20.0), (0.0, 4.0))))
    quads = [(0, 2, 6, 4), (1, 3, 7, 5), (0, 4, 5, 1), (2, 6, 7, 3), (0, 1, 3, 2)]
    walls = np.array([[a, b, c] for q in quads for a, b, c in (q[:3], (q[0], q[2], q[3]))])
    h = 0.4

    def sides(points, more_vertices=None, more_faces=None) -> np.ndarray:
        corners, faces = hall, walls
        if more_vertices is not None:
            corners = np.vstack([hall, more_vertices])
            faces = np.vstack([walls, np.asarray(more_faces) + len(hall)])
        bvh = _native.Bvh(corners, faces)
        points = np.array(points, dtype=float)
        low = np.minimum(corners.min(axis=0), points.min(axis=0) - 2 * h)
        high = np.maximum(corners.max(axis=0), points.max(axis=0) + 2 * h)
        origin = low - h * (2.0 + LAYER_CELLS + np.array([0.0137, 0.0219, 0.0291]))
        shape = np.ceil((high - origin) / h).astype(int) + 2 + LAYER_CELLS
        room = LAYER_ROOM * _spans_across(points[0], points[1:])
        return _open_sides(bvh, points, room, origin, h, shape, LAYER_CELLS)

    inside = sides([[3.0, 7.0, 1.5], [36.0, 9.0, 1.2]])
    assert inside.tolist() == [[True, True], [True, True], [False, False]]
    behind = sides([[3.0, 7.0, 1.5], [36.0, 9.0, 1.2], [-3.0, 9.0, 1.2]])
    assert behind.tolist() == [[True, True], [True, True], [True, True]]
    crate = box_mesh((0.1, 0.1, 0.1))
    shut_in = sides(
        [[3.0, 7.0, 1.5], [36.0, 9.0, 1.2], [20.05, 10.0, 2.0]],
        crate.vertices + [20.0, 9.95, 1.95],
        crate.faces,
    )
    assert shut_in.tolist() == inside.tolist()
    canopy = [[0.0, 0.0, 30.0], [40.0, 0.0, 30.0], [40.0, 20.0, 30.0], [0.0, 20.0, 30.0]]
    under = sides([[3.0, 7.0, 1.5], [36.0, 9.0, 1.2]], canopy, [[0, 1, 2], [0, 2, 3]])
    assert under.tolist() == [[True, True], [True, True], [False, True]]


def test_wave_solver_tells_the_sides_its_air_reaches() -> None:
    # A closed box on a grid that ends halfway along it: the air reaches the grid's edge
    # at +x alone. On a grid round the whole box: nowhere. Round a plate in the open,
    # with a matched layer: everywhere.
    mesh = box_mesh((4.0, 3.0, 3.0))
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    origin = -0.1 * (2.0 + np.array([0.0137, 0.0219, 0.0291]))

    def solver(size):
        shape = np.ceil((np.array(size) - origin) / 0.1).astype(int) + 2
        return _native.WaveSolver(
            bvh,
            np.zeros(12, np.int64),
            np.zeros(1),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            origin,
            0.1,
            *map(int, shape),
            np.array([1.0, 1.5, 1.5]),
            np.zeros((0, 3)),
        )

    assert solver((2.0, 3.0, 3.0)).open_sides == (False, True, False, False, False, False)
    assert solver((4.0, 3.0, 3.0)).open_sides == (False,) * 6
    _, plate, _ = _plate(0.2, LAYER_CELLS, LAYER_CELLS, 0.001)
    assert plate.open_sides == (True,) * 6


def test_a_source_among_the_layers_cells_is_refused() -> None:
    # A grid of 36 cells a side with 8 of matched layer inside each edge, round a plate
    # far below. The source 9.25 cells from the grid's start along x: of the eight cells
    # round it, the four of the layer's last cells are the layer's. 9.75 cells from it:
    # the two cells round it along x, past the layer, and four a side along y and z.
    plate = np.array([[-50.0, -50, -5], [50, -50, -5], [50, 50, -5], [-50, 50, -5]])
    bvh = _native.Bvh(plate, np.array([[0, 1, 2], [0, 2, 3]], np.int64))
    cell, origin = 0.1, np.full(3, 0.0013)

    def solver(x_cells: float):
        return _native.WaveSolver(
            bvh,
            np.zeros(2, np.int64),
            np.zeros(1),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            origin,
            cell,
            36,
            36,
            36,
            origin + cell * np.array([x_cells, 18.3, 18.2]),
            np.zeros((0, 3)),
            layer=8,
        )

    with pytest.raises(ValueError, match="the source is in the matched layer"):
        solver(9.25)
    layout = solver(9.75).layout()
    fed = np.asarray(layout["special"])[np.flatnonzero(np.asarray(layout["source"]))]
    assert len(fed) == 32 and sorted(set(fed % 36)) == [9, 10]


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


# -- a speed of sound per cell (outdoors, in wind and temperature) ------------------------


@STENCILS
def test_one_speed_of_sound_everywhere_changes_nothing(stencil: int) -> None:
    # Every cell at the grid's speed: the run is the one without a speed per cell, to the
    # bit, walls with frequency-dependent admittances included.
    fit = fit_admittance([0.1, 0.3, 0.5, 0.6], 8, C / (_courant(stencil) * 0.1))

    def run(speed):
        solver, rate = _solver(
            (2.0, 1.6, 1.3),
            lambda mesh: [0.05] * len(mesh.faces),
            [0.5, 0.6, 0.7],
            [[1.5, 1.1, 0.4], [0.3, 0.3, 0.3]],
            branches=[(0, fit)],
            stencil=stencil,
            speed=speed,
        )
        return solver.run(_drive(rate, 0.1, 800, stencil))

    assert np.array_equal(run(lambda centres: np.ones(len(centres))), run(None))


def test_a_slower_speed_of_sound_arrives_later_as_loud() -> None:
    # 0.8 times the grid's speed in every cell: the sound arrives 1.25 times later, as
    # loud (the source's strength follows its cell's speed: 1 / r either way), and its
    # velocity, from the Courant number at the listener, keeps its size next to the
    # pressure. 2 m away in a large room, its walls' sound well after the direct sound's.
    def run(ratio):
        solver, rate = _solver(
            (12.0, 12.0, 12.0),
            lambda mesh: [1.0] * len(mesh.faces),
            [5.0, 6.0, 6.0],
            _probe_points(np.array([7.0, 6.0, 6.0]), 0.1),
            speed=None if ratio == 1.0 else lambda centres: np.full(len(centres), ratio),
        )
        field = _first_order(
            solver.run(_drive(rate, 0.1, int(0.03 * rate))), [True] * 13, "l", COURANT * ratio
        )
        signal, delay = source_signal(rate, 0.1)
        arrival = 2.0 / (C * ratio) * rate + delay
        n, half = np.arange(len(field)), 0.012 * rate
        window = np.where(
            abs(n - arrival) < half, 0.5 + 0.5 * np.cos(np.pi * (n - arrival) / half), 0
        )
        n_fft = 1 << 13
        f = np.fft.rfftfreq(n_fft, 1.0 / rate)
        impulse = np.fft.rfft(signal / (4 * math.pi * COURANT**2 / 0.1), n_fft)
        pressure, velocity = (
            np.abs(np.fft.rfft(window * field[:, k], n_fft) / impulse)[(f > 80) & (f < 300)]
            for k in (0, 1)
        )
        # The peak, between the samples (a parabola through the three round it): the
        # sound arrives 36 and 46 steps after it leaves, and a whole step is 2 to 3 % of
        # that.
        k = int(np.argmax(np.abs(field[:, 0])))
        before, at, after = np.abs(field[k - 1 : k + 2, 0])
        peak = k + 0.5 * (before - after) / (before - 2.0 * at + after)
        return (peak - delay) / rate, 2.0 * pressure.mean(), velocity.mean() / pressure.mean()

    fast, slow = run(1.0), run(0.8)
    assert slow[0] / fast[0] == pytest.approx(1.25, rel=0.01)
    assert fast[1] == pytest.approx(1.0, abs=0.05) and slow[1] == pytest.approx(fast[1], rel=0.03)
    assert slow[2] == pytest.approx(fast[2], rel=0.03)


def test_matched_layer_stays_stable_with_a_speed_of_sound_per_cell() -> None:
    ones = lambda centres: np.ones(len(centres))  # noqa: E731
    same, _, _ = _plate(0.5, LAYER_CELLS, LAYER_CELLS, 0.5, speed=ones)
    assert np.array_equal(same, _plate(0.5, LAYER_CELLS, LAYER_CELLS, 0.5)[0])

    # Half the grid's speed at the bottom and all of it from 2 m up, and a fifth slower
    # on one side: still no growth over 9,500 steps.
    def speed(centres):
        rising = np.clip(0.5 + 0.25 * (centres[:, 2] + 3.0) / 2.5, 0.5, 1.0)
        return rising * np.where(centres[:, 0] < 4.0, 0.8, 1.0)

    pressure, _, rate = _plate(0.5, LAYER_CELLS, LAYER_CELLS, 8.0, speed=speed)
    tenth = pressure.shape[1] // 10
    levels = [np.mean(pressure[:, k * tenth : (k + 1) * tenth] ** 2) for k in range(10)]
    assert max(levels[6:]) < 1e-11 * levels[0]


def _warming(table) -> Room:
    """Rigid ground, the source and the listener 30 m up and 60 m apart, in still air
    whose temperature changes with height as ``table`` (rows of height, °C, wind)."""
    return Room.from_scene(
        Scene.model_validate(
            {
                "geometry": {"ground": {"extent": [[-8, -8], [68, 8]], "sky": 60}},
                "materials": {"hard": {"absorption": [0.01] * 8, "scattering": 0.0}},
                "assignments": {"ground": "hard"},
                "atmosphere": {"table": table},
                "sources": [{"name": "s", "position": [0, 0, 30]}],
                "listeners": [{"name": "l", "position": [60, 0, 30]}],
            }
        )
    )


class _Stopped(Exception):
    """What the source's signal was asked for: the rate, the cell, the Courant number and
    the speed of sound at the source over the grid's."""


def test_the_source_takes_its_own_speed_of_sound_and_the_sides_are_found_once(
    monkeypatch,
) -> None:
    # Air warming with height: the grid is laid out more than once, as its cells follow
    # the slowest sound and its time step the fastest, but the sides whose layer needs
    # room are found once. The source's signal takes the speed of sound at the source,
    # 30 m up, over the grid's, the fastest: the speed's square grows evenly with height,
    # so the root of the mean of the slowest's and the fastest's squares, over the
    # fastest.
    room = _warming([[0, 10, 0], [60, 130, 0]])
    calls = {"grids": 0, "sides": 0}
    grid_of, sides_of = wave_stage.band_grid, wave_stage._open_sides

    def grid(*args, **more):
        calls["grids"] += 1
        return grid_of(*args, **more)

    def sides(*args):
        calls["sides"] += 1
        return sides_of(*args)

    def signal(*args):
        raise _Stopped(*args)

    monkeypatch.setattr(wave_stage, "band_grid", grid)
    monkeypatch.setattr(wave_stage, "_open_sides", sides)
    monkeypatch.setattr(wave_stage, "source_signal", signal)
    with pytest.raises(_Stopped) as stopped:
        wave_responses(
            TraceModel.from_room(room),
            np.array([0.0, 0.0, 30.0]),
            [[60.0, 0.0, 30.0]],
            ["l"],
            max_band_hz=63,
            length_s=0.25,
            sample_rate=FS,
            speed_of_sound=room.speed_of_sound(),
        )
    rate, cell, courant, speed = stopped.value.args
    assert calls["grids"] > 1 and calls["sides"] == 1
    assert courant == COURANT and courant * cell * rate == pytest.approx(402.5, rel=1e-3)
    assert speed == pytest.approx(math.sqrt((337.3**2 + 402.5**2) / 2.0) / 402.5, rel=1e-3)


def test_a_speed_of_sound_that_grows_with_height_bends_the_sound_as_the_rays_do() -> None:
    # Air warming from 10 to 130 °C over 60 m (the speed of sound from 337 to 402 m/s)
    # bends the sound down, and cooling as much bends it up: the direct sound arrives
    # from 5 degrees above or below, as the eigenrays do (the 7-point stencil, slower
    # along the grid's axes, bends it 4.6 % more as the air warms and 0.9 % more as it
    # cools).
    elevations = []
    for table in ([[0, 10, 0], [60, 130, 0]], [[0, 130, 0], [60, 10, 0]]):
        room = _warming(table)
        atmosphere = room.scene.atmosphere
        (wave,), info = wave_responses(
            TraceModel.from_room(room),
            np.array([0.0, 0.0, 30.0]),
            [[60.0, 0.0, 30.0]],
            ["l"],
            max_band_hz=63,
            length_s=0.25,
            sample_rate=FS,
            speed_of_sound=room.speed_of_sound(),
        )
        assert info["wave_speed_min_m_s"] == pytest.approx(337.3, abs=0.1)
        assert info["wave_speed_max_m_s"] == pytest.approx(402.5, abs=0.1)
        assert "wave_gpu" not in info
        # The slowest sound, 8 cells per wavelength at the top of the 63 Hz band, stepped
        # as the fastest sound's time step lets it: slower than in still air.
        slowest = info["wave_speed_min_m_s"]
        per_wavelength = slowest / (band_top(0) * wave.cell)
        assert 8.0 <= per_wavelength < 8.5
        ratio = slowest / (COURANT * wave.cell * wave.rate)
        assert ratio == pytest.approx(337.3 / 402.5, rel=1e-3)
        assert info["wave_slow_percent"] == pytest.approx(100 * slow_by(7, per_wavelength, ratio))
        assert info["wave_slow_percent"] > 100 * slow_by(7, per_wavelength) + 0.2
        rays = Rays(Profile.of(atmosphere, room.scene.air.temperature_c), 30.0)
        eigenrays = rays.eigenrays(30.0, 60.0, 0)
        assert len(eigenrays) == 1
        # The direct sound, 30 to 130 Hz, round its time along the eigenray (the ground's
        # reflection comes 50 ms after it).
        n = len(wave.signals)
        f = np.fft.rfftfreq(4 * n, 1.0 / wave.rate)
        gain = np.clip((f - 30) / 15, 0, 1) * np.clip((130 - f) / 30, 0, 1)
        field = np.fft.irfft(
            np.fft.rfft(wave.signals, 4 * n, axis=0) * gain[:, None], 4 * n, axis=0
        )
        # In samples of the run, which starts lead_s before emission.
        arrival = int(round(eigenrays.time[0] * wave.rate)) + round(wave.lead_s * wave.rate)
        w, x, y, z = field[arrival - int(0.008 * wave.rate) : arrival + int(0.008 * wave.rate)].T
        intensity = np.array([w @ x, w @ y, w @ z])  # towards where the sound comes from
        elevation = math.degrees(math.atan2(intensity[2], math.hypot(*intensity[:2])))
        elevations.append(elevation)
        assert elevation == pytest.approx(-math.degrees(eigenrays.arrival[0]), rel=0.12)
    assert elevations[0] > 4.5 and elevations[1] < -4.5


# -- moving air (simulation.wave.wind: moving) --------------------------------------------


@STENCILS
def test_air_that_moves_nowhere_changes_nothing(stencil: int) -> None:
    # Moving air whose every cell is still: the run is the still air's, to the bit, walls
    # with frequency-dependent admittances included.
    fit = fit_admittance([0.1, 0.3, 0.5, 0.6], 8, C / (_courant(stencil) * 0.1))

    def run(flow):
        solver, rate = _solver(
            (2.0, 1.6, 1.3),
            lambda mesh: [0.05] * len(mesh.faces),
            [0.5, 0.6, 0.7],
            [[1.5, 1.1, 0.4], [0.3, 0.3, 0.3]],
            branches=[(0, fit)],
            stencil=stencil,
            flow=flow,
        )
        assert solver.flowing == (flow is not None)
        return solver.run(_drive(rate, 0.1, 800, stencil))

    still = lambda centres: (np.zeros(len(centres)), np.zeros(len(centres)))  # noqa: E731
    assert np.array_equal(run(still), run(None))


def _band_gain(n: int, rate: float, band) -> np.ndarray:
    """A pass band from band[0] to band[1] (Hz), its edges ramps 10 and 30 Hz wide, at the
    frequencies of an rfft of n points."""
    f = np.fft.rfftfreq(n, 1.0 / rate)
    return np.clip((f - band[0]) / 10, 0, 1) * np.clip((band[1] - f) / 30, 0, 1)


def _band_lag(a: np.ndarray, b: np.ndarray, rate: float, band=(40.0, 150.0)) -> float:
    """How much later a is than b (s) in ``band``, from the peak of their correlation
    interpolated 64 times finer."""
    n = 1 << int(math.ceil(math.log2(2 * len(a))))
    gain = _band_gain(n, rate, band)
    spectrum = np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)) * gain**2
    xc = np.fft.irfft(spectrum, 64 * n)
    k = int(np.argmax(xc))
    return (k if k < 32 * n else k - 64 * n) / (64.0 * rate)


def _band_db(a: np.ndarray, b: np.ndarray, rate: float, band=(40.0, 150.0)) -> float:
    """How much louder a is than b (dB) in ``band``."""
    n = 1 << int(math.ceil(math.log2(2 * len(a))))
    gain = _band_gain(n, rate, band)
    energy = [np.sum(np.abs(np.fft.rfft(x, n) * gain) ** 2) for x in (a, b)]
    return 10.0 * math.log10(energy[0] / energy[1])


def test_moving_air_carries_the_sound_downwind() -> None:
    # Air moving along +x at a tenth of the speed of sound: 6 m downwind the sound comes
    # 6 / (c + v) after it leaves, 6 m upwind 6 / (c - v), as loud as in still air (on
    # the wind's axis the convected Green's function is 1 / r either way); across the
    # wind it comes 6 / (c sqrt(1 - M^2)) after, louder by 1 / sqrt(1 - M^2). Against
    # the still air's run, whose grid disperses alike, in the band the grid is for.
    mach = 0.1

    def run(flow):
        solver, rate = _solver(
            (20.0, 20.0, 20.0),
            lambda mesh: [1.0] * len(mesh.faces),
            [10.0, 10.0, 10.0],
            [[16.0, 10.0, 10.0], [4.0, 10.0, 10.0], [10.0, 16.0, 10.0]],
            cell=0.2,
            flow=flow,
        )
        return solver.run(_drive(rate, 0.2, int(0.04 * rate))), rate

    still, rate = run(None)
    shift = mach * COURANT  # cells per step
    moving, _ = run(lambda centres: (np.full(len(centres), shift), np.zeros(len(centres))))
    beta = math.sqrt(1.0 - mach**2)
    expected = [
        6.0 / (C * (1 + mach)) - 6.0 / C,
        6.0 / (C * (1 - mach)) - 6.0 / C,
        6.0 / (C * beta) - 6.0 / C,
    ]
    louder = [0.0, 0.0, -20.0 * math.log10(beta)]
    for k, want in enumerate(expected):
        assert _band_lag(moving[k], still[k], rate) == pytest.approx(want, abs=1.5e-5)
        assert _band_db(moving[k], still[k], rate) == pytest.approx(louder[k], abs=0.05)


def test_moving_air_stays_stable_round_walls_and_in_the_matched_layer() -> None:
    # A wind that grows with height and turns, blowing into a block's face and over a
    # plate in the open: no growth over 9,500 steps, the sound gone into the matched
    # layer down to single precision's rounding.
    def flow(centres):
        rising = np.clip((centres[:, 2] + 0.5) / 3.0, 0.0, 1.0)
        return 0.06 * rising, 0.03 * rising * np.sign(centres[:, 1] - 3.0)

    pressure, solver, _ = _plate(0.5, LAYER_CELLS, LAYER_CELLS, 8.0, flow=flow, block=True)
    assert solver.flowing and solver.layer_cells > 0
    tenth = pressure.shape[1] // 10
    levels = [np.mean(pressure[:, k * tenth : (k + 1) * tenth] ** 2) for k in range(10)]
    assert max(levels[6:]) < 1e-11 * levels[0]


def test_wave_solver_checks_the_air_s_motion() -> None:
    mesh = box_mesh((2.0, 2.0, 2.0))
    bvh = _native.Bvh(mesh.vertices, mesh.faces)
    shape = (26, 26, 26)
    cells = shape[0] * shape[1] * shape[2]

    def make(**flow):
        return _native.WaveSolver(
            bvh,
            np.zeros(len(mesh.faces), np.int64),
            np.zeros(1),
            np.zeros((0, 3)),
            np.zeros(0, np.int64),
            np.array([-0.2, -0.2, -0.2]),
            0.1,
            *shape,
            np.array([1.0, 1.0, 1.0]),
            np.array([[1.2, 1.1, 0.9]]),
            **flow,
        )

    still = np.zeros(cells, np.float32)
    assert make(flow_x=still, flow_y=still).flowing and not make().flowing
    with pytest.raises(ValueError, match="go together"):
        make(flow_x=still)
    with pytest.raises(ValueError, match="one shift of the air per cell"):
        make(flow_x=still[:-1], flow_y=still[:-1])
    for fast in (0.5, -0.5, np.nan):
        flow = still.copy()
        flow[13 + shape[0] * (13 + shape[1] * 13)] = fast  # the room's middle
        with pytest.raises(ValueError, match="less than half a cell per step"):
            make(flow_x=flow, flow_y=still)
    # Outside the air (here the corner, out of the room) the air's motion is not read.
    flow = still.copy()
    flow[0] = 0.9
    assert make(flow_x=still, flow_y=flow).flowing


def _windy_ground(wind: float, model: str, temperature=None) -> Room:
    """Rigid ground, the source 1.5 m up and listeners 6 m downwind and upwind of it, in
    a uniform wind ``wind`` (m/s) along +x taken as ``model``; ``temperature``, a table
    of heights and °C, for still air warming with height."""
    data = {
        "geometry": {"ground": {"extent": [[-10, -6], [10, 6]], "sky": 5}},
        "materials": {"hard": {"absorption": [0.0] * 8, "scattering": 0.0}},
        "default_material": "hard",
        "sources": [{"name": "s", "position": [0, 0, 1.5]}],
        "listeners": [
            {"name": "down", "position": [6, 0, 1.5]},
            {"name": "up", "position": [-6, 0, 1.5]},
        ],
        "simulation": {"wave": {"enabled": True, "max_band_hz": 63, "wind": model}},
    }
    if wind:
        data["atmosphere"] = {
            "wind": {"speed_m_s": wind, "direction_deg": 0.0, "profile": "constant"}
        }
    if temperature is not None:
        data["atmosphere"] = {"table": [[z, t, 0.0] for z, t in temperature]}
    return Room.from_scene(Scene.model_validate(data))


def _windy_run(
    room: Room, wind: str, per_wavelength: float = 8.0, device="cpu", scheme: str = "auto"
):
    said = []
    responses, info = wave_responses(
        TraceModel.from_room(room),
        np.array([0.0, 0.0, 1.5]),
        np.array([listener.position for listener in room.scene.listeners], dtype=float),
        [listener.name for listener in room.scene.listeners],
        max_band_hz=63,
        length_s=0.06,
        sample_rate=FS,
        speed_of_sound=room.speed_of_sound(),
        points_per_wavelength=per_wavelength,
        wind=wind,
        device=device,
        scheme=scheme,
        say=said.append,
        label="s",
    )
    return responses, info, said


def test_still_air_takes_no_moving_air() -> None:
    # Without wind (or with temperature alone) the air does not move: the moving model's
    # run is the effective one's, to the bit.
    for room_of in (
        lambda model: _windy_ground(0.0, model),
        lambda model: _windy_ground(0.0, model, temperature=[[0, 25], [20, 15]]),
    ):
        effective, info, _ = _windy_run(room_of("effective"), "effective")
        moving, moving_info, said = _windy_run(room_of("moving"), "moving")
        assert "wave_moving_air" not in moving_info and "the air moving" not in said[0]
        for a, b in zip(effective, moving, strict=True):
            assert a.rate == b.rate and np.array_equal(a.signals, b.signals)


def test_wave_responses_carry_the_sound_with_the_wind() -> None:
    # A uniform wind of 15 m/s along +x over rigid ground: against the still air's run
    # on the same grid (as fine as sound against the wind needs), the moving air's sound
    # arrives 6 / (c + v) - 6 / c sooner downwind and 6 / (c - v) - 6 / c later upwind.
    still_room = _windy_ground(0.0, "effective")
    c = still_room.speed_of_sound()
    still, _, _ = _windy_run(still_room, "effective", 8.0 * c / (c - 15.0))
    moving, info, said = _windy_run(_windy_ground(15.0, "moving"), "moving")
    assert info["wave_moving_air"] == 1.0
    assert info["wave_cells"] > 0 and "the air moving" in said[0]
    assert info["wave_speed_min_m_s"] == pytest.approx(c - 15.0, abs=0.01)
    assert info["wave_speed_max_m_s"] == pytest.approx(c + 15.0, abs=0.01)
    # The still air's step, in the air's frame: sound against the wind has 8 cells per
    # wavelength at the top of the 63 Hz band, and the still air the grid's time step.
    per_wavelength = (c - 15.0) / (band_top(0) * moving[0].cell)
    assert 8.0 <= per_wavelength < 8.5
    assert info["wave_slow_percent"] == pytest.approx(100 * slow_by(7, per_wavelength))
    for a, b, want in zip(
        moving, still, (6.0 / (c + 15.0) - 6.0 / c, 6.0 / (c - 15.0) - 6.0 / c), strict=True
    ):
        assert a.rate == b.rate and a.cell == b.cell
        lag = _band_lag(a.signals[:, 0], b.signals[:, 0], a.rate, band=(40.0, 90.0))
        assert lag == pytest.approx(want, abs=2.5e-5)
        louder = _band_db(a.signals[:, 0], b.signals[:, 0], a.rate, band=(40.0, 90.0))
        assert louder == pytest.approx(0.0, abs=0.1)


def test_moving_air_stays_on_the_cpu(monkeypatch) -> None:
    # The GPU steps still air only: asked for it (here as if there were one), moving air
    # says so and runs on the CPU, as it does there.
    from acoustic_spaces.engine import wave as wave_module

    room = _windy_ground(15.0, "moving")
    cpu, _, _ = _windy_run(room, "moving")
    monkeypatch.setattr(wave_module, "use_gpu", lambda device, steps=None: True)
    asked, info, said = _windy_run(room, "moving", device="gpu")
    assert "wave_gpu" not in info and info["wave_moving_air"] == 1.0
    assert "the air moving (on the CPU: the GPU steps still air)" in said[0]
    for a, b in zip(cpu, asked, strict=True):
        assert np.array_equal(a.signals, b.signals)


@pytest.mark.parametrize("wind", ["effective", "moving"])
def test_a_grid_too_big_is_refused_before_its_weather_is_made(monkeypatch, wind) -> None:
    from acoustic_spaces.engine import wave as wave_module

    made = []
    for name in ("ratios", "moving_fields"):
        monkeypatch.setattr(wave_module._Weather, name, lambda *_, name=name: made.append(name))
    monkeypatch.setattr(wave_module, "MAX_CELLS", 1_000)
    # Outdoors, the grid with the matched layer, the only one made there.
    with pytest.raises(ValueError, match=r"cells with the matched layer round this open model"):
        _windy_run(_windy_ground(15.0, wind), wind)
    assert made == []


def test_the_wave_stage_takes_its_wind_model() -> None:
    assert WaveStage().wind == "effective"
    assert _windy_ground(15.0, "moving").scene.simulation.wave.wind == "moving"
    with pytest.raises(ValueError, match="wind"):
        WaveStage.model_validate({"wind": "sideways"})
