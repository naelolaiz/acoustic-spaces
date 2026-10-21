"""Grounds with an impedance, the spherical-wave reflection off them, ground zones and
the Fresnel-zone weighting of reflections near where two grounds meet."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.calibration import calibrate
from acoustic_spaces.cli import main
from acoustic_spaces.engine.coherent import coherent_energy, total_energy
from acoustic_spaces.engine.impedance import (
    Admittance,
    band_weights,
    fit_admittance,
    fit_impedance,
    random_incidence_absorption,
)
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.geometry.outdoor import FRESNEL_PATH_DIFFERENCE, Layout, _fresnel_ellipses
from acoustic_spaces.materials.bands import band_centres
from acoustic_spaces.materials.ground import (
    characteristic,
    plane_reflection,
    spherical_reflection,
    surface_impedance,
)
from acoustic_spaces.scene import Impedance, Material

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox" / "scene.yaml"
C = 343.0


def exact_reflection(beta: complex, k: float, r: float, zs: float, zr: float) -> complex:
    """The reflected pressure over the image source's off a locally reacting plane of
    admittance ``beta`` (e^(jwt)), from the exact (Sommerfeld) integral."""
    b = np.conj(beta)
    r2 = math.hypot(r, zs + zr)
    q = np.linspace(0.0, 40.0 / (k * (b.real + (zs + zr) / r2)), 400_001)
    rq = np.sqrt(r * r + (zs + zr + 1j * q) ** 2)
    rq = np.where(rq.real < 0.0, -rq, rq)
    integral = np.trapezoid(np.exp(-k * b * q) * np.exp(1j * k * rq) / rq, q)
    pressure = np.exp(1j * k * r2) / r2 - 2.0 * k * b * integral
    return complex(np.conj(pressure * r2 * np.exp(-1j * k * r2)))


def ground_scene(ground, *, zones=None, road=None, coherent=None, extent=60, **positions):
    """A source and a listener over a ground and nothing else."""
    early = {"method": "image_sources", "max_order": 1, "diffraction": "none"}
    if coherent is not None:
        early["coherent_max_band_hz"] = coherent
    materials = {"field": ground}
    assignments = {}
    if road is not None:
        materials["tar"] = road
        assignments["road"] = "tar"
    data = {
        "geometry": {"ground": {"extent": [[-extent, -extent], [extent, extent]], "sky": 30}},
        "materials": materials,
        "default_material": "field",
        "assignments": assignments,
        "sources": [{"name": "s", "position": list(positions.get("source", (0, 0, 1.0)))}],
        "listeners": [{"name": "l", "position": list(positions.get("listener", (20, 0, 1.5)))}],
        "simulation": {"ir_length_s": 0.2, "seed": 1, "early": early, "late": {"rays": 200}},
    }
    if zones is not None:
        data["geometry"]["ground"]["zones"] = zones
    return Scene.model_validate(data)


def early_level(scene) -> np.ndarray:
    """The early part's energy over the direct sound's, per band (dB)."""
    response = simulate(Room.from_scene(scene))[("s", "l")]
    direct = response.early.energy[response.early.order == 0][0]
    return 10.0 * np.log10(total_energy(response.early) / direct)


# -- impedance models ---------------------------------------------------------------


def test_impedance_models() -> None:
    # At f = sigma (X = 1) the models are their constants.
    z, k = characteristic(np.array([200.0]), 200.0, "miki")
    assert z[0] == pytest.approx(1 + 5.50 - 8.43j) and k[0] == pytest.approx(1 + 7.81 - 11.41j)
    z, k = characteristic(np.array([200.0]), 200.0, "delany_bazley")
    assert z[0] == pytest.approx(1 + 9.08 - 11.9j) and k[0] == pytest.approx(1 + 10.8 - 10.3j)
    # The two-parameter model: 0.436 (1 - j) sqrt(sigma / f) - 19.48 j alpha / f.
    z = surface_impedance(np.array([100.0]), 100.0, "two_parameter", porosity_rate=20.0)
    assert z[0] == pytest.approx(0.436 * (1 - 1j) * math.sqrt(1000.0) - 19.48j * 0.2)
    # A layer: Zc coth(j kc d), Zc itself when deep, stiffer (more reactive) when thin.
    f = np.array([100.0, 1000.0])
    deep = surface_impedance(f, 12.5)
    assert surface_impedance(f, 12.5, thickness=50.0) == pytest.approx(deep, rel=1e-9)
    thin = surface_impedance(f, 12.5, thickness=0.02)
    assert np.all(-thin.imag > -deep.imag)
    with pytest.raises(ValueError, match="no thickness"):
        surface_impedance(f, 100.0, "two_parameter", thickness=0.1)
    # Passive (Re Z > 0) and harder the more resistive.
    for sigma in (12.5, 200.0, 20000.0):
        assert np.all(surface_impedance(np.geomspace(20, 20000, 50), sigma).real > 0)
    assert abs(surface_impedance(f, 20000.0)[0]) > abs(surface_impedance(f, 200.0)[0])


def test_spherical_reflection() -> None:
    # A hard ground reflects the spherical wave whole, at every angle.
    q = spherical_reflection(0.0, np.array([0.0, 0.3, 1.0]), 10.0, 20.0)
    assert np.allclose(q, 1.0)
    beta = 1.0 / surface_impedance(np.array(500.0), 200.0)
    # Far from grazing and far away, Q is the plane wave's R.
    q = spherical_reflection(beta, 0.9, 2 * math.pi * 500 / C, 500.0)
    assert q == pytest.approx(plane_reflection(beta, 0.9), abs=1e-3)
    # At grazing incidence R is -1, but the ground wave keeps Q from cancelling.
    assert plane_reflection(beta, 0.0) == pytest.approx(-1.0)
    assert abs(spherical_reflection(beta, 0.0, 2 * math.pi * 500 / C, 20.0) + 1.0) > 0.3
    # A passive surface's plane wave reflects at most what meets it.
    cos = np.linspace(0, 1, 11)[:, None]
    betas = 1.0 / surface_impedance(np.geomspace(20, 20000, 30), 31.5)
    assert np.all(np.abs(plane_reflection(betas[None], cos)) <= 1.0 + 1e-12)


@pytest.mark.parametrize(("sigma", "thickness"), [(200.0, None), (12.5, 0.1), (2000.0, None)])
def test_spherical_reflection_against_the_exact_solution(sigma, thickness) -> None:
    r, zs, zr = 20.0, 1.0, 1.5
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    for f in (125.0, 500.0, 2000.0):
        k = 2 * math.pi * f / C
        beta = complex(1.0 / surface_impedance(np.array(f), sigma, thickness=thickness))
        q = complex(spherical_reflection(beta, (zs + zr) / r2, k, r2))
        exact = exact_reflection(beta, k, r, zs, zr)
        level = [abs(np.exp(-1j * k * r1) / r1 + x * np.exp(-1j * k * r2) / r2) for x in (q, exact)]
        assert 20 * math.log10(level[0] / level[1]) == pytest.approx(0.0, abs=0.4)


# -- extended reaction ------------------------------------------------------------------


def test_extended_reaction_changes_the_impedance_with_the_angle() -> None:
    f = np.array([125.0, 500.0, 2000.0])
    for sigma, thickness in ((12.5, None), (12.5, 0.1), (200.0, None)):
        local = surface_impedance(f, sigma, thickness=thickness)
        # At normal incidence the extended reaction is the local one.
        normal = surface_impedance(f, sigma, thickness=thickness, cosine=1.0)
        np.testing.assert_allclose(normal, local, rtol=1e-12)
        # Towards grazing incidence it departs, by Zc n / sqrt(n² - sin²), and stays
        # passive.
        grazing = surface_impedance(f, sigma, thickness=thickness, cosine=0.0)
        z, n = characteristic(f, sigma)
        expected = z * n / np.sqrt(n * n - 1.0)
        if thickness is not None:
            expected = expected / np.tanh(1j * 2 * np.pi * f / C * np.sqrt(n * n - 1.0) * thickness)
        np.testing.assert_allclose(grazing, expected, rtol=1e-12)
        assert np.all(grazing.real > 0.0)
    # Soft snow departs by a few per cent at grazing incidence, grass hardly at all.
    for sigma, least, most in ((12.5, 0.02, 0.5), (200.0, 0.0, 0.05)):
        local = surface_impedance(f, sigma)
        grazing = surface_impedance(f, sigma, cosine=0.0)
        change = np.abs(grazing / local - 1.0)
        assert least <= change.max() <= most, (sigma, change)
    # The cosine broadcasts with the frequencies.
    many = surface_impedance(f[None, :], 12.5, cosine=np.array([[0.0], [0.5], [1.0]]))
    assert many.shape == (3, 3) and np.allclose(many[2], surface_impedance(f, 12.5))
    with pytest.raises(ValueError, match="reacts locally"):
        surface_impedance(f, 100.0, "two_parameter", porosity_rate=20.0, cosine=0.5)


def test_extended_reaction_against_the_exact_solution() -> None:
    """The exact field over a ground of extended reaction (the fast field program with
    the plane wave's reflection at every angle) against Q with the impedance at the
    reflection's angle, where Q with the locally reacting impedance is off."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))
    import ffp
    from ground_reflection import extended_impedance

    still = lambda z: np.full(np.shape(z), C)  # noqa: E731
    r, zs, zr = 20.0, 1.0, 1.5
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    cosine = (zs + zr) / r2
    worst = {"at_angle": 0.0, "local": 0.0}
    for f in (78.7, 99.2, 1587.4):
        k = 2 * math.pi * f / C
        exact = ffp.pressure(f, still, zs, [r], zr, extended_impedance(f, 12.5, 0.1), top=35.0)[0]
        level = 20 * math.log10(abs(exact) * r1)
        for key, cos in (("at_angle", cosine), ("local", None)):
            z = complex(surface_impedance(np.array(f), 12.5, thickness=0.1, cosine=cos))
            q = complex(spherical_reflection(1.0 / z, cosine, k, r2))
            got = 20 * math.log10(
                abs(np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2) * r1
            )
            worst[key] = max(worst[key], abs(got - level))
    assert worst["at_angle"] < 0.2 and worst["local"] > 0.35, worst


def test_a_ground_of_extended_reaction_in_a_scene() -> None:
    snow = {"impedance": {"preset": "snow", "thickness": 0.1}}
    extended = {"impedance": {"preset": "snow", "thickness": 0.1, "reaction": "extended"}}
    explicit = {"impedance": {"preset": "snow", "thickness": 0.1, "reaction": "local"}}
    far = {"source": (0, 0, 1.0), "listener": (100, 0, 1.5)}
    plain, same, angled = (
        early_level(ground_scene(g, extent=120, **far)) for g in (snow, explicit, extended)
    )
    # Local by default, written as given; the extended reaction reflects differently.
    np.testing.assert_array_equal(plain, same)
    assert np.max(np.abs(angled - plain)) > 0.5
    scene = ground_scene(extended, extent=120, **far)
    assert scene.to_data()["materials"]["field"]["impedance"]["reaction"] == "extended"
    assert (
        "reaction"
        not in ground_scene(snow, extent=120, **far).to_data()["materials"]["field"]["impedance"]
    )
    assert "extended reaction" in scene.materials["field"].impedance.describe()
    # The trace model holds the admittance at normal and at grazing incidence apart.
    model = TraceModel.from_room(Room.from_scene(scene))
    assert model.impedance_extended.tolist() == [True]
    assert not np.allclose(model.grazing_admittance, model.admittance)
    local_model = TraceModel.from_room(Room.from_scene(ground_scene(snow, extent=120, **far)))
    assert local_model.impedance_extended.tolist() == [False]
    assert np.array_equal(local_model.grazing_admittance, local_model.admittance)
    with pytest.raises(ValidationError, match="reacts locally"):
        Material.model_validate(
            {
                "impedance": {
                    "flow_resistivity": 100,
                    "model": "two_parameter",
                    "porosity_rate": 20,
                    "reaction": "extended",
                }
            }
        )


# -- materials --------------------------------------------------------------------


def test_a_material_with_an_impedance() -> None:
    lawn = Material.model_validate({"impedance": "grass", "scattering": 0.3})
    assert lawn.impedance == Impedance(preset="grass") and lawn.impedance.resistivity == 200.0
    absorption = lawn.coefficients("octave")[0]
    assert len(lawn.absorption) == 24 and np.all(np.diff(absorption) > 0)
    assert absorption[0] == pytest.approx(0.18, abs=0.02)
    assert absorption[-1] == pytest.approx(0.89, abs=0.02)
    assert lawn.coefficients("octave")[1] == pytest.approx([0.3] * 8)
    # Absorption given with it is replaced; harder grounds absorb less.
    given = Material.model_validate({"impedance": "grass", "absorption": [0.5] * 8})
    assert given.absorption == lawn.absorption
    asphalt = Material.model_validate({"impedance": {"preset": "asphalt"}})
    assert max(asphalt.coefficients("octave")[0]) < 0.25
    custom = Material.model_validate({"impedance": {"flow_resistivity": 200.0}})
    assert custom.absorption == lawn.absorption
    assert (
        "Miki: 20000 kPa s/m², 0.05 m thick"
        in Impedance(flow_resistivity=20000.0, thickness=0.05).describe()
    )


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"impedance": "lawn"}, "no ground preset named 'lawn'"),
        ({"impedance": "gras"}, "did you mean grass"),
        ({"impedance": {"preset": "grass", "flow_resistivity": 100}}, "one of them"),
        ({"impedance": {}}, "one of them"),
        ({"impedance": {"flow_resistivity": 100, "model": "two_parameter"}}, "porosity_rate"),
        (
            {
                "impedance": {
                    "preset": "grass",
                    "model": "two_parameter",
                    "porosity_rate": 10,
                    "thickness": 0.1,
                }
            },
            "takes no thickness",
        ),
        ({"impedance": {"preset": "grass", "porosity_rate": 10}}, "two_parameter model"),
        ({"impedance": "grass", "preset": "concrete_floor"}, "give no preset or file"),
        ({"impedance": "grass", "transmission": 0.1}, "lets no sound through"),
    ],
)
def test_impedances_that_are_refused(data, message) -> None:
    with pytest.raises(ValidationError, match=message):
        Material.model_validate(data)


def test_a_scene_with_an_impedance_writes_it_and_not_its_absorption() -> None:
    scene = ground_scene({"impedance": {"preset": "snow", "thickness": 0.1}})
    data = scene.to_data()
    assert data["materials"]["field"]["impedance"] == {
        "preset": "snow",
        "model": "miki",
        "thickness": 0.1,
    }
    assert "absorption" not in data["materials"]["field"]
    again = Scene.model_validate(data)
    assert again.materials["field"].absorption == scene.materials["field"].absorption


# -- in pressure in every band outdoors -----------------------------------------------


def test_outdoors_the_early_paths_add_in_pressure_in_every_band() -> None:
    assert ground_scene({"impedance": "grass"}).simulation.early.coherent_max_band_hz == 8000
    given = ground_scene({"impedance": "grass"}, coherent=125)
    assert given.simulation.early.coherent_max_band_hz == 125
    room = Scene.from_yaml(SHOEBOX).to_data()
    assert Scene.model_validate(room).simulation.early.coherent_max_band_hz == 125
    room["simulation"]["early"]["coherent_max_band_hz"] = 1000
    with pytest.raises(ValidationError, match="up to 500 Hz in a room"):
        Scene.model_validate(room)


def test_the_ground_dip_against_the_exact_solution() -> None:
    """The early part over grass, per octave band, against the direct sound and the
    spherical wave's reflection averaged over each band: the ground dip at 500 Hz,
    however large the ground."""
    r, zs, zr = 20.0, 1.0, 1.5
    level = early_level(ground_scene({"impedance": "grass", "scattering": 0.0}))
    r1, r2 = math.hypot(r, zs - zr), math.hypot(r, zs + zr)
    theory = []
    for centre in band_centres("octave"):
        f = np.linspace(centre / math.sqrt(2), centre * math.sqrt(2), 401)
        k = 2 * math.pi * f / C
        q = spherical_reflection(1.0 / surface_impedance(f, 200.0), (zs + zr) / r2, k, r2)
        field = (np.exp(-1j * k * r1) / r1 + q * np.exp(-1j * k * r2) / r2) * r1
        theory.append(10 * math.log10(np.mean(np.abs(field) ** 2)))
    assert level == pytest.approx(theory, abs=0.05)
    assert level[3] < -5.0 and level.argmin() == 3
    # The ground goes on past its extent: a smaller one scatters no more for its size.
    small = early_level(ground_scene({"impedance": "grass", "scattering": 0.0}, extent=25))
    assert small == pytest.approx(level, abs=1e-9)
    # Over a hard ground the direct sound and its reflection add up in the low bands.
    hard = early_level(ground_scene({"absorption": [0.0] * 8, "scattering": 0.0}))
    assert hard[0] > 5.5 and hard[0] > level[0]


def test_the_points_of_a_line_source_add_in_energy_outdoors() -> None:
    scene = ground_scene({"impedance": "grass", "scattering": 0.0})
    data = scene.to_data()
    data["sources"][0].update(line_m=20.0, spacing_m=10.0, orientation={"yaw": 90})
    room = Room.from_scene(Scene.model_validate(data))
    early = simulate(room)[("s", "l")].early
    points = np.unique(early.point)
    assert len(points) == len(room.scene.sources[0].elements()) > 1
    step = 0.001
    slots = int(0.2 / step)
    together = coherent_energy(early, step, slots).sum(axis=1)
    apart = sum(
        coherent_energy(early[np.flatnonzero(early.point == p)], step, slots).sum(axis=1)
        for p in points
    )
    assert together == pytest.approx(apart, rel=1e-9)


def test_calibration_leaves_grounds_alone() -> None:
    data = Scene.from_yaml(SHOEBOX).to_data()
    data["materials"]["carpet_on_concrete"] = {"impedance": "grass"}
    data["simulation"].update(ir_length_s=0.5, seed=1)
    data["simulation"]["late"]["rays"] = 500
    scene = Scene.model_validate(data)
    room = Room.from_scene(scene)
    pair = (scene.sources[0].name, scene.listeners[0].name)
    measured = {pair: simulate(room)[pair].parameters()}
    with pytest.raises(ValueError, match="take their absorption from an impedance"):
        calibrate(room, measured, materials=["carpet_on_concrete"])
    outdoors = Room.from_scene(ground_scene({"impedance": "grass"}))
    with pytest.raises(ValueError, match="not outdoors"):
        calibrate(outdoors, measured)


# -- ground zones and the Fresnel zone --------------------------------------------------


def test_ground_zones_are_parts_of_the_ground() -> None:
    zones = [
        {"name": "road", "polygon": [[-80, -5], [90, -3], [90, 4], [-80, 6]]},
        {"name": "pavement", "polygon": [[-80, 6], [90, 4], [90, 7], [-80, 9]]},
        {"polygon": [[10, 10], [30, 12], [18, 30]]},
        {"name": "far", "polygon": [[100, 100], [110, 100], [110, 110]]},  # past the extent
    ]
    room = Room.from_scene(ground_scene({"impedance": "grass"}, zones=zones, extent=50))
    assert not room.problems() and room.report.closed
    assert [name for name, _ in room.outdoor.zones] == ["road", "pavement", "zone_3"]
    mesh = room.mesh
    corners = mesh.vertices[mesh.faces]
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    flat = np.all(np.abs(corners[:, :, 2]) < 1e-9, axis=1)
    assert area[flat].sum() == pytest.approx(100 * 100)
    by_label = {
        mesh.labels[k][0]: area[mesh.face_labels == k].sum() for k in range(len(mesh.labels))
    }
    # Cut to the extent: from 11 - 4 * 30 / 170 m wide at x = -50 to 11 - 4 * 130 / 170.
    assert by_label["road"] == pytest.approx(100 * (11 - 4 * 80 / 170))
    assert by_label["pavement"] == pytest.approx(100 * 3.0)
    assert by_label["zone_3"] == pytest.approx(192.0)
    assert ("road", "zones", "ground") in mesh.labels
    assert room.outdoor.zone_at(np.array([[0, 0], [0, 7], [20, 15], [-40, 40]])).tolist() == [
        0,
        1,
        2,
        -1,
    ]


@pytest.mark.parametrize(
    ("zones", "message"),
    [
        (
            [
                {"name": "a", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
                {"name": "b", "polygon": [[5, 5], [15, 5], [15, 15], [5, 15]]},
            ],
            "'a' and 'b' overlap",
        ),
        (
            [
                {"name": "a", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
                {"name": "b", "polygon": [[2, 2], [4, 2], [4, 4], [2, 4]]},
            ],
            "'a' and 'b' overlap",
        ),
        ([{"name": "a", "polygon": [[0, 0], [10, 10], [10, 0], [0, 10]]}], "ground zone 'a'"),
    ],
)
def test_zones_that_are_refused(zones, message) -> None:
    with pytest.raises(ValueError, match=message):
        Room.from_scene(ground_scene({"impedance": "grass"}, zones=zones))


def test_zone_names_are_unique_and_free() -> None:
    zone = {"name": "sky", "polygon": [[0, 0], [10, 0], [10, 10]]}
    with pytest.raises(ValidationError, match="'sky'"):
        ground_scene({"impedance": "grass"}, zones=[zone])
    data = ground_scene({"impedance": "grass"}).to_data()
    data["geometry"]["ground"]["zones"] = [{**zone, "name": "hall"}]
    data["geometry"]["buildings"] = [
        {"name": "hall", "footprint": [[20, 20], [30, 20], [30, 30]], "height": 5}
    ]
    with pytest.raises(ValidationError, match="must be unique"):
        Scene.model_validate(data)


def test_the_fresnel_zone_is_where_the_path_is_half_a_wavelength_longer() -> None:
    rng = np.random.default_rng(3)
    for _ in range(20):
        before = np.array([*rng.uniform(-10, 10, 2), rng.uniform(0.1, 15)])
        after = np.array([*rng.uniform(-10, 10, 2), rng.uniform(0.1, 15)])
        at = before + before[2] / (before[2] + after[2]) * (after - before)
        at[2] = 0.0
        wavelength = rng.uniform(0.05, 6.0)
        centre, axes = _fresnel_ellipses(before[None], at[None], after[None], 0.0, [wavelength])
        angle = np.linspace(0, 2 * np.pi, 37)
        edge = centre[0, 0] + (axes[0, 0] @ np.vstack([np.cos(angle), np.sin(angle)])).T
        edge = np.column_stack([edge, np.zeros(len(edge))])
        image = before * [1, 1, -1]
        longer = (
            np.linalg.norm(edge - before, axis=1)
            + np.linalg.norm(edge - after, axis=1)
            - np.linalg.norm(after - image)
        )
        assert longer == pytest.approx(FRESNEL_PATH_DIFFERENCE * wavelength, rel=1e-9)
        assert np.linalg.norm(np.linalg.solve(axes[0, 0], at[:2] - centre[0, 0])) < 1.0


def test_fresnel_shares() -> None:
    road = np.array([[0, -100], [100, -100], [100, 100], [0, 100]], dtype=float)
    layout = Layout((-100, -100, 100, 100), 0.0, 30.0, (), (("road", road),))
    wavelength = C / np.array([63, 250, 1000, 4000, 8000.0])
    # Across the edge: half on each side.
    shares = layout.fresnel_shares([[-10, 0, 1]], [[0, 0, 0]], [[10, 0, 1]], wavelength)
    assert shares.sum(axis=2) == pytest.approx(1.0)
    assert shares[0, :, 1] == pytest.approx(0.5, abs=0.03)
    # 3 m into the road: more of it the higher the frequency.
    shares = layout.fresnel_shares([[-7, 0, 1]], [[3, 0, 0]], [[13, 0, 1]], wavelength)
    assert np.all(np.diff(shares[0, :, 1]) > 0) and 0.6 < shares[0, 0, 1] < shares[0, -1, 1] < 1
    # Well inside, all of it; along the edge but high up, the zone is small.
    shares = layout.fresnel_shares([[30, 0, 1]], [[40, 0, 0]], [[50, 0, 1]], wavelength)
    assert shares[0, :, 1] == pytest.approx(1.0)
    shares = layout.fresnel_shares([[3, 0, 20]], [[3, 0, 0]], [[3, 1, 30]], wavelength)
    assert shares[0, -1, 1] == pytest.approx(1.0)


def test_a_reflection_mixes_the_grounds_round_it() -> None:
    # The reflection point lies at x = -2: the road's edge moves past it.
    grass, asphalt = {"impedance": "grass"}, {"impedance": "asphalt"}
    positions = {"source": (-10, 0, 1.0), "listener": (10, 0, 1.5)}

    def level(edge):
        zones = (
            None
            if edge is None
            else [{"name": "road", "polygon": [[edge, -60], [80, -60], [80, 60], [edge, 60]]}]
        )
        return early_level(ground_scene(grass, zones=zones, road=asphalt, **positions))

    lawn = level(None)
    road = early_level(ground_scene(asphalt, **positions))
    # A road far past the Fresnel zone changes nothing; one under all of it is all road.
    assert level(40) == pytest.approx(lawn, abs=1e-9)
    assert level(-40) == pytest.approx(road, abs=0.02)
    # At the edge, in between in the 500 Hz band (the dip over grass), and smoothly so.
    middle = [level(x)[3] for x in (-6, -2, 2, 6)]
    assert lawn[3] < middle[3] < middle[2] < middle[1] < middle[0] < road[3]
    # Without impedances the zones mix too, and a far road changes nothing.
    plain, hard = {"absorption": [0.4] * 8, "scattering": 0.0}, {"absorption": [0.02] * 8}
    far = early_level(
        ground_scene(
            plain,
            zones=[{"name": "road", "polygon": [[40, -60], [80, -60], [80, 60], [40, 60]]}],
            road=hard,
            **positions,
        )
    )
    assert far == pytest.approx(early_level(ground_scene(plain, **positions)), abs=1e-9)


# -- the wave solver ------------------------------------------------------------------


def test_the_wave_solver_fits_a_ground_to_its_impedance() -> None:
    for preset in ("snow", "grass", "asphalt"):
        impedance = Impedance(preset=preset)
        for bands, total, rate in ((4, 8, None), (4, 8, 8000.0), (12, 24, 8000.0)):
            fit = fit_impedance(impedance.admittance, bands, total, rate)
            assert fit.beta >= 0 and np.all(fit.relaxation > 0) and len(fit.conductance) == 0
            low, high = band_centres("octave")[0] / 1.4, 500 * 1.4
            f = np.geomspace(low, high, 60)
            target, fitted = impedance.admittance(f), fit(f, rate)
            for cosine in (1.0, 0.3, 0.05):
                miss = np.abs(plane_reflection(fitted, cosine) - plane_reflection(target, cosine))
                assert miss.max() < (0.01 if rate is None else 0.05)
            # Its absorption is the material's.
            achieved = band_weights(f, 1, 8) @ random_incidence_absorption(fitted)
            assert achieved == pytest.approx(
                band_weights(f, 1, 8) @ random_incidence_absorption(target), abs=0.01
            )
    # Branches without mass, as the scheme takes them: no mass, resistance, stiffness.
    branches = fit.branches(8000.0)
    assert np.all(branches[:, 0] == 0.0) and np.all(branches[:, 1:] > 0)
    # Resonant branches as before.
    plain = fit_admittance([0.1, 0.3, 0.5, 0.6], 8, 8000.0)
    assert isinstance(plain, Admittance) and len(plain.relaxation) == 0


def test_the_wave_solver_over_grass() -> None:
    pytest.importorskip("acoustic_spaces._native")
    scene = ground_scene(
        {"impedance": "grass"}, extent=8, source=(0, 0, 1.0), listener=(4, 0, 1.5)
    ).to_data()
    scene["geometry"]["ground"]["sky"] = 4
    scene["simulation"]["wave"] = {"enabled": True, "max_band_hz": 63}
    room = Room.from_scene(Scene.model_validate(scene))
    model = TraceModel.from_room(room)
    assert model.face_impedance is not None
    result = simulate(room)
    info = next(iter(result.details.values()))
    assert info["wave_cells"] > 0 and info["wave_absorption_miss"] < 0.02
    response = result[("s", "l")]
    assert np.all(np.isfinite(response.wave.signals))


# -- the command line -------------------------------------------------------------------


def test_the_command_line(tmp_path, capsys) -> None:
    scene = ground_scene(
        {"impedance": {"flow_resistivity": 20000, "thickness": 0.05}},
        zones=[{"name": "road", "polygon": [[-90, -4], [90, -4], [90, 4], [-90, 4]]}],
        road={"impedance": "asphalt"},
    )
    path = tmp_path / "scene.yaml"
    scene.to_yaml(path)
    assert main(["inspect", str(path)]) == 0
    out = capsys.readouterr().out
    assert "impedance (Miki: 20000 kPa s/m², 0.05 m thick)" in out
    assert "ground zones: road (960 m2)" in out
    assert main(["simulate", str(SHOEBOX), "--coherent-band", "1000", "--out", str(tmp_path)]) == 1
    assert "goes up to 500 Hz in a room" in capsys.readouterr().err
