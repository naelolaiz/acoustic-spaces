"""Vegetation outdoors: trees, hedges and belts of trees, which take sound from the
paths through them and scatter some of it on."""

import math

import numpy as np
import pytest
from pydantic import ValidationError
from test_native import _stochastic_args, _unit_cube
from test_refraction import field

from acoustic_spaces import Room, Scene, simulate
from acoustic_spaces.engine import early_paths
from acoustic_spaces.engine.coverage import coverage_map
from acoustic_spaces.engine.model import TraceModel
from acoustic_spaces.engine.ray_device import _unsupported
from acoustic_spaces.engine.vegetation import (
    AIR_GAMMA,
    AIR_PRANDTL,
    AIR_VISCOSITY_M2_S,
    DB_PER_NEPER,
    DISC_RAYLEIGH,
    SHADOW_SHARE,
    Plants,
    cylinder_scattering,
    disc_scattering,
    foliage_medium,
    leaf_absorption,
    leaves_medium,
    plant_arguments,
    stems_medium,
)
from acoustic_spaces.geometry.outdoor import strip, vegetation_shapes
from acoustic_spaces.scene import Coverage, Foliage, Stems, Vegetation
from acoustic_spaces.scene.model import DENSE_FOLIAGE_DB_PER_M

#: A belt of trees 30 m deep across the field of test_refraction, 15 m high.
BELT = {"name": "wood", "footprint": [[40, -30], [70, -30], [70, 30], [40, 30]], "height": 15}


def planted(*vegetation, atmosphere=None, **kwargs) -> Scene:
    """The field of test_refraction (a source 2 m up, a listener 100 m away) with
    ``vegetation``."""
    scene = field(atmosphere, **kwargs)
    data = scene.to_data()
    data["geometry"]["vegetation"] = list(vegetation)
    return Scene.model_validate(data)


# -- the scene -----------------------------------------------------------------------


def test_shapes_and_what_fills_them() -> None:
    belt = Vegetation.model_validate(BELT)
    assert belt.kind == "belt" and belt.foliage.preset == "dense" and belt.stems is None
    hedge = Vegetation.model_validate({"path": [[0, 0], [10, 0]], "width": 1.5, "height": 2})
    tree = Vegetation.model_validate(
        {
            "tree": [5, 5],
            "crown_diameter": 8,
            "base": 3,
            "height": 12,
            "foliage": {"attenuation": [0.1] * 8, "albedo": 0.5, "asymmetry": 0.3},
            "stems": {"per_m2": 0.1, "diameter": 0.4},
        }
    )
    assert (hedge.kind, tree.kind) == ("hedge", "tree")
    assert belt.describe() == "dense foliage"
    assert (
        tree.describe()
        == "foliage of 0.1/0.1/0.1/0.1/0.1/0.1/0.1/0.1 dB/m, 0.1 stems/m², 0.4 m across"
    )
    bare = Vegetation.model_validate(
        {**BELT, "foliage": "none", "stems": {"per_m2": 0.2, "diameter": 0.2}}
    )
    assert bare.foliage.bare and bare.describe() == "0.2 stems/m², 0.2 m across"


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({**BELT, "tree": [0, 0], "crown_diameter": 5}, "exactly one of footprint, path or tree"),
        ({**BELT, "width": 2}, "width goes with path"),
        ({"path": [[0, 0], [5, 0]], "height": 2}, "a path needs it"),
        ({"tree": [0, 0], "height": 9}, "crown_diameter goes with tree"),
        ({**BELT, "base": 15}, "must be below height"),
        ({**BELT, "foliage": "none"}, "no foliage and no stems"),
        ({**BELT, "foliage": "dence"}, "did you mean dense"),
        ({**BELT, "foliage": {"preset": "dense", "attenuation": [0.1] * 8}}, "one of them"),
        ({**BELT, "foliage": {"preset": "none", "albedo": 0.5}}, "nothing to scatter"),
        ({**BELT, "foliage": {"attenuation": [0.1] * 5}}, "expected 8 attenuation values"),
        ({**BELT, "stems": {"per_m2": 0.1}}, "diameter"),
        ({**BELT, "foliage": {"leaf_area_per_m3": 2}}, "go together"),
        ({**BELT, "foliage": {"leaf_width": 0.1}}, "go together"),
        (
            {**BELT, "foliage": {"preset": "dense", "leaf_area_per_m3": 2, "leaf_width": 0.1}},
            "one of them",
        ),
        (
            {**BELT, "foliage": {"leaf_area_per_m3": 2, "leaf_width": 0.1, "albedo": 0.5}},
            "work out what they scatter",
        ),
        ({**BELT, "foliage": {"leaf_area_per_m3": 2, "leaf_width": 2}}, "less than or equal to 1"),
    ],
)
def test_vegetation_is_checked(data, message) -> None:
    with pytest.raises(ValidationError, match=message):
        Vegetation.model_validate(data)


def test_vegetation_stands_on_the_ground_with_names_of_its_own() -> None:
    with pytest.raises(ValidationError, match="vegetation stands on the ground"):
        Scene.model_validate(
            {"geometry": {"box": [5, 4, 3], "vegetation": [BELT]}, "sources": [], "listeners": []}
        )
    with pytest.raises(ValidationError, match="vegetation names must be unique: wood"):
        planted(BELT, {**BELT, "footprint": [[80, 0], [90, 0], [90, 5]]})
    scene = planted(BELT, {"tree": [20, 10], "crown_diameter": 6, "height": 10})
    assert scene.geometry.vegetation_names() == ["wood", "vegetation_2"]
    assert "1 tree, 1 belt" in scene.geometry.describe()
    # Written out and read back as it was.
    assert Scene.model_validate(scene.to_data()) == scene


def test_the_dense_preset_is_iso_9613s_table() -> None:
    dense = Foliage(preset="dense")
    assert dense.attenuation_db("octave").tolist() == list(DENSE_FOLIAGE_DB_PER_M)
    assert len(dense.attenuation_db("third_octave")) == 24
    medium = foliage_medium(dense, "octave")
    # All of it absorbed: the table counts what the foliage scatters onwards.
    assert medium.absorption * DB_PER_NEPER == pytest.approx(DENSE_FOLIAGE_DB_PER_M)
    assert not medium.scattering.any()
    mine = foliage_medium(Foliage(attenuation=[0.2] * 8, albedo=0.25, asymmetry=0.5), "octave")
    assert mine.extinction * DB_PER_NEPER == pytest.approx([0.2] * 8)
    assert mine.scattering == pytest.approx(mine.extinction / 4)
    assert (mine.asymmetry == 0.5).all()
    assert foliage_medium(Foliage(preset="none"), "octave") is None


def test_a_disc_scatters_as_a_dipole_when_small_and_twice_its_shadow_when_large() -> None:
    kr = np.array([0.01, 0.05, 0.2])
    efficiency, cosine = disc_scattering(kr)
    # The dipole of the air a rigid disc holds back: its added mass 8 rho r^3 / 3 along
    # its normal gives 16 (kr)^4 r^2 / (27 pi) with the wave along the normal, a third
    # of that turned every way alike, over pi r^2.
    assert DISC_RAYLEIGH == pytest.approx(16 / (81 * math.pi**2))
    assert efficiency == pytest.approx(16 * kr**4 / (81 * math.pi**2), rel=2e-3)
    assert cosine == pytest.approx(0.0, abs=1e-3)  # as much back as on
    large, forward = disc_scattering(np.array([30.0, 300.0]))
    # Twice its mean projected area (pi r^2 / 2): half the forward shadow, straight on,
    # half reflected every way alike.
    assert large == pytest.approx(1.0, abs=1e-4)
    assert forward == pytest.approx(SHADOW_SHARE, abs=1e-4)
    rising, _ = disc_scattering(np.linspace(0.1, 10, 50))
    assert np.all(np.diff(rising) > 0)


def test_a_leaf_absorbs_in_its_boundary_layers() -> None:
    k = 2 * math.pi * np.array([125.0, 1000.0, 8000.0]) / 343.0
    omega = k * 343.0
    delta_v = np.sqrt(2 * AIR_VISCOSITY_M2_S / omega)
    delta_t = delta_v / math.sqrt(AIR_PRANDTL)
    # Morse and Ingard's rigid wall, per face (k / 2)(delta_v sin^2 + (gamma - 1) delta_t),
    # sin^2 averaging 2/3 over the leaf's turns, on both faces.
    exact = k * (2 / 3 * delta_v + (AIR_GAMMA - 1) * delta_t)
    assert leaf_absorption(k, 343.0) == pytest.approx(exact, rel=1e-12)
    assert exact[1] == pytest.approx(1.45e-3, rel=0.02)  # a few per mille of its area
    assert leaf_absorption(4 * k, 343.0) == pytest.approx(2 * exact, rel=1e-12)  # as sqrt(f)


def test_leaves_make_a_medium_from_their_area_and_width() -> None:
    foliage = Foliage(leaf_area_per_m3=2.0, leaf_width=0.08)
    assert not foliage.bare and foliage.describe() == "leaves of 2 m²/m³, 0.08 m wide"
    medium = foliage_medium(foliage, "octave", 343.0)
    assert medium is not None and not medium.upright
    # The area per cubic metre times a disc's cross-sections over its area, each band
    # the mean over its frequencies.
    lower, upper = 63 / math.sqrt(2), 63 * math.sqrt(2)
    f = np.exp(np.linspace(math.log(lower), math.log(upper), 10)[:-1] + math.log(2) / 18)
    k = 2 * math.pi * f / 343.0
    efficiency, cosine = disc_scattering(k * 0.04)
    assert medium.scattering[0] == pytest.approx(2.0 * efficiency.mean(), rel=1e-9)
    assert medium.absorption[0] == pytest.approx(2.0 * leaf_absorption(k, 343.0).mean(), rel=1e-9)
    assert medium.asymmetry[0] == pytest.approx((efficiency * cosine).sum() / efficiency.sum())
    # Little taken at low frequencies, and absorbed; at high ones scattered, mostly on.
    assert medium.extinction[0] < 1e-3 < 1.0 < medium.extinction[-1]
    assert medium.absorption[0] > medium.scattering[0]
    assert medium.scattering[-1] > 100 * medium.absorption[-1]
    assert 0.4 < medium.asymmetry[-1] < SHADOW_SHARE and medium.asymmetry[0] < 1e-6
    assert np.all(np.diff(medium.extinction) > 0)
    # What the foliage takes in all, per metre, as a scene reports it, and the same at
    # third-octave resolution, which the scene's own bands take.
    assert foliage.attenuation_db("octave", 343.0) == pytest.approx(
        medium.extinction * DB_PER_NEPER
    )
    assert len(foliage.attenuation_db("third_octave", 343.0)) == 24
    # Wider leaves of the same area scatter more at the frequencies where a leaf's size
    # nears the wavelength, and all leaves the same once far past it.
    wide = leaves_medium(2.0, 0.16, "octave", 343.0)
    assert wide.scattering[4] > 2 * medium.scattering[4]
    assert wide.scattering[-1] == pytest.approx(medium.scattering[-1], rel=0.15)
    # The speed of sound sets the wavelength.
    cold = foliage_medium(foliage, "octave", 330.0)
    assert cold.scattering[4] > medium.scattering[4]
    assert Vegetation.model_validate({**BELT, "foliage": foliage.model_dump()}).describe() == (
        "leaves of 2 m²/m³, 0.08 m wide"
    )


def test_leaves_in_a_scene_lose_and_scatter() -> None:
    scene = planted({**BELT, "foliage": {"leaf_area_per_m3": 1.0, "leaf_width": 0.05}})
    room = Room.from_scene(scene)
    plants = Plants.of(room, "octave", room.speed_of_sound())
    medium = leaves_medium(1.0, 0.05, "octave", room.speed_of_sound())
    assert len(plants.media) == 1 and plants.media[0].scattering == pytest.approx(medium.scattering)
    through = plants.depth(np.array([[0.0, 0, 5]]), np.array([[100.0, 0, 5]]))
    assert through[0] == pytest.approx(30 * medium.extinction)
    result = simulate(scene)
    response = result["s", "l0"]
    # The highest bands' early paths lose most; what they scatter comes back late.
    bare = simulate(field())["s", "l0"]
    early_loss = 10 * np.log10(bare.early.energy.sum(axis=0) / response.early.energy.sum(axis=0))
    assert early_loss[-1] == pytest.approx(30 * medium.extinction[-1] * DB_PER_NEPER, rel=0.05)
    assert early_loss[0] < 0.1
    # At 2 and 4 kHz what the leaves scatter out of the early paths reaches the listener
    # as late sound; at 8 kHz they scatter so much that most of it leaves through the sky.
    assert np.all(response.late.sum(axis=1)[5:7] > 2 * bare.late.sum(axis=1)[5:7])


# -- shapes --------------------------------------------------------------------------


def test_a_hedge_is_a_strip_mitred_at_its_bends() -> None:
    straight = strip([[0, 0], [10, 0]], 2.0)
    assert straight.tolist() == [[0, 1], [10, 1], [10, -1], [0, -1]]
    bent = strip([[0, 0], [10, 0], [10, 10]], 2.0)
    # The outer corner of a right angle lies a half-width out along both sides.
    assert bent[1] == pytest.approx([9, 1]) and bent[4] == pytest.approx([11, -1])
    with pytest.raises(ValueError, match="points 2 and 3 are in one place"):
        strip([[0, 0], [5, 0], [5, 0]], 1.0)


def test_lengths_through_belts_and_crowns() -> None:
    scene = planted(
        BELT, {"name": "oak", "tree": [20, 0], "crown_diameter": 8, "base": 4, "height": 12}
    )
    belt, crown = vegetation_shapes(scene.geometry)
    assert crown.centre.tolist() == [20, 0, 8] and crown.radius == 4 and crown.half_height == 4
    assert belt.inside(np.array([[50.0, 0, 5], [80, 0, 5], [50, 0, 25]])).tolist() == [
        True,
        False,
        False,
    ]
    room = Room.from_scene(scene)
    plants = Plants.of(room, "octave", 343.0)
    starts = np.array([[0.0, 0, 8], [20, 0, 0], [0, 10, 5]])
    ends = np.array([[100.0, 0, 8], [20, 0, 20], [100, 10, 5]])
    assert plants.lengths(starts, ends) == pytest.approx(np.array([[30, 8], [0, 8], [30, 0]]))
    # Through the crown at a slant, and through the belt from inside it.
    slant = plants.lengths(np.array([[16.0, 0, 4]]), np.array([[24.0, 0, 12]]))
    assert slant[0, 1] == pytest.approx(8.0, rel=1e-9)
    assert plants.lengths(np.array([[50.0, 0, 5]]), np.array([[50.0, 50, 5]]))[0, 0] == 30


def test_vegetation_may_run_past_the_edge_of_the_ground() -> None:
    wide = {**BELT, "footprint": [[40, -40], [70, -40], [70, 40], [40, 40]]}
    room = Room.from_scene(planted(wide))
    assert room.outdoor.extent == (-150.0, -20.0, 150.0, 20.0)
    assert room.outdoor.vegetation[0].name == "wood"


# -- media ---------------------------------------------------------------------------


def test_a_rigid_cylinder_scatters_as_the_exact_series_has_it() -> None:
    ka = np.array([0.02, 0.05, 30.0, 60.0])
    efficiency, cosine = cylinder_scattering(ka)
    # Small: 3 pi^2 (ka)^3 / 8, mostly backwards (a monopole and a dipole: -2/3).
    assert efficiency[:2] == pytest.approx(3 * math.pi**2 * ka[:2] ** 3 / 8, rel=1e-3)
    assert cosine[:2] == pytest.approx(-2 / 3, abs=3e-3)
    # Large: towards twice its width, half of it in the forward shadow (towards 1/3).
    assert 1.9 < efficiency[2] < efficiency[3] < 2.0
    assert 0.3 < cosine[2] < cosine[3] < 1 / 3


def test_stems_scatter_and_stand_upright() -> None:
    medium = stems_medium(Stems(per_m2=0.1, diameter=0.3), "octave", 343.0)
    assert medium.upright and not medium.absorption.any()
    assert np.all(np.diff(medium.scattering) > 0)
    assert 0.9 * 0.06 < medium.scattering[-1] < 0.06  # per_m2 times twice the diameter
    assert medium.asymmetry[0] < 0 < medium.asymmetry[-1]
    scene = planted({**BELT, "foliage": "none", "stems": {"per_m2": 0.1, "diameter": 0.3}})
    plants = Plants.of(Room.from_scene(scene), "octave", 343.0)
    # Straight up through the belt sound meets no stems; level, all of them.
    up = plants.depth(np.array([[50.0, 0, 0], [0, 0, 5]]), np.array([[50.0, 0, 20], [100, 0, 5]]))
    assert not up[0].any() and up[1] == pytest.approx(30 * medium.scattering)


# -- early paths ---------------------------------------------------------------------


def test_early_paths_lose_what_the_plants_take_beer_lambert() -> None:
    foliage = {"attenuation": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
    bare = early_paths(field())
    through = early_paths(planted({**BELT, "foliage": foliage}))
    assert len(bare) == len(through)
    rate = np.array(foliage["attenuation"]) / DB_PER_NEPER
    for k in range(len(bare)):
        points = through.points[k]
        inside = sum(
            np.linalg.norm(b - a) * (min(b[0], 70) - max(a[0], 40)) / abs(b[0] - a[0])
            for a, b in zip(points[:-1], points[1:], strict=False)
            if abs(b[0] - a[0]) > 0 and min(b[0], 70) > max(a[0], 40)
        )
        assert inside == pytest.approx(30 * np.linalg.norm(points[-1] - points[0]) / 100, rel=0.01)
        assert through.energy[k] == pytest.approx(bare.energy[k] * np.exp(-rate * inside), rel=1e-9)


def test_without_vegetation_nothing_changes() -> None:
    room = Room.from_scene(field())
    model = TraceModel.from_room(room)
    assert model.plants is None and plant_arguments(model, slice(None)) == {}


# -- late rays -----------------------------------------------------------------------


def _big_box(bands: int, rays: int, **kwargs) -> dict:
    """The tracer's arguments for a box 200 m across, all absorbing, a source in its
    middle and receivers 50 m off in x, y and -x (detector spheres only)."""
    from acoustic_spaces import _native

    vertices, faces = _unit_cube()
    args = _stochastic_args(rays)
    args.update(
        bvh=_native.Bvh(vertices * 200.0 - 100.0, faces),
        source=np.zeros(3),
        absorption=np.ones((12, bands)),
        scattering=np.zeros((12, bands)),
        air=np.zeros(bands),
        receivers=np.array([[50.0, 0, 0], [0, 50, 0], [-50, 0, 0]]),
        receiver_radius=np.full(3, 3.0),
        max_time=0.6,
        diffuse_rain=False,
    )
    args.update(kwargs)
    return args


def _slab(absorption, scattering, asymmetry, upright: bool = False, z=(-100, 100)) -> dict:
    """A slab of plants from x = 10 to 40 m across the box."""
    return {
        "plants": np.array([[0, 0, z[0], z[1], 0, 4, 0, 0]], dtype=float),
        "plant_outline": np.array([[10, -100], [40, -100], [40, 100], [10, 100]], dtype=float),
        "plant_absorption": np.atleast_2d(absorption).astype(float),
        "plant_scattering": np.atleast_2d(scattering).astype(float),
        "plant_asymmetry": np.atleast_2d(asymmetry).astype(float),
        "plant_upright": np.array([upright], dtype=np.uint8),
    }


def _trace(**args) -> np.ndarray:
    from acoustic_spaces import _native

    return _native.trace_stochastic(**args)[0].sum(axis=2)


def test_late_rays_lose_what_the_plants_absorb_beer_lambert() -> None:
    rate = np.array([0.02, 0.05])
    free = _trace(**_big_box(2, 50_000))
    through = _trace(**_big_box(2, 50_000), **_slab(rate, [0, 0], [0, 0]))
    # The same rays (absorbing draws no random numbers), 30 m through the slab or none.
    assert through[0] / free[0] == pytest.approx(np.exp(-30 * rate), rel=3e-3)
    assert (through[1:] == free[1:]).all()


def test_each_band_scatters_as_it_would_alone() -> None:
    absorption, scattering, asymmetry = [0.0, 0.01], [0.03, 0.02], [0.0, 0.6]
    big = {"receiver_radius": np.full(3, 10.0), "diffuse_rain": True}
    both = _trace(**_big_box(2, 300_000, **big), **_slab(absorption, scattering, asymmetry))
    for band in range(2):
        alone = _trace(
            **_big_box(1, 300_000, **big),
            **_slab([absorption[band]], [scattering[band]], [asymmetry[band]]),
        )
        assert both[:, band] == pytest.approx(alone[:, 0], rel=0.03)


def test_rain_and_detector_spheres_count_the_scattered_sound_alike() -> None:
    plants = _slab([0.005], [0.03], [0.3])
    receivers = {"receivers": np.array([[80.0, 0, 0]]), "receiver_radius": np.array([8.0])}
    spheres = _trace(**_big_box(1, 1_000_000, **receivers), **plants)
    rain = _trace(**_big_box(1, 1_000_000, diffuse_rain=True, **receivers), **plants)
    # The same rays: the foliage's rain stands in for the spheres after each scattering.
    assert rain[0, 0] == pytest.approx(spheres[0, 0], rel=0.05)


def test_stems_rain_what_their_next_legs_carry_through_the_spheres() -> None:
    plants = _slab([0.0], [0.03], [-0.3], True)
    # Beside the stems, and among them (the next legs then cross the sphere every way).
    receivers = {
        "receivers": np.array([[0.0, 50.0, 0.0], [25.0, 30.0, 0.0]]),
        "receiver_radius": np.array([10.0, 10.0]),
    }
    free = _trace(**_big_box(1, 1_000_000, **receivers))[:, 0]
    spheres = _trace(**_big_box(1, 1_000_000, **receivers), **plants)[:, 0]
    rain = _trace(**_big_box(1, 1_000_000, diffuse_rain=True, **receivers), **plants)[:, 0]
    # The same rays (the rain draws no random numbers): what the stems scatter beside
    # them, and all that reaches the sphere among them.
    assert rain[0] - free[0] == pytest.approx(spheres[0] - free[0], rel=0.06)
    assert rain[1] == pytest.approx(spheres[1], rel=0.02)


def test_upright_media_meet_sound_by_its_horizontal_share() -> None:
    above = {"receivers": np.array([[0.0, 0, 50]]), "receiver_radius": np.array([5.0])}
    free = _trace(**_big_box(1, 50_000, **above))
    upright = _trace(**_big_box(1, 50_000, **above), **_slab_above([0.05], [0.0], [0.0], True))
    level = _trace(**_big_box(1, 50_000, **above), **_slab_above([0.05], [0.0], [0.0], False))
    # Going up through a layer 30 m thick, sound runs along upright stems, which barely
    # touch it, and through all of anything else. (Absorbing: the same rays each time.)
    assert 0.85 < upright[0, 0] / free[0, 0] < 1.0
    assert level[0, 0] / free[0, 0] == pytest.approx(math.exp(-1.5), rel=5e-3)
    sideways = _trace(**_big_box(1, 50_000), **_slab([0.05], [0.0], [0.0], True))
    assert sideways[0, 0] / _trace(**_big_box(1, 50_000))[0, 0] == pytest.approx(
        math.exp(-1.5), rel=5e-3
    )


def _slab_above(*args) -> dict:
    """A layer of plants from z = 10 to 40 m over the whole box."""
    plants = _slab(*args)
    plants["plants"] = np.array([[0, 0, 10, 40, 0, 4, 0, 0]], dtype=float)
    plants["plant_outline"] = np.array([[-100, -100], [100, -100], [100, 100], [-100, 100]], float)
    return plants


def test_plants_are_checked_by_the_tracer() -> None:
    from acoustic_spaces import _native

    plants = _slab([0.0], [0.1], [0.0])
    plants["plant_asymmetry"] = np.array([[1.5]])
    with pytest.raises(ValueError, match="mean cosines between -1 and 1"):
        _native.trace_stochastic(**_big_box(1, 10), **plants)
    plants = _slab([0.0, 0.0], [0.1, 0.1], [0.0, 0.0])
    with pytest.raises(ValueError, match="one value per band"):
        _native.trace_stochastic(**_big_box(1, 10), **plants)


def test_simulations_through_vegetation() -> None:
    scene = planted(
        {**BELT, "stems": {"per_m2": 0.1, "diameter": 0.3}},
        {"name": "oak", "tree": [20, 5], "crown_diameter": 8, "base": 3, "height": 12},
        listeners=((100.0, 0.0, 1.5), (-100.0, 0.0, 1.5)),
    )
    result = simulate(scene)
    behind, open_side = result["s", "l0"], result["s", "l1"]
    loss = 10 * np.log10(open_side.early.energy.sum(axis=0) / behind.early.energy.sum(axis=0))
    # The belt takes most in the highest bands; the stems scatter some into the late part.
    assert loss[-1] > loss[0] > 0 and loss[-1] > 3


def test_coverage_maps_through_vegetation() -> None:
    scene = planted(BELT)
    shown = coverage_map(Room.from_scene(scene), settings=Coverage(spacing=10.0, rays=500))
    bare = coverage_map(Room.from_scene(field()), settings=Coverage(spacing=10.0, rays=500))
    x = shown.grid.points[:, 0]
    level = shown.get().value("g", 7) - bare.get().value("g", 7)
    assert np.all(level[x > 80] < -2) and np.all(np.abs(level[x < 30]) < 0.5)


def test_the_gpu_traces_vegetation_of_a_few_media() -> None:
    values = {"plants": np.zeros((1, 8)), "transmission": np.zeros(0), "groups": 0}
    assert _unsupported(values | {"plant_upright": np.zeros(16, np.uint8)}) == ""
    reason = _unsupported(values | {"plant_upright": np.zeros(17, np.uint8)})
    assert reason == "vegetation of more than 16 media is traced on the CPU"
