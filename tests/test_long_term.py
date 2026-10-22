"""Levels over a long term outdoors: the scene run in each weather class, combined over
the classes' shares of each period of the day, and rated as L_den."""

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_spaces import Room, Scene
from acoustic_spaces.cli import main
from acoustic_spaces.engine import insertion, long_term
from acoustic_spaces.engine.coverage import LONG_TERM_MAP_PARAMETERS
from acoustic_spaces.engine.reweight import groups_of, reweighted_result
from acoustic_spaces.results import write_results
from acoustic_spaces.scene import LongTerm, WeatherClass
from acoustic_spaces.scene.model import Atmosphere

WALL = {"name": "wall", "path": [[10, -30], [10, 30]], "height": 3.0}
#: How much L_den exceeds a level that is the same all day: 10 lg((12 + 4·10^0.5 +
#: 8·10) / 24).
DEN_OVER_STEADY = 10.0 * math.log10((12.0 + 4.0 * 10.0**0.5 + 8.0 * 10.0) / 24.0)
#: Downwind and upwind by day and in the evening, an inversion in the evening and at
#: night.
CLASSES = [
    {"name": "downwind", "share": {"day": 0.5, "evening": 0.4, "night": 0.2}},
    {
        "name": "upwind",
        "share": {"day": 0.5, "evening": 0.4, "night": 0.2},
        "wind": {"direction_deg": 180.0},
    },
    {
        "name": "inversion",
        "share": {"day": 0.0, "evening": 0.2, "night": 0.6},
        "preset": "clear_night",
    },
]


def field(*, barriers=(), coverage=None, classes=None, atmosphere=None, **simulation):
    """A source 1 m over grass in a 5 m/s wind blowing towards +x, a listener 20 m
    downwind and one 20 m upwind, and the weather classes."""
    geometry = {"ground": {"extent": [[-40, -40], [40, 40]], "sky": 20}}
    if barriers:
        geometry["barriers"] = list(barriers)
    data = {
        "geometry": geometry,
        "materials": {
            "field": {"impedance": "grass", "scattering": 0.0},
            "hard": {"absorption": [0.05] * 8, "scattering": 0.0},
        },
        "default_material": "field",
        "assignments": {"wall": "hard"},
        "sources": [{"name": "s", "position": [0, 0, 1.0]}],
        "listeners": [
            {"name": "down", "position": [20, 0, 1.5]},
            {"name": "up", "position": [-20, 0, 1.5]},
        ],
        "atmosphere": atmosphere or {"preset": "windy"},
        "simulation": {
            "ir_length_s": 0.3,
            "seed": 2,
            "early": {"method": "image_sources", "max_order": 1, "diffraction": "direct"},
            "late": {"rays": 1000},
            **simulation,
        },
    }
    if classes is not None:
        data["long_term"] = {"classes": classes}
    if coverage is not None:
        data["coverage"] = coverage
    return Scene.model_validate(data)


def by_hand(levels, weights) -> np.ndarray:
    """10 lg Σ w 10^(L/10) over the first axis."""
    levels = np.asarray(levels, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1, *[1] * (levels.ndim - 1))
    return 10.0 * np.log10(np.sum(weights * 10.0 ** (levels / 10.0), axis=0))


# -- the scene file -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("classes", "periods", "message"),
    [
        ([{"share": 1.2}], None, "between 0 and 1"),
        ([{"share": 0.5}, {"share": 0.4}], None, "of the day add up to 0.9, not 1"),
        ([{"share": {"day": 1.0, "evening": 1.0}}], None, "of the night add up to 0, not 1"),
        ([{"share": {"day": 1.0, "dusk": 1.0}}], None, "dusk"),
        ([{"name": "a", "share": 0.5}, {"name": "a", "share": 0.5}], None, "'a' twice"),
        ([{"share": 1.0}], [{"name": "day", "hours": 12}] * 2, "'day' twice"),
        (
            [{"share": 1.0, "table": [[0, 15, 0], [10, 15, 2]], "temperature_log_k": 0.5}],
            None,
            "give a weather class a table or the temperature's gradient and log terms",
        ),
        ([{"share": 1.0, "preset": "monsoon"}], None, "monsoon"),
    ],
)
def test_weather_classes_are_checked(classes, periods, message) -> None:
    spec = {"classes": classes} if periods is None else {"classes": classes, "periods": periods}
    with pytest.raises(ValidationError, match=message):
        LongTerm.model_validate(spec)


def test_the_levels_over_a_long_term_are_for_outdoor_scenes() -> None:
    data = {
        "geometry": {"box": [6, 5, 3]},
        "materials": {"plaster": {"absorption": [0.1] * 8}},
        "default_material": "plaster",
        "sources": [{"name": "s", "position": [2, 2, 1.5]}],
        "listeners": [{"name": "l", "position": [4, 3, 1.2]}],
        "simulation": {"ir_length_s": 0.2, "late": {"rays": 200}},
        "long_term": {"classes": [{"share": 1.0}]},
    }
    with pytest.raises(ValidationError, match="long_term is for outdoor scenes"):
        Scene.model_validate(data)
    data.pop("long_term")
    room = Room.from_scene(Scene.model_validate(data))
    spec = LongTerm(classes=[WeatherClass(share=1.0)])
    with pytest.raises(ValueError, match="the levels over a long term are for outdoor"):
        long_term.simulate(room, long_term=spec)


def test_the_periods_are_the_day_the_evening_and_the_night_unless_given() -> None:
    spec = LongTerm.model_validate({"classes": CLASSES})
    assert spec.period_names() == ["day", "evening", "night"]
    assert spec.hours().tolist() == [12.0, 4.0, 8.0]
    assert spec.penalties().tolist() == [0.0, 5.0, 10.0]
    assert spec.class_names() == ["downwind", "upwind", "inversion"]
    assert spec.shares().tolist() == [[0.5, 0.4, 0.2], [0.5, 0.4, 0.2], [0.0, 0.2, 0.6]]
    assert long_term.rating_name(spec.period_names(), spec.hours(), spec.penalties()) == "L_den"
    # A share for the whole day holds in every period; names follow the classes' order.
    plain = LongTerm.model_validate({"classes": [{"share": 0.7}, {"share": 0.3}]})
    assert plain.class_names() == ["class_1", "class_2"]
    assert plain.shares().tolist() == [[0.7] * 3, [0.3] * 3]
    own = LongTerm.model_validate(
        {
            "classes": [{"share": 1.0}],
            "periods": [{"name": "day", "hours": 16}, {"name": "night", "hours": 8}],
        }
    )
    assert own.period_names() == ["day", "night"] and own.penalties().tolist() == [0.0, 0.0]
    assert long_term.rating_name(own.period_names(), own.hours(), own.penalties()) == "L_R"


def test_a_class_takes_what_it_leaves_out_from_the_scene_s_weather() -> None:
    windy = Atmosphere(preset="windy", turbulence={"preset": "moderate"})
    assert windy.wind.speed_m_s == 5.0 and windy.wind.direction_deg == 0.0
    turned = windy.under(WeatherClass(share=1.0, wind={"direction_deg": 180.0}))
    assert turned.wind.speed_m_s == 5.0 and turned.wind.direction_deg == 180.0
    assert turned.turbulence == windy.turbulence
    # A preset gives the class that weather whole: a clear night has no wind.
    night = windy.under(WeatherClass(share=1.0, preset="clear_night"))
    assert night.preset == "clear_night" and night.temperature_log_k == 0.5
    assert night.wind is None or night.wind.speed_m_s == 0.0
    calm = windy.under(WeatherClass(share=1.0, turbulence={"preset": "weak"}))
    assert calm.turbulence.preset == "weak" and calm.wind == windy.wind
    # The scene in a class: its weather, and the ground's temperature if the class gives
    # one; the classes themselves are left out.
    scene = field(classes=CLASSES)
    cool = scene.under(WeatherClass(share=1.0, preset="clear_night", temperature_c=8.0))
    assert cool.long_term is None and cool.air.temperature_c == 8.0
    assert cool.atmosphere.temperature_log_k == 0.5
    same = scene.under(WeatherClass(share=1.0))
    assert same.atmosphere.model_dump(exclude={"preset"}) == scene.atmosphere.model_dump(
        exclude={"preset"}
    )
    assert same.air == scene.air


# -- the arithmetic -------------------------------------------------------------------


def test_the_energy_mean() -> None:
    assert long_term.energy_mean([60.0, 60.0], [0.3, 0.7]) == pytest.approx(60.0)
    half = long_term.energy_mean([60.0, -np.inf], [0.5, 0.5])
    assert half == pytest.approx(60.0 - 10.0 * math.log10(2.0))
    assert long_term.energy_mean([-np.inf, -np.inf], [0.5, 0.5]) == -np.inf
    # A class that never holds takes nothing, whatever its level; one that holds does.
    assert long_term.energy_mean([60.0, np.nan], [1.0, 0.0]) == pytest.approx(60.0)
    assert np.isnan(long_term.energy_mean([60.0, np.nan], [0.5, 0.5]))
    levels = np.array([[50.0, 70.0], [62.0, 40.0]])  # (classes, bands)
    found = long_term.energy_mean(levels, [0.25, 0.75])
    np.testing.assert_allclose(found, by_hand(levels, [0.25, 0.75]))
    # Favourable and homogeneous conditions, as CNOSSOS-EU combines them.
    p, favourable, homogeneous = 0.4, 65.0, 58.0
    expected = 10.0 * math.log10(p * 10.0 ** (favourable / 10.0) + (1 - p) * 10.0**5.8)
    assert long_term.energy_mean([favourable, homogeneous], [p, 1 - p]) == pytest.approx(expected)


def test_l_den() -> None:
    hours, penalties = [12.0, 4.0, 8.0], [0.0, 5.0, 10.0]
    steady = long_term.rating_level([55.0, 55.0, 55.0], hours, penalties)
    assert steady == pytest.approx(55.0 + DEN_OVER_STEADY)
    assert DEN_OVER_STEADY == pytest.approx(6.3952, abs=1e-4)
    day, evening, night = 60.0, 55.0, 50.0
    expected = 10.0 * math.log10(
        (12 * 10 ** (day / 10) + 4 * 10 ** ((evening + 5) / 10) + 8 * 10 ** ((night + 10) / 10))
        / 24
    )
    assert long_term.rating_level([day, evening, night], hours, penalties) == pytest.approx(
        expected
    )
    assert long_term.mean_level([day, evening, night], hours) == pytest.approx(
        10.0 * math.log10((12 * 10**6.0 + 4 * 10**5.5 + 8 * 10**5.0) / 24)
    )
    # Per point of a map: the periods along the first axis.
    levels = np.array([[60.0, 40.0], [55.0, 40.0], [50.0, 40.0]])
    np.testing.assert_allclose(
        long_term.rating_level(levels, hours, penalties), [expected, 40.0 + DEN_OVER_STEADY]
    )


# -- the runs -------------------------------------------------------------------------


def test_one_class_for_all_the_time_is_the_plain_run() -> None:
    scene = field(classes=[{"share": 1.0}])
    result = long_term.simulate(Room.from_scene(scene))
    plain = insertion.simulate(Room.from_scene(field()))
    assert result.details["long_term"]["runs"] == 1.0
    levels = result.long_term
    assert levels.classes == ("class_1",) and levels.rating == "L_den"
    power = scene.sources[0].power_db("octave")
    for key in plain.responses:
        np.testing.assert_array_equal(result[key].energy(), plain[key].energy())
        own = plain[key].levels(power)
        np.testing.assert_allclose(levels.levels[key], own, atol=1e-9)
        np.testing.assert_allclose(levels.period_levels[key], [own] * 3, atol=1e-9)
        a = long_term.a_weighted(own)
        assert levels.levels_a[key] == pytest.approx(a)
        assert levels.rating_level[key] == pytest.approx(a + DEN_OVER_STEADY)
        relative = plain[key].relative_to_free_field()
        np.testing.assert_allclose(levels.re_free_field[key], relative, atol=1e-9)
    # Without classes the runs are the plain ones.
    again = long_term.simulate(Room.from_scene(field()))
    assert again.long_term is None and "long_term" not in again.details


def test_the_scene_is_run_in_each_weather_class() -> None:
    scene = field(classes=CLASSES)
    result = long_term.simulate(Room.from_scene(scene))
    levels = result.long_term
    # The downwind class is the scene's own weather: it takes the scene's run.
    assert result.details["long_term"] == pytest.approx(
        {"classes": 3.0, "runs": 3.0, "seconds": result.details["long_term"]["seconds"]}
    )
    assert levels.runs["downwind"].responses is result.responses
    upwind = insertion.simulate(Room.from_scene(scene.under(WeatherClass(**CLASSES[1]))))
    power = scene.sources[0].power_db("octave")
    shares = np.array([[c["share"][p] for p in ("day", "evening", "night")] for c in CLASSES])
    for key in result.responses:
        per_class = levels.class_levels[key]
        np.testing.assert_array_equal(per_class[0], result[key].levels(power))
        np.testing.assert_allclose(per_class[1], upwind[key].levels(power), atol=1e-9)
        periods = np.array([by_hand(per_class, shares[:, p]) for p in range(3)])
        np.testing.assert_allclose(levels.period_levels[key], periods, atol=1e-9)
        mean = by_hand(periods, np.array([12.0, 4.0, 8.0]) / 24.0)
        np.testing.assert_allclose(levels.levels[key], mean, atol=1e-9)
        a = [long_term.a_weighted(level) for level in periods]
        den = 10.0 * math.log10(
            (12 * 10 ** (a[0] / 10) + 4 * 10 ** ((a[1] + 5) / 10) + 8 * 10 ** ((a[2] + 10) / 10))
            / 24
        )
        assert levels.rating_level[key] == pytest.approx(den, abs=1e-9)
        np.testing.assert_allclose(
            levels.class_a[key], [long_term.a_weighted(x) for x in per_class]
        )
    # The wind turned round changes what each listener hears.
    down, up = ("s", "down"), ("s", "up")
    assert not np.allclose(levels.class_levels[down][0], levels.class_levels[down][1], atol=0.1)
    assert not np.allclose(levels.class_levels[up][0], levels.class_levels[up][1], atol=0.1)


def test_every_class_traces_the_same_rays() -> None:
    scene = field(classes=CLASSES, seed=None)
    result = long_term.simulate(Room.from_scene(scene))
    seeds = {run.settings.seed for run in result.long_term.runs.values()}
    assert result.settings.seed is not None and seeds == {result.settings.seed}


def test_the_insertion_loss_over_a_long_term() -> None:
    scene = field(barriers=[WALL], classes=CLASSES, insertion_loss_of=["wall"])
    result = long_term.simulate(Room.from_scene(scene))
    levels = result.long_term
    key = ("s", "down")
    assert set(levels.insertion_loss) == set(result.responses)
    # The long-term level without the wall, less that with it.
    losses = np.array([levels.runs[name].insertion_loss[key] for name in levels.classes])
    bare = levels.class_levels[key] + losses
    shares = levels.shares
    periods = np.array([by_hand(bare, shares[:, p]) for p in range(3)])
    without = by_hand(periods, np.array([12.0, 4.0, 8.0]) / 24.0)
    np.testing.assert_allclose(levels.insertion_loss[key], without - levels.levels[key], atol=1e-9)
    assert levels.insertion_loss_a[key] == pytest.approx(
        long_term.a_weighted(without) - levels.levels_a[key], abs=1e-9
    )
    assert levels.insertion_loss_a[key] > 10.0  # behind the wall


def test_re_weighting_drops_the_levels_over_a_long_term() -> None:
    scene = field(barriers=[WALL], classes=[{"share": 1.0}])
    room = Room.from_scene(scene)
    face_group, materials = groups_of(room)
    result = long_term.simulate(room, groups=face_group)
    assert result.long_term is not None
    data = scene.to_data()
    data["materials"]["hard"]["absorption"] = [0.3] * 8
    found = reweighted_result(result, scene, face_group, materials, Scene.model_validate(data))
    assert found is not None and found.long_term is None


# -- the results ----------------------------------------------------------------------


def test_the_report(tmp_path) -> None:
    scene = field(barriers=[WALL], classes=CLASSES, insertion_loss_of=["wall"])
    result = long_term.simulate(Room.from_scene(scene))
    report = write_results(result, tmp_path / "out")
    assert report["long_term"]["classes"] == ["downwind", "upwind", "inversion"]
    assert report["long_term"]["shares"]["inversion"] == {"day": 0.0, "evening": 0.2, "night": 0.6}
    assert report["long_term"]["periods"][2] == {"name": "night", "hours": 8.0, "penalty_db": 10.0}
    assert report["long_term"]["rating"] == "L_den"
    levels = result.long_term
    entry = report["responses"][0]["long_term"]
    key = ("s", "down")
    assert entry["rating"] == "L_den"
    assert entry["rating_level_db"] == pytest.approx(levels.rating_level[key], abs=0.006)
    assert entry["spl_db"] == pytest.approx(levels.levels[key], abs=0.006)
    assert entry["periods"]["night"]["spl_a_db"] == pytest.approx(
        levels.period_a[key][2], abs=0.006
    )
    assert entry["classes"]["upwind"]["spl_db"] == pytest.approx(
        levels.class_levels[key][1], abs=0.006
    )
    assert entry["re_free_field_a_db"] == pytest.approx(levels.re_free_field_a[key], abs=0.006)
    assert entry["insertion_loss_db"] == pytest.approx(levels.insertion_loss[key], abs=0.006)
    saved = json.loads((tmp_path / "out" / "parameters.json").read_text())
    assert saved["responses"][0]["long_term"] == entry
    # Without classes the report is as it was.
    plain = write_results(insertion.simulate(Room.from_scene(field())), tmp_path / "plain")
    assert "long_term" not in plain and "long_term" not in plain["responses"][0]


def test_the_command_line(tmp_path, capsys) -> None:
    path = tmp_path / "field.yaml"
    field(classes=CLASSES).to_yaml(path)
    assert main(["inspect", str(path)]) == 0
    text = capsys.readouterr().out
    assert (
        "long term: 3 weather classes over day 12 h, evening 4 h +5 dB, night 8 h +10 dB "
        "(L_den)" in text
    )
    assert "weather class downwind (day 50 %, evening 40 %, night 20 %)" in text
    assert "weather class upwind" in text and "blowing towards 180 deg from +x" in text
    assert "weather class inversion (day 0 %, evening 20 %, night 60 %): clear_night" in text
    out = tmp_path / "out"
    assert main(["simulate", str(path), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "long term L_den" in text and "in the weather classes downwind" in text
    assert "long term re free field" in text
    report = json.loads((out / "parameters.json").read_text())
    assert report["details"]["long_term"]["runs"] == 3.0


def test_maps_over_a_long_term() -> None:
    coverage = {"spacing": 20.0, "height": 1.5, "rays": 100}
    scene = field(classes=[{"share": 1.0}], coverage=coverage)
    result = long_term.coverage_map(Room.from_scene(scene))
    assert result.details["long_term"]["runs"] == 1.0
    assert set(LONG_TERM_MAP_PARAMETERS) <= set(result.parameters())
    found = result.get()
    spl = found.values["spl"]
    np.testing.assert_allclose(found.values["long_term"], spl, atol=1e-9)
    np.testing.assert_allclose(
        found.values["rating_level"], long_term.a_weighted(spl) + DEN_OVER_STEADY, atol=1e-9
    )
    assert found.value("rating_level") == pytest.approx(
        found.value("spl_a") + DEN_OVER_STEADY, abs=0.01
    )
    assert "rating_level" in result.as_dict()["parameters"]
    # In three classes the maps share their points, and the long-term level lies within
    # the classes' levels.
    scene = field(classes=CLASSES, coverage=coverage)
    result = long_term.coverage_map(Room.from_scene(scene))
    assert result.details["long_term"]["runs"] == 3.0
    runs = [
        insertion.coverage_map(
            Room.from_scene(scene.under(WeatherClass(**given))), grid=result.grid
        ).get()
        for given in CLASSES
    ]
    spl = np.array([run.values["spl"] for run in runs])
    mean = result.get().values["long_term"]
    finite = np.isfinite(spl).all(axis=0)
    assert finite.any()
    assert (mean[finite] <= spl.max(axis=0)[finite] + 1e-9).all()
    assert (mean[finite] >= spl.min(axis=0)[finite] - 1e-9).all()
    # Without classes, the maps have no levels over a long term.
    plain = long_term.coverage_map(Room.from_scene(field(coverage=coverage)))
    assert not set(LONG_TERM_MAP_PARAMETERS) & set(plain.parameters())
