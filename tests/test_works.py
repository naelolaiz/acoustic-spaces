"""The road works of the street example (:mod:`acoustic_spaces.demos.plant`, the scene's
``defra`` and ``sae-j994`` models): Defra's rows, their sound power, the EU's limits,
the reversing alarm, the made sounds and works_demo.py, the preset of the demo of the
street's scene that plays them."""

import argparse
import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.demos import plant
from acoustic_spaces.examples import example_path
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import A_WEIGHTING_DB, a_weighted

RATE = 16_000
#: The works' sources in the street's scene (the group Road works), as works_demo.py
#: plays them.
WORKS = (plant.BREAKER.name, plant.GENERATOR.name, plant.DUMPER.name, plant.ALARM)


def test_each_rows_octaves_add_up_to_its_level() -> None:
    """Each row's octave levels, A-weighted and added, give the row's own A-weighted
    level within half a decibel (Defra rounds them), and its sound power is that level
    plus 28 dB in every band."""
    for machine in plant.MACHINES.values():
        assert plant.defra_a_sum(machine) == pytest.approx(machine.level_10m_db_a, abs=0.5)
        np.testing.assert_allclose(machine.sound_power(), np.asarray(machine.levels_10m_db) + 28.0)
        assert machine.sound_power_a() == pytest.approx(plant.defra_a_sum(machine) + 28.0, abs=1e-9)
    # 28 dB is a half sphere 10 m round: 20 lg 10 + 10 lg 2π.
    assert plant.TO_POWER_DB == pytest.approx(20.0 + 10.0 * math.log10(2.0 * math.pi), abs=0.05)
    assert plant.DUMPER.drive_by and not plant.BREAKER.drive_by


def test_the_streets_works_have_their_sound_power() -> None:
    """The street's scene holds each source of the works with its model, Defra's row or
    SAE J994's alarm, at the sound power the code gives it, in the group Road works,
    where the demo says, left out until picked."""
    from acoustic_spaces import Scene

    scene = Scene.from_yaml(example_path("street"))
    found = {s.name: s for s in scene.sources}
    for name in WORKS:
        source = found[name]
        assert not source.enabled and source.group == "Road works"
        machine = plant.MACHINES.get(name)
        if machine is None:
            assert source.model.method == "sae-j994"
            wanted = plant.alarm_power(48_000)
        else:
            assert (source.model.method, source.model.machine) == ("defra", name)
            wanted = machine.sound_power()
        np.testing.assert_allclose(source.power_db(), wanted, atol=1e-6)
    assert found["breaker"].position[2] == plant.BREAKER.height_m
    assert found["generator"].position[2] == plant.GENERATOR.height_m
    assert found["dumper"].position[2] == plant.DUMPER.height_m
    alarm = found[plant.ALARM]
    assert alarm.position[2] == plant.ALARM_HEIGHT_M
    # At the dumper's back, facing the breaker.
    assert alarm.position[0] < found["dumper"].position[0]
    assert found["breaker"].position[0] < alarm.position[0]
    assert alarm.orientation.yaw == 180.0
    assert alarm.directivity.file.name == "siren.csv"


def test_the_eus_limits_and_the_machines() -> None:
    """Stage II of Directive 2000/14/EC, rounded to the nearest decibel: the 6 kVA
    generator's sound power is under its limit, the dumper's drive-by at it."""
    assert plant.eu_limit_db_a("excavator", 15.0) == 93
    assert plant.eu_limit_db_a("excavator", 121.0) == 103
    assert plant.eu_limit_db_a("dumper", 55.0) == 101
    assert plant.eu_limit_db_a("dumper", 75.0) == 103
    assert plant.eu_limit_db_a("generator", 2.0) == 95
    assert plant.eu_limit_db_a("generator", 6.0) == 97
    assert plant.eu_limit_db_a("generator", 20.0) == 96
    with pytest.raises(ValueError):
        plant.eu_limit_db_a("crane", 10.0)
    assert plant.GENERATOR.sound_power_a() < plant.eu_limit_db_a("generator", 6.0) - 3.0
    assert plant.DUMPER.sound_power_a() == pytest.approx(
        plant.eu_limit_db_a("dumper", 75.0), abs=1.0
    )


def test_the_alarm_beeps_as_its_rule_and_its_maker_say() -> None:
    """A tone within SAE J994's, on and off for as long, as many beeps a second as J994
    allows; its sound power gives ALARM_ON_AXIS_DB_A at ALARM_AT_M on its axis in a free
    field, and its sound power in the scene is half that, its average over the beeps."""
    from acoustic_spaces.demos.siren import directivity_index_db
    from acoustic_spaces.synthesis import split_bands

    assert plant.J994_HZ[0] <= plant.ALARM_HZ <= plant.J994_HZ[1]
    assert plant.J994_PER_S[0] <= plant.ALARM_PER_S <= plant.J994_PER_S[1]
    period = 1.0 / plant.ALARM_PER_S
    t = np.linspace(0.0, period, 10_001)[:-1]
    on = plant.beeping(t) > 0.5
    assert np.mean(on) == pytest.approx(0.5, abs=0.01)
    assert plant.alarm_duty() == pytest.approx(
        0.5 - 2 * plant.ALARM_RISE_S * plant.ALARM_PER_S * 0.625, abs=1e-3
    )
    beep = plant.alarm_beep_power(RATE)
    on_axis = a_weighted(beep) + directivity_index_db() - 20.0 * math.log10(plant.ALARM_AT_M)
    on_axis -= 10.0 * math.log10(4.0 * math.pi)
    assert on_axis == pytest.approx(plant.ALARM_ON_AXIS_DB_A, abs=1e-6)
    average = plant.alarm_power(RATE)
    assert a_weighted(average) == pytest.approx(
        a_weighted(beep) + 10.0 * math.log10(plant.alarm_duty()), abs=1e-6
    )
    # The tone is in the 1 kHz octave: the others hold what the filters leak into them.
    assert int(np.argmax(beep)) == 4 and beep[4] - np.max(np.delete(beep, 4)) > 45.0
    x = plant.alarm_sound(RATE, 4 * period)
    assert float(np.sqrt(np.mean(x**2))) == pytest.approx(plant.SOUND_RMS, rel=1e-3)
    spectrum = np.abs(np.fft.rfft(x))
    peak = np.fft.rfftfreq(len(x), 1.0 / RATE)[int(np.argmax(spectrum))]
    assert peak == pytest.approx(plant.ALARM_HZ, abs=1.0)
    bands = np.mean(split_bands(x, RATE) ** 2, axis=1)
    assert int(np.argmax(bands)) == 4


@pytest.mark.parametrize("name", sorted(plant.MACHINES))
def test_a_machines_sound_has_its_rows_spectrum(name) -> None:
    """Each machine's made sound has its row's octave spectrum, within a tenth of a
    decibel up to 4 kHz, at an RMS of SOUND_RMS; the breaker's blows and the engines'
    firing come at their rates."""
    from acoustic_spaces.demos.road import firing_rate_hz
    from acoustic_spaces.synthesis import split_bands

    machine = plant.MACHINES[name]
    x = plant.machine_sound(machine, RATE, 6.0)
    assert float(np.sqrt(np.mean(x**2))) == pytest.approx(plant.SOUND_RMS, rel=1e-9)
    energy = np.mean(split_bands(x, RATE) ** 2, axis=1)
    got = 10.0 * np.log10(energy / energy.sum())
    power = machine.sound_power()
    wanted = power - 10.0 * math.log10(float(np.sum(10.0 ** (power / 10.0))))
    np.testing.assert_allclose(got[:7], wanted[:7], atol=0.1)
    # How often it strikes or fires: the strongest rate of its envelope.
    envelope = np.abs(x)
    window = int(0.002 * RATE)
    envelope = np.convolve(envelope, np.ones(window) / window, mode="same")
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean()))
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / RATE)
    low = (freqs > 2.0) & (freqs < 200.0)
    rate = freqs[low][int(np.argmax(spectrum[low]))]
    expected = (
        plant.BLOWS_PER_MINUTE / 60.0
        if machine.made == "blows"
        else firing_rate_hz(machine.rpm, machine.cylinders)
    )
    assert rate == pytest.approx(expected, rel=0.03)
    with pytest.raises(ValueError):
        plant.machine_sound(machine, RATE, 0.0)


def _street(tmp_path):
    """The street example, quick: 16 kHz, short responses, few rays, seeded."""
    import shutil

    import yaml

    folder = tmp_path / "street"
    shutil.copytree(Path(example_path("street")).parent, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["simulation"].update({"sample_rate": RATE, "ir_length_s": 0.3, "seed": 1})
    data["simulation"]["late"] = {"method": "stochastic_ray_tracing", "rays": 500}
    data["simulation"]["early"]["diffraction"] = "none"
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return folder


def _works_demo(folder):
    from acoustic_spaces.demos import load

    return load(folder / "works_demo.py").at(folder)


def test_the_works_demo_plays_each_machine_at_its_sound_power(tmp_path, monkeypatch) -> None:
    """works_demo.py on the quick street: the scene with the works taken in, at their
    models' sound power, a feed per source as long as the tour at the made sounds' RMS,
    the mixes and the stills; at every listener each source's sound in the mix is as far
    under the breaker's as its simulated level is (within 1.5 dB), not as loud as it;
    each as its response is at the frequencies it holds (the alarm's tone, an engine's
    firing), not on average over each octave."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene, render
    from acoustic_spaces.demos import audio, pipeline, run_folder, scene_demo
    from acoustic_spaces.synthesis import split_bands

    heard = []
    real_convolve = render.convolve

    def convolve(dry, ir):
        wet = real_convolve(dry, ir)
        if np.ndim(wet) == 2:  # a source heard at a listener, not a made sound's pulses
            heard.append(np.array(wet))
        return wet

    monkeypatch.setattr(render, "convolve", convolve)
    folder = _street(tmp_path)
    demo = _works_demo(folder)
    run = ["all", "--seconds", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    assert {k for k, v in demo.plays(args).items() if v != "silent"} == set(WORKS)
    assert demo.main(run) == 0
    scene = Scene.from_yaml(scene_demo.run_scene_file(args))
    found = {s.name: s for s in scene.sources}
    assert all(found[name].in_use for name in WORKS)
    for name in WORKS:
        machine = plant.MACHINES.get(name)
        wanted = plant.alarm_power(48_000) if machine is None else machine.sound_power()
        np.testing.assert_allclose(found[name].power_db(), wanted, atol=1e-6)
    # The feeds of the scene so run, its simulation for the output and the run's
    # stills, each in its folder.
    feeds = scene_demo.feeds_folder(args)
    assert feeds == folder / "demo" / "feeds" / "scene_with-breaker+generator+dumper+dumper_alarm"
    results = demo.simulations(args)[0].results
    assert results == scene_demo.results_of(args) and results.name == "binaural"
    for name in WORKS:
        x, rate = audio.read(feeds / f"{name}.wav")
        assert rate == RATE and len(x) == 12 * RATE
        assert float(np.sqrt(np.mean(x**2))) == pytest.approx(plant.SOUND_RMS, rel=0.01)
    assert run_folder(args).name == "works_seconds-4_binaural"
    assert sorted(p.name for p in (run_folder(args) / "stills").glob("*.png"))
    report = pipeline.read_json(results / pipeline.PARAMETERS)
    levels = {(r["source"], r["listener"]): r["spl_a_db"] for r in report["responses"]}
    weights = 10.0 ** (np.asarray(A_WEIGHTING_DB) / 10.0)

    def a_energy(x: np.ndarray) -> float:
        return sum(
            float(weights @ np.sum(split_bands(x[:, k], RATE) ** 2, axis=1))
            for k in range(x.shape[1])
        )

    def fine_db(name: str, listener: str) -> float:
        """How much louder a source's sound comes through its response than its octave
        bands' levels say, A-weighted: a tone (the alarm's beeps, an engine's firing) is
        heard as the response is at its frequencies, where the street's reflections add
        up or cancel, not as each octave is on average."""
        ir, _ = audio.read(results / f"{name}__{listener}.binaural.wav")
        x, _ = audio.read(feeds / f"{name}.wav")
        n = 1 << math.ceil(math.log2(len(x) + len(ir)))
        sound = np.abs(np.fft.rfft(x[:, 0], n)) ** 2
        through = np.sum(np.abs(np.fft.rfft(ir, n, axis=0)) ** 2, axis=1)
        freqs = np.fft.rfftfreq(n, 1.0 / RATE)
        heard = said = 0.0
        for centre, weight in zip(OCTAVE_BANDS_HZ, weights, strict=True):
            octave = (freqs >= centre / math.sqrt(2.0)) & (freqs < centre * math.sqrt(2.0))
            if not octave.any():
                continue
            heard += weight * float(np.sum(sound[octave] * through[octave]))
            said += weight * float(np.sum(sound[octave])) * float(np.mean(through[octave]))
        return 10.0 * math.log10(heard / said)

    listeners = [lis.name for lis in scene.listeners if lis.in_use]
    count = len(WORKS)
    assert len(heard) == count * len(listeners)
    for k, listener in enumerate(listeners):
        wet = heard[k * count : (k + 1) * count]
        first = a_energy(wet[0])
        for name, x in zip(WORKS[1:], wet[1:], strict=True):
            under = 10.0 * math.log10(first / a_energy(x))
            told = levels[("breaker", listener)] - levels[(name, listener)]
            told += fine_db("breaker", listener) - fine_db(name, listener)
            assert under == pytest.approx(told, abs=1.5), (listener, name)
    # Another rate of blows makes the feeds again, not the simulation.
    again = demo.resolved(demo.parser().parse_args([*run, "--blows", "320"]))
    assert demo.is_done(demo.step("simulate"), again)
    assert not demo.is_done(demo.step("feeds"), again)


def test_the_works_demo_keeps_the_scripts_command_lines(tmp_path) -> None:
    """The script's options and steps, as the demo of the street's scene reads them: a
    rate of blows that is no rate, an input for a source that does not play and a
    listener the street does not have are refused; its listeners are the street's in
    use, its --tour still picks them, its scene step simulates, its --from names the
    scene and its --indoors works and works_and_rumble read as airborne and
    airborne_and_rumble."""
    from acoustic_spaces.demos import run_folder

    folder = _street(tmp_path)
    demo = _works_demo(folder)

    def run(*argv: str):
        return demo.parser().parse_args(list(argv))

    assert demo.check(run("feeds", "--blows", "0")) == "--blows is a positive number, not 0.0"
    assert demo.check(run("feeds", "--input", "car=x.wav")) == (
        "--input car: no such source plays here; it has breaker, generator, dumper, dumper_alarm"
    )
    assert demo.listeners(demo.resolved(run("all"))) == ["pavement", "balcony", "garden"]
    assert "roof" in demo.check(run("all", "--listeners", "garden,roof"))
    toured = run("all", "--tour", "garden,pavement")
    assert demo.check(toured) is None and demo.resolved(toured).tour == "garden,pavement"
    assert demo.steps_named("scene") == ["simulate"]
    given = demo.resolved(run("all", "--from", "scene.yaml"))
    assert Path(given.scene) == folder / "scene.yaml"
    for old, new in (("works", "airborne"), ("works_and_rumble", "airborne_and_rumble")):
        assert demo.resolved(run("render", "--indoors", old)).indoors == new
        assert run_folder(demo.resolved(run("render", "--indoors", old))).name == (
            f"works_indoors-{new}_binaural"
        )


def test_the_works_demo_hears_the_works_and_their_rumble_indoors(tmp_path, monkeypatch) -> None:
    """--indoors, mono: at the pavement and the balcony, 2 and 1 m in front of the
    houses and the offices, the works' sound comes through the window as much under
    its sound outdoors as their levels 2 m in front of the wall through the façade say
    (EN ISO 12354-3; the balcony's simulated at the point 1 m farther out, a scene of
    its own, the pavement's its own), within 2 dB (their tones, the alarm's 1.2 kHz
    beeps and the engines' firing, fall where they fall in the octave bands, near a
    crossover between two bands' gains: 0.06 to 1.6 dB measured): it is the listener's
    own sound, each band changed by the façade's level difference and by as much as the
    works are louder 2 m out (to 1e-4 in energy), and the stills give the rooms' levels
    from there (L_1,2m - D_2m,nT); the rumble has the energy its level gives in the
    engine's units (to 0.05 dB), the breaker's and the dumper's vibration at their
    distances from the building, their energies added, no adjustment for the building,
    50 dB under in dB(A); the two together are the sum of their energies. The garden, in
    front of no wall, hears them outdoors. What is felt at each is the ground's
    vibration, a shaker track. Another --indoors mixes again, in its run's folder."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene, vibration
    from acoustic_spaces.demos import audio, indoors, pipeline, scene_demo, scene_pictures
    from acoustic_spaces.facade import HOUSE

    folder = _street(tmp_path)
    demo = _works_demo(folder)
    assert demo.option("indoors").choices == indoors.choices(indoors.AIRBORNE)
    common = ["--seconds", "4", "--no-mp4", "--format", "mono"]

    def resolved(*more: str):
        return demo.resolved(demo.parser().parse_args(["render", *common, *more]))

    def held(*more: str) -> Path:
        """The folder of the mixes of the run with ``common`` and ``more``."""
        return demo.run_folder(demo.parser().parse_args(["render", *common, *more])) / "seats"

    def mixes(*more: str) -> tuple[dict, dict[str, np.ndarray]]:
        seats = pipeline.read_json(held(*more) / pipeline.SEATS)
        return seats, {
            key: audio.read(held(*more) / f"{key}.mono.wav")[0][:, 0] / seats["gain"]
            for key in seats["seats"]
        }

    def a_energy(x: np.ndarray) -> float:
        """The A-weighted sum of squares of ``x``, by its spectrum."""
        f = np.fft.rfftfreq(len(x), 1.0 / RATE)
        power = np.abs(np.fft.rfft(x) * audio.a_gain(f)) ** 2
        power[1 : (len(x) + 1) // 2] *= 2.0
        return float(np.sum(power)) / len(x)

    assert demo.main(["all", *common]) == 0
    _, outdoors = mixes()
    heard = {}
    for choice in ("airborne", "rumble", "airborne_and_rumble"):
        assert demo.main(["all", *common, "--indoors", choice]) == 0
        heard[choice] = mixes("--indoors", choice)
    args = resolved("--indoors", "airborne")
    scene = Scene.from_yaml(scene_demo.run_scene_file(args))
    sources = {s.name: s for s in scene.sources}
    loudest = max(sources[name].sound_power_db for name in WORKS)
    report = pipeline.read_json(scene_demo.results_of(args) / pipeline.PARAMETERS)
    front_file = scene_demo.front_scene_file(args)
    assert front_file.name == f"{scene_demo.run_scene_file(args).stem}_2m.yaml"
    out = pipeline.read_json(scene_demo.results_at(args, front_file) / pipeline.PARAMETERS)
    (front,) = Scene.from_yaml(front_file).listeners
    assert front.name == "balcony_2m"
    assert front.position == pytest.approx((5.0, -4.0, 10.0), abs=1e-12)
    assert {r["listener"] for r in out["responses"]} == {"balcony_2m"}
    fronts = indoors.facings(scene, [lis.name for lis in scene.listeners if lis.in_use])
    assert {k: v and v.name for k, v in fronts.items()} == {
        "pavement": "houses",
        "balcony": "offices",
        "garden": None,
    }

    def vibration_at(away) -> float:
        """The breaker's and the dumper's vibration (VdB), ``away`` their distance."""
        return 10.0 * math.log10(
            sum(
                10.0 ** (vibration.plant_level(row, float(away(sources[s]))) / 10.0)
                for s, row in (("breaker", "hoe_ram"), ("dumper", "loaded_trucks"))
            )
        )

    felt = math.nan
    for listener in (lis for lis in scene.listeners if lis.in_use):
        name, here = listener.name, np.asarray(listener.position[:2])
        facing = fronts[name]
        if facing is None:
            for choice, (seats, mixed) in heard.items():
                assert "post" not in seats["seats"][name], choice
                assert a_energy(mixed[name]) == pytest.approx(a_energy(outdoors[name]), rel=1e-6)
            felt = vibration_at(lambda s, h=here: np.hypot(*(np.asarray(s.position[:2]) - h)))
            continue
        levels = [r for r in report["responses"] if r["listener"] == name and r["source"] in WORKS]
        outside = sum(10.0 ** (a_weighted(r["spl_db"]) / 10.0) for r in levels)
        there = [
            r for r in out["responses"] if r["listener"] == f"{name}_2m" and r["source"] in WORKS
        ] or levels
        assert len(there) == len(levels) == 4
        inside = sum(
            10.0 ** (a_weighted(np.add(r["spl_db"], HOUSE.indoor_db())) / 10.0) for r in there
        )
        told = 10.0 * math.log10(outside / inside)
        window = heard["airborne"][1][name]
        drop = 10.0 * math.log10(a_energy(outdoors[name]) / a_energy(window))
        assert drop == pytest.approx(told, abs=2.0), name
        # The listener's own sound, each octave band changed by the façade's level
        # difference and by as much as the works, their energies added, are louder 2 m
        # in front of the wall than at the listener: the balcony's 1 m farther out.
        totals = [
            10.0 * np.log10(np.sum([10.0 ** (np.asarray(r["spl_db"]) / 10.0) for r in rows], 0))
            for rows in (there, levels)
        ]
        shift = totals[0] - totals[1]
        assert bool(np.any(np.abs(shift) > 0.5)) == (name == "balcony"), (name, shift)
        house = indoors.House(airborne=True, rumble=False)
        through = house.heard(outdoors[name][:, None], RATE, shift)[:, 0]
        assert a_energy(window) == pytest.approx(a_energy(through), rel=1e-4), name
        vdb = vibration_at(lambda s, f=facing: f.distances([s.position])[0])
        seat = heard["airborne_and_rumble"][0]["seats"][name]
        assert seat["vibration_vdb"] == pytest.approx(vdb, abs=0.01)
        assert seat["rumble_a_db"] == pytest.approx(vdb - 50.0, abs=0.01)
        rumble = heard["rumble"][1][name]
        fade = int(indoors.FADE_S * RATE)
        share = (
            len(rumble)
            - 2 * fade
            + 2 * float(np.sum(np.sin(np.linspace(0.0, np.pi / 2, fade)) ** 4))
        ) / len(rumble)
        wanted = float(indoors.mean_square(10.0 ** (seat["rumble_a_db"] / 10.0), loudest)) * share
        assert 10.0 * math.log10(a_energy(rumble) / len(rumble) / wanted) == pytest.approx(
            0.0, abs=0.05
        )
        both = heard["airborne_and_rumble"][1][name]
        assert a_energy(both) == pytest.approx(a_energy(window) + a_energy(rumble), rel=0.02)
    # The stills: the level in the house, the works through the window and the rumble.
    drawn = []
    real = scene_pictures.Pictures

    def pictures(*args, **kwargs):
        drawn.append(real(*args, **kwargs))
        return drawn[-1]

    monkeypatch.setattr(scene_pictures, "Pictures", pictures)
    # Each --indoors mixes in its run's folder: none over another's.
    assert len({held(), *(held("--indoors", choice) for choice in heard)}) == 4
    assert demo.main(["video", *common, "--indoors", "airborne_and_rumble"]) == 0
    lines = [line if isinstance(line, str) else line[0] for line in drawn[-1].lines(0)]
    first = drawn[-1].stops[0].key
    seat = heard["airborne_and_rumble"][0]["seats"][first]
    inside = sum(
        10.0 ** (a_weighted(np.add(r["spl_db"], HOUSE.indoor_db())) / 10.0)
        for r in report["responses"]
        if r["listener"] == first and r["source"] in WORKS
    )
    level = 10.0 * math.log10(inside + 10.0 ** (seat["rumble_a_db"] / 10.0))
    assert first == "pavement" and f"{level:.0f} dB(A) here, as simulated" in lines[0]
    # The rooms' levels the stills give: each source's 2 m in front of the wall, less the
    # façade's level difference; the balcony's from the point 1 m farther out.
    given = {(r["listener"], r["source"]): r["spl_db"] for r in report["responses"]}
    given |= {(r["listener"], r["source"]): r["spl_db"] for r in out["responses"]}
    for entry in drawn[-1].report["responses"]:
        name = entry["listener"]
        if fronts[name] is not None and entry["source"] in WORKS:
            at = indoors.front_name(name) if name == "balcony" else name
            room = np.add(given[(at, entry["source"])], HOUSE.indoor_db())
            np.testing.assert_allclose(entry["spl_db"], room, atol=0.006, err_msg=name)
    shaking = indoors.shaking_line(seat["vibration_vdb"], True, seat["rumble_a_db"])
    assert shaking in lines
    assert "Inside the houses here, through a closed window (EN ISO 12354-3)" in lines
    told = drawn[-1].indoors["vibration_vdb"]["garden"]
    assert told == pytest.approx(felt, abs=0.01) and "garden" not in drawn[-1].indoors["about"]
    # What is felt: a shaker track at each listener, and along the tour.
    seats = heard["airborne_and_rumble"][0]
    assert seats["shaker"]["hz"] == 25.0
    shaken = held("--indoors", "airborne_and_rumble")
    assert all((shaken / f"{key}.shaker.wav").exists() for key in seats["seats"])
    assert (shaken.parent.with_name(shaken.parent.name + "_shaker.wav")).exists()
    # Outdoors again: the mixes made again, without the house.
    assert demo.main(["render", *common, "--again"]) == 0
    seats, again = mixes()
    assert all("post" not in seat for seat in seats["seats"].values())
    for name, x in again.items():
        assert a_energy(x) == pytest.approx(a_energy(outdoors[name]), rel=1e-6)


def test_the_rumble_indoors_needs_something_that_shakes_the_ground(tmp_path) -> None:
    """With the breaker and the dumper left out, nothing shakes the ground: --indoors
    rumble says so before anything runs; --indoors airborne hears the rest through the
    window once the level 2 m in front of the offices, 1 m out from the balcony, is
    simulated, and says to simulate it first; without the balcony at once. A breaker
    standing in a run where a tram moves past is refused the rumble too."""
    import yaml

    from acoustic_spaces.demos import scene_demo

    folder = _street(tmp_path)
    demo = _works_demo(folder)
    pick = ["--sources", "generator,dumper_alarm"]
    args = demo.parser().parse_args(["render", *pick, "--indoors", "rumble"])
    assert "nothing that plays shakes it (generator, dumper_alarm;" in demo.check(args)
    args = demo.parser().parse_args(["render", *pick, "--indoors", "airborne"])
    assert demo.check(args) is None
    with pytest.raises(SystemExit, match="no simulation in"):
        scene_demo.still_houses(demo.resolved(args), demo)
    args = demo.parser().parse_args(["render", "--sources", "tram,breaker", "--indoors", "rumble"])
    assert "but breaker stands and shakes the ground too" in demo.check(args)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["listeners"] = [each for each in data["listeners"] if each["name"] != "balcony"]
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    args = demo.parser().parse_args(["render", *pick, "--indoors", "airborne"])
    houses = scene_demo.still_houses(demo.resolved(args), demo)
    assert houses is not None and houses.plant == () and not houses.fronts


def test_the_works_demo_plays_the_machines_picked(tmp_path) -> None:
    """``--sources``: every machine and the alarm play by default, under the run's name
    of the script; a pick plays those alone, named after them: the stills' line credits
    them alone, the houses hear them alone (their loudest the reference of the rumble),
    and a pick with nothing that shakes the ground is refused the rumble indoors before
    anything runs, in each part of a run in turn too."""
    from acoustic_spaces.demos import run_folder, scene_demo

    folder = _street(tmp_path)
    demo = _works_demo(folder)

    def run(*argv: str) -> tuple[argparse.Namespace, argparse.Namespace]:
        args = demo.parser().parse_args(["render", *argv])
        return args, demo.resolved(args)

    args, resolved = run()
    assert [(s.name, s.group, s.on) for s in demo.source_list(resolved) if s.name in WORKS] == [
        (name, "Road works", True) for name in WORKS
    ]
    assert resolved.playing == WORKS and demo.check(args) is None
    assert run_folder(resolved).name == "works_binaural"
    args, resolved = run("--sources", "generator,dumper_alarm")
    assert run_folder(resolved).name == "works_generator+dumper_alarm_binaural"
    assert resolved.playing == ("generator", "dumper_alarm") and demo.check(args) is None
    plays = demo.plays(resolved)
    assert {k for k, v in plays.items() if v != "silent"} == {"generator", "dumper_alarm"}
    assert plays["generator"] == (
        "a 6 kVA diesel generator, made to Defra's sound power (its engine firing at "
        f"{plant.GENERATOR.rpm:g} rpm)"
    )
    alarm = {s.name: s for s in scene_demo.scene_of(resolved).sources}["dumper_alarm"]
    assert f"{plant.ALARM_HZ:g} Hz beeps" in scene_demo.models.made_credit(alarm)
    # The rumble indoors needs a machine that shakes the ground among those picked.
    args, _ = run("--sources", "generator,dumper_alarm", "--indoors", "rumble")
    assert demo.check(args).startswith(
        "--indoors rumble hears the rumble of the ground's vibration, but nothing that "
        "plays shakes it (generator, dumper_alarm;"
    )
    args, _ = run("--sources", "generator,dumper", "--indoors", "rumble")
    assert demo.check(args) is None
    args, _ = run("--sources-play", "in-turn", "--indoors", "airborne_and_rumble")
    assert [run_folder(part).name for part in demo.each(args)] == [
        f"works_{name}_indoors-airborne_and_rumble_binaural" for name in WORKS
    ]
    assert demo.check(args).startswith(
        "works_generator_indoors-airborne_and_rumble_binaural: --indoors airborne_and_rumble "
        "hears the rumble"
    )
    # The houses hear the machines picked alone: outdoors, nothing then shakes the
    # ground unless the breaker or the dumper plays.
    assert scene_demo.still_houses(run("--sources", "generator")[1]) is None
    heard = scene_demo.still_houses(run("--sources", "generator,breaker")[1])
    assert [p.source for p in heard.plant] == ["breaker"]
    assert heard.loudest_db == pytest.approx(max(_power(n) for n in ("generator", "breaker")))


def _power(name: str) -> float:
    """The sound power of a source of the works (dB re 1 pW), all bands together."""
    machine = plant.MACHINES.get(name)
    level = plant.alarm_power(48_000) if machine is None else machine.sound_power()
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in level))
