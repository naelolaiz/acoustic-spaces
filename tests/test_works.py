"""The road works of the street example (:mod:`acoustic_spaces.demos.plant`,
:mod:`acoustic_spaces.demos.works`): Defra's rows, their sound power, the EU's limits,
the reversing alarm, the made sounds and the demo that plays them."""

import math
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces.demos import plant, works
from acoustic_spaces.examples import example_path
from acoustic_spaces.materials.bands import OCTAVE_BANDS_HZ
from acoustic_spaces.parameters import A_WEIGHTING_DB, a_weighted

RATE = 16_000


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
    """The street's scene holds each source of the works at the sound power the code
    gives it, at the scene's rate, where the demo says, left out until ticked."""
    from acoustic_spaces import Scene

    scene = Scene.from_yaml(example_path("street"))
    rate = int(scene.simulation.sample_rate)
    found = {s.name: s for s in scene.sources}
    for name in works.SOURCES:
        source = found[name]
        assert not source.enabled
        assert list(source.level_db) == works.level_of(name, rate)
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


def test_the_works_demo_plays_each_machine_at_its_sound_power(tmp_path, monkeypatch) -> None:
    """works_demo.py on the quick street: the scene with the works alone, at the sound
    power of the scene's rate, a feed per source as long as the tour at the made sounds'
    RMS, the mixes and the stills; at every listener each source's sound in the mix is as
    far under the breaker's as its simulated level is (within 1.5 dB), not as loud as
    it; each as its response is at the frequencies it holds (the alarm's tone, an
    engine's firing), not on average over each octave."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene, render
    from acoustic_spaces.demos import audio, load, pipeline
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
    demo = load(folder / "works_demo.py").at(folder)
    run = ["all", "--seconds", "4", "--no-mp4"]
    args = demo.resolved(demo.parser().parse_args(run))
    assert demo.check(args) is None
    assert set(demo.plays(args)) == set(works.SOURCES)
    assert demo.main(run) == 0
    scene = Scene.from_yaml(folder / "scene_works.yaml")
    assert [s.name for s in scene.sources] == list(works.SOURCES)
    assert all(s.in_use for s in scene.sources)
    for source in scene.sources:
        assert list(source.level_db) == works.level_of(source.name, RATE)
    work = folder / "demo_works"
    table = pipeline.read_json(work / "feeds" / "feeds.json")
    assert table["length_s"] == pytest.approx(12.0)
    for name in works.SOURCES:
        x, rate = audio.read(work / "feeds" / f"{name}.wav")
        assert rate == RATE and len(x) == 12 * RATE
        assert float(np.sqrt(np.mean(x**2))) == pytest.approx(plant.SOUND_RMS, rel=0.01)
    assert (work / "stills" / "00_pavement.png").exists()
    report = pipeline.read_json(work / "results" / pipeline.PARAMETERS)
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
        ir, _ = audio.read(work / "results" / f"{name}__{listener}.binaural.wav")
        x, _ = audio.read(work / "feeds" / f"{name}.wav")
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
    count = len(works.SOURCES)
    assert len(heard) == count * len(listeners)
    for k, listener in enumerate(listeners):
        wet = heard[k * count : (k + 1) * count]
        first = a_energy(wet[0])
        for name, x in zip(works.SOURCES[1:], wet[1:], strict=True):
            under = 10.0 * math.log10(first / a_energy(x))
            told = levels[("breaker", listener)] - levels[(name, listener)]
            told += fine_db("breaker", listener) - fine_db(name, listener)
            assert under == pytest.approx(told, abs=1.5), (listener, name)
    # Another rate of blows makes the feeds again, not the scene.
    again = demo.resolved(demo.parser().parse_args([*run, "--blows", "320"]))
    assert demo.is_done(demo.step("scene"), again)
    assert not demo.is_done(demo.step("feeds"), again)


def test_the_works_demo_says_what_is_wrong(tmp_path) -> None:
    """A rate of blows that is no rate, an input for a source the works do not have,
    and a listener the street does not have; the listeners it can play are the street's
    in use, and its earlier --tour still picks them."""
    from acoustic_spaces.demos import load

    folder = _street(tmp_path)
    demo = load(folder / "works_demo.py").at(folder)
    args = demo.resolved(demo.parser().parse_args(["feeds", "--blows", "0"]))
    assert "--blows is a positive number" in demo.check(args)
    args = demo.resolved(demo.parser().parse_args(["feeds", "--input", "car=x.wav"]))
    assert "--input car: no such source plays here" in demo.check(args)
    args = demo.resolved(demo.parser().parse_args(["all"]))
    assert demo.listeners(args) == ["pavement", "balcony", "garden"]
    args = demo.resolved(demo.parser().parse_args(["all", "--listeners", "garden,roof"]))
    assert "roof" in demo.check(args)
    args = demo.resolved(demo.parser().parse_args(["all", "--tour", "garden,pavement"]))
    assert demo.check(args) is None and args.listeners == "garden,pavement"


def test_the_works_demo_hears_the_works_and_their_rumble_indoors(tmp_path, monkeypatch) -> None:
    """--indoors, mono: in a house at each listener the works' sound comes through the
    window as much under its sound outdoors as their levels through the façade say,
    within 2 dB (their tones, the alarm's 1.2 kHz beeps and the engines' firing, fall
    where they fall in the octave bands, near a crossover between two bands' gains:
    0.06 to 1.6 dB measured); the rumble has the energy its level gives in the engine's
    units (to 0.05 dB), the breaker's and the dumper's vibration at their distances,
    their energies added, no adjustment for the building, 50 dB under in dB(A); the
    two together are the sum of their energies. Another --indoors mixes again, and the
    video refuses mixes made for another."""
    pytest.importorskip("PIL")
    from acoustic_spaces import Scene, vibration
    from acoustic_spaces.demos import audio, indoors, load, pipeline, scene_pictures
    from acoustic_spaces.facade import HOUSE

    folder = _street(tmp_path)
    demo = load(folder / "works_demo.py").at(folder)
    assert demo.option("indoors").choices == indoors.choices("works")
    common = ["--seconds", "4", "--no-mp4", "--format", "mono"]
    work = folder / "demo_works"

    def mixes() -> tuple[dict, dict[str, np.ndarray]]:
        seats = pipeline.read_json(work / "seats" / pipeline.SEATS)
        return seats, {
            key: audio.read(work / "seats" / f"{key}.mono.wav")[0][:, 0] / seats["gain"]
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
    for choice in ("works", "rumble", "works_and_rumble"):
        assert demo.main(["render", *common, "--indoors", choice]) == 0
        heard[choice] = mixes()
    scene = Scene.from_yaml(folder / "scene_works.yaml")
    loudest = max(s.sound_power_db for s in scene.sources if s.in_use)
    sources = {s.name: s for s in scene.sources}
    report = pipeline.read_json(work / "results" / pipeline.PARAMETERS)
    for listener in scene.listeners:
        name, here = listener.name, np.asarray(listener.position[:2])
        levels = [r for r in report["responses"] if r["listener"] == name]
        outside = sum(10.0 ** (a_weighted(r["spl_db"]) / 10.0) for r in levels)
        inside = sum(
            10.0 ** (a_weighted(np.add(r["spl_db"], HOUSE.indoor_db())) / 10.0) for r in levels
        )
        told = 10.0 * math.log10(outside / inside)
        window = heard["works"][1][name]
        drop = 10.0 * math.log10(a_energy(outdoors[name]) / a_energy(window))
        assert drop == pytest.approx(told, abs=2.0), name
        shaking = sum(
            10.0
            ** (
                vibration.plant_level(
                    row, float(np.hypot(*(np.asarray(sources[s].position[:2]) - here)))
                )
                / 10.0
            )
            for s, row in (("breaker", "hoe_ram"), ("dumper", "loaded_trucks"))
        )
        vdb = 10.0 * math.log10(shaking)
        seat = heard["works_and_rumble"][0]["seats"][name]
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
        both = heard["works_and_rumble"][1][name]
        assert a_energy(both) == pytest.approx(a_energy(window) + a_energy(rumble), rel=0.02)
    # The stills: the level in the house, the works through the window and the rumble.
    drawn = []
    real = scene_pictures.Pictures

    def pictures(*args, **kwargs):
        drawn.append(real(*args, **kwargs))
        return drawn[-1]

    monkeypatch.setattr(scene_pictures, "Pictures", pictures)
    with pytest.raises(SystemExit, match="another --indoors"):
        demo.main(["video", *common, "--indoors", "works"])
    assert demo.main(["video", *common, "--indoors", "works_and_rumble"]) == 0
    lines = [line if isinstance(line, str) else line[0] for line in drawn[-1].lines(0)]
    first = drawn[-1].stops[0].key
    seat = heard["works_and_rumble"][0]["seats"][first]
    inside = sum(
        10.0 ** (a_weighted(np.add(r["spl_db"], HOUSE.indoor_db())) / 10.0)
        for r in report["responses"]
        if r["listener"] == first
    )
    level = 10.0 * math.log10(inside + 10.0 ** (seat["rumble_a_db"] / 10.0))
    assert lines[0] == f"{level:.0f} dB(A) here, as simulated"
    assert f"Ground shaking {seat['vibration_vdb']:.0f} VdB, its rumble" in "\n".join(lines)
    assert "In a house here, through its closed window (EN ISO 12354-3)" in lines
    # Outdoors again: the mixes made again, without the house.
    assert demo.main(["render", *common]) == 0
    seats, again = mixes()
    assert all("post" not in seat for seat in seats["seats"].values())
    for name, x in again.items():
        assert a_energy(x) == pytest.approx(a_energy(outdoors[name]), rel=1e-6)
