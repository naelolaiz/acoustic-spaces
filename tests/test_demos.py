"""The demos' shared package: how a demo describes its steps and options, the audio it
measures, the recordings it fetches, and the pipeline from the mixes to the video."""

from __future__ import annotations

import argparse
import os
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from acoustic_spaces import fetch
from acoustic_spaces.audio import read_wav, write_wav
from acoustic_spaces.demos import (
    SIMULATE,
    Demo,
    Option,
    Step,
    audio,
    common_options,
    demo_for,
    list_demos,
    music,
    pipeline,
)

RATE = 8000


def _tone(seconds: float, hz: float = 440.0, rate: int = RATE, gain: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return gain * np.sin(2 * np.pi * hz * t)


# --------------------------------------------------------------------------- the audio


def test_levels_are_gated_and_weighted() -> None:
    """A tone with a long silence after it keeps its level, as a meter's gate leaves the
    silence out; the A weighting drops the bass far more than the K weighting, which is
    why the tours match their stops by loudness."""
    loud = np.concatenate([_tone(2.0), np.zeros(int(8.0 * RATE))])
    assert audio.level_db(loud, RATE) == pytest.approx(audio.level_db(_tone(2.0), RATE), abs=0.5)
    assert audio.db(audio.gated_rms(_tone(2.0, 1000.0), RATE)) == pytest.approx(-9.0, abs=0.5)
    # At 1 kHz both weightings pass the signal; at 50 Hz the A weighting takes 30 dB and
    # the K weighting 4 (its high-pass at 38 Hz), which is why far places keep their bass.
    for weighting, at_50 in ((audio.a_weighted, -30.0), (audio.k_weighted, -4.0)):
        for hz, want in ((1000.0, 0.0), (50.0, at_50)):
            x = _tone(2.0, hz)
            got = audio.db(audio.gated_rms(weighting(x, RATE), RATE)) - audio.db(
                audio.gated_rms(x, RATE)
            )
            assert got == pytest.approx(want, abs=1.5), (weighting.__name__, hz)
    quiet = np.concatenate([_tone(1.0), _tone(1.0, gain=0.01)])
    assert audio.loudness_db(quiet, RATE) > audio.level_db(quiet, RATE) - 6.0


def test_lag_and_shift_line_two_takes_up() -> None:
    """``lag_of`` finds how far a signal runs behind another and ``shifted`` moves it
    back, which is how the tours bring a mix forward by its arrival."""
    x = np.concatenate([np.zeros(1000), _tone(0.5), np.zeros(1000)])
    late = audio.shifted(x, -400, len(x))  # 400 samples later
    assert audio.lag_of(late, x, RATE, 1.0) == 400
    back = audio.shifted(late, 400, len(x))  # and brought forward again
    assert audio.lag_of(back, x, RATE, 1.0) == 0
    assert np.allclose(back[:-400], x[:-400])
    stereo = audio.stereo(x)
    assert stereo.shape == (len(x), 2) and np.allclose(stereo[:, 0], stereo[:, 1])
    assert audio.stereo(np.zeros((10, 4))).shape == (10, 2)
    assert audio.mono(np.ones((10, 2))).shape == (10,)


def test_quiet_moment_finds_the_gap_near_a_time() -> None:
    """A cut looks for the quietest moment near where it falls, so that it lands between
    phrases."""
    loud = np.concatenate([_tone(6.0), _tone(2.0, gain=0.001), _tone(6.0)])
    env = audio.envelope(loud, RATE)
    assert 6.0 <= audio.quiet_moment(env, 0.1, 7.0, 3.0) <= 8.5
    assert audio.quiet_moment(env, 0.1, 2.0, 0.0) == pytest.approx(2.0, abs=0.3)
    assert audio.mmss(0.0) == "0:00" and audio.mmss(125.3) == "2:05"


def test_resampled_keeps_the_tone(tmp_path: Path) -> None:
    up = audio.resampled(_tone(0.5, 440.0), RATE, 2 * RATE)
    assert len(up) == pytest.approx(RATE, rel=0.01)
    assert audio.resampled(up, RATE, RATE) is up
    audio.write(tmp_path / "a" / "tone.wav", _tone(0.2), RATE)
    x, rate = audio.read(tmp_path / "a" / "tone.wav")
    assert rate == RATE and len(x) == int(0.2 * RATE)
    assert audio.decoded(tmp_path / "a" / "tone.wav")[1] == RATE


# --------------------------------------------------------------------------- downloads


def test_fetch_takes_single_members_out_of_archives(tmp_path: Path) -> None:
    """A demo fetches the files it needs and nothing else: one member of a zip archive,
    the members it wants out of a tar read as it comes, or a whole file."""
    wanted = tmp_path / "wanted.txt"
    wanted.write_text("a take", encoding="utf-8")
    other = tmp_path / "other.txt"
    other.write_text("not this one", encoding="utf-8")
    zip_path = tmp_path / "takes.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(wanted, "takes/wanted.txt")
        archive.write(other, "takes/other.txt")
    with zipfile.ZipFile(zip_path) as archive:
        out = fetch.copy_member(archive, "takes/wanted.txt", tmp_path / "out" / "take.txt")
    assert out.read_text(encoding="utf-8") == "a take"
    assert not list((tmp_path / "out").glob("*.part"))
    tar_path = tmp_path / "takes.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(other, "takes/other.txt")
        archive.add(wanted, "takes/wanted.txt")
    read: list[int] = []
    got = fetch.extract_tar(
        tar_path.as_uri(),
        {"takes/wanted.txt": tmp_path / "from_tar.txt"},
        lambda done, total: read.append(done),
    )
    assert got["takes/wanted.txt"].read_text(encoding="utf-8") == "a take"
    assert read and read[-1] > 0
    with pytest.raises(OSError, match="no takes/missing.txt"):
        fetch.extract_tar(tar_path.as_uri(), {"takes/missing.txt": tmp_path / "no.txt"})
    assert fetch.download(wanted.as_uri(), tmp_path / "whole.txt").read_text("utf-8") == "a take"
    assert fetch.fetch(wanted.as_uri(), tmp_path / "again.txt").exists()


def test_music_fetches_tracks_once_and_sums_them(tmp_path: Path) -> None:
    """A track is the sum of its files, fetched once: a second run writes nothing, and
    the files it downloaded stay for it."""
    left, right = tmp_path / "left.wav", tmp_path / "right.wav"
    write_wav(left, _tone(1.0, 440.0), RATE)
    write_wav(right, _tone(1.0, 660.0), RATE)
    folder = tmp_path / "music"
    tracks = {"master": [(left.as_uri(), None), (right.as_uri(), None)]}
    paths = music.write_tracks(tracks, folder, rate=2 * RATE, channels=2)
    x, rate = read_wav(paths["master"])
    assert rate == 2 * RATE and x.shape[1] == 2
    assert np.abs(x).max() == pytest.approx(0.9, abs=0.15)  # two tones of 0.5, not clipped
    written = paths["master"].stat().st_mtime_ns
    music.write_tracks(tracks, folder)
    assert paths["master"].stat().st_mtime_ns == written
    assert (folder / "left.wav").exists()  # kept, so a second run downloads nothing


def test_phrases_set_where_a_tour_cuts() -> None:
    """The stops of a tour end on the song's phrases (8 bars), none runs past the end of
    the music, and a stop with no time left is left out."""
    song = {"bpm": 120.0}
    phrase = music.phrase_s(song)
    assert phrase == pytest.approx(16.0)
    tour = music.phrase_tour([("a", 1), ("b", 2), ("c", 0)], 100.0, phrase, first_s=0.5)
    assert tour == [("a", 0.0, 16.5), ("b", 16.5, 48.5), ("c", 48.5, 100.0)]
    # The last stop takes what is left, or as many phrases as asked for.
    assert music.phrase_tour([("a", 1), ("b", 1)], 100.0, phrase)[-1] == ("b", 16.0, 100.0)
    assert music.phrase_tour([("a", 1), ("b", 1)], 100.0, phrase, last_phrases=2)[-1] == (
        "b",
        16.0,
        48.0,
    )
    short = music.phrase_tour([("a", 1), ("b", 1), ("c", 1)], 20.0, phrase)
    assert [key for key, _, _ in short] == ["a", "b"]


# --------------------------------------------------------------------------- a demo


def _demo(tmp_path: Path, **more) -> tuple[Demo, list[str]]:
    """A demo of three steps and one simulation, writing into ``tmp_path``."""
    ran: list[str] = []

    def step(name: str):
        def run(args) -> None:
            ran.append(name)

        return run

    scene = tmp_path / "scene.yaml"
    results = tmp_path / "results"
    demo = Demo(
        name="test",
        example="shoebox",
        title="A demo",
        description="The first line.\n\nThe rest of the docstring.",
        script=tmp_path / "test_demo.py",
        steps=[
            Step("scene", step("scene"), "write the scene", lambda args: args.scene.exists()),
            SIMULATE,
            Step("render", step("render"), "mix every listener"),
        ],
        options=[
            Option("scene", "the scene", Path("scene.yaml"), "path"),
            Option("results", "simulate's folder", Path("results"), "folder"),
            *common_options(),
        ],
        simulations=lambda args: [pipeline.Simulation(args.scene, args.results, args.format)],
        scene_from="scene",
        **more,
    )
    scene.parent.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return demo, ran


def test_a_demo_describes_its_steps_and_options(tmp_path: Path) -> None:
    """A demo's options make its parser, its defaults and the command line that runs a
    step; its description is the first paragraph of the script's docstring."""
    demo, _ran = _demo(tmp_path)
    assert demo.description == "The first line."
    args = demo.parser().parse_args(["render", "--hrtf", "me.sofa"])
    assert args.step == "render" and args.hrtf == Path("me.sofa")
    assert demo.defaults().format == "binaural" and demo.defaults().hrtf is None
    assert demo.command("render") == "python test_demo.py render"
    assert demo.command("render", args) == "python test_demo.py render --hrtf me.sofa"
    assert demo.command("render", demo.resolved(demo.defaults())) == "python test_demo.py render"
    assert demo.first_step() == "python test_demo.py scene"
    assert demo.scene_step().name == "scene"
    # Paths count from the demo's folder, wherever it is run from.
    resolved = demo.resolved(demo.defaults())
    assert resolved.scene == tmp_path / "scene.yaml"
    elsewhere = demo.at(tmp_path / "copy").resolved(demo.defaults())
    assert elsewhere.scene == tmp_path / "copy" / "scene.yaml"
    with pytest.raises(KeyError):
        demo.step("nothing")
    assert demo.option("hrtf").filter == "SOFA files (*.sofa)"
    assert "simulate" in [s.name for s in demo.steps]
    with pytest.raises(ValueError, match="listed twice"):
        Demo(
            name="twice",
            example="shoebox",
            title="",
            description="",
            script=tmp_path / "x.py",
            steps=[SIMULATE, SIMULATE],
            options=[],
        )
    with pytest.raises(ValueError, match="no option kind"):
        Option("x", "", None, "colour")


def test_a_demo_leaves_out_what_is_done_already(tmp_path: Path, capsys) -> None:
    """``all`` runs the steps in turn and leaves out the scene and the simulation when
    they are there; a step asked for on its own always runs."""
    demo, ran = _demo(tmp_path)
    args = demo.defaults()
    simulated = []
    demo.simulate = lambda args, only_missing=False: simulated.append(only_missing)
    demo.run([s.name for s in demo.steps], args, skip_done=True)
    assert ran == ["scene", "render"] and simulated == [True]
    (tmp_path / "scene.yaml").write_text("name: a scene\n", encoding="utf-8")
    ran.clear()
    demo.run([s.name for s in demo.steps], args, skip_done=True)
    assert ran == ["render"] and "done already" in capsys.readouterr().out
    ran.clear()
    demo.run(["scene"], args)
    assert ran == ["scene"]


def test_a_demo_refuses_an_hrtf_for_another_format(tmp_path: Path) -> None:
    demo, _ran = _demo(tmp_path)
    args = demo.defaults()
    args.hrtf, args.format = Path("me.sofa"), "ambix3"
    assert demo.check(args) == "--hrtf is for binaural mixes"
    with pytest.raises(SystemExit, match="binaural"):
        demo.run(["render"], args)
    with pytest.raises(SystemExit) as caught:
        demo.main(["render", "--hrtf", "me.sofa", "--format", "ambix3"])
    assert caught.value.code == 2  # argparse's own error


def test_a_demo_stops_when_the_scene_is_missing(tmp_path: Path) -> None:
    """The simulation needs the scene: without it the demo says which step writes it."""
    demo, _ran = _demo(tmp_path)
    args = demo.resolved(demo.defaults())
    with pytest.raises(SystemExit, match="test_demo.py scene"):
        demo.simulate(args)


def test_the_examples_demos_are_listed() -> None:
    """Every demo that comes with the examples describes itself, and each one belongs to
    an example that is there."""
    demos = list_demos()
    assert {demo.name for demo in demos} >= {"church", "stadium", "walk"}
    for demo in demos:
        assert demo.title and demo.description and demo.steps
        assert demo.script.exists() and demo.script.parent.name == demo.example
        assert demo_for(demo.example).name == demo.name
        assert {o.name for o in demo.options} >= {"work", "hrtf", "no_mp4"}
        for option in demo.options:
            assert option.help and option.description
            assert all(name in {s.name for s in demo.steps} for name in option.steps)
        if demo.scene_from:
            assert demo.scene_step() is not None
    assert demo_for("shoebox") is None


# --------------------------------------------------------------------------- the mixes


def _results(folder: Path, listeners=("near", "far"), sources=("a", "b"), fmt="binaural") -> dict:
    """A simulation's folder: a response per source and listener, the far one later and
    source ``b`` later still, and the parameters.json that names them."""
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for listener in listeners:
        for source in sources:
            name = f"{source}__{listener}.wav"
            response = np.zeros((RATE, 2))
            delay = 0.1 if listener == "near" else 0.3
            response[int((delay + (0.05 if source == "b" else 0.0)) * RATE)] = 1.0
            write_wav(folder / name, response, RATE)
            entries.append(
                {
                    "source": source,
                    "listener": listener,
                    "outputs": [{"format": fmt, "wav": name}],
                }
            )
    pipeline.write_json(folder / pipeline.PARAMETERS, {"responses": entries})
    return {"responses": entries}


def test_arrival_weighs_the_sources_by_how_loud_they_play(tmp_path: Path) -> None:
    """When the sound arrives is the peak of the responses added up, each weighted by its
    source's level, so a loud late loudspeaker sets the beat and a quiet early one does
    not."""
    results = tmp_path / "results"
    entries = _results(results)["responses"]
    assert pipeline.arrival_s(results, entries, "near") == pytest.approx(0.1, abs=0.002)
    assert pipeline.arrival_s(results, entries, "far") == pytest.approx(0.3, abs=0.002)
    loud_b = {"a": 0.0, "b": 20.0}
    assert pipeline.arrival_s(results, entries, "near", loud_b) == pytest.approx(0.15, abs=0.002)
    only_a = {"a": 0.0}
    assert pipeline.arrival_s(results, entries, "near", only_a) == pytest.approx(0.1, abs=0.002)
    assert pipeline.response_rate(pipeline.Simulation(Path("s"), results), entries) == RATE


def _mixes(tmp_path: Path, monkeypatch, **more) -> tuple[list[pipeline.Mix], Path, list[str]]:
    """Two mixes of one simulation, auralized by a fake that writes a tone, the far one
    16 dB quieter."""
    results = tmp_path / "results"
    _results(results)
    feeds = {"a": tmp_path / "feeds" / "a.wav", "b": tmp_path / "feeds" / "b.wav"}
    for path in feeds.values():
        audio.write(path, _tone(1.0), RATE)
    simulation = pipeline.Simulation(tmp_path / "scene.yaml", results, "binaural")
    mixes = [
        pipeline.Mix(listener, simulation, listener, feeds, info={"where": listener}, **more)
        for listener in ("near", "far")
    ]
    made: list[str] = []

    def auralize(results, feeds, listener, label, out, equal_power=False) -> None:
        made.append(listener)
        gain = 0.5 if listener == "near" else 0.08
        audio.write(out, audio.stereo(_tone(2.0, gain=gain)), RATE)

    monkeypatch.setattr(pipeline, "auralize", auralize)
    return mixes, tmp_path / "seats", made


def test_the_mixes_keep_one_gain_and_are_measured(tmp_path: Path, monkeypatch) -> None:
    """Every stop is mixed with one gain for all (the loudest peak at -1 dBFS), measured
    in dB(A) and in loudness, and kept in seats.json with when its sound arrives."""
    mixes, folder, made = _mixes(tmp_path, monkeypatch)
    args = argparse.Namespace(format="binaural", hrtf=None, again=False)
    seats = pipeline.render_mixes(mixes, folder, args)
    assert made == ["near", "far"]
    assert seats["rate"] == RATE and seats["format"] == "binaural"
    peaks = [np.abs(audio.read(folder / f"{k}.binaural.wav")[0]).max() for k in seats["seats"]]
    peak = max(peaks)
    assert audio.db(peak) == pytest.approx(-1.0, abs=0.01)
    near, far = seats["seats"]["near"], seats["seats"]["far"]
    assert near["relative_db"] == 0.0 and far["relative_db"] == pytest.approx(-15.9, abs=0.2)
    assert near["lag_s"] == pytest.approx(0.1, abs=0.002)
    assert far["lag_s"] == pytest.approx(0.3, abs=0.002)
    assert near["length_s"] == 2.0 and near["where"] == "near"
    assert near["loudness_db"] < 0.0
    assert pipeline.seats_of(folder, "run render")["seats"] == seats["seats"]
    # A second run keeps the mixes and the gain; --again mixes them afresh.
    made.clear()
    again = pipeline.render_mixes(mixes, folder, args)
    assert made == [] and again["gain"] == pytest.approx(seats["gain"])
    assert again["seats"]["far"]["rms_db"] == pytest.approx(far["rms_db"], abs=0.01)
    pipeline.render_mixes(mixes, folder, argparse.Namespace(**vars(args) | {"again": True}))
    assert made == ["near", "far"]
    # A feed written after the mixes makes them stale, so that run mixes them again
    # (both play that feed), and a mix whose responses were simulated again as well.
    made.clear()
    path = folder / "near.binaural.wav"
    later = path.stat().st_mtime + 10
    os.utime(mixes[0].feeds["a"], (later, later))
    pipeline.render_mixes(mixes, folder, args)
    assert made == ["near", "far"]
    made.clear()
    older = folder.stat().st_mtime - 60
    os.utime(mixes[0].feeds["a"], (older, older))
    pipeline.render_mixes(mixes, folder, args)
    assert made == []
    os.utime(mixes[0].simulation.results / pipeline.PARAMETERS, (later + 20, later + 20))
    pipeline.render_mixes(mixes, folder, args)
    assert made == ["near", "far"]


def test_the_mixes_stop_when_the_simulation_has_no_responses(tmp_path: Path) -> None:
    args = argparse.Namespace(format="binaural", hrtf=None, again=False)
    simulation = pipeline.Simulation(tmp_path / "scene.yaml", tmp_path / "none", "binaural")
    assert not pipeline.simulated(simulation, args)
    with pytest.raises(SystemExit, match="run first"):
        pipeline.responses_of(simulation, args, "python demo.py scene")
    with pytest.raises(SystemExit, match="no mixes"):
        pipeline.seats_of(tmp_path / "none", "python demo.py render")
    seats = {"seats": {"near": {}}}
    pipeline.missing_mixes(seats, ["near"], "python demo.py render")
    with pytest.raises(SystemExit, match="no mix for far"):
        pipeline.missing_mixes(seats, ["near", "far"], "python demo.py render")


# --------------------------------------------------------------------------- the tour


def _seats(folder: Path, keys=("near", "far"), lag=(0.0, 0.2)) -> dict:
    info = {}
    for key, lag_s in zip(keys, lag, strict=True):
        audio.write(folder / f"{key}.binaural.wav", audio.stereo(_tone(10.0)), RATE)
        info[key] = {"lag_s": lag_s, "rms_db": -20.0, "loudness_db": -18.0, "length_s": 10.0}
    info[keys[1]]["loudness_db"] = -28.0
    return {"rate": RATE, "format": "binaural", "gain": 1.0, "seats": info}


def test_a_tour_crossfades_its_stops_and_ends_at_full_scale(tmp_path: Path) -> None:
    """The tour plays each stop in turn, crossfaded over the cut, each mix brought
    forward by its arrival so the music runs on, and the whole ends at -1 dBFS."""
    folder = tmp_path / "seats"
    seats = _seats(folder)
    stops = [
        pipeline.TourStop("near", 0.0, 0.0, 4.0),
        pipeline.TourStop("far", 4.0, 4.0, 8.0),
    ]
    assert stops[0].length == 4.0 and stops[1].end == 8.0
    out = pipeline.tour_audio(stops, folder, seats, crossfade_s=1.0, fade_out_s=1.0)
    assert out.shape == (8 * RATE, 2)
    assert audio.db(float(np.abs(out).max())) == pytest.approx(-1.0, abs=0.01)
    # Nothing drops out at the cut (equal power), and the end fades away.
    at_cut = float(np.abs(out[int(3.9 * RATE) : int(4.1 * RATE)]).max())
    before = float(np.abs(out[int(2.0 * RATE) : int(3.0 * RATE)]).max())
    assert at_cut > before * 0.9
    assert np.abs(out[-10:]).max() < 1e-3
    # The gains: every stop up to the loudest one's loudness, not its dB(A).
    gains = pipeline.gains_of(seats, ["near", "far"])
    assert gains == {"near": 0.0, "far": 10.0}
    louder = pipeline.tour_audio(
        [pipeline.TourStop("far", 0.0, 0.0, 4.0, gain_db=10.0)], folder, seats, fade_out_s=0.0
    )
    assert audio.db(float(np.abs(louder).max())) == pytest.approx(-1.0, abs=0.01)
    # Seats measured before loudness was kept fall back to their level.
    for seat in seats["seats"].values():
        del seat["loudness_db"]
    assert pipeline.gains_of(seats, ["near", "far"]) == {"near": 0.0, "far": 0.0}


def test_the_video_writes_the_stills_and_the_chapters(tmp_path: Path) -> None:
    """Without ffmpeg (``--no-mp4``) the video step still writes every still, the list
    the concat demuxer reads and the chapters."""
    pytest.importorskip("PIL")
    from PIL import Image

    drawn: list[int] = []

    def draw(k: int):
        drawn.append(k)
        return Image.new("RGB", (64, 36), (10, 10, 10))

    folder = tmp_path / "demo"
    audio.write(folder / "tour.wav", audio.stereo(_tone(1.0)), RATE)
    pipeline.video(
        folder,
        [("near", 2.0), ("far", 3.0)],
        draw,
        folder / "tour.wav",
        no_mp4=True,
        chapters=["0:00 Near", "0:02 Far"],
    )
    assert drawn == [0, 1]
    stills = sorted(p.name for p in (folder / "stills").glob("*.png"))
    assert stills == ["00_near.png", "01_far.png"]
    lines = (folder / "stills" / "stills.txt").read_text(encoding="utf-8").splitlines()
    assert lines[1] == "duration 2.000" and lines[3] == "duration 3.000"
    assert lines[-1] == lines[-3]  # the last picture again, as the demuxer wants
    assert (folder / "chapters.txt").read_text(encoding="utf-8").startswith("0:00 Near")
    # An animation draws by time instead, and writes the stills it is asked for.
    times: list[float] = []

    def frame(t: float):
        times.append(t)
        return Image.new("RGB", (64, 36), (10, 10, 10))

    pipeline.animation(
        tmp_path / "walk",
        3.0,
        frame,
        folder / "tour.wav",
        stills=[("setting-off", 0.5)],
        no_mp4=True,
    )
    assert times == [0.5]
    assert [p.name for p in (tmp_path / "walk" / "stills").glob("*.png")] == ["00_setting-off.png"]
