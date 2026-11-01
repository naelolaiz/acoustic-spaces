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
    demos_of,
    input_option,
    list_demos,
    listener_names,
    listeners_in,
    listeners_option,
    music,
    picked,
    pipeline,
    scripts_of,
    shell_line,
    sibling,
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


def test_decoded_reads_other_formats_through_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    """A file that is no WAV (a band's FLAC mix, say) is decoded by ffmpeg with its own
    channels, and refused with a reason where ffmpeg is not on the PATH."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    mix = np.stack([_tone(0.5, 440.0), 0.5 * _tone(0.5, 660.0)], axis=1)
    audio.write(tmp_path / "mix.wav", mix, RATE)
    flac = tmp_path / "mix.flac"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(tmp_path / "mix.wav"), str(flac)], check=True
    )
    x, rate = audio.decoded(flac)
    assert rate == RATE and x.shape == mix.shape
    np.testing.assert_allclose(x, mix, atol=1e-3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mix.flac", "mix.wav"]
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(SystemExit, match="ffmpeg is needed to decode mix.flac"):
        audio.decoded(flac)


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


def test_a_remote_block_cut_short_every_time_fails(monkeypatch) -> None:
    """A block of a remote archive that keeps arriving short is asked for again, and in
    the end the read fails rather than hand on a member with a hole in it."""
    asked: list[str] = []

    class Response:
        def __init__(self, headers: dict, body: bytes) -> None:
            self.headers, self.body = headers, body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def urlopen(url, headers=None, timeout=fetch.TIMEOUT):
        asked.append(headers["Range"])
        if headers["Range"] == "bytes=0-0":
            return Response({"Content-Range": "bytes 0-0/100"}, b"x")
        return Response({}, b"only part")

    monkeypatch.setattr(fetch, "urlopen", urlopen)
    remote = fetch.RangeFile("https://example.org/takes.zip")
    with pytest.raises(OSError, match="came 9 long, 5 times"):
        remote.read(10)
    assert asked.count("bytes=0-99") == fetch.RETRIES


def test_a_remote_file_is_read_in_blocks_of_the_size_asked(monkeypatch) -> None:
    """A remote archive is asked for in blocks of the size given (large ones read large
    members in fewer requests), and keeps 64 MiB of them, two blocks at the least."""
    data = bytes(range(256)) * 400
    asked: list[str] = []

    class Response:
        def __init__(self, headers: dict, body: bytes) -> None:
            self.headers, self.body = headers, body

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def read(self) -> bytes:
            return self.body

    def urlopen(url, headers=None, timeout=fetch.TIMEOUT):
        asked.append(headers["Range"])
        a, b = (int(x) for x in headers["Range"].removeprefix("bytes=").split("-"))
        return Response({"Content-Range": f"bytes {a}-{b}/{len(data)}"}, data[a : b + 1])

    monkeypatch.setattr(fetch, "urlopen", urlopen)
    remote = fetch.RangeFile("https://example.org/tracks.zip", 40_000)
    remote.seek(39_990)
    assert remote.read(20) == data[39_990:40_010]
    remote.seek(-10, 2)
    assert remote.read() == data[-10:]
    assert asked == ["bytes=0-0", "bytes=0-39999", "bytes=40000-79999", "bytes=80000-102399"]
    assert fetch.RangeFile("https://example.org/tracks.zip").kept == 16
    assert fetch.RangeFile("https://example.org/tracks.zip", 64 << 20).kept == 2


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
    # A file of your own in place of the files to fetch: decoded and written as the
    # others are, nothing downloaded, and read again each time, as it may have changed.
    mine = tmp_path / "mine.wav"
    write_wav(mine, _tone(0.5), RATE)
    paths = music.write_tracks({"own": mine}, folder, channels=1)
    x, rate = read_wav(paths["own"])
    assert rate == RATE and x.squeeze().ndim == 1 and len(x) == RATE // 2
    write_wav(mine, _tone(1.0), RATE)
    music.write_tracks({"own": mine}, folder)
    assert len(read_wav(paths["own"])[0]) == RATE


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


def test_a_tour_without_some_places_keeps_its_length_and_cuts() -> None:
    """A tour that leaves places out: each stop left out gives its time to the stop
    before it (those first, to the first one kept), stops at one place in a row become
    one, and the tour lasts as long as before."""
    tour = [("a", 0.0, 10.0), ("b", 10.0, 20.0), ("c", 20.0, 30.0), ("a", 30.0, 40.0)]
    assert pipeline.picked_tour(tour, lambda stop: True) == tour
    assert pipeline.picked_tour(tour, lambda stop: stop[0] != "b") == [
        ("a", 0.0, 20.0),
        ("c", 20.0, 30.0),
        ("a", 30.0, 40.0),
    ]
    assert pipeline.picked_tour(tour, lambda stop: stop[0] == "c") == [("c", 0.0, 40.0)]
    assert pipeline.picked_tour(tour, lambda stop: stop[0] == "a") == [("a", 0.0, 40.0)]
    assert pipeline.picked_tour(tour, lambda stop: False) == []
    # A place may be more than a name: a listener in a weather.
    weathers = [("x", "evening", 0.0, 5.0), ("x", "still", 5.0, 9.0), ("y", "evening", 9.0, 12)]
    assert pipeline.picked_tour(weathers, lambda stop: stop[0] == "x") == [
        ("x", "evening", 0.0, 5.0),
        ("x", "still", 5.0, 12),
    ]


def test_a_demo_plays_the_listeners_named(tmp_path: Path) -> None:
    """``--listeners``: the names in its order, each once, none when it is not given;
    the listeners a demo plays keep the demo's order; a name the demo cannot play, or
    none, is refused with the names it can, and the editor labels it Listeners."""
    assert listener_names(None) is None
    assert listener_names(" b, a,,b ") == ["b", "a"]
    assert listener_names("") == []
    every = ["a", "b", "c"]
    assert picked(argparse.Namespace(listeners=None), every) == every
    assert picked(argparse.Namespace(listeners="c,a"), every) == ["a", "c"]
    assert picked(argparse.Namespace(), every) == every
    demo, _ran = _demo(tmp_path, listeners=lambda args: listeners_in(args.scene))
    demo.options.append(listeners_option())
    option = demo.option("listeners")
    assert option.kind == "listeners" and option.label == "Listeners" and not option.advanced
    assert option.steps == ("render", "video") and option.choices == ()
    (tmp_path / "scene.yaml").write_text(
        "listeners:\n"
        "  - {name: front, position: [1, 1, 1]}\n"
        "  - {name: left_out, position: [2, 1, 1], enabled: false}\n"
        "  - {name: back, position: [3, 1, 1]}\n",
        encoding="utf-8",
    )
    assert listeners_in(tmp_path / "scene.yaml") == ["front", "back"]
    args = demo.parser().parse_args(["render", "--listeners", "back"])
    assert args.listeners == "back" and demo.check(args) is None
    assert demo.command("render", args) == "python test_demo.py render --listeners back"
    args.listeners = "back,left_out"
    assert demo.check(args) == "--listeners left_out: no such listener here; it has front, back"
    args.listeners = " , "
    assert demo.check(args) == "--listeners names no listener"
    # A scene the demo cannot read yet leaves the names to the steps.
    (tmp_path / "scene.yaml").unlink()
    args.listeners = "anyone"
    assert demo.check(args) is None


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


def test_a_command_line_is_quoted_for_the_system(monkeypatch) -> None:
    """The command lines the demos show are pasted into a shell: on Windows a path such
    as ``C:\\Users\\me\\ears.sofa`` stays bare and an argument with a space takes double
    quotes; everywhere else the quoting is sh's."""
    plain = ["simulate", "scene.yaml", "--out", "results"]
    monkeypatch.setattr("acoustic_spaces.demos.sys.platform", "win32")
    assert shell_line(plain) == "simulate scene.yaml --out results"
    assert shell_line(["--hrtf", r"C:\Users\me\ears.sofa"]) == r"--hrtf C:\Users\me\ears.sofa"
    assert shell_line(["--hrtf", r"C:\My Files\ears.sofa"]) == r'--hrtf "C:\My Files\ears.sofa"'
    monkeypatch.setattr("acoustic_spaces.demos.sys.platform", "linux")
    assert shell_line(plain) == "simulate scene.yaml --out results"
    assert shell_line(["--hrtf", "/home/me/my ears.sofa"]) == "--hrtf '/home/me/my ears.sofa'"
    assert shell_line([]) == ""


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


def test_a_demo_takes_a_file_for_each_name(tmp_path: Path) -> None:
    """A ``files`` option (``--stem STEM=WAV``) takes a file for each of the names it
    lists, once per name, the paths counted from the demo's folder; a name it does not
    list, or a file that is not there, is refused. A number typed where the demo takes
    one (the editor's field) has to be a number."""
    import dataclasses

    demo, _ran = _demo(tmp_path)
    stems = Option("stems", "a stem of the song", None, "files", ("drums", "vocal"), flag="--stem")
    assert stems.metavar == "STEM=WAV" and input_option().metavar == "SOURCE=WAV"
    demo = dataclasses.replace(
        demo, options=[*demo.options, stems, Option("bpm", "the tempo", None, "float")]
    )
    args = demo.parser().parse_args(["render", "--stem", "drums=d.wav", "--stem", "vocal=in/v.wav"])
    assert args.stems == {"drums": Path("d.wav"), "vocal": Path("in/v.wav")}
    resolved = demo.resolved(args)
    assert resolved.stems == {"drums": tmp_path / "d.wav", "vocal": tmp_path / "in" / "v.wav"}
    assert demo.arguments(resolved, names_only=True) == [
        "--stem",
        "drums=d.wav",
        "--stem",
        "vocal=v.wav",
    ]
    assert demo.check(args) == f"--stem drums: no {tmp_path / 'd.wav'}"
    audio.write(tmp_path / "d.wav", _tone(0.5), RATE)
    audio.write(tmp_path / "in" / "v.wav", _tone(0.5), RATE)
    assert demo.check(args) is None
    args.stems["keys"] = Path("d.wav")
    assert demo.check(args) == "--stem keys: not one of drums, vocal"
    with pytest.raises(SystemExit) as caught:
        demo.parser().parse_args(["render", "--stem", "drums"])
    assert caught.value.code == 2  # argparse's own error: STEM=WAV
    args = demo.parser().parse_args(["render", "--bpm", "120"])
    assert args.bpm == 120.0 and demo.check(args) is None
    args.bpm = "fast"  # as the editor's field may hold it
    assert demo.check(args) == "--bpm: a number, not 'fast'"
    with pytest.raises(ValueError, match="lists the names it takes"):
        Option("stems", "a stem", None, "files")


def test_a_demo_takes_a_file_for_each_source(tmp_path: Path) -> None:
    """``--input SOURCE=WAV``, once per source: the sources and the files they play
    instead, on the command line and back, the paths counted from the demo's folder; a
    source the demo does not feed, or a file that is not there, is refused."""
    import dataclasses

    def plays(args) -> dict[str, str]:
        return {"a": "the song, left", "b": "the song, right"}

    demo, _ran = _demo(tmp_path, plays=plays)
    demo = dataclasses.replace(demo, options=[*demo.options, input_option()])
    assert demo.option("inputs").kind == "inputs" and demo.defaults().inputs is None
    args = demo.parser().parse_args(["render", "--input", "a=x.wav", "--input", " b = in/y.wav"])
    assert args.inputs == {"a": Path("x.wav"), "b": Path("in/y.wav")}
    # Unresolved paths keep the system's separators; resolved relative paths use '/'.
    assert demo.command("render", args) == (
        f"python test_demo.py render --input a=x.wav --input b={Path('in/y.wav')}"
    )
    resolved = demo.resolved(args)
    assert resolved.inputs == {"a": tmp_path / "x.wav", "b": tmp_path / "in" / "y.wav"}
    assert demo.arguments(resolved, relative_to=tmp_path)[-2:] == [
        "--input",
        "b=in/y.wav",
    ]
    assert demo.arguments(resolved, names_only=True) == ["--input", "a=x.wav", "--input", "b=y.wav"]
    assert demo.check(args) == f"--input a: no {tmp_path / 'x.wav'}"
    audio.write(tmp_path / "x.wav", _tone(0.5), RATE)
    audio.write(tmp_path / "in" / "y.wav", _tone(0.5), RATE)
    assert demo.check(args) is None
    args.inputs["c"] = Path("x.wav")
    assert demo.check(args) == "--input c: no such source plays here; it has a, b"
    with pytest.raises(SystemExit, match="no such source"):
        demo.run(["render"], args)
    with pytest.raises(SystemExit) as caught:
        demo.parser().parse_args(["render", "--input", "a"])
    assert caught.value.code == 2  # argparse's own error: SOURCE=WAV


def test_a_demo_stops_when_the_scene_is_missing(tmp_path: Path) -> None:
    """The simulation needs the scene: without it the demo says which step writes it."""
    demo, _ran = _demo(tmp_path)
    args = demo.resolved(demo.defaults())
    with pytest.raises(SystemExit, match="test_demo.py scene"):
        demo.simulate(args)


def test_the_examples_demos_are_listed() -> None:
    """Every demo that comes with the examples describes itself, and each one belongs to
    an example that is there; an example's own demo comes first, before those that build
    on it (the helicopter over the stadium's concert)."""
    demos = list_demos()
    assert {demo.name for demo in demos} >= {"church", "stadium", "walk", "crossing"}
    for demo in demos:
        assert demo.title and demo.description and demo.steps
        assert demo.script.exists() and demo.script.parent.name == demo.example
        own = demos_of(demo.example)
        assert demo.name in [d.name for d in own] and own[0].builds_on is None
        assert demo_for(demo.example).name == own[0].name
        assert demo.builds_on in (None, own[0].name)
        assert {o.name for o in demo.options} >= {"work", "hrtf", "no_mp4"}
        for option in demo.options:
            assert option.help and option.description
            assert all(name in {s.name for s in demo.steps} for name in option.steps)
        if demo.scene_from:
            assert demo.scene_step() is not None
    assert demo_for("shoebox") is None


def test_an_examples_script_runs_as_a_demo_of_one_step(tmp_path: Path) -> None:
    """An example's scripts with a parser() (not its demos, nor what writes its model)
    run as demos of one step: their options from their own arguments, the scene they
    read given first without a flag, a file they write under its own flag."""
    scripts = {demo.name: demo for demo in scripts_of("stadium_concert")}
    assert set(scripts) == {"align", "feedback", "near_field"}
    feedback = scripts["feedback"]
    assert feedback.stepless and [s.name for s in feedback.steps] == ["run"]
    assert feedback.title == "Feedback of the singer's microphone"
    assert feedback.description.startswith("Feedback of the singer's microphone: how much")
    options = {o.name: o for o in feedback.options}
    scene, rays, out = options["scene"], options["rays"], options["out"]
    assert scene.positional and scene.kind == "file" and scene.default == Path("scene.yaml")
    assert scene.advanced and out.advanced and not rays.advanced
    assert (rays.kind, rays.default, rays.option) == ("int", None, "--rays")
    assert (out.kind, out.default) == ("path", Path("feedback.wav"))
    assert feedback.command("run", feedback.defaults()) == "python feedback.py"
    args = feedback.defaults()
    args.scene, args.rays, args.out = tmp_path / "copy.yaml", 30, Path("loop.wav")
    assert feedback.arguments(args) == [
        str(tmp_path / "copy.yaml"),
        "--rays",
        "30",
        "--out",
        "loop.wav",
    ]
    argv = feedback.run_argv("run", args)
    assert argv[1:] == [
        str(feedback.script),
        str(tmp_path / "copy.yaml"),
        "--rays",
        "30",
        "--out",
        str(feedback.folder / "loop.wav"),
    ]
    # Its arguments parse as the script's own parser parses them.
    parsed = sibling(feedback.script, "feedback").parser().parse_args(argv[2:])
    assert Path(parsed.scene) == tmp_path / "copy.yaml" and parsed.rays == 30
    assert [d.name for d in scripts_of("town_hall_pa")] == ["delays"]
    assert [d.name for d in scripts_of("listening_room")] == ["bass"]
    assert scripts_of("shoebox") == [] and scripts_of("measurement") == []


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


def test_a_file_given_to_a_source_takes_the_place_of_its_feed(tmp_path: Path, capsys) -> None:
    """``--input``: the file, mixed down to mono and at the feed's rate, brought to the
    gated RMS of the feed it replaces and cut or padded with silence to its length;
    made again when it is another file, even an older one, or either is newer."""
    feed = tmp_path / "feeds" / "movement1" / "a.wav"
    audio.write(feed, np.stack([_tone(2.0, gain=0.1)] * 2, axis=1), RATE)
    mine = tmp_path / "mine.wav"
    audio.write(mine, np.stack([_tone(3.0, 300.0, 16_000, 0.8), np.zeros(48_000)], axis=1), 16_000)
    out = pipeline.input_feed("a", mine, feed, tmp_path / "inputs")
    assert out == tmp_path / "inputs" / "movement1" / "a.wav"
    x, rate = audio.read(out)
    assert rate == RATE and x.shape == (2 * RATE, 1)
    target = audio.gated_rms(audio.read(feed)[0], RATE)
    assert audio.gated_rms(x, rate) == pytest.approx(target, rel=1e-3)
    told = capsys.readouterr().out
    assert "a: plays mine.wav instead of its feed" in told and "cut to 2.0 s" in told
    made = out.stat().st_mtime_ns
    assert pipeline.input_feed("a", mine, feed, tmp_path / "inputs") == out
    assert out.stat().st_mtime_ns == made  # the same file: kept
    short = tmp_path / "short.wav"
    audio.write(short, _tone(0.5, 300.0), RATE)
    older = mine.stat().st_mtime - 100
    os.utime(short, (older, older))
    pipeline.input_feed("a", short, feed, tmp_path / "inputs")
    x, _rate = audio.read(out)
    assert len(x) == 2 * RATE and not x[RATE:].any()  # padded with silence
    assert "padded with silence to 2.0 s" in capsys.readouterr().out
    silent = tmp_path / "silent.wav"
    audio.write(silent, np.zeros(RATE), RATE)
    with pytest.raises(SystemExit, match="silent.wav is silent"):
        pipeline.input_feed("a", silent, feed, tmp_path / "inputs")


def test_the_mixes_play_the_files_given_to_their_sources(tmp_path: Path, monkeypatch) -> None:
    """``render_mixes(inputs=...)`` hands each mix the feeds made from the files, keeps
    them in seats.json, and mixes again when the inputs change; the credit line names
    them, and only them when every source plays one."""
    mixes, folder, made = _mixes(tmp_path, monkeypatch)
    args = argparse.Namespace(format="binaural", hrtf=None, again=False)
    mine = tmp_path / "mine.wav"
    audio.write(mine, _tone(1.0, 300.0), RATE)
    seats = pipeline.render_mixes(mixes, folder, args, inputs={"a": mine})
    assert made == ["near", "far"]
    assert seats["seats"]["near"]["inputs"] == {"a": str(mine)}
    assert (tmp_path / "inputs" / "feeds" / "a.wav").exists()
    made.clear()
    pipeline.render_mixes(mixes, folder, args, inputs={"a": mine})
    assert made == []  # the same inputs: kept
    pipeline.render_mixes(mixes, folder, args)
    assert made == ["near", "far"]  # the feeds again
    with pytest.raises(SystemExit, match="no such source plays here; it has a, b"):
        pipeline.render_mixes(mixes, folder, args, inputs={"c": mine})
    credit = "Music: someone, CC BY 4.0"
    assert pipeline.credited(credit, pipeline.read_json(folder / pipeline.SEATS), "ab") == credit
    seats = {"seats": {"near": {"inputs": {"a": str(mine)}}}}
    assert pipeline.credited(credit, seats, ["a", "b"]) == f"{credit} · instead, a: mine.wav"
    seats["seats"]["far"] = {"inputs": {"b": "/x/theirs.wav"}}
    assert pipeline.credited(credit, seats, ["a", "b"]) == "Playing a: mine.wav, b: theirs.wav"


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


# --------------------------------------------------------------------------- any scene

SHOEBOX = Path(__file__).resolve().parent.parent / "examples" / "shoebox"


def _any_scene(tmp_path: Path, **outputs) -> Path:
    """The shoebox, small and quick to simulate, with three sources: the talker plays a
    loud stereo recording at 22.05 kHz, the radio a quiet mono one at 8 kHz, the fan
    nothing; and two listeners."""
    import shutil

    import yaml

    folder = tmp_path / "room"
    shutil.copytree(SHOEBOX, folder)
    data = yaml.safe_load((folder / "scene.yaml").read_text(encoding="utf-8"))
    data["simulation"] = {
        "early": {"method": "ray_tracing", "max_order": 1, "rays": 500},
        "late": {"method": "stochastic_ray_tracing", "rays": 500},
        "ir_length_s": 0.3,
        "sample_rate": 16_000,
    }
    data["outputs"] = outputs.get("outputs", ["binaural"])
    # The talker plays a file of its own, mixed down, and the scene no recording.
    data.pop("recording", None)
    data["sources"][0].pop("channel", None)
    data["sources"][0]["signal"] = "sounds/talk.wav"
    data["sources"] += [
        {"name": "radio", "position": [3.0, 2.0, 1.0], "level_db": 74.0, "signal": "radio.wav"},
        {"name": "fan", "position": [7.0, 5.0, 2.5]},
    ]
    data["listeners"].append(
        {"name": "seat-b", "position": [4.0, 1.5, 1.2], "orientation": {"yaw": 90}}
    )
    (folder / "scene.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    talk = _tone(2.0, 300.0, 22_050, 0.5)
    audio.write(folder / "sounds" / "talk.wav", np.stack([talk, 0.5 * talk], axis=1), 22_050)
    audio.write(folder / "radio.wav", _tone(2.0, 600.0, RATE, 0.02), RATE)
    return folder / "scene.yaml"


def test_a_demo_of_any_scene_needs_a_source_that_plays(tmp_path: Path) -> None:
    """The demo of any scene says what is missing: the scene, a source that plays, the
    file a source plays."""
    from acoustic_spaces.demos.scene_demo import DEMO

    scene = _any_scene(tmp_path)
    demo = DEMO.at(scene.parent)
    args = demo.defaults()
    assert demo.check(args) == "give the scene to demo (--scene)"
    args.scene = Path("other.yaml")
    assert demo.check(args) == f"no {scene.parent / 'other.yaml'}"
    args.scene = Path("scene.yaml")
    assert demo.check(args) is None
    (scene.parent / "radio.wav").unlink()
    assert (
        demo.check(args)
        == f"radio plays {(scene.parent / 'radio.wav').resolve()}, which is not there"
    )
    text = scene.read_text(encoding="utf-8")
    text = text.replace("signal: sounds/talk.wav", "").replace("signal: radio.wav", "")
    scene.write_text(text, encoding="utf-8")
    assert demo.check(args).startswith("no source plays anything")
    # Where it writes, and how a terminal runs it: names relative to where it runs.
    assert demo.work_folder(args) == scene.parent / "demo"
    args.hrtf = Path("ears.sofa")
    assert demo.command("feeds", args) == "acoustic-spaces demo scene.yaml feeds --hrtf ears.sofa"
    argv = demo.run_argv("feeds", args)
    assert argv[1:4] == ["-m", "acoustic_spaces.demos.scene_demo", "feeds"]
    assert argv[4:] == ["--scene", str(scene), "--hrtf", str(scene.parent / "ears.sofa")]


def test_a_demo_of_any_scene_takes_the_scene_hrtfs(tmp_path: Path) -> None:
    """A scene whose binaural output names a SOFA file is heard through it unless
    ``--hrtf`` says otherwise; other formats take none."""
    from acoustic_spaces.demos.scene_demo import DEMO

    scene = _any_scene(tmp_path, outputs=["mono", {"format": "binaural", "hrtf": "ears.sofa"}])
    demo = DEMO.at(scene.parent)
    args = demo.defaults()
    args.scene = Path("scene.yaml")
    assert demo.resolved(args).hrtf == (scene.parent / "ears.sofa").resolve()
    args.hrtf = Path("mine.sofa")
    assert demo.resolved(args).hrtf == scene.parent / "mine.sofa"
    args.hrtf, args.format = None, "ambix1"
    assert demo.resolved(args).hrtf is None


def test_a_demo_of_any_scene_plays_its_sources(tmp_path: Path, monkeypatch, capsys) -> None:
    """``acoustic-spaces demo``: each source that plays gets a feed at one gated RMS and
    the scene's rate, the silent ones are listed, every listener gets a mix and the
    video a still for each, with what was heard there and the settings of the run."""
    pytest.importorskip("PIL")
    from acoustic_spaces.cli import main as cli
    from acoustic_spaces.demos import pictures
    from acoustic_spaces.demos.scene_demo import DEMO, FEED_RMS

    scene = _any_scene(tmp_path)
    monkeypatch.chdir(scene.parent)
    assert cli(["demo", "scene.yaml", "--seconds", "0.8", "--no-mp4"]) == 0
    work = scene.parent / "demo"
    table = pipeline.read_json(work / "feeds" / "feeds.json")
    assert table["rate"] == 16_000 and list(table["sources"]) == ["talker", "radio"]
    assert table["silent"] == ["fan"]
    for name in ("talker", "radio"):
        x, rate = audio.read(work / "feeds" / f"{name}.wav")
        assert rate == 16_000 and x.reshape(len(x), -1).shape[1] == 1
        assert audio.gated_rms(x, rate) == pytest.approx(FEED_RMS, rel=0.02)
    seats = pipeline.read_json(work / "seats" / "seats.json")
    assert set(seats["seats"]) == {"seat-a", "seat-b"}
    stills = sorted(p.name for p in (work / "stills").glob("*.png"))
    assert stills == ["00_seat-a.png", "01_seat-b.png"]
    assert (work / "thumbnail.png").exists() and (work / "tour.wav").exists()
    # Again: what is there is left out, and a step on its own runs.
    capsys.readouterr()
    assert cli(["demo", "scene.yaml", "--seconds", "0.8", "--no-mp4"]) == 0
    out = capsys.readouterr().out
    assert "feeds: done already" in out and "simulate: done already" in out
    assert "there already" in out  # the mixes, newer than their feeds and responses
    # The stills say how the sound was made and how to make it again.
    args = DEMO.at(scene.parent).defaults()
    args.scene, args.seconds, args.no_mp4 = Path("scene.yaml"), 0.8, True
    args = DEMO.at(scene.parent).resolved(args)
    report = pipeline.read_json(work / "results" / pipeline.PARAMETERS)
    from acoustic_spaces.demos import scene_pictures
    from acoustic_spaces.scene import Scene

    # What each still says: the level of the sources that play, all of them or none
    # (a response too short for a level gives None), and T30 at mid frequencies, the
    # mean of the 500 Hz and 1 kHz bands when both are there.
    levels = {("talker", "seat-a"): None, ("radio", "seat-a"): 70.0, ("fan", "seat-b"): None}
    t30 = {("talker", "seat-b"): [1.0, 2.0], ("radio", "seat-b"): [1.0, None]}
    made = {
        "bands_hz": [500, 1000],
        "responses": [
            {
                "source": s,
                "listener": lis,
                "spl_a_db": levels.get((s, lis), 70.0),
                "parameters": {"t30": t30.get((s, lis), [None, None])},
            }
            for s in ("talker", "radio", "fan")
            for lis in ("seat-a", "seat-b")
        ],
    }
    stops = [
        pipeline.TourStop("seat-a", 0.0, 0.0, 0.8),
        pipeline.TourStop("seat-b", 0.8, 0.8, 1.6, gain_db=6.0),
    ]
    drawing = scene_pictures.Pictures(
        Scene.from_yaml(scene).to_data(),
        np.zeros((0, 2, 2)),
        made,
        seats["seats"],
        stops,
        playing=["talker", "radio"],
    )
    assert drawing.title("seat-b") == ("seat-b", "Listener 2 of 2")
    assert drawing.level_at("seat-a") is None
    seat_b = drawing.lines(1)
    assert seat_b[0] == "73 dB(A) here, as simulated"
    assert "Reverberation time T30 1.5 s (mean of 500 Hz and 1 kHz)" in seat_b
    assert "Turned up 6 dB here so you can hear it" in seat_b
    assert not any("dB(A)" in str(line) for line in drawing.lines(0))
    first, run = pictures.settings_lines(DEMO.at(scene.parent), args, {}, report)
    assert first.startswith("Simulated: early reflections traced to order 1 with 500 rays")
    assert "stochastic ray tracing for the late sound, 500 rays per band" in first
    assert run.startswith("acoustic-spaces ") and run.endswith(
        " at 16 kHz, mixed binaural through a spherical head model. "
        "Run as: acoustic-spaces demo scene.yaml all --seconds 0.8 --no-mp4"
    )
    # A scene changed since is fed and simulated again.
    later = (work / "feeds" / "feeds.json").stat().st_mtime + 10
    os.utime(scene, (later, later))
    from acoustic_spaces.demos.scene_demo import feeds_done

    assert not feeds_done(args)
    assert not pipeline.simulated(pipeline.Simulation(scene, work / "results", "binaural"), args)
    # A missing scene is the command line's error.
    assert cli(["demo", "nothing.yaml"]) == 2
    assert "no " in capsys.readouterr().err


def test_a_demo_of_any_scene_plays_the_listeners_named(tmp_path: Path, monkeypatch, capsys) -> None:
    """``--listeners`` (or ``--tour``, its earlier name): the demo of any scene mixes
    those listeners alone and its video visits them in that order, from a simulation of
    every listener, so that another choice needs no new one; a listener the scene has
    not is refused with those it has."""
    pytest.importorskip("PIL")
    from acoustic_spaces.cli import main as cli

    scene = _any_scene(tmp_path)
    monkeypatch.chdir(scene.parent)
    run = ["demo", "scene.yaml", "--seconds", "0.8", "--no-mp4"]
    assert cli([*run, "--listeners", "seat-b"]) == 0
    work = scene.parent / "demo"
    report = pipeline.read_json(work / "results" / pipeline.PARAMETERS)
    assert {e["listener"] for e in report["responses"]} == {"seat-a", "seat-b"}
    assert set(pipeline.read_json(work / "seats" / "seats.json")["seats"]) == {"seat-b"}
    assert sorted(p.name for p in (work / "stills").glob("*.png")) == ["00_seat-b.png"]
    capsys.readouterr()
    assert cli([*run, "--tour", "seat-b,seat-a"]) == 0
    assert "simulate: done already" in capsys.readouterr().out
    assert set(pipeline.read_json(work / "seats" / "seats.json")["seats"]) == {"seat-a", "seat-b"}
    stills = sorted(p.name for p in (work / "stills").glob("*.png"))
    assert stills == ["00_seat-b.png", "01_seat-a.png"]
    assert cli(["demo", "scene.yaml", "--listeners", "nobody"]) == 2
    assert "--listeners nobody: no such listener here; it has seat-a, seat-b" in (
        capsys.readouterr().err
    )


def test_a_demo_of_any_scene_moves_its_walkers(tmp_path: Path, monkeypatch, capsys) -> None:
    """With a walk on, the demo of any scene hears every listener with the walkers moving
    on one clock: the mixes go to moving/, one gain for all, with the level of each
    source at each moment; the frames move the walker and say the level then.
    ``--moving`` picks the walks of a run: none keeps every walker standing, a name with
    no walk is refused, and so is a walk with Doppler with another walker."""
    pytest.importorskip("PIL")
    import yaml

    from acoustic_spaces.cli import main as cli
    from acoustic_spaces.demos import scene_pictures
    from acoustic_spaces.demos.scene_demo import DEMO, moving_of, moving_scene
    from acoustic_spaces.walkthrough import knots, movers

    scene = _any_scene(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    turn = {"time_s": 1.5, "position": [6.0, 2.5, 1.2], "orientation": {"yaw": 180}}
    data["listeners"][1]["walk"] = {
        "keyframes": [{"time_s": 1.0, "position": [6.0, 2.5, 1.2]}, turn]
    }
    data["sources"][1]["walk"] = {"enabled": False, "to": [[5.0, 2.0, 1.0]], "doppler": True}
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(scene.parent)
    demo = DEMO.at(scene.parent)
    args = demo.defaults()
    args.scene = Path("scene.yaml")
    assert moving_of(demo.resolved(args)) == ["seat-b"] and demo.check(args) is None
    args.moving = "seat-b,radio"
    assert demo.check(args) == (
        "radio walks with Doppler, which plays alone: leave the others standing "
        "(--moving radio), or turn its Doppler off"
    )
    args.moving = "seat-a"
    assert demo.check(args) == (
        "--moving: 'seat-a' has no walk to move along; the scene's walks: seat-b, radio"
    )
    args.moving = "none"
    assert moving_of(demo.resolved(args)) == [] and demo.check(args) is None
    assert demo.command("render", args) == "acoustic-spaces demo scene.yaml render --moving none"

    assert cli(["demo", "scene.yaml", "--seconds", "0.8", "--no-mp4"]) == 0
    work = scene.parent / "demo"
    seats = pipeline.read_json(work / "moving" / "seats.json")
    assert seats["moving"] == ["seat-b"] and set(seats["seats"]) == {"seat-a", "seat-b"}
    seat_a, seat_b = seats["seats"]["seat-a"], seats["seats"]["seat-b"]
    assert seat_b["moves"] and not seat_a["moves"]
    assert seat_a["arrival_s"] > 0 and seat_b["arrival_s"] is None  # it changes on the way
    assert seat_a["lag_s"] == seat_b["lag_s"] == 0.0  # the tour keeps the walks' clock
    moments = seats["movement"]["times"]
    assert moments[0] == 0.0 and moments[-1] == pytest.approx(1.5)
    levels = seats["movement"]["levels_a_db"]["seat-b"]
    assert set(levels) == {"talker", "radio"}
    assert all(len(v) == len(moments) for v in levels.values())
    peaks = [
        np.abs(audio.read(work / "moving" / f"{k}.binaural.wav")[0]).max() for k in seats["seats"]
    ]
    assert max(peaks) == pytest.approx(10.0 ** (-1.0 / 20.0), rel=1e-3)
    assert not (work / "seats").exists()
    stills = sorted(p.name for p in (work / "stills").glob("*.png"))
    assert stills == ["00_seat-a.png", "01_seat-b.png"]
    # The frames: the walker where its walk has it then, the level of that moment.
    resolved = demo.resolved(argparse.Namespace(**vars(demo.defaults()) | {"scene": scene}))
    moving = moving_scene(resolved)
    stops = [pipeline.TourStop("seat-a", 0.0, 0.0, 0.8), pipeline.TourStop("seat-b", 0.8, 0.8, 1.6)]
    drawing = scene_pictures.Pictures(
        moving.to_data(),
        np.zeros((0, 2, 2)),
        pipeline.read_json(work / "results" / pipeline.PARAMETERS),
        seats["seats"],
        stops,
        playing=["talker", "radio"],
        paths={(kind, item.name): knots(item) for kind, item in movers(moving)},
        movement=seats["movement"],
    )
    walker = drawing.listeners["seat-b"]
    assert drawing.placed("listeners", walker, 1.2)["position"] == pytest.approx([6.0, 2.5, 1.2])
    assert drawing.placed("listeners", walker, None) is walker
    assert drawing.moment(1.5) == len(moments) - 1 and drawing.moment(0.05) == 0
    words = [str(line) for line in drawing.lines(1, 1.2)]
    assert any("Moving along their walks: seat-b" in line for line in words)
    assert not any("arrives" in line for line in words)
    drawing.t30[("talker", "seat-b")] = 1.5  # the quick scene's responses are too short
    assert "Reverberation time T30 1.5 s where the walks start (mean of 500 Hz and 1 kHz)" in (
        drawing.lines(1, 1.2)
    )
    assert drawing.frame(1.2).getbbox() is not None
    # Again: there already; with every walker standing, the still mixes instead.
    capsys.readouterr()
    assert cli(["demo", "scene.yaml", "render"]) == 0
    assert "there already" in capsys.readouterr().out
    assert cli(["demo", "scene.yaml", "render", "--moving", "none"]) == 0
    assert set(pipeline.read_json(work / "seats" / "seats.json")["seats"]) == {"seat-a", "seat-b"}


def test_a_demo_of_any_scene_plays_a_walk_with_doppler(tmp_path: Path, monkeypatch) -> None:
    """A walk with Doppler moves alone: every listener hears the radio drive past, from
    one simulation with the radio at each place, its level at each place (the talker's,
    standing, the same at every one), one gain for all."""
    pytest.importorskip("PIL")
    import yaml

    from acoustic_spaces.cli import main as cli

    scene = _any_scene(tmp_path)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    data["sources"][1]["walk"] = {"to": [[6.0, 2.0, 1.0]], "speed_m_s": 10.0, "doppler": True}
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(scene.parent)
    assert cli(["demo", "scene.yaml", "--seconds", "0.8", "--no-mp4"]) == 0
    work = scene.parent / "demo"
    seats = pipeline.read_json(work / "moving" / "seats.json")
    assert seats["moving"] == ["radio"] and len(seats["seats"]) == 2
    moments = seats["movement"]["times"]
    assert moments[0] == 0.0 and moments[-1] == pytest.approx(0.3)  # 3 m at 10 m/s
    for name, levels in seats["movement"]["levels_a_db"].items():
        assert seats["seats"][name]["moves"] and seats["seats"][name]["arrival_s"] is None
        assert len(levels["radio"]) == len(moments) and len(set(levels["radio"])) > 1
        assert len(set(levels["talker"])) == 1  # it stands
    peaks = [
        np.abs(audio.read(work / "moving" / f"{k}.binaural.wav")[0]).max() for k in seats["seats"]
    ]
    assert max(peaks) == pytest.approx(10.0 ** (-1.0 / 20.0), rel=1e-3)


def test_a_demo_of_any_scene_plays_the_files_given_to_its_sources(
    tmp_path: Path, monkeypatch
) -> None:
    """``--input``: a source plays the file instead of its signal, a silent one too, fed
    at the same gated RMS as every feed; the feeds are made again when the inputs
    change, and a name the scene has no source in use for is refused."""
    from acoustic_spaces.cli import main as cli
    from acoustic_spaces.demos.scene_demo import DEMO, FEED_RMS, feeds_done, plays

    scene = _any_scene(tmp_path)
    monkeypatch.chdir(scene.parent)
    audio.write(scene.parent / "hum.wav", _tone(1.0, 100.0, RATE, 0.3), RATE)
    argv = ["demo", "scene.yaml", "feeds", "--input", "fan=hum.wav", "--input", "radio=hum.wav"]
    assert cli(argv) == 0
    work = scene.parent / "demo"
    table = pipeline.read_json(work / "feeds" / "feeds.json")
    hum = str(scene.parent / "hum.wav")
    assert table["sources"]["fan"]["signal"] == hum and table["sources"]["radio"]["signal"] == hum
    assert table["sources"]["talker"]["signal"].endswith("talk.wav") and table["silent"] == []
    x, rate = audio.read(work / "feeds" / "fan.wav")
    assert audio.gated_rms(x, rate) == pytest.approx(FEED_RMS, rel=0.02)
    demo = DEMO.at(scene.parent)
    args = demo.defaults()
    args.scene, args.inputs = Path("scene.yaml"), {"fan": Path("hum.wav"), "radio": Path("hum.wav")}
    args = demo.resolved(args)
    assert feeds_done(args)
    assert plays(args) == {"talker": "talk.wav", "radio": "radio.wav", "fan": "silent"}
    args.inputs = None
    assert not feeds_done(args)  # the radio's own signal and a silent fan now
    args.inputs = {"nobody": scene.parent / "hum.wav"}
    assert demo.check(args).startswith("--input nobody: no such source plays here")


def test_a_demo_of_any_scene_plays_the_channels_of_its_recording(
    tmp_path: Path, monkeypatch, recordings_folder: Path
) -> None:
    """Sources with a channel and no signal play that channel of the scene's recording:
    left and right keep the recording's stereo balance (one gain for both, which brings
    their mean square to the feeds' gated RMS, so that unlike channels do not play over
    their sound power), mono plays both mixed down at the feeds' gated RMS; a source's
    own signal still wins, and its channel picks a channel of it. A free recording not
    downloaded yet is the recording step's download; there, it plays as a file would,
    and the stills credit it as its licence asks."""
    import yaml

    from acoustic_spaces import recordings
    from acoustic_spaces.cli import main as cli
    from acoustic_spaces.demos.scene_demo import (
        DEMO,
        FEED_RMS,
        feeds_done,
        plays,
        plays_line,
        recording_download,
    )

    scene = _any_scene(tmp_path)
    monkeypatch.chdir(scene.parent)
    data = yaml.safe_load(scene.read_text(encoding="utf-8"))
    rate = 16_000
    band = np.stack([_tone(2.0, 200.0, rate, 0.4), _tone(2.0, 300.0, rate, 0.1)], axis=1)
    audio.write(scene.parent / "band.wav", band, rate)
    data["recording"] = "band.wav"
    data["sources"][0]["channel"] = "right"  # of its own signal, talk.wav
    data["sources"][1].pop("signal")
    data["sources"][1]["channel"] = "left"  # the radio: the band's left
    data["sources"][2]["channel"] = "mono"  # the fan: both mixed down
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    demo = DEMO.at(scene.parent)
    args = demo.defaults()
    args.scene = Path("scene.yaml")
    args = demo.resolved(args)
    assert demo.check(args) is None
    assert plays(args) == {
        "talker": "talk.wav, right channel",
        "radio": "band.wav, left channel",
        "fan": "band.wav, mixed down to mono",
    }
    assert recording_download(args) is None  # a file of its own: nothing to download
    assert cli(["demo", "scene.yaml", "feeds"]) == 0
    table = pipeline.read_json(scene.parent / "demo" / "feeds" / "feeds.json")
    assert {k: v["channel"] for k, v in table["sources"].items()} == {
        "talker": "right",
        "radio": "left",
        "fan": "mono",
    }
    assert feeds_done(args)
    feed = {
        k: audio.read(scene.parent / "demo" / "feeds" / f"{k}.wav")[0].ravel()
        for k in table["sources"]
    }

    def pair_gain(x: np.ndarray) -> float:
        x = audio.stereo(x)
        squares = [audio.gated_rms(x[:, i], rate) ** 2 for i in (0, 1)]
        return FEED_RMS / float(np.sqrt(np.mean(squares)))

    np.testing.assert_allclose(feed["radio"], band[:, 0] * pair_gain(band), atol=1e-6)
    left = band[:, 0] * pair_gain(band)
    right = band[:, 1] * pair_gain(band)
    mean_square = (audio.gated_rms(left, rate) ** 2 + audio.gated_rms(right, rate) ** 2) / 2
    assert np.sqrt(mean_square) == pytest.approx(FEED_RMS, rel=1e-6)
    # One gain from the mixdown would play the band's two tones (unlike each other) over.
    assert pair_gain(band) < FEED_RMS / audio.gated_rms(band, rate)
    mixdown_gain = FEED_RMS / audio.gated_rms(band, rate)
    np.testing.assert_allclose(feed["fan"], band.mean(axis=1) * mixdown_gain, atol=1e-6)
    assert audio.gated_rms(feed["fan"], rate) == pytest.approx(FEED_RMS, rel=1e-6)
    talk, talk_rate = audio.read(scene.parent / "sounds" / "talk.wav")
    talk = audio.resampled(talk, talk_rate, rate)
    np.testing.assert_allclose(
        feed["talker"], audio.stereo(talk)[:, 1] * pair_gain(talk), atol=1e-6
    )
    # A free recording: the recording step downloads it, the feeds wait for it.
    data["recording"] = "potassium"
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert not feeds_done(args)
    assert demo.check(args) is None
    assert plays(args)["radio"] == '"Potassium", Software-Entwicklungskit, left channel'
    assert recording_download(args) == (
        '"Potassium", Software-Entwicklungskit (CC0 1.0, 6 MB), kept for every scene'
    )
    assert cli(["demo", "scene.yaml", "feeds"]) == 1
    kept = recordings.file_of("potassium")
    assert kept.parent == recordings_folder
    audio.write(kept, band, rate)
    assert recording_download(args) is None
    assert cli(["demo", "scene.yaml", "feeds"]) == 0
    table = pipeline.read_json(scene.parent / "demo" / "feeds" / "feeds.json")
    assert table["sources"]["fan"]["signal"] == str(kept)
    credit = plays_line(table, recordings.named("potassium"))
    assert credit.startswith(recordings.named("potassium").credit)
    assert credit.endswith("Playing talker: talk.wav")
    # A channel with no recording plays nothing.
    data.pop("recording")
    scene.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert plays(args) == {"talker": "talk.wav, right channel", "radio": "silent", "fan": "silent"}


def test_free_recordings_are_downloaded_once(tmp_path: Path, monkeypatch) -> None:
    """A free recording comes as one WAV file in the recordings folder: a song's files
    decoded and added together, or a talk taken out of its corpus's tar archive; one
    there already is kept, and what it was made from is not."""
    import dataclasses

    from acoustic_spaces import recordings

    rate = 8000
    source = tmp_path / "server"
    left = np.stack([_tone(1.0, 200.0, rate, 0.2), _tone(1.0, 300.0, rate, 0.1)], axis=1)
    audio.write(source / "a.wav", left, rate)
    audio.write(source / "b.wav", left[:, ::-1], rate)
    audio.write(source / "talk.wav", _tone(1.0, 150.0, rate, 0.3), rate)
    with tarfile.open(source / "corpus.tar.bz2", "w:bz2") as archive:
        archive.add(source / "talk.wav", arcname="Speech/talk.wav")
    song = dataclasses.replace(
        recordings.named("potassium"),
        name="test_song",
        files=((str((source / "a.wav").as_uri()), None), ((source / "b.wav").as_uri(), None)),
    )
    talk = dataclasses.replace(
        recordings.named("speech_male"),
        name="test_talk",
        tar=((source / "corpus.tar.bz2").as_uri(), "Speech/talk.wav"),
    )
    monkeypatch.setattr(recordings, "RECORDINGS", (song, talk))
    path = recordings.get("test_song")
    assert path == recordings.data_folder() / "test_song.wav"
    x, got = audio.read(path)
    assert got == rate
    np.testing.assert_allclose(x, left + left[:, ::-1], atol=1e-6)
    assert sorted(p.name for p in path.parent.iterdir()) == ["test_song.wav"]
    (source / "a.wav").unlink()
    assert recordings.get("test_song") == path  # kept: nothing fetched again
    talked = recordings.get("test_talk")
    np.testing.assert_allclose(
        audio.read(talked)[0].ravel(), _tone(1.0, 150.0, rate, 0.3), atol=1e-6
    )
    with pytest.raises(KeyError, match="no free recording 'nothing'"):
        recordings.named("nothing")


def test_a_demo_scene_written_elsewhere_still_finds_its_files(tmp_path: Path) -> None:
    """A demo run in another folder than its example (the editor's copy under the home
    folder) writes scenes whose files are found from there: relative under the new
    folder, in full elsewhere; absolute ones and the example's own folder unchanged."""
    import yaml

    example = tmp_path / "example"
    ears = (tmp_path / "data" / "ears.sofa").as_posix()  # absolute on every system
    data = {
        "geometry": {"file": "model.obj"},
        "outputs": [{"format": "binaural", "hrtf": ears}],
        "sources": [{"name": "a", "signal": "work/talk.wav"}],
    }
    pipeline.write_scene(example / "scene.yaml", data, "# a header\n", example)
    assert yaml.safe_load((example / "scene.yaml").read_text(encoding="utf-8")) == data
    pipeline.write_scene(example / "work" / "scene.yaml", data, "", example)
    moved = yaml.safe_load((example / "work" / "scene.yaml").read_text(encoding="utf-8"))
    assert moved["geometry"]["file"] == (example / "model.obj").resolve().as_posix()
    assert moved["sources"][0]["signal"] == "talk.wav"
    assert moved["outputs"][0]["hrtf"] == ears
    assert (example / "scene.yaml").read_text(encoding="utf-8").startswith("# a header\n")


def test_the_stills_say_how_the_sound_was_made(tmp_path: Path) -> None:
    """The lines under every still state the simulation as it ran, the engine's defaults
    for what a scene leaves out, the wave solver and, outdoors in wind, what the weather
    bent; a line too long for the still is set smaller to fit."""
    pytest.importorskip("PIL")
    import yaml
    from PIL import ImageDraw

    from acoustic_spaces.demos import pictures

    demo, _ran = _demo(tmp_path)
    args = demo.resolved(demo.defaults())
    # A scene that names nothing gets the engine's defaults, not guesses.
    first, mixed = pictures.settings_lines(demo, args, {"simulation": {}})
    assert "early reflections traced to order 3 with 20,000 rays" in first
    assert "the direct sound and first reflections diffracted round edges" in first
    assert "50,000 rays per band" in first
    assert mixed.endswith(
        " at 48 kHz, mixed binaural through a spherical head model. "
        "Run as: python test_demo.py all (every option at its default)"
    )
    simulation = {
        "early": {"method": "image_sources", "max_order": 2, "diffraction": "none"},
        "late": {"method": "statistical"},
        "wave": {"enabled": True, "max_band_hz": 250},
    }
    first, _mixed = pictures.settings_lines(demo, args, {"simulation": simulation})
    assert first.endswith(
        "image sources to order 2; a statistical late sound; the bands up to 250 Hz from "
        "the wave solver"
    )
    # Outdoors in wind: the paths, the late rays and the parabolic equation.
    lake = SHOEBOX.parent / "lakeside_concert" / "scene.yaml"
    scene = yaml.safe_load(lake.read_text(encoding="utf-8"))
    first, _mixed = pictures.settings_lines(demo, args, scene)
    assert "paths and late rays bent by the wind and temperature" in first
    assert "up to 500 Hz (2 kHz in shadows) by the narrow-angle parabolic equation" in first
    scene["atmosphere"] = None
    assert "bent" not in pictures.settings_lines(demo, args, scene)[0]
    # Every line fits the still, made smaller when it is long.
    draw = ImageDraw.Draw(pictures.canvas())
    room = pictures.SIZE[0] - 2 * pictures.MARGIN
    line, f = pictures.fitted(draw, "word " * 40, 18, room)
    assert line == "word " * 40 and f.size < 18 and draw.textlength(line, font=f) <= room
    line, f = pictures.fitted(draw, "word " * 400, 18, room)
    assert line.endswith("…") and f.size == pictures.FOOTER_MIN
