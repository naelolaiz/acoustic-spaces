import importlib.util
import io
import math
import sys
from pathlib import Path

import pytest
import yaml

from acoustic_spaces.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHOEBOX = EXAMPLES / "shoebox" / "scene.yaml"


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "acoustic_spaces" in out
    assert "compiled core: available" in out
    assert "CPU cores for ray tracing: " in out
    assert "device for scenes that name none: cpu" in out
    assert "device choices for auto: none yet" in out


def test_tune_needs_a_gpu(capsys, monkeypatch) -> None:
    from acoustic_spaces.engine import gpu

    monkeypatch.setattr(gpu, "_adapter", lambda: None)
    assert main(["tune", str(SHOEBOX), "--rays", "10"]) == 1
    assert "error: tune compares the CPU with a GPU" in capsys.readouterr().err


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_errors_and_warnings_in_colour_on_a_terminal(monkeypatch, tmp_path) -> None:
    from acoustic_spaces import cli

    terminal = _Terminal()
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setattr(cli, "_windows_escape_codes", lambda: True)
    missing = str(tmp_path / "missing.yaml")
    red, yellow, plain = "\033[1;31m", "\033[1;33m", "\033[0m"
    assert main(["validate", missing]) == 1
    assert terminal.getvalue().startswith(f"{red}error:{plain} ")
    assert sys.stderr is terminal  # put back
    main(["validate", str(EXAMPLES / "leaky_classroom" / "scene.yaml")])
    assert f"{yellow}warning:{plain} the model is not closed" in terminal.getvalue()
    with pytest.raises(SystemExit):
        main(["tune", "--default", "gpu"])
    assert f"tune: {red}error:{plain} argument --default" in terminal.getvalue()
    # Plain with NO_COLOR, or written anywhere but a terminal.
    monkeypatch.setenv("NO_COLOR", "1")
    terminal.seek(0)
    terminal.truncate()
    assert main(["validate", missing]) == 1
    assert terminal.getvalue().startswith("error: ")
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    assert main(["validate", missing]) == 1
    assert sys.stderr.getvalue().startswith("error: ")


def test_tune_switches_the_default_back_to_the_cpu(capsys, monkeypatch) -> None:
    from acoustic_spaces.engine import gpu, tuning

    tuning.set_default("auto")
    monkeypatch.setattr(gpu, "_adapter", lambda: None)  # times nothing, so needs no GPU
    assert main(["tune", "--default", "cpu"]) == 0
    assert "default device is now cpu" in capsys.readouterr().out
    assert tuning.default_device() == "cpu"
    with pytest.raises(SystemExit):
        main(["tune", "--default", "gpu"])


def test_validate_example(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", str(SHOEBOX)]) == 0
    assert "1 sources, 1 listeners, 12 triangles" in capsys.readouterr().out


def test_validate_reports_problems(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["geometry"]["file"] = str(EXAMPLES / "shoebox" / "shoebox.obj")
    del data["assignments"]["walls"]
    data["listeners"][0]["position"] = [6.0, 3.0, 4.0]
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "surfaces without a material: walls (84.0 m2)" in err
    assert "listener 'seat-a'" in err


def test_validate_reports_schema_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["materials"]["painted_concrete"]["preset"] = "smooth_brickwork"
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    assert main(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "materials.painted_concrete: no absorption material named 'smooth_brickwork'" in err


def test_validate_reports_unreadable_scenes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # These ended in tracebacks.
    path = tmp_path / "broken.yaml"
    path.write_text("sources: [{name: s\n", encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    assert capsys.readouterr().err.startswith("error: broken.yaml: not valid YAML: ")
    assert main(["validate", str(tmp_path)]) == 1  # a folder
    assert capsys.readouterr().err.startswith("error: ")
    monkeypatch.setitem(sys.modules, "ifcopenshell", None)
    assert main(["validate", str(EXAMPLES / "bim_office" / "scene.yaml")]) == 1
    assert capsys.readouterr().err.startswith("error: reading IFC files needs IfcOpenShell")


def test_validate_refuses_a_model_of_degenerate_faces(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "line.obj").write_text("v 0 0 0\nv 1 0 0\nv 2 0 0\nf 1 2 3\n", encoding="utf-8")
    data = yaml.safe_load(SHOEBOX.read_text(encoding="utf-8"))
    data["geometry"]["file"] = "line.obj"
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["validate", str(path)]) == 1
    err = capsys.readouterr().err
    assert "error: line.obj has no faces left: all 1 of them have no area" in err
    assert "outside the room" not in err


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(SHOEBOX)]) == 0
    out = capsys.readouterr().out
    assert "volume 144.00 m3" in out
    assert "Sabine" in out and "Eyring" in out and "Arau-Puchades" in out
    # The example's absorption sits on the floor and ceiling: its decay is no diffuse one.
    assert "note: the absorption is spread unevenly" in out
    assert "seat-a" in out and "inside" in out
    assert "sound power (dB re 1 pW per octave band" in out
    row = next(line for line in out.splitlines() if line.startswith("talker ") and "85.0" in line)
    assert row.split()[1:9] == ["85.0"] * 8


def test_inspect_gives_the_air_volume_of_walls_drawn_as_a_shell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from acoustic_spaces.geometry import box_mesh

    # A closed box inside a closed box, as a solid wall is drawn: the report counts the
    # cavity between them as the room, the sources and the listener stand in the inner one.
    vertices, faces = [], []
    for low, size in (((0, 0, 0), (4.0, 3.0, 2.5)), ((1, 1, 1), (1.0, 1.0, 1.0))):
        mesh = box_mesh(size)
        faces += [f"f {' '.join(str(len(vertices) + 1 + i) for i in f)}" for f in mesh.faces]
        vertices += [f"v {x} {y} {z}" for x, y, z in mesh.vertices + low]
    obj = tmp_path / "shell.obj"
    obj.write_text("\n".join([*vertices, "g walls", *faces]) + "\n", encoding="utf-8")
    data = {
        "geometry": {"file": obj.as_posix()},
        "materials": {"m": {"absorption": [0.2] * 8}},
        "default_material": "m",
        "sources": [{"name": "s", "position": [1.5, 1.5, 1.5]}],
        "listeners": [{"name": "l", "position": [1.7, 1.6, 1.4]}],
    }
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["inspect", str(scene)]) == 0
    out = capsys.readouterr().out
    assert "volume 29.00 m3" in out  # the cavity, as the report counts it
    assert "air volume 1.00 m3" in out
    # An ordinary room has one volume only.
    assert main(["inspect", str(SHOEBOX)]) == 0
    assert "air volume" not in capsys.readouterr().out


def test_inspect_of_an_open_model_with_a_closed_room_in_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from acoustic_spaces.geometry import box_mesh

    # A closed box and a stray panel outside it: the model is not closed (the panel has
    # open edges) while the box still encloses a room, and inspect says what is open.
    mesh = box_mesh((4.0, 3.0, 2.5))
    vertices = [f"v {x} {y} {z}" for x, y, z in mesh.vertices]
    faces = [f"f {' '.join(str(i + 1) for i in f)}" for f in mesh.faces]
    n = len(vertices)
    vertices += [f"v 10 {y} {z}" for y, z in ((0, 0), (1, 0), (1, 1), (0, 1))]
    faces += [f"f {n + 1} {n + 2} {n + 3} {n + 4}"]
    obj = tmp_path / "leaky.obj"
    obj.write_text("\n".join([*vertices, "g walls", *faces]) + "\n", encoding="utf-8")
    data = {
        "geometry": {"file": obj.as_posix()},
        "materials": {"m": {"absorption": [0.2] * 8}},
        "default_material": "m",
        "sources": [{"name": "s", "position": [1.0, 1.0, 1.0]}],
        "listeners": [{"name": "l", "position": [2.0, 1.5, 1.2]}],
    }
    scene = tmp_path / "scene.yaml"
    scene.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert main(["inspect", str(scene)]) == 0
    out = capsys.readouterr().out
    assert "open:" in out
    assert "air volume" not in out


def test_inspect_shows_third_octave_data_in_octaves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = SHOEBOX.read_text(encoding="utf-8")
    text = text.replace("file: shoebox.obj", f"file: {(SHOEBOX.parent / 'shoebox.obj').as_posix()}")
    third = ", ".join(["0.1", "0.4", "0.1"] * 8)
    text = text.replace("preset: ceiling_fissured_tile", f"absorption: [{third}]")
    scene = tmp_path / "scene.yaml"
    scene.write_text(text, encoding="utf-8")
    assert main(["inspect", str(scene)]) == 0
    out = capsys.readouterr().out
    row = next(line for line in out.splitlines() if line.startswith("acoustic_ceiling"))
    assert row.split()[1:9] == ["0.20"] * 8
    assert "third-octave data, shown as octave averages" in out


def test_inspect_shows_line_and_surface_sources(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = SHOEBOX.read_text(encoding="utf-8")
    text = text.replace("file: shoebox.obj", f"file: {(SHOEBOX.parent / 'shoebox.obj').as_posix()}")
    text = text.replace("    directivity: omni\n", "    directivity: lambert\n    line_m: 3\n")
    scene = tmp_path / "scene.yaml"
    scene.write_text(text, encoding="utf-8")
    assert main(["inspect", str(scene)]) == 0
    out = capsys.readouterr().out
    assert "inside  line 3 m, 6 points" in out and "lambert" in out
    scene.write_text(text.replace("line_m: 3", "surface_m: [3, 9]"), encoding="utf-8")
    assert main(["inspect", str(scene)]) == 1  # higher than the room
    captured = capsys.readouterr()
    assert "OUTSIDE  surface 3 x 9 m, 108 points" in captured.out
    assert "source 'talker' at [2.0, 3.0, 1.5] reaches outside the room" in captured.err
    assert main(["directivity", "lambert"]) == 0
    assert "directivity index dB     6.0   6.0" in capsys.readouterr().out
    column = tmp_path / "column.yaml"
    column.write_text("count: 8\nspacing_m: 0.2\n", encoding="utf-8")
    assert main(["directivity", str(column)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("array of 8 elements") and "   9.0" in out  # 10 log10(8) at 8 kHz


def test_materials_search(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["materials", "fissured"]) == 0
    assert "ceiling_fissured_tile" in capsys.readouterr().out
    assert main(["materials", "--kind", "scattering"]) == 0
    assert "rpg_qrd" in capsys.readouterr().out
    assert main(["materials", "no-such-material"]) == 1


def test_export_for_blender(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "shoebox.glb"
    args = ["export", str(SHOEBOX), str(out), "--paths", "--max-order", "1", "--particles"]
    assert main([*args, "--count", "20", "--duration", "30", "--frames", "4"]) == 0
    printed = capsys.readouterr().out
    assert "3 surfaces, 1 sources, 1 listeners" in printed
    assert "7 paths from talker to seat-a: 1 of order 0, 6 of order 1" in printed
    assert "20 particles from talker: 30 ms of sound over 1.2 s (29 frames at 24 fps)" in printed
    assert out.read_bytes()[:4] == b"glTF"
    assert main(["export", str(SHOEBOX), str(out), "--particles", "--source", "piano"]) == 1
    assert "no source 'piano'" in capsys.readouterr().err


def test_simulate_writes_responses_and_parameters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from acoustic_spaces.audio import read_wav

    out = tmp_path / "out"
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--rays", "2000", "--length", "0.6"]
    assert main([*args, "--seed", "3", "--sample-rate", "16000", "--normalize"]) == 0
    stdout = capsys.readouterr().out
    assert "talker -> seat-a" in stdout
    assert "T30 (s)" in stdout
    data, fs = read_wav(out / "talker__seat-a.wav")
    assert fs == 16000 and data.shape == (9600, 1)
    assert abs(data).max() == pytest.approx(10 ** (-1 / 20), rel=1e-6)
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["seed"] == 3
    (response,) = report["responses"]
    assert response["early_paths"] == 63
    assert len(response["parameters"]["c80"]) == 8
    # The listener 4 m away: lateral fractions, no stage support; IACC from the scene's
    # binaural output.
    params = response["parameters"]
    assert [v is None for v in params["lf"]] == [v is None for v in params["c80"]]
    assert params["lf"][4] is not None and params["iacc_e"][4] is not None
    assert all(v is None for v in params["st_early"])
    assert 0.0 < response["speech"]["sti"] < 1.0 and 0.0 < response["speech"]["stipa"] < 1.0
    assert set(response["echo"]) == {"speech", "music"}
    assert "LF " in stdout and "STI 0." in stdout and "echo criterion" in stdout
    assert "SPL " in stdout and "dB(A)" in stdout
    assert len(response["spl_db"]) == 8 and 40.0 < response["spl_a_db"] < 100.0


def test_simulate_checks_the_options_that_override_the_scene(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    out = tmp_path / "out"
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--length", "0.3"]
    for option, value, message in [
        ("--rays", "0", "greater than or equal to 1"),  # was a ZeroDivisionError
        ("--sample-rate", "-8000", "greater than or equal to 8000"),
        ("--sample-rate", "100", "greater than or equal to 8000"),  # was accepted
        ("--max-order", "21", "less than or equal to 20"),
        ("--length", "nan", "a finite number"),
    ]:
        assert main([*args, option, value]) == 1
        assert f"error: {option}: Input should be {message}" in capsys.readouterr().err
    assert not out.exists()
    assert main([*args, "--length", "0", "--rays", "500"]) == 1
    assert "error: --length: Input should be greater than 0" in capsys.readouterr().err

    # The sample rate is the simulation's, which the wave solver works at.
    assert main([*args, "--rays", "500", "--max-order", "1", "--sample-rate", "16000"]) == 0
    capsys.readouterr()
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["sample_rate"] == 16000
    assert report["settings"]["sample_rate"] == 16000
    assert report["settings"]["early"]["coherent_max_band_hz"] == 125
    assert main([*args, "--rays", "500", "--max-order", "1", "--coherent-band", "none"]) == 0
    capsys.readouterr()
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["early"]["coherent_max_band_hz"] is None
    assert report["settings"]["performance"] == {"threads": 0, "ray_tree": "sah"}
    assert report["settings"]["early"]["diffraction_reflections"] == "one_side"
    both = ["--rays", "500", "--max-order", "1", "--diffraction-reflections", "both_sides"]
    assert main([*args, *both]) == 0
    capsys.readouterr()
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["early"]["diffraction_reflections"] == "both_sides"
    assert report["settings"]["early"]["parabolic_equation"] == "narrow_angle"
    wide = ["--rays", "500", "--max-order", "1", "--parabolic-equation", "wide_angle"]
    assert main([*args, *wide]) == 0
    capsys.readouterr()
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["early"]["parabolic_equation"] == "wide_angle"

    performance = ["--threads", "1", "--ray-tree", "median"]
    assert main([*args, "--rays", "500", "--max-order", "1", *performance]) == 0
    capsys.readouterr()
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["performance"] == {"threads": 1, "ray_tree": "median"}

    from acoustic_spaces.engine import ray_device

    monkeypatch.setattr(ray_device, "_trial", True)
    device = ["--late-device", "cpu", "--no-device-trial"]
    assert main([*args, "--rays", "500", "--max-order", "1", *device]) == 0
    capsys.readouterr()
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["late"]["device"] == "cpu"
    assert not ray_device.trial_enabled()
    for option, value in [
        ("--threads", "-1"),
        ("--seed", "-1"),
        ("--ray-tree", "octree"),
        ("--late-device", "tpu"),
    ]:
        with pytest.raises(SystemExit):
            main([*args, option, value])
        assert option in capsys.readouterr().err


def test_edit_hands_the_device_trial_switch_to_the_editor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import acoustic_spaces.gui

    opened = []
    monkeypatch.setattr(acoustic_spaces.gui, "main", lambda argv: opened.append(argv) or 0)
    assert main(["edit", str(SHOEBOX), "--no-device-trial"]) == 0
    assert main(["edit"]) == 0
    assert opened == [[str(SHOEBOX), "--no-device-trial"], []]


def test_alone_it_opens_the_editor(monkeypatch, capsys) -> None:
    import acoustic_spaces.gui
    from acoustic_spaces import cli

    opened = []
    monkeypatch.setattr(acoustic_spaces.gui, "main", lambda argv: opened.append(argv) or 0)
    monkeypatch.setattr(cli, "_no_editor", lambda: "")
    assert main([]) == 0 and opened == [[]]
    # Without the gui extra or a display: the usage, as for a wrong argument.
    monkeypatch.setattr(cli, "_no_editor", lambda: "there is no display to show it on")
    with pytest.raises(SystemExit) as stopped:
        main([])
    err = capsys.readouterr().err
    assert stopped.value.code == 2 and err.startswith("usage: acoustic-spaces")
    assert "required: command (alone, acoustic-spaces opens the editor, but there is no" in err
    with pytest.raises(SystemExit) as stopped:
        main(["--frobnicate"])
    assert stopped.value.code == 2 and capsys.readouterr().err.startswith("usage:")
    assert opened == [[]]

    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "PySide6", None)  # not installed
    assert cli._no_editor() == acoustic_spaces.gui.INSTALL_HINT
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(sys, "platform", "linux")
    for name in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    assert cli._no_editor() == "there is no display to show it on"
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert cli._no_editor() == ""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert cli._no_editor() == ""


def test_simulate_in_third_octave_bands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    out = tmp_path / "out"
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--rays", "500", "--length", "0.3"]
    assert main([*args, "--max-order", "1", "--bands", "third_octave"]) == 0
    assert "T30 (s)" in capsys.readouterr().out
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["bands"] == "third_octave"
    (response,) = report["responses"]
    assert len(response["parameters"]["c80"]) == 8  # parameters stay in octave bands
    with pytest.raises(SystemExit):
        main([*args, "--bands", "sixth_octave"])
    capsys.readouterr()


def test_simulate_with_the_wave_solver(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    out = tmp_path / "out"
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--rays", "1000", "--length", "0.4"]
    wave = ["--wave", "--wave-band", "125", "--wave-device", "auto", "--wave-wind", "moving"]
    assert main([*args, *wave, "--output", "ambisonics:1"]) == 0
    captured = capsys.readouterr()
    assert "talker -> seat-a: direct sound at" in captured.out
    assert "63-125 Hz from the wave solver" in captured.out
    assert "talker: wave solver, 63-125 Hz (" in captured.err
    report = json.loads((out / "parameters.json").read_text(encoding="utf-8"))
    assert report["settings"]["wave"] == {
        "enabled": True,
        "max_band_hz": 125,
        "points_per_wavelength": 8.0,
        "scheme": "auto",
        "wind": "moving",
        "device": "auto",
    }
    assert report["details"]["talker"]["wave_stencil"] == 19
    assert "wave_moving_air" not in report["details"]["talker"]  # no wind in a room
    assert report["details"]["talker"]["wave_cells"] > 0
    assert main([*args, "--no-wave"]) == 0
    assert "wave solver" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*args, "--wave-band", "100"])


def test_simulate_writes_spatial_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from acoustic_spaces.audio import read_wav

    out = tmp_path / "out"
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--rays", "1000", "--length", "0.3"]
    formats = ["binaural", "ambisonics:2", "loudspeakers:5.0"]
    assert main([*args, "--seed", "1", *(f"--output={f}" for f in formats)]) == 0
    capsys.readouterr()
    channels = {"binaural": 2, "ambix2": 9, "5.0": 5}
    for label, count in channels.items():
        data, fs = read_wav(out / f"talker__seat-a.{label}.wav")
        assert data.shape == (int(0.3 * fs), count)
    assert not (out / "talker__seat-a.wav").exists()
    (response,) = json.loads((out / "parameters.json").read_text(encoding="utf-8"))["responses"]
    assert [o["format"] for o in response["outputs"]] == list(channels)
    assert all(0.0 < v <= 1.0 for v in response["parameters"]["iacc_e"])  # from the ears
    assert response["outputs"][2]["channels"] == ["L", "R", "C", "Ls", "Rs"]
    assert main([*args, "--output", "loudspeakers:9.1"]) == 1
    assert "unknown layout" in capsys.readouterr().err
    assert main([*args, "--hrtf", str(tmp_path / "missing.sofa")]) == 1


def test_map_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    out = tmp_path / "map.json"
    args = ["map", str(SHOEBOX), "--spacing", "2", "--rays", "2000", "--length", "0.8"]
    assert main([*args, "--seed", "4", "--noise", "35", "--json", str(out)]) == 0
    stdout = capsys.readouterr().out
    assert "12 map points 2 m apart, 1.2 m above the audience surfaces" in stdout
    assert "talker (bands: 1 kHz)" in stdout
    assert "STI" in stdout and "Ts (ms)" in stdout
    text = out.read_text(encoding="utf-8")
    assert text.startswith('{\n  "') and text.endswith("}\n")  # as the other JSON files
    report = json.loads(text)
    assert report["settings"]["noise_db"] == 35.0 and report["settings"]["rays"] == 2000
    assert len(report["points"]) == 12
    assert len(report["maps"]["talker"]["values"]["c80"][0]) == 8

    if importlib.util.find_spec("matplotlib") is not None:
        plot = tmp_path / "map.png"
        assert main([*args, "--parameter", "sti", "--plot", str(plot)]) == 0
        assert plot.stat().st_size > 10_000

    assert main([*args, "--surfaces", "ceiling"]) == 1
    assert "no map point lies inside the room" in capsys.readouterr().err
    assert main([*args, "--noise", "30", "40"]) == 1
    assert "noise_db" in capsys.readouterr().err


def test_map_checks_the_options_that_override_the_scene(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["map", str(SHOEBOX), "--spacing", "2", "--length", "0.3"]
    for option, value, message in [
        ("--rays", "0", "greater than or equal to 1"),
        ("--length", "0", "greater than 0"),
        ("--max-order", "-1", "greater than or equal to 0"),
        ("--spacing", "-1", "greater than 0"),
    ]:
        assert main([*args, option, value]) == 1
        err = capsys.readouterr().err
        assert f"error: {option}: Input should be {message}" in err
        assert "validation error" not in err and "errors.pydantic.dev" not in err
    assert main([*args, "--noise", "30", "40"]) == 1
    assert "error: --noise: expected 8 noise_db values, got 2" in capsys.readouterr().err


def _write_recording(sweep_wav: Path, out: Path, latency: int = 0) -> None:
    """Play the sweep through a synthetic room: direct sound and a 0.5 s decay."""
    import math

    import numpy as np

    from acoustic_spaces.audio import read_wav, write_wav

    x, fs = read_wav(sweep_wav)
    rng = np.random.default_rng(5)
    t = np.arange(fs) / fs
    room = 0.05 * rng.standard_normal(fs) * np.exp(-3 * math.log(10) * t / 0.5)
    room[:120] = 0.0
    room[120] = 1.0
    n_fft = 1 << (len(x) + fs).bit_length()
    y = np.fft.irfft(np.fft.rfft(x[:, 0], n_fft) * np.fft.rfft(room, n_fft), n_fft)
    played = np.concatenate([np.zeros(latency), x[:, 0]])[: len(x)]
    heard = np.concatenate([np.zeros(latency), y])[: len(x)]
    write_wav(out, np.column_stack([heard, played]), fs)


def test_sweep_deconvolve_and_analyze(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    from acoustic_spaces.audio import read_wav

    sweep = tmp_path / "new" / "sweep.wav"  # the folder is made
    assert main(["sweep", str(sweep), "--duration", "1.5", "--silence", "1", "--repeats", "2"]) == 0
    assert (tmp_path / "new" / "sweep.json").exists()
    _write_recording(sweep, tmp_path / "recording.wav", latency=48)
    ir = tmp_path / "ir.wav"
    args = ["deconvolve", str(tmp_path / "recording.wav"), "--sweep", str(sweep)]
    assert main([*args, "--out", str(ir), "--loopback", "2"]) == 0
    assert "latency from the loopback channel: 1.00 ms" in capsys.readouterr().out
    data, fs = read_wav(ir)
    assert data.shape == (fs, 1)

    report = tmp_path / "ir.json"
    assert main(["analyze", str(ir), "--reference-energy", "1", "--json", str(report)]) == 0
    out = capsys.readouterr().out
    assert "T30 (s)" in out
    channel = json.loads(report.read_text())["channels"][0]
    assert channel["onset_s"] == pytest.approx(120 / fs, abs=5 / fs)  # side lobes
    t30 = [v for v in channel["parameters"]["t30"][2:] if v is not None]
    assert len(t30) == 6 and all(abs(v - 0.5) < 0.05 for v in t30)


def test_deconvolve_rejects_a_mismatched_sample_rate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np

    from acoustic_spaces.audio import write_wav

    main(["sweep", str(tmp_path / "sweep.wav"), "--duration", "1", "--silence", "0.5"])
    write_wav(tmp_path / "rec.wav", np.zeros(1000), 44_100)
    args = ["deconvolve", str(tmp_path / "rec.wav"), "--sweep", str(tmp_path / "sweep.json")]
    assert main([*args, "--out", str(tmp_path / "ir.wav")]) == 1
    assert "44100 Hz" in capsys.readouterr().err


def _arrivals(tmp_path: Path) -> Path:
    """First-order AmbiX: the direct sound from the front, a reflection from the left
    20 ms later at half its energy, and a tail decaying in 0.5 s from all around."""
    import numpy as np

    from acoustic_spaces.audio import write_wav

    fs, rng = 48_000, np.random.default_rng(0)
    t = np.arange(fs) / fs
    tail = rng.standard_normal((fs, 3)) * np.exp(-6.9 * t / 0.5)[:, None] * 0.02
    tail[: int(0.02 * fs)] = 0.0
    xyz = tail / np.sqrt(3.0)
    w = tail.sum(axis=1) / np.sqrt(3.0)
    w[480] += 1.0
    xyz[480, 0] += 1.0
    w[1440] += math.sqrt(0.5)
    xyz[1440, 1] += math.sqrt(0.5)
    path = tmp_path / "ambix.wav"
    write_wav(path, np.column_stack([w, xyz[:, 1], xyz[:, 2], xyz[:, 0]]), fs)
    return path


def test_analyze_ambisonic_binaural_and_lateral_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    path = _arrivals(tmp_path)
    report = tmp_path / "ambix.json"
    assert main(["analyze", str(path), "--ambisonic", "--stage", "--json", str(report)]) == 0
    out = capsys.readouterr().out
    assert "ambix.wav, W" in out and "T30_X (s)" in out and "ST_early (dB)" in out
    (channel,) = json.loads(report.read_text(encoding="utf-8"))["channels"]
    lf = channel["parameters"]["lf"][4]  # 1 kHz
    assert lf == pytest.approx(0.5 / 1.5, abs=0.05)
    assert 0.0 < channel["speech"]["sti"] <= 1.0 and channel["echo"]["speech"]["ec"] > 0
    # The same W with its figure of eight as a separate channel.
    assert main(["analyze", str(path), "--lateral", "2", "--json", str(report)]) == 0
    assert "channel 1" in capsys.readouterr().out
    (again,) = json.loads(report.read_text(encoding="utf-8"))["channels"]
    assert again["parameters"]["lf"] == channel["parameters"]["lf"]
    # Two of the channels as ears.
    assert main(["analyze", str(path), "--binaural", "--json", str(report)]) == 0
    assert "left ear" in capsys.readouterr().out
    ears = json.loads(report.read_text(encoding="utf-8"))["channels"]
    assert len(ears) == 2 and ears[0]["parameters"]["iacc_e"] == ears[1]["parameters"]["iacc_e"]
    for options, message in (
        (["--ambisonic", "--channel", "1"], "--channel does not go with --ambisonic"),
        (["--lateral", "9"], "lateral channel"),
    ):
        assert main(["analyze", str(path), *options]) == 1
        assert message in capsys.readouterr().err


def test_analyze_ambisonic_needs_four_channels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import numpy as np

    from acoustic_spaces.audio import write_wav

    write_wav(tmp_path / "stereo.wav", np.eye(4800, 2), 48_000)
    assert main(["analyze", str(tmp_path / "stereo.wav"), "--ambisonic"]) == 1
    assert "--ambisonic needs 4 channels, the file has 2" in capsys.readouterr().err


def test_deconvolve_and_analyze_report_unreadable_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["sweep", str(tmp_path / "sweep.wav"), "--duration", "1", "--silence", "0.5"])
    capsys.readouterr()
    (tmp_path / "notes.wav").write_text("not audio", encoding="utf-8")
    sweep = ["--sweep", str(tmp_path / "sweep.json"), "--out", str(tmp_path / "ir.wav")]
    # These ended in tracebacks.
    for args, message in [
        (["deconvolve", str(tmp_path / "missing.wav"), *sweep], "missing.wav"),
        (["deconvolve", str(tmp_path / "notes.wav"), *sweep], "notes.wav: not a WAV file"),
        (["analyze", str(tmp_path / "missing.wav")], "missing.wav"),
        (["analyze", str(tmp_path / "notes.wav")], "notes.wav: not a WAV file"),
    ]:
        assert main(args) == 1
        err = capsys.readouterr().err
        assert err.startswith("error: ") and message in err


def test_deconvolve_and_analyze_make_their_output_folders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    sweep = tmp_path / "sweep.wav"
    main(["sweep", str(sweep), "--duration", "1", "--silence", "0.5"])
    _write_recording(sweep, tmp_path / "recording.wav")
    ir = tmp_path / "irs" / "ir.wav"
    args = ["deconvolve", str(tmp_path / "recording.wav"), "--sweep", str(sweep)]
    assert main([*args, "--out", str(ir)]) == 0
    assert ir.exists()
    report = tmp_path / "reports" / "ir.json"
    args = ["analyze", str(ir), "--channel", "1", "--json", str(report)]
    if importlib.util.find_spec("matplotlib") is not None:
        args += ["--plot", str(tmp_path / "plots" / "ir.png")]
    assert main(args) == 0
    capsys.readouterr()
    assert json.loads(report.read_text(encoding="utf-8"))["channels"][0]["channel"] == 1
    if importlib.util.find_spec("matplotlib") is not None:
        assert (tmp_path / "plots" / "ir.png").exists()


def test_measure_with_a_mocked_audio_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    import numpy as np

    from acoustic_spaces import measurement
    from acoustic_spaces.audio import read_wav

    def play_and_record(signal, sample_rate, outputs, inputs, device):
        assert list(inputs) == [1, 2] and list(outputs) == [1] and device == "usb"
        delayed = np.concatenate([np.zeros(96), signal])[: len(signal)]
        return np.column_stack([0.5 * delayed, delayed])

    monkeypatch.setattr(measurement, "play_and_record", play_and_record)
    out = tmp_path / "m"
    args = ["measure", "--out", str(out), "--device", "usb", "--input", "1", "--loopback", "2"]
    assert main([*args, "--duration", "1", "--silence", "0.5", "--no-noise"]) == 0
    assert "latency from the loopback channel: 2.00 ms" in capsys.readouterr().out
    ir, fs = read_wav(out / "ir.wav")
    assert ir.shape == (fs // 2, 1)
    assert np.argmax(np.abs(ir[:, 0])) == 0
    report = json.loads((out / "measurement.json").read_text())
    assert report["latency_s"] == pytest.approx(0.002)
    assert {p.name for p in out.iterdir()} == {
        "sweep.wav",
        "sweep.json",
        "recording.wav",
        "ir.wav",
        "measurement.json",
    }


def test_devices_without_sounddevice(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    assert main(["devices"]) == 1
    assert 'pip install "acoustic-spaces[measure]"' in capsys.readouterr().err


def _decay_wav(path: Path, t60: float, sample_rate: int, length: float = 2.0, channels: int = 1):
    """A response that decays exactly exponentially in every octave band (a tone at each
    mid-band frequency under one envelope, after 10 ms of silence), in every channel."""
    import math

    import numpy as np

    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.parameters import octave_midbands

    t = np.arange(int(length * sample_rate)) / sample_rate
    tones = sum(np.cos(2 * np.pi * f * t + 0.7 * b) for b, f in enumerate(octave_midbands()))
    decay = 0.1 * tones * np.exp(-3.0 * math.log(10.0) * t / t60)
    ir = np.concatenate([np.zeros(sample_rate // 100), decay])
    write_wav(path, np.tile(ir[:, None], (1, channels)), sample_rate)


def test_compare_a_simulated_response_with_a_measured_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    # Simulated 10 % longer than measured, at another sample rate: 2 JNDs.
    _decay_wav(tmp_path / "sim.wav", 1.1, 44_100)
    _decay_wav(tmp_path / "meas.wav", 1.0, 48_000, channels=2)
    report = tmp_path / "reports" / "compare.json"  # the folder is made
    args = ["compare", str(tmp_path / "sim.wav"), str(tmp_path / "meas.wav")]
    args += ["--measured-channel", "2", "--json", str(report)]
    assert main([*args, "--simulated-reference-energy", "0.01"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""  # the channel was picked: no note
    lines = captured.out.splitlines()
    assert lines[:2] == [
        "simulated: sim.wav, channel 1, 44100 Hz, direct sound at 10.0 ms",
        "measured:  meas.wav, channel 2, 48000 Hz, direct sound at 10.0 ms",
    ]
    t30 = next(i for i, line in enumerate(lines) if line.startswith("T30 (s)"))
    assert lines[t30].split() == ["T30", "(s)", "sim"] + ["1.100"] * 8
    assert lines[t30 + 1].split() == ["meas"] + ["1.000"] * 8
    assert lines[t30 + 2].split() == ["JNDs"] + ["+2.0*"] * 8  # marked: beyond one JND
    assert "  G in every band: no reference for the measured G" in lines
    summary = next(line for line in reversed(lines) if line.startswith("T30 (s)"))
    assert summary.split() == ["T30", "(s)", "0", "of", "8", "2.0"]

    text = report.read_text(encoding="utf-8")
    assert text.startswith('{\n  "simulated": {') and text.endswith("}\n")
    data = json.loads(text)
    assert data["simulated"] == {
        "file": "sim.wav",
        "channel": 1,
        "sample_rate": 44_100,
        "onset_s": pytest.approx(0.01, abs=1e-4),
    }
    assert data["measured"]["channel"] == 2 and data["measured"]["sample_rate"] == 48_000
    assert data["bands_hz"] == [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    t30 = data["parameters"]["t30"]
    assert t30["difference_jnd"] == [pytest.approx(2.0, abs=0.02)] * 8
    assert (t30["compared"], t30["within_jnd"]) == (8, 0)
    assert t30["mean_abs_jnd"] == pytest.approx(2.0, abs=0.02)
    g = data["parameters"]["g"]
    assert g["difference"] == [None] * 8 and g["mean_abs_jnd"] is None
    assert g["missing"] == ["no reference for the measured G"] * 8

    # With a reference on both sides, G is compared too; a free-field response works
    # as for analyze.
    from acoustic_spaces.audio import write_wav

    impulse = [0.0] * 4800
    impulse[100] = 1.0
    write_wav(tmp_path / "ff.wav", impulse, 48_000)
    reference = ["--measured-free-field", str(tmp_path / "ff.wav"), "--measured-distance", "1"]
    assert main([*args, "--simulated-reference-energy", "0.01", *reference]) == 0
    capsys.readouterr()
    g = json.loads(report.read_text(encoding="utf-8"))["parameters"]["g"]
    assert g["compared"] == 8 and None not in g["difference_jnd"]


def test_compare_reports_missing_bands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import json

    # The measured response stops after 0.15 s of a 2 s decay: less than 5 dB.
    _decay_wav(tmp_path / "sim.wav", 2.0, 48_000)
    _decay_wav(tmp_path / "meas.wav", 2.0, 48_000, length=0.15, channels=2)
    report = tmp_path / "compare.json"
    args = ["compare", str(tmp_path / "sim.wav"), str(tmp_path / "meas.wav")]
    assert main([*args, "--json", str(report)]) == 0
    captured = capsys.readouterr()
    assert "meas.wav has 2 channels; comparing the first (--measured-channel" in captured.err
    lines = captured.out.splitlines()
    c80 = next(i for i, line in enumerate(lines) if line.startswith("C80 (dB)"))
    assert lines[c80 + 1].split() == ["meas"] + ["-"] * 8
    assert lines[c80 + 2].split() == ["JNDs"] + ["-"] * 8  # missing, not zero
    assert (
        "  EDT, T20, T30, C50, C80, D50, Ts in every band: "
        "the measured response decays by less than 20 dB"
    ) in lines
    summary = next(line for line in reversed(lines) if line.startswith("C80 (dB)"))
    assert summary.split() == ["C80", "(dB)", "0", "of", "0", "-"]
    c80 = json.loads(report.read_text(encoding="utf-8"))["parameters"]["c80"]
    assert c80["measured"] == c80["difference_jnd"] == [None] * 8
    assert c80["missing"] == ["the measured response decays by less than 20 dB"] * 8


def test_compare_reports_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _decay_wav(tmp_path / "ir.wav", 1.0, 48_000, length=0.5)
    _decay_wav(tmp_path / "ff44.wav", 1.0, 44_100, length=0.1)
    (tmp_path / "notes.wav").write_text("not audio", encoding="utf-8")
    ir, missing = str(tmp_path / "ir.wav"), str(tmp_path / "missing.wav")
    for args, message in [
        ([missing, ir], "missing.wav"),
        ([ir, missing], "missing.wav"),
        ([ir, str(tmp_path / "notes.wav")], "notes.wav: not a WAV file"),
        ([ir, ir, "--measured-channel", "2"], "measured channel 2 is not among channels 1 to 1"),
        ([ir, ir, "--simulated-channel", "0"], "simulated channel 0 is not among channels"),
        (
            [ir, ir, "--measured-free-field", ir],
            "--measured-free-field needs --measured-distance",
        ),
        (
            [ir, ir, "--simulated-free-field", str(tmp_path / "ff44.wav")]
            + ["--simulated-distance", "2"],
            "ff44.wav is at 44100 Hz, the simulated response at 48000 Hz",
        ),
    ]:
        assert main(["compare", *args, "--json", str(tmp_path / "c.json")]) == 1
        err = capsys.readouterr().err
        assert err.startswith("error: ") and message in err, err
    assert not (tmp_path / "c.json").exists()


def test_compare_plots_both_decays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    _decay_wav(tmp_path / "sim.wav", 1.1, 44_100, length=1.0)
    _decay_wav(tmp_path / "meas.wav", 1.0, 48_000, length=1.0)
    plot = tmp_path / "plots" / "compare.png"
    args = ["compare", str(tmp_path / "sim.wav"), str(tmp_path / "meas.wav"), "--plot", str(plot)]
    if importlib.util.find_spec("matplotlib") is not None:
        assert main(args) == 0
        assert f"saved {plot}" in capsys.readouterr().out
        assert plot.stat().st_size > 10_000
        plot.unlink()
    # Without matplotlib the comparison is still printed, with a warning.
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "T30 (s)" in captured.out and "saved" not in captured.out
    assert captured.err.startswith("warning: response plots need matplotlib")
    assert not plot.exists()


@pytest.fixture(scope="module")
def simulated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sim")
    args = ["simulate", str(SHOEBOX), "--out", str(out), "--rays", "1000", "--length", "0.3"]
    assert main([*args, "--seed", "1", "--output=mono", "--output=loudspeakers:5.0"]) == 0
    return out


def _dry(path: Path, sample_rate: int = 44_100, channels: int = 1) -> None:
    import numpy as np

    from acoustic_spaces.audio import write_wav

    click = np.zeros((sample_rate // 10, channels))
    click[0] = 0.5
    write_wav(path, click, sample_rate)


def test_convolve_and_auralize(
    simulated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from acoustic_spaces.audio import channel_mask, read_wav

    _dry(tmp_path / "dry.wav", channels=2)
    ir, fs = read_wav(simulated / "talker__seat-a.5.0.wav")
    wet = tmp_path / "wet.wav"
    assert (
        main(
            [
                "convolve",
                str(tmp_path / "dry.wav"),
                str(simulated / "talker__seat-a.5.0.wav"),
                "--out",
                str(wet),
            ]
        )
        == 0
    )
    err = capsys.readouterr().err
    assert "mixed the 2 channels" in err and "from 44100 Hz to 48000 Hz" in err
    data, rate = read_wav(wet)
    assert rate == fs and data.shape == (len(ir) + fs // 10 - 1, 5)
    assert channel_mask(wet) == channel_mask(simulated / "talker__seat-a.5.0.wav")

    mix = tmp_path / "mix.wav"
    args = ["auralize", str(simulated), f"talker={tmp_path / 'dry.wav'}", "--out", str(mix)]
    assert main([*args, "--normalize"]) == 0
    assert "peak -1.0 dBFS" in capsys.readouterr().out
    data, _ = read_wav(mix)
    assert data.shape[1] == 1
    assert abs(abs(data).max() - 10 ** (-1 / 20)) < 1e-6
    assert main([*args, "--format", "binaural"]) == 1
    assert "the results have 5.0, mono" in capsys.readouterr().err
    assert main([*args[:2], "piano=x.wav", "--out", str(mix)]) == 1
    assert "no source 'piano'" in capsys.readouterr().err
    assert main([*args, "--listener", "seat-b"]) == 1


def test_play_command(
    simulated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_sounddevice
) -> None:
    ir = str(simulated / "talker__seat-a.wav")
    _dry(tmp_path / "dry.wav", sample_rate=48_000)
    assert main(["play", ir]) == 1
    assert "give a signal to play or --live" in capsys.readouterr().err
    assert main(["play", ir, str(tmp_path / "dry.wav"), "--block", "512"]) == 0
    captured = capsys.readouterr()
    assert "as loud as the dry signal" in captured.err
    assert "dropped out" in captured.err  # the fake sound card reports one
    assert "played 0.4 s" in captured.out
    assert main(["play", ir, "--live", "--seconds", "0.1", "--absolute"]) == 0
    assert "as loud" not in capsys.readouterr().err


def test_play_with_head_tracking(
    simulated: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_sounddevice
) -> None:
    import numpy as np

    from acoustic_spaces.audio import write_wav
    from acoustic_spaces.spatial import Ambisonics, from_angles

    ir = np.zeros((2_400, 4))
    ir[100] = 0.5 * Ambisonics(1).gains(from_angles(90.0, 0.0)[None])[0]
    write_wav(tmp_path / "ambix1.wav", ir, 48_000)
    _dry(tmp_path / "dry.wav", sample_rate=48_000)
    args = ["play", str(tmp_path / "ambix1.wav"), str(tmp_path / "dry.wav"), "--head-tracking"]
    assert main([*args, "--osc-port", "0", "--invert", "yaw, Pitch"]) == 0
    captured = capsys.readouterr()
    assert "response scaled by +6.0 dB" in captured.err
    assert "(order 1 Ambisonics) to the ears, turned by OSC head-tracker messages" in captured.err
    assert "on 127.0.0.1 port" in captured.err
    assert "0 head-tracker message(s) received" in captured.out
    played = np.concatenate(fake_sounddevice.played)
    assert played.shape[1] == 2 and np.abs(played[:, 0]).max() > 2 * np.abs(played[:, 1]).max()

    assert main([*args, "--osc-port", "0", "--invert", "nod"]) == 1
    assert "cannot invert nod" in capsys.readouterr().err
    mono = ["play", str(simulated / "talker__seat-a.wav"), str(tmp_path / "dry.wav")]
    assert main([*mono, "--head-tracking", "--osc-port", "0"]) == 1
    assert "order 1 or more: 4, 9, 16... channels, not 1" in capsys.readouterr().err


def _test_set(tmp_path: Path):
    """A set of one tiny file next to the tests, so nothing is downloaded."""
    from acoustic_spaces.spatial import hrtf_sets

    source = tmp_path / "source" / "ears.sofa"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"SOFA" + bytes(64))
    return hrtf_sets.HrtfSet(
        name="test",
        title="Test ears",
        ears="a test",
        licence="CC0 1.0",
        credit="the tests",
        page="https://example.invalid/ears",
        url=source.as_uri(),
        size_mb=0.0001,
        note="nothing much in it",
    )


def test_hrtf_lists_the_sets_and_downloads_one(tmp_path: Path, capsys, monkeypatch) -> None:
    """acoustic-spaces hrtf names every set with its licence, says which are there, and
    downloads the one asked for."""
    from acoustic_spaces.spatial import hrtf_sets

    folder = tmp_path / "sets"
    assert main(["hrtf", "--folder", str(folder)]) == 0
    listed = capsys.readouterr().out
    for item in hrtf_sets.sets():
        assert item.name in listed and item.licence in listed and item.page in listed
    assert "not here" not in listed and "[have]" not in listed  # nothing downloaded yet

    monkeypatch.setattr(hrtf_sets, "SETS", (_test_set(tmp_path),))
    assert main(["hrtf", "test", "--folder", str(folder)]) == 0
    written = capsys.readouterr().out.strip()
    assert written == str(folder / "test.sofa")
    assert (folder / "test.sofa").read_bytes().startswith(b"SOFA")

    # It is not downloaded again, and the list marks it.
    assert main(["hrtf", "test", "--folder", str(folder)]) == 0
    assert "there already" in capsys.readouterr().out
    assert main(["hrtf", "--folder", str(folder)]) == 0
    assert "[have] test" in capsys.readouterr().out

    # A name that is not on offer says what is.
    assert main(["hrtf", "nope", "--folder", str(folder)]) == 1
    assert "no HRTF set 'nope'; try: test" in capsys.readouterr().err


def test_hrtf_downloads_another_sofa_file(tmp_path: Path, capsys) -> None:
    """--url takes any other SOFA file, saved under the name given."""
    source = tmp_path / "mine.sofa"
    source.write_bytes(b"SOFA" + bytes(16))
    folder = tmp_path / "sets"
    assert main(["hrtf", "me", "--url", source.as_uri(), "--folder", str(folder)]) == 0
    assert capsys.readouterr().out.strip() == str(folder / "me.sofa")
    assert (folder / "me.sofa").read_bytes() == source.read_bytes()
    assert main(["hrtf", "--url", "file:///nothing/here.sofa", "--folder", str(folder)]) == 1
    assert "error:" in capsys.readouterr().err
