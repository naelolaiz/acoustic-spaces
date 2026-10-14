import importlib.util
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


def test_inspect(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["inspect", str(SHOEBOX)]) == 0
    out = capsys.readouterr().out
    assert "volume 144.00 m3" in out
    assert "Sabine" in out and "Eyring" in out and "Arau-Puchades" in out
    # The example's absorption sits on the floor and ceiling: its decay is no diffuse one.
    assert "note: the absorption is spread unevenly" in out
    assert "seat-a" in out and "inside" in out


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
    report = json.loads(out.read_text(encoding="utf-8"))
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

    sweep = tmp_path / "sweep.wav"
    assert main(["sweep", str(sweep), "--duration", "1.5", "--silence", "1", "--repeats", "2"]) == 0
    assert (tmp_path / "sweep.json").exists()
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
    assert "acoustic-spaces[measure]" in capsys.readouterr().err


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
