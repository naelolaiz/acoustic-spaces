"""Write a simulation's responses to a folder: the files ``acoustic-spaces simulate``
and the editor's Export produce.

For every (source, listener) pair, one WAV file per output format
(``<source>__<listener>.wav`` for mono, ``<source>__<listener>.<format>.wav`` for the
others), and for the whole run ``parameters.json``: the settings, run details and the
ISO 3382-1 parameters of every response (with IACC when a binaural response is written),
its STI, STIPA and U50, and its echo criteria. ``acoustic-spaces auralize`` reads that
folder.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from acoustic_spaces import parameters as iso3382
from acoustic_spaces.audio import write_wav
from acoustic_spaces.materials import OCTAVE_BANDS_HZ

if TYPE_CHECKING:
    from acoustic_spaces.engine import SimulationResult
    from acoustic_spaces.spatial import Format

REPORT_FILE = "parameters.json"


def file_name(text: str) -> str:
    """``text`` with anything but letters, digits, dots, dashes and underscores replaced."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unnamed"


def response_stem(source: str, listener: str) -> str:
    return f"{file_name(source)}__{file_name(listener)}"


def write_results(
    result: SimulationResult,
    out: str | Path,
    formats: Sequence[Format] | None = None,
    *,
    sample_rate: int | None = None,
    normalize: bool = False,
    plots: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Write every response of ``result`` into the folder ``out`` and return the report
    saved as ``parameters.json``.

    ``formats`` are formats of :mod:`acoustic_spaces.spatial` (default: the scene's
    ``outputs``); ``sample_rate`` defaults to the simulation's. ``normalize`` scales all
    files together so the loudest sample is at -1 dBFS; otherwise they keep the
    simulation's levels, which exceed full scale when a listener is closer than 1 m to
    a source (the report's ``peak`` says by how much). ``plots`` adds the echogram and
    direction plots of :mod:`acoustic_spaces.viz` when matplotlib is installed.
    """
    from acoustic_spaces.spatial import from_output

    say = progress or (lambda _: None)
    if formats is None:
        formats = [from_output(o) for o in result.room.scene.outputs]
    sample_rate = int(sample_rate or result.settings.sample_rate)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    irs = {}
    for key, response in result.responses.items():
        for fmt in formats:
            say(f"{key[0]} -> {key[1]}: rendering {fmt.label}")
            irs[key, fmt.label] = response.spatial_impulse_response(fmt, sample_rate)
    peak = max((float(abs(ir).max(initial=0.0)) for ir in irs.values()), default=0.0)
    gain = 10.0 ** (-1.0 / 20.0) / peak if normalize and peak > 0 else 1.0

    scene = result.room.scene
    sources = {s.name: s for s in scene.sources}
    noise_db = scene.coverage.noise_db if scene.coverage is not None else None
    report = {
        "scene": result.room.scene.name,
        "bands_hz": list(OCTAVE_BANDS_HZ),
        "sample_rate": sample_rate,
        "ir_gain": gain,
        "peak": peak * gain,
        "settings": result.settings.model_dump(mode="json"),
        "details": result.details,
        "responses": [],
    }
    for (source, listener), response in result.responses.items():
        stem = response_stem(source, listener)
        written = []
        for fmt in formats:
            label = fmt.label
            name = f"{stem}.wav" if label == "mono" else f"{stem}.{file_name(label)}.wav"
            mask = fmt.channel_mask() if hasattr(fmt, "channel_mask") else None
            write_wav(out / name, irs[(source, listener), label] * gain, sample_rate, mask)
            written.append({"format": label, "wav": name, "channels": fmt.channel_names})
        params = response.parameters()
        binaural = [fmt.label for fmt in formats if fmt.label == "binaural"]
        if binaural:
            ears = irs[(source, listener), binaural[0]]
            params = dataclasses.replace(
                params, **iso3382.interaural_cross_correlation(ears, sample_rate)
            )
        if ((source, listener), "mono") in irs:
            mono = irs[(source, listener), "mono"][:, 0]
        else:
            mono = response.impulse_response(sample_rate)
        power = sources[source].power_db("octave") if source in sources else 94.0
        talker = sources[source].talker if source in sources else "male"
        spl = response.levels(power, params)
        entry = {
            "source": source,
            "listener": listener,
            "outputs": written,
            "direct_delay_s": response.direct_delay_s,
            "early_paths": len(response.early),
            "diffracted_paths": int(response.early.diffracted.sum()),
            "parameters": params.as_dict(),
            "spl_db": [_plain(v) for v in spl],
            "spl_a_db": _plain(iso3382.a_weighted(spl)),
            "speech": response.speech(power, noise_db, parameters=params, talker=talker).as_dict(),
            "echo": _echo(mono, sample_rate),
        }
        if plots:
            try:
                from acoustic_spaces.viz import plot_directions, plot_response

                plot_response(response, out / f"{stem}.png")
                plot_directions(response, out / f"{stem}.directions.png")
                entry["plot"] = f"{stem}.png"
                entry["directions_plot"] = f"{stem}.directions.png"
            except ImportError as exc:
                say(f"warning: {exc}")
                plots = False
        report["responses"].append(entry)
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _plain(value: float) -> float | None:
    """A level for JSON: None where it is not finite."""
    value = float(value)
    return round(value, 2) if math.isfinite(value) else None


def _echo(ir, sample_rate: int) -> dict:
    """The echo criteria of a mono impulse response, for the report."""
    out = {}
    for kind in iso3382.ECHO_CRITERIA:
        value, at = iso3382.echo_criterion(ir, sample_rate, kind)
        out[kind] = {
            "ec": value if math.isfinite(value) else None,
            "at_s": at if math.isfinite(at) else None,
        }
    return out


__all__ = ["REPORT_FILE", "file_name", "response_stem", "write_results"]
