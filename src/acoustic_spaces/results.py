"""Write a simulation's responses to a folder: the files ``acoustic-spaces simulate``
and the editor's Export produce.

For every (source, listener) pair, one WAV file per output format
(``<source>__<listener>.wav`` for mono, ``<source>__<listener>.<format>.wav`` for the
others, with ``-2``, ``-3``... after the names or the format where two files would
otherwise be the same), and for the whole run ``parameters.json``: the settings, run details and the
ISO 3382-1 parameters of every response (with IACC when a binaural response is written,
from the first if there are several), its STI, STIPA and U50, and its echo criteria.
Outdoors also its level relative to free field, and the insertion loss of the buildings,
barriers or vegetation the run was also made without (``simulation.insertion_loss_of``).
``acoustic-spaces auralize`` reads that folder.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
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


def unique_stems(stems: Sequence[str], suffixes: Sequence[str] = ("",)) -> list[str]:
    """``stems``, each that would name the same file as one before it with any of
    ``suffixes`` given ``-2``, ``-3``... in turn, also where the names differ only in
    case (the file systems of Windows and macOS ignore it): :func:`file_name` turns
    ``A/B`` and ``A_B`` into the same name, and names in other scripts into
    ``unnamed``."""
    taken: set[str] = set()
    out = []
    for stem in stems:
        unique, k = stem, 1
        while any((unique + suffix).casefold() in taken for suffix in suffixes):
            k += 1
            unique = f"{stem}-{k}"
        taken.update((unique + suffix).casefold() for suffix in suffixes)
        out.append(unique)
    return out


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
    from acoustic_spaces.spatial import Binaural, from_output

    say = progress or (lambda _: None)
    if formats is None:
        formats = [from_output(o) for o in result.room.scene.outputs]
    sample_rate = int(sample_rate or result.settings.sample_rate)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    from acoustic_spaces.engine.run_size import worker_threads

    # The responses side by side, on the run's threads: each is rendered and read as it
    # would be alone, so the files and the report are the same as one after another.
    threads = worker_threads(result.settings.performance.threads)
    workers = max(1, min(threads, len(result.responses)))

    def render(key) -> dict:
        rendered = {}
        for k, fmt in enumerate(formats):
            say(f"{key[0]} -> {key[1]}: rendering {fmt.label}")
            rendered[key, k] = result.responses[key].spatial_impulse_response(fmt, sample_rate)
        return rendered

    irs = {}
    with ThreadPoolExecutor(workers) as pool:
        for rendered in pool.map(render, result.responses):
            irs.update(rendered)
    peak = max((float(abs(ir).max(initial=0.0)) for ir in irs.values()), default=0.0)
    gain = 10.0 ** (-1.0 / 20.0) / peak if normalize and peak > 0 else 1.0

    # File names that no two responses or formats share, though their names may give
    # the same (two binaural outputs with HRTF files of the same name, say).
    parts = unique_stems(
        ["" if fmt.label == "mono" else f".{file_name(fmt.label)}" for fmt in formats]
    )
    suffixes = [f"{part}.wav" for part in parts] + [".png", ".directions.png"]
    stems = dict(
        zip(
            result.responses,
            unique_stems([response_stem(*key) for key in result.responses], suffixes),
            strict=True,
        )
    )
    binaural = [k for k, fmt in enumerate(formats) if isinstance(fmt, Binaural)]
    mono_at = [k for k, fmt in enumerate(formats) if fmt.label == "mono"]

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
        # What auralize needs to mix the sources at their power: the responses are per
        # unit of it.
        "sources": {
            source.name: {"sound_power_db": source.sound_power_db} for source in scene.sources
        },
        "responses": [],
    }
    if result.removed:
        report["removed"] = list(result.removed)
    long_term = result.long_term
    if long_term is not None:
        report["long_term"] = {
            "classes": list(long_term.classes),
            "shares": {
                name: {
                    period: round(float(share), 6)
                    for period, share in zip(long_term.periods, row, strict=True)
                }
                for name, row in zip(long_term.classes, long_term.shares, strict=True)
            },
            "periods": [
                {"name": name, "hours": float(hours), "penalty_db": float(penalty)}
                for name, hours, penalty in zip(
                    long_term.periods, long_term.hours, long_term.penalties, strict=True
                )
            ],
            "rating": long_term.rating,
        }

    def write(key) -> dict:
        """Write the response's files; its entry in the report, without plots."""
        source, listener = key
        response = result.responses[key]
        stem = stems[key]
        written = []
        for k, (fmt, part) in enumerate(zip(formats, parts, strict=True)):
            name = f"{stem}{part}.wav"
            mask = fmt.channel_mask() if hasattr(fmt, "channel_mask") else None
            write_wav(out / name, irs[key, k] * gain, sample_rate, mask)
            written.append({"format": fmt.label, "wav": name, "channels": fmt.channel_names})
        params = response.parameters()
        if binaural:
            ears = irs[key, binaural[0]]
            params = dataclasses.replace(
                params, **iso3382.interaural_cross_correlation(ears, sample_rate)
            )
        if mono_at:
            mono = irs[key, mono_at[0]][:, 0]
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
        if response.free_field is not None:
            relative = response.relative_to_free_field()
            entry["re_free_field_db"] = [_plain(v) for v in relative]
            entry["re_free_field_a_db"] = _plain(
                iso3382.a_weighted(spl) - iso3382.a_weighted(spl - relative)
            )
        loss = result.insertion_loss.get((source, listener))
        if loss is not None:
            entry["insertion_loss_db"] = [_plain(v) for v in loss]
            entry["insertion_loss_a_db"] = _plain(
                iso3382.a_weighted(spl + loss) - iso3382.a_weighted(spl)
            )
        if long_term is not None and (source, listener) in long_term.levels:
            entry["long_term"] = _long_term_entry(long_term, (source, listener))
        return entry

    with ThreadPoolExecutor(workers) as pool:
        report["responses"] = list(pool.map(write, result.responses))
    # Plots one after another: matplotlib draws in one thread.
    for entry, (key, response) in zip(report["responses"], result.responses.items(), strict=True):
        stem = stems[key]
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
    (out / REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _long_term_entry(long_term, key) -> dict:
    """The levels over a long term at one (source, listener), for the report: the
    long-term level, the rating level, and those in each period and each weather class,
    per octave band and A-weighted."""

    def levels(bands, a) -> dict:
        return {"spl_db": [_plain(v) for v in bands], "spl_a_db": _plain(a)}

    entry = {
        **levels(long_term.levels[key], long_term.levels_a[key]),
        "rating_level_db": _plain(long_term.rating_level[key]),
        "rating": long_term.rating,
        "periods": {
            name: levels(long_term.period_levels[key][k], long_term.period_a[key][k])
            for k, name in enumerate(long_term.periods)
        },
        "classes": {
            name: levels(long_term.class_levels[key][k], long_term.class_a[key][k])
            for k, name in enumerate(long_term.classes)
        },
    }
    for what in ("re_free_field", "insertion_loss"):
        bands = getattr(long_term, what).get(key)
        if bands is not None:
            entry[f"{what}_db"] = [_plain(v) for v in bands]
            entry[f"{what}_a_db"] = _plain(getattr(long_term, f"{what}_a")[key])
    return entry


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


__all__ = ["REPORT_FILE", "file_name", "response_stem", "unique_stems", "write_results"]
