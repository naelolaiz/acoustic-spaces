"""Beethoven's Symphony No. 8 for the catalogue of recordings: the anechoic recording of
the TU Berlin (Böhm, Ackermann and Weinzierl, Orchester Wiener Akademie; CC BY 4.0,
https://doi.org/10.14279/depositonce-6729.2), a track per player, 61 of them, of the
three movements it has (I, II and IV).

Each movement comes as a zip (movement1.zip, movement2.zip, movement4.zip, each the
:attr:`~acoustic_spaces.recordings.Piece.archive` of its piece; 2.0 GB for the three):
:func:`build` reads a movement's 61 tracks out of it on the TU Berlin's server by HTTP
range requests, without keeping the zip, or out of a copy downloaded before, and keeps
them as they were recorded, each the stem of the source named after its player
(:data:`TRACKS`), so that the balance between the players is the recording's own.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np

from acoustic_spaces import fetch, recordings
from acoustic_spaces.demos.audio import mono, read, write

#: Where the TU Berlin's server (DepositOnce) keeps each movement's zip, by the piece's
#: name; it answers range requests.
BITSTREAMS = "https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/"
ZIPS = {
    "I": BITSTREAMS + "9fc848cb-33ec-435f-a6ea-bb73294a01b8/content",
    "II": BITSTREAMS + "5e704ff9-53c0-4378-a8ef-5d2e6cdc8388/content",
    "IV": BITSTREAMS + "2b54758e-148e-439d-b687-2a960d704611/content",
}
#: The folder of the tracks in the zip, and in the folder a copy was unpacked into.
ANECHOIC = "anechoic"
#: Bytes a range request reads of a zip: the tracks lie one after the other in it, and
#: the server takes about half a second to answer each request (in blocks of 4 MiB the
#: three movements took 622 s, in these 189 s).
BLOCK = 32 << 20
#: Each player's stem and the track it is in the zips, as auralize.sh has them.
TRACKS = {
    "fl1": "01_Fl_1.wav",
    "fl2": "02_Fl_2.wav",
    "ob1": "03_Ob_1.wav",
    "ob2": "04_Ob_2.wav",
    "cl1": "05_Cl_1.wav",
    "cl2": "06_Cl_2.wav",
    "bn1": "07_Bn_1.wav",
    "bn2": "08_Bn_2.wav",
    "hn1": "09_Hrn_1.wav",
    "hn2": "10_Hrn_2.wav",
    "tp1": "11_Trp_1.wav",
    "tp2": "12_Trp_2.wav",
    "timpani": "13_Tp.wav",
    **{
        f"vl1_{k:02d}": f"{13 + k}_Vl_1_{1 + (k - 1) // 7}_{1 + (k - 1) % 7}.wav"
        for k in range(1, 15)
    },
    **{
        f"vl2_{k:02d}": f"{27 + k}_Vl_2_{1 + (k - 1) // 6}_{1 + (k - 1) % 6}.wav"
        for k in range(1, 7)
    },
    **{f"vl2_{k:02d}": f"{27 + k}_Vl_2_2_{k - 6}.wav" for k in range(7, 12)},
    **{f"vla_{k:02d}": f"{38 + k}_Vla_1_{k}.wav" for k in range(1, 7)},
    **{f"vla_{k:02d}": f"{38 + k}_Vla_2_{k - 6}.wav" for k in range(7, 12)},
    **{f"vc_{k}": f"{49 + k}_Vc_{k}.wav" for k in range(1, 7)},
    **{f"db_{k}": f"{55 + k}_Db_{k}.wav" for k in range(1, 7)},
}


def build(
    item: recordings.Recording,
    folder: str | Path | None = None,
    piece: str = "I",
    copy: Path | None = None,
) -> None:
    """The 61 tracks of the movement ``piece`` (not the zip's mix), each the stem of its
    player (:func:`~acoustic_spaces.recordings.stem_file`), as recorded, read out of the
    movement's zip on the TU Berlin's server by range requests of :data:`BLOCK`, or out
    of ``copy`` (:func:`~acoustic_spaces.recordings.copy_of`), the zip downloaded before
    or the folder it was unpacked into, its tracks in :data:`ANECHOIC` or in it; and the
    movement's master, the stems added up. Tracks there already are kept, so that a
    download cut short goes on where it stopped."""
    stems = {name: recordings.stem_file(item, name, folder, piece) for name in TRACKS}
    missing = [name for name, path in stems.items() if not path.exists()]
    if missing and copy is not None and copy.is_dir():
        # The zip unpacked (its tracks in anechoic/), or the tracks themselves, as the
        # orchestra's script kept them before 4.4.0 (demo/tracks/movement1).
        tracks = copy / ANECHOIC if (copy / ANECHOIC).is_dir() else copy
        print(f"copying the tracks from {tracks}")
        absent = [TRACKS[name] for name in missing if not (tracks / TRACKS[name]).exists()]
        if absent:
            raise SystemExit(f"error: {tracks} has no {absent[0]}")
        for name in missing:
            part = stems[name].with_name(stems[name].name + ".part")
            part.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tracks / TRACKS[name], part)
            part.replace(stems[name])
    elif missing:
        source = str(copy) if copy is not None else ZIPS[piece]
        print(f"reading {source}" + ("" if copy is not None else " (range requests)"))
        archive = zipfile.ZipFile(copy) if copy is not None else fetch.open_zip(source, BLOCK)
        with archive:
            members = {Path(name).name: name for name in archive.namelist()}
            absent = [TRACKS[name] for name in missing if TRACKS[name] not in members]
            if absent:
                raise SystemExit(f"error: {source} has no {absent[0]}")
            for k, name in enumerate(missing, 1):
                fetch.copy_member(archive, members[TRACKS[name]], stems[name])
                print(f"  {TRACKS[name]} ({k} of {len(missing)})")
    master = recordings.file_of(item, folder, piece)
    if missing or not master.exists():
        total: np.ndarray | None = None
        rate = 0
        for path in stems.values():
            x, rate = read(path)
            x = mono(x)
            if total is None:
                total = np.zeros(len(x))
            if len(x) > len(total):
                total = np.concatenate([total, np.zeros(len(x) - len(total))])
            total[: len(x)] += x
        part = master.with_name(master.name + ".part")
        write(part, total if total is not None else np.zeros(0), rate)
        part.replace(master)
        print(f"wrote the {len(stems)} stems of {piece} and their sum, {master}")


__all__ = ["ANECHOIC", "BLOCK", "TRACKS", "ZIPS", "build"]
