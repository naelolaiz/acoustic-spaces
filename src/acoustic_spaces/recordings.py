"""Free recordings a scene can play in a demo of it: songs in stereo and speech, each
published under a licence that lets anyone use it.

A scene names one by its name as its ``recording`` (:class:`acoustic_spaces.scene.Scene`),
and each source with a ``channel`` plays that channel of it in the demo of the scene
(``acoustic-spaces demo``, the editor's *This scene*). Nothing is fetched until a demo
needs it: :func:`get` downloads the recording into :func:`data_folder` the first time
(the song out of its release by HTTP range requests, or the talk out of its corpus's
archive, read as far as the talk), decodes it and keeps it there as one WAV file, so
that every scene that names it plays the same file and none fetches it again. The
licences ask for a credit: :attr:`Recording.credit` is the line a demo's video shows.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from acoustic_spaces.i18n import N_

#: The folder the recordings are kept in, under the user's data folder.
FOLDER = "recordings"
#: The ACE challenge corpus's anechoic speech (CC BY 4.0): one tar archive of 148 MB.
ACE = "https://zenodo.org/api/records/6257551/files/ACE_Corpus_Speech.tbz2/content"
ACE_DOI = "10.5281/zenodo.6257551"
_EIF = (
    "https://raw.githubusercontent.com/ichbinsoftware/everythingisfree/"
    "e1d0d5c0422d2626c211d5d877e6f69c37ce8712/src/"
)


@dataclass(frozen=True)
class Recording:
    """One free recording: where it comes from, what it is, and what it costs to fetch."""

    #: The name a scene names it by (``spicy_funk_cake``).
    name: str
    #: What it is called, for the editor and the demo's messages (marked for translation).
    title: str
    #: ``song`` (a stereo master) or ``speech`` (one talker, mono).
    kind: str
    licence: str
    #: The line a video that plays it shows, as its licence asks.
    credit: str
    #: The files it is made of, added together: ``(url, member)`` for a member of the zip
    #: archive at ``url`` (read by HTTP range requests), ``(url, None)`` for the file at
    #: ``url`` (decoded with ffmpeg unless it is a WAV file).
    files: tuple[tuple[str, str | None], ...] = ()
    #: For a recording taken out of a tar archive instead: the archive and its member.
    tar: tuple[str, str] | None = None
    #: MB read from the server to fetch it.
    download_mb: float = 0.0
    #: A song's tempo, beats a minute (the demos' tours move on its phrases).
    bpm: float | None = None

    def summary(self) -> str:
        """``Title (kind, licence, N MB download)``: what the editor offers."""
        return f"{self.title} ({self.kind}, {self.licence}, {self.download_mb:g} MB download)"


RECORDINGS = (
    Recording(
        name="spicy_funk_cake",
        title=N_('"Spicy Funk Cake", Rhythmussportgruppe'),
        kind="song",
        licence="CC BY 4.0",
        credit='Music: Rhythmussportgruppe, "Spicy Funk Cake" (Niklas Dahlheimer), recorded '
        "for the DEGA at Hochschule Düsseldorf, CC BY 4.0 · doi.org/10.5281/zenodo.3601032",
        files=(
            (
                "https://zenodo.org/records/3601032/files/Rhythmussportgruppe_Preview.zip",
                "Rhythmussportgruppe_Preview/Spicy_Fun_cake_Preview/1_REFERENCE-MIX/"
                "Spicy_Funk_Cake.wav",
            ),
        ),
        download_mb=61,
        bpm=110.0,
    ),
    Recording(
        name="great_british_summer",
        title=N_('"Great British Summer", Tom McKenzie'),
        kind="song",
        licence="CC BY 4.0",
        credit='Music: Tom McKenzie, "Great British Summer" (album "Extraordinary", AlbumDB), '
        "CC BY 4.0 · doi.org/10.5281/zenodo.19683001",
        files=(
            (
                "https://zenodo.org/records/19683001/files/masters_stereo.zip",
                "masters_stereo/09 - Great British Summer.wav",
            ),
        ),
        download_mb=50,
        bpm=120.0,
    ),
    Recording(
        name="potassium",
        title=N_('"Potassium", Software-Entwicklungskit'),
        kind="song",
        licence="CC0 1.0",
        credit='Music: Software-Entwicklungskit, "Potassium" (album "Everything is Free"), '
        "CC0 1.0 · github.com/ichbinsoftware/everythingisfree",
        files=((_EIF + "4.Potassium/4.Potassium_Master.m4a", None),),
        download_mb=6,
        bpm=90.0,
    ),
    Recording(
        name="speech_male",
        title=N_("A man talking (ACE corpus, talker M4)"),
        kind="speech",
        licence="CC BY 4.0",
        credit="Speech: ACE challenge corpus, talker M4, by J. Eaton, N. D. Gaubitch, "
        f"A. H. Moore and P. A. Naylor, CC BY 4.0 · doi.org/{ACE_DOI}",
        tar=(ACE, "Speech/M4s4.wav"),
        download_mb=16,
    ),
    Recording(
        name="speech_female",
        title=N_("A woman talking (ACE corpus, talker F2)"),
        kind="speech",
        licence="CC BY 4.0",
        credit="Speech: ACE challenge corpus, talker F2, by J. Eaton, N. D. Gaubitch, "
        f"A. H. Moore and P. A. Naylor, CC BY 4.0 · doi.org/{ACE_DOI}",
        tar=(ACE, "Speech/F2s4.wav"),
        download_mb=49,
    ),
)
#: The names a scene can give as its recording.
NAMES = tuple(item.name for item in RECORDINGS)


def recordings() -> tuple[Recording, ...]:
    """The recordings on offer."""
    return RECORDINGS


def named(name: str) -> Recording:
    """The recording called ``name``."""
    for item in RECORDINGS:
        if item.name == name:
            return item
    raise KeyError(f"no free recording {name!r}; try: {', '.join(NAMES)}")


def data_folder() -> Path:
    """The folder downloaded recordings are kept in: ``acoustic-spaces/recordings``
    under the user's data folder (``ACOUSTIC_SPACES_RECORDINGS`` moves it)."""
    from acoustic_spaces.fetch import data_folder as user_folder

    return user_folder(FOLDER, "ACOUSTIC_SPACES_RECORDINGS")


def file_of(item: Recording | str, folder: str | Path | None = None) -> Path:
    """Where the recording's WAV file goes: in ``folder``, or in :func:`data_folder`."""
    item = named(item) if isinstance(item, str) else item
    base = Path(folder) if folder is not None else data_folder()
    return base / f"{item.name}.wav"


def get(item: Recording | str, folder: str | Path | None = None) -> Path:
    """The recording as a WAV file in ``folder`` (:func:`data_folder` by default),
    downloaded and decoded unless it is there already. What it was made from is not
    kept."""
    from acoustic_spaces import fetch
    from acoustic_spaces.demos import music
    from acoustic_spaces.demos.audio import write

    item = named(item) if isinstance(item, str) else item
    path = file_of(item, folder)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if item.tar is not None:
        url, member = item.tar
        print(f"reading {url} for {item.title}")
        fetch.extract_tar(url, {member: path})
        return path
    with tempfile.TemporaryDirectory(dir=path.parent) as parts:
        with music.Archives() as archives:
            x, rate = music.summed(item.files, Path(parts), archives)
        part = path.with_name(path.name + ".part")
        write(part, x, rate)
        part.replace(path)
    print(f"wrote {path}")
    return path


__all__ = [
    "ACE",
    "ACE_DOI",
    "FOLDER",
    "NAMES",
    "RECORDINGS",
    "Recording",
    "data_folder",
    "file_of",
    "get",
    "named",
    "recordings",
]
