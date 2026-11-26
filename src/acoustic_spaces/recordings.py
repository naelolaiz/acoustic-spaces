"""Free recordings a scene can play in a demo of it: songs in stereo and speech, each
published under a licence that lets anyone use it.

A scene names one by its name as its ``recording`` (:class:`acoustic_spaces.scene.Scene`),
and each source with a ``channel`` plays that channel of it in the demo of the scene
(``acoustic-spaces demo``, the editor's *This scene*). Nothing is fetched until a demo
needs it: :func:`get` downloads the recording into :func:`data_folder` the first time
(the song out of its release by HTTP range requests, or the talk out of its corpus's
archive, read as far as the talk), decodes it and keeps it there as one WAV file, so
that every scene that names it plays the same file and none fetches it again. A song
may also have its stems, the parts it was mixed from (:attr:`Recording.stems`): a
source whose channel names one, or a mix of them (a monitor's), plays it, and
:func:`get_stems` fetches them the first time one is played, into a folder named
after the song, lined up with it. A multitrack recording has a stem per player or
singer; one of several pieces (a symphony's movements, :attr:`Recording.pieces`) has
them for each piece, in a folder of its own. The licences ask for a credit:
:attr:`Recording.credit` is the line a demo's video shows.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from acoustic_spaces.i18n import N_

#: A file a recording is made of: its URL, and the member to take out of the zip archive
#: there (None: the whole file).
File = tuple[str, str | None]

#: The folder the recordings are kept in, under the user's data folder.
FOLDER = "recordings"
#: The ACE challenge corpus's anechoic speech (CC BY 4.0): one tar archive of 148 MB.
ACE = "https://zenodo.org/api/records/6257551/files/ACE_Corpus_Speech.tbz2/content"
ACE_DOI = "10.5281/zenodo.6257551"
#: The ACE corpus's talkers (5 women, 9 men), each one's take 4, about a minute on how
#: they get to work: its title, and the MB of the archive read before the take is out
#: (the archive is read as it comes).
ACE_TALKERS = {
    "F1": (N_("A woman talking (ACE corpus, talker F1)"), 145),
    "F2": (N_("A woman talking (ACE corpus, talker F2)"), 49),
    "F3": (N_("A woman talking (ACE corpus, talker F3)"), 125),
    "F4": (N_("A woman talking (ACE corpus, talker F4)"), 120),
    "F5": (N_("A woman talking (ACE corpus, talker F5)"), 136),
    "M1": (N_("A man talking (ACE corpus, talker M1)"), 79),
    "M2": (N_("A man talking (ACE corpus, talker M2)"), 43),
    "M3": (N_("A man talking (ACE corpus, talker M3)"), 110),
    "M4": (N_("A man talking (ACE corpus, talker M4)"), 16),
    "M5": (N_("A man talking (ACE corpus, talker M5)"), 99),
    "M6": (N_("A man talking (ACE corpus, talker M6)"), 87),
    "M7": (N_("A man talking (ACE corpus, talker M7)"), 133),
    "M8": (N_("A man talking (ACE corpus, talker M8)"), 49),
    "M9": (N_("A man talking (ACE corpus, talker M9)"), 135),
}
_EIF = (
    "https://raw.githubusercontent.com/ichbinsoftware/everythingisfree/"
    "e1d0d5c0422d2626c211d5d877e6f69c37ce8712/src/"
)
#: The Rhythmussportgruppe's whole project, with the stems of "Spicy Funk Cake".
_RSG = "https://zenodo.org/records/3601032/files/RSG_Complete_Project.zip"
_RSG_STEMS = "RSG_Complete_Project/Abgabe_RSG_DEGA/1_Audio/2_SPICY_FUNK_CAKE/2_STEMS_PROCESSED/"
#: The names a song's stems go by, in the order a mix of them lists them: what a
#: source's channel names (``drums``), or each of a mix's (a monitor's).
STEM_NAMES = ("drums", "bass", "guitar", "keys", "brass", "vocal", "backing")
#: How far a song's stems may be off its master (s), which they are lined up with.
STEMS_LAG_S = 5.0
#: The gated RMS (ITU-R BS.1770) a source's feed has when it plays at its sound power:
#: a multitrack recording's stems are built or recorded so where they play loudest.
LEVEL_RMS = 0.1
#: How far (s) a tour's cut may move to a quieter moment in a recording whose stops cut
#: so (:attr:`Recording.cut`).
QUIET_S = 10.0
#: The stems of the church's "Dona nobis pacem" (built by
#: :mod:`acoustic_spaces.demos.dona_nobis_pacem`), named after the church's sources
#: that play them: its 19 singers, by their names in VocalSet, and the two sides of its
#: organ's manual and pedal.
_DONA_STEMS = (
    *(
        f"singer_{singer}"
        for singer in (
            *("f1", "f5", "f2", "f3", "f8", "f4", "f6", "f7", "f9"),
            *("m1", "m4", "m2", "m10", "m3", "m5", "m8", "m7", "m6", "m11"),
        )
    ),
    "organ_manual_c",
    "organ_manual_cis",
    "organ_pedal_c",
    "organ_pedal_cis",
)
#: The stems of Beethoven's 8th symphony (built by
#: :mod:`acoustic_spaces.demos.beethoven_8`): a track per player of the Orchester Wiener
#: Akademie, named after the orchestra example's sources that play them, the winds, the
#: brass and the timpani first, then the strings desk by desk.
_BEETHOVEN_STEMS = (
    *("fl1", "fl2", "ob1", "ob2", "cl1", "cl2", "bn1", "bn2"),
    *("hn1", "hn2", "tp1", "tp2", "timpani"),
    *(f"vl1_{k:02d}" for k in range(1, 15)),
    *(f"vl2_{k:02d}" for k in range(1, 12)),
    *(f"vla_{k:02d}" for k in range(1, 12)),
    *(f"vc_{k}" for k in range(1, 7)),
    *(f"db_{k}" for k in range(1, 7)),
)


@dataclass(frozen=True)
class Piece:
    """A piece of a recording of several (a movement of a symphony): its stems and its
    master in a folder of its own (:func:`stems_folder`)."""

    #: The name a tour's stop and ``--piece`` give it by (``I``, ``IV``).
    name: str
    #: What it is called (``Allegro vivace e con brio``).
    title: str
    #: MB read from the server to fetch it.
    download_mb: float = 0.0
    #: The archive its build reads (:attr:`Recording.build`), by its file name
    #: (``movement1.zip``), that a folder of copies downloaded before may hold in its
    #: place, or the folder it was unpacked into (:func:`copy_of`).
    archive: str | None = None
    #: The stems that rest through the whole piece, as the score has them (the
    #: Allegretto scherzando of Beethoven's 8th has no trumpets and no timpani): a tour
    #: of only them leaves the piece out.
    rests: tuple[str, ...] = ()


@dataclass(frozen=True)
class Recording:
    """One free recording: where it comes from, what it is, and what it costs to fetch."""

    #: The name a scene names it by (``spicy_funk_cake``).
    name: str
    #: What it is called, for the editor and the demo's messages (marked for translation).
    title: str
    #: ``song`` (a stereo master, and maybe its stems), ``speech`` (one talker, mono) or
    #: ``multitrack`` (a stem per player or singer, at the levels they play at: a
    #: source plays its stem as it is, so that at :data:`LEVEL_RMS` it plays at its
    #: sound power; its master is the stems added up).
    kind: str
    licence: str
    #: The line a video that plays it shows, as its licence asks.
    credit: str
    #: The files it is made of, added together: ``(url, member)`` for a member of the zip
    #: archive at ``url`` (read by HTTP range requests), ``(url, None)`` for the file at
    #: ``url`` (decoded with ffmpeg unless it is a WAV file).
    files: tuple[File, ...] = ()
    #: For a recording taken out of a tar archive instead: the archive and its member.
    tar: tuple[str, str] | None = None
    #: MB read from the server to fetch it.
    download_mb: float = 0.0
    #: A song's tempo, beats a minute (the demos' tours move on its phrases).
    bpm: float | None = None
    #: When a song's first downbeat falls (s), which its phrases of eight bars run from.
    downbeat_s: float = 0.0
    #: Speech's talker, ``male`` or ``female``: the speech spectrum a source that says it
    #: takes (IEC 60268-16), which the STI weights for.
    talker: str | None = None
    #: A song's stems, the parts it was mixed from, by their names (:data:`STEM_NAMES`),
    #: each made of files added together as :attr:`files` is: a source whose channel
    #: names one plays it, or a mix of them. Fetched together, only when one is played.
    stems: tuple[tuple[str, tuple[File, ...]], ...] = ()
    #: MB read from the server to fetch the stems.
    stems_mb: float = 0.0
    #: For a recording built rather than downloaded as it is: ``module:function``, which
    #: is called with the recording, the folder (:func:`data_folder` for None) and, for
    #: one of several pieces, the piece, and writes its stems and its master there from
    #: what it downloads (:attr:`download_mb`), or, with ``copy``, from a copy of the
    #: archive of the piece it reads (:attr:`Piece.archive`) downloaded before
    #: (:func:`copy_of`).
    build: str | None = None
    #: The parts of the music in time: when each starts (s) and what plays in it, as the
    #: still of a stop that starts in it says.
    sections: tuple[tuple[float, str], ...] = ()
    #: For a recording of several pieces, each with its own stems (a symphony's
    #: movements), in their order; a demo plays one piece in a run, a tour's stop names
    #: the piece it is heard in. None: one piece, the recording itself.
    pieces: tuple[Piece, ...] = ()
    #: How a tour whose stops give no length shares the music out: None, each stop as
    #: long as a stop is by default (``--seconds``); ``share``, the piece in even
    #: shares; ``quiet``, so, each cut moved to the quietest moment within
    #: :data:`QUIET_S` of where an even share puts it, so that it falls between
    #: phrases.
    cut: str | None = None

    @property
    def stem_names(self) -> tuple[str, ...]:
        """The names of its stems, in the order :data:`STEM_NAMES` has them."""
        return tuple(name for name, _ in self.stems)

    @property
    def piece_names(self) -> tuple[str, ...]:
        """The names of its pieces, in their order; none for a recording of one."""
        return tuple(piece.name for piece in self.pieces)

    def piece(self, name: str) -> Piece:
        """The piece called ``name``; KeyError naming those it has."""
        for piece in self.pieces:
            if piece.name == name:
                return piece
        have = ", ".join(self.piece_names) or "none"
        raise KeyError(f"{self.name} has no piece {name!r}; it has {have}")

    def summary(self) -> str:
        """``Title (kind, licence, N MB download)``: what the editor offers."""
        more = f", {self.stems_mb:g} MB more" if self.stems_mb else ""
        stems = f"; {len(self.stems)} stems{more}" if self.stems else ""
        return (
            f"{self.title} ({self.kind}, {self.licence}, {self.download_mb:g} MB download{stems})"
        )


def _album_song(number: int, name: str, title: str, groups: dict[str, list[str]], **more):
    """A song of the album "Everything is Free" by Software-Entwicklungskit (CC0 1.0): its
    master, and its stems, each the album's own in ``groups`` added together
    (``{"drums": ["KICK", "HATS"]}``). The album is electronic: its synth lead is the
    guitar's stem, its beats the drums'."""
    import urllib.parse

    song = name.capitalize()
    folder = f"{_EIF}{number}.{song}/"

    def file(part: str) -> File:
        return (folder + urllib.parse.quote(f"{number}.{song}_{part}.m4a"), None)

    return Recording(
        name=name,
        title=title,
        kind="song",
        licence="CC0 1.0",
        credit=f'Music: Software-Entwicklungskit, "{song}" (album "Everything is Free"), '
        "CC0 1.0 · github.com/ichbinsoftware/everythingisfree",
        files=(file("Master"),),
        stems=tuple(
            (name, tuple(file(f"Stem_{part}") for part in groups[name]))
            for name in STEM_NAMES
            if name in groups
        ),
        **more,
    )


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
        downbeat_s=0.5,  # after a pickup
        # The processed stems of the whole project; the shouts are the backing vocals.
        stems=(
            ("drums", ((_RSG, _RSG_STEMS + "1_STEM_PROCESSED_DRUMS.wav"),)),
            ("bass", ((_RSG, _RSG_STEMS + "2_STEM_PROCESSED_Bass.wav"),)),
            ("guitar", ((_RSG, _RSG_STEMS + "3_STEM_PROCESSED_GIT.wav"),)),
            ("keys", ((_RSG, _RSG_STEMS + "4_STEM_PROCESSED_KEYS.wav"),)),
            ("brass", ((_RSG, _RSG_STEMS + "5_STEM_PROCESSED_BRASS.wav"),)),
            ("vocal", ((_RSG, _RSG_STEMS + "6_STEM_PROCESSED_VOX.wav"),)),
            ("backing", ((_RSG, _RSG_STEMS + "7_STEM_PROCESSED_SHOUTS.wav"),)),
        ),
        stems_mb=249,
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
    _album_song(
        7,
        "francium",
        N_('"Francium", Software-Entwicklungskit'),
        {
            "drums": [
                "BIG BEAT",
                "GARAGE BEAT",
                "FILL BEAT",
                "SCRATCH BEAT",
                "BREAK",
                "KICK",
                "HATS",
                "OPEN HATS",
                "CYMBALS",
                "CRASH",
                "CLAPS",
                "PERC BUILD",
                "FILL BUILD",
                "FILL SCRATCH",
            ],
            "bass": ["SINE"],
            "guitar": ["MINILOGUE_SYNTH"],
            "keys": ["MELODY GLITCH", "MICROCOSM_EFFECT", "TEXTURE"],
            "vocal": ["VOX LEAD"],
            "backing": [
                "BGVOX ALT OUTRO",
                "BGVOX BOP",
                "BGVOX BUILD",
                "BGVOX ECHO",
                "BGVOX OUTRO",
                "BGVOX SCATTER OUTRO",
            ],
        },
        download_mb=7,
        stems_mb=35,
        # It starts a beat into a bar, and its intro is a bar short of a phrase.
        bpm=127.75,
        downbeat_s=14.76,
    ),
    _album_song(
        5,
        "rubidium",
        N_('"Rubidium", Software-Entwicklungskit'),
        {
            "drums": ["DRUMS MAIN", "DRUMS 2", "DRUMS OUTRO", "BUILD"],
            "guitar": ["MINILOGUE_SYNTH"],
            "keys": ["PLUCK", "MELODY BREAK"],
            "vocal": ["VOX LEAD"],
            "backing": ["BGVOX"],
        },
        download_mb=6,
        stems_mb=37,
        bpm=132.0,  # from its first downbeat, at its start
    ),
    Recording(
        name="dona_nobis_pacem",
        title=N_('"Dona nobis pacem", a choir of 19 singers and an organ'),
        kind="multitrack",
        licence="CC BY 4.0",
        credit="Singers: VocalSet by J. Wilkins, P. Seetharaman, A. Wahl, B. Pardo (CC BY "
        "4.0, doi:10.5281/zenodo.1442513), with note annotations from Annotated-VocalSet by "
        "B. Faghih and J. Timoney (CC BY 4.0, doi:10.5281/zenodo.7061507); retuned and "
        "time-aligned · Organ: VCSL pipe organ, sampled by Simon Dalzell (Ivy Audio), "
        'Versilian Studios (CC0) · "Dona nobis pacem", traditional round',
        download_mb=127,
        stems=tuple((name, ()) for name in _DONA_STEMS),
        build="acoustic_spaces.demos.dona_nobis_pacem:build",
        sections=(
            (0.0, "Verse 1: the organ alone, full"),
            (26.0, "Verse 2: the choir alone"),
            (52.0, "Verse 3: the choir with the organ, quiet"),
        ),
    ),
    Recording(
        name="beethoven_8",
        title=N_("Beethoven's Symphony No. 8, an orchestra of 61 players"),
        kind="multitrack",
        licence="CC BY 4.0",
        credit="Anechoic recording: Orchester Wiener Akademie, cond. Martin Haselböck · TU "
        "Berlin (Böhm, Ackermann, Weinzierl), CC BY 4.0, doi.org/10.14279/depositonce-6729.2",
        download_mb=1990,
        stems=tuple((name, ()) for name in _BEETHOVEN_STEMS),
        build="acoustic_spaces.demos.beethoven_8:build",
        pieces=(
            Piece("I", "Allegro vivace e con brio", 642, "movement1.zip"),
            Piece("II", "Allegretto scherzando", 605, "movement2.zip", ("tp1", "tp2", "timpani")),
            Piece("IV", "Allegro vivace", 743, "movement4.zip"),
        ),
        cut="quiet",
    ),
    *(
        Recording(
            name=f"ace_{talker.lower()}",
            title=title,
            kind="speech",
            licence="CC BY 4.0",
            credit=f"Speech: ACE challenge corpus, talker {talker}, by J. Eaton, N. D. "
            f"Gaubitch, A. H. Moore and P. A. Naylor, CC BY 4.0 · doi.org/{ACE_DOI}",
            tar=(ACE, f"Speech/{talker}s4.wav"),
            download_mb=mb,
            talker="female" if talker.startswith("F") else "male",
        )
        for talker, (title, mb) in ACE_TALKERS.items()
    ),
)
#: The names a scene can give as its recording.
NAMES = tuple(item.name for item in RECORDINGS)
#: Every stem a source's channel may name: those of the recordings on offer, a song's
#: (:data:`STEM_NAMES`) first.
ALL_STEMS = tuple(
    dict.fromkeys([*STEM_NAMES, *(name for item in RECORDINGS for name in item.stem_names)])
)
#: The names two of the ACE talkers had before 4.4.0, still read as theirs.
ALIASES = {"speech_male": "ace_m4", "speech_female": "ace_f2"}


def recordings() -> tuple[Recording, ...]:
    """The recordings on offer."""
    return RECORDINGS


def canonical(name: str) -> str:
    """``name``, or the name it has now when it is an old one (:data:`ALIASES`)."""
    return ALIASES.get(name, name)


def named(name: str) -> Recording:
    """The recording called ``name`` (or by its old name, :data:`ALIASES`)."""
    name = canonical(name)
    for item in RECORDINGS:
        if item.name == name:
            return item
    raise KeyError(f"no free recording {name!r}; try: {', '.join(NAMES)}")


def data_folder() -> Path:
    """The folder downloaded recordings are kept in: ``acoustic-spaces/recordings``
    under the user's data folder (``ACOUSTIC_SPACES_RECORDINGS`` moves it)."""
    from acoustic_spaces.fetch import data_folder as user_folder

    return user_folder(FOLDER, "ACOUSTIC_SPACES_RECORDINGS")


def _piece_of(item: Recording, piece: str | None) -> str | None:
    """The piece of ``item`` meant: ``piece`` (KeyError for one it lacks), its first for
    None, or None for a recording of one piece."""
    if not item.pieces:
        if piece is not None:
            raise KeyError(f"{item.name} is one piece; it has no piece {piece!r}")
        return None
    return item.piece(piece).name if piece is not None else item.pieces[0].name


def file_of(
    item: Recording | str, folder: str | Path | None = None, piece: str | None = None
) -> Path:
    """Where the recording's WAV file goes: in ``folder``, or in :func:`data_folder`; for
    one of several pieces, the master of ``piece`` (its first for None), beside the
    folder of that piece's stems."""
    item = named(item) if isinstance(item, str) else item
    base = Path(folder) if folder is not None else data_folder()
    which = _piece_of(item, piece)
    if which is not None:
        return stems_folder(item, folder) / f"{which}.wav"
    return base / f"{item.name}.wav"


def stems_folder(item: Recording | str, folder: str | Path | None = None) -> Path:
    """The folder a song's stems go in: named after it, in ``folder`` or in
    :func:`data_folder`; a recording of several pieces has a folder in it for each."""
    item = named(item) if isinstance(item, str) else item
    base = Path(folder) if folder is not None else data_folder()
    return base / item.name


def to_download_mb(item: Recording | str, folder: str | Path | None = None) -> float:
    """The MB the recording still downloads: none once its master (each piece's, for one
    of several) is in ``folder`` or in :func:`data_folder`, its stems apart."""
    item = named(item) if isinstance(item, str) else item
    if item.pieces:
        return sum(
            piece.download_mb
            for piece in item.pieces
            if not file_of(item, folder, piece.name).exists()
        )
    return 0.0 if file_of(item, folder).exists() else float(item.download_mb)


def stem_file(
    item: Recording | str, stem: str, folder: str | Path | None = None, piece: str | None = None
) -> Path:
    """Where the song's stem ``stem`` goes (:func:`stems_folder`); for a recording of
    several pieces, that of ``piece`` (its first for None), in the piece's folder."""
    item = named(item) if isinstance(item, str) else item
    which = _piece_of(item, piece)
    folder_of = stems_folder(item, folder)
    return (folder_of / which if which is not None else folder_of) / f"{stem}.wav"


def stems_there(
    item: Recording | str, folder: str | Path | None = None, piece: str | None = None
) -> bool:
    """Whether every stem of the song (of ``piece``, for one of several) is downloaded
    (one with none has them all)."""
    item = named(item) if isinstance(item, str) else item
    return all(stem_file(item, name, folder, piece).exists() for name in item.stem_names)


def get_stems(
    item: Recording | str,
    folder: str | Path | None = None,
    piece: str | None = None,
    archives: str | Path | None = None,
) -> dict[str, Path]:
    """Every stem of the song as a WAV file (:func:`stem_file`), downloaded, decoded
    and added up from its files unless they are all there already, at the rate of the
    song (:func:`get`, fetched first if it is not there), lined up with it (their sum
    against it, within :data:`STEMS_LAG_S`) and cut or padded to its length, so that a
    stem plays in time with the song and with the others. What they were made from is
    not kept. A recording that is built (:attr:`Recording.build`) is built instead,
    those of ``piece`` for one of several, from the copies of its archives in
    ``archives`` where it holds them."""
    from acoustic_spaces.demos import music
    from acoustic_spaces.demos.audio import decoded, resampled, write

    item = named(item) if isinstance(item, str) else item
    paths = {name: stem_file(item, name, folder, piece) for name in item.stem_names}
    if all(path.exists() for path in paths.values()):
        return paths
    if item.build is not None:
        _built(item, folder, piece, archives)
        return paths
    master, rate = decoded(get(item, folder))
    out = stems_folder(item, folder)
    out.mkdir(parents=True, exist_ok=True)
    stems: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory(dir=out) as parts:
        with music.Archives() as opened:
            for name, files in item.stems:
                x, got = music.summed(files, Path(parts), opened)
                stems[name] = resampled(x, got, rate)
    stems, lag = lined_up(stems, master, rate)
    print(f"the stems of {item.name} line up with it {lag / rate:+.3f} s later")
    for name, x in stems.items():
        part = paths[name].with_name(paths[name].name + ".part")
        write(part, x, rate)
        part.replace(paths[name])
        print(f"wrote {paths[name]}")
    return paths


def lined_up(
    stems: dict[str, np.ndarray], master: np.ndarray, rate: int
) -> tuple[dict[str, np.ndarray], int]:
    """A song's ``stems`` lined up with its ``master`` (all at ``rate``) and cut or padded
    to its length, and the samples they were moved earlier by: their sum's onsets
    against the master's over its first 90 s, within :data:`STEMS_LAG_S`
    (:func:`~acoustic_spaces.demos.audio.onset_lag`). Stems often start before the
    song (Spicy Funk Cake's by 58 ms) or after a count-in (Rubidium's by 1.8 s)."""
    from acoustic_spaces.demos.audio import mono, onset_lag, shifted

    if not stems:
        return {}, 0
    head = rate * 90
    heads = [mono(x)[:head] for x in stems.values()]
    together = np.zeros(max(len(x) for x in heads))
    for x in heads:
        together[: len(x)] += x
    lag = onset_lag(together, mono(master)[:head], rate, STEMS_LAG_S)
    return {name: shifted(x, lag, len(master)) for name, x in stems.items()}, lag


def get(
    item: Recording | str,
    folder: str | Path | None = None,
    piece: str | None = None,
    archives: str | Path | None = None,
) -> Path:
    """The recording as a WAV file in ``folder`` (:func:`data_folder` by default),
    downloaded and decoded unless it is there already; a recording that is built, built
    (:func:`get_stems`), for one of several pieces that of ``piece``. What it was made
    from is not kept."""
    from acoustic_spaces import fetch
    from acoustic_spaces.demos import music
    from acoustic_spaces.demos.audio import write

    item = named(item) if isinstance(item, str) else item
    path = file_of(item, folder, piece)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for old, new in ALIASES.items():
        kept = path.with_name(f"{old}.wav")
        if new == item.name and kept.exists():
            kept.replace(path)  # fetched under its old name
            return path
    if item.build is not None:
        _built(item, folder, piece, archives)
        return path
    if item.tar is not None:
        url, member = item.tar
        print(f"reading {url} for {item.title}")
        fetch.extract_tar(url, {member: path})
        return path
    with tempfile.TemporaryDirectory(dir=path.parent) as parts:
        with music.Archives() as opened:
            x, rate = music.summed(item.files, Path(parts), opened)
        part = path.with_name(path.name + ".part")
        write(part, x, rate)
        part.replace(path)
    print(f"wrote {path}")
    return path


def copy_of(item: Recording | str, piece: str | None, archives: str | Path | None) -> Path | None:
    """The copy in ``archives`` of the archive the build of ``piece`` reads
    (:attr:`Piece.archive`), downloaded before: the archive itself, else the folder it
    was unpacked into (named as it is, without its suffix); None when neither is there,
    or the piece reads none."""
    item = named(item) if isinstance(item, str) else item
    which = _piece_of(item, piece)
    archive = item.piece(which).archive if which is not None else None
    if archives is None or archive is None:
        return None
    path = Path(archives) / archive
    if path.is_file():
        return path
    unpacked = path.with_suffix("")
    return unpacked if unpacked.is_dir() else None


def _built(
    item: Recording,
    folder: str | Path | None,
    piece: str | None = None,
    archives: str | Path | None = None,
) -> None:
    """The recording built (:attr:`Recording.build`): its stems and its master, of
    ``piece`` for one of several (its first for None), from the copy of the archive it
    reads in ``archives`` where that holds one (:func:`copy_of`), passed as ``copy``."""
    import importlib

    module, function = (item.build or "").split(":")
    build = getattr(importlib.import_module(module), function)
    more: dict = {}
    which = _piece_of(item, piece)
    if which is not None:
        more["piece"] = which
    copy = copy_of(item, which, archives)
    if copy is not None:
        more["copy"] = copy
    print(f"building {item.title}" + (f", {which}" if which is not None else ""))
    build(item, folder, **more)


__all__ = [
    "ACE",
    "ACE_DOI",
    "ACE_TALKERS",
    "ALIASES",
    "ALL_STEMS",
    "FOLDER",
    "LEVEL_RMS",
    "NAMES",
    "QUIET_S",
    "RECORDINGS",
    "STEMS_LAG_S",
    "STEM_NAMES",
    "File",
    "Piece",
    "Recording",
    "canonical",
    "copy_of",
    "data_folder",
    "file_of",
    "get",
    "get_stems",
    "lined_up",
    "named",
    "recordings",
    "stem_file",
    "stems_folder",
    "stems_there",
    "to_download_mb",
]
