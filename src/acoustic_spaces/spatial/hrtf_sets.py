"""Free HRTF sets to download, so that binaural renders go through measured ears
instead of the spherical head model.

Every set here is published under a licence that allows anyone to use it; the entry
names the licence and who measured it, and ``acoustic-spaces hrtf`` and the editor's
**Download HRTFs…** show both before downloading anything. Nothing is fetched until it
is asked for, and a set already in the folder is left alone.

A set is one SOFA file: either a plain download or one member of a remote archive,
taken out of it without the rest (:mod:`acoustic_spaces.fetch`). ``ears`` says whose
head it is: a dummy head measures the same way every time, a person's own ears place
sounds better for that person and worse for everybody else.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from acoustic_spaces.fetch import Progress, fetch

#: The folder the sets are kept in, under the user's data folder.
FOLDER = "hrtf"


@dataclass(frozen=True)
class HrtfSet:
    """One SOFA file to download: where it comes from, and what it costs to fetch."""

    #: The short name it is asked for by (``ku100``).
    name: str
    title: str
    #: Whose ears: a dummy head, or a person.
    ears: str
    licence: str
    #: Who measured it, for the credit a work that uses it owes.
    credit: str
    #: Where it is published, to read the terms in full.
    page: str
    #: The file to download: a SOFA file, or an archive to take ``members`` out of.
    url: str
    #: How big the download is.
    size_mb: float
    #: What the file holds: directions, sample rate, length.
    note: str
    #: The members of the archive that are SOFA files, the first by default; empty when
    #: the download is the SOFA file itself.
    members: tuple[str, ...] = ()
    #: Which member the choice of subject is, for the help (``subject``).
    choice: str = ""

    @property
    def member(self) -> str | None:
        """The member taken out of the archive by default, or None."""
        return self.members[0] if self.members else None

    def file(self, member: str | None = None) -> str:
        """What the downloaded file is called: the set's name, and the member when it is
        not the first (``viking-subj_K.sofa``)."""
        if member is None or member == self.member:
            return f"{self.name}.sofa"
        return f"{self.name}-{Path(member).stem}.sofa"

    def summary(self) -> str:
        """One line for a list: the title, the ears, the licence and the size."""
        return f"{self.title} — {self.ears}; {self.licence}; {self.size_mb:g} MB"


#: Viking v2 holds one file per head, so the subjects are the members to choose from.
_VIKING = tuple(f"subj_{letter}.sofa" for letter in "ABCDEFGHIJKLMNOPQRSTZ")

#: The sets on offer. Each is a free licence and a direct download.
SETS: tuple[HrtfSet, ...] = (
    HrtfSet(
        name="ku100",
        title="Neumann KU100 (TH Köln)",
        ears="a dummy head",
        licence="CC BY 3.0",
        credit="Benjamin Bernschütz, TH Köln",
        page="https://doi.org/10.5281/zenodo.3928297",
        url="https://zenodo.org/api/records/3928297/files/HRIR_L2702.sofa/content",
        size_mb=3.4,
        note="2702 directions over the whole sphere, 48 kHz, 128 taps",
    ),
    HrtfSet(
        name="kemar",
        title="KEMAR (SADIE II)",
        ears="a dummy head",
        licence="Apache 2.0",
        credit="AudioLab, University of York (SADIE II)",
        page="https://doi.org/10.5281/zenodo.12092466",
        url="https://zenodo.org/api/records/12092466/files/D2_HRIR_SOFA.zip/content",
        size_mb=38.0,
        note="8802 directions, 48 kHz, 256 taps",
        members=("D2_HRIR_SOFA/D2_48K_24bit_256tap_FIR_SOFA.sofa",),
    ),
    HrtfSet(
        name="viking",
        title="The Viking HRTF dataset v2",
        ears="one of 20 people, and a dummy head (Z)",
        licence="CC BY 4.0",
        credit="Spagnol, Miccini and Unnthorsson, Reykjavík University",
        page="https://doi.org/10.5281/zenodo.4160401",
        url="https://zenodo.org/api/records/4160401/files/HRTFs.zip/content",
        size_mb=58.4,
        note="1513 directions, 48 kHz, 128 taps; one file per head, A by default",
        members=_VIKING,
        choice="head",
    ),
)


def sets() -> tuple[HrtfSet, ...]:
    """The sets on offer."""
    return SETS


def named(name: str) -> HrtfSet:
    """The set called ``name``."""
    for item in SETS:
        if item.name == name:
            return item
    raise KeyError(f"no HRTF set {name!r}; try: {', '.join(item.name for item in SETS)}")


def data_folder() -> Path:
    """The folder downloaded sets are kept in: ``acoustic-spaces/hrtf`` under the user's
    data folder (``ACOUSTIC_SPACES_HRTF`` moves it)."""
    named_folder = os.environ.get("ACOUSTIC_SPACES_HRTF")
    if named_folder:
        return Path(named_folder)
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "acoustic-spaces" / FOLDER


def file_of(item: HrtfSet, member: str | None = None, folder: str | Path | None = None) -> Path:
    """Where the set's file goes: in ``folder``, or in :func:`data_folder`."""
    base = Path(folder) if folder is not None else data_folder()
    return base / item.file(member)


def installed(folder: str | Path | None = None) -> dict[str, Path]:
    """The sets that are there already, by name."""
    base = Path(folder) if folder is not None else data_folder()
    found = {}
    for item in SETS:
        path = base / item.file()
        if path.exists():
            found[item.name] = path
    return found


def get(
    item: HrtfSet | str,
    member: str | None = None,
    folder: str | Path | None = None,
    progress: Progress | None = None,
    again: bool = False,
) -> Path:
    """The set's SOFA file, downloaded into ``folder`` (:func:`data_folder` by default)
    unless it is there already; ``progress(done, total)`` follows the download."""
    item = named(item) if isinstance(item, str) else item
    if member is not None and item.members and member not in item.members:
        raise KeyError(f"{item.name}: no {member!r}; try: {', '.join(item.members)}")
    path = file_of(item, member, folder)
    if path.exists() and not again:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    return fetch(item.url, path, member=member or item.member, progress=progress)


def other(url: str, name: str, folder: str | Path | None = None, progress=None) -> Path:
    """Any other SOFA file, by its address, saved as ``name`` beside the sets."""
    base = Path(folder) if folder is not None else data_folder()
    base.mkdir(parents=True, exist_ok=True)
    path = base / (name if name.endswith(".sofa") else f"{name}.sofa")
    return fetch(url, path, progress=progress)


__all__ = [
    "FOLDER",
    "SETS",
    "HrtfSet",
    "data_folder",
    "file_of",
    "get",
    "installed",
    "named",
    "other",
    "sets",
]
