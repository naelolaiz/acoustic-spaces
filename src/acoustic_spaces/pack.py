"""A scene packed in one file, to run on another computer: a server with a GPU, or a
job on a cloud service.

A pack is a zip file holding ``scene.yaml`` and every file the scene names (its model,
terrain, buildings map, material and directivity files, the sources' signals, the
scene's recording and HRTFs) with the files those name in turn (a glTF model's buffers
and images, a directivity array's element data). Files under the scene's folder keep
their place relative to it; files elsewhere go under ``external/<their folder>/``. The
scene file names them all relatively, so the pack runs wherever it is unpacked, and
every command that reads a scene takes a pack in its place::

    acoustic-spaces pack hall/scene.yaml hall.zip
    acoustic-spaces simulate hall.zip --out results     # on the other computer

``pack.json`` records the version that wrote the pack and a SHA-256 of every file in it,
which unpacking checks. Packing uploads nothing: the pack is a file, to take where you
choose, and it holds everything needed to rebuild the scene.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import unicodedata
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from acoustic_spaces._version import __version__
from acoustic_spaces.scene import Scene

#: The scene file in a pack.
SCENE_FILE = "scene.yaml"
#: What the pack holds: its format, the version that wrote it and the files' hashes.
MANIFEST_FILE = "pack.json"
#: The folder of a pack that holds the files from outside the scene's folder.
EXTERNAL = "external"
#: The pack format this version writes and reads.
FORMAT = 1
#: Bytes read at a time from the files packed and unpacked (models and HRTFs can be
#: hundreds of megabytes).
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Packed:
    """A pack written by :func:`pack`."""

    path: Path
    #: (name in the pack, size in bytes) of every file of the scene in it.
    files: list[tuple[str, int]]

    @property
    def size(self) -> int:
        """The size of the scene's files in bytes, before compression."""
        return sum(size for _name, size in self.files)


@dataclass(frozen=True)
class Unpacked:
    """A pack unpacked by :func:`unpack`."""

    #: The scene file, ready to load.
    scene: Path
    #: The version of acoustic_spaces that wrote the pack.
    version: str


def is_pack(path: str | Path) -> bool:
    """Whether a path names a pack rather than a scene file: it ends in ``.zip``."""
    return Path(path).suffix.lower() == ".zip"


def pack(scene: Scene, out: str | Path, base_dir: str | Path | None = None) -> Packed:
    """Write ``scene`` and every file it names to the zip file ``out``.

    ``base_dir`` is the scene file's folder: the files under it keep their place
    relative to it. Without it (a scene never saved), every file goes under
    ``external/``. The pack is written to ``<out>.part`` and takes the place of a file
    ``out`` only when complete: when packing fails, an earlier ``out`` stays.
    """
    out = Path(out)
    if not is_pack(out):
        raise ValueError(f"a pack is a zip file: name it {out.stem}.zip")
    base = None if base_dir is None else Path(base_dir).resolve()
    names, entries = _layout([Path(file).resolve() for file in scene.files()], base)
    text = scene.yaml_text(lambda file: names[Path(file).resolve()])
    hashes = {}
    # Written beside ``out`` and put in its place once complete, so a pack that fails
    # leaves an earlier one of the same name as it was.
    part = out.with_name(out.name + ".part")
    try:
        with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(SCENE_FILE, text)
            for name, file in entries.items():
                archive.write(file, name)
                hashes[name] = _sha256(file)
            manifest = {
                "format": FORMAT,
                "acoustic_spaces": __version__,
                "scene": scene.name,
                "files": {SCENE_FILE: _sha256_of(text.encode("utf-8")), **hashes},
            }
            archive.writestr(MANIFEST_FILE, json.dumps(manifest, indent=2) + "\n")
        os.replace(part, out)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return Packed(out, [(name, file.stat().st_size) for name, file in entries.items()])


def unpack(path: str | Path, into: str | Path) -> Unpacked:
    """Unpack the pack ``path`` into the folder ``into``, checking every file against
    the hash the pack records.

    Nothing is written unless the folder has room for every file at the size the pack
    records (which is as much as reading one gives). Each file is written to
    ``<file>.part`` and takes its name once its hash matches, so a damaged file is
    never left in the folder."""
    path, into = Path(path), Path(into)
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path.name} is not a pack: {exc}") from None
    with archive:
        try:
            manifest = json.loads(archive.read(MANIFEST_FILE))
        except KeyError:
            raise ValueError(f"{path.name} is not a pack: it has no {MANIFEST_FILE}") from None
        except ValueError as exc:
            raise ValueError(f"{path.name}: {MANIFEST_FILE} is not valid JSON: {exc}") from None
        if not isinstance(manifest, dict):
            raise ValueError(f"{path.name}: {MANIFEST_FILE} is not a pack's manifest")
        if manifest.get("format") != FORMAT:
            raise ValueError(
                f"{path.name} is a pack this version cannot read (it was written by "
                f"acoustic_spaces {manifest.get('acoustic_spaces', '?')}): update to that "
                "version or later"
            )
        files = manifest.get("files")
        if not isinstance(files, dict) or SCENE_FILE not in files:
            raise ValueError(f"{path.name}: {MANIFEST_FILE} lists no {SCENE_FILE}")
        targets, size = {}, 0
        for name in files:
            targets[name] = into.joinpath(*_safe_parts(name, path))
            try:
                size += archive.getinfo(name).file_size
            except KeyError:
                raise ValueError(f"{path.name} lacks {name}, which its manifest lists") from None
        into.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(into).free
        if size > free:
            raise ValueError(
                f"{path.name} unpacks to {size / 2**20:,.1f} MiB; {into} has "
                f"{free / 2**20:,.1f} MiB free"
            )
        for name, expected in files.items():
            target = targets[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            digest = hashlib.sha256()
            try:
                with archive.open(name) as source, part.open("wb") as sink:
                    while chunk := source.read(_CHUNK):
                        digest.update(chunk)
                        sink.write(chunk)
                if digest.hexdigest() != expected:
                    raise zipfile.BadZipFile("its hash does not match")
            # Damaged on the way, encrypted (RuntimeError) or compressed in a way zipfile
            # cannot read (NotImplementedError).
            except (
                zipfile.BadZipFile,
                zlib.error,
                EOFError,
                RuntimeError,
                NotImplementedError,
            ) as exc:
                part.unlink(missing_ok=True)
                raise ValueError(f"{path.name}: {name} is damaged ({exc})") from None
            except BaseException:
                part.unlink(missing_ok=True)
                raise
            os.replace(part, target)
    return Unpacked(into / SCENE_FILE, str(manifest.get("acoustic_spaces", "?")))


def _layout(files: list[Path], base: Path | None) -> tuple[dict[Path, str], dict[str, Path]]:
    """The name in the pack of each file the scene names, and every file to pack by its
    name. A file and the files it names keep their places relative to each other: under
    the scene's folder when they all lie in it, else in a folder of their own under
    ``external/`` named after the file's folder.

    Names are told apart only where they differ in more than case or the way a letter
    is encoded (:func:`_folded`), as macOS tells them apart. A name that a pack may not
    hold (:func:`_safe_parts`), two files whose names macOS or Windows take for one, or
    a file and a folder that would have one name there, are refused here rather than
    when the pack is unpacked."""
    reserved = {_folded(SCENE_FILE), _folded(MANIFEST_FILE)}
    inside, outside = {}, {}
    for file in files:
        if not file.is_file():
            raise ValueError(f"the scene names {file}, which is not there")
        group = [file, *_named_by(file)]
        missing = [f for f in group if not f.is_file()]
        if missing:
            raise ValueError(f"{file.name} names {missing[0]}, which is not there")
        if base is not None and all(f.is_relative_to(base) for f in group):
            relative = {f: f.relative_to(base).as_posix() for f in group}
            if not reserved & {_folded(name) for name in relative.values()}:
                inside[file] = relative
                continue
        folder = file.parent
        try:
            outside[file] = {f: f.relative_to(folder).as_posix() for f in group}
        except ValueError:
            stray = next(f for f in group if not f.is_relative_to(folder))
            raise ValueError(
                f"{file.name} names {stray}, outside its folder: move that next to "
                f"{file.name} to pack it"
            ) from None
    names: dict[Path, str] = {}
    entries: dict[str, Path] = {}
    for file, relative in inside.items():
        names[file] = relative[file]
        entries.update((name, f) for f, name in relative.items())
    folders: dict[Path, str] = {}
    for file, relative in outside.items():
        if file.parent not in folders:
            # A name of the packer's choosing: one a pack may hold, apart from the
            # others even where case does not count.
            label = file.parent.name.replace(":", "-").replace("\\", "-") or "files"
            prefix, count = f"{EXTERNAL}/{label}", 1
            # Neither a folder already taken nor a file's name (external/a.wav, a file
            # of the scene's folder, beside a folder a.wav from elsewhere).
            while any(
                n == _folded(prefix) or n.startswith(_folded(prefix) + "/")
                for n in map(_folded, entries)
            ):
                count += 1
                prefix = f"{EXTERNAL}/{label}-{count}"
            folders[file.parent] = prefix
        prefix = folders[file.parent]
        names[file] = f"{prefix}/{relative[file]}"
        entries.update((f"{prefix}/{name}", f) for f, name in relative.items())
    seen: dict[str, tuple[str, Path]] = {}
    for name, file in entries.items():
        try:
            _safe_parts(name, file)
        except ValueError:
            raise ValueError(
                f"{file} cannot go in a pack as {name!r}: a name in a pack holds no ':' "
                "or '\\', which Windows refuses; rename the file or its folder"
            ) from None
        other = seen.setdefault(_folded(name), (name, file))
        if other[1] != file and not other[1].samefile(file):
            what, where = _difference(other[0], name)
            raise ValueError(
                f"{other[1]} and {file} would go in a pack as {other[0]!r} and {name!r}, "
                f"which differ only in {what}: unpacked on {where}, one would take the "
                "other's place; rename one"
            )
    for name, file in entries.items():
        parts = name.split("/")
        for k in range(1, len(parts)):
            folder = "/".join(parts[:k])
            other = seen.get(_folded(folder))
            if other is None:
                continue
            if other[0] == folder:
                alike = "the file's name: a file and a folder cannot have one name"
            else:
                what, where = _difference(other[0], folder)
                alike = (
                    f"a name that differs from the file's only in {what}: unpacked on "
                    f"{where}, a file and a folder would have one name"
                )
            raise ValueError(
                f"{other[1]} would go in a pack as {other[0]!r}, and {file} as {name!r}, "
                f"in a folder {folder!r}, {alike}; rename one"
            )
    return names, entries


def _folded(name: str) -> str:
    """A name as macOS compares it: names of one fold are one name there. APFS compares
    names in Unicode's canonical caseless form: NFD, full case folding, NFD again (The
    Unicode Standard, D145; Apple's xnu, bsd/sys/unicode.h), so that straße and strasse
    are one name, as are é as one character and as e and an accent. NTFS (Windows)
    compares them otherwise (:func:`_upcased`), and keeps those two pairs apart."""
    return unicodedata.normalize("NFD", unicodedata.normalize("NFD", name).casefold())


def _upcased(name: str) -> str:
    """A name as NTFS compares it: each UTF-16 code unit upper-cased one for one through
    the volume's ``$UpCase`` table, here Unicode's one-for-one upper cases, and not
    normalised (Microsoft, FsRtlAreNamesEqual)."""
    return "".join(c.upper() if ord(c) < 0x10000 and len(c.upper()) == 1 else c for c in name)


#: What names that APFS takes for one can differ in besides case.
_ENCODED = "the way a letter is encoded (é as one character, or as e and an accent)"


def _difference(a: str, b: str) -> tuple[str, str]:
    """What two names of one fold (:func:`_folded`) differ in, case or the way a letter
    is encoded, and where they are one name: macOS, and Windows where NTFS takes them
    for one too (:func:`_upcased`)."""
    case = unicodedata.normalize("NFD", a) != unicodedata.normalize("NFD", b)
    encoding = not case or a.casefold() != b.casefold()
    what = " and in ".join(
        text for text, differ in (("case", case), (_ENCODED, encoding)) if differ
    )
    return what, "Windows or macOS" if _upcased(a) == _upcased(b) else "macOS"


def _named_by(file: Path) -> list[Path]:
    """The files that ``file`` names in turn, which must come with it: a glTF model's
    buffers and images, and the element's file of a directivity array (YAML, the only
    YAML file a scene names)."""
    suffix = file.suffix.lower()
    if suffix == ".gltf":
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read {file.name}: {exc}") from None
        uris = [item.get("uri") for key in ("buffers", "images") for item in data.get(key, [])]
        return [
            (file.parent / unquote(uri)).resolve()
            for uri in uris
            if isinstance(uri, str) and not uri.startswith("data:")
        ]
    if suffix in (".yaml", ".yml"):
        from acoustic_spaces.directivity import read_array

        file = getattr(read_array(file).element, "file", None)
        return [] if file is None else [Path(file).resolve()]
    return []


def _safe_parts(name: str, pack: Path) -> tuple[str, ...]:
    """The parts of a name in a pack, refusing any that would land outside the folder
    it is unpacked into, or that Windows refuses (``:`` and ``\\``)."""
    parts = PurePosixPath(name).parts
    if (
        not parts
        or "\\" in name
        or name.startswith("/")
        or any(part in ("", ".", "..") or ":" in part for part in parts)
    ):
        raise ValueError(f"{pack.name}: {name!r} is not a name a pack may hold")
    return parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
