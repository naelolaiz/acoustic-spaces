"""The language of the text the command line, the editor and the library show.

English is the text in the code; other languages come from catalogs in ``locale/``
(``es.po`` for Spanish), read when the first text is translated. The language is, in
order:

1. ``ACOUSTIC_SPACES_LANG`` in the environment (``en``, ``es``; ``pseudo`` shows each
   translatable text in brackets, to find text that is not);
2. the one picked in the editor (View > Language), kept per user in ``settings.json``;
3. the system's: ``LANGUAGE``, ``LC_ALL``, ``LC_MESSAGES`` or ``LANG``, else the display
   language of Windows or macOS.

A language without a catalog falls back to English. Scene files, command and option names
and the files the commands write stay in English whatever the language.

Text is wrapped in ``_()`` where it is shown, with named fields filled in afterwards, so a
translation can put them in another order: ``_("{count} rooms").format(count=n)``. Text
defined before the language is known (a module's constants) is marked with ``N_()`` and
translated with ``_()`` where it is shown. ``tools/catalog.py`` collects the marked text
into the catalogs and checks that every entry is translated.
"""

from __future__ import annotations

import gettext
import io
import json
import os
import plistlib
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: The languages with a catalog, in their own words (as a language menu shows them).
LANGUAGES = {"en": "English", "es": "Español"}
#: The environment variable that sets the language, over every other choice.
ENV = "ACOUSTIC_SPACES_LANG"
#: Shows each translatable text as ⟦text⟧: text without brackets is not translatable.
PSEUDO = "pseudo"
#: Where the catalogs are.
LOCALE_DIR = Path(__file__).resolve().parent / "locale"

_active: tuple[str, gettext.NullTranslations] | None = None


# -- the catalogs ---------------------------------------------------------------------


@dataclass
class Entry:
    """One message of a ``.po`` catalog."""

    msgid: str
    msgstr: list[str] = field(default_factory=lambda: [""])
    context: str | None = None
    plural: str | None = None
    flags: list[str] = field(default_factory=list)
    #: Translator comments (``# ``), extracted comments (``#.``), references (``#:``)
    #: and the text a fuzzy entry was translated from (``#|``).
    comments: list[str] = field(default_factory=list)
    extracted: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    previous: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str | None, str]:
        return self.context, self.msgid

    @property
    def fuzzy(self) -> bool:
        return "fuzzy" in self.flags

    @property
    def translated(self) -> bool:
        return not self.fuzzy and all(self.msgstr)


_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def _unquote(text: str, where: str) -> str:
    text = text.strip()
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        raise ValueError(f"{where}: expected a quoted string, found {text!r}")
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(0)), text[1:-1])


def read_po(text: str, name: str = "catalog") -> list[Entry]:
    """The entries of a ``.po`` catalog, its header (``msgid ""``) first. Obsolete
    entries (``#~``) are left out."""
    entries: list[Entry] = []
    entry: Entry | None = None
    target: tuple[str, int] | None = None  # the field the next continuation line adds to
    seen_msgid = False
    for number, line in enumerate(text.splitlines(), 1):
        where = f"{name}:{number}"
        line = line.rstrip()
        if not line or line.startswith("#~"):
            continue
        if line.startswith("#") and seen_msgid:  # comments start the next entry
            entry, seen_msgid, target = None, False, None
        if entry is None:
            entry = Entry(msgid="")
            entries.append(entry)
        if line.startswith("#,"):
            entry.flags += [flag.strip() for flag in line[2:].split(",") if flag.strip()]
        elif line.startswith("#."):
            entry.extracted.append(line[3:])
        elif line.startswith("#:"):
            entry.references += line[2:].split()
        elif line.startswith("#|"):
            entry.previous.append(line[3:])
        elif line.startswith("#"):
            entry.comments.append(line[2:])
        elif line.startswith('"'):
            if target is None:
                raise ValueError(f"{where}: a string that continues nothing")
            name_, index = target
            value = _unquote(line, where)
            if name_ == "msgstr":
                entry.msgstr[index] += value
            else:
                setattr(entry, name_, (getattr(entry, name_) or "") + value)
        else:
            keyword, _, rest = line.partition(" ")
            if keyword == "msgctxt":
                if seen_msgid:
                    entry = Entry(msgid="")
                    entries.append(entry)
                    seen_msgid = False
                entry.context, target = _unquote(rest, where), ("context", 0)
            elif keyword == "msgid":
                if seen_msgid:
                    entry = Entry(msgid="")
                    entries.append(entry)
                entry.msgid, target, seen_msgid = _unquote(rest, where), ("msgid", 0), True
            elif keyword == "msgid_plural":
                entry.plural, target = _unquote(rest, where), ("plural", 0)
            elif keyword == "msgstr":
                entry.msgstr, target = [_unquote(rest, where)], ("msgstr", 0)
            elif (found := re.fullmatch(r"msgstr\[(\d+)\]", keyword)) is not None:
                index = int(found.group(1))
                if index == 0:
                    entry.msgstr = []
                if index != len(entry.msgstr):
                    raise ValueError(f"{where}: msgstr[{index}] out of order")
                entry.msgstr.append(_unquote(rest, where))
                target = ("msgstr", index)
            else:
                raise ValueError(f"{where}: unknown line {line!r}")
    return [e for e in entries if e.msgid or e.context is None and e.msgstr != [""]]


def _mo(entries: list[Entry]) -> bytes:
    """A compiled (``.mo``) catalog of the translated entries, as msgfmt writes it."""
    messages = {}
    for entry in entries:
        if entry.msgid and not entry.translated:
            continue  # untranslated and fuzzy text shows in English
        key = entry.msgid if entry.plural is None else f"{entry.msgid}\0{entry.plural}"
        if entry.context is not None:
            key = f"{entry.context}\x04{key}"
        messages[key.encode()] = "\0".join(entry.msgstr).encode()
    keys = sorted(messages)
    ids = strs = b""
    offsets = []
    for key in keys:
        offsets.append((len(ids), len(key), len(strs), len(messages[key])))
        ids += key + b"\0"
        strs += messages[key] + b"\0"
    start = 7 * 4 + 16 * len(keys)  # the header and both tables
    table_ids, table_strs = [], []
    for id_at, id_len, str_at, str_len in offsets:
        table_ids += [id_len, start + id_at]
        table_strs += [str_len, start + len(ids) + str_at]
    header = struct.pack("Iiiiiii", 0x950412DE, 0, len(keys), 7 * 4, 7 * 4 + 8 * len(keys), 0, 0)
    tables = struct.pack(f"{len(table_ids)}i", *table_ids)
    tables += struct.pack(f"{len(table_strs)}i", *table_strs)
    return header + tables + ids + strs


class _Pseudo(gettext.NullTranslations):
    """Every text in brackets, plurals and contexts too."""

    def gettext(self, message):
        return f"⟦{message}⟧"

    def ngettext(self, msgid1, msgid2, n):
        return f"⟦{msgid1 if n == 1 else msgid2}⟧"

    def pgettext(self, context, message):
        return f"⟦{message}⟧"

    def npgettext(self, context, msgid1, msgid2, n):
        return f"⟦{msgid1 if n == 1 else msgid2}⟧"


def catalog_path(code: str) -> Path:
    return LOCALE_DIR / f"{code}.po"


def _translations(code: str) -> gettext.NullTranslations:
    if code == PSEUDO:
        return _Pseudo()
    path = catalog_path(code)
    if code == "en" or not path.is_file():
        return gettext.NullTranslations()
    entries = read_po(path.read_text(encoding="utf-8"), path.name)
    return gettext.GNUTranslations(io.BytesIO(_mo(entries)))


# -- choosing the language ------------------------------------------------------------


def _code(tag: str) -> str | None:
    """The language of a locale name such as ``es_AR.UTF-8`` or ``es-419``, if it has a
    catalog; ``en`` for the C locale."""
    tag = tag.strip().split(".")[0].split("@")[0]
    if tag in ("C", "POSIX"):
        return "en"
    code = re.split(r"[_-]", tag)[0].lower()
    return code if code in LANGUAGES else None


def system_language() -> str:
    """The system's language, if it has a catalog, else English."""
    for name in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(name)
        if value:  # the first one set decides, as for gettext; LANGUAGE lists several
            for tag in value.split(":"):
                code = _code(tag) if tag else None
                if code is not None:
                    return code
            return "en"
    if sys.platform == "win32":
        import ctypes

        primary = ctypes.windll.kernel32.GetUserDefaultUILanguage() & 0x3FF
        return {0x09: "en", 0x0A: "es"}.get(primary, "en")
    if sys.platform == "darwin":
        path = Path.home() / "Library" / "Preferences" / ".GlobalPreferences.plist"
        try:
            with path.open("rb") as file:
                preferred = plistlib.load(file).get("AppleLanguages") or []
        except (OSError, ValueError, plistlib.InvalidFileException):
            preferred = []
        if preferred:  # the first is the display language
            return _code(str(preferred[0])) or "en"
    return "en"


def settings_file() -> Path:
    """The user's settings: ``acoustic-spaces/settings.json`` in the user's configuration
    folder."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "acoustic-spaces" / "settings.json"


def _settings() -> dict:
    try:
        saved = json.loads(settings_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return saved if isinstance(saved, dict) else {}


def saved_language() -> str | None:
    """The language picked in the editor, or None to follow the system."""
    code = _settings().get("language")
    return code if code in LANGUAGES else None


def save_language(code: str | None) -> None:
    """Keep ``code`` as the user's language (None: follow the system), from the next
    start on."""
    if code is not None and code not in LANGUAGES:
        raise ValueError(f"no catalog for the language {code!r}")
    settings = _settings()
    if code is None:
        settings.pop("language", None)
    else:
        settings["language"] = code
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def chosen_language() -> tuple[str, str]:
    """The language to use and what chose it: ``ENV``, ``"saved"`` or ``"system"``."""
    value = os.environ.get(ENV, "").strip()
    if value:
        return (PSEUDO if value == PSEUDO else _code(value) or "en"), ENV
    saved = saved_language()
    if saved is not None:
        return saved, "saved"
    return system_language(), "system"


def language() -> str:
    """The language in use (``en``, ``es`` or ``pseudo``)."""
    return _current()[0]


def set_language(code: str | None) -> None:
    """Use ``code`` from now on; None chooses again as at start-up."""
    global _active
    if code is not None and code not in LANGUAGES and code != PSEUDO:
        raise ValueError(f"no catalog for the language {code!r}")
    _active = None if code is None else (code, _translations(code))


def _current() -> tuple[str, gettext.NullTranslations]:
    global _active
    if _active is None:
        code = chosen_language()[0]
        _active = (code, _translations(code))
    return _active


# -- translating ----------------------------------------------------------------------


def _(message: str) -> str:
    """``message`` in the language in use."""
    return _current()[1].gettext(message)


def ngettext(singular: str, plural: str, n: int) -> str:
    """The singular or plural form, as the language in use counts ``n``."""
    return _current()[1].ngettext(singular, plural, n)


def pgettext(context: str, message: str) -> str:
    """``message`` in the language in use, where English uses the same words for
    different things: ``context`` tells them apart."""
    return _current()[1].pgettext(context, message)


def N_(message: str) -> str:
    """Marks ``message`` for translation without translating it (yet)."""
    return message


__all__ = [
    "ENV",
    "LANGUAGES",
    "N_",
    "PSEUDO",
    "Entry",
    "chosen_language",
    "language",
    "ngettext",
    "pgettext",
    "read_po",
    "save_language",
    "saved_language",
    "set_language",
    "settings_file",
    "system_language",
]
