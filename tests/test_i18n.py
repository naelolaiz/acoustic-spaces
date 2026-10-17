"""Translations: the catalogs, choosing the language and the catalog tool."""

import gettext
import io
import json
import sys
import textwrap
from pathlib import Path

import pytest

from acoustic_spaces import i18n
from acoustic_spaces.i18n import Entry, _, ngettext, pgettext, read_po

TOOLS = Path(__file__).resolve().parent.parent / "tools"

PO = textwrap.dedent(
    r"""
    # A header comment.
    msgid ""
    msgstr ""
    "Language: es\n"
    "Content-Type: text/plain; charset=UTF-8\n"
    "Plural-Forms: nplurals=2; plural=(n != 1);\n"

    #. Translators: a room of the scene.
    #: acoustic_spaces/room.py
    #, python-brace-format
    msgid "{count} room"
    msgid_plural "{count} rooms"
    msgstr[0] "{count} sala"
    msgstr[1] "{count} salas"

    msgctxt "frequency band"
    msgid "Band"
    msgstr "Banda"

    msgid ""
    "A long text, on two lines, "
    "with \"quotes\" and a\n"
    "line break"
    msgstr "Un texto largo con «comillas» y un\nsalto de línea"

    #, fuzzy
    msgid "Run"
    msgstr "Correr"

    msgid "Stop"
    msgstr ""

    #~ msgid "Gone"
    #~ msgstr "Ido"
    """
)


def _catalog(text: str = PO) -> gettext.GNUTranslations:
    return gettext.GNUTranslations(io.BytesIO(i18n._mo(read_po(text))))


def test_reading_a_catalog() -> None:
    header, plural, band, long, fuzzy, empty = read_po(PO)
    assert header.msgid == "" and "nplurals=2" in header.msgstr[0]
    assert header.comments == ["A header comment."]
    assert plural.plural == "{count} rooms" and plural.msgstr == ["{count} sala", "{count} salas"]
    assert plural.extracted == ["Translators: a room of the scene."]
    assert plural.references == ["acoustic_spaces/room.py"]
    assert plural.flags == ["python-brace-format"]
    assert band.context == "frequency band" and band.msgstr == ["Banda"]
    assert long.msgid == 'A long text, on two lines, with "quotes" and a\nline break'
    assert fuzzy.fuzzy and not fuzzy.translated
    assert not empty.translated


def test_a_compiled_catalog_translates_as_gettext_does() -> None:
    catalog = _catalog()
    assert catalog.ngettext("{count} room", "{count} rooms", 1) == "{count} sala"
    assert catalog.ngettext("{count} room", "{count} rooms", 3) == "{count} salas"
    assert catalog.pgettext("frequency band", "Band") == "Banda"
    assert catalog.gettext("Band") == "Band"  # without its context it is another text
    assert catalog.gettext("Run") == "Run"  # fuzzy: English until reviewed
    assert catalog.gettext("Stop") == "Stop"
    assert catalog.gettext("Gone") == "Gone"


def test_unreadable_lines_are_reported() -> None:
    with pytest.raises(ValueError, match="cat.po:2"):
        read_po('msgid "a"\nmsgstr b\n', "cat.po")
    with pytest.raises(ValueError, match="continues nothing"):
        read_po('"a"\n')


def test_spanish_translates_the_text_of_the_code() -> None:
    i18n.set_language("es")
    assert i18n.language() == "es"
    assert _("&Language") == "&Idioma"
    assert _("A text without a translation") == "A text without a translation"
    entries = read_po(i18n.catalog_path("es").read_text(encoding="utf-8"))
    for entry in entries[1:]:
        if entry.context is None and entry.plural is None:
            assert _(entry.msgid) == entry.msgstr[0]
    i18n.set_language("en")
    assert _("&Language") == "&Language"
    assert ngettext("{n} room", "{n} rooms", 2) == "{n} rooms"
    assert pgettext("band", "Band") == "Band"


def test_pseudo_language_brackets_every_text() -> None:
    i18n.set_language(i18n.PSEUDO)
    assert _("Open") == "⟦Open⟧"
    assert ngettext("{n} room", "{n} rooms", 1).format(n=1) == "⟦1 room⟧"
    assert pgettext("band", "Band") == "⟦Band⟧"
    with pytest.raises(ValueError, match="no catalog"):
        i18n.set_language("fr")


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({"LANG": "es_AR.UTF-8"}, "es"),
        ({"LANG": "es-419"}, "es"),
        ({"LANG": "en_GB.UTF-8"}, "en"),
        ({"LANG": "C.UTF-8"}, "en"),
        ({"LANG": "fr_FR.UTF-8"}, "en"),  # no French catalog
        ({"LANGUAGE": "fr:es", "LANG": "fr_FR"}, "es"),  # the first with a catalog
        ({"LANGUAGE": "en:es", "LANG": "es_ES"}, "en"),
        ({"LC_ALL": "es_MX", "LANG": "en_US"}, "es"),
        ({"LC_MESSAGES": "es_CL.UTF-8"}, "es"),
    ],
)
def test_the_system_language(monkeypatch, variables, expected) -> None:
    for name in ("LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(name, raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    assert i18n.system_language() == expected


def test_what_chooses_the_language(monkeypatch, english) -> None:
    monkeypatch.setenv("LANG", "es_ES.UTF-8")
    monkeypatch.delenv("LANGUAGE", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    monkeypatch.setenv(i18n.ENV, "en")
    assert i18n.chosen_language() == ("en", i18n.ENV)
    monkeypatch.setenv(i18n.ENV, "pseudo")
    assert i18n.chosen_language() == ("pseudo", i18n.ENV)
    monkeypatch.setenv(i18n.ENV, "de")
    assert i18n.chosen_language() == ("en", i18n.ENV)  # no German catalog

    monkeypatch.delenv(i18n.ENV)
    assert i18n.saved_language() is None
    assert i18n.chosen_language() == ("es", "system")
    english.write_text(json.dumps({"other": 1}))
    i18n.save_language("en")
    assert json.loads(english.read_text()) == {"other": 1, "language": "en"}
    assert i18n.chosen_language() == ("en", "saved")
    i18n.set_language(None)
    assert i18n.language() == "en"
    i18n.save_language(None)
    assert json.loads(english.read_text()) == {"other": 1}
    assert i18n.chosen_language() == ("es", "system")
    with pytest.raises(ValueError, match="no catalog"):
        i18n.save_language("fr")

    english.write_text("not json")
    assert i18n.saved_language() is None
    english.write_text(json.dumps({"language": "klingon"}))
    assert i18n.saved_language() is None


def test_info_names_the_language(capsys, english) -> None:
    from acoustic_spaces.cli import main

    assert main(["info"]) == 0
    assert "language: English (ACOUSTIC_SPACES_LANG)" in capsys.readouterr().out


# -- the catalog tool -------------------------------------------------------------------


@pytest.fixture
def catalog_tool(monkeypatch):
    if not (TOOLS / "catalog.py").exists():
        pytest.skip("tools/ is in the source checkout only")
    monkeypatch.syspath_prepend(str(TOOLS))
    import catalog

    yield catalog
    sys.modules.pop("catalog", None)


def test_the_catalogs_are_complete(catalog_tool, capsys) -> None:
    """What CI checks: every catalog holds the marked text of the code, translated."""
    assert catalog_tool.check() == 0, capsys.readouterr().err


def _package(tmp_path: Path, source: str) -> Path:
    package = tmp_path / "acoustic_spaces"
    package.mkdir(exist_ok=True)
    (package / "module.py").write_text(textwrap.dedent(source))
    return package


def test_collecting_the_marked_text(catalog_tool, tmp_path) -> None:
    package = _package(
        tmp_path,
        """
        from acoustic_spaces.i18n import N_, _, ngettext, pgettext

        LABELS = [N_("Open"), N_("Save")]

        def show(label, n):
            # Translators: the number of rooms
            # in the scene.
            print(_("{count} rooms").format(count=n), ngettext("one", "{n} many", n))
            print(_(label), pgettext("band", "Band"), _("Open"))
        """,
    )
    entries = catalog_tool.extract(package)
    assert [(e.context, e.msgid, e.plural) for e in entries] == [
        (None, "Open", None),
        (None, "Save", None),
        (None, "{count} rooms", None),
        (None, "one", "{n} many"),
        ("band", "Band", None),
    ]
    rooms = entries[2]
    assert rooms.extracted == ["Translators: the number of rooms in the scene."]
    assert rooms.flags == ["python-brace-format"]
    assert rooms.references == ["acoustic_spaces/module.py"]


def test_collecting_the_examples(catalog_tool, tmp_path) -> None:
    """Each example scene's title and the comment at its top go into the catalogs."""
    scene = tmp_path / "examples" / "hall" / "scene.yaml"
    scene.parent.mkdir(parents=True)
    scene.write_text(
        "# A hall.\n# Two seats.\nname: Hall (10 m)\nmaterials:\n  name: not this\n",
        encoding="utf-8",
    )
    entries = catalog_tool.extract_examples(tmp_path / "examples")
    assert [(e.context, e.msgid) for e in entries] == [
        ("example", "Hall (10 m)"),
        ("example", "A hall. Two seats."),
    ]
    assert entries[0].references == ["examples/hall/scene.yaml"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ('_(f"{x} rooms")', "f-string"),
        ('_("{} rooms")', "names"),
        ('_("{0} rooms")', "names"),
        ('_("")', "empty"),
        ('pgettext(context, "Band")', "context"),
        ('_("{x rooms")', "in"),
    ],
)
def test_text_that_cannot_be_translated(catalog_tool, tmp_path, source, message) -> None:
    package = _package(tmp_path, f"x = 1\nprint({source})\n")
    with pytest.raises(catalog_tool.MarkError, match=message):
        catalog_tool.extract(package)


def test_a_variable_named_underscore_hides_the_translation(catalog_tool, tmp_path) -> None:
    package = _package(
        tmp_path,
        """
        def pick():
            path, _ = open_dialog()
            return _("Open")
        """,
    )
    with pytest.raises(catalog_tool.MarkError, match="rename the variable"):
        catalog_tool.extract(package)
    _package(tmp_path, "f = lambda *_: print(1)\nprint(_('Open'))\n")
    assert [e.msgid for e in catalog_tool.extract(tmp_path / "acoustic_spaces")] == ["Open"]
    # A comprehension or a lambda has a scope of its own: its _ hides nothing outside.
    _package(
        tmp_path,
        """
        def pick(pairs):
            names = [name for name, _ in pairs]
            return _("Open {names}").format(names=names)
        """,
    )
    assert [e.msgid for e in catalog_tool.extract(tmp_path / "acoustic_spaces")] == ["Open {names}"]


def test_problems_of_a_translation(catalog_tool) -> None:
    def problems(msgid, msgstr, plural=None):
        return catalog_tool.problems(Entry(msgid, msgstr, plural=plural))

    assert problems("Open {path}", ["Abrir {path}"]) == []
    assert problems("Open {path}", ["Abrir {ruta}"])
    assert problems("{value:.1f} m", ["{value} m"])
    assert problems("%(prog)s: error", ["%(prog)s: fallo"]) == []
    assert problems("%(prog)s: error", ["error"])
    assert problems("&Open", ["&Abrir"]) == []
    assert problems("&Open", ["Abrir"])
    assert problems("Save && quit", ["Guardar y salir"]) == []
    assert problems("{n} room", ["una sala", "{n} salas"], plural="{n} rooms") == []
    assert problems("{n} room", ["una sala", "salas"], plural="{n} rooms")
    assert problems("{n} room", ["{x} sala", "{n} salas"], plural="{n} rooms")
    assert problems("Open", [""]) == ["'Open' is not translated"]
    fuzzy = Entry("Open", ["Abrir"], flags=["fuzzy"])
    assert "fuzzy" in catalog_tool.problems(fuzzy)[0]


def test_updating_keeps_translations_and_suggests_close_ones(catalog_tool) -> None:
    header = Entry("", ["Plural-Forms: nplurals=2; plural=(n != 1);\n"])
    catalog = [
        header,
        Entry("Open", ["Abrir"], comments=["checked"]),
        Entry("Save the scene as a file", ["Guardar la escena en un archivo"]),
        Entry("Gone", ["Ido"]),
    ]
    marked = [
        Entry("Open"),
        Entry("Save the scene in a file"),
        Entry("{n} room", plural="{n} rooms"),
    ]
    merged = catalog_tool.merge(catalog, marked)
    opened, saved, rooms = merged[1:]
    assert opened.msgstr == ["Abrir"] and opened.comments == ["checked"]
    assert saved.fuzzy and saved.msgstr == ["Guardar la escena en un archivo"]
    assert saved.previous == ['msgid "Save the scene as a file"']
    assert rooms.msgstr == ["", ""]
    text = catalog_tool.write_po(merged)
    assert catalog_tool.write_po(read_po(text)) == text
    assert "#| msgid" in text and "#, fuzzy" in text


def test_long_text_is_written_in_lines(catalog_tool) -> None:
    long = "A tooltip " * 20 + "\nand a second line"
    text = catalog_tool.write_po([Entry("", ["Language: es\n"]), Entry(long, ["x"])])
    assert all(len(line) <= catalog_tool.WIDTH for line in text.splitlines())
    assert read_po(text)[1].msgid == long
