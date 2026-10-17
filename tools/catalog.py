"""Keep the translation catalogs (src/acoustic_spaces/locale/*.po) in step with the code.

    python tools/catalog.py update   collect the text marked in the code into every catalog
    python tools/catalog.py check    fail when a catalog is out of date or not translated

``update`` adds new text untranslated (or, when it is close to text that is gone, with the
old translation marked fuzzy for review), and drops text the code no longer has. ``check``
is what CI runs: every catalog must hold exactly the marked text, written as ``update``
writes it, with every entry translated, none fuzzy, and the same fields ({name}, %s) and
menu shortcut (&) as the English. It needs only Python's standard library.

Text is marked with ``_("...")``, ``N_("...")``, ``ngettext("...", "...", n)`` and
``pgettext("context", "...")`` (see acoustic_spaces.i18n). A comment starting with
``Translators:`` on the lines just above goes into the catalog for the translator.
The example scenes' titles and descriptions (examples/*/scene.yaml) are collected too,
in the context ``example``.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import io
import re
import string
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "acoustic_spaces"
LOCALE = PACKAGE / "locale"
EXAMPLES = ROOT / "examples"
#: The context of the example scenes' titles and descriptions.
EXAMPLE_CONTEXT = "example"
WIDTH = 79

# acoustic_spaces.i18n by its path, without importing the package (and numpy with it).
_spec = importlib.util.spec_from_file_location("_i18n", PACKAGE / "i18n.py")
i18n = importlib.util.module_from_spec(_spec)
sys.modules["_i18n"] = i18n  # dataclasses look their module up
_spec.loader.exec_module(i18n)
Entry = i18n.Entry

#: The marking functions: where their context, text and plural are among the arguments.
MARKERS = {
    "_": (None, 0, None),
    "N_": (None, 0, None),
    "ngettext": (None, 0, 1),
    "pgettext": (0, 1, None),
}


class MarkError(Exception):
    pass


# -- collecting the marked text -------------------------------------------------------


def _translator_comments(source: str) -> dict[int, str]:
    """The ``# Translators:`` comments, by the line they end on (continued comment lines
    join the first)."""
    found: dict[int, str] = {}
    last = None
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line, text = token.start[0], token.string.lstrip("#").strip()
        if text.startswith("Translators:"):
            found[line], last = text, line
        elif last is not None and line == last + 1:
            found[line] = found.pop(last) + " " + text
            last = line
        else:
            last = None
    return found


def _literal(node: ast.expr, where: str) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        raise MarkError(f"{where}: an f-string cannot be translated; use named fields")
    return None  # a variable: text marked elsewhere with N_()


def _check_fields(text: str, where: str) -> None:
    try:
        fields = [name for _, name, _, _ in string.Formatter().parse(text) if name is not None]
    except ValueError as exc:
        raise MarkError(f"{where}: {exc} in {text!r}") from exc
    if any(name == "" or name.isdigit() for name in fields):
        raise MarkError(f"{where}: give fields names, so a translation can reorder them")


def _calls_translation(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_"


_SCOPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _own_scope(function: ast.AST):
    """The nodes of ``function``'s own scope: not those of the functions, lambdas and
    comprehensions inside it."""
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _check_shadowing(tree: ast.Module, reference: str) -> None:
    """A function that names a variable ``_`` (``path, _ = ...``) cannot call ``_()``,
    nor can the functions inside it."""
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        args = function.args
        names = [a.arg for a in [*args.posonlyargs, *args.args, *args.kwonlyargs]]
        names += [a.arg for a in (args.vararg, args.kwarg) if a is not None]
        bound = "_" in names or any(
            isinstance(n, ast.Name) and n.id == "_" and isinstance(n.ctx, ast.Store)
            for n in _own_scope(function)
        )
        if bound and any(_calls_translation(n) for n in ast.walk(function)):
            raise MarkError(
                f"{reference}:{function.lineno}: _ is a variable here and the translation "
                "too; rename the variable"
            )


def extract(package: Path = PACKAGE) -> list[Entry]:
    """The marked text of the package's modules, in order of first appearance."""
    entries: dict[tuple, Entry] = {}
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        reference = path.relative_to(package.parent).as_posix()
        comments = _translator_comments(source)
        tree = ast.parse(source, str(path))
        _check_shadowing(tree, reference)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in MARKERS
        ]
        for node in sorted(calls, key=lambda n: (n.lineno, n.col_offset)):
            where = f"{reference}:{node.lineno}"
            context_at, text_at, plural_at = MARKERS[node.func.id]
            needed = max(i for i in (context_at, text_at, plural_at) if i is not None)
            if len(node.args) <= needed:
                continue
            text = _literal(node.args[text_at], where)
            if text is None:
                continue
            if not text:
                raise MarkError(f"{where}: empty text")
            context = plural = None
            if context_at is not None:
                context = _literal(node.args[context_at], where)
                if context is None:
                    raise MarkError(f"{where}: the context must be a literal string")
            if plural_at is not None:
                plural = _literal(node.args[plural_at], where)
                if plural is None:
                    raise MarkError(f"{where}: the plural must be a literal string")
            for value in (text, plural):
                if value is not None:
                    _check_fields(value, where)
            entry = entries.setdefault((context, text), Entry(text, context=context))
            if plural is not None:
                if entry.plural not in (None, plural):
                    raise MarkError(f"{where}: {text!r} has two plurals")
                entry.plural = plural
            if reference not in entry.references:
                entry.references.append(reference)
            note = comments.get(node.lineno - 1) or comments.get(node.lineno)
            if note and note not in entry.extracted:
                entry.extracted.append(note)
            entry.flags = _format_flags(text, plural)
    return list(entries.values())


def _format_flags(*texts: str | None) -> list[str]:
    flags = []
    if any(t and _brace_fields(t) for t in texts):
        flags.append("python-brace-format")
    if any(t and _percent_fields(t) for t in texts):
        flags.append("python-format")
    return flags


# -- comparing a translation with its English -----------------------------------------


def _brace_fields(text: str) -> list[str]:
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError:
        return ["<unbalanced braces>"]
    fields = []
    for _, name, spec, conversion in parsed:
        if name is not None:
            fields.append(
                f"{{{name}{'!' + conversion if conversion else ''}{':' + spec if spec else ''}}}"
            )
    return sorted(fields)


_PERCENT = re.compile(r"%(\([A-Za-z_]\w*\))?[-#0 +]*\d*(?:\.\d+)?[sdirfgeExXc%]")


def _percent_fields(text: str) -> list[str]:
    return [m.group(0) for m in _PERCENT.finditer(text) if m.group(0) != "%%"]


def _accelerators(text: str) -> int:
    return len(re.findall(r"&(?=[^&\s])", text.replace("&&", "")))


def problems(entry: Entry) -> list[str]:
    """What is wrong with a translated entry."""
    if not entry.msgid:
        return []
    name = repr(entry.msgid if len(entry.msgid) < 60 else entry.msgid[:57] + "...")
    if entry.fuzzy:
        return [f"{name} is fuzzy: check the translation and drop the flag"]
    if not entry.translated:
        return [f"{name} is not translated"]
    found = []
    if entry.plural is None:
        pairs = [(entry.msgstr[0], entry.msgid, True)]
    else:
        # Every form but the last may leave out a field ("una sala" for "1 room").
        last = len(entry.msgstr) - 1
        pairs = [(text, entry.plural, k == last) for k, text in enumerate(entry.msgstr)]
    for text, source, exact in pairs:
        for kind in (_brace_fields, _percent_fields):
            fields, wanted = kind(text), kind(source)
            if fields != wanted if exact else not set(fields) <= set(wanted):
                found.append(f"{name}: fields {fields} where the English has {wanted}")
        if _accelerators(text) != _accelerators(source):
            found.append(f"{name}: the shortcut (&) of the English is missing or doubled")
    return found


# -- writing catalogs ------------------------------------------------------------------


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n")


def _field(keyword: str, text: str) -> list[str]:
    """``keyword "text"``, on one line when it fits, else in lines of words."""
    if len(keyword) + len(_escape(text)) + 3 <= WIDTH and "\n" not in text[:-1]:
        return [f'{keyword} "{_escape(text)}"']
    lines = [f'{keyword} ""']
    for piece in re.findall(r"[^\n]*\n|[^\n]+", text):
        line = ""
        for word in re.findall(r"\S*\s*", piece):
            if line and len(_escape(line + word)) + 2 > WIDTH:
                lines.append(f'"{_escape(line)}"')
                line = ""
            line += word
        if line:
            lines.append(f'"{_escape(line)}"')
    return lines


def write_po(entries: list[Entry]) -> str:
    """The catalog's text, as ``update`` writes it."""
    blocks = []
    for entry in entries:
        lines = [f"# {c}".rstrip() for c in entry.comments]
        lines += [f"#. {c}" for c in entry.extracted]
        if entry.references:
            lines.append("#: " + " ".join(entry.references))
        flags = (["fuzzy"] if entry.fuzzy else []) + [f for f in entry.flags if f != "fuzzy"]
        if flags:
            lines.append("#, " + ", ".join(flags))
        lines += [f"#| {p}" for p in entry.previous]
        if entry.context is not None:
            lines += _field("msgctxt", entry.context)
        lines += _field("msgid", entry.msgid)
        if entry.plural is None:
            lines += _field("msgstr", entry.msgstr[0] if entry.msgstr else "")
        else:
            lines += _field("msgid_plural", entry.plural)
            for k, text in enumerate(entry.msgstr):
                lines += _field(f"msgstr[{k}]", text)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def _plurals(header: Entry) -> int:
    found = re.search(r"nplurals\s*=\s*(\d+)", header.msgstr[0])
    return int(found.group(1)) if found else 2


def merge(catalog: list[Entry], marked: list[Entry]) -> list[Entry]:
    """``marked`` with the translations (and translator comments) of ``catalog``."""
    header, old = catalog[0], {e.key: e for e in catalog[1:]}
    forms = _plurals(header)
    gone = [e for key, e in old.items() if key not in {m.key for m in marked} and e.translated]
    merged = [header]
    for entry in marked:
        new = Entry(
            entry.msgid,
            context=entry.context,
            plural=entry.plural,
            flags=list(entry.flags),
            extracted=list(entry.extracted),
            references=list(entry.references),
        )
        known = old.get(entry.key)
        if known is not None and known.plural == entry.plural:
            new.msgstr, new.comments = list(known.msgstr), list(known.comments)
            if known.fuzzy:
                new.flags.insert(0, "fuzzy")
                new.previous = list(known.previous)
        else:
            close = difflib.get_close_matches(entry.msgid, [e.msgid for e in gone], 1, 0.75)
            source = next((e for e in gone if close and e.msgid == close[0]), None)
            if source is not None and (source.plural is None) == (entry.plural is None):
                new.msgstr, new.comments = list(source.msgstr), list(source.comments)
                new.flags.insert(0, "fuzzy")
                new.previous = _field("msgid", source.msgid)
            else:
                new.msgstr = [""] * (1 if entry.plural is None else forms)
        merged.append(new)
    return merged


def extract_examples(folder: Path = EXAMPLES) -> list[Entry]:
    """The title (``name``) and description (the comment at the top) of every example
    scene, read as acoustic_spaces.examples reads them."""
    entries = []
    for scene in sorted(folder.glob("*/scene.yaml")):
        reference = scene.relative_to(folder.parent).as_posix()
        lines = scene.read_text(encoding="utf-8").splitlines()
        comment = []
        for line in lines:
            if not line.startswith("#"):
                break
            comment.append(line.lstrip("#").strip())
        names = [line[len("name:") :].strip() for line in lines if line.startswith("name:")]
        title = names[0] if names else ""
        for text in (title.strip("'\""), " ".join(comment)):
            if text:
                entry = Entry(text, context=EXAMPLE_CONTEXT, references=[reference])
                entry.flags = _format_flags(text)
                entries.append(entry)
    return entries


def marked_text() -> list[Entry]:
    """Everything the catalogs translate: the code's marked text, then the examples'."""
    return extract() + extract_examples()


def catalogs() -> list[Path]:
    return sorted(LOCALE.glob("*.po"))


def update() -> int:
    marked = marked_text()
    for path in catalogs():
        catalog = i18n.read_po(path.read_text(encoding="utf-8"), path.name)
        merged = merge(catalog, marked)
        path.write_text(write_po(merged), encoding="utf-8", newline="\n")
        left = sum(1 for e in merged[1:] if not e.translated)
        print(f"{path.name}: {len(merged) - 1} messages, {left} to translate")
    return 0


def check() -> int:
    marked = marked_text()
    failed = False
    for path in catalogs():
        text = path.read_text(encoding="utf-8")
        catalog = i18n.read_po(text, path.name)
        found = []
        keys = {e.key for e in catalog[1:]}
        missing = [m.msgid for m in marked if m.key not in keys]
        if missing:
            found.append(f"{len(missing)} new messages, first {missing[0]!r}")
        stale = len(keys - {m.key for m in marked})
        if stale:
            found.append(f"{stale} messages the code no longer has")
        if not found and write_po(merge(catalog, marked)) != text:
            found.append("written differently from what update writes")
        if found:
            found[-1] += ": run python tools/catalog.py update"
        for entry in catalog[1:]:
            found += problems(entry)
        if "Plural-Forms:" not in catalog[0].msgstr[0]:
            found.append("the header has no Plural-Forms")
        for problem in found:
            print(f"{path.relative_to(ROOT).as_posix()}: {problem}", file=sys.stderr)
        failed |= bool(found)
        if not found:
            print(f"{path.name}: {len(catalog) - 1} messages, all translated")
    return int(failed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["update", "check"])
    args = parser.parse_args(argv)
    try:
        return update() if args.command == "update" else check()
    except MarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
