"""Checks of the code that need no run: every name a module takes from another is there
and every call of a module's function fits its signature. A mistake there shows only
when the code runs, which for a demo's video is at the end of a long run, and for a
command's lazy import only when someone runs that command."""

import ast
import importlib
import inspect
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
#: Every Python file of the project: the package, the examples, the tools, the
#: validation scripts and the tests.
FILES = sorted(
    [
        *SRC.glob("acoustic_spaces/**/*.py"),
        *EXAMPLES.glob("*/*.py"),
        *(ROOT / "tools").glob("*.py"),
        *(ROOT / "validation").glob("*.py"),
        *(ROOT / "tests").glob("*.py"),
    ]
)


def _name_of(path: Path) -> list[str]:
    """The dotted name of a module of the package, as parts."""
    parts = list(path.relative_to(SRC).with_suffix("").parts)
    return parts[:-1] if parts[-1] == "__init__" else parts


def _module_of(path: Path) -> types.ModuleType:
    """``path`` loaded as it is used: by its name in the package, or beside its script as
    the demos load their picture modules; skipped when it needs an extra not installed."""
    from acoustic_spaces.demos import sibling

    try:
        if SRC in path.parents:
            return importlib.import_module(".".join(_name_of(path)))
        return sibling(path, path.stem)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "acoustic_spaces":
            raise
        pytest.skip(f"{path.name} needs {exc.name}")


def _imports_that_are_not_there(path: Path) -> list[str]:
    """The imports of the package in ``path``, at its top or inside a function, that
    name a module or a name that is not there."""
    problems = []
    package = _name_of(path)[:-1] if SRC in path.parents else None
    if package is not None and path.stem == "__init__":
        package = _name_of(path)
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None:
                    continue
                base = package[: len(package) - node.level + 1]
                name = ".".join(base + ([node.module] if node.module else []))
            else:
                name = node.module or ""
            wanted = [a.name for a in node.names if a.name != "*"]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "acoustic_spaces":
                    try:
                        importlib.import_module(alias.name)
                    except ModuleNotFoundError as exc:
                        if exc.name and exc.name.split(".")[0] == "acoustic_spaces":
                            problems.append(f"{path.name}:{node.lineno}: no {alias.name}")
            continue
        else:
            continue
        if name.split(".")[0] != "acoustic_spaces":
            continue
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "acoustic_spaces":
                problems.append(f"{path.name}:{node.lineno}: no {name}")
            continue  # otherwise it needs an extra that is not installed
        for item in wanted:
            if hasattr(module, item):
                continue
            try:
                importlib.import_module(f"{name}.{item}")
            except ModuleNotFoundError as exc:
                # A module that is there but needs an extra that is not installed is not
                # a mistake of the code.
                if exc.name and exc.name.split(".")[0] == "acoustic_spaces":
                    problems.append(f"{path.name}:{node.lineno}: {name} has no {item}")
    return problems


def _what_code_uses_that_is_not_there(path: Path) -> list[str]:
    """The attributes ``path`` takes from modules that do not have them, and the calls of
    their functions (and of the functions it imports) that their signatures refuse: the
    mistakes that only show when a step runs, such as a demo's video at its end."""
    module = _module_of(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems: list[str] = []

    def siblings_in(node) -> dict[str, types.ModuleType]:
        found = {}
        for item in ast.walk(node):
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and isinstance(item.value, ast.Call)
                and getattr(item.value.func, "id", None) == "sibling"
                and len(item.value.args) == 2
                and isinstance(item.value.args[1], ast.Constant)
            ):
                found[item.targets[0].id] = _module_of(
                    path.parent / f"{item.value.args[1].value}.py"
                )
        return found

    def imports_in(node) -> dict[str, object]:
        """What the imports of the package inside ``node`` bind: the names a lazy import
        brings in are checked as the module's own are."""
        found: dict[str, object] = {}
        for item in ast.walk(node):
            if not (
                isinstance(item, ast.ImportFrom)
                and not item.level
                and (item.module or "").split(".")[0] == "acoustic_spaces"
            ):
                continue
            try:
                source = importlib.import_module(item.module)
            except ModuleNotFoundError:
                continue  # an extra not installed; a missing module fails the import test
            for alias in item.names:
                if hasattr(source, alias.name):
                    found[alias.asname or alias.name] = getattr(source, alias.name)
                    continue
                try:
                    found[alias.asname or alias.name] = importlib.import_module(
                        f"{item.module}.{alias.name}"
                    )
                except ModuleNotFoundError:
                    continue
        return found

    def assigned_in(node) -> set[str]:
        names = set()
        for item in ast.walk(node):
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                names.add(item.id)
            elif isinstance(item, ast.arg):
                names.add(item.arg)
            elif isinstance(item, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in item.names)
        return names

    def check(scope, names: dict) -> None:
        for item in ast.walk(scope):
            if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name):
                target = names.get(item.value.id)
                if isinstance(target, types.ModuleType) and not hasattr(target, item.attr):
                    problems.append(
                        f"{path.name}:{item.lineno}: {item.value.id}.{item.attr} is not there"
                    )
            if not isinstance(item, ast.Call):
                continue
            func = item.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                owner = names.get(func.value.id)
                target = (
                    getattr(owner, func.attr, None) if isinstance(owner, types.ModuleType) else None
                )
            elif isinstance(func, ast.Name):
                target = names.get(func.id)
                target = None if isinstance(target, types.ModuleType) else target
            else:
                target = None
            if (
                not callable(target)
                or isinstance(target, type)
                and issubclass(target, BaseException)
            ):
                continue
            if any(isinstance(a, ast.Starred) for a in item.args) or any(
                k.arg is None for k in item.keywords
            ):
                continue
            try:
                signature = inspect.signature(target)
            except (TypeError, ValueError):
                continue
            try:
                signature.bind(*item.args, **{k.arg: None for k in item.keywords})
            except TypeError as exc:
                problems.append(f"{path.name}:{item.lineno}: {ast.unparse(func)}(): {exc}")

    top = {**vars(module), **siblings_in(tree)}
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda))]
    inner = {id(n) for f in functions for n in ast.walk(f) if n is not f}
    for function in functions:
        if id(function) in inner:
            continue  # checked with the function round it
        local = {**imports_in(function), **siblings_in(function)}
        shadowed = assigned_in(function) - set(local)
        names = {k: v for k, v in top.items() if k not in shadowed} | local
        check(function, names)
    # The module's own statements, outside its functions.
    body = ast.Module(
        body=[n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef))],
        type_ignores=[],
    )
    check(body, top)
    return sorted(set(problems))


def _id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


@pytest.mark.parametrize("path", FILES, ids=_id)
def test_every_import_of_the_package_is_there(path: Path) -> None:
    """Every ``from acoustic_spaces... import name``, lazy ones inside functions too,
    names a module and a name that are there."""
    assert _imports_that_are_not_there(path) == []


@pytest.mark.parametrize(
    "path",
    [*sorted(EXAMPLES.glob("*/*.py")), *sorted(SRC.glob("acoustic_spaces/demos/*.py"))],
    ids=_id,
)
def test_the_examples_use_only_what_is_there(path: Path, tmp_path, monkeypatch) -> None:
    """Every attribute an example script or a demo module takes from a module is there
    and every call of a module's function fits its signature (the five demos once
    called their picture modules for a function only the shared one has, and failed at
    their video). Loading a script writes nothing: its work waits for it to be run."""
    pytest.importorskip("PIL")
    monkeypatch.chdir(tmp_path)
    assert _what_code_uses_that_is_not_there(path) == []
    assert list(tmp_path.iterdir()) == []
