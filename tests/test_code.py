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


#: The products the package's sources leave to the linear-algebra library, per file:
#: ``@``, and the calls of ``dot``, ``matmul``, ``tensordot``, ``inner``, ``vdot`` and
#: ``multi_dot``, with the library calls whose sums also go to it and so depend on how
#: many threads it has: ``np.linalg.norm`` without an axis, ``convolve`` and
#: ``correlate`` of numpy and scipy, ``least_squares``, ``RBFInterpolator`` and
#: ``einsum`` with ``optimize``. Each stays for the reason given: for most, on these
#: operands the library uses one thread, or its result is not one the package keeps.
LIBRARY_SUMS = {
    # ratio @ self.mean: 16 groups at most times the moment slots (one thread up to
    # 28,800 slots, 288 s of response); least_squares: a solver.
    "calibration.py": 2,
    # The demos' pictures and sounds: band weights, 2- and 3-vectors, a 2 ms smoothing.
    "demos/audio.py": 2,
    "demos/indoors.py": 1,
    "demos/passby.py": 3,
    "demos/pipeline.py": 1,
    # 3-vectors and 3 x 3 frames; the positions of one balloon's measurement points.
    "directivity.py": 6,
    # 2- and 3-vectors.
    "doc29/noise.py": 1,
    "doc29/path.py": 1,
    "doc29/source.py": 1,
    # A 3-vector times a 3 x 3 frame, one path at a time.
    "doppler.py": 1,
    # A 3 x 3 check; ways @ centre over at most 65,536 ways (one thread below 153,600);
    # the 5-degree grid's 2664 directions times 3 x 3 axes.
    "emission.py": 3,
    # (paths, 3) @ axis: one thread below 153,600 paths (at most 5,262 seen).
    "engine/coherent.py": 2,
    # A 3-vector's norm.
    "engine/coverage.py": 1,
    # 2- and 3-vectors.
    "engine/crests.py": 22,
    # 3-vectors, and one plane's normal times a point.
    "engine/diffraction.py": 4,
    # 3-vectors, and the few planes or faces of one path or one plane.
    "engine/early.py": 22,
    "engine/edge_fade.py": 1,
    # Integers: exact in any order.
    "engine/gpu.py": 1,
    # (paths, 3) @ normal: one thread below 153,600 paths.
    "engine/grazing.py": 1,
    # The fit's band weights and frequencies, and least_squares: a solver and its input.
    "engine/impedance.py": 7,
    # 3-vectors; in refracted.py also the points on a row's upright edges times one (one
    # thread below 153,600 edges).
    "engine/leaks.py": 2,
    "engine/refracted.py": 22,
    # (paths, 3) and (late directions, 3) @ axis: one thread below 153,600 rows;
    # 3-vectors.
    "engine/response.py": 4,
    # 2-vectors; the ends of a row's upright faces times a 2-vector, two terms each.
    "engine/row_paths.py": 5,
    # 3-vectors.
    "engine/section.py": 1,
    "engine/simulate.py": 2,
    # A terrain profile's points times a 2-vector; 2- and 3-vectors.
    "engine/terrain.py": 6,
    # np.convolve of the high-pass filter (at most about 5,900 taps) and of 3 taps.
    "engine/wave.py": 3,
    # A weather profile's nodes times 3-vectors.
    "engine/weather.py": 7,
    # np.convolve of order + 1 terms.
    "engine/wide_angle.py": 1,
    # The shoelace sum of a builder's outline.
    "geometry/builders.py": 2,
    # 2- and 3-vectors and 4 x 4 transforms.
    "geometry/cells.py": 2,
    "geometry/io.py": 5,
    "geometry/outdoor.py": 4,
    "geometry/triangulate.py": 1,
    # _TO_GLTF is a signed permutation: each element of a product is one exact term,
    # the same in any order; norms of a quaternion and of a bounding box's diagonal.
    "gltf.py": 8,
    # The editor's views.
    "gui/movement.py": 1,
    "gui/plan_view.py": 11,
    "gui/viewport.py": 10,
    # Live head-tracked playback: single precision, in real time, not a result on file.
    "headtracking.py": 4,
    # A fit's band weights times its frequencies.
    "materials/ground.py": 1,
    # The STI's seven bands.
    "parameters.py": 2,
    # A 3-vector's norm; integers, exact in any order.
    "room.py": 2,
    # 3-vectors, a 3 x 3 frame, and the fixed 15-degree grid that counts the harmonics.
    "rotorcraft.py": 7,
    "scene/model.py": 1,
    # The spherical head's grid (2048 directions) times an ear's axis.
    "spatial/hrtf.py": 1,
    # Loudspeaker layouts: tens of speakers.
    "spatial/loudspeakers.py": 13,
    # 3-vectors and a quaternion.
    "spatial/microphone.py": 3,
    "spatial/rotation.py": 1,
    # Plots.
    "viz.py": 9,
}

_PRODUCTS = {"dot", "matmul", "tensordot", "inner", "vdot", "multi_dot"}
_SUMS = {"convolve", "correlate", "least_squares", "RBFInterpolator"}
_LIBRARIES = {"np", "numpy", "scipy", "linalg", "signal", "optimize", "interpolate"}


def _library_sums(source: str) -> list[int]:
    """The lines of the products and sums ``source`` leaves to the library (see
    :data:`LIBRARY_SUMS`), once for each."""
    tree = ast.parse(source)
    imported = {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and not node.level
        and (node.module or "").split(".")[0] in ("numpy", "scipy")
        for alias in node.names
    }
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
            lines.append(node.lineno)
            continue
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in imported:
            name = imported[func.id]
        elif isinstance(func, ast.Attribute):
            name, owner = func.attr, ast.unparse(func.value).split(".")
            if owner == ["products"]:
                continue
            # Any x.dot(y), and the library's functions by its modules' names.
            if name not in _PRODUCTS and not {owner[0], owner[-1]} & _LIBRARIES:
                continue
        else:
            continue
        keywords = {k.arg for k in node.keywords}
        if (
            name in _PRODUCTS
            or name in _SUMS
            or (name == "norm" and len(node.args) < 3 and "axis" not in keywords)
            or (name == "einsum" and "optimize" in keywords)
        ):
            lines.append(node.lineno)
    return sorted(lines)


def test_the_package_adds_its_sums_in_a_fixed_order() -> None:
    """Every product the package's sources leave to the linear-algebra library is one
    that :data:`LIBRARY_SUMS` counts and says why: a new one would give results whose
    last digits depend on how many threads the library has."""
    package = SRC / "acoustic_spaces"
    if not package.is_dir():
        pytest.skip("the package's sources are in the source checkout only")
    found = {}
    for path in sorted(package.rglob("*.py")):
        lines = _library_sums(path.read_text(encoding="utf-8"))
        if lines:
            found[path.relative_to(package).as_posix()] = lines
    problems = [
        f"{name}: {len(found.get(name, []))} products or sums left to the library "
        f"(lines {', '.join(map(str, found.get(name, []))) or 'none'}), "
        f"LIBRARY_SUMS has {LIBRARY_SUMS.get(name, 0)}"
        for name in sorted(set(found) | set(LIBRARY_SUMS))
        if len(found.get(name, [])) != LIBRARY_SUMS.get(name, 0)
    ]
    assert not problems, (
        "\n".join(problems) + "\nSums are added in a fixed order with acoustic_spaces.products "
        "(products.product, add_product, dot, dots), so that a result has the same bits on any "
        "number of library threads: write a new product with it. For a product of small fixed "
        "operands, raise this file's count in LIBRARY_SUMS (tests/test_code.py) and say there "
        "why it stays; where a product moved to the module, lower it."
    )
