# Contributing

Pull requests are welcome.

## Development setup

You need Python 3.10 or newer, a C++17 compiler and CMake 3.18 or newer.

```sh
uv sync --extra viz                  # or, with pip: python -m pip install -e ".[dev,viz]"
```

`uv sync` installs the `dev` group (pytest, h5py, ruff) along with the package, and
rebuilds the compiled core when the C++ sources or the build files change; pip rebuilds
it on every `pip install -e .`.

## Checks

CI runs these on Linux, macOS and Windows, and its logs are kept free of warnings, not
only of errors (with uv, put `uv run` in front of each):

```sh
ruff check .
ruff format --check .
pytest -W error
```

Tests that draw with OpenGL carry the `render` marker; without a display, run them with
`VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow` (Linux, with OSMesa installed).

## Documentation

```sh
python -m pip install -e ".[docs]"     # or: uv sync --extra docs
python -m sphinx -W -b html docs docs/_build/html
```

The front page is the README; `docs/validation.md` is the validation report and
`docs/api.md` the API reference, generated from the docstrings.

## Validation scripts

`validation/` holds the longer checks behind the validation report. They are not part of
the test suite:

```sh
python -m pip install pyroomacoustics matplotlib
python validation/pyroomacoustics_crosscheck.py --json results.json --plots docs/validation
python validation/convolver_benchmark.py
```

## Releasing

The version is written in one place, `src/acoustic_spaces/_version.py`.

1. Set the version there and add its section to `CHANGELOG.md` (`## X.Y.Z`).
2. Merge to `main`, then tag that commit `vX.Y.Z` and push the tag.
3. CI builds and tests the wheels for every platform and Python version, checks that the
   tag matches the version, and attaches the wheels and the source distribution to a
   GitHub release whose notes are that version's changelog section.

Publishing to PyPI needs a one-time setup by a maintainer:

1. On PyPI, add a trusted publisher for the project `acoustic-spaces`: this repository,
   workflow `ci.yml`, environment `pypi`.
2. In the repository settings, create the environment `pypi` (optionally requiring an
   approval) and set the repository variable `PYPI_PUBLISH` to `true`.

From then on every release tag also uploads to PyPI, with no token stored anywhere.
