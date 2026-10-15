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

The editor's tests (`tests/test_gui.py`, with the `gui` extra installed:
`uv sync --extra gui`) use Qt's off-screen platform unless `QT_QPA_PLATFORM` says
otherwise; there the 3D view shows a note instead and its click tests are skipped. To
run those too on Linux without a display:
`QT_QPA_PLATFORM=xcb xvfb-run -a uv run pytest tests/test_gui.py`.

Example scenes live in `examples/<name>/scene.yaml`, with a comment at the top that
`acoustic-spaces examples` shows; wheels carry them as `acoustic_spaces/example_scenes`.

## Documentation

```sh
python -m pip install -e ".[docs]"     # or: uv sync --extra docs
python -m sphinx -W -b html docs docs/_build/html
```

The front page is the README, a short summary that links to the user guide's pages in
`docs/` (getting started, the editor, scene files, simulation and so on); new features
get their details there and a line on the front page if they deserve one.
`docs/validation.md` is the validation report and
`docs/api.md` the API reference, generated from the docstrings.

`docs/screenshots.py` draws most of the documentation's pictures into `docs/images` from the
example scenes, with fixed seeds; run it again after a change that shows in them. The 3D
pictures and the editor need a display with OpenGL (`xvfb-run` gives one):

```sh
xvfb-run -a -s "-screen 0 1920x1080x24" python docs/screenshots.py --bras path/to/BRAS
```

## Validation scripts

`validation/` holds the longer checks behind the validation report. They are not part of
the test suite:

```sh
python -m pip install pyroomacoustics matplotlib
python validation/pyroomacoustics_crosscheck.py --json results.json --plots docs/validation
python validation/convolver_benchmark.py
python validation/wave_solver.py --plot docs/images/wave.png
```

## Releasing

The version is written in one place, `src/acoustic_spaces/_version.py`.

1. Set the version there and add its section to `CHANGELOG.md` (`## X.Y.Z`).
2. Merge to `main`, then tag that commit `vX.Y.Z`: push the tag
   (`git tag vX.Y.Z && git push origin vX.Y.Z`), or on GitHub's Releases page draft a new
   release, create the tag `vX.Y.Z` on `main` there and publish it.
3. CI builds and tests the wheels for every platform and Python version, checks that the
   tag matches the version, and attaches the wheels and the source distribution to the
   GitHub release. A release made by CI takes that version's changelog section as its
   notes; one made on the Releases page keeps its own notes, or gets the changelog's
   when it has none.

Publishing to PyPI is off. It needs a one-time setup by a maintainer:

1. On PyPI, add a trusted publisher for the project `acoustic-spaces`: this repository,
   workflow `ci.yml`, environment `pypi`.
2. In the repository settings, create the environment `pypi` (optionally requiring an
   approval) and set the repository variable `PYPI_PUBLISH` to `true`.

From then on every release tag also uploads to PyPI, with no token stored anywhere.
