"""What the demo scripts share: how a demo describes its steps and options, so that
``python NAME_demo.py STEP`` and the editor's **Run demo…** run the same code, and the
parts of the pipeline every demo goes through.

A demo script (``examples/church/church_demo.py``, say) builds a :class:`Demo` from its
steps, functions of the parsed arguments, and its :class:`Option` list, and ends with
``sys.exit(DEMO.main())``. The steps that write a scene and the simulation
(:data:`SIMULATE`, run by the framework through ``acoustic-spaces simulate``) are steps
like the others, so ``all`` runs a demo from nothing to its video, leaving out what is
there already (a scene file, a simulation's results).

- :mod:`acoustic_spaces.demos.audio`: WAV files, gated levels, A and K weighting.
- :mod:`acoustic_spaces.demos.music`: the recordings a demo plays, fetched and summed,
  and the phrases its tour moves on.
- :mod:`acoustic_spaces.demos.pipeline`: the simulate command and its HRTF, auralizing
  every listener with one gain, the tour's crossfades, the stills and the video.
- :mod:`acoustic_spaces.demos.pictures`: the stills' shared colours, fonts and parts,
  and the lines that say how the sound was made.
- :mod:`acoustic_spaces.demos.scene_demo` and :mod:`~acoustic_spaces.demos.scene_pictures`:
  the demo of any scene (``acoustic-spaces demo``), each source playing the WAV file its
  ``signal`` names.
- :mod:`acoustic_spaces.fetch`: downloads, whole or one member of a remote zip archive.
"""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from acoustic_spaces.examples import examples_dir

#: The kinds of option, for the parser and for the editor's form: a free text, a choice
#: from a list, a number, a flag, a file to read, a file the demo writes (``path``), a
#: folder, ``inputs``: a WAV file for each of some sources (``--input NAME=WAV``, as
#: often as needed), what they play instead of the demo's own recordings, and ``files``:
#: a WAV file for each of some names the option lists as its choices (``--stem
#: NAME=WAV``), such as the stems of a song.
KINDS = ("text", "choice", "int", "float", "flag", "file", "path", "folder", "inputs", "files")
PATH_KINDS = ("file", "path", "folder")
#: The kinds whose value is a dict of names and files.
MAP_KINDS = ("inputs", "files")


def shell_line(argv: Sequence[str]) -> str:
    """``argv`` as one line to paste into the shell of this system: quoted as ``sh``
    reads it, or on Windows as ``cmd`` and PowerShell do, where ``C:\\Users\\me`` stays
    as it is and only an argument with a space gets its double quotes."""
    if sys.platform == "win32":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


class _Inputs(argparse.Action):
    """``--input NAME=WAV``, once per source (or stem): a dict of names and files."""

    def __call__(self, parser, namespace, value, option_string=None) -> None:
        name, sep, file = str(value).partition("=")
        if not sep or not name.strip() or not file.strip():
            parser.error(f"{option_string} takes {self.metavar}, not {value!r}")
        inputs = dict(getattr(namespace, self.dest, None) or {})
        inputs[name.strip()] = Path(file.strip())
        setattr(namespace, self.dest, inputs)


@dataclass(frozen=True)
class Option:
    """A command line option of a demo (``--song``), and the editor's field for it."""

    #: The attribute on the parsed arguments (``source`` for ``--from``).
    name: str
    help: str
    default: Any = None
    kind: str = "text"
    choices: tuple[str, ...] = ()
    #: The steps it matters to, for the help; () for every step.
    steps: tuple[str, ...] = ()
    #: The flag, when it is not ``--name`` with dashes.
    flag: str | None = None
    #: The editor's file filter for a ``file`` option ("SOFA files (*.sofa)").
    filter: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: no option kind {self.kind!r}")
        if self.kind == "files" and not self.choices:
            raise ValueError(f"{self.name}: a files option lists the names it takes as choices")

    @property
    def option(self) -> str:
        return self.flag or "--" + self.name.replace("_", "-")

    @property
    def metavar(self) -> str:
        """How a ``files`` option is given (``STEM=WAV``, for ``--stem``), or ``inputs``."""
        return "SOURCE=WAV" if self.kind == "inputs" else f"{self.option.lstrip('-').upper()}=WAV"

    @property
    def description(self) -> str:
        """The help with the steps in front and the default behind."""
        text = f"{', '.join(self.steps)}: {self.help}" if self.steps else self.help
        if self.default is not None and self.kind != "flag":
            text += f" (default {self.default})"
        return text

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        if self.kind == "flag":
            parser.add_argument(self.option, dest=self.name, action="store_true", help=self.help)
            return
        if self.kind in MAP_KINDS:
            parser.add_argument(
                self.option,
                dest=self.name,
                action=_Inputs,
                default=None,
                metavar=self.metavar,
                help=self.description,
            )
            return
        kwargs: dict[str, Any] = {
            "dest": self.name,
            "default": self.default,
            "help": self.description,
        }
        if self.kind in PATH_KINDS:
            kwargs["type"] = Path
        elif self.kind == "int":
            kwargs["type"] = int
        elif self.kind == "float":
            kwargs["type"] = float
        if self.choices:
            kwargs["choices"] = list(self.choices)
        parser.add_argument(self.option, **kwargs)


@dataclass(frozen=True)
class Step:
    """A step of a demo: what ``python NAME_demo.py STEP`` runs."""

    name: str
    run: Callable[[argparse.Namespace], None] | None
    help: str
    #: Whether ``all`` may leave the step out (its result is there already).
    done: Callable[[argparse.Namespace], bool] | None = None


def input_option(steps: tuple[str, ...] = ("render",)) -> Option:
    """``--input SOURCE=WAV``: what a source plays instead of the demo's own recording,
    for the steps that read the feeds. The demo's :attr:`Demo.plays` lists the sources."""
    return Option(
        "inputs",
        "a WAV file a source plays instead of the demo's own recording, as SOURCE=WAV, once "
        "per source: mixed down to mono, brought to the gated RMS of the feed it replaces "
        "and cut or padded with silence to its length, so the demo's balance and tour hold",
        None,
        "inputs",
        steps,
        flag="--input",
        filter="WAV files (*.wav)",
    )


#: The simulation, run by the framework: ``acoustic-spaces simulate`` on each scene the
#: demo names (:meth:`Demo.simulations`), unless its results are there.
SIMULATE = Step("simulate", None, "simulate each scene of the demo (acoustic-spaces simulate)")


def common_options(work: str = "demo") -> list[Option]:
    """The options every demo takes: the work folder, the mixes' format and HRTF, ``--again``
    and ``--no-mp4``."""
    return [
        Option("work", "folder for everything the demo writes", Path(work), "folder"),
        Option("format", "output format the mixes take", "binaural"),
        Option(
            "hrtf",
            "binaural mixes through this SOFA file of HRTFs, your own or measured, instead of "
            "a spherical head model (the simulate command takes it too)",
            None,
            "file",
            filter="SOFA files (*.sofa)",
        ),
        Option("again", "render: mix the stops again, even the mixes that exist", False, "flag"),
        Option("no_mp4", "video: write the audio and the stills only", False, "flag"),
    ]


@dataclass
class Demo:
    """A demo: its steps in order (with :data:`SIMULATE` where the simulation goes),
    its options, and the scenes it simulates."""

    #: A short name (``stadium``), the example it plays in and a title for the editor.
    name: str
    example: str
    title: str
    #: What the demo does: the first paragraph of the script's docstring, on one line
    #: (the whole docstring may be given).
    description: str
    #: The script file (``Path(__file__)``), run from its folder.
    script: Path
    steps: list[Step]
    options: list[Option]
    #: The scenes ``acoustic-spaces simulate`` runs on, from the arguments
    #: (:class:`pipeline.Simulation` each); none for a demo that simulates in its steps.
    simulations: Callable[[argparse.Namespace], list] = lambda args: []
    #: The step that writes those scenes, when one does: the step to run before the
    #: simulation. A demo that simulates the example's own scene names none.
    scene_from: str | None = None
    #: The demo's own check of the arguments, run by :meth:`check` with the demo (in its
    #: folder) and the arguments: what is wrong, or None.
    problems: Callable[[Demo, argparse.Namespace], str | None] | None = None
    #: How a terminal runs the demo when not as ``python <script>``: the command
    #: (``acoustic-spaces demo``), the option it takes bare before the step (``scene``)
    #: and the module the editor runs with ``python -m``.
    program: str | None = None
    leading: str | None = None
    module: str | None = None
    #: Where the demo writes, from the arguments (paths resolved); by default the
    #: ``work`` option, or the demo's folder.
    work_of: Callable[[argparse.Namespace], Path] | None = None
    #: Fills in what the arguments leave to the scene (paths resolved): the demo of any
    #: scene takes the HRTFs its binaural output names when ``--hrtf`` gives none.
    completes: Callable[[argparse.Namespace], argparse.Namespace] | None = None
    #: The sources the demo feeds and what each plays by default (``{"subs": "Spicy Funk
    #: Cake, both channels"}``), from the arguments (paths resolved): the rows of the
    #: editor's map of inputs, and the names ``--input`` takes.
    plays: Callable[[argparse.Namespace], dict[str, str]] | None = None
    _folder: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.script = Path(self.script).resolve()
        self.description = " ".join((self.description or "").strip().split("\n\n")[0].split())
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: a step is listed twice")

    # -- description ----------------------------------------------------------------

    @property
    def folder(self) -> Path:
        """The folder the demo runs in: the script's, unless :meth:`at` moved it."""
        return self._folder or self.script.parent

    def at(self, folder: str | Path) -> Demo:
        """The same demo run in ``folder`` (a copy of the example): relative paths in its
        arguments count from there."""
        return replace(self, _folder=Path(folder))

    def step(self, name: str) -> Step:
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"{self.name}: no step {name!r}")

    def option(self, name: str) -> Option:
        for option in self.options:
            if option.name == name:
                return option
        raise KeyError(f"{self.name}: no option {name!r}")

    def work_folder(self, args: argparse.Namespace) -> Path:
        """Where a run with ``args`` writes (:attr:`work_of`)."""
        resolved = self.resolved(args)
        if self.work_of is not None:
            return Path(self.work_of(resolved))
        work = getattr(resolved, "work", None)
        return Path(work) if work else self.folder

    def arguments(
        self,
        args: argparse.Namespace | None = None,
        *,
        names_only: bool = False,
        relative_to: Path | None = None,
    ) -> list[str]:
        """The options of ``args`` that differ from their defaults, as a command line
        (``["--hrtf", "me.sofa"]``): what :meth:`command` shows and the editor runs.
        ``names_only`` gives a path by its file name, for a caption, and leaves out a
        path named as its default is, wherever it lies; ``relative_to`` gives a path
        under that folder relative to it."""
        argv: list[str] = []
        for option in self.options:
            value = getattr(args, option.name, option.default) if args is not None else None
            if value is None or value == option.default:
                continue
            if option.kind in MAP_KINDS:
                for name, file in value.items():
                    if names_only:
                        shown = Path(file).name
                    elif relative_to is not None:
                        shown = _relative(Path(file), relative_to)
                    else:
                        shown = str(file)
                    argv += [option.option, f"{name}={shown}"]
                continue
            if option.kind in PATH_KINDS and option.default is not None:
                if Path(value) == self.folder / option.default:  # the default, resolved
                    continue
                if names_only and Path(value).name == Path(option.default).name:
                    continue
            if option.kind == "flag":
                argv.append(option.option)
            elif names_only and option.kind in PATH_KINDS:
                argv += [option.option, Path(value).name]
            elif relative_to is not None and option.kind in PATH_KINDS:
                argv += [option.option, _relative(Path(value), relative_to)]
            else:
                argv += [option.option, str(value)]
        return argv

    def command(
        self, step: str, args: argparse.Namespace | None = None, *, names_only: bool = False
    ) -> str:
        """How to run ``step`` from the command line, with the options that differ from
        their defaults (``python church_demo.py scene``, or for a demo with a
        :attr:`program`, ``acoustic-spaces demo scene.yaml feeds``); ``names_only`` as
        :meth:`arguments` takes it. Paths in the folder the command runs in are given
        relative to it where the command reads them so: a demo run as a module, or a
        script run in its own folder."""
        here = self.module is not None or self.folder == self.script.parent
        argv = self.arguments(
            args, names_only=names_only, relative_to=self.folder if here else None
        )
        if self.program is None:
            return f"python {self.script.name} {step} {shell_line(argv)}".rstrip()
        lead: list[str] = []
        if self.leading is not None:
            flag = self.option(self.leading).option
            if flag in argv:
                k = argv.index(flag)
                lead, argv = argv[k + 1 : k + 2], argv[:k] + argv[k + 2 :]
        parts = [self.program, shell_line(lead), step, shell_line(argv)]
        return " ".join(part for part in parts if part)

    def run_argv(self, step: str, args: argparse.Namespace | None = None) -> list[str]:
        """What the editor runs for ``step``: this Python on the script (or with ``-m`` on
        the demo's :attr:`module`), the step and the options that differ from their
        defaults, paths in full so that it runs from any folder."""
        start = ["-m", self.module] if self.module else [str(self.script)]
        values = self.resolved(args) if args is not None else None
        return [sys.executable, *start, step, *self.arguments(values)]

    # -- arguments ------------------------------------------------------------------

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.description)
        choices = [s.name for s in self.steps] + ["all"]
        parser.add_argument(
            "step",
            choices=choices,
            help="; ".join(f"{s.name}: {s.help}" for s in self.steps)
            + "; all: every step, leaving out what is done already",
        )
        for option in self.options:
            option.add_to(parser)
        return parser

    def defaults(self) -> argparse.Namespace:
        """Every option at its default."""
        return argparse.Namespace(**{o.name: o.default for o in self.options})

    def resolved(self, args: argparse.Namespace) -> argparse.Namespace:
        """The arguments with relative paths counted from the demo's folder, and what
        they leave to the scene filled in (:attr:`completes`)."""
        values = vars(args).copy()
        for option in self.options:
            value = values.get(option.name)
            if option.kind in PATH_KINDS and value is not None:
                path = Path(value)
                values[option.name] = path if path.is_absolute() else self.folder / path
            elif option.kind in MAP_KINDS and value:
                values[option.name] = {
                    name: Path(f) if Path(f).is_absolute() else self.folder / f
                    for name, f in value.items()
                }
        resolved = argparse.Namespace(**values)
        return self.completes(resolved) if self.completes is not None else resolved

    def check(self, args: argparse.Namespace) -> str | None:
        """What is wrong with the arguments, or None."""
        if getattr(args, "hrtf", None) and getattr(args, "format", "binaural") != "binaural":
            return "--hrtf is for binaural mixes"
        for option in self.options:
            value = getattr(args, option.name, None)
            if option.kind in ("int", "float") and value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return f"{option.option}: a number, not {value!r}"
        resolved = self.resolved(args)
        inputs = getattr(resolved, "inputs", None) or {}
        if inputs and self.plays is not None:
            fed = self.plays(resolved)
            for name, file in inputs.items():
                if name not in fed:
                    return f"--input {name}: no such source plays here; it has {', '.join(fed)}"
                if not Path(file).is_file():
                    return f"--input {name}: no {file}"
        for option in self.options:
            if option.kind != "files":
                continue
            for name, file in (getattr(resolved, option.name, None) or {}).items():
                if name not in option.choices:
                    return f"{option.option} {name}: not one of {', '.join(option.choices)}"
                if not Path(file).is_file():
                    return f"{option.option} {name}: no {file}"
        if self.problems is not None:
            return self.problems(self, args)
        return None

    # -- running --------------------------------------------------------------------

    def simulate(self, args: argparse.Namespace, only_missing: bool = False) -> None:
        """``acoustic-spaces simulate`` on each scene of the demo, in this process."""
        from acoustic_spaces.cli import main as cli
        from acoustic_spaces.demos.pipeline import simulate_argv, simulate_command, simulated

        for simulation in self.simulations(args):
            if only_missing and simulated(simulation, args):
                print(f"{simulation.results}: simulated already")
                continue
            if not simulation.scene.exists():
                raise SystemExit(
                    f"error: no {simulation.scene}; run first: {self.first_step(args)}"
                )
            hrtf = getattr(args, "hrtf", None)
            print(simulate_command(simulation, hrtf))
            code = cli(simulate_argv(simulation, hrtf))
            if code:
                raise SystemExit(code)

    def scene_step(self) -> Step | None:
        """The step that writes the demo's scenes (:attr:`scene_from`), or None."""
        return self.step(self.scene_from) if self.scene_from else None

    def first_step(self, args: argparse.Namespace | None = None) -> str:
        """The command that writes the scenes, or the first step of the demo."""
        step = self.scene_step() or next(s for s in self.steps if s is not SIMULATE)
        return self.command(step.name, args)

    def is_done(self, step: Step, args: argparse.Namespace) -> bool:
        """Whether ``all`` may leave ``step`` out: its result is there already."""
        from acoustic_spaces.demos.pipeline import simulated

        if step is SIMULATE:
            return all(simulated(s, args) for s in self.simulations(args))
        return step.done is not None and step.done(args)

    def run(self, names: Sequence[str], args: argparse.Namespace, skip_done: bool = False) -> None:
        """Run the steps ``names`` in order with ``args`` (paths counted from the demo's
        folder); ``skip_done`` leaves out the steps whose result is there."""
        problem = self.check(args)
        if problem:
            raise SystemExit(f"error: {problem}")
        args = self.resolved(args)
        for name in names:
            step = self.step(name)
            if skip_done and self.is_done(step, args):
                print(f"{name}: done already")
                continue
            if step is SIMULATE:
                self.simulate(args, only_missing=skip_done)
            else:
                step.run(args)

    def main(self, argv: Sequence[str] | None = None) -> int:
        """Run the steps a command line asks for. A demo of the examples counts relative
        paths from its script's folder, where its files are; one run as a module
        (:attr:`module`, a demo of any scene) from the folder it is run in."""
        demo = self if self.module is None or self._folder else self.at(Path.cwd())
        parser = demo.parser()
        args = parser.parse_args(argv)
        problem = demo.check(args)
        if problem:
            parser.error(problem)
        names = [s.name for s in demo.steps] if args.step == "all" else [args.step]
        demo.run(names, args, skip_done=args.step == "all")
        return 0


def _relative(path: Path, folder: Path) -> str:
    """``path`` relative to ``folder`` when it lies under it, else as it is."""
    try:
        return Path(path).resolve().relative_to(Path(folder).resolve()).as_posix()
    except ValueError:
        return str(path)


def sibling(script: str | Path, name: str) -> ModuleType:
    """The module ``name``.py next to ``script`` (a demo's pictures), imported by its
    path, so that a script runs from any folder and the editor can load it."""
    script = Path(script).resolve()
    key = f"acoustic_spaces_demos.{script.parent.name}.{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, script.parent / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"no {name}.py next to {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def load(script: str | Path) -> Demo:
    """The :class:`Demo` a script describes (its ``DEMO``)."""
    script = Path(script).resolve()
    module = sibling(script, script.stem)
    demo = getattr(module, "DEMO", None)
    if not isinstance(demo, Demo):
        raise ValueError(f"{script} describes no demo (no DEMO)")
    return demo


def list_demos() -> list[Demo]:
    """Every demo that comes with the examples (``examples/*/*_demo.py``)."""
    found = []
    for script in sorted(examples_dir().glob("*/*_demo.py")):
        found.append(load(script))
    return found


def demo_for(example: str) -> Demo | None:
    """The demo of the example ``example`` (a folder name), or None."""
    for script in sorted((examples_dir() / example).glob("*_demo.py")):
        return load(script)
    return None


__all__ = [
    "KINDS",
    "MAP_KINDS",
    "PATH_KINDS",
    "SIMULATE",
    "Demo",
    "Option",
    "Step",
    "common_options",
    "input_option",
    "demo_for",
    "list_demos",
    "load",
    "shell_line",
    "sibling",
]
