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
- :mod:`acoustic_spaces.demos.passby` and :mod:`~acoustic_spaces.demos.passby_pictures`:
  the pass-by demos of the roadside and the street, a car driving past with Doppler,
  and the flyover demo; :mod:`acoustic_spaces.demos.car`, the car's sound power
  (CNOSSOS-EU) and a sound made to it; :mod:`acoustic_spaces.demos.helicopter`, the
  helicopter that flies over and its rotors' sound.
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
from acoustic_spaces.i18n import N_

#: The kinds of option, for the parser and for the editor's form: a free text, a choice
#: from a list, a number, a flag, a file to read, a file the demo writes (``path``), a
#: folder, ``inputs``: a WAV file for each of some sources (``--input NAME=WAV``, as
#: often as needed), what they play instead of the demo's own recordings, ``files``: a
#: WAV file for each of some names the option lists as its choices (``--stem
#: NAME=WAV``), such as the stems of a song, and ``movers``: the sources and listeners
#: that move in a run, by name and comma-separated, or ``none`` (:func:`moving_names`).
KINDS = (
    "text",
    "choice",
    "int",
    "float",
    "flag",
    "file",
    "path",
    "folder",
    "inputs",
    "files",
    "movers",
)
PATH_KINDS = ("file", "path", "folder")
#: The kinds whose value is a dict of names and files.
MAP_KINDS = ("inputs", "files")


#: What a ``movers`` option takes for nothing moving.
NO_MOVERS = "none"

#: The editor's label of each option, in words, by its flag (the Run demo window gives
#: the flag in the label's tip); an option not named here shows its flag.
LABELS = {
    "--scene": N_("Scene"),
    "--seconds": N_("Time per listener"),
    "--tour": N_("Tour"),
    "--credit": N_("Credit"),
    "--work": N_("Work folder"),
    "--input": N_("What the sources play"),
    "--format": N_("Format"),
    "--hrtf": N_("HRTFs"),
    "--again": N_("Mix again"),
    "--no-mp4": N_("No video"),
    "--from": N_("Example's scene"),
    "--results": N_("Results folder"),
    "--listeners": N_("Listeners"),
    "--talker": N_("Talker"),
    "--voice": N_("Voice"),
    "--stand": N_("Where the visitor stands"),
    "--turn": N_("Head turns"),
    "--speed": N_("Speed"),
    "--broadband": N_("Broadband noise"),
    "--step": N_("Places apart"),
    "--song": N_("Song"),
    "--music": N_("Your own song"),
    "--bpm": N_("Tempo"),
    "--still": N_("Scene in still air"),
    "--still-results": N_("Results in still air"),
    "--last-phrases": N_("Last stop's phrases"),
    "--zips": N_("Downloads folder"),
    "--movements": N_("Movements"),
    "--rpm": N_("Engine speed"),
    "--flight": N_("Flight's scene"),
    "--stadium": N_("Stadium's scene"),
    "--height": N_("Height"),
    "--at": N_("Overhead at"),
    "--band": N_("Band's scene"),
    "--downbeat": N_("First downbeat"),
    "--stem": N_("Stems"),
    "--rays": N_("Rays per band"),
    "--out": N_("Written to"),
}
#: The options the editor keeps under Advanced besides the files and folders a demo
#: writes, the example's scenes it reads and the flags, by flag: what fine-tunes a run
#: rather than what it plays.
ADVANCED = frozenset(
    {
        "--scene",
        "--credit",
        "--tour",
        "--listeners",
        "--movements",
        "--bpm",
        "--downbeat",
        "--last-phrases",
        "--step",
        "--stand",
        "--turn",
    }
)


def moving_names(text: str | None) -> list[str] | None:
    """The names a ``movers`` option gives (``--moving visitor,guide``): None when it is
    not given, so that the scene's own walks decide, and none for ``none``."""
    if text is None:
        return None
    if str(text).strip().casefold() == NO_MOVERS:
        return []
    return [name.strip() for name in str(text).split(",") if name.strip()]


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
    #: A number's unit, shown after it in the editor ("km/h").
    unit: str = ""
    #: What a number without a default stands for when not given ("as in the scene"):
    #: the editor's field shows it at its lowest, zero.
    unset: str = ""
    #: Given without its flag, before the others (an example script's scene).
    positional: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: no option kind {self.kind!r}")
        if self.kind == "files" and not self.choices:
            raise ValueError(f"{self.name}: a files option lists the names it takes as choices")

    @property
    def option(self) -> str:
        return self.flag or "--" + self.name.replace("_", "-")

    @property
    def label(self) -> str:
        """The editor's label, in English (:data:`LABELS`), or the flag."""
        return LABELS.get(self.option, self.option)

    @property
    def advanced(self) -> bool:
        """Whether the editor keeps it under Advanced: a file or a folder the demo
        writes, a scene of the example's it reads, a flag, or one of :data:`ADVANCED`."""
        return (
            self.kind in ("path", "folder", "flag")
            or (self.kind == "file" and self.default is not None)
            or self.option in ADVANCED
        )

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
        if self.positional:
            kind = Path if self.kind in PATH_KINDS else None
            parser.add_argument(
                self.name, nargs="?", default=self.default, type=kind, help=self.description
            )
            return
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
    #: What the step downloads with the arguments, in words and with its size (the Run
    #: demo window says it before a run); None when it downloads nothing.
    downloads: Callable[[argparse.Namespace], str | None] | None = None


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
    #: The sources and listeners the demo plays moving along their walks, by name, from
    #: the arguments (paths resolved); None for a demo that plays every one standing where
    #: it starts. The editor's Run demo window ticks them in its table of what moves, which
    #: a ``movers`` option lets the user change for a run.
    moving: Callable[[argparse.Namespace], list[str]] | None = None
    #: What a run with the arguments (paths resolved) does, in a line under the editor's
    #: form that follows the options (how far and how fast the pass-by demos' car drives,
    #: and so for how long); None to say nothing.
    summary: Callable[[argparse.Namespace], str | None] | None = None
    #: The demo of the same example whose work this one plays (``stadium``, for the
    #: helicopter over its concert), to run first in the same work folder; None for the
    #: example's own demo, the one :func:`demo_for` gives.
    builds_on: str | None = None
    #: Whether it is a script of one step, run without naming it (an example's script
    #: that is no demo, :func:`script_demo`).
    stepless: bool = False
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
        lead: list[str] = []
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
                continue
            if names_only and option.kind in PATH_KINDS:
                shown = Path(value).name
            elif relative_to is not None and option.kind in PATH_KINDS:
                shown = _relative(Path(value), relative_to)
            else:
                shown = str(value)
            if option.positional:
                lead.append(shown)
            else:
                argv += [option.option, shown]
        return lead + argv

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
            named = "" if self.stepless else f" {step}"
            return f"python {self.script.name}{named} {shell_line(argv)}".rstrip()
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
        named = [] if self.stepless else [step]
        return [sys.executable, *start, *named, *self.arguments(values)]

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


def demos_of(example: str) -> list[Demo]:
    """Every demo of the example ``example`` (a folder name): its own demo first, then
    those that build on it (:attr:`Demo.builds_on`)."""
    folder = examples_dir() / example
    found = [load(script) for script in sorted(folder.glob("*_demo.py"))]
    return sorted(found, key=lambda demo: demo.builds_on is not None)


def demo_for(example: str) -> Demo | None:
    """The demo of the example ``example`` (a folder name), or None: its own, when it
    has more than one (:func:`demos_of`)."""
    found = demos_of(example)
    return found[0] if found else None


#: The step of an example's script run as a demo: the script itself.
RUN = "run"


def _option_of(action: argparse.Action, folder: Path) -> Option:
    """The :class:`Option` of a script's argument: a number, a flag, a choice, a path
    (read when the example has the file its default names, else written) or a text."""
    default = action.default
    if isinstance(action, argparse._StoreTrueAction):
        kind = "flag"
    elif action.choices:
        kind = "choice"
    elif action.type in (int, float):
        kind = action.type.__name__
    elif action.type is Path or isinstance(default, Path):
        if default is not None:
            default = Path(default)
            if default.is_absolute() and default.parent == folder:
                default = Path(default.name)  # the example's own file, by its name
        kind = "file" if default is not None and (folder / default).is_file() else "path"
    else:
        kind = "text"
    flag = next((s for s in action.option_strings if s.startswith("--")), None)
    return Option(
        action.dest,
        action.help or "",
        default,
        kind,
        tuple(action.choices or ()),
        flag=flag,
        filter="Scene files (*.yaml *.yml)" if action.dest == "scene" else "",
        positional=not action.option_strings,
    )


def script_demo(script: str | Path) -> Demo:
    """An example's script that is no demo (``feedback.py``) as a demo of one step the
    editor runs: its ``TITLE``, its docstring, and its options from its own
    ``parser()``, the scene it reads given first without a flag; run as ``python
    feedback.py [SCENE] [--rays N]``."""
    script = Path(script).resolve()
    module = sibling(script, script.stem)
    parser = module.parser()
    options = [
        _option_of(action, script.parent)
        for action in parser._actions
        if not isinstance(action, argparse._HelpAction)
    ]
    return Demo(
        name=script.stem,
        example=script.parent.name,
        title=module.TITLE,
        description=module.__doc__ or "",
        script=script,
        steps=[Step(RUN, None, "run the script")],
        options=options,
        stepless=True,
    )


def scripts_of(example: str) -> list[Demo]:
    """The scripts of the example ``example`` (a folder name) that the editor runs as
    demos (:func:`script_demo`): those with a ``parser()``, which are neither its demos
    nor what writes its model."""
    folder = examples_dir() / example
    found = []
    for script in sorted(folder.glob("*.py")):
        stem = script.stem
        if stem.endswith(("_demo", "_pictures")) or stem.startswith("make_"):
            continue
        if "\ndef parser(" in script.read_text(encoding="utf-8"):
            found.append(script_demo(script))
    return found


__all__ = [
    "ADVANCED",
    "KINDS",
    "LABELS",
    "MAP_KINDS",
    "NO_MOVERS",
    "PATH_KINDS",
    "RUN",
    "SIMULATE",
    "Demo",
    "Option",
    "Step",
    "common_options",
    "input_option",
    "demo_for",
    "demos_of",
    "list_demos",
    "load",
    "moving_names",
    "script_demo",
    "scripts_of",
    "shell_line",
    "sibling",
]
