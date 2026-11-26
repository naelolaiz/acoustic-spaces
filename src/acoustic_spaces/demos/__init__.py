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
  the pass-by demos of the roadside and the street, a car, a truck, a bus, a motorcycle
  or a police car with its siren driving past with Doppler, and the flyover demo;
  :mod:`acoustic_spaces.demos.road`, the road vehicles' sound power (CNOSSOS-EU) and a
  sound made to it; :mod:`acoustic_spaces.demos.siren`, the siren's sound, level and
  directivity; :mod:`acoustic_spaces.demos.helicopter`, the helicopter that flies over
  and its rotors' sound.
- :mod:`acoustic_spaces.fetch`: downloads, whole or one member of a remote zip archive.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from acoustic_spaces.examples import examples_dir
from acoustic_spaces.i18n import N_

if TYPE_CHECKING:
    from acoustic_spaces.scene import Scene

#: The kinds of option, for the parser and for the editor's form: a free text, a choice
#: from a list, a number, a flag, a file to read, a file the demo writes (``path``), a
#: folder, ``inputs``: a WAV file for each of some sources (``--input NAME=WAV``, as
#: often as needed), what they play instead of the demo's own recordings, ``files``: a
#: WAV file for each of some names the option lists as its choices (``--stem
#: NAME=WAV``), such as the stems of a song, ``movers``: the sources and listeners
#: that move in a run, by name and comma-separated, or ``none`` (:func:`moving_names`),
#: ``listeners``: the listeners a run plays, by name and comma-separated
#: (:func:`listener_names`), which the editor ticks in a list, and ``sources``: the
#: sources a run plays, by name or by their scene's group, comma-separated
#: (:attr:`Demo.sources`), ticked alike.
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
    "listeners",
    "sources",
)
PATH_KINDS = ("file", "path", "folder")
#: The kinds whose value is a dict of names and files.
MAP_KINDS = ("inputs", "files")
#: How the line begins that a demo prints once every step it was asked for has run
#: (:meth:`Demo.run`; for an example's script that is no demo,
#: :mod:`acoustic_spaces.demos.run_script` once it has ended well): from it the editor's
#: Run demo knows that what the demo writes is complete, whatever its process does next.
DONE = "every step has run"


#: What a ``movers`` option takes for nothing moving.
NO_MOVERS = "none"

#: How an option counts in the name of a run (:attr:`Option.named`): ``plays``, it picks
#: what plays (``--vehicle``, ``--song``; named always, at its default too, or for a file,
#: when one is given); ``changed``, it changes the sound or the pictures (named when it
#: differs from its default, as ``speed-80``); ``never``, it does neither (``--credit``,
#: ``--again``, a folder the demo writes into).
NAMINGS = ("plays", "changed", "never")
#: The longest a run's name gets, in characters: a longer one keeps its first part (the
#: demo) and its last (the output) and gives the rest cut short, with a digest of it.
RUN_NAME_MOST = 96
#: The longest the listeners of a run (``--listeners``) get in its name; more are named by
#: their number and a digest of their names (``12-listeners-3f9a1c``).
LISTENERS_MOST = 48
#: The longest a text (``--tour``, ``--movements``) gets in a run's name; a longer one is
#: named by a digest of it (``tour-3f9a1c``).
TEXT_MOST = 24
#: What a series takes in turn besides the options given several values (``--series``,
#: :meth:`Demo.parts`): the picked sources one at a time (``--sources-play in-turn``) and
#: the listeners one part each (only when ``--series`` names them; otherwise the tour of
#: each part visits them).
SOURCES_DIM = "sources"
LISTENERS_DIM = "listeners"
#: How the picked sources play (``--sources-play``): all in one mix, or each alone, one
#: after another in the run's video.
TOGETHER = "together"
IN_TURN = "in-turn"
SOURCES_PLAY = (TOGETHER, IN_TURN)
#: What :meth:`Demo.resolved` adds to the arguments, which a run's parts and its
#: siblings work out again: the run's folder, what it plays (in its mixes' names), its
#: parts (the arguments of each, for a run of several) and the sources that play.
RESOLVED_ONLY = ("run_folder", "run_plays", "series_parts", "playing")
#: The file each run's folder keeps of it (:func:`record_run`): the demo, the run's
#: name and its parts, the command, and once the video is there, its file, length and
#: chapters.
RUN_RECORD = "run.json"


def digest(text: str) -> str:
    """Six hexadecimal digits that stand for ``text`` in a name (of SHA-1)."""
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:6]


def name_part(text: str) -> str:
    """``text`` as part of a file's name: letters (of any script), digits, ``_``, ``-`` and
    ``+`` as they are, a point between digits as ``p`` (``80.5`` as ``80p5``, so that a
    folder named so has no extension), anything else as ``-``; a digest of it when
    nothing is left."""
    out = re.sub(r"(?<=\d)\.(?=\d)", "p", str(text))
    out = re.sub(r"[^\w+-]+", "-", out).strip("-")
    return out or digest(text)


def values_of(value: Any) -> tuple:
    """The values an option holds: several (a tuple, :attr:`Option.varies`) or one."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def given_names(text: str | None) -> list[str] | None:
    """The names a comma-separated option gives, in its order, each once: None when it
    is not given."""
    if text is None:
        return None
    if isinstance(text, (tuple, list)):
        text = ",".join(str(t) for t in text)
    names = [name.strip() for name in str(text).split(",") if name.strip()]
    return list(dict.fromkeys(names))


#: The editor's label of each option, in words, by its flag (the Run demo window gives
#: the flag in the label's tip); an option not named here shows its flag.
LABELS = {
    "--scene": N_("Scene"),
    "--recording": N_("Recording"),
    "--vocal-effort": N_("Vocal effort"),
    "--seconds": N_("Time per listener"),
    "--tour": N_("Tour"),
    "--atmosphere": N_("Atmosphere"),
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
    "--stand": N_("Where a listener stands"),
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
    "--last-stop": N_("Last stop"),
    "--zips": N_("Downloads folder"),
    "--movements": N_("Movements"),
    "--rpm": N_("Engine speed"),
    "--blows": N_("Blows a minute"),
    "--flight": N_("Flight's scene"),
    "--stadium": N_("Stadium's scene"),
    "--height": N_("Height"),
    "--at": N_("Overhead at"),
    "--band": N_("Band's scene"),
    "--downbeat": N_("First downbeat"),
    "--stem": N_("Stems"),
    "--rays": N_("Rays per band"),
    "--out": N_("Written to"),
    "--vehicle": N_("Vehicle"),
    "--train": N_("Train"),
    "--track": N_("Track"),
    "--siren": N_("Siren"),
    "--horn": N_("Horn"),
    "--indoors": N_("Heard indoors"),
    "--sources": N_("Sources"),
    "--sources-play": N_("Sources play"),
    "--series": N_("Order of the parts"),
}
#: The options the editor keeps under Advanced besides the files and folders a demo
#: writes, the example's scenes it reads and the flags, by flag: what fine-tunes a run
#: rather than what it plays.
ADVANCED = frozenset(
    {
        "--scene",
        "--credit",
        "--tour",
        "--movements",
        "--bpm",
        "--downbeat",
        "--last-phrases",
        "--last-stop",
        "--step",
        "--stand",
        "--turn",
    }
)


def moving_names(text: str | None) -> list[str] | None:
    """The names a ``movers`` option gives (``--moving visitor,guide``): None when it is
    not given, so that the scene's own trajectories decide, and none for ``none``."""
    if text is None:
        return None
    if str(text).strip().casefold() == NO_MOVERS:
        return []
    return [name.strip() for name in str(text).split(",") if name.strip()]


def listener_names(text: str | None) -> list[str] | None:
    """The listeners a ``listeners`` option names (``--listeners front,back``), in its
    order, each once: None when it is not given, so that the demo plays every one."""
    return given_names(text)


def source_matches(source: Source, name: str) -> bool:
    """Whether ``--sources`` picks ``source`` by ``name``: its own, its group's
    (``PA/Mains``) or that of a group its group is in (``PA``)."""
    group = source.group or ""
    return name == source.name or bool(group) and (group == name or group.startswith(name + "/"))


def playing(args: argparse.Namespace, names: Sequence[str]) -> list[str]:
    """``names`` (the sources a step could play, in its order) without those the run
    leaves out (``--sources``, :meth:`Demo.picked_sources`): all of them for a demo with
    no choice of its sources."""
    chosen = getattr(args, "playing", None)
    if chosen is None:
        return list(names)
    return [name for name in names if name in chosen]


def picked(args: argparse.Namespace, names: Sequence[str]) -> list[str]:
    """``names`` (the listeners a demo plays, in its order) without those ``--listeners``
    leaves out: all of them when it is not given."""
    chosen = listener_names(getattr(args, "listeners", None))
    if chosen is None:
        return list(names)
    return [name for name in names if name in chosen]


def chosen_order(args: argparse.Namespace, names: Sequence[str]) -> list[str]:
    """``names`` (the listeners a demo can play) in the order ``--listeners`` gives
    them, the others left out; all of them in their own order when it is not given."""
    chosen = listener_names(getattr(args, "listeners", None))
    if chosen is None:
        return list(names)
    return [name for name in chosen if name in names]


def listeners_in(scene: str | Path) -> list[str]:
    """The listeners in use (not ``enabled: false``) of the scene file ``scene``, in its
    order, read as the file has them (no check of the scene): what a demo playing that
    scene can play."""
    path = Path(scene)
    stat = path.stat()
    return list(_listeners_in(str(path.resolve()), stat.st_mtime_ns, stat.st_size))


@functools.lru_cache(maxsize=16)
def _listeners_in(path: str, _mtime_ns: int, _size: int) -> tuple[str, ...]:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return tuple(
        str(item["name"])
        for item in data.get("listeners") or []
        if isinstance(item, dict) and "name" in item and item.get("enabled", True) is not False
    )


LISTENERS_HELP = (
    "only these listeners, comma separated (default: every one): the demo mixes those "
    "alone and its video's tour leaves the others out; the simulation keeps every "
    "listener, so another choice needs no new one"
)


def listeners_option(steps: tuple[str, ...] = ("render", "video"), help: str = "") -> Option:
    """``--listeners a,b``: the listeners a run plays, for the steps that mix and tour
    them; the editor ticks them in a list of the listeners the demo can play
    (:attr:`Demo.listeners`)."""
    return Option("listeners", help or LISTENERS_HELP, None, "listeners", steps=steps)


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
    #: the editor's field shows it while empty.
    unset: str = ""
    #: Given without its flag, before the others (an example script's scene).
    positional: bool = False
    #: Another option and the values of it this one matters with, as ``("vehicle",
    #: ("police_car",))`` for the police car's siren: the editor shows its field only
    #: while that option holds one of them, and leaves it at its default otherwise.
    #: None: it always matters.
    only: tuple[str, tuple[str, ...]] | None = None
    #: Other flags it is given by, as before it took this one (``--tour``, for the demo
    #: of any scene's ``--listeners``).
    aliases: tuple[str, ...] = ()
    #: Flags an example's script gave it by before the script became a preset of the
    #: demo of any scene, with values of another form: each flag, and what a value of
    #: it is in this option's (``("--talker", lambda code: f"ace_{code.lower()}")`` for
    #: ``--recording``), each of several for an option that :attr:`varies`. Its own
    #: flag reads so too (the gallery's ``--stand 16,4``, for ``--stand
    #: standing=16,4``).
    renamed: tuple[tuple[str, Callable[[str], str]], ...] = ()
    #: How it counts in the name of a run (:meth:`Demo.run_name`), one of
    #: :data:`NAMINGS`; None: as its kind has it (:attr:`naming`).
    named: str | None = None
    #: For a file the demo writes (``path``) that depends on the run's settings (a scene
    #: with the vehicle at ``--speed``): its name in the run's folder
    #: (:func:`run_folder`), where it is written when the option is not given.
    in_run: str | None = None
    #: Another option that, given, plays in this one's place (``music`` for ``--song``:
    #: a song of your own in place of the preset): the run's name leaves this one out
    #: then.
    replaced_by: str | None = None
    #: Whether it takes several values, comma separated, each played in a part of its
    #: own, one after another in one video (:meth:`Demo.parts`); None: as its kind has
    #: it (:attr:`varies`).
    several: bool | None = None
    #: Why it changes nothing in a run with the arguments given (paths resolved, one of
    #: everything), as the check says it, or None where it changes something:
    #: ``--indoors`` at listeners none of whom stands in front of a house. A part of a
    #: series (:meth:`Demo.parts`) where it changes nothing takes its default, as the
    #: run of one of everything would hear it; a run where it changes nothing in any
    #: part is refused (:meth:`Demo.check`). None: it always changes something.
    moot: Callable[[argparse.Namespace], str | None] | None = None
    #: For a choice: whether it also takes a file of yours in place of one of its
    #: choices (``--recording song.flac``), counted from the demo's folder as a file
    #: option is.
    or_file: bool = False

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.name}: no option kind {self.kind!r}")
        if self.named is not None and self.named not in NAMINGS:
            raise ValueError(f"{self.name}: no naming {self.named!r}")
        if self.in_run is not None and (self.kind != "path" or self.default is not None):
            raise ValueError(f"{self.name}: only a path with no default is written in the run")
        if self.kind == "files" and not self.choices:
            raise ValueError(f"{self.name}: a files option lists the names it takes as choices")
        if self.several and self.kind not in ("choice", "int", "float", "text"):
            raise ValueError(f"{self.name}: only a choice, a number or a text takes several")
        if self.or_file and self.kind != "choice":
            raise ValueError(f"{self.name}: only a choice takes a file in place of one")

    @property
    def option(self) -> str:
        return self.flag or "--" + self.name.replace("_", "-")

    @property
    def naming(self) -> str:
        """How it counts in the name of a run (:data:`NAMINGS`): as :attr:`named` says,
        else ``plays`` for the files some sources play (``--input``, ``--stem``),
        ``never`` for a file or folder the demo writes, the listeners and the runs (named
        their own way), the format and the HRTFs (in the run's label) and ``--credit`` (a
        line of words on the stills), and ``changed`` for the rest."""
        if self.named is not None:
            return self.named
        if self.kind in MAP_KINDS:
            return "plays"
        if self.kind in ("path", "folder", "listeners", "sources") or self.positional:
            return "never"
        if self.name in ("format", "hrtf", "credit"):
            return "never"
        return "changed"

    @property
    def varies(self) -> bool:
        """Whether it takes several values, each played in a part of its own
        (:attr:`several`): by default a choice or a number that counts in the run's name
        (what plays, or what changes the sound or the pictures), and the format."""
        if self.several is not None:
            return self.several
        if self.positional:
            return False
        if self.name == "format":
            return True
        return self.kind in ("choice", "int", "float") and self.naming != "never"

    def parse(self, text: Any) -> Any:
        """Its value from the command line: one, or for an option that :attr:`varies`,
        several, comma separated, each once, as a tuple; each a number for a number, one
        of its choices for a choice."""
        items = given_names(text) if self.varies else [str(text)]
        if not items:
            raise argparse.ArgumentTypeError("no value")
        out: list[Any] = []
        for item in items:
            try:
                value: Any = (
                    int(item)
                    if self.kind == "int"
                    else float(item)
                    if self.kind == "float"
                    else item
                )
            except ValueError:
                raise argparse.ArgumentTypeError(f"not a number: {item!r}") from None
            if self.choices and value not in self.choices and not self.or_file:
                raise argparse.ArgumentTypeError(
                    f"invalid choice: {item!r} (choose from {', '.join(self.choices)})"
                )
            if value not in out:
                out.append(value)
        return out[0] if len(out) == 1 else tuple(out)

    def applies(self, values: dict[str, Any]) -> bool:
        """Whether it matters with the options at ``values`` (by name): always, unless
        it matters only with some values of another (:attr:`only`), with any of them
        when that one holds several."""
        if self.only is None:
            return True
        name, among = self.only
        return any(value in among for value in values_of(values.get(name)))

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
        if self.varies:
            text += "; several, comma separated, play one after another in one video"
        if self.default is not None and self.kind != "flag":
            text += f" (default {self.default})"
        elif self.in_run is not None:
            text += f" (default {self.in_run} in the run's folder)"
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
        elif self.varies:
            kwargs["type"] = self.parse
            if self.choices:
                files = "|FILE" if self.or_file else ""
                kwargs["metavar"] = "{" + ",".join(self.choices) + "}" + files + "[,...]"
        elif self.kind == "int":
            kwargs["type"] = int
        elif self.kind == "float":
            kwargs["type"] = float
        if self.choices and not self.varies and not self.or_file:
            kwargs["choices"] = list(self.choices)
        own = (self.option, *self.aliases)
        for flag, form in self.renamed:
            if flag in own:  # its own flag, in the old form or its own
                kwargs["type"] = self._renamed(form, kwargs.get("type"))
        parser.add_argument(self.option, *self.aliases, **kwargs)
        for flag, form in self.renamed:
            if flag in own:
                continue
            parser.add_argument(
                flag,
                dest=self.name,
                default=argparse.SUPPRESS,
                type=self._renamed(form),
                help=f"{self.option} as the example's script took it before",
            )

    def _renamed(
        self, form: Callable[[str], str], then: Callable[[str], Any] | None = None
    ) -> Callable[[str], Any]:
        """How a value of a :attr:`renamed` flag reads: in this option's form
        (``form``), then as one of its own (:meth:`parse`, or ``then``)."""

        def read(text: str) -> Any:
            if self.varies:
                return self.parse(",".join(form(item) for item in given_names(text) or []))
            return form(text) if then is None else then(form(text))

        return read


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
    #: The names steps that it took over had in an example's script before the script
    #: became a preset of the demo of any scene (``walk`` and ``turn``, for ``render``):
    #: a command line that names one runs it.
    aliases: tuple[str, ...] = ()


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


def simulates(step: Step) -> bool:
    """Whether ``step`` is the framework's simulation (:data:`SIMULATE`, under its own
    name or with :attr:`Step.aliases` of a preset's)."""
    return step.run is None and step.name == SIMULATE.name


@dataclass(frozen=True)
class Source:
    """A source a demo can play (:attr:`Demo.sources`): its name, its group in the scene
    (``PA/Mains``; ``--sources PA`` picks every source of the groups under ``PA``), and
    whether a run plays it when ``--sources`` names none."""

    name: str
    group: str | None = None
    on: bool = True


#: What ``--sources`` says of a demo that simulates every source once and mixes those
#: picked (:func:`sources_options`).
SOURCES_HELP = (
    "only these sources, by name or by the scene's group (PA takes PA/Mains and "
    "PA/Fills), comma separated (default: the demo's own pick); the simulation keeps "
    "every source, so another pick needs no new one"
)


def sources_options(
    play: str = TOGETHER, help: str = SOURCES_HELP, steps: tuple[str, ...] = ()
) -> list[Option]:
    """``--sources a,b``: the sources a run plays, by name or by group, the editor ticking
    them in a list of the demo's (:attr:`Demo.sources`), with ``help`` for the steps
    ``steps``; and ``--sources-play``: whether they play together, in one mix, or one
    after another, each alone, in a part of its own of the run's video (``play`` by
    default)."""
    return [
        Option("sources", help, None, "sources", steps=steps),
        Option(
            "sources_play",
            f"{TOGETHER}: the picked sources in one mix; {IN_TURN}: each name --sources "
            "gives (a group as one), or without it each top group of the demo's own pick, "
            "alone, one after another in one video, each after a title card (a run with one "
            "of them plays as one together)",
            play,
            "choice",
            SOURCES_PLAY,
            named="never",
        ),
    ]


def series_option(
    several: Sequence[str] = ("format",), sources: bool = False, listeners: bool = False
) -> Option:
    """``--series vehicle,listeners``: the order of a run's parts, when it plays several
    of something (:meth:`Demo.parts`), its help naming what of it a demo has: the
    options that take several values by their flags (``several``), the sources played
    in turn and the listeners."""
    named = [f"an option given several values ({', '.join(several)})"] if several else []
    if sources:
        named.append(f"{SOURCES_DIM} (with --sources-play {IN_TURN})")
    if listeners:
        named.append(
            f"{LISTENERS_DIM}, each listener a part of its own (by default the tour of each "
            "part visits them)"
        )
    return Option(
        "series",
        "the order of the parts when the run plays several of something, outermost first, "
        f"comma separated: {', '.join(named) or 'nothing here'}; the rest follow in the "
        "order of the options",
        None,
        named="never",
    )


def run_folder(args: argparse.Namespace) -> Path:
    """The folder of a run with ``args`` (resolved, :meth:`Demo.resolved`): in the work
    folder, named as the run is (:meth:`Demo.run_name`). What depends on the run's
    settings goes there (the mixes, the trajectories, the stills), so that a run with others
    writes over none of it."""
    folder = getattr(args, "run_folder", None)
    if folder is None:
        raise ValueError("the arguments are not resolved by a demo (Demo.resolved)")
    return Path(folder)


def product(args: argparse.Namespace, suffix: str) -> Path:
    """A file of a run to keep or share, in the work folder beside the run's folder and
    named as the run is, then ``suffix``: ``product(args, ".mp4")`` is the video
    (``demo/flight_departure_binaural.mp4``), ``"_thumbnail.png"`` its thumbnail."""
    folder = run_folder(args)
    return folder.parent / f"{folder.name}{suffix}"


def run_plays(args: argparse.Namespace) -> str | None:
    """What a run with ``args`` (resolved) plays, as its name gives it (``arrival``,
    ``violin-my_song``), or None when the demo gives no choice of it: the front of
    each mix's name (``arrival__lateral_450m.binaural.wav``)."""
    return getattr(args, "run_plays", None)


def mix_stem(args: argparse.Namespace, key: str) -> str:
    """The name of a mix file of a run (before ``.<output>.wav``): what plays
    (:func:`run_plays`), then the listener or seat ``key``, as ``arrival__lateral_450m``;
    the key alone where the demo gives no choice of what plays."""
    plays = run_plays(args)
    return f"{plays}__{name_part(key)}" if plays else name_part(key)


def record_run(folder: Path, **fields: Any) -> None:
    """Keep ``fields`` in the run folder's :data:`RUN_RECORD`, beside what it holds."""
    path = Path(folder) / RUN_RECORD
    record = {}
    if path.exists():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
    record.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def common_options(work: str = "demo") -> list[Option]:
    """The options every demo takes: the work folder, the mixes' format and HRTF, ``--again``
    and ``--no-mp4``."""
    return [
        Option("work", "folder for everything the demo writes", Path(work), "folder"),
        Option(
            "format",
            "output format the mixes take (binaural, mono, ambix3, 5.0...); the tour decodes "
            "Ambisonics and loudspeaker mixes to the two ears",
            "binaural",
        ),
        Option(
            "hrtf",
            "binaural mixes through this SOFA file of HRTFs, your own or measured, instead of "
            "a spherical head model (the simulate command takes it too)",
            None,
            "file",
            filter="SOFA files (*.sofa)",
            only=("format", ("binaural",)),
        ),
        Option(
            "again",
            "render: mix the stops again, even the mixes that exist",
            False,
            "flag",
            named="never",
        ),
        Option(
            "no_mp4",
            "video: write the audio and the stills only (and remove the run's older video)",
            False,
            "flag",
            named="never",
        ),
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
    #: The sources and listeners the demo plays moving along their trajectories, by name, from
    #: the arguments (paths resolved); None for a demo that plays every one standing where
    #: it starts. The editor's Run demo window ticks them in its table of what moves, which
    #: a ``movers`` option lets the user change for a run.
    moving: Callable[[argparse.Namespace], list[str]] | None = None
    #: The scene whose trajectories a run simulates its movers along, from the arguments (paths
    #: resolved), when it is not the scene the editor has open: the pass-by demos'
    #: vehicle at --speed in a scene of its own. While the demo runs, the editor draws the
    #: movers where the run simulates them, on these trajectories; None: the open scene's.
    trajectories: Callable[[argparse.Namespace], Scene] | None = None
    #: The listeners a run can play, in the demo's order, from the arguments (paths
    #: resolved): the names its ``listeners`` option (``--listeners``) takes, and the
    #: rows of the editor's list of them, all ticked by default.
    listeners: Callable[[argparse.Namespace], list[str]] | None = None
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
    #: The first part of a run's name (:meth:`run_name`), from the arguments (paths
    #: resolved): the scene file's name for the demo of any scene; None: the demo's
    #: :attr:`name`.
    stem_of: Callable[[argparse.Namespace], str] | None = None
    #: The sources a run can play, in the demo's order, each with its group and whether
    #: it plays by default (:class:`Source`), from the arguments (paths resolved): what
    #: ``--sources`` takes and the rows of the editor's list of them. A demo with them
    #: takes ``--sources`` and ``--sources-play`` (:func:`sources_options`); its steps
    #: play those :func:`playing` gives.
    sources: Callable[[argparse.Namespace], list[Source]] | None = None
    #: How the picked sources play by default (:data:`SOURCES_PLAY`).
    sources_play: str = TOGETHER
    #: How the sources a run plays count in its name (:data:`NAMINGS`): ``plays``, always
    #: (the airport's ``departure``); ``changed``, when they are not the demo's own pick.
    sources_named: str = "changed"
    _folder: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.script = Path(self.script).resolve()
        self.description = " ".join((self.description or "").strip().split("\n\n")[0].split())
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: a step is listed twice")
        if self.sources_play not in SOURCES_PLAY or self.sources_named not in NAMINGS:
            raise ValueError(f"{self.name}: no way for the sources to play or be named so")
        have = {o.name for o in self.options}
        if self.sources is not None and "sources" not in have:
            self.options = [*self.options, *sources_options(self.sources_play)]
        if not self.stepless and "series" not in have:
            several = [o.option.lstrip("-") for o in self.options if o.varies]
            listeners = any(o.kind == "listeners" for o in self.options)
            self.options = [
                *self.options,
                series_option(several, self.sources is not None, listeners),
            ]

    # -- description ----------------------------------------------------------------

    @property
    def folder(self) -> Path:
        """The folder the demo runs in: the script's, unless :meth:`at` moved it."""
        return self._folder or self.script.parent

    def at(self, folder: str | Path) -> Demo:
        """The same demo run in ``folder`` (a copy of the example): relative paths in its
        arguments count from there."""
        return replace(self, _folder=Path(folder))

    @property
    def in_place(self) -> bool:
        """Whether the demo's command line counts relative paths, given or at their
        default, from the folder it runs in: a demo run as a module, which counts them
        from the working folder, or a script in its own folder. Moved elsewhere
        (:meth:`at`), a demo's script counts them from its own folder; an example's
        script that is no demo (:attr:`stepless`) counts them from the working folder,
        where the editor runs it, but its default scene is the one beside it. Either
        way, :meth:`arguments` gives it every path that has a default, in full."""
        return self.module is not None or self.folder == self.script.parent

    def default_path(self, option: Option) -> Path:
        """Where the path ``option`` names at its default: in the folder the demo runs in,
        or for a file it reads that the folder has no copy of, the example's beside the
        script."""
        path = self.folder / option.default
        beside = self.script.parent / option.default
        if option.kind == "file" and not path.exists() and beside.exists():
            return beside
        return path

    def steps_named(self, name: str) -> list[str]:
        """The steps a command line that names ``name`` runs, in order: the step of that
        name, or every step that took over one of that name (:attr:`Step.aliases`): the
        lakeside's ``music``, ``recording`` and ``feeds``."""
        if any(step.name == name for step in self.steps):
            return [name]
        found = [step.name for step in self.steps if name in step.aliases]
        if not found:
            raise KeyError(f"{self.name}: no step {name!r}")
        return found

    def step(self, name: str) -> Step:
        """The step called ``name``, or that took over a step of that name
        (:attr:`Step.aliases`)."""
        for step in self.steps:
            if step.name == name:
                return step
        for step in self.steps:
            if name in step.aliases:
                return step
        raise KeyError(f"{self.name}: no step {name!r}")

    def option(self, name: str) -> Option:
        for option in self.options:
            if option.name == name:
                return option
        raise KeyError(f"{self.name}: no option {name!r}")

    def work_folder(self, args: argparse.Namespace) -> Path:
        """Where a run with ``args`` writes (:attr:`work_of`)."""
        return self._work(self.resolved(args))

    def run_folder(self, args: argparse.Namespace) -> Path:
        """The folder of a run with ``args`` (:func:`run_folder`): in the work folder,
        named after the run's settings (:meth:`run_name`)."""
        return run_folder(self.resolved(args))

    def sibling(self, args: argparse.Namespace, **changes: Any) -> argparse.Namespace:
        """The arguments (resolved) of the run with ``args`` (resolved) but for
        ``changes``, by option name: ``sibling(args, listeners=None)`` is the run of every
        listener with the same settings, whose work a run of some listeners may play
        (a trajectory's passes). The files a run writes in its folder (:attr:`Option.in_run`)
        are that run's."""
        values = vars(args) | changes
        own = getattr(args, "run_folder", None)
        for option in self.options:
            if option.in_run is not None and option.name not in changes and own is not None:
                if values.get(option.name) == Path(own) / option.in_run:
                    values[option.name] = None
        for key in RESOLVED_ONLY:
            values.pop(key, None)
        return self.resolved(argparse.Namespace(**values))

    def upstream(self, args: argparse.Namespace, step: str) -> argparse.Namespace:
        """The arguments (resolved) of the run with ``args`` (resolved) but for the
        options that matter to none of the steps up to ``step`` (:attr:`Option.steps`),
        at their defaults: ``upstream(args, "walk")`` is the run whose trajectory a run with
        another ``--tour`` or ``--at`` may play, as its trajectory is the same."""
        names = [s.name for s in self.steps]
        before = set(names[: names.index(step) + 1])
        # The sources a run plays name it apart (:meth:`run_parts`), not by their naming;
        # how they play goes with them.
        later = {
            option.name: option.default
            for option in self.options
            if option.steps
            and not before & set(option.steps)
            and (option.naming != "never" or option.name == SOURCES_DIM)
        }
        play = next((o for o in self.options if o.name == "sources_play"), None)
        if SOURCES_DIM in later and play is not None:
            later[play.name] = play.default
        return self.sibling(args, **later)

    def _work(self, resolved: argparse.Namespace) -> Path:
        if self.work_of is not None:
            return Path(self.work_of(resolved))
        work = getattr(resolved, "work", None)
        return Path(work) if work else self.folder

    # -- sources and series -------------------------------------------------------------

    def source_list(self, args: argparse.Namespace) -> list[Source]:
        """The sources a run with ``args`` (resolved) can play (:attr:`sources`); none
        when the demo gives no choice of them, or cannot tell yet (no scene)."""
        if self.sources is None:
            return []
        try:
            return list(self.sources(args))
        except (Exception, SystemExit):  # the steps say what keeps them from the scene
            return []

    def own_pick(self, args: argparse.Namespace) -> list[str]:
        """The sources a run with ``args`` (resolved) plays when ``--sources`` names none:
        those the demo has on by default, in its order."""
        return [source.name for source in self.source_list(args) if source.on]

    def picked_sources(
        self, args: argparse.Namespace, names: Sequence[str] | None = None
    ) -> list[str] | None:
        """The sources that play in a run with ``args`` (resolved), in the demo's order:
        those ``names`` (``--sources`` by default) names, by name or by group
        (:func:`source_matches`); the demo's own pick (:meth:`own_pick`) when it names
        none. None for a demo with no choice of sources, or that cannot tell yet."""
        known = self.source_list(args)
        if not known:
            return None
        if names is None:
            names = given_names(getattr(args, "sources", None))
        if names is None:
            return [source.name for source in known if source.on]
        return [s.name for s in known if any(source_matches(s, name) for name in names)]

    def each(self, args: argparse.Namespace) -> list[argparse.Namespace]:
        """The arguments (resolved) of each part of a run with ``args`` (:meth:`parts`),
        in turn: the run's own, when it plays one of everything."""
        resolved = self.resolved(args)
        return list(getattr(resolved, "series_parts", None) or [resolved])

    def _dim_of(self, text: str) -> str:
        """What ``--series`` names by ``text``: an option's name (by its flag without
        dashes, or its name), or :data:`SOURCES_DIM`, :data:`LISTENERS_DIM`."""
        bare = str(text).strip().lstrip("-")
        for option in self.options:
            if bare in (option.option.lstrip("-"), option.name):
                return option.name
        return bare

    def _dim_flag(self, dim: str) -> str:
        """``dim`` as ``--series`` gives it (an option by its flag, without dashes)."""
        for option in self.options:
            if option.name == dim:
                return option.option.lstrip("-")
        return dim

    def _single(self, args: argparse.Namespace) -> argparse.Namespace:
        """The arguments of the first part of a run with ``args``, as the demo's hooks
        take them (paths resolved, what they leave to the scene filled in), without
        naming the run: the first value of each option given several, the sources
        together, no series."""
        values = {
            name: values_of(value)[0] if isinstance(value, tuple) else value
            for name, value in vars(args).items()
            if name not in RESOLVED_ONLY
        }
        values["series"] = None
        if "sources_play" in values:
            values["sources_play"] = TOGETHER
        single = self._paths(argparse.Namespace(**values))
        if self.completes is not None:
            single = self.completes(single)
        return single

    def _turns(self, args: argparse.Namespace) -> list[str]:
        """The sources a run with ``args`` (paths resolved) plays in turn: each name
        ``--sources`` gives (a group plays as one part); without it, each top group of
        the demo's own pick (``PA`` for ``PA/Mains`` and ``PA/Fills``), a source of no
        group by its name."""
        given = given_names(getattr(args, "sources", None))
        if given is not None:
            return given
        single = self._single(args)
        on = [source for source in self.source_list(single) if source.on]
        tops = [source.group.split("/")[0] if source.group else source.name for source in on]
        return list(dict.fromkeys(tops))

    def _listener_turns(self, args: argparse.Namespace) -> list[str]:
        """The listeners of a run with ``args`` (paths resolved), one part each: those
        ``--listeners`` names, in its order, or every one the demo can play."""
        option = next((o for o in self.options if o.kind == "listeners"), None)
        if option is None:
            return []
        given = listener_names(getattr(args, option.name, None))
        if given is not None:
            return given
        if self.listeners is None:
            return []
        try:
            return list(self.listeners(self._single(args)))
        except (Exception, SystemExit):
            return []

    def _combos(self, args: argparse.Namespace, dims: Sequence[str]) -> list[dict[str, Any]]:
        """Every mix of the values of the options ``dims`` in ``args``, the first
        outermost."""
        combos: list[dict[str, Any]] = [{}]
        for dim in dims:
            values = values_of(getattr(args, dim, None))
            combos = [combo | {dim: value} for combo in combos for value in values]
        return combos

    def _options_given_several(self, args: argparse.Namespace) -> list[str]:
        """The options given several values in ``args``, by name, in the options' order."""
        return [
            option.name
            for option in self.options
            if option.varies and len(values_of(getattr(args, option.name, None))) > 1
        ]

    def varying(self, args: argparse.Namespace) -> list[str]:
        """What a run with ``args`` plays several of, in the options' order: each option
        given several values, by name, and :data:`SOURCES_DIM`, where ``--sources``
        stands among them, when its sources play in turn, more than one with any of
        those values (each vehicle its own)."""
        out = self._options_given_several(args)
        if getattr(args, "sources_play", TOGETHER) == IN_TURN and self.sources is not None:
            if any(
                len(self._turns(argparse.Namespace(**(vars(args) | combo)))) > 1
                for combo in self._combos(args, out)
            ):
                names = [o.name for o in self.options]
                at = names.index("sources") if "sources" in names else len(names)
                before = {o.name for o in self.options[:at]}
                out.insert(sum(1 for name in out if name in before), SOURCES_DIM)
        return out

    def series_order(self, args: argparse.Namespace) -> list[str]:
        """The order of the parts of a run with ``args``, outermost first: what
        ``--series`` names of what varies (:meth:`varying`) and :data:`LISTENERS_DIM`
        (with more than one listener), then the rest of what varies in the options'
        order. Empty for a run of one part. The listeners split into parts only when
        ``--series`` names them."""
        varying = self.varying(args)
        named = [self._dim_of(name) for name in given_names(getattr(args, "series", None)) or []]
        split = LISTENERS_DIM in named and len(self._listener_turns(self._single(args))) > 1
        order = [d for d in named if d in varying or (d == LISTENERS_DIM and split)]
        order = list(dict.fromkeys(order))
        return order + [d for d in varying if d not in order]

    def parts(self, args: argparse.Namespace) -> list[argparse.Namespace]:
        """The parts of a run with ``args``, in the order of :meth:`series_order`: one for
        each value of each option given several (``--vehicle departure,arrival``), each
        source in turn (``--sources-play in-turn``: each vehicle's own when it plays
        several, one of none when it plays one), each listener (when ``--series`` names
        them), and every mix of them; each the arguments (paths resolved) of an ordinary
        run of one of everything. An option that does not matter with a part's values
        (:attr:`Option.only`) takes its default there, and parts that come out as the
        same run (an option that changes nothing with the sources picked) are played
        once. One part, the arguments themselves, when the run plays one of everything."""
        base = self._paths(args)
        order = self.series_order(base)
        if not order:
            return [base]
        listeners = next((o for o in self.options if o.kind == "listeners"), None)
        given = [d for d in order if d not in (SOURCES_DIM, LISTENERS_DIM)]
        ranks: dict[str, dict[Any, int]] = {
            dim: {repr(value): k for k, value in enumerate(values_of(getattr(base, dim, None)))}
            for dim in given
        }
        ranks[SOURCES_DIM], ranks[LISTENERS_DIM] = {}, {}
        rows: list[tuple[tuple[int, ...], dict[str, Any]]] = []
        for combo in self._combos(base, given):
            mixes = [combo]
            here = argparse.Namespace(**(vars(base) | combo))
            if SOURCES_DIM in order:
                turns = self._turns(here)
                mixes = [m | {"sources": t, "sources_play": TOGETHER} for m in mixes for t in turns]
                for turn in turns:
                    ranks[SOURCES_DIM].setdefault(repr(turn), len(ranks[SOURCES_DIM]))
            if LISTENERS_DIM in order and listeners is not None:
                names = self._listener_turns(self._single(here))
                mixes = [m | {listeners.name: name} for m in mixes for name in names]
                for name in names:
                    ranks[LISTENERS_DIM].setdefault(repr(name), len(ranks[LISTENERS_DIM]))
            for mix in mixes:
                key = []
                for dim in order:
                    value = mix.get("sources") if dim == SOURCES_DIM else mix.get(dim)
                    if dim == LISTENERS_DIM and listeners is not None:
                        value = mix.get(listeners.name)
                    key.append(ranks[dim].get(repr(value), 0))
                rows.append((tuple(key), mix))
        rows.sort(key=lambda row: row[0])
        out: list[argparse.Namespace] = []
        seen: set[Any] = set()
        for _key, combo in rows:
            values = {k: v for k, v in vars(base).items() if k not in RESOLVED_ONLY} | combo
            values["series"] = None
            for option in self.options:
                if option.only is None or option.applies(values):
                    continue
                if option.name in combo or option.only[0] in combo:
                    values[option.name] = option.default
            part = argparse.Namespace(**values)
            for option in self.options:
                if option.moot is None or values.get(option.name) in (None, option.default):
                    continue
                if option.moot(self._single(part)) is not None:
                    setattr(part, option.name, option.default)
            same = self._same_run(part)
            if same in seen:
                continue
            seen.add(same)
            out.append(part)
        return out

    def _same_run(self, part: argparse.Namespace) -> Any:
        """What tells the run of ``part`` (one of everything) from others: its folder,
        or its values when it has none yet (no scene)."""
        folder = getattr(self.resolved(part), "run_folder", None)
        if folder is not None:
            return str(folder)
        return tuple(sorted((k, repr(v)) for k, v in vars(part).items()))

    def _sources_part(self, args: argparse.Namespace, in_turn: bool) -> str | None:
        """The sources of a run with ``args`` (paths resolved) as part of its name: those
        ``--sources`` names (``departure``, ``PA+Band``), always for a demo whose sources
        are what plays (:attr:`sources_named`), else when they are not the demo's own
        pick, then by how they differ from it where that is shorter
        (``with-train_horn``, :meth:`_pick_change`); ``each-`` in front when they play
        in turn. None for nothing to name."""
        given = given_names(getattr(args, "sources", None))
        names = given
        change = None
        if names is None and self.sources_named == "plays" and not in_turn:
            names = self.own_pick(self._single(args)) or None
        elif names is not None and self.sources_named == "changed" and not in_turn:
            single = self._single(args)
            picked = self.picked_sources(single, names)
            if picked == self.own_pick(single):
                names = None
            elif picked is not None:
                change = self._pick_change(single, picked)
        if names is None:
            return "each-source" if in_turn else None
        text = "+".join(name_part(name) for name in names)
        if change is not None and len(change) < len(text):
            text = change
        if len(text) > LISTENERS_MOST:
            text = f"{len(names)}-sources-{digest(','.join(names))}"
        return f"each-{text}" if in_turn else text

    def _pick_change(self, args: argparse.Namespace, picked: Sequence[str]) -> str:
        """How the sources ``picked`` differ from the demo's own pick, as part of a
        run's name: ``with-`` those added, ``without-`` those left out, each set a group
        where it holds every source of that group (``without-Band``)."""
        known = self.source_list(args)
        own = set(self.own_pick(args))
        added = [s for s in known if s.name in picked and s.name not in own]
        dropped = [s for s in known if s.name in own and s.name not in picked]
        words = []
        for word, these in (("with", added), ("without", dropped)):
            if these:
                words.append(f"{word}-" + "+".join(name_part(n) for n in _by_groups(known, these)))
        return "_".join(words)

    # -- the names of runs --------------------------------------------------------------

    def run_parts(self, args: argparse.Namespace) -> list[tuple[str, str]]:
        """The parts of the name of a run with ``args`` (paths resolved), each as its kind
        and its text: ``demo``, the demo (:attr:`stem_of`); ``plays``, what plays where
        the demo gives a choice of it (:data:`NAMINGS`); ``listeners``, those
        ``--listeners`` names; ``changed``, each option that changes the sound or the
        pictures and differs from its default, as ``speed-80``; ``label``, the output the
        mixes take (``binaural``, ``binaural-kemar``, ``ambix3``)."""
        stem = self.stem_of(args) if self.stem_of is not None else self.name
        parts = [("demo", name_part(stem))]
        values = dict(vars(args))
        series = getattr(args, "series_parts", None)
        if series:  # what several values come to in the parts, those that changed nothing one
            for name, value in list(values.items()):
                if isinstance(value, tuple):
                    held = list(dict.fromkeys(getattr(part, name, None) for part in series))
                    values[name] = held[0] if len(held) == 1 else tuple(held)
        changed: list[tuple[str, str]] = []
        for option in self.options:
            naming = option.naming
            if naming == "never" or not option.applies(values):
                continue
            if option.replaced_by is not None and values.get(option.replaced_by):
                continue
            value = values.get(option.name)
            if naming == "plays":
                if value is None or value == {} or value is False:
                    continue
                parts.append(("plays", self._value_part(option, value)))
            elif not self._at_default(option, value):
                if option.kind == "flag":
                    changed.append(("changed", name_part(option.option.lstrip("-"))))
                else:
                    text = self._value_part(option, value)
                    changed.append(("changed", f"{name_part(option.option.lstrip('-'))}-{text}"))
        order = self.series_order(args) if not self.stepless else []
        if self.sources is not None:
            text = self._sources_part(args, SOURCES_DIM in order)
            if text:
                parts.append(("plays" if self.sources_named == "plays" else "sources", text))
        listeners = next((o for o in self.options if o.kind == "listeners"), None)
        names = None if listeners is None else listener_names(values.get(listeners.name))
        split = LISTENERS_DIM in order
        if names:
            text = "+".join(name_part(name) for name in names)
            if len(text) > LISTENERS_MOST:
                text = f"{len(names)}-listeners-{digest(','.join(names))}"
            parts.append(("listeners", f"each-{text}" if split else text))
        elif split:
            parts.append(("listeners", "each-listener"))
        parts += changed
        if given_names(values.get("series")) and order != self.series_order(
            argparse.Namespace(**(values | {"series": None}))
        ):
            text = "-".join(name_part(self._dim_flag(dim)) for dim in order)
            parts.append(("changed", f"series-{text}"))
        if any(o.name in ("format", "hrtf") for o in self.options):
            parts.append(("label", self._label_part(args)))
        return parts

    def _label_part(self, args: argparse.Namespace) -> str:
        """The output the mixes of a run take, as part of its name (``binaural-kemar``),
        or of each of its parts, joined by ``+`` (``binaural+mono``)."""
        from acoustic_spaces.demos.pipeline import hrtf_label

        labels = []
        for each in getattr(args, "series_parts", None) or [args]:
            for fmt in values_of(getattr(each, "format", None) or "binaural"):
                label = name_part(hrtf_label(str(fmt), getattr(each, "hrtf", None)))
                if label not in labels:
                    labels.append(label)
        return "+".join(labels)

    def run_name(self, args: argparse.Namespace) -> str:
        """The name of a run with ``args`` (paths resolved): its parts
        (:meth:`run_parts`) joined by ``_``, as ``flight_arrival_lateral_450m_binaural``
        or ``passby_truck_speed-80_binaural-kemar``; no longer than
        :data:`RUN_NAME_MOST`. Its folder (:func:`run_folder`) and the files it keeps
        (:func:`product`) are named so, so that runs with other settings write over none
        of each other's files, and a run with the same settings finds what it made."""
        return joined_name([text for _kind, text in self.run_parts(args)])

    def _at_default(self, option: Option, value: Any) -> bool:
        """Whether ``value`` (resolved) is the option's default. A path is at its default
        where :meth:`default_path` puts it, or as the default names it in the working
        folder: the editor starts a demo moved to a copy of its example (:meth:`at`) in
        that copy, with every path in full, and the script, counting from its own
        folder, then names its run as the editor does."""
        if value is None:
            return True
        if isinstance(value, tuple):  # several values: a series
            return False
        if option.kind == "flag":
            return not value
        if option.kind in MAP_KINDS:
            return not value
        if option.kind in PATH_KINDS and option.default is not None:
            try:
                given = Path(value).resolve()
                return given in (
                    self.default_path(option).resolve(),
                    (Path.cwd() / option.default).resolve(),
                )
            except (OSError, ValueError):
                return False
        return value == option.default

    @staticmethod
    def _value_part(option: Option, value: Any) -> str:
        """``value`` of ``option`` as part of a run's name; several, joined by ``+``."""
        if isinstance(value, tuple):
            return "+".join(Demo._value_part(option, each) for each in value)
        if option.kind in MAP_KINDS:
            return "+".join(
                f"{name_part(name)}-{name_part(Path(file).stem)}" for name, file in value.items()
            )
        if option.kind in PATH_KINDS or isinstance(value, Path):  # a file of yours
            return name_part(Path(value).stem)
        if option.kind in ("int", "float") and isinstance(value, (int, float)):
            return name_part(f"{value:g}")
        if option.kind == "movers":
            names = moving_names(value)
            text = "+".join(name_part(n) for n in names) if names else NO_MOVERS
            return text if len(text) <= TEXT_MOST else digest(value)
        if option.kind == "text":
            text = "+".join(name_part(p) for p in str(value).split(",") if p.strip())
            return text if 0 < len(text) <= TEXT_MOST else digest(value)
        return name_part(value)

    def arguments(
        self,
        args: argparse.Namespace | None = None,
        *,
        names_only: bool = False,
        relative_to: Path | None = None,
    ) -> list[str]:
        """The options of ``args`` that differ from their defaults, as a command line
        (``["--hrtf", "me.sofa"]``): what :meth:`command` shows and the editor runs. A
        script moved out of its folder (not :attr:`in_place`, which says where it would
        count its paths from) is given every path that has a default, in full, even one
        at its default or left empty (its :meth:`default_path`).
        ``names_only`` gives a path by its file name, for a caption, and leaves out a
        path named as its default is, wherever it lies, or at its default counted from
        the working folder, where the editor runs a moved script (a folder whose default
        is ``.`` has no name of its own); ``relative_to`` gives a path under that folder
        relative to it."""
        moved = not names_only and not self.in_place
        argv: list[str] = []
        lead: list[str] = []
        own = None if args is None else getattr(args, "run_folder", None)
        for option in self.options:
            value = getattr(args, option.name, option.default) if args is not None else None
            if option.in_run is not None and own is not None and value is not None:
                if Path(value) == Path(own) / option.in_run:  # where the run writes it
                    continue
            given = moved and option.kind in PATH_KINDS and option.default is not None
            if given:  # in full, counted from the folder the demo runs in
                value = self.default_path(option) if value is None else self.folder / value
            elif value is None or value == option.default:
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
            if option.kind in PATH_KINDS and option.default is not None and not given:
                if Path(value) == self.folder / option.default:  # the default, resolved
                    continue
                if names_only and (
                    Path(value).name == Path(option.default).name
                    or Path(value).resolve() == (Path.cwd() / option.default).resolve()
                ):
                    continue
            if option.kind == "flag":
                argv.append(option.option)
                continue
            if names_only and option.kind in PATH_KINDS:
                shown = Path(value).name
            elif relative_to is not None and option.kind in PATH_KINDS:
                shown = _relative(Path(value), relative_to)
            elif isinstance(value, tuple):  # several values
                shown = ",".join(str(each) for each in value)
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
        relative to it where the command reads them so (:attr:`in_place`: a demo run as
        a module, or a script run in its own folder). A script moved out of its folder
        is shown as the editor runs it (:meth:`run_argv`): by its full path, with every
        path in full."""
        if not self.in_place and not names_only:
            return "python " + shell_line(self.run_argv(step, args)[1:])
        argv = self.arguments(
            args, names_only=names_only, relative_to=self.folder if self.in_place else None
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
        defaults (every path, for a script moved out of its folder: :meth:`arguments`),
        paths in full so that it runs from any folder."""
        start = ["-m", self.module] if self.module else [str(self.script)]
        values = self.resolved(args) if args is not None else None
        named = [] if self.stepless else [step]
        return [sys.executable, *start, *named, *self.arguments(values)]

    # -- arguments ------------------------------------------------------------------

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=self.description)
        choices = [s.name for s in self.steps] + ["all"]
        choices += list(dict.fromkeys(alias for s in self.steps for alias in s.aliases))
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

    def _paths(self, args: argparse.Namespace) -> argparse.Namespace:
        """The arguments with relative paths counted from the demo's folder."""
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
            elif option.or_file and value is not None:
                files = [
                    v if v in option.choices or Path(v).is_absolute() else self.folder / v
                    for v in values_of(value)
                ]
                files = [v if v in option.choices else Path(v) for v in files]
                values[option.name] = files[0] if len(files) == 1 else tuple(files)
        return argparse.Namespace(**values)

    def resolved(self, args: argparse.Namespace) -> argparse.Namespace:
        """The arguments with relative paths counted from the demo's folder, and what
        they leave to the scene filled in (:attr:`completes`); the run's folder
        (:func:`run_folder`), what it plays (:func:`run_plays`) and the sources that play
        (:func:`playing`). A run of several parts (:meth:`parts`) keeps the values as
        given, its folder named after them all, and the arguments of each part (resolved)
        as ``series_parts``, in turn."""
        resolved = self._paths(args)
        if not self.stepless:
            parts = self.parts(resolved)
            if len(parts) > 1:
                return self._series(resolved, parts)
            # Several values that come out as one run (an option that changes nothing
            # with the sources picked) are that run.
            resolved = parts[0]
        if self.completes is not None:
            resolved = self.completes(resolved)
        if self.sources is not None:
            picked = self.picked_sources(resolved)
            resolved.playing = None if picked is None else tuple(picked)
        if not self.stepless:
            try:
                work = self._work(resolved)
            except (TypeError, ValueError, OSError):  # no work folder yet (no scene given)
                return resolved
            resolved.run_folder = work / self.run_name(resolved)
            plays = [text for kind, text in self.run_parts(resolved) if kind == "plays"]
            resolved.run_plays = "+".join(plays) or None
            for option in self.options:
                if option.in_run is not None and getattr(resolved, option.name, None) is None:
                    setattr(resolved, option.name, resolved.run_folder / option.in_run)
        return resolved

    def _series(
        self, resolved: argparse.Namespace, parts: list[argparse.Namespace]
    ) -> argparse.Namespace:
        """The arguments (paths resolved) of a run of several ``parts``: its folder in the
        work folder of the first, named after them all (``flight_departure+arrival``)."""
        done = [self.resolved(part) for part in parts]
        resolved.series_parts = done
        resolved.run_plays = None
        resolved.playing = None
        resolved.run_folder = self._work(done[0]) / self.run_name(resolved)
        return resolved

    def check(self, args: argparse.Namespace) -> str | None:
        """What is wrong with the arguments, or None; for a run of several parts
        (:meth:`parts`), with the first part that has something wrong, named."""
        for option in self.options:
            value = getattr(args, option.name, None)
            if isinstance(value, tuple) and not option.varies:
                return f"{option.option} takes one value"
            for each in values_of(value):
                if option.kind in ("int", "float") and each is not None:
                    if isinstance(each, bool) or not isinstance(each, (int, float)):
                        return f"{option.option}: a number, not {each!r}"
                if option.kind == "choice" and option.choices and each is not None:
                    if each in option.choices:
                        continue
                    listed = ", ".join(option.choices)
                    if not option.or_file:
                        return f"{option.option} {each}: not one of {listed}"
                    file = Path(each) if Path(each).is_absolute() else self.folder / each
                    if not file.is_file():
                        return f"{option.option} {each}: no such file, nor one of {listed}"
        hrtf = getattr(args, "hrtf", None)
        if hrtf and "binaural" not in values_of(getattr(args, "format", "binaural")):
            return "--hrtf is for binaural mixes"
        for option in self.options:
            if option.moot is None or getattr(args, option.name, None) in (None, option.default):
                continue
            why = option.moot(self._single(args))
            if why is not None:
                return why
        if not self.stepless:
            for name in given_names(getattr(args, "series", None)) or []:
                dim = self._dim_of(name)
                if dim == SOURCES_DIM:
                    if self.sources is None:
                        return f"--series {name}: this demo gives no choice of sources"
                elif dim == LISTENERS_DIM:
                    if not any(o.kind == "listeners" for o in self.options):
                        return f"--series {name}: this demo gives no choice of listeners"
                elif not any(o.name == dim and o.varies for o in self.options):
                    return (
                        f"--series {name}: neither an option that takes several values here, "
                        f"nor {SOURCES_DIM} or {LISTENERS_DIM}"
                    )
            parts = self.parts(args)
            if len(parts) > 1:
                for part in parts:
                    problem = self.check(part)
                    if problem:
                        return f"{self.run_name(self.resolved(part))}: {problem}"
                return None
            args = parts[0]  # several values that come out as one run: that run
        resolved = self.resolved(args)
        if getattr(args, "hrtf", None) and not Path(resolved.hrtf).is_file():
            return f"--hrtf: no {resolved.hrtf}"
        inputs = getattr(resolved, "inputs", None) or {}
        if inputs and self.plays is not None:
            fed = self.plays(resolved)
            chosen = getattr(resolved, "playing", None)
            if chosen is not None:  # the sources --sources leaves out play nothing
                # A source the demo plays beyond its pick (the crossing's helicopter
                # over the concert's sources) takes an --input as ever.
                listed = {source.name for source in self.source_list(resolved)}
                fed = {
                    name: what for name, what in fed.items() if name in chosen or name not in listed
                }
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
            problem = self.problems(self, args)
            if problem:
                return problem
        return self._sources_problem(resolved) or self._listeners_problem(args, resolved)

    def _sources_problem(self, resolved: argparse.Namespace) -> str | None:
        """What is wrong with the sources ``--sources`` names: none at all, or one that is
        neither a source of the demo nor a group of them (when it can tell)."""
        names = given_names(getattr(resolved, "sources", None))
        if names is None or self.sources is None:
            return None
        if not names:
            return "--sources names no source"
        known = self.source_list(resolved)
        if not known:
            return None
        for name in names:
            if not any(source_matches(source, name) for source in known):
                groups = sorted({s.group for s in known if s.group})
                have = ", ".join([s.name for s in known] + groups)
                return f"--sources {name}: no such source or group here; it has {have}"
        return None

    def _listeners_problem(
        self, args: argparse.Namespace, resolved: argparse.Namespace
    ) -> str | None:
        """What is wrong with the listeners ``--listeners`` names: none at all, or one
        the demo cannot play (:attr:`listeners`, when it can tell)."""
        option = next((o for o in self.options if o.kind == "listeners"), None)
        if option is None:
            return None
        names = listener_names(getattr(args, option.name, None))
        if names is None:
            return None
        if not names:
            return f"{option.option} names no listener"
        if self.listeners is None:
            return None
        try:
            known = self.listeners(resolved)
        except (Exception, SystemExit):  # the steps say what keeps them from the scene
            return None
        unknown = [name for name in names if name not in known]
        if unknown:
            return f"{option.option} {unknown[0]}: no such listener here; it has " + (
                ", ".join(known) or "none"
            )
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

    def all_steps(self) -> list[str]:
        """The steps ``all`` runs: every one."""
        return [s.name for s in self.steps]

    def scene_step(self) -> Step | None:
        """The step that writes the demo's scenes (:attr:`scene_from`), or None."""
        return self.step(self.scene_from) if self.scene_from else None

    def first_step(self, args: argparse.Namespace | None = None) -> str:
        """The command that writes the scenes, or the first step of the demo."""
        step = self.scene_step() or next(s for s in self.steps if not simulates(s))
        return self.command(step.name, args)

    def is_done(self, step: Step, args: argparse.Namespace) -> bool:
        """Whether ``all`` may leave ``step`` out: its result is there already."""
        from acoustic_spaces.demos.pipeline import simulated

        if simulates(step):
            return all(simulated(s, args) for s in self.simulations(args))
        return step.done is not None and step.done(args)

    def run(self, names: Sequence[str], args: argparse.Namespace, skip_done: bool = False) -> None:
        """Run the steps ``names`` in order with ``args`` (paths counted from the demo's
        folder); ``skip_done`` leaves out the steps whose result is there. A run of
        several parts (:meth:`parts`) runs them for each part in turn, each an ordinary
        run in its own folder, and with the ``video`` step, joins the parts' videos into
        one (:func:`~acoustic_spaces.demos.join.join_parts`). Ends with a line that
        begins with :data:`DONE`."""
        problem = self.check(args)
        if problem:
            raise SystemExit(f"error: {problem}")
        args = self.resolved(args)
        parts = getattr(args, "series_parts", None)
        if not parts:
            self._run_steps(names, args, skip_done)
        else:
            for k, part in enumerate(parts):
                print(f"part {k + 1} of {len(parts)}: {run_folder(part).name}", flush=True)
                self._run_steps(names, part, skip_done)
            if "video" in names and not getattr(args, "no_mp4", False):
                from acoustic_spaces.demos.join import join_parts

                join_parts(self, args)
            elif "video" in names:
                from acoustic_spaces.demos.pipeline import chapters_file

                out = product(args, ".mp4")
                for old in (out, chapters_file(out)):
                    if old.is_file():
                        old.unlink()
                        print(f"removed {old}, an older video of this run (--no-mp4)")
        print(f"{DONE}: {', '.join(names)}", flush=True)

    def _run_steps(self, names: Sequence[str], args: argparse.Namespace, skip_done: bool) -> None:
        """The steps ``names`` of a run of one part with ``args`` (resolved), and its
        record (:func:`record_run`)."""
        for name in names:
            step = self.step(name)
            if skip_done and self.is_done(step, args):
                print(f"{name}: done already")
                continue
            if simulates(step):
                self.simulate(args, only_missing=skip_done)
            else:
                step.run(args)
        if not self.stepless and run_folder(args).is_dir():
            record_run(
                run_folder(args),
                demo=self.name,
                title=self.title,
                run=run_folder(args).name,
                parts=[list(part) for part in self.run_parts(args)],
                command=self.command("all", args, names_only=True),
            )

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
        names = demo.all_steps() if args.step == "all" else demo.steps_named(args.step)
        demo.run(names, args, skip_done=args.step == "all")
        return 0


def _by_groups(known: Sequence[Source], these: Sequence[Source]) -> list[str]:
    """The names of the sources ``these`` (of ``known``, in its order), a group of two
    sources or more by its path where they hold every source of it, the widest such
    group first."""
    chosen = {source.name for source in these}
    paths: list[str] = []
    for source in known:
        group = source.group or ""
        while group:
            if group not in paths:
                paths.append(group)
            group = group.rpartition("/")[0]
    out: list[str] = []
    covered: set[str] = set()
    for path in sorted(paths, key=lambda p: p.count("/")):
        members = {s.name for s in known if source_matches(s, path)}
        if len(members) > 1 and members <= chosen and not members & covered:
            out.append(path)
            covered |= members
    out += [source.name for source in these if source.name not in covered]
    return out


def joined_name(parts: Sequence[str]) -> str:
    """``parts`` (each made for a file's name, :func:`name_part`) joined by ``_``: the
    name of a run (:meth:`Demo.run_name`). A name longer than :data:`RUN_NAME_MOST`
    keeps its first part and its last and gives those between cut short, with a digest
    of them, so that two long names stay apart."""
    parts = [part for part in parts if part]
    name = "_".join(parts)
    if len(name) <= RUN_NAME_MOST:
        return name
    if len(parts) < 3:
        return f"{name[: RUN_NAME_MOST - 7].rstrip('_-')}-{digest(name)}"
    first, last, middle = parts[0], parts[-1], "_".join(parts[1:-1])
    room = max(RUN_NAME_MOST - len(first) - len(last) - 9, 0)
    return f"{first}_{middle[:room].rstrip('_-')}-{digest(middle)}_{last}"


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
    # One module per folder: two copies of an example are two demos, each in its own.
    key = f"acoustic_spaces_demos.{script.parent.name}_{digest(str(script.parent))}.{name}"
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
    (read when the example has the file its default names, else written), the
    listeners to tick (``--listeners``) or a text."""
    default = action.default
    if action.dest == "listeners":
        kind = "listeners"  # ticked in a list of the scene's listeners
    elif isinstance(action, argparse._StoreTrueAction):
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
        several=False,  # a script runs once: no parts
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
    names = {option.name for option in options}
    return Demo(
        name=script.stem,
        example=script.parent.name,
        title=module.TITLE,
        description=module.__doc__ or "",
        script=script,
        steps=[Step(RUN, None, "run the script")],
        options=options,
        stepless=True,
        # A --listeners option ticks the listeners of the scene it reads.
        listeners=(
            (lambda args: listeners_in(args.scene)) if {"listeners", "scene"} <= names else None
        ),
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
    "DONE",
    "IN_TURN",
    "LISTENERS_DIM",
    "RESOLVED_ONLY",
    "SOURCES_DIM",
    "SOURCES_HELP",
    "SOURCES_PLAY",
    "TOGETHER",
    "KINDS",
    "LABELS",
    "LISTENERS_MOST",
    "MAP_KINDS",
    "NAMINGS",
    "NO_MOVERS",
    "PATH_KINDS",
    "RUN",
    "RUN_NAME_MOST",
    "RUN_RECORD",
    "SIMULATE",
    "simulates",
    "TEXT_MOST",
    "Demo",
    "Option",
    "Source",
    "Step",
    "chosen_order",
    "common_options",
    "digest",
    "input_option",
    "joined_name",
    "mix_stem",
    "name_part",
    "product",
    "record_run",
    "run_folder",
    "run_plays",
    "given_names",
    "playing",
    "series_option",
    "source_matches",
    "sources_options",
    "values_of",
    "demo_for",
    "demos_of",
    "list_demos",
    "listener_names",
    "listeners_in",
    "listeners_option",
    "load",
    "moving_names",
    "picked",
    "script_demo",
    "scripts_of",
    "shell_line",
    "sibling",
]
