"""Example scenes that come with the package.

They are the scenes in the repository's ``examples`` folder, which wheels carry as
``acoustic_spaces/example_scenes``::

    from acoustic_spaces import Room
    from acoustic_spaces.examples import example_path, list_examples

    for example in list_examples():
        print(example.name, example.title)
    room = Room.from_yaml(example_path("concert_hall"))

``acoustic-spaces examples`` lists them, and ``--copy DIR`` copies them to a folder you
can edit; the editor opens them from its File menu, in the groups of :data:`GROUPS`.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from acoustic_spaces.geometry.io import extra_for, missing_extra
from acoustic_spaces.i18n import N_

SCENE_FILE = "scene.yaml"

#: The groups the examples are shown in, in order, each with its examples (folder names)
#: from the simplest. The titles are English; the editor translates them.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        N_("Rooms"),
        ("shoebox", "room_modes", "studio", "listening_room", "apartment", "archway", "workshop"),
    ),
    (
        N_("Halls and churches"),
        ("lecture_hall", "concert_hall", "orchestra", "church", "gallery"),
    ),
    (N_("Sound systems"), ("town_hall_pa", "concert_stage", "stadium_concert")),
    (
        N_("Outdoors"),
        (
            "street",
            "roadside",
            "tree_belt",
            "hillside",
            "open_air_stage",
            "lakeside_concert",
            "helicopter_flyover",
        ),
    ),
    (N_("Models from CAD and BIM files"), ("rehearsal_room", "bim_office", "leaky_classroom")),
    (N_("Measurements and material libraries"), ("fitting", "material_libraries")),
)
#: The group of an example that :data:`GROUPS` leaves out.
OTHER_GROUP = N_("Other examples")


@dataclass(frozen=True)
class Example:
    #: Folder name, used to pick the example.
    name: str
    #: The scene's ``name``.
    title: str
    #: The comment at the top of the scene file.
    description: str
    #: The scene file.
    path: Path
    #: Its model file, or None for a room built from dimensions.
    model: Path | None = None
    #: The group it is shown in (English: see :data:`GROUPS`).
    group: str = OTHER_GROUP

    @property
    def extra(self) -> str | None:
        """The extra its model is read with (``ifc`` for an IFC file), or None."""
        return None if self.model is None else extra_for(self.model)

    @property
    def missing_extra(self) -> str | None:
        """The extra its model needs when that is not installed, or None."""
        return None if self.model is None else missing_extra(self.model)


def examples_dir() -> Path:
    """The folder that holds one sub-folder per example.

    Installed packages carry it as ``acoustic_spaces/example_scenes``; a source checkout
    (an editable install) uses the repository's ``examples`` folder.
    """
    installed = _installed_dir()
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "examples"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("the example scenes are missing from this installation")


def _installed_dir() -> Path:
    """Where an installed package keeps its examples (``acoustic_spaces/example_scenes``),
    whether or not it is there."""
    return Path(str(resources.files("acoustic_spaces"))) / "example_scenes"


def is_installed_example(path: str | Path) -> bool:
    """Whether ``path`` lies among the examples of an installed package, which an upgrade
    or an uninstall replaces or removes: nothing is to be written there, even where the
    folder can be written (a virtual environment). A source checkout's ``examples``
    folder is not."""
    installed = _installed_dir()
    if not installed.is_dir():
        return False
    return Path(path).resolve().is_relative_to(installed.resolve())


def _description(scene_file: Path) -> str:
    lines = []
    for line in scene_file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("#").strip())
    return " ".join(lines)


def list_examples() -> list[Example]:
    """Every example scene, sorted by folder name."""
    out = []
    for folder in sorted(examples_dir().iterdir()):
        scene_file = folder / SCENE_FILE
        if not scene_file.is_file():
            continue
        data = yaml.safe_load(scene_file.read_text(encoding="utf-8"))
        model = (data.get("geometry") or {}).get("file")
        out.append(
            Example(
                name=folder.name,
                title=str(data.get("name", folder.name)),
                description=_description(scene_file),
                path=scene_file,
                model=None if model is None else folder / model,
                group=_GROUP_OF.get(folder.name, OTHER_GROUP),
            )
        )
    return out


_GROUP_OF = {name: group for group, names in GROUPS for name in names}


def example_groups(examples: list[Example] | None = None) -> list[tuple[str, list[Example]]]:
    """The examples (default: all) by group, in the order of :data:`GROUPS` and within a
    group from the simplest; those it leaves out last, in :data:`OTHER_GROUP`. Groups
    with no example are left out."""
    examples = list_examples() if examples is None else examples
    rank = {name: k for _, names in GROUPS for k, name in enumerate(names)}
    order = [group for group, _ in GROUPS] + [OTHER_GROUP]
    out = []
    for group in order:
        members = [e for e in examples if e.group == group]
        if members:
            out.append((group, sorted(members, key=lambda e: (rank.get(e.name, 0), e.name))))
    return out


def example_path(name: str) -> Path:
    """The scene file of the example ``name`` (a folder name from :func:`list_examples`)."""
    scene_file = examples_dir() / name / SCENE_FILE
    if not scene_file.is_file():
        names = ", ".join(e.name for e in list_examples())
        raise ValueError(f"no example named {name!r}; examples: {names}")
    return scene_file


def example_of(scene_file: str | Path) -> str | None:
    """The example the scene file ``scene_file`` is, or is a copy of: its folder's name
    when that names an example, else the example whose scene has the same ``name``, so
    that a copy saved anywhere keeps its example; None for any other scene, or one that
    cannot be read."""
    path = Path(scene_file)
    folder = path.parent.name
    if (examples_dir() / folder / SCENE_FILE).is_file():
        return folder
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    name = data.get("name") if isinstance(data, dict) else None
    if name is None:
        return None
    return next((e.name for e in list_examples() if e.title == str(name)), None)


def copy_examples(destination: str | Path, names: list[str] | None = None) -> list[Path]:
    """Copy examples (all, or ``names``) into ``destination``, one folder each, and
    return their scene files. Raises ``FileExistsError`` rather than overwrite a folder."""
    destination = Path(destination)
    chosen = list_examples()
    if names is not None:
        chosen = [e for e in chosen if e.name in names]
        found = {e.name for e in chosen}
        missing = [name for name in names if name not in found]
        if missing:
            raise ValueError(f"no example named {', '.join(missing)}")
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for example in chosen:
        target = destination / example.name
        if target.exists():
            raise FileExistsError(f"{target} already exists")
        shutil.copytree(example.path.parent, target)
        copied.append(target / SCENE_FILE)
    return copied


__all__ = [
    "GROUPS",
    "OTHER_GROUP",
    "Example",
    "copy_examples",
    "example_groups",
    "example_of",
    "example_path",
    "examples_dir",
    "is_installed_example",
    "list_examples",
]
