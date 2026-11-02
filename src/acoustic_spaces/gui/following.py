"""What a background run works on now, for the 3D view and the plan to show.

A run says what it does in lines of text, which the status bar and the Messages panel
show: the editor's own runs and a demo, which runs as a process of its own, alike. Each
line of a step on sources and listeners begins with them: ``hall: early reflections``,
``hall -> seat: rendering binaural``, ``car #12 @ 3.40 s: stochastic ray tracing``, the
last a place along a walk (:func:`~acoustic_spaces.walkthrough.place_name`).
:class:`RunFocus` reads those names from each line: the views ring the sources and
listeners they are, and at a place along a walk draw every walker of the run where it
is at that place's time, on the walks the run simulates (:class:`RunView`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from acoustic_spaces.gui.movement import KINDS, on_path, path_of, pose_at
from acoustic_spaces.walkthrough import Knots, movers, place_of

#: Where a line's names end: the first colon and space.
_HEAD = ": "
#: Between a source and a listener of a line.
_PAIR = " -> "
#: One point of a line or surface source: ``hall (2)``.
_POINT = re.compile(r"(?P<name>.+) \(\d+\)")
#: No walks to follow.
_NO_WALKS: Mapping[tuple[str, str], Knots] = MappingProxyType({})


@dataclass(frozen=True)
class RunView:
    """What the views show of a run: the sources and listeners it works on, as (kind,
    name) pairs, and the time (s) of the place along a walk it simulates, None when at
    none, with the walks it simulates, (kind, name) to their knots (compared as the same
    object: knots hold arrays)."""

    names: frozenset[tuple[str, str]] = frozenset()
    time: float | None = None
    walks: Mapping[tuple[str, str], Knots] = field(default=_NO_WALKS, compare=False)

    def same(self, other: RunView) -> bool:
        """Whether the views show ``other`` as they show this."""
        return self == other and self.walks is other.walks

    def pose(self, kind: str, item, playhead: float | None):
        """Where the views draw ``item``, a source or listener of ``kind``: at a place of
        the run, where the run's walk has it then (standing still when the run does not
        walk it); else where the Movement panel's ``playhead`` has it
        (:func:`~acoustic_spaces.gui.movement.pose_at`)."""
        if self.time is None:
            return pose_at(item, playhead)
        path = self.walks.get((kind, item.name))
        return item if path is None else on_path(item, path, self.time)


#: Nothing to show: no run, or one that names nothing.
NO_RUN = RunView()


def walks_of(scene) -> dict[tuple[str, str], Knots]:
    """The walks of ``scene``'s walkers (:func:`~acoustic_spaces.walkthrough.movers`), by
    (kind, name)."""
    walks = {}
    item: Any
    for kind, item in movers(scene):
        path = path_of(item)
        if path is not None:
            walks[kind, item.name] = path
    return walks


class RunFocus:
    """Reads a run's lines (:meth:`read`) against ``scene``, the open scene, and keeps
    what the views show of it (:attr:`view`). ``walks`` gives the scene whose walks the
    run simulates (None: no walks to follow); it is asked at the first place of each
    walk, as a demo may write that scene as it runs."""

    def __init__(self, scene, walks: Callable[[], object | None] | None = None) -> None:
        self._kinds = {item.name: kind for kind in KINDS for item in getattr(scene, kind)}
        self._walks_of = walks
        self._walks: Mapping[tuple[str, str], Knots] | None = None
        self.view = NO_RUN

    def named(self, line: str) -> tuple[list[tuple[str, str]], tuple[str, float] | None]:
        """The scene's sources and listeners that ``line`` begins with, as (kind, name),
        and the walker and time of the place it names, if any."""
        head, colon, _rest = line.strip().partition(_HEAD)
        if not colon:
            return [], None
        named, place = [], None
        for part in head.split(_PAIR):
            found = self._name(part.strip())
            if found is None:
                return [], None  # not a step on sources and listeners
            name, at = found
            named.append((self._kinds[name], name))
            if at is not None and place is None:
                place = (name, at)
        return named, place

    def _name(self, part: str) -> tuple[str, float | None] | None:
        """The scene's source or listener ``part`` names (itself, one of its points or a
        place along its walk), with the place's time; None when it names none."""
        point = _POINT.fullmatch(part)
        for label in (part, point["name"] if point else None):
            if label is None:
                continue
            if label in self._kinds:
                return label, None
            place = place_of(label)
            if place is not None and place[0] in self._kinds:
                return place
        return None

    def read(self, line: str) -> bool:
        """Take the run's latest line; True when what the views show changes. A line of
        sources alone keeps the listeners of the line before (a pass-by names its
        listener, then each place of the vehicle); a line that names nothing shows
        nothing."""
        named, place = self.named(line)
        sources = {pair for pair in named if pair[0] == "sources"}
        listeners = {pair for pair in named if pair[0] == "listeners"}
        if sources and not listeners:
            listeners = {pair for pair in self.view.names if pair[0] == "listeners"}
        time, walks = None, _NO_WALKS
        if place is None:
            self._walks = None  # read again at the next walk: a demo may write it anew
        else:
            if self._walks is None:
                self._walks = self._read_walks()
            if any(name == place[0] for _kind, name in self._walks):
                time, walks = place[1], self._walks
        view = RunView(frozenset(sources | listeners), time, walks)
        changed = not view.same(self.view)
        self.view = view
        return changed

    def _read_walks(self) -> Mapping[tuple[str, str], Knots]:
        if self._walks_of is None:
            return _NO_WALKS
        try:
            scene = self._walks_of()
        except (Exception, SystemExit):  # a demo's scene not written, or unreadable
            return _NO_WALKS
        return _NO_WALKS if scene is None else MappingProxyType(walks_of(scene))


__all__ = ["NO_RUN", "RunFocus", "RunView", "walks_of"]
