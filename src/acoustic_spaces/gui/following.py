"""What a background run works on now, for the 3D view and the plan to show.

A run says what it does in lines of text, which the status bar and the Messages panel
show: the editor's own runs and a demo, which runs as a process of its own, alike. Each
line of a step on sources and listeners begins with them: ``hall: early reflections``,
``hall -> seat: rendering binaural``, ``car #12 @ 3.40 s: stochastic ray tracing``, the
last a place along a walk (:func:`~acoustic_spaces.walkthrough.place_name`); a demo's
video names what each moment of it shows, ``stage, car @ 3.40 s -> seat: encoding the
video, ...`` (:func:`~acoustic_spaces.walkthrough.moment_name`), and a head it turns
with its yaw, ``guide -> standing facing 90.0 deg: turning the head, ...``
(:func:`~acoustic_spaces.walkthrough.facing_name`). :class:`RunFocus` reads those names
from each line: the views ring the sources and listeners they are, turn each to its yaw,
and at a place along a walk draw every walker of the run where it is at that place's
time, on the walks the run simulates (:class:`RunView`).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from acoustic_spaces.gui.movement import KINDS, on_path, path_of, pose_at
from acoustic_spaces.walkthrough import Knots, facing_of, movers, place_of

#: Where a line's names end: the first colon and space.
_HEAD = ": "
#: Between a line's sources and its listeners.
_PAIR = " -> "
#: Between two sources, or two listeners, of a line.
_AND = ", "
#: One point of a line or surface source: ``hall (2)``.
_POINT = re.compile(r"(?P<name>.+) \(\d+\)")
#: A name a line gives: the source's or listener's, the time of the place along its walk
#: and its yaw (degrees), when it gives them.
_Named = tuple[str, float | None, float | None]
#: No walks to follow.
_NO_WALKS: Mapping[tuple[str, str], Knots] = MappingProxyType({})


@dataclass(frozen=True)
class RunView:
    """What the views show of a run: the sources and listeners it works on, as (kind,
    name) pairs, and the time (s) of the place along a walk it simulates, None when at
    none, with the walks it simulates, (kind, name) to their knots (compared as the same
    object: knots hold arrays); and the yaw (degrees) of each it turns, as a scripted
    head turn has it then, ((kind, name), yaw) pairs."""

    names: frozenset[tuple[str, str]] = frozenset()
    time: float | None = None
    walks: Mapping[tuple[str, str], Knots] = field(default=_NO_WALKS, compare=False)
    facing: tuple[tuple[tuple[str, str], float], ...] = ()

    def same(self, other: RunView) -> bool:
        """Whether the views show ``other`` as they show this."""
        return self == other and self.walks is other.walks

    def pose(self, kind: str, item, playhead: float | None):
        """Where the views draw ``item``, a source or listener of ``kind``: at a place of
        the run, where the run's walk has it then (standing still when the run does not
        walk it); else where the Movement panel's ``playhead`` has it
        (:func:`~acoustic_spaces.gui.movement.pose_at`). Turned to the run's yaw for it,
        when it gives one."""
        if self.time is None:
            shown = pose_at(item, playhead)
        else:
            path = self.walks.get((kind, item.name))
            shown = item if path is None else on_path(item, path, self.time)
        yaw = dict(self.facing).get((kind, item.name))
        if yaw is None:
            return shown
        turned = shown.orientation.model_copy(update={"yaw": yaw})
        return shown.model_copy(update={"orientation": turned})


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

    def named(
        self, line: str
    ) -> tuple[list[tuple[str, str]], tuple[str, float] | None, dict[tuple[str, str], float]]:
        """The scene's sources and listeners that ``line`` begins with, as (kind, name);
        the walker and time of the place it names, if any; and the yaw of each it says
        is turned (:func:`~acoustic_spaces.walkthrough.facing_name`)."""
        head, colon, _rest = line.strip().partition(_HEAD)
        if not colon:
            return [], None, {}
        named: list[tuple[str, str]] = []
        place, facing = None, {}
        for side in head.split(_PAIR):
            found = self._names(side.strip())
            if found is None:
                return [], None, {}  # not a step on sources and listeners
            for name, at, yaw in found:
                pair = (self._kinds[name], name)
                named.append(pair)
                if at is not None and place is None:
                    place = (name, at)
                if yaw is not None:
                    facing[pair] = yaw
        return named, place, facing

    def _names(self, side: str) -> list[_Named] | None:
        """The scene's sources or listeners that one side of a line's head names: one,
        or several between commas; None unless each names one."""
        alone = self._name(side)
        if alone is not None:
            return [alone]
        found = []
        for part in side.split(_AND):
            name = self._name(part.strip())
            if name is None:
                return None
            found.append(name)
        return found

    def _name(self, part: str) -> _Named | None:
        """The scene's source or listener ``part`` names (itself, one of its points, a
        place along its walk, each turned to a yaw or not), with the place's time and
        the yaw; None when it names none."""
        found = self._named(part)
        if found is not None:
            return (*found, None)
        turned = facing_of(part)
        found = None if turned is None else self._named(turned[0])
        return None if found is None or turned is None else (*found, turned[1])

    def _named(self, part: str) -> tuple[str, float | None] | None:
        point = _POINT.fullmatch(part)
        for label in (part, point["name"] if point else None):
            if label is None:
                continue
            if label in self._kinds:
                return label, None
            place = place_of(label)
            if place is None:
                continue
            # A walker's name may end as a place's number does: ``seat #3 @ 1.00 s``.
            for name in (label.rpartition(" @ ")[0], place[0]):
                if name in self._kinds:
                    return name, place[1]
        return None

    def read(self, line: str) -> bool:
        """Take the run's latest line; True when what the views show changes. A line of
        sources alone keeps the listeners of the line before (a pass-by names its
        listener, then each place of the vehicle); a line that names nothing shows
        nothing."""
        named, place, facing = self.named(line)
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
        view = RunView(frozenset(sources | listeners), time, walks, tuple(sorted(facing.items())))
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
