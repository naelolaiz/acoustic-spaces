"""The waves that are one wave, in the bands where the early paths add in energy.

Indoors, above ``simulation.early.coherent_max_band_hz``, the early paths add in energy,
as if their phases were random. The sound bent round an edge does not: near its boundary
each term of the uniform theory of diffraction continues the geometric wave it belongs
to, and its pressure adds to that wave's, so that the two together change smoothly where
the geometric wave stops (:mod:`~acoustic_spaces.engine.diffraction`,
:mod:`~acoustic_spaces.engine.edge_fade`). Away from it the term is a wave of its own,
bent round the edge, whose terms (one for each boundary of the edge) add up as one wave
by how far each lies from its boundary.

Each term therefore counts in two waves. It continues its geometric wave by c and is
part of its diffracted wave by s = sqrt(1 - c^2): its own energy stays
``|t|^2 = (c^2 + s^2) |t|^2``, and two terms cohere by the products of their shares,
through every wave both are part of. On the lit side of its boundary, where the
geometric wave arrives, c is the term's coherence with it over the band, as the term
goes X / k farther (X the argument of its transition function); on the dark side, where
the wave is gone, c = ``sqrt(1 - |F(X)|^2)``, so that the term continues it near the
boundary only, and deep in the shadow all the terms of an edge are one wave
(:func:`~acoustic_spaces.engine.diffraction.transition`). All the edges of an event are
one diffracted wave, as the edges round an opening are in the boundary diffraction wave
(Rubinowicz, 1917): the terms whose paths over an edge run from the same image of the
source to the same image of the listener, whatever the edge, so that round a narrow face
the two edges' terms nearly cancel as they do in pressure, and the many edges of a
curved surface's facets add up to the surface's own reflection.

A wave's energy is that of the pressures of its pieces added up, averaged over each
band's frequencies and over the slice of the band each frequency stands for, where two
pieces that arrive t seconds apart interfere by sinc(t times the slice's width) of what
they would at its frequency: it is never negative, as it is for the pressures of any
set of waves. A wave of a few pieces takes them pair by pair; in one of more, the slice
is averaged over with a Gauss-Legendre rule of as many points as the pieces' spread in
arrival needs (within 1e-10 of sinc), so that the work grows with the pieces, not with
their pairs: the facets of a curved surface give events of hundreds of thousands of
terms.

The wave of a specular path, found by the early stage, is that path's: the cross terms of
the pieces that continue it go to its energy, and each piece keeps its own; where the
cross terms take more than the path has, the rest comes off the pieces in proportion to
their energy, which is always enough. In every other wave each piece takes its own
energy and half its cross term with each other piece; where that comes to less than
nothing it takes none, and the others give it in proportion, so that the wave's energy
stays what it is.
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: Images closer than this, relative to the model's size, are the same.
_STEP = 1e-5
#: At most this many values (pieces times frequencies times points of the rule) at once.
_VALUES = 1 << 22
#: At most this many points of the rule in one panel.
_PANEL = 128
#: Waves of at most this many pieces take their pairs one by one.
_PAIRS = 32


class Clusters:
    """The pieces of the waves that are one wave (see the module notes), in the bands
    where the early paths add in energy, whose slices are ``slice_hz`` (Hz) wide, for
    sound at ``speed_of_sound``; images closer than :data:`_STEP` times ``size`` (m)
    are the same."""

    def __init__(self, slice_hz: np.ndarray, speed_of_sound: float, size: float) -> None:
        self.slice_hz = np.asarray(slice_hz, dtype=float)
        self.speed_of_sound = float(speed_of_sound)
        self.step = _STEP * max(float(size), 1e-300)
        self._ids: dict[bytes, int] = {}
        self._private = 0
        self._parts: list[dict] = []
        self._alone: list[tuple[np.ndarray, np.ndarray]] = []

    @property
    def bands(self) -> int:
        return len(self.slice_hz)

    def geometric(self, image: np.ndarray, order: np.ndarray) -> np.ndarray:
        """The waves of the specular paths of ``order`` (n,) reflections whose image of
        the source is ``image`` (n, 3)."""
        image = np.asarray(image, dtype=float).reshape(-1, 3)
        return self._keys(0, image, np.broadcast_to(order, len(image)))

    def diffracted(self, source_side: np.ndarray, listener_side: np.ndarray) -> np.ndarray:
        """The diffracted waves whose paths over an edge run from the image of the source
        ``source_side`` (n, 3) to that of the listener ``listener_side`` (n, 3)."""
        points = np.hstack(
            [
                np.asarray(source_side, dtype=float).reshape(-1, 3),
                np.asarray(listener_side, dtype=float).reshape(-1, 3),
            ]
        )
        return self._keys(1, points)

    def private(self, count: int) -> np.ndarray:
        """``count`` waves of their own."""
        out = -1 - (self._private + np.arange(count, dtype=np.int64))
        self._private += count
        return out

    def _keys(self, kind: int, points: np.ndarray, order=None) -> np.ndarray:
        if len(points) == 0:
            return np.zeros(0, dtype=np.int64)
        grid = np.round(points / self.step).astype(np.int64)
        tag = np.full((len(grid), 1), kind, dtype=np.int64)
        if order is not None:
            tag = np.hstack([tag, np.asarray(order, dtype=np.int64).reshape(-1, 1)])
        grid = np.hstack([tag, grid])
        unique, inverse = np.unique(grid, axis=0, return_inverse=True)
        ids = np.empty(len(unique), dtype=np.int64)
        for i, row in enumerate(unique):
            key = row.tobytes()
            ids[i] = self._ids.setdefault(key, len(self._ids))
        return ids[inverse.reshape(-1)]

    def add(
        self,
        rows: np.ndarray,
        keys: np.ndarray,
        pressure: np.ndarray,
        distance: np.ndarray,
        anchor: bool = False,
        own: np.ndarray | None = None,
    ) -> None:
        """Pieces of the rows ``rows`` (-1 for a row left out) in the waves ``keys``:
        their pressure (n, bands, M) with the phase of their whole way, which is
        ``distance`` (n,) metres long. ``anchor``: the rows are specular paths, whose
        waves they are, with energy ``own`` (n, bands)."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        own = np.mean(np.abs(pressure) ** 2, axis=2) if own is None else np.asarray(own, float)
        keys = np.broadcast_to(np.asarray(keys, dtype=np.int64), rows.shape)
        distance = np.broadcast_to(np.asarray(distance, dtype=float), rows.shape)
        # Pieces with nothing in them change nothing.
        some = own.max(axis=1, initial=0.0) > 0.0
        if not some.any():
            return
        pressure = np.asarray(pressure, dtype=np.complex64)
        if not some.all():
            rows, keys, pressure = rows[some], keys[some], pressure[some]
            distance, own = distance[some], own[some]
        self._parts.append(
            {
                "rows": rows,
                "keys": np.array(keys),
                "pressure": pressure,
                "distance": np.array(distance),
                "anchor": np.full(len(rows), anchor),
                "own": own,
            }
        )

    def alone(self, rows: np.ndarray, energy: np.ndarray) -> None:
        """Pieces of the rows ``rows`` that are waves of their own, with ``energy``
        (n, bands)."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        if len(rows):
            self._alone.append((rows, np.asarray(energy, dtype=float)))

    def renumber(self, mapping: np.ndarray) -> None:
        """Rows ``i`` become ``mapping[i]`` (-1: left out)."""
        mapping = np.asarray(mapping, dtype=np.int64)

        def move(rows):
            return np.where(rows >= 0, mapping[np.maximum(rows, 0)], -1)

        for part in self._parts:
            part["rows"] = move(part["rows"])
        self._alone = [(move(rows), energy) for rows, energy in self._alone]

    def _gather(self) -> dict | None:
        """The pieces sorted by wave: ``order`` their places among all the pieces and
        ``keys`` their waves in that order. All else stays in the parts, as the pieces of
        a curved surface's facets take gigabytes (:meth:`_take`)."""
        if not self._parts:
            return None
        keys = np.concatenate([p["keys"] for p in self._parts])
        order = np.argsort(keys, kind="stable")
        return {
            "order": order,
            "keys": keys[order],
            "starts": np.cumsum([0] + [len(p["rows"]) for p in self._parts]),
        }

    def _take(self, part: dict, sel: np.ndarray, names: tuple[str, ...]) -> list[np.ndarray]:
        """The values ``names`` of the pieces ``sel`` of :meth:`_gather`'s ``part``."""
        index = part["order"][sel]
        which = np.searchsorted(part["starts"], index, side="right") - 1
        by_part = np.argsort(which, kind="stable")
        bounds = np.searchsorted(which[by_part], np.arange(len(self._parts) + 1))
        out = []
        for name in names:
            first = self._parts[0][name]
            kind = np.result_type(*(p[name] for p in self._parts))
            out.append(np.empty((len(index), *first.shape[1:]), dtype=kind))
        for i in np.flatnonzero(np.diff(bounds)):
            mine = by_part[bounds[i] : bounds[i + 1]]
            local = index[mine] - part["starts"][i]
            for values, name in zip(out, names, strict=True):
                values[mine] = self._parts[i][name][local]
        return out

    def resolve(self, count: int) -> tuple[np.ndarray, np.ndarray]:
        """(count, bands) the energy of each of ``count`` rows from its pieces, and
        (count,) which rows have any."""
        energy = np.zeros((count, self.bands))
        has = np.zeros(count, dtype=bool)
        for rows, alone in self._alone:
            kept = rows >= 0
            np.add.at(energy, rows[kept], alone[kept])
            has[rows[kept]] = True
        part = self._gather() if self.bands else None
        if part is None:
            return energy, has
        _, start, size = np.unique(part["keys"], return_index=True, return_counts=True)
        del part["keys"]
        # The waves with more than one piece, a batch of whole waves at a time, each with
        # the waves of one piece up to the next batch, in the order of the waves.
        many = np.flatnonzero(size > 1)
        values = size[many] * self._parts[0]["pressure"].shape[2]
        batch = np.cumsum(values) // _VALUES
        batches = np.unique(batch)
        bounds = [0, *start[many[np.searchsorted(batch, batches[1:])]], len(part["order"])]
        for i in range(len(bounds) - 1):
            low = bounds[i]
            rows, given = self._take(part, np.arange(low, bounds[i + 1]), ("rows", "own"))
            if i < len(batches):
                waves = many[batch == batches[i]]
                local = np.repeat(np.arange(len(waves)), size[waves])
                before = np.cumsum(size[waves]) - size[waves]
                sel = start[waves][local] + np.arange(len(local)) - before[local]
                pressure, distance, anchor = self._take(
                    part, sel, ("pressure", "distance", "anchor")
                )
                share = self._shares(pressure, distance, local)
                del pressure
                mine = sel - low
                given[mine] = _distribute(share, given[mine], local, anchor, rows[mine] >= 0)
            kept = rows >= 0
            np.add.at(energy, rows[kept], given[kept])
            has[rows[kept]] = True
        return np.maximum(energy, 0.0), has

    def _shares(self, pressure: np.ndarray, distance: np.ndarray, cluster: np.ndarray):
        """(n, bands) each piece's own energy and half its cross terms with the other
        pieces of its wave (``cluster``, sorted), averaged over the band's frequencies
        and the slices they stand for."""
        n, bands, per_band = pressure.shape
        _, start, size = np.unique(cluster, return_index=True, return_counts=True)
        local = np.repeat(np.arange(len(start)), size)
        low = np.minimum.reduceat(distance, start)
        high = np.maximum.reduceat(distance, start)
        middle = 0.5 * (low + high)
        relative = (distance - middle[local]) / self.speed_of_sound  # (n,) s
        spread = (high - low) / self.speed_of_sound
        out = np.zeros((n, bands))
        for b in range(bands):
            width = self.slice_hz[b]
            # A wave of few pieces (no more than the rule's points) takes its pairs one
            # by one.
            panels, points = _rule_size(width, spread)
            pairs = (size <= np.minimum(panels * points, _PAIRS))[local]
            mine = np.flatnonzero(pairs)
            if len(mine):
                out[mine, b] = _pair_shares(pressure[mine, b], relative[mine], local[mine], width)
            rules, which = np.unique(np.column_stack([panels, points]), axis=0, return_inverse=True)
            which = which.reshape(-1)[local]
            for i, (count, per) in enumerate(rules.astype(int)):
                mine = np.flatnonzero((which == i) & ~pairs)
                if len(mine):
                    out[mine, b] = _slice_shares(
                        pressure[mine, b], relative[mine], local[mine], width, count, per
                    )
        return out


def _pair_shares(pressure, relative, cluster, width) -> np.ndarray:
    """(n,) the shares of pieces ``pressure`` (n, M) of the waves ``cluster`` (sorted),
    arriving ``relative`` seconds apart as they are, pair by pair: two pieces' cross
    term fades by sinc over slices ``width`` Hz wide."""
    _, start, size = np.unique(cluster, return_index=True, return_counts=True)
    local = np.repeat(np.arange(len(start)), size)
    n, per_band = pressure.shape
    counts = size[local]  # each piece's pairs
    total = np.cumsum(counts)
    limit = max(_VALUES // per_band, 1)
    out = np.zeros(n)
    first = 0
    while first < n:
        last = max(
            int(np.searchsorted(total, total[first] - counts[first] + limit, "right")), first + 1
        )
        mine = np.arange(first, last)
        c = counts[mine]
        i = np.repeat(mine, c)
        j = np.repeat(start[local[mine]], c) + np.arange(len(i)) - np.repeat(np.cumsum(c) - c, c)
        product = np.einsum("pm,pm->p", pressure[i], np.conj(pressure[j])).real / per_band
        product *= np.sinc(width * (relative[i] - relative[j]))
        out[first:last] = np.bincount(i - first, weights=product, minlength=last - first)
        first = last
    return out


def _rule_size(width: float, spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The panels and the points in each of the rule over slices ``width`` Hz wide for
    waves whose pieces arrive over ``spread`` seconds: 1.75 X + 14 points, X = width *
    spread the turns of phase between the first piece and the last, in panels of at most
    :data:`_PANEL` points."""
    need = 1.75 * width * np.asarray(spread, dtype=float)
    panels = np.maximum(np.ceil(need / (_PANEL - 14)), 1.0)
    points = 2 ** np.ceil(np.log2(np.maximum(need / panels + 14.0, 16.0)))
    return panels, points


@functools.cache
def _rule(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights over (0, 1)."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _slice_shares(pressure, relative, cluster, width, panels, points) -> np.ndarray:
    """(n,) the shares of pieces ``pressure`` (n, M) of the waves ``cluster`` (sorted),
    arriving ``relative`` seconds after the middle of their wave, averaged over slices
    ``width`` Hz wide by ``panels`` panels of ``points`` Gauss-Legendre points (each
    point of the slice turns a piece's phase by its frequency times its delay)."""
    nodes, weights = _rule(points)
    _, start, size = np.unique(cluster, return_index=True, return_counts=True)
    n, per_band = pressure.shape
    out = np.zeros(n)
    turn = -2j * math.pi * width
    for first, count in zip(start, size, strict=True):
        p, r = pressure[first : first + count], relative[first : first + count]
        # At most _VALUES values at once: these panels, for these pieces.
        step = max(1, _VALUES // max(count * points, 1))
        rows = max(1, _VALUES // (min(step, panels) * points))
        chunks = range(0, count, rows)
        for low in range(0, panels, step):
            ends = np.arange(low, min(low + step, panels)) / panels - 0.5
            within = nodes / panels
            total = 0.0  # (q, M): the wave at each point
            for a in chunks:
                phase = _phases(turn * r[a : a + rows], ends, within)
                total = total + phase.T @ p[a : a + rows]
            weight = np.tile(weights / panels, phase.shape[1] // points)
            for a in chunks:
                if len(chunks) > 1:
                    phase = _phases(turn * r[a : a + rows], ends, within)
                mine = np.real(phase * (p[a : a + rows] @ np.conj(total).T)) / per_band
                out[first + a : first + a + len(phase)] += mine @ weight
    return out


def _phases(turn: np.ndarray, ends: np.ndarray, within: np.ndarray) -> np.ndarray:
    """(n, q) the phases of pieces that turn by ``turn`` (n,) over a slice, at the rule's
    points ``within`` (relative to the slice) of each of the panels from ``ends``."""
    turned = turn[:, None, None]
    return (np.exp(turned * ends[None, :, None]) * np.exp(turned * within)).reshape(len(turn), -1)


def _distribute(share, own, cluster, anchor, kept) -> np.ndarray:
    """(n, bands) the energy each piece is given of its wave's (``cluster``, sorted),
    from its ``share`` and ``own`` energy (see the module notes); pieces not ``kept``
    are given none."""
    _, start, size = np.unique(cluster, return_index=True, return_counts=True)
    local = np.repeat(np.arange(len(start)), size)
    total = np.maximum(np.add.reduceat(share, start, axis=0), 0.0)  # (waves, bands)
    given = np.zeros_like(share)
    # The wave of a specular path: the first such piece is the path's.
    first = np.full(len(start), -1, dtype=np.int64)
    rows = np.flatnonzero(anchor)[::-1]
    first[local[rows]] = rows
    anchored = first[local] >= 0
    path = np.zeros(len(share), dtype=bool)
    path[first[first >= 0]] = True
    rest = anchored & ~path
    if anchored.any():
        others = np.zeros_like(total)
        np.add.at(others, local[rest], own[rest])
        left = total - others  # what the path keeps
        given[rest & kept] = own[rest & kept]
        owed = np.maximum(-left, 0.0)
        mine = first[first >= 0]
        given[mine] = np.maximum(left[local[mine]], 0.0)
        # Where the cross terms take more than the path has: off the pieces that are kept.
        have = np.zeros_like(total)
        np.add.at(have, local[rest & kept], own[rest & kept])
        part = np.where(have > 0.0, np.minimum(owed / np.where(have > 0.0, have, 1.0), 1.0), 0.0)
        sel = rest & kept
        given[sel] = own[sel] * (1.0 - part[local[sel]])
    free = ~anchored & kept
    if free.any():
        positive = np.where(free[:, None], np.maximum(share, 0.0), 0.0)
        weight = np.zeros_like(total)
        np.add.at(weight, local, positive)
        fallback = np.zeros_like(total)
        np.add.at(fallback, local[free], own[free])
        use = np.where(weight[local] > 0.0, positive, np.where(free[:, None], own, 0.0))
        norm = np.where(weight > 0.0, weight, fallback)
        scale = np.where(norm > 0.0, total / np.where(norm > 0.0, norm, 1.0), 0.0)
        given[free] = (use * scale[local])[free]
    return given
