"""The paraxial Kirchhoff field over edges in a row that a straight way passes close by.

Near the straight way from a point source to a listener, at small angles to it, a row of
obstacles that each stand across the way (thin screens, the tops of blocks) lets through
the share of the free field that the Fresnel-Kirchhoff integral over the planes of the
obstacles gives. Over every obstacle the integral is Gaussian in the two coordinates
across the way, with the complex variance of the free field's Fresnel zone there: the
field relative to the free field is the probability that a Brownian bridge from the
source to the listener misses every obstacle, analytically continued to the complex
variance i d / k of the bridge at the distance d = x (D - x) / D across, x along the way
of D. The bridge is Markov, so the probability is an integral over a chain of Gaussian
steps, one for each obstacle in turn: with the bridge at the obstacles standardised
(unit variance) the correlation between two in turn is

    rho = sqrt(x_i (D - x_j) / (x_j (D - x_i)))    (x_i < x_j),

and an edge whose line runs across the way at h standard deviations from it lets the
bridge through on one side of h. Two knife edges right on the way let through
1/4 + asin(rho) / (2 pi), three 1/8 + (asin rho_12 + asin rho_13 + asin rho_23) / (4 pi).

The thresholds are complex (h has the phase pi / 4 of the bridge's deviation), and the
integrals run from each threshold along the real direction: on the clear side towards
+infinity and on the blocked side towards -infinity, where the Gaussian falls off in
both directions. An edge that stops only part of the bridge on its blocked side
(:func:`bridge_chain`'s ``blocked`` weights below 1) stands for an edge that is in the
row only in part: the mean over which edges are in it of the field over those that
are, as each edge's term in the product is linear in whether it is there.

The edges of a row (:func:`find_row`) are of two families: those that lie across the
way at right angles to one direction ``q`` across it (the tops of screens and blocks,
whose obstacles stand on one side of them along ``q``), and the edges of thin screens
along ``q`` (a screen's upright end), whose screens stand on one side of them along the
third direction ``p``. The bridge's two coordinates along ``q`` and ``p`` are
independent, so the probability is the product of a chain in each. A screen whose top
and end are both in the row stops the bridge where it is on the screen's side of both:
the product of the two edges' conditions, which the two chains give as the difference
of two products (:meth:`Row.branches`). A rigid flat top between two edges of the
first family (a block's top) reflects the bridge: by Neumann's images, the field over
it is the bridge's plus that of the bridge from the source's image in the top's plane,
which crosses the edges before the top mirrored in it.

Each edge is in the row by how close the way passes it (:func:`membership`): far from
the way the edges' paths of the uniform theory of diffraction hold, and the bridge's
chain is not needed; near it the paraxial field holds where the series of those paths
over one and two edges is cut short. As Kirchhoff's approximation, it has the terms of
the edges' paths on the incident wave's boundary and on a top's reflection only: where
it stands for the edges' paths, their other terms (the reflections off the other faces,
smooth near the way) go out from each edge on their own, past the row's other nodes
(:meth:`Row.past`). Rows of the edges of one obstacle alone (a block's top, a screen's
corner) are the edges' paths'. The field the edges' paths give, with weights that make
up the rest, and the row's field are added (:class:`Row`); where its chains cannot be
computed in double precision or resolved by the quadrature's points, the row takes
nothing (:meth:`Row.computable`).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass

import numpy as np

from acoustic_spaces import products

#: The integrals over each step of the chain run this many standard deviations from the
#: threshold, where the Gaussian has fallen to exp(-32) of its peak.
_REACH = 8.0
#: Gauss-Legendre points on each of the two rays from a threshold: at least _FEWEST, and
#: _PER_WIDTH over the width sqrt(1 - rho^2) of the narrowest step (edges close together
#: along the way) and as many again for each _TURN radians per standard deviation at
#: which a node's integrand turns along its ray at fastest (a Gaussian along a real ray
#: from a complex threshold is a wave, :func:`_chain_cost`), up to _MOST. On 435 chains
#: of two to four nodes within 3 standard deviations, at random, that the row takes
#: (:meth:`Row.growth` below _GROWTH_FAR), they are within 5e-8 of the 400-point integral.
_FEWEST = 24
_PER_WIDTH = 10.0
_TURN = 3.0
_MOST = 128
#: A step narrower than _APART takes more than _MOST / 2 points, and one narrower than
#: _CLOSE more than _MOST (two edges of one family 5 cm apart half way along 25 m). Over
#: so narrow a step the bridge hardly moves: the chain takes the two nodes as one, the
#: bridge at the second at the first's place (Y_j = +-Y_i), whose weight is the product
#: of theirs, below _CLOSE, and goes over from the two to the one between _APART and
#: _CLOSE (:func:`_chain`). One node for two knife edges at the bridge's mean lets
#: through 1/2 for 1/4 + asin(rho) / (2 pi) = 1/2 - asin(sqrt(1 - rho^2)) / (2 pi):
#: within _CLOSE / pi (2.5 %) of it.
_APART = 2.0 * _PER_WIDTH / _MOST
_CLOSE = _PER_WIDTH / _MOST
#: An edge is in the row where the way passes it within ROW_NEAR standard deviations of
#: the bridge there, and not beyond ROW_FAR: the bridge's density at the edge is then
#: within exp(-1/2) of its peak, and below exp(-9/2) (1 %), where the edge's paths over
#: one and two edges are a few hundredths of the free field and their series converges.
ROW_NEAR = 1.0
ROW_FAR = 3.0
#: From one node of a family to the next, the chain's integrand along the real rays
#: from the thresholds grows, before it falls, by exp(E) with E = (t_j - rho t_i)^2 /
#: (4 (1 - rho^2)) (t the thresholds in standard deviations, the clear side up; t_j^2 / 4
#: for a node alone): where the next edge lies far outside the bridge's spread from the
#: one before; along the chain, by the product of its steps'. The row takes a frequency
#: in full where the sum of its chain's E is at most _GROWTH_NEAR, and none from
#: _GROWTH_FAR, where the quadrature in double precision has lost its digits (exp(20)
#: is 5e8); and in full where the quadrature takes at most _MOST _GROWTH_NEAR /
#: _GROWTH_FAR points, none where it would take more than _MOST (:meth:`Row.growth`).
_GROWTH_NEAR = 10.0
_GROWTH_FAR = 20.0
#: Edges within this angle of a direction lie along it or at right angles to it.
_PARALLEL = math.radians(1.0)
#: An edge crosses the way where the sine of its angle to the way is at least this.
_CROSSING = 0.25
#: An edge's obstacle stands across the way where the direction from the edge into it
#: has at least this share along the direction of the edge's constraint.
_ACROSS = 0.5


def bridge_chain(
    thresholds: np.ndarray, correlations: np.ndarray, clear: np.ndarray, blocked: np.ndarray
) -> np.ndarray:
    """E[prod_j w_j(Y_j)] over a chain Y_1..Y_n of unit Gaussians, each correlated by
    ``correlations`` (n - 1,) with the next, where w_j is ``clear[j]`` for Y_j above
    ``thresholds[j]`` and ``blocked[j]`` below it: (F,) for the thresholds and weights
    (n, F) at F frequencies (complex: the integral is continued analytically). With
    ``clear`` 1 and ``blocked`` 0 it is the orthant probability P(Y_j > h_j for all j);
    with both 1, 1.

    A node with one weight on both sides is that weight and leaves the chain at that
    frequency, the bridge's nodes either side of it correlated by the product of the
    correlations across it (the chain is Markov). The integral of the others runs
    along the real direction from their thresholds, where the Gaussian along one of the
    two rays grows as exp(|h| u / sqrt 2) before it falls: within _REACH for thresholds
    of a few standard deviations, as those of the row's nodes are (:func:`membership`),
    but not for an edge far from the way, which is in the row by none of its share.
    Nodes across a step narrower than the quadrature resolves are taken as one
    (:func:`_chain`)."""
    return _bridge(thresholds, correlations, clear, blocked, True)


def _bridge(
    thresholds: np.ndarray,
    correlations: np.ndarray,
    clear: np.ndarray,
    blocked: np.ndarray,
    merge: bool,
) -> np.ndarray:
    """:func:`bridge_chain`, with the nodes across narrow steps taken as one (``merge``,
    :func:`_chain`) or each by the quadrature however narrow its steps."""
    h = np.asarray(thresholds, dtype=complex)
    count, frequencies = h.shape
    out = np.ones(frequencies, dtype=complex)
    if count == 0:
        return out
    rho = np.asarray(correlations, dtype=float)
    clear = np.broadcast_to(np.asarray(clear, dtype=complex), h.shape)
    blocked = np.broadcast_to(np.asarray(blocked, dtype=complex), h.shape)
    even = clear == blocked
    out = out * np.prod(np.where(even, clear, 1.0), axis=0)
    patterns, which = np.unique(~even.T, axis=0, return_inverse=True)
    for pattern, chain in zip(patterns, range(len(patterns)), strict=True):
        columns = np.flatnonzero((which.reshape(-1) == chain) & (out != 0.0))
        nodes = np.flatnonzero(pattern)
        if len(columns) and len(nodes):
            link = np.array([np.prod(rho[a:b]) for a, b in itertools.pairwise(nodes)])
            out[columns] *= (_chain if merge else _quadrature)(
                h[np.ix_(nodes, columns)],
                link,
                clear[np.ix_(nodes, columns)],
                blocked[np.ix_(nodes, columns)],
            )
    return out


def _chain(h: np.ndarray, rho: np.ndarray, clear: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    """:func:`bridge_chain` for nodes that each weigh the two sides of their thresholds
    differently: by :func:`_quadrature` where every step is at least _APART wide; the
    nodes across the steps narrower than that taken as one (:func:`_merged`) below
    _CLOSE, and between the two by half a cosine's period in the step's width (in the
    product of those over the chain's steps), the quadrature's part also taken down by
    :func:`_fade` of its growth and its points over those steps, as :meth:`Row.growth`
    has them."""
    width = np.sqrt(np.maximum(1.0 - rho**2, 0.0))
    narrow = width < _APART
    if not narrow.any():
        return _quadrature(h, rho, clear, blocked)
    x = np.clip((width[narrow] - _CLOSE) / (_APART - _CLOSE), 0.0, 1.0)
    apart = float(np.prod(0.5 * (1.0 - np.cos(np.pi * x))))
    one = _merged(h, rho, clear, blocked, narrow)
    if apart == 0.0:
        return one
    close = width < _CLOSE
    t = np.abs(h) * np.sign(h.real)
    growth, points = _chain_cost(t, rho, np.ones(t.shape, dtype=bool), _CLOSE)
    share = apart * _fade(np.maximum(growth, _GROWTH_FAR * points / _MOST))
    if close.any():
        two = _merged(h, rho, clear, blocked, close)
    else:
        two = _quadrature(h, rho, clear, blocked)
    return share * two + (1.0 - share) * one


def _merged(
    h: np.ndarray, rho: np.ndarray, clear: np.ndarray, blocked: np.ndarray, narrow: np.ndarray
) -> np.ndarray:
    """:func:`bridge_chain` with the nodes across each of the steps ``narrow`` (n - 1,)
    taken as one, at the first of them, the bridge at the others where it is there (the
    sign of the correlations across, Y_j = +-Y_i): the product of their weights, a weight
    that steps at each of their thresholds (in the order of the thresholds' real
    coefficient of exp(i pi / 4), the phase they share), which is its value below them
    all and its step at each (a node of weights 1 and 0) in turn; the chain over the
    others and those (each by the quadrature, :func:`_bridge`), summed."""
    count = len(h)
    # The runs of nodes across narrow steps: each from its first node.
    runs, start = [], 0
    for j in range(1, count + 1):
        if j == count or not narrow[j - 1]:
            runs.append(list(range(start, j)))
            start = j
    options = []
    for run in runs:
        if len(run) == 1:
            i = run[0]
            options.append([(np.ones(h.shape[1], dtype=complex), i, h[i], clear[i], blocked[i])])
            continue
        # Each node's weight at the first's place: above and below its threshold there.
        sign = np.cumprod(np.concatenate([[1.0], np.sign(rho[run[0] : run[-1]])]))
        at = sign[:, None] * h[run]
        above = np.where(sign[:, None] > 0.0, clear[run], blocked[run])
        below = np.where(sign[:, None] > 0.0, blocked[run], clear[run])
        order = np.argsort((at * np.exp(-0.25j * np.pi)).real, axis=0, kind="stable")
        rank = np.argsort(order, axis=0, kind="stable")
        value = [
            np.prod(np.where(rank < passed, above, below), axis=0) for passed in range(len(run) + 1)
        ]
        steps = np.take_along_axis(at, order, axis=0)
        ones, nothing = np.ones(h.shape[1], dtype=complex), np.zeros(h.shape[1], dtype=complex)
        option = [(value[0], -1, None, None, None)]
        for passed in range(1, len(run) + 1):
            step = value[passed] - value[passed - 1]
            option.append((step, run[0], steps[passed - 1], ones, nothing))
        options.append(option)
    out = np.zeros(h.shape[1], dtype=complex)
    for choice in itertools.product(*options):
        coefficient = np.prod([c[0] for c in choice], axis=0)
        kept = [c for c in choice if c[1] >= 0]
        if not kept:
            out += coefficient
            continue
        places = [c[1] for c in kept]
        link = np.array([np.prod(rho[a:b]) for a, b in itertools.pairwise(places)])
        out += coefficient * _bridge(
            np.array([c[2] for c in kept]),
            link,
            np.array([c[3] for c in kept]),
            np.array([c[4] for c in kept]),
            False,
        )
    return out


def _quadrature(
    h: np.ndarray, rho: np.ndarray, clear: np.ndarray, blocked: np.ndarray
) -> np.ndarray:
    """:func:`bridge_chain` by Gauss-Legendre quadrature, for nodes that each weigh the
    two sides of their thresholds differently."""
    count = len(h)
    points = int(min(max(_FEWEST, math.ceil(_points(h, rho))), _MOST))
    x, w = np.polynomial.legendre.leggauss(points)
    u, du = 0.5 * _REACH * (x + 1.0), 0.5 * _REACH * w

    def nodes(j: int) -> tuple[np.ndarray, np.ndarray]:
        # (F, 2 points): the ray up from the threshold, then the one down from it.
        y = np.concatenate([h[j][:, None] + u, h[j][:, None] - u], axis=1)
        weight = np.concatenate([clear[j][:, None] * du, blocked[j][:, None] * du], axis=1)
        return y, weight

    y, weight = nodes(0)
    density = np.exp(-0.5 * y**2) / math.sqrt(2.0 * math.pi) * weight
    for j in range(1, count):
        following, weight = nodes(j)
        r = float(rho[j - 1])
        spread = 1.0 - r * r
        step = following[:, None, :] - r * y[:, :, None]
        kernel = np.exp(-0.5 * step**2 / spread) / math.sqrt(2.0 * math.pi * spread)
        density = products.product(density[:, None, :], kernel)[:, 0, :] * weight
        y = following
    return density.sum(axis=1)


def _chain_cost(t: np.ndarray, rho: np.ndarray, active: np.ndarray, merged: float = 0.0) -> tuple:
    """For the chain of the nodes ``active`` (n, F) of thresholds ``t`` (n, F, in
    standard deviations, the clear side up) and correlations ``rho`` (n - 1,), from one
    active node to the next by the product of the correlations between: (F,) the
    exponent by which its integrand along the real rays grows, the growths E of its
    steps summed (the first node's density alone, then each kernel from one node to
    the next), and (F,) the Gauss-Legendre points its quadrature takes (_PER_WIDTH,
    _TURN): over the width of the narrowest kernel, and over the fastest turn of a
    node's integrand along its ray, by the step into it (sqrt(2 E) radians over the
    kernel's width, the first node's |t| / sqrt 2 over its density's) and the kernel
    out of it in proportion to the correlation; none where no node is active. A node a
    step narrower than ``merged`` from the active node before is one with it
    (:func:`_merged`): it adds no step, and the next step is from that node."""
    count, frequencies = t.shape
    growth, fastest = np.zeros(frequencies), np.zeros(frequencies)
    narrowest = np.ones(frequencies)
    last, link, into = np.zeros(frequencies), np.ones(frequencies), np.zeros(frequencies)
    seen = np.zeros(frequencies, dtype=bool)
    for j in range(count):
        spread = np.maximum(1.0 - link**2, 1e-12)
        width = np.sqrt(spread)
        joined = active[j] & seen & (width < merged)
        on = active[j] & ~joined
        offset = np.abs(t[j] - link * last)
        step = np.where(seen, offset**2 / (4.0 * spread), t[j] ** 2 / 4.0)
        turn = np.where(seen, offset / (math.sqrt(2.0) * spread), np.abs(t[j]) / math.sqrt(2.0))
        # The node before turns with the kernel out of it too.
        fastest = np.where(on & seen, np.maximum(fastest, into + np.abs(link) * turn), fastest)
        narrowest = np.where(on & seen, np.minimum(narrowest, width), narrowest)
        growth = growth + np.where(on, step, 0.0)
        into = np.where(on, turn, into)
        last = np.where(on, t[j], last)
        seen = seen | on
        if j < count - 1:
            link = np.where(on | ~seen, 1.0, link) * rho[j]
    fastest = np.maximum(fastest, into)
    points = np.where(seen, _PER_WIDTH * (1.0 / narrowest + fastest / _TURN), 0.0)
    return growth, points


def _points(h: np.ndarray, rho: np.ndarray) -> float:
    """The Gauss-Legendre points the quadrature of the chain of thresholds ``h`` (n, F)
    and correlations ``rho`` (n - 1,) takes (:func:`_chain_cost`), at the frequency that
    takes the most."""
    t = np.abs(h) * np.sign(h.real)
    return float(_chain_cost(t, rho, np.ones(t.shape, dtype=bool))[1].max())


def correlations(along: np.ndarray, length: float) -> np.ndarray:
    """(n - 1,) the correlations of the bridge from the source to the listener ``length``
    apart between the points ``along`` (n,) the way from the source (ascending), each with
    the next."""
    t = np.asarray(along, dtype=float) / length
    return np.sqrt(t[:-1] * (1.0 - t[1:]) / ((1.0 - t[:-1]) * t[1:]))


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.einsum("i,i->", a, b))


def membership(distance: np.ndarray) -> np.ndarray:
    """The share an edge has in the row where the way passes it ``distance`` standard
    deviations of the bridge away: 1 up to :data:`ROW_NEAR`, 0 from :data:`ROW_FAR`,
    and half a cosine's period between."""
    x = np.clip((np.asarray(distance, dtype=float) - ROW_NEAR) / (ROW_FAR - ROW_NEAR), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _fade(growth: np.ndarray) -> np.ndarray:
    """The share of a frequency the row takes at its chains' ``growth``
    (:meth:`Row.growth`): 1 up to _GROWTH_NEAR, 0 from _GROWTH_FAR, and half a cosine's
    period between."""
    x = np.clip((growth - _GROWTH_NEAR) / (_GROWTH_FAR - _GROWTH_NEAR), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


@dataclass
class Row:
    """Edges in a row along the way from the source to the listener, ``length`` long
    (:func:`find_row`), at F frequencies. Node i is edge ``edge[i]``, ``along[i]`` from
    the source, of ``family[i]`` (0: its constraint is along q, 1: along p); its line is
    at ``level[i]`` along that direction, the way at ``way[family]`` (the source's and
    the listener's coordinates), and the bridge passes it on the side ``clear[i]``
    (+1: above its level, -1: below). ``point[i]`` is the way's point nearest the edge.

    Which edges are in the row is uncertain. The row holds by ``hold`` (F,), and where it
    does not, no node is in it: the field is then the uniform theory of diffraction's
    over every edge, so that where the row's share of the field is taken down at once
    (:meth:`computable`) the field goes over from the row's to that one, each whole.
    Where the row holds, each node of ``share`` (n, F) is in it by that share, alone of
    the others; the two nodes of each of ``corners`` (the top and the end of one screen,
    in families 0 and 1) are in it together by the states (none, the first, the second,
    both; each (F,)). The field over the edges is that of the uniform theory of
    diffraction over those not in the row, the row's paraxial field over those in it
    where they are two or more: :meth:`weight` is the share of a term over some edges
    that the row takes, and :meth:`branches` the row's field.

    ``tops`` (front node, back node, level of the face's plane along q, the face's
    reflection factor (F,)) are the rigid flat tops between two nodes of family 0, in
    the row where both edges are."""

    edge: np.ndarray
    along: np.ndarray
    family: np.ndarray
    level: np.ndarray
    clear: np.ndarray
    point: np.ndarray
    way: np.ndarray
    length: float
    share: np.ndarray
    hold: np.ndarray
    corners: list[tuple[int, int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]
    tops: list[tuple[int, int, float, np.ndarray]]
    #: The model's plane of each of ``tops``, and the side of the face that is the top
    #: at each of its nodes.
    top_planes: list[int]
    top_sides: dict[int, int]
    #: The source and the unit vector along the way.
    origin: np.ndarray
    direction: np.ndarray
    #: (2, 3) the directions q and p; (n, F) each node's share by its ends alone (in
    #: full but near an end, :func:`find_row`); (F,) the wavenumbers.
    axes: np.ndarray
    ends: np.ndarray
    wavenumber: np.ndarray
    #: Whether a node the bridge passes beyond ROW_FAR standard deviations away weighs
    #: by the side it lies on alone (its geometric limit, which the chain's integrals
    #: do not reach, :func:`bridge_chain`), and one between ROW_NEAR and ROW_FAR by its
    #: two sides in part (:func:`membership`): the nodes of a leg (:meth:`past`), each
    #: in it by its share in the row, which the leg may pass far from.
    limit_far: bool = False

    def __post_init__(self) -> None:
        cornered = {i for first, second, _ in self.corners for i in (first, second)}
        units: list[dict[frozenset, np.ndarray]] = []
        for i in range(len(self.edge)):
            if i not in cornered:
                units.append({frozenset(): 1.0 - self.share[i], frozenset([i]): self.share[i]})
        for first, second, (none, one, other, both) in self.corners:
            units.append(
                {
                    frozenset(): none,
                    frozenset([first]): one,
                    frozenset([second]): other,
                    frozenset([first, second]): both,
                }
            )
        self._units = units
        self._cornered = cornered
        self._node = {int(e): i for i, e in enumerate(self.edge)}
        partner = {}
        for front, back, _, _ in self.tops:
            partner[front], partner[back] = back, front
        self._partner = partner
        self._together = {frozenset([first, second]) for first, second, _ in self.corners}
        # The obstacles: a screen's corner, a block's top, and each other node alone.
        groups = [frozenset([first, second]) for first, second, _ in self.corners]
        groups += [frozenset([front, back]) for front, back, _, _ in self.tops]
        grouped = frozenset().union(*groups)
        groups += [frozenset([i]) for i in range(len(self.edge)) if i not in grouped]
        self._groups = groups

    @property
    def frequencies(self) -> int:
        return self.share.shape[1]

    def node(self, edge: int) -> int:
        """The node of model edge ``edge``, -1 if it is not in the row."""
        return self._node.get(int(edge), -1)

    def partner(self, node: int) -> int:
        """The node across the rigid top of ``node``, -1 if it has none."""
        return self._partner.get(int(node), -1)

    def together(self, i: int, j: int) -> bool:
        """Whether nodes ``i`` and ``j`` are the top and the end of one screen's corner."""
        return frozenset([int(i), int(j)]) in self._together

    def top_of(self, plane: int, point: np.ndarray) -> int:
        """The top of the row in ``plane`` (a model plane) whose stretch along the way
        is nearest ``point`` (a reflection off the plane), -1 if no top lies in it."""
        x = float(np.einsum("i,i->", np.asarray(point, dtype=float) - self.origin, self.direction))
        best, apart = -1, math.inf
        for q, ((front, back, _, _), own) in enumerate(
            zip(self.tops, self.top_planes, strict=True)
        ):
            if own == plane:
                gap = max(self.along[front] - x, x - self.along[back], 0.0)
                if gap < apart:
                    best, apart = q, gap
        return best

    def members(self, edge: int, wave: int, planes, points) -> frozenset | None:
        """The nodes whose rows hold the term of ``wave`` (0: the incident wave, 1 and 2:
        the reflections off face 0 and face n, as the edges' paths have them) of model
        edge ``edge`` on a path along the way that reflects off ``planes`` at ``points``
        in turn (-1 for no more): the two of a top for the reflection off that top, its
        node for the incident wave and the reflection off any other face (which the
        row's field has from the edge on its own, :meth:`past`), and those of the tops
        the path reflects off. None where the term is not one of the row's field: over
        an edge not in the row, or on a path that reflects off a plane without a top of
        the row."""
        node = self.node(edge)
        if node < 0:
            return None
        if wave > 0 and self.partner(node) >= 0 and wave == 1 + self.top_sides[node]:
            out = {node, self.partner(node)}
        else:
            out = {node}
        for plane, point in zip(planes, points, strict=False):
            if plane < 0:
                break
            q = self.top_of(int(plane), point)
            if q < 0:
                return None
            out |= {self.tops[q][0], self.tops[q][1]}
        return frozenset(out)

    def contains(self, members) -> np.ndarray:
        """(F,) the probability that the nodes ``members`` are all in the row."""
        members = frozenset(members)
        if not members:
            return np.ones(self.frequencies)
        return self.hold * self._contains(members)

    def _contains(self, members: frozenset) -> np.ndarray:
        """(F,) the probability that the nodes ``members`` are all in the row where it
        holds."""
        out = np.ones(self.frequencies)
        for unit in self._units:
            mine = members & frozenset().union(*unit)
            if mine:
                out = out * sum(p for state, p in unit.items() if mine <= state)
        return out

    def _empty(self, group: frozenset = frozenset()) -> np.ndarray:
        """(F,) the probability that no node but those of ``group`` is in the row where it
        holds."""
        out = np.ones(self.frequencies)
        for unit in self._units:
            if not frozenset().union(*unit) & group:
                out = out * unit[frozenset()]
        return out

    def _within(self, group: frozenset, members: frozenset) -> np.ndarray:
        """(F,) the probability that the nodes ``members`` are in the row and none but
        those of ``group``, where it holds."""
        out = self._empty(group)
        for unit in self._units:
            mine = frozenset().union(*unit)
            if mine & group:
                out = out * sum(p for state, p in unit.items() if members & mine <= state)
        return out

    def weight(self, members) -> np.ndarray:
        """(F,) the share of a term over the nodes ``members`` (none: the way's own
        wave) that the row's field stands for: the probability that the row holds, that
        they are in it and that its nodes are of two obstacles or more (the paths over
        the edges of one obstacle, a block's top or a screen's corner, are those of the
        uniform theory of diffraction)."""
        members = frozenset(members)
        out = self._contains(members)
        if not members:
            out = out - self._empty()
        for group in self._groups:
            if members <= group:
                within = self._within(group, members)
                out = out - (within - self._empty() if not members else within)
        return self.hold * out

    def _steps(self, family: int, source: float, items: list, scale: np.ndarray) -> tuple:
        """The chain over ``items`` (node, level, clear side, weight on the clear side, on
        the blocked side) of ``family``, the bridge from ``source`` (that family's
        coordinate of the source or its image) to the listener, in the terms of
        :func:`bridge_chain`: its thresholds (n, F), correlations (n - 1,) and weights
        on the clear and the blocked side (n, F), in order along the way."""
        items = sorted(items, key=lambda item: self.along[item[0]])
        nodes = np.array([item[0] for item in items])
        listener = self.way[family, 1]
        mean = source + self.along[nodes] / self.length * (listener - source)
        level = np.array([item[1] for item in items])
        sign = np.array([item[2] for item in items], dtype=float)
        thresholds = (sign * (level - mean))[:, None] * scale[nodes]
        rho = sign[:-1] * sign[1:] * correlations(self.along[nodes], self.length)
        weights = [
            np.array([np.broadcast_to(item[c], (self.frequencies,)) for item in items])
            for c in (3, 4)
        ]
        if self.limit_far:
            # The side the bridge's mean lies on, towards which a far node weighs.
            limit = np.where(thresholds.real < 0.0, weights[0], weights[1])
            near = membership(np.abs(thresholds))
            weights = [limit + near * (weight - limit) for weight in weights]
        return thresholds, rho, *weights

    def _layouts(self, k: np.ndarray):
        """For each set of tops the bridge reflects off: the set (their indices in order
        along the way), the distance from the source's image in their planes in turn to
        the listener, the set's weight (F,) and the image's spreading and delay relative
        to those of the straight line from the source (F,), and the chains whose sum is
        the field: (coefficient, the steps of family 0, those of family 1,
        :meth:`_steps`)."""
        count = len(self.edge)
        frequencies = self.frequencies
        across = self.along * (self.length - self.along) / self.length
        scale = np.sqrt(np.asarray(k, dtype=float)[None, :] / across[:, None]) * np.exp(
            0.25j * np.pi
        )
        order = np.argsort(-self.along, kind="stable")
        for flips in itertools.product((False, True), repeat=len(self.tops)):
            chosen = [top for top, on in zip(self.tops, flips, strict=True) if on]
            weight = np.ones(frequencies, dtype=complex)
            hard = set()
            for front, back, _, reflection in chosen:
                weight = weight * self.share[front] * self.share[back] * reflection
                hard |= {front, back}
            # The mirror each node of family 0 is seen in from the listener's side: the
            # planes of the tops after it (and its own, at a top's front edge).
            fronts = {front: level for front, _, level, _ in chosen}
            sign, shift = np.ones(count), np.zeros(count)
            mirror, offset = 1.0, 0.0
            for i in order:
                if i in fronts:
                    plane = mirror * fronts[i] + offset
                    mirror, offset = -mirror, 2.0 * plane - offset
                sign[i], shift[i] = mirror, offset
            source = mirror * self.way[0, 0] + offset
            # The straight line's and the image's distances to the listener.
            square = self.length**2 + (self.way[1, 1] - self.way[1, 0]) ** 2
            direct = math.sqrt(square + (self.way[0, 1] - self.way[0, 0]) ** 2)
            image = math.sqrt(square + (self.way[0, 1] - source) ** 2)
            ratio = direct / image * np.exp(-1j * np.asarray(k) * (image - direct))

            def item(i: int, on_clear, on_blocked, sign=sign, shift=shift) -> tuple:
                if self.family[i] == 0:
                    level = sign[i] * self.level[i] + shift[i]
                    return (i, level, sign[i] * self.clear[i], on_clear, on_blocked)
                return (i, self.level[i], self.clear[i], on_clear, on_blocked)

            chains = []
            for split in itertools.product((False, True), repeat=len(self.corners)):
                coefficient = 1.0
                items: list[list] = [[], []]
                for i in range(count):
                    if i not in self._cornered:
                        blocked = 0.0 if i in hard else 1.0 - self.share[i]
                        items[self.family[i]].append(item(i, 1.0, blocked))
                # A screen's top and end in the row together stop the bridge on the
                # screen's side of both: 1 - [first] p_first - [second] p_second
                # - [first][second] p_both, the brackets 1 on the blocked side, which is
                # (1 - [second] p_second) - [first] (p_first + [second] p_both).
                for (first, second, (_, one, other, both)), part in zip(
                    self.corners, split, strict=True
                ):
                    if part:
                        coefficient = -coefficient
                        items[0].append(item(first, 0.0, 1.0))
                        items[1].append(item(second, one, one + both))
                    else:
                        items[1].append(item(second, 1.0, 1.0 - other))
                chains.append(
                    (
                        coefficient,
                        self._steps(0, source, items[0], scale) if items[0] else None,
                        self._steps(1, self.way[1, 0], items[1], scale) if items[1] else None,
                    )
                )
            yield tuple(q for q, on in enumerate(flips) if on), image, weight, ratio, chains

    def growth(self) -> np.ndarray:
        """(F,) the largest, over the row's chains at the frequencies where their set of
        tops has a share and the chain is not none, of the exponent by which the
        integrand of :func:`bridge_chain` grows along a chain of the nodes that weigh
        their two sides differently (_GROWTH_NEAR, :func:`_chain_cost`), and of
        _GROWTH_FAR in proportion to the points its quadrature takes over _MOST."""
        out = np.zeros(self.frequencies)
        for _, _, weight, _, chains in self._layouts(self.wavenumber):
            for _, *families in chains:
                steps_of = [steps for steps in families if steps is not None]
                # A chain with a node that weighs both sides by none is none there.
                some = weight != 0.0
                for _, _, clear, blocked in steps_of:
                    some &= ~((clear == 0.0) & (blocked == 0.0)).any(axis=0)
                for thresholds, rho, clear, blocked in steps_of:
                    t = np.abs(thresholds) * np.sign(thresholds.real)
                    active = (clear != blocked) & some
                    growth, points = _chain_cost(t, rho, active, _APART)
                    out = np.maximum(out, np.maximum(growth, _GROWTH_FAR * points / _MOST))
        return out

    def computable(self) -> Row:
        """The row with its hold at each frequency taken down by :func:`_fade` of its
        :meth:`growth`: where its chains cannot be computed, the field goes over to the
        uniform theory of diffraction's over every edge."""
        fade = _fade(self.growth())
        if (fade == 1.0).all():
            return self
        return dataclasses.replace(self, hold=self.hold * fade)

    def _part(self, inside: np.ndarray, start, end, own: bool) -> Row:
        """The row of the nodes ``inside`` (ascending) alone for the bridge from ``start``
        to ``end`` (``start`` first along the way), each node in it by its share in this
        row where it holds (``own``: the part holds in full), or by how close the straight
        line between the two passes it (:func:`membership`, by its ends as
        :func:`find_row` has them, the part holding by the largest of those shares,
        :func:`_held`)."""
        index = {int(old): new for new, old in enumerate(inside)}
        a = float(np.einsum("i,i->", start - self.origin, self.direction))
        b = float(np.einsum("i,i->", end - self.origin, self.direction))
        length = b - a
        along = self.along[inside] - a
        way = np.array(
            [
                [float(np.einsum("i,i->", point, axis)) for point in (start, end)]
                for axis in self.axes
            ]
        )
        family = self.family[inside]
        level = self.level[inside]
        clear = self.clear[inside]
        mean = way[family, 0] + along / length * (way[family, 1] - way[family, 0])
        if own:
            share, hold = self.share[inside], np.ones(self.frequencies)
        else:
            across = (along * (length - along) / length)[:, None]
            sigma = np.sqrt(across / self.wavenumber[None, :])
            hold, share = _held(
                membership(np.abs(level - mean)[:, None] / sigma) * self.ends[inside]
            )
        corners = []
        for first, second, states in self.corners:
            if first in index and second in index:
                f, g = index[first], index[second]
                if not own:
                    beside = clear[f] * (mean[f] - level[f]) > 0.0
                    aside = clear[g] * (mean[g] - level[g]) > 0.0
                    states = _corner_states(share[f], share[g], beside, aside)
                corners.append((f, g, states))
        tops, top_planes, top_sides = [], [], {}
        for (front, back, height, reflection), plane in zip(
            self.tops, self.top_planes, strict=True
        ):
            if front in index and back in index:
                tops.append((index[front], index[back], height, reflection))
                top_planes.append(plane)
                top_sides[index[front]] = self.top_sides[front]
                top_sides[index[back]] = self.top_sides[back]
        part = Row(
            edge=self.edge[inside],
            along=along,
            family=family,
            level=level,
            clear=clear,
            point=self.point[inside],
            way=way,
            length=length,
            share=share,
            hold=hold,
            corners=corners,
            tops=tops,
            top_planes=top_planes,
            top_sides=top_sides,
            origin=start,
            direction=self.direction,
            axes=self.axes,
            ends=self.ends[inside],
            wavenumber=self.wavenumber,
        )
        return part if own else part.computable()

    def passing(self, legs) -> np.ndarray:
        """(F,) the field at the end of each of ``legs`` (pairs of points in turn, the
        path's) of a point source at its start relative to the free field, past the nodes
        of the row between the two along the way (not those at either), over every row
        of them (:meth:`branches`), each node in the row by how close the straight line
        between the two passes it (:func:`membership`, by its ends as :func:`find_row`
        has them), multiplied along the legs. The legs of one path pass one row: where it
        holds on one leg it does on each that holds by as much or more (the row on each
        leg holds by the largest of its nodes' shares there, :func:`_held`)."""
        holds, fields = [], []
        for start, end in legs:
            hold, field = self._passing(
                np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            )
            holds.append(hold)
            fields.append(field)
        holds, fields = np.array(holds), np.array(fields)
        order = np.argsort(-holds, axis=0, kind="stable")
        holds = np.take_along_axis(holds, order, axis=0)
        fields = np.take_along_axis(fields, order, axis=0)
        ones, nothing = np.ones((1, self.frequencies)), np.zeros((1, self.frequencies))
        # Where the row holds on the first k legs (by its hold on them) and on no other.
        share = np.concatenate([ones, holds]) - np.concatenate([holds, nothing])
        field = np.concatenate([ones.astype(complex), np.cumprod(fields, axis=0)])
        return np.sum(share * field, axis=0)

    def _passing(self, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(F,) how much the row of the nodes between ``start`` and ``end`` holds for the
        bridge between them, and (F,) its field there where it holds (:meth:`passing`)."""
        a = float(np.einsum("i,i->", start - self.origin, self.direction))
        b = float(np.einsum("i,i->", end - self.origin, self.direction))
        if b < a:
            start, end, a, b = end, start, b, a
        margin = 1e-6 * self.length
        inside = np.flatnonzero((self.along > a + margin) & (self.along < b - margin))
        ones = np.ones(self.frequencies)
        if len(inside) == 0 or b - a <= margin:
            return ones, ones.astype(complex)
        part = self._part(inside, start, end, own=False)
        held = dataclasses.replace(part, hold=ones)
        fields = [field for _, _, field in held.branches(self.wavenumber, every=True)]
        return part.hold, np.sum(fields, axis=0)

    def _leg(self, inside: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """(F,) the field at ``end`` of a point source at ``start`` relative to the free
        field past the nodes ``inside`` (ascending, between the two), each in the row by
        its share in this row, those the straight line passes far from at their
        geometric limit (``limit_far``), over every row of them (:meth:`branches`)."""
        if len(inside) == 0:
            return np.ones(self.frequencies, dtype=complex)
        part = self._part(inside, start, end, own=True)
        part = dataclasses.replace(part, limit_far=True).computable()
        return np.sum([field for _, _, field in part.branches(self.wavenumber, every=True)], axis=0)

    def past(self, node: int, point: np.ndarray) -> np.ndarray:
        """(F,) the share the row's field stands for of a wave that the edge of ``node``
        sends from ``point`` on it alone, with the legs from the source to ``point`` and
        on to the listener past the row's other nodes (but the other edge of a screen's
        corner with it, which stands at ``point``): over the rows that hold the node and
        are of two obstacles or more (as :meth:`weight` has them), in proportion to
        their probability, the field of each along the two legs (the row holding, as it
        does where the node is in it)."""
        held = self.contains([node])
        if not held.any():
            return np.zeros(self.frequencies, dtype=complex)
        point = np.asarray(point, dtype=float)
        x, margin = self.along[node], 1e-6 * self.length
        group = next(group for group in self._groups if node in group)
        others = np.array(
            [i for i in range(len(self.edge)) if i != node and not self.together(i, node)],
            dtype=np.int64,
        )
        end = self.origin + self.length * self.direction
        legs, alone = [], []
        for inside, start, stop in (
            (others[self.along[others] < x - margin], self.origin, point),
            (others[self.along[others] > x + margin], point, end),
        ):
            legs.append(self._leg(inside, start, stop))
            alone.append(self._leg(inside[np.isin(inside, list(group))], start, stop))
        within = self._empty(group) * alone[0] * alone[1]
        return held * (legs[0] * legs[1] - within)

    def branches(
        self, k: np.ndarray, every: bool = False
    ) -> list[tuple[tuple[int, ...], float, np.ndarray]]:
        """The row's field relative to the free field at wavenumbers ``k`` (F,), over the
        rows of two nodes or more in proportion to their probability (with ``every``,
        over every row, of any number of nodes, and the free field where the row does not
        hold), by the tops the bridge reflects off: for
        each set of ``tops`` (their indices, in order along the way), the distance from
        the source's image in their planes in turn to the listener and (F,) that part of
        the field, the image's spreading and delay relative to the straight line's
        included. Where a set of tops has no share (their edges are in no row), its part
        is none: its chains, far from the bridge there, are not worked out."""
        out = []
        for flips, image, weight, ratio, chains in self._layouts(k):
            live = np.flatnonzero((weight != 0.0) & (self.hold != 0.0))
            total = np.zeros(len(live), dtype=complex)
            for coefficient, *families in chains:
                term = np.full(len(live), coefficient, dtype=complex)
                for steps in families:
                    if steps is not None:
                        thresholds, rho, clear, blocked = steps
                        term *= bridge_chain(
                            thresholds[:, live], rho, clear[:, live], blocked[:, live]
                        )
                total += term
            if not flips and not every:
                # The row of no node is the way's free field.
                total -= self._empty()[live]
            field = np.zeros(self.frequencies, dtype=complex)
            field[live] = (self.hold * weight * ratio)[live] * total
            if not flips and every:
                # Where the row does not hold, no node is in it: the free field.
                field = field + (1.0 - self.hold)
            out.append((flips, image, field))
        if not every:
            # The rows of the nodes of one obstacle are the edges' paths'.
            places = {flips: q for q, (flips, _, _) in enumerate(out)}
            for group in self._groups:
                inside = np.array(sorted(group))
                part = self._part(
                    inside, self.origin, self.origin + self.length * self.direction, True
                )
                rest = self._empty(group)
                tops = [self.tops.index(top) for top in self.tops if {top[0], top[1]} <= group]
                for flips, _, field in part.branches(k, every=True):
                    if not flips:
                        field = field - part._empty()
                    q = places[tuple(tops[f] for f in flips)]
                    out[q] = (out[q][0], out[q][1], out[q][2] - self.hold * rest * field)
        return out


def _held(share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(F,) how much a row of nodes in it by ``share`` (n, F) each holds, the largest of
    the shares at each frequency, and (n, F) their shares where it holds (each over that,
    none where it does not): a row whose nodes are all in it by less than in full holds
    by that much as a whole (its field goes over to the uniform theory of diffraction's
    over every edge), not by each node apart from the others."""
    hold = share.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(hold > 0.0, share / np.where(hold > 0.0, hold, 1.0), 0.0)
    return hold, relative


def _corner_states(a: np.ndarray, b: np.ndarray, beside: bool, aside: bool) -> tuple:
    """The states (none, the first, the second, both) of a screen's top and end in the
    row, ``a`` and ``b`` their shares, the way on the open side of the first
    (``beside``) or of the second (``aside``): each alone is in the row only where the
    way passes on the screen's side of the other, as alone it would stand across the
    way where the screen does not; both by the product of their shares, so that each
    state is continuous where the way crosses either line."""
    both = a * b
    one = np.zeros_like(a) if aside else a * (1.0 - b)
    other = np.zeros_like(a) if beside else (1.0 - a) * b
    return (1.0 - both - one - other, one, other, both)


def _directions(edges, model, u: np.ndarray, nearest: int, candidates: np.ndarray) -> tuple:
    """The row's directions q and p for the way along ``u`` with ``nearest`` its reference
    edge, and each edge's family, the direction of its constraint and the share of its
    obstacle along that (:func:`find_row`): q at right angles to the way and to the
    edge, or the normal of a face along the way (its sine to the way at most
    _CROSSING, a block's top) of the first of ``candidates`` (by distance) parallel to
    the edge that has one."""
    d = edges.direction
    q = np.cross(u, d[nearest])
    q /= math.sqrt(_dot(q, q))
    for e in candidates:
        if edges.wedge[e] >= 2.0 - 1e-9 or abs(_dot(d[e], q)) > math.sin(_PARALLEL):
            continue
        normals = model.plane_normal[edges.plane[e]]
        along_way = np.abs(np.einsum("ij,j->i", normals, u)) <= _CROSSING
        if along_way.any():
            normal = normals[np.flatnonzero(along_way)[0]]
            q = normal * np.sign(_dot(normal, q))
            break
    p = np.cross(u, q)
    p /= math.sqrt(_dot(p, p))
    lies = np.abs(np.einsum("ij,j->i", d, q))
    family = np.full(len(edges), -1, dtype=np.int64)
    family[lies <= math.sin(_PARALLEL)] = 0
    family[(lies >= math.cos(_PARALLEL)) & np.isclose(edges.wedge, 2.0)] = 1
    constraint = np.where((family == 0)[:, None], q[None, :], p[None, :])
    stands = np.einsum("ij,ij->i", -edges.bisector(), constraint)
    return q, p, family, constraint, stands


def find_row(edges, model, source, listener, wavenumber, tolerance, reflection) -> Row | None:
    """The row of edges that the way from ``source`` to ``listener`` passes close by, at
    the ``wavenumber`` (F,), or None where it passes fewer than two obstacles so: the
    ``edges`` (:class:`~acoustic_spaces.engine.diffraction.Edges` of ``model``) whose
    line crosses the way between the two at more than :data:`_CROSSING` (the sine of its
    angle to the way), within :data:`ROW_FAR` standard deviations of the bridge at the
    lowest wavenumber, and whose obstacle stands across the way. ``reflection(edge,
    side)`` is the (F,) reflection factor of face ``side`` of ``edge`` as the edge's
    paths have it. Where such an edge cannot be in the row (it leans across the
    families' directions, or is an edge of a solid that runs along q), two edges of one
    family are too close along the way for the chain, or two are the edges of one
    obstacle across the way (a strip's top and foot, a panel's two ends), there is no
    row either.

    The row's directions: q at right angles to the way and to the edge nearest it in
    Fresnel units (the nearest for which every such edge is of one of the families), or
    the normal of a face along the way (at most :data:`_CROSSING` its sine to it: a
    block's top) of the nearest edge parallel to that one that has such a face, and p at
    right angles to the way and to q. An edge's share (:func:`membership`)
    is that of the way's distance from it, and, near an end of it that is not the corner
    of a screen whose other edge is in the row, of the way's distance from that end too:
    the end's own paths (and the fades of the edge's) take it there, and past the end the
    edge is not in the row. A screen's top and end at its corner are in the row together
    (:class:`Row`'s corners): where the way passes on the screen's side of both, each
    alone or both; on the open side of one of them, that one is in the row only with the
    other, as alone it would stand across the way where the screen does not."""
    source, listener = np.asarray(source, dtype=float), np.asarray(listener, dtype=float)
    way = listener - source
    length = math.sqrt(_dot(way, way))
    if len(edges) < 2 or length <= tolerance:
        return None
    u = way / length
    k = np.asarray(wavenumber, dtype=float).reshape(-1)
    k_min = float(k.min())
    d = edges.direction
    cosine = np.einsum("ij,j->i", d, u)
    sine = np.sqrt(np.maximum(1.0 - cosine**2, 0.0))
    crossing = sine >= _CROSSING
    w0 = source[None, :] - edges.start
    dd = np.einsum("ij,j->i", w0, u)
    ee = np.einsum("ij,ij->i", w0, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(crossing, (cosine * ee - dd) / np.maximum(sine**2, 1e-300), -1.0)
        t = np.where(crossing, (ee - cosine * dd) / np.maximum(sine**2, 1e-300), 0.0)
    inside = crossing & (x > tolerance) & (x < length - tolerance)
    on_edge = edges.start + t[:, None] * d
    on_way = source[None, :] + x[:, None] * u[None, :]
    offset = np.linalg.norm(on_edge - on_way, axis=1)
    across = np.where(inside, x * (length - x) / length, 1.0)
    distance = offset * np.sqrt(k_min / across)
    # How far the way passes beyond each end, along the edge across the way.
    beyond = np.column_stack([-t, t - edges.length]) * sine[:, None]
    reach = ROW_FAR * np.sqrt(across / k_min)
    near = inside & (distance < ROW_FAR) & (beyond.max(axis=1) < reach)
    candidates = np.flatnonzero(near)
    if len(candidates) < 2:
        return None
    # The row's directions: those of the edge nearest the way (in Fresnel units) for
    # which every edge near the way is of one of the two families (edges equally near,
    # where the way passes a screen's corner, may set directions that leave another
    # edge in neither).
    by_distance = candidates[np.argsort(distance[candidates], kind="stable")]
    for nearest in by_distance:
        q, p, family, constraint, stands = _directions(edges, model, u, nearest, by_distance)
        if ((family[candidates] >= 0) & (np.abs(stands[candidates]) >= _ACROSS)).all():
            break
    else:
        return None
    nodes = candidates[np.argsort(x[candidates], kind="stable")]
    count = len(nodes)
    node_of = {int(e): i for i, e in enumerate(nodes)}
    fam = family[nodes]
    along = x[nodes]
    # The two edges of one obstacle across the way (a strip between its top and its
    # foot, a panel between its ends): faces in one plane across the way from each
    # towards the other. The chain takes each edge's obstacle to stand to infinity on
    # its side, which these do not: no row. Faces in one plane across the way that run
    # the same way from two edges are those of two solids that touch (a step from one
    # roof up to the next): the edge where they meet is no edge of the solid they make,
    # and the chain would take it for one: no row either.
    turn = np.pi * edges.wedge[:, None]
    facing = np.stack(
        [edges.along, np.cos(turn) * edges.along + np.sin(turn) * edges.normal], axis=1
    )  # (E, 2, 3): the direction of each face from its edge
    for i, j in itertools.combinations(range(count), 2):
        e, f = nodes[i], nodes[j]
        if fam[i] != fam[j]:
            continue
        apart = on_edge[f] - on_edge[e]
        for side in (0, 1):
            plane = edges.plane[e, side]
            sides = np.flatnonzero(edges.plane[f] == plane)
            if len(sides) == 0 or abs(_dot(model.plane_normal[plane], u)) <= _CROSSING:
                continue
            towards = _dot(facing[e, side], apart) > 0.0
            back = _dot(facing[f, sides[0]], apart) < 0.0
            if towards or back:
                return None
    level = np.einsum("ij,ij->i", on_edge[nodes], constraint[nodes])
    clear = -np.sign(stands[nodes])
    way_coordinates = np.array(
        [[_dot(source, q), _dot(listener, q)], [_dot(source, p), _dot(listener, p)]]
    )
    sigma = np.sqrt(across[nodes][:, None] / k[None, :])  # (n, F)
    mean = way_coordinates[fam, 0] + along / length * (
        way_coordinates[fam, 1] - way_coordinates[fam, 0]
    )
    own = membership(np.abs(level - mean)[:, None] / sigma)
    # The ends: a screen's corner with its other edge in the row is a corner of the row;
    # any other end within reach takes the edge out of the row near it and past it.
    corners: list[tuple[int, int]] = []
    paired: set[int] = set()
    ends = np.ones((count, k.size))
    for i, e in enumerate(nodes):
        for end in (0, 1):
            past = float(beyond[e, end])
            if past < -float(reach[e]):
                continue
            j = node_of.get(int(edges.corner[e, end]), -1)
            pair = (i, j) if fam[i] == 0 else (j, i)
            if pair in corners:
                continue
            screens = j >= 0 and np.isclose(edges.wedge[[e, nodes[j]]], 2.0).all()
            if screens and fam[j] != fam[i]:
                if not {i, j} & paired:
                    corners.append(pair)
                    paired |= {i, j}
                    continue
            if past >= 0.0:
                ends[i] = 0.0
            else:
                ends[i] *= 1.0 - membership(-past / sigma[i])
    hold, share = _held(own * ends)
    units = []
    for first, second in corners:
        beside = clear[first] * (mean[first] - level[first]) > 0.0
        aside = clear[second] * (mean[second] - level[second]) > 0.0
        units.append((first, second, _corner_states(share[first], share[second], beside, aside)))
    # A block's top: faces in one plane along q from two edges of family 0 towards each
    # other.
    tops, top_planes, top_sides = [], [], {}
    for i in range(count):
        e = nodes[i]
        if fam[i] != 0 or edges.wedge[e] >= 2.0 - 1e-9 or i in top_sides:
            continue
        for side in (0, 1):
            plane = edges.plane[e, side]
            normal = model.plane_normal[plane]
            if abs(_dot(normal, q)) < math.cos(_PARALLEL) or abs(_dot(normal, u)) > _CROSSING:
                continue
            for j in range(i + 1, count):
                f = nodes[j]
                if fam[j] != 0 or j in top_sides or edges.wedge[f] >= 2.0 - 1e-9:
                    continue
                sides = np.flatnonzero(edges.plane[f] == plane)
                if len(sides) == 0:
                    continue
                apart = on_edge[f] - on_edge[e]
                if _dot(facing[e, side], apart) <= 0.0 or _dot(facing[f, sides[0]], apart) >= 0.0:
                    continue
                height = float(model.plane_offset[plane]) * _dot(normal, q)
                tops.append((i, j, height, reflection(int(e), side, on_edge[e])))
                top_planes.append(int(plane))
                top_sides[i], top_sides[j] = side, int(sides[0])
                break
            if i in top_sides:
                break
    if len({round(float(v), 9) for v in along}) < 2:
        return None
    row = Row(
        edge=nodes,
        along=along,
        family=fam,
        level=level,
        clear=clear,
        point=on_way[nodes],
        way=way_coordinates,
        length=length,
        share=share,
        hold=hold,
        corners=units,
        tops=tops,
        top_planes=top_planes,
        top_sides=top_sides,
        origin=source,
        direction=u,
        axes=np.stack([q, p]),
        ends=ends,
        wavenumber=k,
    ).computable()
    return row if row.hold.any() else None


__all__ = ["ROW_FAR", "ROW_NEAR", "Row", "bridge_chain", "correlations", "find_row", "membership"]
