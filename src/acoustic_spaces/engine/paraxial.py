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
way nearer right angles to one direction ``q`` across it than along it (the tops of
screens and blocks, whose obstacles stand on one side of them along ``q``), and the
edges of thin screens nearer along ``q`` (a screen's upright end), whose screens stand
on one side of them along the third direction ``p``. The bridge's two coordinates along
``q`` and ``p`` are independent, so the probability is the product of a chain in each.
An edge that leans from its family's direction stops the bridge's coordinate along its
own normal across the way: the chain takes its threshold at its distance from the way
along that normal, and its correlation with each other node as if it did not lean,
where the two coordinates correlate by the cosine of the angle between the nodes'
normals times that (:func:`_lean_error`). A screen whose top and end are both in the
row stops the bridge where it is on the screen's side of both: the product of the two
edges' conditions, which the two chains give as the difference of two products
(:meth:`Row._splits`). A rigid flat top between two edges of the
first family (a block's top) reflects the bridge: by Neumann's images, the field over
it from its front edge to the next node is the bridge's plus that to the node's image
in the top's plane, the plane through the two edges' lines, whether it lies level or
slopes along the way; the fields over every set of tops the bridge reflects off are
worked out together, node by node along the chain (:func:`_reflected`).

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
#: A face is a block's top where its normal lies within this angle of q, and an edge
#: is parallel to the row's reference edge (and may set q by its top) within it.
_PARALLEL = math.radians(1.0)
#: A row holds in full where taking its edges for lines of its two families changes
#: the share of the bridge past any two of them right on the way by at most _LEAN_NEAR
#: dB, and not from _LEAN_FAR dB (:func:`_lean_error`).
_LEAN_NEAR = 0.1
_LEAN_FAR = 0.5
#: A face lies along the way where the sine of its angle to the way's direction across
#: its edge (:func:`_across`) is at most this (a block's top, level or pitched: 45
#: degrees), and across it otherwise (a wall). Where a pitched top bends the way much,
#: the row hands over by its paraxial error (:func:`_paraxial_excess`).
_ALONG = math.sqrt(0.5)
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
    differently: by :func:`_quadrature` where every step is at least _APART wide, and
    with the nodes across each step narrower than that taken as one (:func:`_merged`)
    by a share of its own, all of it below _CLOSE and none at _APART, half a cosine's
    period in the step's width between. The narrower steps are taken as one first (a
    step is where a uniform number lies at or above the share it keeps apart): the
    chain is the mean, over that number, of the chains with the narrow steps taken as
    one from the narrowest on, one more at a time. Each of those but the one with every
    narrow step taken as one is taken down by :func:`_fade` of its growth and its points
    (:func:`_chain_cost`, as :meth:`Row.growth` has them) towards the next."""
    width = np.sqrt(np.maximum(1.0 - rho**2, 0.0))
    narrow = width < _APART
    if not narrow.any():
        return _quadrature(h, rho, clear, blocked)
    x = np.clip((width - _CLOSE) / (_APART - _CLOSE), 0.0, 1.0)
    apart = 0.5 * (1.0 - np.cos(np.pi * x))
    # The steps taken as one in part, the narrowest (the least kept apart) first.
    partial = np.flatnonzero(narrow & (apart > 0.0))
    partial = partial[np.argsort(apart[partial], kind="stable")]
    sets = [narrow & (apart == 0.0)]
    for step in partial:
        sets.append(sets[-1].copy())
        sets[-1][step] = True
    # Where the uniform number lies between the shares of two steps in turn.
    shares = np.diff(np.concatenate([[0.0], apart[partial], [1.0]]))
    t = np.abs(h) * np.sign(h.real)
    active = np.ones(t.shape, dtype=bool)
    value = _merged(h, rho, clear, blocked, sets[-1])
    out = shares[-1] * value
    for steps, share in zip(sets[-2::-1], shares[-2::-1], strict=True):
        if steps.any():
            own = _merged(h, rho, clear, blocked, steps)
        else:
            own = _quadrature(h, rho, clear, blocked)
        growth, points = _chain_cost(t, rho, active, join=steps)
        fade = _fade(np.maximum(growth, _GROWTH_FAR * points / _MOST))
        value = fade * own + (1.0 - fade) * value
        out = out + share * value
    return out


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


def _reflected(
    h: np.ndarray,
    rho: np.ndarray,
    clear: np.ndarray,
    blocked: np.ndarray,
    hard: tuple[np.ndarray, np.ndarray],
    tops: list[tuple[int, int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """:func:`bridge_chain` over a chain some of whose links cross rigid tops, over each
    of which the field adds, by the top's own share, that of the bridge reflected off
    it: (F,) for the thresholds and weights (n, F) and correlations (n - 1,) of
    :func:`bridge_chain`, ``hard`` the weights (on the clear side, on the blocked side;
    (n, F) each) of the nodes where a top reflects (its two edges, then in the row),
    and ``tops`` (front node, back node, the top's share (F,): the probability that
    both its edges are in the row times its face's reflection factor, and each node's
    offset from the top's plane (n, F), in the chain's terms: its sign times its scale
    times its level less the plane's there, the plane through the two edges' lines).

    By Neumann's images, over a rigid plane the paraxial field from one point to the
    next is that from the point and from its image in the plane. In the bridge's terms
    the image's kernel from node a to node b is the free one times exp(-2 rho (A_a +
    v_a) (A_b + v_b) / (1 - rho^2)), v the bridge's place from each node's threshold, A
    the node's offset from the plane (none at the top's own edges, whose plane it is,
    whether it lies level or slopes along the way) and rho the correlation from a to
    b. The bridge reflects off a top on the link from its front node to the next node,
    its two nodes taking their hard weights; the fields over every set of tops it
    reflects off are summed together, node by node (the chain is Markov), the field
    from a top's front node to its back node in two parts, with and without the
    reflection. Nodes that weigh both sides alike leave the chain as in
    :func:`bridge_chain`, the hard weight of a top's node taking the place of its weight
    in the top's part; a top whose front node has left reflects on the link of the
    nodes left on either side of it, by the plane's offsets there, and adds nothing
    where no node is left on one side. Two tops on one link reflect there each alone.
    Every step is to be at least _CLOSE wide (:func:`_chain`) where there are tops; with
    none, it is :func:`bridge_chain`'s (which takes narrow steps as it does)."""
    if not tops:
        return bridge_chain(h, rho, clear, blocked)
    h = np.asarray(h, dtype=complex)
    count, frequencies = h.shape
    out = np.ones(frequencies, dtype=complex)
    if count == 0:
        return out
    rho = np.asarray(rho, dtype=float)
    clear = np.broadcast_to(np.asarray(clear, dtype=complex), h.shape)
    blocked = np.broadcast_to(np.asarray(blocked, dtype=complex), h.shape)
    hard_clear = np.broadcast_to(np.asarray(hard[0], dtype=complex), h.shape)
    hard_blocked = np.broadcast_to(np.asarray(hard[1], dtype=complex), h.shape)
    even = clear == blocked
    alike = np.where(even, clear, 1.0)
    out = out * np.prod(alike, axis=0)
    patterns, which = np.unique(~even.T, axis=0, return_inverse=True)
    for chain, pattern in enumerate(patterns):
        columns = np.flatnonzero((which.reshape(-1) == chain) & (out != 0.0))
        nodes = np.flatnonzero(pattern)
        if len(columns) == 0 or len(nodes) == 0:
            continue
        link = np.array([np.prod(rho[a:b]) for a, b in itertools.pairwise(nodes)])
        # Each top's link (from the last node left at or before its front), whether
        # its front and back are left, and its share with the hard weights of those
        # that are not in place of their weights.
        reflections = []
        for front, back, share, offset in tops:
            start = int(np.searchsorted(nodes, front, side="right")) - 1
            if start < 0 or start + 1 >= len(nodes):
                continue
            factor = np.asarray(share, dtype=complex)[columns]
            for node in (front, back):
                if not pattern[node]:
                    same = hard_clear[node, columns] == hard_blocked[node, columns]
                    factor = np.where(
                        same, factor * hard_clear[node, columns] / alike[node, columns], 0.0
                    )
            end = int(np.searchsorted(nodes, back, side="left"))
            reflections.append(
                (
                    start,
                    bool(nodes[start] == front),
                    end,
                    bool(end < len(nodes) and nodes[end] == back),
                    factor,
                    np.asarray(offset)[np.ix_(nodes, columns)],
                )
            )
        out[columns] *= _marched(
            h[np.ix_(nodes, columns)],
            link,
            (clear[np.ix_(nodes, columns)], blocked[np.ix_(nodes, columns)]),
            (hard_clear[np.ix_(nodes, columns)], hard_blocked[np.ix_(nodes, columns)]),
            reflections,
        )
    return out


def _marched(h: np.ndarray, rho: np.ndarray, free: tuple, hard: tuple, tops: list) -> np.ndarray:
    """:func:`_reflected` for its nodes left in the chain (thresholds ``h`` (n, F),
    correlations ``rho`` (n - 1,), the weights ``free`` and ``hard``, each (clear side,
    blocked side)), by Gauss-Legendre quadrature as :func:`_quadrature` (its points as
    the chain without the tops' reflections takes them), the tops (the node their
    link starts from, whether that is the front, the first node at or after the back,
    whether that is the back, the top's share (F,) and offsets (n, F)) by the parts of
    the field in turn: each part the field with the reflections off a set of tops whose
    links have begun and whose backs have not been passed."""
    count = len(h)
    points = int(min(max(_FEWEST, math.ceil(_points(h, rho))), _MOST))
    x, w = np.polynomial.legendre.leggauss(points)
    u, du = 0.5 * _REACH * (x + 1.0), 0.5 * _REACH * w
    # The bridge's place from the threshold on the ray up, then on the one down.
    v = np.concatenate([u, -u])

    def weights(j: int, pair: tuple) -> np.ndarray:
        return np.concatenate([pair[0][j][:, None] * du, pair[1][j][:, None] * du], axis=1)

    y = h[0][:, None] + v[None, :]
    parts: dict[frozenset, np.ndarray] = {
        frozenset(): np.exp(-0.5 * y**2) / math.sqrt(2.0 * math.pi)
    }
    for j in range(count):
        # The parts of the tops whose backs have left the chain before this node join
        # the others.
        for t, (_, _, end, at_back, _, _) in enumerate(tops):
            if end == j and not at_back:
                parts = _closed(parts, t)
        # The parts that reflect off the tops whose links begin here.
        for t, (start, _, _, _, _, _) in enumerate(tops):
            if start == j:
                for key in list(parts):
                    if not any(tops[other][0] == j for other in key):
                        parts[key | {t}] = parts[key]
        for key in list(parts):
            hard_here = any(
                (tops[t][0] == j and tops[t][1]) or (tops[t][2] == j and tops[t][3]) for t in key
            )
            parts[key] = parts[key] * weights(j, hard if hard_here else free)
        for t, (_, _, end, at_back, _, _) in enumerate(tops):
            if end == j and at_back:
                parts = _closed(parts, t)
        if j == count - 1:
            break
        r = float(rho[j])
        spread = 1.0 - r * r
        following = h[j + 1][:, None] + v[None, :]
        step = following[:, None, :] - r * y[:, :, None]
        exponent = -0.5 * step**2 / spread
        norm = 1.0 / math.sqrt(2.0 * math.pi * spread)
        moved: dict[frozenset, np.ndarray] = {}
        for key, density in parts.items():
            reflecting = [t for t in key if tops[t][0] == j]
            if reflecting:
                (t,) = reflecting
                offset = tops[t][5]
                here = offset[j][:, None, None] + v[None, :, None]
                there = offset[j + 1][:, None, None] + v[None, None, :]
                kernel = np.exp(exponent - 2.0 * r * here * there / spread) * norm
                field = products.product(density[:, None, :], kernel)[:, 0, :]
                field = field * tops[t][4][:, None]
            else:
                kernel = np.exp(exponent) * norm
                field = products.product(density[:, None, :], kernel)[:, 0, :]
            moved[key] = moved[key] + field if key in moved else field
        parts, y = moved, following
    return np.sum([density.sum(axis=1) for density in parts.values()], axis=0)


def _closed(parts: dict, top: int) -> dict:
    """``parts`` (:func:`_marched`) with that of each set of tops with ``top`` added to
    that of the set without it."""
    out: dict[frozenset, np.ndarray] = {}
    for key, density in parts.items():
        key = key - {top}
        out[key] = out[key] + density if key in out else density
    return out


def _chain_cost(
    t: np.ndarray, rho: np.ndarray, active: np.ndarray, merged: float = 0.0, join=None
) -> tuple:
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
    (:func:`_merged`), and so is each node after a step of ``join`` (n - 1,) where that
    is given (every node active): it adds no step, and the next step is from that
    node."""
    count, frequencies = t.shape
    growth, fastest = np.zeros(frequencies), np.zeros(frequencies)
    narrowest = np.ones(frequencies)
    last, link, into = np.zeros(frequencies), np.ones(frequencies), np.zeros(frequencies)
    seen = np.zeros(frequencies, dtype=bool)
    for j in range(count):
        spread = np.maximum(1.0 - link**2, 1e-12)
        width = np.sqrt(spread)
        if join is None:
            joined = active[j] & seen & (width < merged)
        else:
            joined = active[j] & seen & bool(j > 0 and join[j - 1])
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
    return _between(growth, _GROWTH_NEAR, _GROWTH_FAR)


def _between(value, near: float, far: float):
    """1 up to ``near``, 0 from ``far``, and half a cosine's period between."""
    x = np.clip((value - near) / (far - near), 0.0, 1.0)
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
    that the row takes, and :meth:`field` the row's field.

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
    #: The model's planes of the nodes' faces that cross the way between its ends (a
    #: block's wall across it), but those of tops, each with the nodes it is a face of.
    walls: dict[int, list[int]]
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
    #: The share by which the row holds as its chain takes its edges for lines of their
    #: families (:func:`_lean_error`): its hold has it, and so has each part of it on a
    #: leg of a path past it (:meth:`passing`).
    fit: float = 1.0

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
        # The legs worked out by :meth:`_passing`, by their two ends.
        self._legs: dict[tuple[bytes, bytes], tuple[np.ndarray, np.ndarray]] = {}

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

    def wall_of(self, plane: int, point: np.ndarray) -> int:
        """The node of the row with a face in ``plane`` (a model plane) that crosses the
        way (:attr:`walls`) nearest ``point`` (a reflection off the plane) along the way,
        -1 if none has."""
        nodes = self.walls.get(int(plane), [])
        if not nodes:
            return -1
        x = float(np.einsum("i,i->", np.asarray(point, dtype=float) - self.origin, self.direction))
        return int(min(nodes, key=lambda i: abs(self.along[i] - x)))

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
        row's field has from the edge on its own, :meth:`past`), those of the tops the
        path reflects off, and the node of each face it reflects off that crosses the
        way (:attr:`walls`): off such a face the path runs back along the way, and the
        row's field, which goes on from each node towards the listener, has none of it.
        None where the term is not one of the row's field: over an edge not in the row,
        or on a path that reflects off a plane with neither a top of the row nor a face
        of a node that crosses the way."""
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
            if q >= 0:
                out |= {self.tops[q][0], self.tops[q][1]}
                continue
            wall = self.wall_of(int(plane), point)
            if wall < 0:
                return None
            out.add(wall)
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
        the blocked side, and with two more, the hard weights on the clear and the
        blocked side) of ``family``, the bridge from ``source`` (that family's coordinate
        of the source or its image) to the listener, in the terms of
        :func:`bridge_chain`: its thresholds (n, F), correlations (n - 1,) and weights
        on the clear and the blocked side (n, F), in order along the way, and with hard
        weights, those (n, F each) and the nodes in that order."""
        items = sorted(items, key=lambda item: self.along[item[0]])
        nodes = np.array([item[0] for item in items])
        listener = self.way[family, 1]
        mean = source + self.along[nodes] / self.length * (listener - source)
        level = np.array([item[1] for item in items])
        sign = np.array([item[2] for item in items], dtype=float)
        thresholds = (sign * (level - mean))[:, None] * scale[nodes]
        rho = sign[:-1] * sign[1:] * correlations(self.along[nodes], self.length)
        columns = range(3, len(items[0]))
        weights = [
            np.array([np.broadcast_to(item[c], (self.frequencies,)) for item in items])
            for c in columns
        ]
        if self.limit_far:
            # The side the bridge's mean lies on, towards which a far node weighs.
            near = membership(np.abs(thresholds))
            for c in range(0, len(weights), 2):
                limit = np.where(thresholds.real < 0.0, weights[c], weights[c + 1])
                weights[c : c + 2] = [
                    limit + near * (weight - limit) for weight in weights[c : c + 2]
                ]
        if len(weights) > 2:
            return thresholds, rho, *weights[:2], tuple(weights[2:]), nodes
        return thresholds, rho, *weights

    def _layouts(self, k: np.ndarray):
        """For each set of tops the bridge reflects off, as :meth:`field` takes them where
        a step of the chain is narrow (:meth:`_narrow`): the set (their indices in order
        along the way), the distance from the source's image in their planes in turn to
        the listener, the set's weight (F,) and the image's spreading and delay relative
        to those of the straight line from the source (F,), and the chains whose sum is
        the field: (coefficient, the steps of family 0, those of family 1,
        :meth:`_steps`)."""
        count = len(self.edge)
        frequencies = self.frequencies
        scale = self._scale(k)
        order = np.argsort(-self.along, kind="stable")
        for flips in itertools.product((False, True), repeat=len(self.tops)):
            chosen = [top for top, on in zip(self.tops, flips, strict=True) if on]
            weight = np.ones(frequencies, dtype=complex)
            hard = set()
            for front, back, _, reflection in chosen:
                weight = weight * self.share[front] * self.share[back] * reflection
                hard |= {front, back}
            # The mirror each node of family 0 is seen in from the listener's side: the
            # planes of the tops after it (and its own, at a top's front edge), each
            # through its two edges' lines, level or sloping along the way (as
            # _reflections has them): the image is mirror * y + offset + tilt * along.
            fronts = {}
            for front, back, _, _ in chosen:
                slope = (self.level[back] - self.level[front]) / (
                    self.along[back] - self.along[front]
                )
                fronts[front] = (self.level[front] - slope * self.along[front], slope)
            sign, shift = np.ones(count), np.zeros(count)
            mirror, offset, tilt = 1.0, 0.0, 0.0
            for i in order:
                if i in fronts:
                    level, slope = fronts[i]
                    plane, incline = mirror * level + offset, mirror * slope + tilt
                    mirror, offset, tilt = -mirror, 2.0 * plane - offset, 2.0 * incline - tilt
                sign[i], shift[i] = mirror, offset + tilt * self.along[i]
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
        """(F,) the largest, over the row's chains (:meth:`_splits`, those of the bridge
        that reflects off no top) at the frequencies where the chain is not none, of the
        exponent by which the integrand of :func:`bridge_chain` grows along a chain of
        the nodes that weigh their two sides differently (_GROWTH_NEAR,
        :func:`_chain_cost`), and of _GROWTH_FAR in proportion to the points its
        quadrature takes over _MOST."""
        out = np.zeros(self.frequencies)
        for _, *families in self._splits(self.wavenumber):
            steps_of = [steps[:4] for steps in families if steps is not None]
            # A chain with a node that weighs both sides by none is none there.
            some = np.ones(self.frequencies, dtype=bool)
            for _, _, clear, blocked in steps_of:
                some &= ~((clear == 0.0) & (blocked == 0.0)).any(axis=0)
            for thresholds, rho, clear, blocked in steps_of:
                t = np.abs(thresholds) * np.sign(thresholds.real)
                active = (clear != blocked) & some
                growth, points = _chain_cost(t, rho, active, _APART)
                out = np.maximum(out, np.maximum(growth, _GROWTH_FAR * points / _MOST))
        return out

    def _scale(self, k: np.ndarray) -> np.ndarray:
        """(n, F) each node's chain variable per metre across the way at wavenumbers
        ``k`` (F,): sqrt(k / d) exp(i pi / 4), d = x (D - x) / D."""
        across = self.along * (self.length - self.along) / self.length
        return np.sqrt(np.asarray(k, dtype=float)[None, :] / across[:, None]) * np.exp(
            0.25j * np.pi
        )

    def _splits(self, k: np.ndarray):
        """For each split of the screens' corners (see :meth:`_layouts`), the chains of the
        bridge from the source, whose sum over the splits is the row's field over the
        tops' reflections too (:func:`_reflected`): (coefficient, the steps of family
        0 with the hard weights of the tops' nodes, those of family 1; :meth:`_steps`,
        none where a family has no node)."""
        count = len(self.edge)
        scale = self._scale(k)
        hard = {i for front, back, _, _ in self.tops for i in (front, back)}
        for split in itertools.product((False, True), repeat=len(self.corners)):
            coefficient = 1.0
            items: list[list] = [[], []]
            for i in range(count):
                if i not in self._cornered:
                    item = (i, self.level[i], self.clear[i], 1.0, 1.0 - self.share[i])
                    if self.family[i] == 0:
                        # The hard weights, those of the tops' nodes in the tops' parts.
                        item += (1.0, 0.0) if i in hard else (1.0, 1.0 - self.share[i])
                    items[self.family[i]].append(item)
            for (first, second, (_, one, other, both)), part in zip(
                self.corners, split, strict=True
            ):
                if part:
                    coefficient = -coefficient
                    items[0].append(
                        (first, self.level[first], self.clear[first], 0.0, 1.0, 0.0, 1.0)
                    )
                    items[1].append(
                        (second, self.level[second], self.clear[second], one, one + both)
                    )
                else:
                    items[1].append(
                        (second, self.level[second], self.clear[second], 1.0, 1.0 - other)
                    )
            yield (
                coefficient,
                self._steps(0, self.way[0, 0], items[0], scale) if items[0] else None,
                self._steps(1, self.way[1, 0], items[1], scale) if items[1] else None,
            )

    def _reflections(self, nodes: np.ndarray, scale: np.ndarray) -> list:
        """The tops of the chain of family 0 over ``nodes`` (in order along the way), in
        the terms of :func:`_reflected`: (front node, back node, share (F,), offsets of
        the nodes from the top's plane (n, F))."""
        place = {int(node): j for j, node in enumerate(nodes)}
        out = []
        for front, back, _, reflection in self.tops:
            # The plane through the two edges' lines, as its level along the way.
            slope = (self.level[back] - self.level[front]) / (self.along[back] - self.along[front])
            plane = self.level[front] + slope * (self.along[nodes] - self.along[front])
            offset = (self.clear[nodes] * (self.level[nodes] - plane))[:, None] * scale[nodes]
            share = self.share[front] * self.share[back] * reflection
            out.append((place[int(front)], place[int(back)], share, offset))
        return out

    def _narrow(self) -> bool:
        """Whether a step of the chain of family 0 is narrower than _CLOSE, where its
        quadrature would take more than _MOST points and takes nodes as one
        (:func:`_chain`), which the tops' reflections (:func:`_reflected`) do not: the
        row's field is then worked out over each set of tops apart (:meth:`_layouts`).
        Down to _CLOSE the march takes the step with up to _MOST points."""
        along = np.sort(self.along[self.family == 0])
        if not self.tops or len(along) < 2:
            return False
        rho = correlations(along, self.length)
        return bool((np.sqrt(np.maximum(1.0 - rho**2, 0.0)) < _CLOSE).any())

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
            hold = hold * self.fit
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
        walls = {}
        for plane, owners in self.walls.items():
            kept = [index[i] for i in owners if i in index]
            if kept:
                walls[plane] = kept
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
            walls=walls,
            origin=start,
            direction=self.direction,
            axes=self.axes,
            ends=self.ends[inside],
            wavenumber=self.wavenumber,
            fit=self.fit,
        )
        return part if own else part.computable()

    def passing(self, legs) -> np.ndarray:
        """(F,) the field at the end of each of ``legs`` (pairs of points in turn, the
        path's) of a point source at its start relative to the free field, past the nodes
        of the row between the two along the way (not those at either), over every row
        of them (:meth:`field`), each node in the row by how close the straight line
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
        bridge between them, and (F,) its field there where it holds (:meth:`passing`;
        where it does not, 1). The legs of the paths over two edges come again and again
        (on a way at right angles to the edges, from the source to each edge and from
        each edge to the listener): each pair of ends is worked out once."""
        key = (start.tobytes(), end.tobytes())
        if key not in self._legs:
            self._legs[key] = self._passing_once(start, end)
        return self._legs[key]

    def _passing_once(self, start: np.ndarray, end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`_passing`, worked out."""
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
        held = dataclasses.replace(part, hold=(part.hold > 0.0).astype(float))
        return part.hold, held.field(self.wavenumber, every=True)

    def _leg(self, inside: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """(F,) the field at ``end`` of a point source at ``start`` relative to the free
        field past the nodes ``inside`` (ascending, between the two), each in the row by
        its share in this row, those the straight line passes far from at their
        geometric limit (``limit_far``), over every row of them (:meth:`field`)."""
        if len(inside) == 0:
            return np.ones(self.frequencies, dtype=complex)
        part = self._part(inside, start, end, own=True)
        part = dataclasses.replace(part, limit_far=True).computable()
        return part.field(self.wavenumber, every=True)

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

    def field(self, k: np.ndarray, every: bool = False) -> np.ndarray:
        """(F,) the row's field relative to the free field at wavenumbers ``k`` (F,), over
        the rows of two nodes or more in proportion to their probability (with
        ``every``, over every row, of any number of nodes, and the free field where the
        row does not hold), with the bridge's reflections off every set of tops
        (:func:`_reflected` over the chains of :meth:`_splits`; where a step is narrow,
        :meth:`_narrow`, over each set apart, :meth:`_layouts`)."""
        live = np.flatnonzero(self.hold != 0.0)
        total = np.zeros(len(live), dtype=complex)
        if self._narrow():
            for _, _, weight, ratio, chains in self._layouts(k):
                # Where a set of tops has no share (their edges are in no row), its part
                # is none: its chains, far from the bridge there, are not worked out.
                mine = np.flatnonzero(weight[live] != 0.0)
                columns = live[mine]
                part = np.zeros(len(columns), dtype=complex)
                for coefficient, *families in chains:
                    term = np.full(len(columns), coefficient, dtype=complex)
                    for steps in families:
                        if steps is not None:
                            thresholds, rho, clear, blocked = steps
                            term *= bridge_chain(
                                thresholds[:, columns], rho, clear[:, columns], blocked[:, columns]
                            )
                    part += term
                total[mine] += (weight * ratio)[columns] * part
        else:
            scale = self._scale(k)
            for coefficient, *families in self._splits(k):
                term = np.full(len(live), coefficient, dtype=complex)
                if families[0] is not None:
                    thresholds, rho, clear, blocked, hard, nodes = families[0]
                    tops = [
                        (front, back, np.asarray(share)[live], offset[:, live])
                        for front, back, share, offset in self._reflections(nodes, scale)
                    ]
                    term *= _reflected(
                        thresholds[:, live],
                        rho,
                        clear[:, live],
                        blocked[:, live],
                        (hard[0][:, live], hard[1][:, live]),
                        tops,
                    )
                if families[1] is not None:
                    thresholds, rho, clear, blocked = families[1]
                    term *= bridge_chain(thresholds[:, live], rho, clear[:, live], blocked[:, live])
                total += term
        if not every:
            # The row of no node is the way's free field.
            total -= self._empty()[live]
        out = np.zeros(self.frequencies, dtype=complex)
        out[live] = self.hold[live] * total
        if every:
            # Where the row does not hold, no node is in it: the free field.
            return out + (1.0 - self.hold)
        # The rows of the nodes of one obstacle are the edges' paths'.
        for group in self._groups:
            inside = np.array(sorted(group))
            part = self._part(inside, self.origin, self.origin + self.length * self.direction, True)
            out = out - self.hold * self._empty(group) * (part.field(k, every=True) - part._empty())
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


def _across(d: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(3,) the way's direction ``u`` at right angles to the unit direction ``d`` of an
    edge it crosses: the way as seen along the edge."""
    w = u - _dot(u, d) * d
    return w / math.sqrt(_dot(w, w))


def _lean_error(
    normal: np.ndarray, family: np.ndarray, clear: np.ndarray, along: np.ndarray, length: float
) -> float:
    """The most, in dB, by which the chain changes the share of the bridge that passes
    any two of a row's nodes on their clear sides where the way runs through both
    (1/4 + asin(r) / (2 pi) for the correlation r of the bridge's two coordinates
    there), as it takes each node for a line of its family: r is rho (by the clear
    sides) for two of one family and 0 for two of different families, whose coordinates
    are independent, where it is rho times the cosine of the angle between the nodes'
    normals in the plane across the way (``normal`` (n, 3), each towards its node's
    clear side). ``family`` and ``clear`` (n,) are the nodes', ``along`` (n,) their
    places along the way of ``length``."""
    if len(along) < 2:
        return 0.0
    i, j = np.triu_indices(len(along), 1)
    x, y = np.minimum(along[i], along[j]), np.maximum(along[i], along[j])
    rho = np.sqrt(np.clip(x * (length - y) / (y * (length - x)), 0.0, 1.0))
    chain = np.where(family[i] == family[j], clear[i] * clear[j] * rho, 0.0)
    exact = rho * np.einsum("ij,ij->i", normal[i], normal[j])
    through = [0.25 + np.arcsin(np.clip(r, -1.0, 1.0)) / (2.0 * np.pi) for r in (chain, exact)]
    with np.errstate(divide="ignore"):
        error = np.abs(
            20.0 * np.log10(np.maximum(through[0], 1e-300) / np.maximum(through[1], 1e-300))
        )
    return float(error.max())


def _taut(along: np.ndarray, level: np.ndarray, clear: np.ndarray, ends, length: float):
    """The likeliest path of a bridge along one coordinate (its mean's limit): the
    shortest line from ``ends`` (the coordinate at 0 and at ``length``) that passes
    each node (``along``, ``level``) on its ``clear`` side (+1: above its level), as
    its corners (places along the way, coordinates), by adding the node it passes
    furthest on the blocked side and dropping a corner the line past it no longer
    needs, in turn."""
    active: list[int] = []
    for _ in range(4 * len(along) + 4):
        xs = np.concatenate([[0.0], along[active], [length]])
        ys = np.concatenate([[ends[0]], level[active], [ends[1]]])
        slack = []
        for m, i in enumerate(active):
            line = ys[m] + (along[i] - xs[m]) / (xs[m + 2] - xs[m]) * (ys[m + 2] - ys[m])
            slack.append(clear[i] * (line - level[i]))
        if slack and max(slack) > 1e-12:
            del active[int(np.argmax(slack))]
            continue
        passes = clear * (np.interp(along, xs, ys) - level)
        worst = int(np.argmin(passes)) if len(passes) else -1
        if worst < 0 or passes[worst] >= -1e-12 or worst in active:
            break
        active = sorted(active + [worst], key=lambda i: along[i])
    xs = np.concatenate([[0.0], along[active], [length]])
    ys = np.concatenate([[ends[0]], level[active], [ends[1]]])
    return xs, ys


def _paraxial_excess(along, level, clear, family, way, length: float) -> float:
    """How much longer the bridge's paraxial measure makes the row's likeliest path
    (:func:`_taut`, in each family's coordinate) than it is: the sum over its straight
    pieces of da + (dq^2 + dp^2) / (2 da) - sqrt(da^2 + dq^2 + dp^2). The paraxial field
    holds while the phase of that, k times it, stays small."""
    corners = []
    for f in (0, 1):
        mine = family == f
        corners.append(_taut(along[mine], level[mine], clear[mine], way[f], length))
    xs = np.unique(np.concatenate([corners[0][0], corners[1][0]]))
    q = np.interp(xs, *corners[0])
    p = np.interp(xs, *corners[1])
    da, dq, dp = np.diff(xs), np.diff(q), np.diff(p)
    da = np.maximum(da, 1e-12 * length)
    across = dq**2 + dp**2
    return float(np.sum(da + across / (2.0 * da) - np.sqrt(da**2 + across)))


def _directions(edges, model, u: np.ndarray, nearest: int, candidates: np.ndarray) -> tuple:
    """The row's directions q and p for the way along ``u`` with ``nearest`` its reference
    edge, and each edge's family, the direction of its constraint and the share of its
    obstacle along that (:func:`find_row`): q at right angles to the way and to the
    edge, or the normal of a face along the way (its sine to the way across the edge,
    :func:`_across`, at most _ALONG: a block's top), at right angles to the way, of
    the first of ``candidates``
    (by distance) parallel to the edge that has one; an edge is of the first family
    where its line across the way lies nearer p than q, and of the second where it lies
    nearer q and is a thin screen's (and of neither, -1, where it is a solid's along
    q)."""
    d = edges.direction
    q = np.cross(u, d[nearest])
    q /= math.sqrt(_dot(q, q))
    for e in candidates:
        if edges.wedge[e] >= 2.0 - 1e-9 or abs(_dot(d[e], q)) > math.sin(_PARALLEL):
            continue
        normals = model.plane_normal[edges.plane[e]]
        along_way = np.abs(np.einsum("ij,j->i", normals, _across(d[e], u))) <= _ALONG
        if along_way.any():
            normal = normals[np.flatnonzero(along_way)[0]]
            normal = normal - _dot(normal, u) * u
            normal /= math.sqrt(_dot(normal, normal))
            q = normal * np.sign(_dot(normal, q))
            break
    p = np.cross(u, q)
    p /= math.sqrt(_dot(p, p))
    lines = _lines(d, u)
    nearer_q = np.abs(np.einsum("ij,j->i", lines, q)) > np.abs(np.einsum("ij,j->i", lines, p))
    family = np.where(nearer_q, -1, 0)
    family[nearer_q & np.isclose(edges.wedge, 2.0)] = 1
    constraint = np.where((family == 0)[:, None], q[None, :], p[None, :])
    stands = np.einsum("ij,ij->i", -edges.bisector(), constraint)
    return q, p, family, constraint, stands


def _lines(d: np.ndarray, u: np.ndarray) -> np.ndarray:
    """(E, 3) the unit directions ``d`` (E, 3) of edges as seen along the way's ``u``: in
    the plane across the way (those along the way, which do not cross it, as they are)."""
    across = d - np.einsum("ij,j->i", d, u)[:, None] * u[None, :]
    size = np.linalg.norm(across, axis=1)
    return np.where((size > 1e-12)[:, None], across / np.maximum(size, 1e-300)[:, None], d)


def find_row(edges, model, source, listener, wavenumber, tolerance, reflection) -> Row | None:
    """The row of edges that the way from ``source`` to ``listener`` passes close by, at
    the ``wavenumber`` (F,), or None where it passes fewer than two obstacles so: the
    ``edges`` (:class:`~acoustic_spaces.engine.diffraction.Edges` of ``model``) whose
    line crosses the way between the two (at any angle: over parallel edges at a slant
    to it the chain's thresholds and correlations are those of the way at right angles
    to them at the wavenumber times the sine of that angle), within :data:`ROW_FAR`
    standard deviations of the bridge at the lowest wavenumber, and whose obstacle
    stands across the way. ``reflection(edge, side)`` is the (F,) reflection factor of
    face ``side`` of ``edge`` as the edge's paths have it. Where such an edge cannot be
    in the row (an edge of a solid that runs nearer q than across it), or two are the
    edges of one obstacle across the way (a strip's top and foot, a panel's two ends)
    or of two solids that touch, there is no row either. The row holds in full where
    taking its edges for lines of their families changes the share of the bridge past
    any two of them by at most :data:`_LEAN_NEAR` dB, and not from :data:`_LEAN_FAR`
    (:func:`_lean_error`): edges that lean from one another.

    The row's directions: q at right angles to the way and to the edge nearest it in
    Fresnel units (the nearest for which every such edge is of one of the families,
    :func:`_directions`), or the normal of a face along the way (at most
    :data:`_ALONG` its sine to the way across the edge, :func:`_across`: a block's
    top) of the nearest edge parallel to that one that has such a face, and p at right
    angles to the way and to q. An edge's share (:func:`membership`) is that of the
    way's distance from it where the way passes it on its clear side (1 where it passes
    it on its blocked side: the bridge goes round it), and, near an end of it that is
    not the corner of a screen whose other edge is in the row, of the way's distance
    from that end too:
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
    crossing = sine * length > tolerance
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
            crosses = abs(_dot(model.plane_normal[plane], _across(d[e], u))) > _ALONG
            if len(sides) == 0 or not crosses:
                continue
            towards = _dot(facing[e, side], apart) > 0.0
            back = _dot(facing[f, sides[0]], apart) < 0.0
            if towards or back:
                return None
    clear = -np.sign(stands[nodes])
    way_coordinates = np.array(
        [[_dot(source, q), _dot(listener, q)], [_dot(source, p), _dot(listener, p)]]
    )
    sigma = np.sqrt(across[nodes][:, None] / k[None, :])  # (n, F)
    mean = way_coordinates[fam, 0] + along / length * (
        way_coordinates[fam, 1] - way_coordinates[fam, 0]
    )
    # Each node's line and normal across the way (in the plane at right angles to it), the
    # normal towards its clear side, and the cosine of the line's angle to its family's
    # there: the reference edge's line, or that at right angles to it.
    lines = _lines(d[nodes], u)
    clear_normal = np.cross(u[None, :], lines)
    towards = np.sign(np.einsum("ij,ij->i", clear_normal, constraint[nodes]))
    clear_normal *= (clear * towards)[:, None]
    reference = _lines(d[nearest][None, :], u)[0]
    families = np.stack([reference, np.cross(u, reference)])
    lean_cosine = np.abs(np.einsum("ij,ij->i", lines, families[fam]))
    # Each edge's level along its constraint: the way's there and the edge's offset from
    # the way along the constraint over the cosine, as along its own normal (across which
    # the bridge's coordinate that it stops has the spread of the constraint's); for an
    # edge parallel to its family's line, its own level along the constraint.
    offsets = np.einsum("ij,ij->i", on_edge[nodes] - on_way[nodes], constraint[nodes])
    level = np.einsum("ij,ij->i", on_edge[nodes], constraint[nodes]) + offsets * (
        1.0 / lean_cosine - 1.0
    )
    # A node the way passes on its blocked side is in the row: the bridge goes round it.
    passes = ((mean - level) * clear)[:, None] / sigma
    own = np.where(passes < 0.0, 1.0, membership(np.abs(passes)))
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
    fit = float(
        _between(_lean_error(clear_normal, fam, clear, along, length), _LEAN_NEAR, _LEAN_FAR)
    )
    # The row hands over to the edges' paths as the paraxial measure of its likeliest
    # path errs by a phase of pi / 4 to pi / 2 (an eighth to a quarter of a wavelength).
    excess = _paraxial_excess(along, level, clear, fam, way_coordinates, length)
    hold = hold * _between(k * excess, 0.25 * np.pi, 0.5 * np.pi)
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
            along_way = abs(_dot(normal, _across(d[e], u))) <= _ALONG
            # A top lies along q and the way, level or pitched along the way, not rolled
            # towards p.
            if abs(_dot(normal, p)) > math.sin(_PARALLEL) or not along_way:
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
    # The nodes' faces whose planes cross the way between its ends, but tops'.
    walls: dict[int, list[int]] = {}
    for i in range(count):
        e = nodes[i]
        for side in (0, 1):
            plane = int(edges.plane[e, side])
            if plane < 0 or top_sides.get(i) == side or i in walls.get(plane, []):
                continue
            normal, offset = model.plane_normal[plane], float(model.plane_offset[plane])
            if (_dot(normal, source) - offset) * (_dot(normal, listener) - offset) < 0.0:
                walls.setdefault(plane, []).append(i)
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
        hold=hold * fit,
        corners=units,
        tops=tops,
        top_planes=top_planes,
        top_sides=top_sides,
        walls=walls,
        origin=source,
        direction=u,
        axes=np.stack([q, p]),
        ends=ends,
        wavenumber=k,
        fit=fit,
    ).computable()
    return row if row.hold.any() else None


__all__ = ["ROW_FAR", "ROW_NEAR", "Row", "bridge_chain", "correlations", "find_row", "membership"]
