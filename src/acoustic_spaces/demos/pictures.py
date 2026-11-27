"""What the demos' stills share: the colours, the fonts, the parts of a still (the heading,
the headphones badge, the facts, the tour, the credit line and the lines that say how
the sound was made) and the marks for listeners and loudspeakers. Needs Pillow (which
the viz extra brings); Inter is used when installed, else DejaVu Sans.

Each demo draws its own scene (a stadium from above, a church in plan and section, a
lake with its bent rays, a concert hall) with its own module next to its script; the
frame round it comes from here, so the videos look alike.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from acoustic_spaces._version import __version__
from acoustic_spaces.demos.audio import mmss
from acoustic_spaces.demos.pipeline import SPECTRUM_BOX

BG = (13, 17, 28)
PANEL = (21, 27, 42)
LINE = (58, 68, 92)
TEXT = (232, 236, 244)
MUTED = (148, 158, 182)
FAINT = (96, 106, 132)
#: The listener of the moment.
LISTENER = (102, 217, 239)
#: What makes the sound: the loudspeakers of a sound system, the organ, the guide, a
#: scene's sources that play.
SOURCE = (255, 166, 77)

#: A still and a thumbnail (as YouTube wants it).
SIZE = (1920, 1080)
THUMBNAIL = (1280, 720)
BADGE = "BINAURAL  ·  USE HEADPHONES"
#: The still's margin, and where the stop's column starts by default.
MARGIN = 80
RIGHT = 1840

_FONTS: dict = {}


def font(weight: str, size: int, display: bool = False) -> ImageFont.FreeTypeFont:
    """Inter (Display for titles) when installed, else DejaVu Sans, else Pillow's own."""
    key = (weight, size, display)
    if key not in _FONTS:
        family = "InterDisplay" if display else "Inter"
        bold = weight in ("SemiBold", "Bold", "Black")
        for name in (f"{family}-{weight}.otf", f"{family}-{weight}.ttf"):
            try:
                _FONTS[key] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            try:
                _FONTS[key] = ImageFont.truetype(
                    "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
                )
            except OSError:
                _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


def text(draw, xy, s, f, fill=TEXT, anchor="la", halo: int = 0) -> None:
    """``s`` at ``xy``; with ``halo``, ringed that many pixels wide in the background's
    colour, so that the lines and marks it is written over do not run through it."""
    draw.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=halo, stroke_fill=BG)


def arrow(draw, start, direction, length: float, head: float, colour, width: int) -> None:
    """A line from ``start`` along the unit vector ``direction``, with a head at its end."""
    (x, y), (dx, dy) = start, direction
    tip = (x + dx * length, y + dy * length)
    base = (tip[0] - dx * head, tip[1] - dy * head)
    draw.line([(x, y), base], fill=colour, width=width)
    draw.polygon(
        [
            tip,
            (base[0] - dy * head * 0.6, base[1] + dx * head * 0.6),
            (base[0] + dy * head * 0.6, base[1] - dx * head * 0.6),
        ],
        fill=colour,
    )


def cone(draw, apex, direction, length: float, half_angle: float, rgba) -> None:
    """A fan from ``apex`` along the unit vector ``direction``, ``half_angle`` degrees to
    either side: where a loudspeaker aims, or the way a source faces."""
    (x, y), (dx, dy) = apex, direction
    aim = math.atan2(dy, dx)
    rim = [
        (x + length * math.cos(aim + math.radians(a)), y + length * math.sin(aim + math.radians(a)))
        for a in range(-int(half_angle), int(half_angle) + 1, 4)
    ]
    draw.polygon([(x, y), *rim], fill=rgba)


def glow(img: Image.Image, shapes, blur: float) -> None:
    """Blurred discs, ``shapes`` as (x, y, radius, rgba), under what comes next; blurred
    in a patch round them only, which an animation draws every frame."""
    if not shapes:
        return
    margin = 3.0 * blur + 2.0
    left = max(0, int(math.floor(min(x - r for x, _, r, _ in shapes) - margin)))
    top = max(0, int(math.floor(min(y - r for _, y, r, _ in shapes) - margin)))
    right = min(img.width, int(math.ceil(max(x + r for x, _, r, _ in shapes) + margin)))
    bottom = min(img.height, int(math.ceil(max(y + r for _, y, r, _ in shapes) + margin)))
    if right <= left or bottom <= top:
        return
    layer = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for x, y, r, rgba in shapes:
        draw.ellipse([x - r - left, y - r - top, x + r - left, y + r - top], fill=rgba)
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(blur)), (left, top))


def canvas(size: tuple[int, int] = SIZE) -> Image.Image:
    return Image.new("RGBA", size, BG)


def title_card(size: tuple[int, int], title: str, name: str, lines) -> Image.Image:
    """A title card of ``size``: ``title`` in large, ``name`` under it and ``lines`` under
    that, each cut to the width (the card before each part of a run of several parts,
    :func:`~acoustic_spaces.demos.join.join_parts`)."""
    img = canvas(size)
    draw = ImageDraw.Draw(img)
    text(draw, (MARGIN, 300), title, font("Bold", 64, display=True))
    text(draw, (MARGIN, 390), name, font("Regular", 34), LISTENER)
    y = 480
    for line in lines:
        shown, face = fitted(draw, line, 30, size[0] - 2 * MARGIN)
        text(draw, (MARGIN, y), shown, face, MUTED)
        y += 52
    return img


def save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, optimize=True)


class Plan:
    """Scene x and y to pixels, from above with north (+y) up; or ``rotated``: as the
    audience sees a stage, the scene's +x down the picture and +y to the right."""

    def __init__(self, left, top, scale, box, rotated: bool = False):
        self.left, self.top, self.s = left, top, scale
        self.x0, self.y0, self.x1, self.y1 = box
        self.rotated = rotated
        #: The panel the plan is drawn in (left, top, right, bottom), where labels may go
        #: beyond the scene's own box; None where it is not given.
        self.area: tuple[float, float, float, float] | None = None

    def __call__(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        if self.rotated:
            return (self.left + (y - self.y0) * self.s, self.top + (x - self.x0) * self.s)
        return (self.left + (x - self.x0) * self.s, self.top + (self.y1 - y) * self.s)

    def vec(self, dx: float, dy: float) -> tuple[float, float]:
        """A scene direction on the picture."""
        return (dy, dx) if self.rotated else (dx, -dy)

    def rect(self, x0, y0, x1, y1) -> list[float]:
        """A scene rectangle as Pillow wants it: left, top, right, bottom."""
        (a, b), (c, d) = self(x0, y1), self(x1, y0)
        return [min(a, c), min(b, d), max(a, c), max(b, d)]

    @property
    def size(self) -> tuple[float, float]:
        w, h = (self.x1 - self.x0) * self.s, (self.y1 - self.y0) * self.s
        return (h, w) if self.rotated else (w, h)


def facing(item: dict) -> tuple[float, float]:
    """The way a source or listener faces in the scene (unit vector): yaw turns from +x
    towards +y."""
    yaw = math.radians(item.get("orientation", {}).get("yaw", 0.0))
    return math.cos(yaw), math.sin(yaw)


def axes(plan, item: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """The way ``item`` faces and its left, as unit vectors on the picture (``plan.vec``
    maps scene directions): the left ear points to (-sin yaw, cos yaw)."""
    fx, fy = facing(item)
    return plan.vec(fx, fy), plan.vec(-fy, fx)


# --------------------------------------------------------------------------- marks


def head(
    img: Image.Image,
    at: tuple[float, float],
    forward: tuple[float, float],
    left: tuple[float, float],
    r: float,
    *,
    pointer: bool = True,
    nose: bool = True,
    ears: bool = True,
    letters: bool = True,
) -> None:
    """The listener of the moment: a glowing head with a nose, ears marked L and R and an
    arrow the way it faces (``forward`` and ``left`` on the picture, from :func:`axes`)."""
    hx, hy = at
    fx, fy = forward
    lx, ly = left
    glow(img, [(hx, hy, 3 * r, LISTENER + (110,))], r)
    draw = ImageDraw.Draw(img)
    width = max(3, round(r / 3))
    if pointer:
        start = (hx + fx * r, hy + fy * r)
        arrow(draw, start, forward, r * 3.4, r * 1.2, BG, width + 4)  # an outline, over the rays
        arrow(draw, start, forward, r * 3.2, r * 1.0, LISTENER, width)
    if nose:
        points = [
            (hx + fx * r * 1.45, hy + fy * r * 1.45),
            (hx + fx * r * 0.8 + lx * r * 0.45, hy + fy * r * 0.8 + ly * r * 0.45),
            (hx + fx * r * 0.8 - lx * r * 0.45, hy + fy * r * 0.8 - ly * r * 0.45),
        ]
        draw.polygon(points, fill=LISTENER)
    if ears:
        for side in (1, -1):
            ex, ey = hx + lx * r * 0.9 * side, hy + ly * r * 0.9 * side
            draw.ellipse(
                [ex - r * 0.38, ey - r * 0.38, ex + r * 0.38, ey + r * 0.38],
                fill=LISTENER,
                outline=BG,
            )
    draw.ellipse([hx - r, hy - r, hx + r, hy + r], fill=LISTENER, outline=BG, width=2)
    if letters:
        f = font("Bold", max(13, round(r * 1.1)))
        for side, letter in ((1, "L"), (-1, "R")):
            px, py = hx + lx * r * 2.0 * side, hy + ly * r * 2.0 * side
            draw.text(
                (px, py), letter, font=f, fill=TEXT, anchor="mm", stroke_width=3, stroke_fill=BG
            )


def ring(
    draw,
    at: tuple[float, float],
    forward: tuple[float, float],
    r: float,
    length: float,
    headsize: float,
    *,
    fill=None,
    colour=MUTED,
) -> None:
    """Another listener: a ring of radius ``r`` with an arrow the way it faces, ``length``
    pixels long with a head of ``headsize``."""
    x, y = at
    fx, fy = forward
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=colour, width=2)
    arrow(draw, (x + fx * r, y + fy * r), forward, length, headsize, colour, 2)


def rays(
    img,
    plan,
    levels: dict[str, float],
    sources: Sequence[dict],
    to,
    colour_of,
    *,
    alpha: tuple[int, int] = (40, 170),
    width: int = 3,
) -> None:
    """A line from each source to the point ``to`` on the picture, brighter the louder the
    source is there (``levels``, dB, over the 24 dB below the loudest): its opacity from
    ``alpha[0]`` to the sum of both, its width from 1 to ``1 + width`` pixels;
    ``colour_of`` gives a source's colour from its name, ``plan`` maps its position (a
    plan, a section). A source whose level is not a number (minus infinity: nothing of it
    heard there yet, as before a train's first bogie comes by) gets no line."""
    levels = {name: v for name, v in levels.items() if math.isfinite(v)}
    if not levels:
        return
    top = max(levels.values())
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for source in sorted(sources, key=lambda s: levels.get(s["name"], -999)):
        weight = 1.0 + (levels.get(source["name"], -999) - top) / 24.0
        if weight <= 0:
            continue
        draw.line(
            [plan(*source["position"]), to],
            fill=colour_of(source["name"]) + (int(alpha[0] + alpha[1] * weight),),
            width=max(1, int(1 + width * weight)),
        )
    img.alpha_composite(layer)


def scale_bar(draw, left: float, bottom: float, length: float, label: str, up: bool = True):
    """A bar of ``length`` pixels with its label, at the picture's ``left`` and
    ``bottom``; the label over it, or under it with ``up`` False."""
    draw.line([(left, bottom), (left + length, bottom)], fill=MUTED, width=3)
    where = (left + length / 2, bottom - 6) if up else (left + length / 2, bottom + 6)
    text(draw, where, label, font("Medium", 16), MUTED, "md" if up else "ma")


# --------------------------------------------------------------------------- the frame


def heading(draw, title: str, subtitle: str, y: int = 44) -> None:
    """The video's title and subtitle at the top left."""
    text(draw, (MARGIN, y), title, font("Bold", 50, display=True))
    text(draw, (MARGIN, y + 68), subtitle, font("Regular", 26), MUTED)


def badge(draw, x: float, y: float, size: int = 24, label: str = BADGE) -> float:
    """The headphones badge, a pill with its top left at (x, y); returns its height."""
    pill = font("SemiBold", size)
    height = round(size * 1.9)
    pad = round(size * 0.85)
    draw.rounded_rectangle(
        [x, y, x + draw.textlength(label, font=pill) + 2 * pad, y + height], height / 2, LISTENER
    )
    text(draw, (x + pad, y + height / 2), label, pill, BG, "lm")
    return height


def place(
    draw, x: float, y: float, title: str, where: str, size: int = 44, where_size: int = 26
) -> float:
    """The stop's name and where it is; returns the y under them."""
    text(draw, (x, y), title, font("Bold", size, display=True))
    text(draw, (x, y + size + 14), where, font("Regular", where_size), MUTED)
    return y + size + 14 + 50


def _fact_rows(draw, line, size: int, width: float) -> tuple[list[str], object, str]:
    """A fact (a string or (string, colour)) in rows no wider than ``width`` pixels,
    broken between words, with its font and colour."""
    s, colour = line if isinstance(line, tuple) else (line, TEXT)
    f = font("SemiBold" if colour == LISTENER else "Regular", size)
    rows: list[str] = []
    for word in s.split(" "):
        longer = f"{rows[-1]} {word}" if rows else word
        if rows and draw.textlength(longer, font=f) > width:
            rows.append(word)
        else:
            rows[-1:] = [longer]
    return rows, f, colour


def facts(
    draw,
    x: float,
    y: float,
    lines: Sequence,
    size: int = 24,
    step: int = 36,
    right: float = RIGHT,
) -> float:
    """Lines of facts under the stop's name, each a string or (string, colour), one
    longer than reaches ``right`` going on on the next, between words; returns the y
    under them."""
    for line in lines:
        rows, f, colour = _fact_rows(draw, line, size, right - x)
        for row in rows:
            text(draw, (x, y), row, f, colour)
            y += step
    return y


#: The sizes the facts under a stop's name are set in, with their rows' step (px),
#: largest first (:func:`fitted_facts`).
FACT_SIZES = ((23, 32), (21, 29), (19, 26), (17, 23))


def fitted_facts(
    draw, x: float, y: float, lines: Sequence, bottom: float, right: float = RIGHT
) -> float:
    """:func:`facts` in the largest of :data:`FACT_SIZES` whose rows end by ``bottom``
    (the smallest where none does), so that what comes under them keeps its room;
    returns the y under them."""
    for size, step in FACT_SIZES:
        rows = sum(len(_fact_rows(draw, line, size, right - x)[0]) for line in lines)
        if y + rows * step <= bottom:
            break
    return facts(draw, x, y, lines, size=size, step=step, right=right)


def tour_column(
    draw,
    x: float,
    y: float,
    entries: Sequence[tuple[str, float]],
    current: int,
    *,
    row: int,
    right: float = RIGHT,
    bottom: float = 860,
    size: int = 19,
    title: str = "The tour",
) -> None:
    """The tour's stops (label, start s) with their times under ``title``, the current
    one bright, the ones to come faint, in rows of ``row`` pixels. Where they would pass
    ``bottom``, as many as fit, the current one among them, with how many stops come
    before and after them (:func:`tour_window`)."""
    text(draw, (x, y), title, font("SemiBold", 22), MUTED)
    y += 36
    first, last = tour_window(len(entries), current, max(1, int((bottom - y) // row)))
    if first > 0:
        text(draw, (x, y), f"   … {_stops(first)} before", font("Regular", size), FAINT)
        y += row
    for j in range(first, last):
        label, t0 = entries[j]
        colour = LISTENER if j == current else (MUTED if j < current else FAINT)
        weight = "SemiBold" if j == current else "Regular"
        mark = "▶ " if j == current else "   "
        text(draw, (x, y), mark + label, font(weight, size), colour)
        text(draw, (right, y), mmss(t0), font(weight, size), colour, "ra")
        y += row
    if last < len(entries):
        more = len(entries) - last
        text(draw, (x, y), f"   … {_stops(more)} after", font("Regular", size), FAINT)


#: The tour column's title and the gap over it, its rows and how far down it may go (px).
TOUR_HEAD, TOUR_ROW, TOUR_BOTTOM = 58, 28, 860


def tour_room(count: int) -> float:
    """How tall a tour column of ``count`` stops is at the least (px): its title and
    three rows, the current stop with a row for those before it and one for those after
    (all of them where there are fewer)."""
    return TOUR_HEAD + min(count, 3) * TOUR_ROW


def _stops(n: int) -> str:
    """``1 stop``, ``3 stops``."""
    return f"{n} stop" if n == 1 else f"{n} stops"


def tour_window(count: int, current: int, rows: int) -> tuple[int, int]:
    """Which of ``count`` stops a tour column of ``rows`` rows lists (first, end): all
    where they fit; else as many as fit with a row saying how many come before them and
    one how many after (where some do), the ``current`` one among them, as near their
    middle as the ends let it be."""
    if count <= rows:
        return 0, count
    shown = max(1, rows - 2)
    first = min(max(current - shown // 2, 0), count - shown)
    # At either end no row says what is hidden there: one more stop takes it.
    if first == 0:
        return 0, shown + 1
    if first + shown == count:
        return count - shown - 1, count
    return first, first + shown


#: The smallest size a footer line is set in to fit the width; a line that does not fit
#: even so is cut with an ellipsis.
FOOTER_MIN = 13
#: The smallest size the credit is set in on one line: a longer credit (the church's
#: singers and organ, each with its licence) takes two lines, so that it is read whole.
CREDIT_ONE_MIN = 16
#: The size of the footer's lines under a credit of two lines.
FOOTER_TWO = 16


def fitted(draw, line: str, size: int, width: float) -> tuple[str, ImageFont.FreeTypeFont]:
    """``line`` in the largest size from ``size`` down to :data:`FOOTER_MIN` that fits
    ``width`` pixels, cut with an ellipsis if none does."""
    for s in range(size, FOOTER_MIN - 1, -1):
        f = font("Regular", s)
        if draw.textlength(line, font=f) <= width:
            return line, f
    while line and draw.textlength(line + "…", font=f) > width:
        line = line[:-1]
    return line.rstrip() + "…", f


def credit_lines(draw, credit: str, width: float) -> list[str]:
    """``credit`` as the footer sets it: one line where it fits ``width`` pixels in
    :data:`CREDIT_ONE_MIN`, else two, split nearest its middle where each line fits the
    width in :data:`FOOTER_TWO`: at the `` · `` between its parts if one does, else
    after a semicolon, a comma or at a space; nearest the middle where none does."""
    if draw.textlength(credit, font=font("Regular", CREDIT_ONE_MIN)) <= width:
        return [credit]
    f = font("Regular", FOOTER_TWO)
    middle = len(credit) / 2
    for mark in (" · ", "; ", ", ", " "):
        cuts = [k for k in range(len(credit)) if credit.startswith(mark, k)]
        cuts.sort(key=lambda k: abs(k - middle))
        kept = mark.strip() if mark.strip() in (";", ",") else ""  # the · goes with the break
        for k in cuts:
            first, second = credit[:k] + kept, credit[k + len(mark) :]
            if max(draw.textlength(first, font=f), draw.textlength(second, font=f)) <= width:
                return [first, second]
    k = credit.rfind(" ", 0, round(middle) + 1)
    k = k if k > 0 else round(middle)
    return [credit[:k].rstrip(), credit[k:].lstrip()]


def footer(draw, credit: str, tool: str | Sequence[str], size: int = 19) -> None:
    """The spectrogram's panel (the video draws over it), the credit line and under it
    up to two lines that say how the sound was made (:func:`settings_lines`, or one
    line such as :func:`tool_line`'s). The credit is set in ``size`` pixels, the others
    in 18; each is made smaller, down to :data:`FOOTER_MIN`, to fit the still's width.
    A credit too long for one line (:func:`credit_lines`) takes two, and every line
    under the panel is then set in :data:`FOOTER_TWO` at most."""
    left, top, width, height = SPECTRUM_BOX
    draw.rounded_rectangle([left - 20, top - 7, left + width + 20, top + height + 7], 14, PANEL)
    lines = [tool] if isinstance(tool, str) else list(tool)
    if len(lines) > 2:
        raise ValueError("the footer has room for two lines about the sound")
    room = SIZE[0] - 2 * MARGIN
    credits = credit_lines(draw, credit, room)
    if len(credits) == 1:
        line, f = fitted(draw, credit, size, room)
        text(draw, (MARGIN, 1002), line, f, MUTED)
        y, step, tool_size = 1028, 24, 18
    else:
        for k, line in enumerate(credits):
            line, f = fitted(draw, line, FOOTER_TWO, room)
            text(draw, (MARGIN, 1000 + 19 * k), line, f, MUTED)
            y, step, tool_size = 1040, 20, FOOTER_TWO
    for k, line in enumerate(lines):
        line, f = fitted(draw, line, tool_size, room)
        text(draw, (MARGIN, y + step * k), line, f, MUTED)


def thumbnail_badge(draw, x: float, y: float, label: str = "BINAURAL", size: int = 34) -> None:
    """The pill a thumbnail carries."""
    pill = font("Bold", size)
    height = round(size * 1.75)
    pad = round(size * 0.7)
    draw.rounded_rectangle(
        [x, y, x + draw.textlength(label, font=pill) + 2 * pad, y + height], height / 2, LISTENER
    )
    text(draw, (x + pad, y + height / 2), label, pill, BG, "lm")


#: What each kind of diffraction (``simulation.early.diffraction``) bends round edges;
#: without it, a path an edge blocks is cut off at once as something moves.
DIFFRACTION = {
    "none": ", without diffraction round edges",
    "direct": ", the direct sound diffracted round edges",
    "first_order": ", the direct sound and first reflections diffracted round edges",
    "second_order": (
        ", the direct sound (over up to two edges) and first reflections diffracted round edges"
    ),
}


def _settings(simulation: dict):
    """A scene's ``simulation`` (or a run's settings) as the engine takes it, with the
    defaults of what it leaves out."""
    from acoustic_spaces.scene.model import Simulation

    return Simulation.model_validate(simulation or {})


def _early_text(early) -> str:
    """The early reflections as the simulation runs them (``simulation.early``): traced,
    with their rays, or image sources, to which order, and what diffracts."""
    if early.method == "image_sources":
        first = f"image sources to order {early.max_order}"
    else:
        first = f"early reflections traced to order {early.max_order} with {early.rays:,} rays"
    return first + DIFFRACTION[early.diffraction]


def _late_text(late) -> str:
    """The late sound as the simulation makes it (``simulation.late``)."""
    if late.method == "statistical":
        return "a statistical late sound"
    return f"stochastic ray tracing for the late sound, {late.rays:,} rays per band"


def _more_text(scene: dict, settings) -> str:
    """What else made the sound: the wave solver's bands, and outdoors in wind or a
    temperature gradient what the weather bent (the paths, the late rays, the parabolic
    equation over the open ground up to ``WAVE_MAX_BAND_HZ``, short of half the way to
    the shadow up to ``LIT_WAVE_MAX_BAND_HZ`` with ``open_ground_frequencies: rays``);
    empty for none."""
    from acoustic_spaces.engine.refracted import LIT_WAVE_MAX_BAND_HZ, WAVE_MAX_BAND_HZ
    from acoustic_spaces.scene.model import Atmosphere

    more = ""
    if settings.wave.enabled:
        more += f"; the bands up to {settings.wave.max_band_hz} Hz from the wave solver"
    outdoors = bool((scene.get("geometry") or {}).get("ground"))
    atmosphere = scene.get("atmosphere")
    if outdoors and atmosphere and not Atmosphere.model_validate(atmosphere).still:
        angle = settings.early.parabolic_equation.replace("_", "-")
        reach = f"{WAVE_MAX_BAND_HZ / 1000:g} kHz"
        if settings.early.open_ground_frequencies == "rays":
            reach = f"{LIT_WAVE_MAX_BAND_HZ:g} Hz ({reach} in and near shadows)"
        more += (
            "; paths and late rays bent by the wind and temperature; the open ground in the "
            f"bands up to {reach} by the {angle} parabolic equation"
        )
    return more


def ears_text(fmt: str, hrtf: str | Path | None) -> str:
    """What the mixes went through: ``binaural through the HRTFs of me.sofa``, ``binaural
    through a spherical head model``, or another format: ``as ambix3``."""
    if fmt != "binaural":
        return f"as {fmt}"
    ears = f"the HRTFs of {Path(hrtf).name}" if hrtf else "a spherical head model"
    return f"binaural through {ears}"


def tool_line(example: str, scene: dict, hrtf: str | Path | None = None) -> str:
    """How the sound was made, in one line, as the scene asks for it: the example, the
    early reflections, the late sound, what the wave solver and the weather did, and
    what the binaural mixes went through. :func:`settings_lines` says more, from a
    run's own settings."""
    settings = _settings(scene.get("simulation", {}))
    return (
        f"Simulated with acoustic-spaces ({example} example): {_early_text(settings.early)}; "
        f"{_late_text(settings.late)}{_more_text(scene, settings)}; "
        f"{ears_text('binaural', hrtf)}"
    )


def settings_lines(demo, args, scene: dict, report: dict | None = None) -> list[str]:
    """How the sound was made, for :func:`footer`, in two lines. First the early
    reflections, the late sound, the wave solver's bands and what the weather bent
    outdoors, as the simulation ran them (``report`` is the results' parameters.json;
    without it, as the scene asks, with the engine's defaults for what it leaves out).
    Then the tool, its version and the sample rate, what the mixes went through, and
    the command with the options the run was given, paths by their names, so that a
    video says which HRTFs and which song it came from."""
    settings = _settings((report or {}).get("settings") or scene.get("simulation", {}))
    rate = (report or {}).get("sample_rate") or settings.sample_rate
    fmt = getattr(args, "format", None) or "binaural"
    first = (
        f"Simulated: {_early_text(settings.early)}; {_late_text(settings.late)}"
        f"{_more_text(scene, settings)}"
    )
    run = demo.command("all", args, names_only=True)
    if not demo.arguments(args, names_only=True):
        run += " (every option at its default)"
    second = (
        f"acoustic-spaces {__version__} at {rate / 1000:g} kHz, mixed "
        f"{ears_text(fmt, getattr(args, 'hrtf', None))}. Run as: {run}"
    )
    return [first, second]


__all__ = [
    "BADGE",
    "BG",
    "CREDIT_ONE_MIN",
    "DIFFRACTION",
    "FAINT",
    "FOOTER_MIN",
    "FOOTER_TWO",
    "LINE",
    "LISTENER",
    "MARGIN",
    "MUTED",
    "PANEL",
    "RIGHT",
    "SIZE",
    "SOURCE",
    "SPECTRUM_BOX",
    "TEXT",
    "THUMBNAIL",
    "Plan",
    "arrow",
    "axes",
    "badge",
    "canvas",
    "cone",
    "credit_lines",
    "ears_text",
    "facing",
    "facts",
    "fitted_facts",
    "fitted",
    "font",
    "footer",
    "glow",
    "head",
    "heading",
    "mmss",
    "place",
    "rays",
    "ring",
    "save",
    "scale_bar",
    "settings_lines",
    "text",
    "thumbnail_badge",
    "title_card",
    "tool_line",
    "tour_column",
    "tour_room",
    "tour_window",
]
