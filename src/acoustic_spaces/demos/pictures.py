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


def text(draw, xy, s, f, fill=TEXT, anchor="la") -> None:
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


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
    plan, a section)."""
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


def facts(draw, x: float, y: float, lines: Sequence, size: int = 24, step: int = 36) -> float:
    """Lines of facts under the stop's name, each a string or (string, colour); returns
    the y under them."""
    for line in lines:
        s, colour = line if isinstance(line, tuple) else (line, TEXT)
        weight = "SemiBold" if colour == LISTENER else "Regular"
        text(draw, (x, y), s, font(weight, size), colour)
        y += step
    return y


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
    one bright, the ones to come faint, in rows of ``row`` pixels, closer when they
    would pass ``bottom``."""
    text(draw, (x, y), title, font("SemiBold", 22), MUTED)
    y += 36
    row = min(row, (bottom - y) // max(1, len(entries)))
    for j, (label, t0) in enumerate(entries):
        colour = LISTENER if j == current else (MUTED if j < current else FAINT)
        weight = "SemiBold" if j == current else "Regular"
        mark = "▶ " if j == current else "   "
        text(draw, (x, y), mark + label, font(weight, size), colour)
        text(draw, (right, y), mmss(t0), font(weight, size), colour, "ra")
        y += row


#: The smallest size a footer line is set in to fit the width; a line that does not fit
#: even so is cut with an ellipsis.
FOOTER_MIN = 13


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


def footer(draw, credit: str, tool: str | Sequence[str], size: int = 19) -> None:
    """The spectrogram's panel (the video draws over it), the credit line and under it
    up to two lines that say how the sound was made (:func:`settings_lines`, or one
    line such as :func:`tool_line`'s). The credit is set in ``size`` pixels, the others
    in 18; each is made smaller, down to :data:`FOOTER_MIN`, to fit the still's width."""
    left, top, width, height = SPECTRUM_BOX
    draw.rounded_rectangle([left - 20, top - 7, left + width + 20, top + height + 7], 14, PANEL)
    lines = [tool] if isinstance(tool, str) else list(tool)
    if len(lines) > 2:
        raise ValueError("the footer has room for two lines about the sound")
    room = SIZE[0] - 2 * MARGIN
    line, f = fitted(draw, credit, size, room)
    text(draw, (MARGIN, 1002), line, f, MUTED)
    for k, line in enumerate(lines):
        line, f = fitted(draw, line, 18, room)
        text(draw, (MARGIN, 1028 + 24 * k), line, f, MUTED)


def thumbnail_badge(draw, x: float, y: float, label: str = "BINAURAL", size: int = 34) -> None:
    """The pill a thumbnail carries."""
    pill = font("Bold", size)
    height = round(size * 1.75)
    pad = round(size * 0.7)
    draw.rounded_rectangle(
        [x, y, x + draw.textlength(label, font=pill) + 2 * pad, y + height], height / 2, LISTENER
    )
    text(draw, (x + pad, y + height / 2), label, pill, BG, "lm")


#: What each kind of diffraction (``simulation.early.diffraction``) bends round edges.
DIFFRACTION = {
    "none": "",
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
    equation over the open ground in the low bands); empty for none."""
    from acoustic_spaces.engine.refracted import SHADOW_WAVE_MAX_BAND_HZ, WAVE_MAX_BAND_HZ
    from acoustic_spaces.scene.model import Atmosphere

    more = ""
    if settings.wave.enabled:
        more += f"; the bands up to {settings.wave.max_band_hz} Hz from the wave solver"
    outdoors = bool((scene.get("geometry") or {}).get("ground"))
    atmosphere = scene.get("atmosphere")
    if outdoors and atmosphere and not Atmosphere.model_validate(atmosphere).still:
        angle = settings.early.parabolic_equation.replace("_", "-")
        more += (
            "; paths and late rays bent by the wind and temperature; the open ground in the "
            f"bands up to {WAVE_MAX_BAND_HZ:g} Hz ({SHADOW_WAVE_MAX_BAND_HZ / 1000:g} kHz in "
            f"shadows) by the {angle} parabolic equation"
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
    "DIFFRACTION",
    "FAINT",
    "FOOTER_MIN",
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
    "ears_text",
    "facing",
    "facts",
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
    "tool_line",
    "tour_column",
]
