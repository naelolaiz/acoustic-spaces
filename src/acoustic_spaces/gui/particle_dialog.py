"""The window that asks how to export sound particles to Blender: how much sound to
animate, how slowly, and how many particles, and in wind or a temperature gradient
whether to add the same particles in still air. Outdoors it starts with sound enough to
reach the farthest listener, slowed down to last as long as the default."""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gltf import PARTICLE_COUNT, SLOW_MOTION
from acoustic_spaces.gui.inspector import int_spin, spin
from acoustic_spaces.gui.numbers import fmt
from acoustic_spaces.i18n import _
from acoustic_spaces.room import Room
from acoustic_spaces.scene import Source

#: The sound the export animates where the scene needs no more, ms.
DURATION_MS = 100.0


def reach_ms(room: Room, source: Source) -> float:
    """The sound to animate from ``source``, ms: outdoors, a tenth more than the
    direct sound takes to reach the farthest listener in use, rounded up to 100 ms;
    indoors, or when that is shorter, :data:`DURATION_MS`."""
    listeners = [item.position for item in room.scene.listeners if item.in_use]
    if room.outdoor is None or not listeners:
        return DURATION_MS
    start = np.asarray(source.position, dtype=float)
    farthest = float(np.linalg.norm(np.asarray(listeners, dtype=float) - start, axis=1).max())
    needed = 1.1e3 * farthest / room.speed_of_sound()
    return max(DURATION_MS, 100.0 * math.ceil(needed / 100.0))


def slow_motion_for(duration_ms: float) -> float:
    """The slow motion that makes ``duration_ms`` of sound last as long as the default
    sound does at the default slow motion, at least 1 (real time)."""
    return max(1.0, round(SLOW_MOTION * DURATION_MS / duration_ms, 1))


class ParticleDialog(QDialog):
    """The particle animation's settings for ``room``'s ``source``: :attr:`duration`
    (ms), :attr:`slow_motion`, :attr:`count` and :attr:`still_air` give what OK chose."""

    def __init__(self, room: Room, source: Source, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Export particle animation to Blender"))
        layout = QVBoxLayout(self)
        intro = QLabel(
            _(
                "Sound particles leave {source}: how much of the sound, how slowly, how many?"
            ).format(source=source.name)
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        duration = reach_ms(room, source)
        self.sound = spin(1.0, 60_000.0, 0, 100.0, "ms")
        self.sound.setValue(duration)
        self.sound.setToolTip(
            _(
                "How much of the sound to animate, from when it leaves the source. Outdoors "
                "it starts long enough to reach the farthest listener"
            )
        )
        self.slow = spin(1.0, 10_000.0, 1, 1.0, "×")
        self.slow.setValue(slow_motion_for(duration))
        self.slow.setToolTip(
            _(
                "Seconds of animation per second of sound: 40 shows 100 ms of sound over "
                "4 s, 1 in real time"
            )
        )
        self.particles = int_spin(10, 100_000, 100)
        self.particles.setValue(PARTICLE_COUNT)
        self.particles.setToolTip(
            _("How many particles leave the source: 1000 make a file of about 3 MB")
        )
        for text, box in (
            (_("Sound to animate"), self.sound),
            (_("Slow motion"), self.slow),
            (_("Particles"), self.particles),
        ):
            label = QLabel(text)
            label.setToolTip(box.toolTip())
            form.addRow(label, box)
        layout.addLayout(form)
        atmosphere = room.scene.atmosphere
        bent = atmosphere is not None and not atmosphere.still
        self.compare = QCheckBox(_("Compare with still air"))
        self.compare.setChecked(bent)
        self.compare.setEnabled(bent)
        self.compare.setToolTip(
            _(
                "Adds the same particles traced in still air, in blue, beside those in the "
                "scene's wind and temperature, which bend them: the wind is a few metres a "
                "second against the sound's 343, so the two part by metres, near the ground"
            )
        )
        layout.addWidget(self.compare)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.sound.valueChanged.connect(self._summarize)
        self.slow.valueChanged.connect(self._summarize)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._summarize()

    @property
    def duration(self) -> float:
        return float(self.sound.value())

    @property
    def slow_motion(self) -> float:
        return float(self.slow.value())

    @property
    def count(self) -> int:
        return int(self.particles.value())

    @property
    def still_air(self) -> bool:
        return self.compare.isEnabled() and self.compare.isChecked()

    def _summarize(self) -> None:
        self.summary.setText(
            fmt(
                _("The animation lasts {seconds:g} s."),
                seconds=round(self.duration * self.slow_motion / 1e3, 2),
            )
        )


__all__ = ["DURATION_MS", "ParticleDialog", "reach_ms", "slow_motion_for"]
