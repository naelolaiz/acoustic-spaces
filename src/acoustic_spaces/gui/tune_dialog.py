"""The window that shows what Tune CPU and GPU timed and asks what to keep: the timings
and this computer's default device are saved on OK only; Cancel saves nothing."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.engine import tuning
from acoustic_spaces.i18n import N_, _

#: The check box that makes ``auto`` this computer's default.
MAKE_DEFAULT = N_("Make auto the default on this computer")
#: The check box that sets the open scene's devices back to the default.
FOLLOW = N_("Let this scene follow this computer's default too")


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setIndent(24)  # under the check box's text
    return label


class TuneDialog(QDialog):
    """The timings of ``result`` (a :class:`~acoustic_spaces.engine.tuning.Tuning`),
    with a check box for this computer's default and, when the open scene names its
    own devices (``late`` and ``wave``, None where it names none), one that lets it
    follow the default. Nothing is saved until OK."""

    def __init__(
        self,
        result: tuning.Tuning,
        late: str | None,
        wave: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.timings = result
        self.setWindowTitle(_("Tune CPU and GPU"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("What the CPU and the GPU took on this computer:")))
        table = QLabel(result.summary())
        table.setTextFormat(Qt.TextFormat.PlainText)
        table.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        table.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(table)

        self.make_default = QCheckBox(_(MAKE_DEFAULT))
        self.make_default.setChecked(True)
        self.make_default.setToolTip(
            _(
                "Scenes whose Run on is default then run on the fastest device found here; "
                "unticked, on the CPU"
            )
        )
        layout.addSpacing(8)
        layout.addWidget(self.make_default)
        current = tuning.default_device()
        layout.addWidget(
            _note(
                _(
                    "Scenes whose Run on is set to default, in the editor and on the command "
                    "line, then run the late reverberation and the wave solver on the "
                    "devices marked auto above. Unticked, they run on the CPU. "
                    "The default now: {device}."
                ).format(device=current)
            )
        )

        self.follow: QCheckBox | None = None
        if late is not None and wave is not None:
            # Translators: {late} and {wave} are devices, such as cpu or gpu.
            named = _("late reverberation on {late} and wave solver on {wave}")
        elif late is not None:
            named = _("late reverberation on {late}")
        else:
            named = _("wave solver on {wave}")
        named = named.format(late=late, wave=wave)
        if late is not None or wave is not None:
            self.follow = QCheckBox(_(FOLLOW))
            self.follow.setChecked(True)
            self.follow.setToolTip(
                _(
                    "Set this scene's Run on back to default, so that it follows this computer's "
                    "default"
                )
            )
            layout.addWidget(self.follow)
            layout.addWidget(
                _note(
                    _(
                        "This scene names its own devices ({devices}), so the default does "
                        "not apply to it. Ticked, OK sets its Run on back to default "
                        "(Edit > Undo takes that back)."
                    ).format(devices=named)
                )
            )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setToolTip(
            _("Save the timings and the choices above in {path}").format(path=result.file)
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setToolTip(
            _("Close without saving anything, not even the timings")
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addSpacing(8)
        layout.addWidget(
            QLabel(_("OK saves the timings and the choices above. Cancel saves nothing."))
        )
        layout.addWidget(buttons)
        self.buttons = buttons

    def default(self) -> str:
        """The default device the window asks for: auto or cpu."""
        return "auto" if self.make_default.isChecked() else "cpu"

    def follows(self) -> bool:
        """Whether the open scene should drop its own devices for the default."""
        return self.follow is not None and self.follow.isChecked()
