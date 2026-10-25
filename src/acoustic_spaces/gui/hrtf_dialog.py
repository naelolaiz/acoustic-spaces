"""The window behind Download HRTFs…: the free HRTF sets
(:mod:`acoustic_spaces.spatial.hrtf_sets`), what each one is and what it allows, and a
download that keeps the editor working while it runs.

Measured ears place sounds better than the spherical head model, above all front from
back. Every set on offer is published under a free licence; the licence and who
measured it are on the row, because a work that uses the set owes them a credit. The
file goes into the user's data folder, and the window hands its path back to whoever
opened it: the binaural output of the scene, or the demo window's HRTF field.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from acoustic_spaces.gui.sizing import fit_width, scroll_bar_width
from acoustic_spaces.gui.worker import DownloadJob
from acoustic_spaces.i18n import _
from acoustic_spaces.spatial import hrtf_sets


class HrtfDialog(QDialog):
    """The HRTF sets on offer. :attr:`path` is the file the window ends on: the set that
    was downloaded or is there already, or None."""

    def __init__(self, parent: QWidget | None = None, folder: str | Path | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_("Download HRTFs"))
        self.folder = Path(folder) if folder is not None else hrtf_sets.data_folder()
        #: The file the window ends on, for whoever opened it.
        self.path: Path | None = None
        self.job: DownloadJob | None = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            _(
                "Head-related transfer functions (HRTF) are how a head and its ears colour "
                "sound from every direction. Rendering through a measured set places sounds "
                "better than the spherical head model, front from back above all. Every set "
                "here is free to use; its licence asks for the credit on the row."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.list = QListWidget()
        self.list.setToolTip(_("The HRTF sets on offer; each row says whose ears and whose terms"))
        for item in hrtf_sets.sets():
            entry = QListWidgetItem()
            entry.setData(Qt.ItemDataRole.UserRole, item.name)
            entry.setToolTip(f"{item.note}\n{item.credit}\n{item.page}")
            self.list.addItem(entry)
        layout.addWidget(self.list)

        self.heads = QComboBox()
        self.heads.setToolTip(_("Which head of a set that holds several"))
        self.heads_label = QLabel(_("Head"))
        self.heads_label.setBuddy(self.heads)
        self.heads_label.setToolTip(self.heads.toolTip())
        heads_row = QHBoxLayout()
        heads_row.addWidget(self.heads_label)
        heads_row.addWidget(self.heads, 1)
        layout.addLayout(heads_row)

        other = QHBoxLayout()
        other_label = QLabel(_("Another SOFA file"))
        self.url = QLineEdit()
        self.url.setPlaceholderText(_("https://…/ears.sofa"))
        self.url.setToolTip(
            _("Any other SOFA file, by its address: your own measurements, or a set from a site")
        )
        other_label.setBuddy(self.url)
        other_label.setToolTip(self.url.toolTip())
        other.addWidget(other_label)
        other.addWidget(self.url, 1)
        layout.addLayout(other)

        self.bar = QProgressBar()
        self.bar.setToolTip(_("How much of the download is done"))
        self.bar.hide()
        layout.addWidget(self.bar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.where = QLabel(_("The sets are kept in {folder}").format(folder=self.folder))
        self.where.setWordWrap(True)
        layout.addWidget(self.where)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close
        )
        self.download = QPushButton(_("Download"))
        self.download.setToolTip(_("Fetch the set on the row, or the address above"))
        self.download.clicked.connect(self._download)
        self.buttons.addButton(self.download, QDialogButtonBox.ButtonRole.ActionRole)
        self.use = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.use.setText(_("Use"))
        self.use.setToolTip(
            _("Take the file just downloaded, or else the set on the row, and close")
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setToolTip(
            _("Close; what was downloaded stays in the folder")
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.list.currentRowChanged.connect(self._chose)
        self.list.itemDoubleClicked.connect(self._download)
        self.url.textChanged.connect(self._refresh)
        self.heads.currentIndexChanged.connect(self._refresh)
        self.list.setCurrentRow(0)
        self._refresh()
        # Wide enough for the rows, whose credits are long, and for their scroll bar.
        margins = layout.contentsMargins()
        rows = self.list.sizeHintForColumn(0) + 2 * self.list.frameWidth()
        fit_width(self, rows + scroll_bar_width(self.list) + margins.left() + margins.right())

    # -- the rows -------------------------------------------------------------------

    @property
    def chosen(self):
        """The set on the row."""
        row = max(self.list.currentRow(), 0)
        return hrtf_sets.sets()[row]

    def member(self) -> str | None:
        """Which head of the set on the row, for a set that holds several."""
        item = self.chosen
        return self.heads.currentText() if len(item.members) > 1 else item.member

    def file(self) -> Path:
        """Where the set on the row goes."""
        return hrtf_sets.file_of(self.chosen, self.member(), self.folder)

    def _chose(self) -> None:
        item = self.chosen
        self.heads.blockSignals(True)
        self.heads.clear()
        if len(item.members) > 1:
            self.heads.addItems(list(item.members))
        self.heads.blockSignals(False)
        several = len(item.members) > 1
        self.heads.setVisible(several)
        self.heads_label.setVisible(several)
        self._refresh()

    def _refresh(self) -> None:
        """The rows, with what is there already, and what the buttons can do now."""
        for row, item in enumerate(hrtf_sets.sets()):
            here = hrtf_sets.file_of(item, None, self.folder).exists()
            mark = _("downloaded") if here else _("not here yet")
            by = _("by {credit}").format(credit=item.credit)
            self.list.item(row).setText(f"{item.summary()}; {by} — {mark}")
        running = self.job is not None
        url = self.url.text().strip()
        self.download.setEnabled(not running and (bool(url) or not self.file().exists()))
        self.use.setEnabled(not running and (self.path is not None or self.file().exists()))
        if not running and not self.status.text() and self.file().exists():
            self.status.setText(_("{file} is here already").format(file=self.file()))

    # -- the download ---------------------------------------------------------------

    def _download(self) -> None:
        if self.job is not None:
            return
        url = self.url.text().strip()
        if url:
            name = Path(url.split("?")[0]).stem or "hrtf"
            job = DownloadJob(url, self.folder / f"{name}.sofa")
            self.status.setText(_("Downloading {name}…").format(name=name))
        else:
            item = self.chosen
            job = DownloadJob(item.url, self.file(), member=self.member())
            self.status.setText(
                _("Downloading {title} ({licence}, by {credit})…").format(
                    title=item.title, licence=item.licence, credit=item.credit
                )
            )
        self.job = job
        self.bar.setRange(0, 0)
        self.bar.setValue(0)
        self.bar.show()
        job.progress.connect(self._step)
        job.done.connect(self._downloaded)
        job.failed.connect(self._failed)
        job.finished.connect(self._ended)
        self._refresh()
        job.start()

    def _step(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
        else:
            self.bar.setRange(0, 0)

    def _downloaded(self, path: str) -> None:
        self.path = Path(path)
        self.status.setText(_("Downloaded {file}").format(file=path))

    def _failed(self, message: str) -> None:
        self.status.setText(_("The download failed: {reason}").format(reason=message))

    def _ended(self) -> None:
        job, self.job = self.job, None
        if job is not None:
            job.wait()
        self.bar.hide()
        self._refresh()

    # -- closing --------------------------------------------------------------------

    def accept(self) -> None:
        if self.path is None and self.file().exists():
            self.path = self.file()
        super().accept()

    def reject(self) -> None:
        if self.job is not None:
            self.job.cancel()
            self.job.wait()
        super().reject()


__all__ = ["HrtfDialog"]
