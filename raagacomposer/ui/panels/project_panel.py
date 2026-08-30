"""Project header panel (spec section 14A)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout)


class ProjectPanel(QGroupBox):
    newRequested = Signal()
    openRequested = Signal()
    openPathRequested = Signal(str)
    saveRequested = Signal()
    saveAsRequested = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__("Project", parent)
        self.app = app

        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Song title")
        self.title_edit.editingFinished.connect(self._title_changed)

        self.stage_label = QLabel("-")
        self.stage_label.setObjectName("hint")
        self.path_label = QLabel("(not saved yet)")
        self.path_label.setObjectName("hint")
        self.path_label.setWordWrap(True)

        self.recent = QComboBox()
        self.recent.setToolTip("Recently opened projects")
        self.recent.activated.connect(self._recent_chosen)

        buttons = QHBoxLayout()
        for text, signal, primary in (("New", self.newRequested, False),
                                      ("Open", self.openRequested, False),
                                      ("Save", self.saveRequested, True),
                                      ("Save As", self.saveAsRequested, False)):
            b = QPushButton(text)
            if primary:
                b.setObjectName("primary")
            b.clicked.connect(signal.emit)
            buttons.addWidget(b)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_edit)
        layout.addWidget(self.stage_label)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Recent:"))
        layout.addWidget(self.recent)
        layout.addWidget(self.path_label)
        self.refresh()

    def _title_changed(self) -> None:
        text = self.title_edit.text().strip()
        if text and text != self.app.project.title:
            self.app.project.title = text
            self.app._changed("project.title", f"Renamed to {text!r}")

    def _recent_chosen(self, index: int) -> None:
        path = self.recent.itemData(index)
        if path:
            self.openPathRequested.emit(str(path))

    def refresh(self) -> None:
        project = self.app.project
        if self.title_edit.text() != project.title and not self.title_edit.hasFocus():
            self.title_edit.setText(project.title)
        dirty = " *" if self.app.dirty else ""
        self.stage_label.setText(
            f"Stage: {project.current_stage.value}{dirty}   "
            f"Duration: {project.duration:.0f}s")
        self.path_label.setText(str(self.app.project_dir or "(not saved yet)"))

        current = self.recent.currentData()
        self.recent.blockSignals(True)
        self.recent.clear()
        for entry in self.app.recent_projects():
            self.recent.addItem(f"{entry['title']}  -  {Path(entry['directory']).name}",
                                entry["directory"])
        idx = self.recent.findData(current)
        if idx >= 0:
            self.recent.setCurrentIndex(idx)
        self.recent.blockSignals(False)
