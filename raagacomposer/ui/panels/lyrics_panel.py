"""Lyrics panel (spec section 14E)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QTextEdit, QVBoxLayout, QWidget)


class LyricsPanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self._loading = False

        self.generate_btn = QPushButton("Write lyrics for the locked tune")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.clicked.connect(lambda: self.app.generate_lyrics())
        self.accept_btn = QPushButton("Accept lyrics")
        self.accept_btn.clicked.connect(self._accept)
        self.regen_line_btn = QPushButton("Rewrite this line")
        self.regen_line_btn.clicked.connect(self._regenerate_line)
        self.lock_line_btn = QPushButton("Lock / unlock line")
        self.lock_line_btn.clicked.connect(self._toggle_lock)
        self.play_line_btn = QPushButton("Play this line")
        self.play_line_btn.clicked.connect(self._play_line)

        self.versions = QComboBox()
        self.versions.activated.connect(self._version_chosen)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Section", "Start", "Syllables", "Line", "Locked"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._item_changed)

        self.alignment = QTextEdit()
        self.alignment.setReadOnly(True)
        self.alignment.setFixedHeight(88)

        top = QHBoxLayout()
        top.addWidget(self.generate_btn)
        top.addWidget(self.accept_btn)
        top.addWidget(QLabel("Version:"))
        top.addWidget(self.versions, 1)

        row = QHBoxLayout()
        row.addWidget(self.regen_line_btn)
        row.addWidget(self.lock_line_btn)
        row.addWidget(self.play_line_btn)
        row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(QLabel(
            "Lines are fitted to the tune. Edit the Line column directly; "
            "the syllables are re-fitted to the same notes."))
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        layout.addWidget(QLabel("Fit report:"))
        layout.addWidget(self.alignment)
        self.setMinimumWidth(520)
        self.setMinimumHeight(260)
        self.refresh()

    # -- actions -----------------------------------------------------------
    def _accept(self) -> None:
        self.app.accept_lyrics()
        self.changed.emit()

    def _version_chosen(self, index: int) -> None:
        version = self.versions.itemData(index)
        if version is not None:
            self.app.project.approved_lyrics = int(version)
            self.app._changed("lyrics.select", f"Switched to lyrics v{version}")
            self.changed.emit()

    def _current_line(self):
        lyrics = self.app.project.lyrics_version()
        row = self.table.currentRow()
        if lyrics is None or row < 0 or row >= len(lyrics.lines):
            return None
        return lyrics.lines[row]

    def _regenerate_line(self) -> None:
        line = self._current_line()
        if line is None:
            QMessageBox.information(self, "Lyrics", "Select a line first.")
            return
        try:
            warnings = self.app.regenerate_lyric_line(line.id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Lyrics", str(exc))
            return
        if warnings:
            self.app.status("; ".join(warnings))
        self.refresh()
        self.changed.emit()

    def _toggle_lock(self) -> None:
        line = self._current_line()
        if line is None:
            return
        self.app.set_lyric_line_lock(line.id, not line.locked)
        self.refresh()

    def _play_line(self) -> None:
        line = self._current_line()
        if line is None:
            return
        self.app.play_range(line.start, line.end)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading or item.column() != 3:
            return
        lyrics = self.app.project.lyrics_version()
        if lyrics is None or item.row() >= len(lyrics.lines):
            return
        line = lyrics.lines[item.row()]
        try:
            warnings = self.app.edit_lyric_line(line.id, item.text())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Lyrics", str(exc))
            self.refresh()
            return
        if warnings:
            self.app.status("; ".join(warnings))
        self.refresh()

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        self._loading = True
        project = self.app.project
        lyrics = project.lyrics_version()
        melody = project.melody()

        self.versions.blockSignals(True)
        self.versions.clear()
        for lv in project.lyrics:
            self.versions.addItem(f"v{lv.version}  {len(lv.lines)} lines "
                                  f"[{lv.state.value}]", lv.version)
        if lyrics is not None:
            idx = self.versions.findData(lyrics.version)
            self.versions.setCurrentIndex(idx if idx >= 0 else 0)
        self.versions.blockSignals(False)

        self.generate_btn.setEnabled(melody is not None)
        lines = lyrics.lines if lyrics else []
        self.table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            section = melody.section_by_id(line.section_id) if melody else None
            sung = len([s for s in line.syllables if not s.startswith("~")])
            values = [section.name if section else "-", f"{line.start:.1f}s",
                      f"{sung}/{len(line.note_indices)}", line.text,
                      "yes" if line.locked else ""]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col != 3:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(3, max(260, self.table.columnWidth(3)))
        self.alignment.setPlainText(self.app.lyric_alignment())
        self._loading = False
