"""Lyrics panel (spec section 14E).

What the creator reads here has to answer four questions without a
glossary: which version is approved and in use, whether the latest
version is a draft and why, which lines are missing and where, and who
wrote each line.  A writer that answers short leaves lines missing; the
application keeps such a version as a draft with the approved words
untouched, and this panel is where that has to be unmistakable.

Viewing a version is not approving it.  The version list shows any
version; only a complete one becomes the approved words by being chosen,
and Accept lyrics refuses a draft with missing lines and says so.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QTextEdit, QVBoxLayout, QWidget)

COLUMNS = ["Section", "Start", "Syllables", "Line", "Written by", "Locked"]
LINE_COLUMN = COLUMNS.index("Line")


def written_by(source: str) -> str:
    """A line's provenance in the creator's words."""
    if not source:
        return "not recorded"
    if source.startswith("llm:"):
        return source[4:] or "a writer"
    if source.startswith("missing:"):
        return f"missing ({source[8:] or 'the writer'} returned nothing)"
    return source


def missing_writer(lyrics) -> str:
    """Who left the missing lines, for the standing line."""
    for line in lyrics.lines:
        if line.source.startswith("missing:"):
            return line.source[8:] or "the writer"
    return "the writer"


class LyricsPanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self._loading = False
        #: The version on the table, when it is not the approved one.
        #: Viewing a draft must not make it the song's words.
        self._viewing: Optional[int] = None

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

        #: One or two sentences on what stands: approved, draft, missing.
        self.standing = QLabel()
        self.standing.setWordWrap(True)
        self.standing.setObjectName("lyricsStanding")

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._item_changed)

        self.alignment = QTextEdit()
        self.alignment.setReadOnly(True)
        self.alignment.setFixedHeight(110)

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
        layout.addWidget(self.standing)
        layout.addWidget(QLabel(
            "Lines are fitted to the tune. Edit the Line column directly; "
            "the syllables are re-fitted to the same notes."))
        layout.addWidget(self.table, 1)
        layout.addLayout(row)
        layout.addWidget(QLabel("Fit report:"))
        layout.addWidget(self.alignment)
        self.setMinimumWidth(520)
        self.setMinimumHeight(280)
        self.refresh()

    # -- which version is on the table ------------------------------------
    def shown(self):
        """The version the table shows: the one being viewed, else the
        approved one, else the latest."""
        project = self.app.project
        if self._viewing is not None:
            found = project.lyrics_version(self._viewing)
            if found is not None:
                return found
            self._viewing = None
        return project.lyrics_version()

    # -- actions -----------------------------------------------------------
    def _accept(self) -> None:
        shown = self.shown()
        if shown is None:
            return
        if self.app.accept_lyrics(version=shown.version):
            self._viewing = None
        self.refresh()
        self.changed.emit()

    def _version_chosen(self, index: int) -> None:
        version = self.versions.itemData(index)
        if version is None:
            return
        chosen = self.app.project.lyrics_version(int(version))
        if chosen is None:
            return
        if chosen.unfitted:
            # Looking at a draft is not choosing it.  The approved words
            # stay in use until the missing lines are written and the
            # draft is accepted.
            self._viewing = chosen.version
            self.app.status(f"Viewing lyrics v{chosen.version}, a draft with "
                            f"{chosen.unfitted} missing line(s); v"
                            f"{self.app.project.approved_lyrics} stays approved.")
        else:
            self._viewing = None
            self.app.project.approved_lyrics = chosen.version
            self.app._changed("lyrics.select", f"Switched to lyrics v{chosen.version}")
        self.refresh()
        self.changed.emit()

    def _current_line(self):
        lyrics = self.shown()
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
        if self._loading or item.column() != LINE_COLUMN:
            return
        lyrics = self.shown()
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
    def _standing_text(self, shown, melody) -> str:
        project = self.app.project
        approved = project.approved_lyrics
        if shown is None:
            return "No lyrics yet."
        latest = project.lyrics[-1] if project.lyrics else None
        parts = []
        if shown.unfitted:
            names = []
            for line in shown.lines:
                if line.unfitted:
                    section = melody.section_by_id(line.section_id) if melody else None
                    names.append(section.name if section else "?")
            where = ", ".join(dict.fromkeys(names))
            parts.append(f"v{shown.version} is a draft: {shown.unfitted} of "
                         f"{len(shown.lines)} requested line(s) missing ({where}); "
                         f"{missing_writer(shown)} returned nothing for them.")
            if approved is not None and approved != shown.version:
                parts.append(f"Your approved words (v{approved}) are unchanged and "
                             f"still in use.")
            else:
                parts.append("No lyrics are approved yet.")
            parts.append("Write the missing lines here, or rewrite them, then "
                         "Accept lyrics.")
        elif shown.version == approved:
            parts.append(f"v{shown.version} is approved: {len(shown.lines)} lines fitted.")
        elif approved is None:
            parts.append(f"v{shown.version}: {len(shown.lines)} lines fitted; not yet "
                         f"accepted.")
        else:
            parts.append(f"Viewing v{shown.version} ({len(shown.lines)} lines, complete); "
                         f"v{approved} is approved. Accept lyrics to use v{shown.version}.")
        if latest is not None and latest.unfitted and latest.version != shown.version:
            parts.append(f"The latest version, v{latest.version}, is a draft with "
                         f"{latest.unfitted} missing line(s) - {missing_writer(latest)} "
                         f"returned nothing for them. Choose it in the version list "
                         f"to see and fix it; nothing here has changed.")
        return " ".join(parts)

    def refresh(self) -> None:
        self._loading = True
        project = self.app.project
        shown = self.shown()
        melody = project.melody()
        approved = project.approved_lyrics

        self.versions.blockSignals(True)
        self.versions.clear()
        for lv in project.lyrics:
            if lv.version == approved:
                label = f"v{lv.version} - approved, {len(lv.lines)} lines"
            elif lv.unfitted:
                label = (f"v{lv.version} - draft, {lv.unfitted} missing of "
                         f"{len(lv.lines)} lines")
            else:
                label = f"v{lv.version} - {len(lv.lines)} lines, not accepted"
            self.versions.addItem(label, lv.version)
        if shown is not None:
            idx = self.versions.findData(shown.version)
            self.versions.setCurrentIndex(idx if idx >= 0 else 0)
        self.versions.blockSignals(False)

        self.standing.setText(self._standing_text(shown, melody))
        self.generate_btn.setEnabled(melody is not None)
        self.accept_btn.setEnabled(shown is not None and not shown.unfitted)
        self.accept_btn.setToolTip(
            "Write the missing lines first." if shown is not None and shown.unfitted
            else "")
        lines = shown.lines if shown else []
        self.table.setRowCount(len(lines))
        for row, line in enumerate(lines):
            section = melody.section_by_id(line.section_id) if melody else None
            sung = len([s for s in line.syllables if not s.startswith("~")])
            if line.unfitted and not line.text.strip():
                fit = "MISSING"
            elif line.unfitted:
                fit = "unfitted"
            else:
                fit = f"{sung}/{len(line.note_indices)}"
            values = [section.name if section else "-", f"{line.start:.1f}s", fit,
                      line.text, written_by(line.source), "yes" if line.locked else ""]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col != LINE_COLUMN:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if line.unfitted:
                    item.setBackground(QColor(255, 235, 205))
                    if col == LINE_COLUMN:
                        item.setToolTip("No words were written for this line. "
                                        "Type them here, or use Rewrite this line.")
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        self.table.setColumnWidth(LINE_COLUMN, max(260, self.table.columnWidth(LINE_COLUMN)))
        report = self.app.lyric_alignment(version=shown.version) if shown else "No lyrics yet."
        if shown is not None and shown.notes:
            report = shown.notes + "\n\n" + report
        self.alignment.setPlainText(report)
        self._loading = False
