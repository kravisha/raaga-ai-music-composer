"""Tune panel (spec section 14D)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QGroupBox, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from ...music import beat as beat_engine


class TunePanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        self.generate_btn = QPushButton("Generate tune")
        self.generate_btn.setObjectName("primary")
        self.generate_btn.clicked.connect(lambda: self.app.generate_tune())
        self.variation_btn = QPushButton("Variation")
        self.variation_btn.clicked.connect(lambda: self.app.make_variation())
        self.play_btn = QPushButton("Play tune")
        self.play_btn.clicked.connect(lambda: self.app.play_render("tune"))
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.app.stop)
        self.accept_btn = QPushButton("Accept and lock")
        self.accept_btn.clicked.connect(self._accept)

        # The beat is its own layer (specification 11): made, heard and
        # varied without touching the tune, because it is written against
        # the tala rather than against the melody's notes.
        self.beat_btn = QPushButton("Generate beat")
        self.beat_btn.clicked.connect(lambda: self.app.generate_beat(autoplay=True))
        self.beat_variation_btn = QPushButton("Beat variation")
        self.beat_variation_btn.clicked.connect(
            lambda: self.app.beat_variation(self.beat_strength.currentText(),
                                            autoplay=True))
        self.beat_strength = QComboBox()
        self.beat_strength.addItems(list(beat_engine.STRENGTHS))
        self.beat_strength.setCurrentText(beat_engine.DEFAULT_STRENGTH)
        self.play_beat_btn = QPushButton("Play beat")
        self.play_beat_btn.clicked.connect(lambda: self.app.play_render("beat"))
        self.beat_label = QLabel("No beat yet")
        self.beat_label.setObjectName("hint")

        self.versions = QComboBox()
        self.versions.activated.connect(self._version_chosen)

        self.tempo = QSpinBox()
        self.tempo.setRange(30, 220)
        self.tempo.setSuffix(" bpm")
        tempo_btn = QPushButton("Set tempo")
        tempo_btn.clicked.connect(lambda: self.app.set_tempo(int(self.tempo.value())))

        self.sections = QTableWidget(0, 5)
        self.sections.setHorizontalHeaderLabels(
            ["Section", "Start", "End", "Locked", "Notes"])
        self.sections.horizontalHeader().setStretchLastSection(True)
        self.sections.verticalHeader().setVisible(False)
        self.sections.setSelectionBehavior(QTableWidget.SelectRows)

        regen_btn = QPushButton("Regenerate selected section")
        regen_btn.clicked.connect(self._regenerate_section)
        lock_btn = QPushButton("Lock / unlock section")
        lock_btn.clicked.connect(self._toggle_section_lock)
        play_section_btn = QPushButton("Play section")
        play_section_btn.clicked.connect(self._play_section)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setFixedHeight(90)

        top = QHBoxLayout()
        for w in (self.generate_btn, self.variation_btn, self.play_btn,
                  self.stop_btn, self.accept_btn):
            top.addWidget(w)
        top.addStretch(1)

        second = QHBoxLayout()
        second.addWidget(QLabel("Version:"))
        second.addWidget(self.versions, 1)
        second.addWidget(self.tempo)
        second.addWidget(tempo_btn)

        beat_row = QHBoxLayout()
        beat_row.addWidget(self.beat_btn)
        beat_row.addWidget(self.beat_variation_btn)
        beat_row.addWidget(self.beat_strength)
        beat_row.addWidget(self.play_beat_btn)
        beat_row.addWidget(self.beat_label, 1)

        third = QHBoxLayout()
        third.addWidget(regen_btn)
        third.addWidget(lock_btn)
        third.addWidget(play_section_btn)
        third.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(second)
        layout.addLayout(beat_row)
        layout.addWidget(QLabel("Sections - each is separately playable, "
                                "regeneratable and lockable:"))
        layout.addWidget(self.sections, 1)
        layout.addLayout(third)
        layout.addWidget(QLabel("Raaga check:"))
        layout.addWidget(self.report)
        self.setMinimumWidth(520)
        self.setMinimumHeight(260)
        self.refresh()

    # -- actions -----------------------------------------------------------
    def _accept(self) -> None:
        self.app.accept_tune(lock=True)
        self.changed.emit()

    def _version_chosen(self, index: int) -> None:
        version = self.versions.itemData(index)
        if version is not None:
            self.app.select_melody_version(int(version))
            self.changed.emit()

    def _selected_section(self):
        row = self.sections.currentRow()
        melody = self.app.project.melody()
        if melody is None or row < 0 or row >= len(melody.sections):
            return None
        return melody.sections[row]

    def _regenerate_section(self) -> None:
        section = self._selected_section()
        if section is None:
            QMessageBox.information(self, "Tune", "Select a section first.")
            return
        try:
            self.app.regenerate_tune_section(section.id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Tune", str(exc))

    def _toggle_section_lock(self) -> None:
        section = self._selected_section()
        if section is None:
            return
        self.app.set_section_lock(section.id, not section.locked)
        self.changed.emit()

    def _play_section(self) -> None:
        section = self._selected_section()
        if section is None:
            return
        self.app.play_range(section.start, section.end)

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        project = self.app.project
        melody = project.melody()

        current = self.versions.currentData()
        self.versions.blockSignals(True)
        self.versions.clear()
        for m in project.melodies:
            mark = " *" if m.version == project.approved_melody else ""
            self.versions.addItem(
                f"v{m.version}  {m.tempo_bpm} bpm  {m.duration:.0f}s  "
                f"[{m.state.value}]{mark}", m.version)
        if melody is not None:
            idx = self.versions.findData(melody.version)
            self.versions.setCurrentIndex(idx if idx >= 0 else 0)
        elif current is not None:
            idx = self.versions.findData(current)
            if idx >= 0:
                self.versions.setCurrentIndex(idx)
        self.versions.blockSignals(False)

        has_tune = melody is not None
        self.variation_btn.setEnabled(has_tune)
        self.play_btn.setEnabled(self.app.rendered("tune") is not None)
        self.accept_btn.setEnabled(has_tune)

        # The beat needs a length to fill, which a tune gives it - but a
        # brief with a target length is enough, so it is not gated on the
        # tune existing.
        beat = project.beat()
        self.beat_variation_btn.setEnabled(beat is not None)
        self.play_beat_btn.setEnabled(self.app.rendered("beat") is not None)
        self.beat_label.setText(
            f"v{beat.version}: {beat.summary()}" if beat else "No beat yet")

        if melody is not None and not self.tempo.hasFocus():
            self.tempo.setValue(int(melody.tempo_bpm))

        sections = melody.sections if melody else []
        self.sections.setRowCount(len(sections))
        for row, section in enumerate(sections):
            count = sum(1 for n in melody.notes if n.section_id == section.id)
            values = [section.name, f"{section.start:.1f}s", f"{section.end:.1f}s",
                      "yes" if section.locked else "", str(count)]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.sections.setItem(row, col, item)
        self.sections.resizeColumnsToContents()

        self.report.setPlainText(self.app.validation_report())
