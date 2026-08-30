"""Creative brief panel (spec section 14B)."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QLineEdit, QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

LANGUAGES = ["Tamil", "Hindi", "Telugu", "English", "Kannada", "Malayalam"]
SONG_TYPES = ["film song", "devotional", "simple", "pop", "ghazal"]
MOODS = ["romantic", "longing", "sad", "celebration", "devotional", "aggressive",
         "high-energy", "intimate", "hopeful", "nostalgia", "heroic"]


class BriefPanel(QGroupBox):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__("Creative brief", parent)
        self.app = app

        self.situation = QLineEdit()
        self.situation.setPlaceholderText("Film situation, scene or character view")
        self.mood = QComboBox()
        self.mood.setEditable(True)
        self.mood.addItems(MOODS)
        self.feel = QPlainTextEdit()
        self.feel.setPlaceholderText(
            "Describe the feel in your own words - \"lonely, late at night, but "
            "still warm\"")
        self.feel.setFixedHeight(56)
        self.language = QComboBox()
        self.language.setEditable(True)
        self.language.addItems(LANGUAGES)
        self.song_type = QComboBox()
        self.song_type.setEditable(True)
        self.song_type.addItems(SONG_TYPES)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(20.0, 900.0)
        self.duration.setSuffix(" s")
        self.duration.setSingleStep(15.0)
        self.tempo = QSpinBox()
        self.tempo.setRange(0, 200)
        self.tempo.setSpecialValueText("auto")
        self.tempo.setSuffix(" bpm")
        self.prefer = QLineEdit()
        self.prefer.setPlaceholderText("Instruments to use, comma separated")
        self.avoid = QLineEdit()
        self.avoid.setPlaceholderText("Instruments to avoid")
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Anything else")
        self.notes.setFixedHeight(46)

        form = QFormLayout()
        form.addRow("Situation", self.situation)
        form.addRow("Mood", self.mood)
        form.addRow("Feel", self.feel)
        form.addRow("Language", self.language)
        form.addRow("Song type", self.song_type)
        form.addRow("Target length", self.duration)
        form.addRow("Tempo", self.tempo)
        form.addRow("Prefer", self.prefer)
        form.addRow("Avoid", self.avoid)
        form.addRow("Notes", self.notes)

        apply_btn = QPushButton("Apply brief")
        apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(self.apply)

        container = QWidget()
        container.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(container)
        layout.addWidget(apply_btn)
        self.refresh()

    def apply(self) -> None:
        self.app.update_brief(
            situation=self.situation.text().strip(),
            mood=self.mood.currentText().strip(),
            feel=self.feel.toPlainText().strip(),
            language=self.language.currentText().strip() or "Tamil",
            song_type=self.song_type.currentText().strip() or "film song",
            duration_target=float(self.duration.value()),
            tempo_preference=(int(self.tempo.value()) or None),
            instruments_preferred=[s.strip() for s in self.prefer.text().split(",")
                                   if s.strip()],
            instruments_avoided=[s.strip() for s in self.avoid.text().split(",")
                                 if s.strip()],
            notes=self.notes.toPlainText().strip())
        self.changed.emit()

    def refresh(self) -> None:
        brief = self.app.project.brief
        if self.hasFocus() or self.feel.hasFocus() or self.situation.hasFocus():
            return
        self.situation.setText(brief.situation)
        self.mood.setCurrentText(brief.mood)
        if self.feel.toPlainText() != brief.feel:
            self.feel.setPlainText(brief.feel)
        self.language.setCurrentText(brief.language)
        self.song_type.setCurrentText(brief.song_type)
        self.duration.setValue(float(brief.duration_target))
        self.tempo.setValue(int(brief.tempo_preference or 0))
        self.prefer.setText(", ".join(brief.instruments_preferred))
        self.avoid.setText(", ".join(brief.instruments_avoided))
        if self.notes.toPlainText() != brief.notes:
            self.notes.setPlainText(brief.notes)
