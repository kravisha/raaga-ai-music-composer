"""Creative brief panel (spec section 14B)."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QLabel, QLineEdit, QPlainTextEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from ...core.actions import ActionState
from ..theme import DARK

LANGUAGES = ["Tamil", "Hindi", "Telugu", "English", "Kannada", "Malayalam"]
SONG_TYPES = ["film song", "devotional", "simple", "pop", "ghazal"]
MOODS = ["romantic", "longing", "sad", "celebration", "devotional", "aggressive",
         "high-energy", "intimate", "hopeful", "nostalgia", "heroic"]

# Action state -> theme colour for the status label (v0.3 section 6.1).
_STATE_COLOR = {
    ActionState.IDLE: DARK["muted"],
    ActionState.STARTING: DARK["muted"],
    ActionState.WORKING: DARK["accent"],
    ActionState.COMPLETED: DARK["ok"],
    ActionState.FAILED: DARK["error"],
    ActionState.CANCELLED: DARK["warn"],
}


class BriefPanel(QGroupBox):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__("Creative brief", parent)
        self.app = app

        self.title = QLineEdit()
        self.title.setPlaceholderText("Song title")
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
        form.addRow("Title", self.title)
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

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        container = QWidget()
        container.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(container)
        layout.addWidget(apply_btn)
        layout.addWidget(self.status_label)
        self.refresh()

        # Chain onto whatever is already listening (spec section 6.1): this
        # panel must never silently replace another subscriber.
        previous_on_action = self.app.on_action

        def _on_action(status) -> None:
            if previous_on_action:
                previous_on_action(status)
            self._show_action_status(status)

        self.app.on_action = _on_action

    def apply(self) -> None:
        self.app.apply_brief(
            title=self.title.text().strip(),
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

    def _show_action_status(self, status) -> None:
        if status.action != "apply_brief":
            return
        color = _STATE_COLOR.get(status.state, DARK["text"])
        prefix = "" if status.state in (ActionState.COMPLETED,) else \
            f"{status.state.value.title()}: "
        self.status_label.setText(f"{prefix}{status.text}")
        self.status_label.setStyleSheet(f"color: {color};")

    def refresh(self) -> None:
        brief = self.app.project.brief
        if self.hasFocus() or self.feel.hasFocus() or self.situation.hasFocus():
            return
        if not self.title.hasFocus():
            self.title.setText(brief.title)
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
