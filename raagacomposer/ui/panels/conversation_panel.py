"""Conversational control panel (spec sections 5, 14H)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QProgressBar,
                               QPushButton, QTextEdit, QVBoxLayout, QWidget)

from .. import theme

STATUS_COLORS = {
    "applied": theme.DARK["ok"],
    "failed": theme.DARK["error"],
    "ignored": theme.DARK["warn"],
    "superseded": theme.DARK["muted"],
    "cancelled": theme.DARK["muted"],
}


class ConversationPanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        self.mic_btn = QPushButton("Start listening")
        self.mic_btn.setObjectName("record")
        self.mic_btn.setCheckable(True)
        self.mic_btn.clicked.connect(self._toggle_mic)

        self.cancel_btn = QPushButton("Cancel current operation")
        self.cancel_btn.clicked.connect(self._cancel)

        self.mic_state = QLabel("Microphone off")
        self.mic_state.setObjectName("hint")
        self.mic_state.setWordWrap(True)

        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(8)

        self.partial = QLabel("")
        self.partial.setObjectName("hint")
        self.partial.setWordWrap(True)

        self.history = QListWidget()
        self.history.setWordWrap(True)

        self.interpretation = QTextEdit()
        self.interpretation.setReadOnly(True)
        self.interpretation.setFixedHeight(64)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText(
            "Type an instruction (same pipeline as speech) and press Enter")
        self.entry.returnPressed.connect(self._submit)

        jobs_row = QHBoxLayout()
        self.job_label = QLabel("No background work")
        self.job_label.setObjectName("hint")
        self.job_bar = QProgressBar()
        self.job_bar.setRange(0, 100)
        self.job_bar.setFixedHeight(12)
        jobs_row.addWidget(self.job_label, 1)
        jobs_row.addWidget(self.job_bar, 1)

        top = QHBoxLayout()
        top.addWidget(self.mic_btn)
        top.addWidget(self.cancel_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.mic_state)
        layout.addWidget(self.level)
        layout.addWidget(QLabel("Live transcript:"))
        layout.addWidget(self.partial)
        layout.addWidget(QLabel("What I understood:"))
        layout.addWidget(self.interpretation)
        layout.addWidget(QLabel("Command history:"))
        layout.addWidget(self.history, 1)
        layout.addWidget(self.entry)
        layout.addLayout(jobs_row)
        self.setMinimumWidth(300)
        self.setMinimumHeight(300)
        self.refresh()

    # -- actions -----------------------------------------------------------
    def _toggle_mic(self) -> None:
        if self.app.context.listening:
            self.app.stop_listening()
        else:
            self.app.start_listening()
        self.refresh()

    def _cancel(self) -> None:
        n = self.app.jobs.cancel_all("cancelled by the creator")
        self.app.status(f"Cancelled {n} operation(s)" if n else "Nothing to cancel")

    def _submit(self) -> None:
        text = self.entry.text().strip()
        if not text:
            return
        self.entry.clear()
        self.app.handle_utterance(text)
        self.refresh()
        self.changed.emit()

    # -- refresh -----------------------------------------------------------
    def refresh(self) -> None:
        state = self.app.voice_input.state
        listening = state.listening
        self.mic_btn.setChecked(listening)
        self.mic_btn.setText("Stop listening" if listening else "Start listening")
        self.mic_btn.setProperty("listening", "true" if listening else "false")
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)
        self.mic_state.setText(self.app.voice_input.status_text())
        self.level.setValue(int(min(100.0, state.level * 900)))
        self.partial.setText(self.app.context.partial or "-")

        turns = self.app.context.turns[-60:]
        self.history.clear()
        for turn in turns:
            text = f"{turn.text}"
            if turn.interpretation:
                text += f"\n    -> {turn.interpretation}  [{turn.status}]"
            item = QListWidgetItem(text)
            colour = STATUS_COLORS.get(turn.status)
            if colour:
                from PySide6.QtGui import QColor
                item.setForeground(QColor(colour))
            self.history.addItem(item)
        self.history.scrollToBottom()
        if turns:
            self.interpretation.setPlainText(turns[-1].interpretation or "-")

        active = self.app.jobs.active_jobs()
        if active:
            job = active[0]
            self.job_label.setText(
                f"{job.description} ({len(active)} running)"
                + (f" - {job.message}" if job.message else ""))
            self.job_bar.setValue(int(job.progress * 100))
        else:
            self.job_label.setText("No background work")
            self.job_bar.setValue(0)
