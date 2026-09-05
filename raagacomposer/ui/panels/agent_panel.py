"""Music agent / learning panel (learning specification section 17).

Shows what the student is doing, what it knows, where it learned it, and lets
the creator talk to it directly.  The creator never has to touch the database.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QComboBox, QFileDialog, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)


class AgentPanel(QWidget):
    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app

        # ---- status ------------------------------------------------------
        # Kept as an attribute (not a local) so the LEARN workspace's
        # Dashboard area can re-home this whole box (spec v0.3 section 4.2A)
        # instead of rebuilding an equivalent readout from scratch.
        status_box = self.status_box = QGroupBox("The student")
        self.headline = QLabel("-")
        self.headline.setWordWrap(True)
        self.activity = QLabel("-")
        self.activity.setObjectName("hint")
        self.activity.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.mastery = QProgressBar()
        self.mastery.setRange(0, 100)
        self.counts = QLabel("-")
        self.counts.setObjectName("hint")
        self.counts.setWordWrap(True)

        status_layout = QVBoxLayout(status_box)
        status_layout.addWidget(self.headline)
        status_layout.addWidget(self.activity)
        status_layout.addWidget(QLabel("Curriculum progress"))
        status_layout.addWidget(self.progress)
        status_layout.addWidget(QLabel("Confidence in the current raaga"))
        status_layout.addWidget(self.mastery)
        status_layout.addWidget(self.counts)

        # ---- controls ----------------------------------------------------
        # Same reasoning as status_box: kept alive as self.controls so the
        # LEARN workspace's Dashboard area can re-home it.
        controls = self.controls = QGroupBox("Learning")
        self.raaga_box = QComboBox()
        self.raaga_box.addItems(self.app.raagas.names())
        study_btn = QPushButton("Study this raaga")
        study_btn.clicked.connect(self._study)

        step_btn = QPushButton("One lesson")
        step_btn.setObjectName("primary")
        step_btn.clicked.connect(self._step)
        # No Start / Pause / Resume / Stop here.  The LEARN dashboard
        # re-homes this whole group directly beneath its own Start / Pause /
        # Resume / Stop row, so running the learner from here put the same
        # actions twice in one column, inches apart - a tri-state toggle
        # beside the four buttons it duplicates.  The dashboard's row is the
        # one that survives; this group keeps what is its own, which is
        # choosing what to study and studying one lesson by hand.
        # ``_toggle_learning`` and ``_stop`` stay - the Learning menu calls
        # them, and they refresh the panel where the raw controller calls
        # would leave it stale.
        corpus_btn = QPushButton("Choose my learning folder...")
        corpus_btn.clicked.connect(self._choose_corpus)

        row1 = QHBoxLayout()
        row1.addWidget(self.raaga_box, 1)
        row1.addWidget(study_btn)
        row2 = QHBoxLayout()
        row2.addWidget(step_btn)
        row2.addStretch(1)
        row3 = QHBoxLayout()
        row3.addWidget(corpus_btn)
        self.corpus_label = QLabel("no folder chosen")
        self.corpus_label.setObjectName("hint")
        self.corpus_label.setWordWrap(True)
        row3.addWidget(self.corpus_label, 1)

        controls_layout = QVBoxLayout(controls)
        controls_layout.addLayout(row1)
        controls_layout.addLayout(row2)
        controls_layout.addLayout(row3)

        # ---- tabs --------------------------------------------------------
        self.knowledge = QTextEdit()
        self.knowledge.setReadOnly(True)
        self.curriculum = QTableWidget(0, 5)
        self.curriculum.setHorizontalHeaderLabels(
            ["Unit", "Goal", "Status", "Mastery", "Waiting on"])
        self.curriculum.verticalHeader().setVisible(False)
        self.sources = QTableWidget(0, 5)
        self.sources.setHorizontalHeaderLabels(
            ["Source", "Rights", "Provider", "Status", "Confidence"])
        self.sources.verticalHeader().setVisible(False)
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.critique = QTextEdit()
        self.critique.setReadOnly(True)

        tabs = QTabWidget()
        tabs.addTab(_wrap(self.knowledge), "What it knows")
        tabs.addTab(_wrap(self.curriculum), "Curriculum")
        tabs.addTab(_wrap(self.sources), "Where it learned it")
        tabs.addTab(_wrap(self.activity_log), "Recent activity")
        tabs.addTab(_wrap(self.critique), "Its own critique")
        self.tabs = tabs

        # Kept as self.critique_btn so the LEARN workspace's Practice/Quiz
        # area can re-home it alongside the critique text it fills in.
        critique_btn = self.critique_btn = QPushButton("Mark the current tune")
        critique_btn.clicked.connect(self._critique)

        # ---- talking to the agent ---------------------------------------
        self.ask = QLineEdit()
        self.ask.setPlaceholderText(
            'Ask or instruct: "Learn Keeravani", "Why did you choose this '
            'phrase?", "This does not sound like Keeravani."')
        self.ask.returnPressed.connect(self._ask)
        self.answer = QTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setFixedHeight(96)

        # The status and controls keep their natural height and the panel
        # scrolls, rather than squeezing the labels until they vanish.
        self.knowledge.setMinimumHeight(180)
        self.curriculum.setMinimumHeight(180)
        self.sources.setMinimumHeight(180)
        self.activity_log.setMinimumHeight(180)
        self.critique.setMinimumHeight(180)
        tabs.setMinimumHeight(240)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.addWidget(status_box)
        inner_layout.addWidget(controls)
        inner_layout.addWidget(critique_btn)
        inner_layout.addWidget(tabs, 1)
        inner_layout.addWidget(self.ask)
        inner_layout.addWidget(self.answer)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self.setMinimumWidth(460)
        self.setMinimumHeight(300)
        self.refresh()

    # ==================================================================
    def _toggle_learning(self) -> None:
        status = self.app.agent_status()
        if status["learning"]:
            self.app.pause_learning()
        elif status["paused"]:
            self.app.resume_learning()
        else:
            self.app.start_learning()
        self.refresh()

    def _step(self) -> None:
        message = self.app.learn_now(1)
        self.answer.setPlainText(message)
        self.refresh()
        self.changed.emit()

    def _stop(self) -> None:
        self.app.stop_learning()
        self.refresh()

    def _study(self) -> None:
        message = self.app.study_raaga(self.raaga_box.currentText())
        self.answer.setPlainText(message)
        self.refresh()
        self.changed.emit()

    def _choose_corpus(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose a folder of your own recordings to learn from",
            self.app.settings.learning_corpus_dir or str(Path.home()))
        if not directory:
            return
        self.app.settings.learning_corpus_dir = directory
        self.app.settings.save()
        self.app.agent.research = type(self.app.agent.research)(
            self.app.agent.repo, self.app.raagas, self.app.settings,
            self.app.providers.llm)
        QMessageBox.information(
            self, "Learning folder",
            "I will look here for audio to study.\n\nName the files or folders "
            "after the raaga - for example Keeravani-alapana.wav - so I know "
            "what I am listening to.\n\nOnly put material there that you are "
            "entitled to use.")
        self.refresh()

    def _critique(self) -> None:
        self.critique.setPlainText(self.app.critique_tune())
        self.tabs.setCurrentIndex(4)

    def _ask(self) -> None:
        text = self.ask.text().strip()
        if not text:
            return
        self.ask.clear()
        low = text.lower()
        if low.startswith(("learn ", "study ")):
            name = text.split(" ", 1)[1].strip(" .")
            self.answer.setPlainText(self.app.study_raaga(name))
        elif any(word in low for word in
                 ("does not sound", "doesn't sound", "too ", "i like",
                  "keep ", "wrong", "not right", "sounds ")):
            self.answer.setPlainText(self.app.give_feedback(text))
        else:
            self.answer.setPlainText(self.app.ask_agent(text))
        self.refresh()
        self.changed.emit()

    # ==================================================================
    def refresh(self) -> None:
        status = self.app.agent_status()
        learning = status["learning"]
        paused = status["paused"]
        self.headline.setText(
            f"Stage {status['stage']} - studying {status['current_raaga']}\n"
            f"Next: {status['next_goal']}")
        self.activity.setText(
            f"{'learning' if learning else ('paused' if paused else 'idle')} - "
            f"{status['last_step'] or 'nothing yet this session'}")
        self.progress.setValue(int(status["overall_percent"]))
        self.mastery.setValue(int(status["mastery"] * 100))
        self.counts.setText(
            f"foundations {status['foundations']} - raaga units "
            f"{status['raaga_units']} - {status['phrases']} phrases learned from "
            f"{status['sources_analysed']}/{status['sources']} sources - "
            f"{status['facts']} facts"
            + (f" ({status['disputed_facts']} disputed)"
               if status["disputed_facts"] else "")
            + f" - memory {status['repository_bytes'] / 1024:.0f} KB")

        idx = self.raaga_box.findText(status["current_raaga"])
        if idx >= 0 and not self.raaga_box.hasFocus():
            self.raaga_box.setCurrentIndex(idx)
        self.corpus_label.setText(
            self.app.settings.learning_corpus_dir or "no folder chosen")

        self.knowledge.setPlainText(self.app.agent_knowledge())

        rows = self.app.agent.curriculum.progress_table(status["current_raaga"])
        self.curriculum.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, key in enumerate(("unit", "goal", "status", "mastery",
                                     "blocked_by")):
                item = QTableWidgetItem(str(row[key]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.curriculum.setItem(r, c, item)
        self.curriculum.resizeColumnsToContents()

        sources = self.app.agent.repo.sources(limit=60)
        self.sources.setRowCount(len(sources))
        for r, source in enumerate(sources):
            values = [source.title, source.rights_status, source.provider,
                      source.status, f"{source.confidence:.2f}"]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.sources.setItem(r, c, item)
        self.sources.resizeColumnsToContents()

        events = self.app.agent_events(40)
        self.activity_log.setPlainText("\n".join(
            f"{e['kind']:<22} {e['detail']}" for e in events))

        for error in status["errors"]:
            if error not in self.activity_log.toPlainText():
                self.activity_log.append(f"error: {error}")


def _wrap(widget: QWidget) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    return holder
