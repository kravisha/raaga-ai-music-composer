"""The Training tab - specification section 3.

A view over :class:`TrainingController` and nothing more: every decision this
panel appears to make is a method call on the controller, which is what lets
the whole feature be tested without a window.

The layout follows the specification's own three sections - search, results,
queue - with the report and the history behind their own tabs so the working
area stays legible.  The one piece of judgement in here is the selection
model: results arrive with a checkbox and *nothing is checked*, because
section 20 rule 1 makes approval the creator's alone and a pre-ticked box is
not approval.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog,
                               QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from ...training.models import Accessibility, ObjectiveStatus, RunStatus

RESULT_COLUMNS = ("Select", "Title", "Source", "Author", "Duration",
                  "Relevance", "Accessibility", "Already learned", "URL")
QUEUE_COLUMNS = ("#", "Source", "Current objective", "Status", "Progress",
                 "Started", "Completed", "Result")
HISTORY_COLUMNS = ("Source", "Learned", "Search phrase", "Objectives",
                   "Knowledge", "Conflicts", "Confidence", "Status")


def _stamp(value: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(value)) if value else "-"


def _table(columns) -> QTableWidget:
    table = QTableWidget(0, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setStretchLastSection(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    return table


class TrainingPanel(QWidget):
    """Search for material, approve it, watch it be learned, read the report."""

    changed = Signal()

    def __init__(self, app, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self.training = getattr(app, "training", None)
        self._results: List[Any] = []
        self._queue_rows: List[Dict[str, Any]] = []
        self._history_rows: List[Dict[str, Any]] = []

        layout = QVBoxLayout(self)
        layout.addWidget(self._search_box())

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._results_box())
        splitter.addWidget(self._lower_tabs())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter, 1)

        self.status = QLabel("-")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # The queue works on its own thread; the view is refreshed on a timer
        # rather than from that thread, exactly as the rest of the app does.
        self._timer = QTimer(self)
        self._timer.setInterval(700)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    # ==================================================================
    # construction
    # ==================================================================
    def _search_box(self) -> QGroupBox:
        box = QGroupBox("Find something to learn from")
        self.phrase = QLineEdit()
        self.phrase.setPlaceholderText(
            "Kamboji raga beginner lesson, Carnatic gamaka techniques, "
            "Learn Adi tala ...")
        self.phrase.returnPressed.connect(self._search)

        self.max_results = QSpinBox()
        self.max_results.setRange(1, 50)
        self.max_results.setValue(10)

        self.source_filter = QComboBox()
        self.source_filter.addItem("Every source", "")
        self.source_filter.addItem("My learning folder", "library")
        self.source_filter.addItem("Exercises it can play itself", "exercises")
        self.source_filter.addItem("Web leads", "web")

        self.content_type = QComboBox()
        for label, value in (("Any kind", ""), ("Exercise", "exercise"),
                             ("My audio", "local_file"),
                             ("Transcript", "transcript"), ("Lead", "lead")):
            self.content_type.addItem(label, value)

        self.difficulty = QComboBox()
        for label, value in (("Any level", ""), ("Beginner", "beginner"),
                             ("Intermediate", "intermediate"),
                             ("Advanced", "advanced")):
            self.difficulty.addItem(label, value)

        self.duration = QComboBox()
        for label, value in (("Any length", ""), ("Short", "short"),
                             ("Medium", "medium"), ("Long", "long")):
            self.duration.addItem(label, value)

        self.language = QLineEdit()
        self.language.setPlaceholderText("Language")
        self.language.setMaximumWidth(140)
        self.include = QLineEdit()
        self.include.setPlaceholderText("Include words")
        self.exclude = QLineEdit()
        self.exclude.setPlaceholderText("Exclude words")

        search_btn = QPushButton("Search")
        search_btn.setObjectName("primary")
        search_btn.clicked.connect(self._search)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        suggest_btn = QPushButton("Suggest from what it is missing")
        suggest_btn.clicked.connect(self._suggest)

        top = QHBoxLayout()
        top.addWidget(self.phrase, 1)
        top.addWidget(search_btn)
        top.addWidget(clear_btn)

        filters = QHBoxLayout()
        for widget in (QLabel("Results"), self.max_results,
                       self.source_filter, self.content_type, self.difficulty,
                       self.duration, self.language, self.include,
                       self.exclude):
            filters.addWidget(widget)
        filters.addStretch(1)
        filters.addWidget(suggest_btn)

        layout = QVBoxLayout(box)
        layout.addLayout(top)
        layout.addLayout(filters)
        return box

    def _results_box(self) -> QGroupBox:
        box = QGroupBox("What was found - tick what it may learn from")
        self.results_table = _table(RESULT_COLUMNS)
        self.results_table.itemDoubleClicked.connect(self._open_result)

        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._check_all(True))
        clear_sel = QPushButton("Clear selection")
        clear_sel.clicked.connect(lambda: self._check_all(False))
        add_btn = QPushButton("Add to learning queue")
        add_btn.setObjectName("primary")
        add_btn.clicked.connect(self._add_to_queue)
        supply_btn = QPushButton("Supply the file myself...")
        supply_btn.clicked.connect(self._supply_file)
        transcript_btn = QPushButton("Provide transcript...")
        transcript_btn.clicked.connect(self._supply_transcript)

        buttons = QHBoxLayout()
        for widget in (select_all, clear_sel, supply_btn, transcript_btn):
            buttons.addWidget(widget)
        buttons.addStretch(1)
        buttons.addWidget(add_btn)

        note = QLabel(
            "Nothing is learned from until you tick it. A source marked "
            "'Metadata only' has not been fetched - supply the file or a "
            "transcript if you are entitled to use it.")
        note.setObjectName("hint")
        note.setWordWrap(True)

        layout = QVBoxLayout(box)
        layout.addWidget(self.results_table, 1)
        layout.addLayout(buttons)
        layout.addWidget(note)
        return box

    def _lower_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.addTab(self._queue_tab(), "Learning queue")
        tabs.addTab(self._report_tab(), "Report")
        tabs.addTab(self._history_tab(), "History")
        tabs.addTab(self._knowledge_tab(), "Knowledge base")
        self._tabs = tabs
        return tabs

    def _queue_tab(self) -> QWidget:
        page = QWidget()
        self.queue_table = _table(QUEUE_COLUMNS)
        self.queue_table.itemSelectionChanged.connect(self._queue_selected)

        self.queue_progress = QProgressBar()
        self.queue_progress.setRange(0, 100)

        start = QPushButton("Start learning")
        start.setObjectName("primary")
        start.clicked.connect(self._start)

        controls = QHBoxLayout()
        controls.addWidget(start)
        for text, slot in (("Pause", self._pause), ("Resume", self._resume),
                           ("Cancel current", self._cancel),
                           ("Remove from queue", self._remove),
                           ("Retry", self._retry),
                           ("View report", self._view_report)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            controls.addWidget(button)
        controls.addStretch(1)

        self.objectives_view = QTextEdit()
        self.objectives_view.setReadOnly(True)
        self.objectives_view.setMaximumHeight(150)
        add_objective = QPushButton("Add an objective...")
        add_objective.clicked.connect(self._add_objective)

        objectives_row = QHBoxLayout()
        objectives_row.addWidget(QLabel("Objectives for the selected source"))
        objectives_row.addStretch(1)
        objectives_row.addWidget(add_objective)

        layout = QVBoxLayout(page)
        layout.addWidget(self.queue_table, 1)
        layout.addWidget(self.queue_progress)
        layout.addLayout(controls)
        layout.addLayout(objectives_row)
        layout.addWidget(self.objectives_view)
        return page

    def _report_tab(self) -> QWidget:
        page = QWidget()
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setLineWrapMode(QTextEdit.WidgetWidth)
        self.report_view.setPlainText(
            "Select a source in the queue and press 'View report'.")
        layout = QVBoxLayout(page)
        layout.addWidget(self.report_view)
        return page

    def _history_tab(self) -> QWidget:
        page = QWidget()
        self.history_table = _table(HISTORY_COLUMNS)
        self.history_filter = QLineEdit()
        self.history_filter.setPlaceholderText("Filter by raaga or topic")
        self.history_filter.returnPressed.connect(self.refresh)
        self.history_totals = QLabel("-")
        self.history_totals.setObjectName("hint")

        row = QHBoxLayout()
        row.addWidget(self.history_filter, 1)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)

        layout = QVBoxLayout(page)
        layout.addLayout(row)
        layout.addWidget(self.history_table, 1)
        layout.addWidget(self.history_totals)
        return page

    def _knowledge_tab(self) -> QWidget:
        page = QWidget()
        self.knowledge_query = QLineEdit()
        self.knowledge_query.setPlaceholderText(
            "Search what has been learned - keyword, raaga, concept ...")
        self.knowledge_query.returnPressed.connect(self._search_knowledge)
        find = QPushButton("Search")
        find.clicked.connect(self._search_knowledge)
        wrong = QPushButton("Mark selected incorrect")
        wrong.clicked.connect(self._mark_incorrect)
        approve = QPushButton("Approve selected")
        approve.clicked.connect(self._approve)

        row = QHBoxLayout()
        row.addWidget(self.knowledge_query, 1)
        row.addWidget(find)
        row.addWidget(approve)
        row.addWidget(wrong)

        self.knowledge_table = _table(
            ("Statement", "Category", "Raaga", "Confidence", "Source",
             "Status"))
        self.knowledge_table.itemSelectionChanged.connect(self._show_provenance)
        self.provenance_view = QTextEdit()
        self.provenance_view.setReadOnly(True)
        self.provenance_view.setMaximumHeight(160)

        layout = QVBoxLayout(page)
        layout.addLayout(row)
        layout.addWidget(self.knowledge_table, 1)
        layout.addWidget(QLabel("Where this came from"))
        layout.addWidget(self.provenance_view)
        return page

    # ==================================================================
    # actions
    # ==================================================================
    def _unavailable(self) -> bool:
        if self.training is None:
            self.status.setText("Training is not available in this session.")
            return True
        return False

    def _search(self) -> None:
        if self._unavailable():
            return
        phrase = self.phrase.text().strip()
        if not phrase:
            self.status.setText("Type something to search for first.")
            return
        self._results = self.training.search(
            phrase,
            max_results=self.max_results.value(),
            source_filter=self.source_filter.currentData(),
            content_type=self.content_type.currentData(),
            difficulty=self.difficulty.currentData(),
            duration_preference=self.duration.currentData(),
            language=self.language.text().strip(),
            include_keywords=[w for w in self.include.text().split(",") if w.strip()],
            exclude_keywords=[w for w in self.exclude.text().split(",") if w.strip()])
        self._fill_results()
        self.status.setText(
            f"{len(self._results)} result(s). Tick the ones it may learn from.")

    def _clear(self) -> None:
        if self.training is not None:
            self.training.clear_search()
        self._results = []
        self.results_table.setRowCount(0)
        self.phrase.clear()

    def _suggest(self) -> None:
        if self._unavailable():
            return
        self.phrase.setText(self.training.suggested_phrase())
        self.status.setText(
            "Suggested from what the agent is still missing. It will not "
            "search or approve anything on its own.")

    def _fill_results(self) -> None:
        table = self.results_table
        table.setRowCount(len(self._results))
        for row, source in enumerate(self._results):
            check = QTableWidgetItem()
            check.setFlags(check.flags() | Qt.ItemIsUserCheckable)
            # Never pre-ticked: approval is the creator's, not a default.
            check.setCheckState(Qt.Unchecked)
            table.setItem(row, 0, check)
            values = (
                source.title, source.provider or source.source_type,
                source.author, source.duration_label,
                f"{source.relevance_score:.2f}",
                Accessibility.LABELS.get(source.accessibility_status,
                                         source.accessibility_status),
                "yes" if source.previously_learned else "",
                source.url)
            for column, value in enumerate(values, start=1):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    def _check_all(self, checked: bool) -> None:
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item is not None:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _checked_sources(self) -> List[Any]:
        out = []
        for row in range(self.results_table.rowCount()):
            item = self.results_table.item(row, 0)
            if item is not None and item.checkState() == Qt.Checked:
                out.append(self._results[row])
        return out

    def _add_to_queue(self) -> None:
        if self._unavailable():
            return
        chosen = self._checked_sources()
        if not chosen:
            self.status.setText("Tick at least one result first.")
            return
        relearn = [s for s in chosen if s.previously_learned]
        if relearn:
            answer = QMessageBox.question(
                self, "Already learned",
                f"{len(relearn)} of these has been learned before. Learn them "
                f"again? The earlier reports are kept either way.",
                QMessageBox.Yes | QMessageBox.No)
            if answer == QMessageBox.No:
                chosen = [s for s in chosen if not s.previously_learned]
        if not chosen:
            self.status.setText("Nothing was added.")
            return
        runs = self.training.add_to_queue([s.source_id for s in chosen])
        self.status.setText(f"{len(runs)} source(s) queued. Press "
                            f"'Start learning' when you are ready.")
        self.refresh()
        self.changed.emit()

    def _selected_run(self) -> str:
        row = self.queue_table.currentRow()
        if 0 <= row < len(self._queue_rows):
            return str(self._queue_rows[row]["run_id"])
        return ""

    def _selected_source(self) -> Optional[Any]:
        row = self.results_table.currentRow()
        if 0 <= row < len(self._results):
            return self._results[row]
        return None

    def _supply_file(self) -> None:
        if self._unavailable():
            return
        source = self._selected_source()
        if source is None:
            self.status.setText("Select a result first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Supply the audio for this source", "",
            "Audio (*.wav *.flac *.ogg *.aiff *.aif *.mp3)")
        if not path:
            return
        if self.training.supply_file(source.source_id, path):
            source.local_path = path
            source.accessibility_status = Accessibility.ACCESSIBLE
            self._fill_results()
            self.status.setText(f"{Path(path).name} will be used for this "
                                f"source.")

    def _supply_transcript(self) -> None:
        if self._unavailable():
            return
        source = self._selected_source()
        if source is None:
            self.status.setText("Select a result first.")
            return
        text, ok = QInputDialog.getMultiLineText(
            self, "Provide transcript",
            "Paste the transcript or captions you are entitled to use:")
        if not ok or not text.strip():
            return
        if self.training.supply_transcript(source.source_id, text):
            source.accessibility_status = Accessibility.TRANSCRIPT
            source.transcript_available = True
            self._fill_results()
            self.status.setText("Transcript stored for this source. It will "
                                "be read, but nothing will be heard.")

    # -- queue ---------------------------------------------------------
    def _start(self) -> None:
        if self._unavailable():
            return
        self.training.start_learning()
        self.status.setText("Learning. One source at a time.")

    def _pause(self) -> None:
        if not self._unavailable():
            self.training.pause_learning()
            self.status.setText("Paused after the current source.")

    def _resume(self) -> None:
        if not self._unavailable():
            self.training.resume_learning()
            self.status.setText("Resumed.")

    def _cancel(self) -> None:
        if not self._unavailable():
            self.training.cancel_current()
            self.status.setText("Cancelling the source being worked on.")

    def _remove(self) -> None:
        run_id = self._selected_run()
        if run_id and not self._unavailable():
            self.training.remove_from_queue(run_id)
            self.refresh()

    def _retry(self) -> None:
        run_id = self._selected_run()
        if run_id and not self._unavailable():
            self.training.retry(run_id)
            self.refresh()

    def _view_report(self) -> None:
        run_id = self._selected_run()
        if not run_id or self._unavailable():
            return
        self.report_view.setPlainText(self.training.render_report(run_id))
        self._tabs.setCurrentIndex(1)

    def _queue_selected(self) -> None:
        run_id = self._selected_run()
        if not run_id or self.training is None:
            return
        lines = []
        for objective in self.training.objectives(run_id):
            label = ObjectiveStatus.LABELS.get(objective.status,
                                               objective.status)
            lines.append(f"[{label}] {objective.description}")
            if objective.outcome:
                lines.append(f"      {objective.outcome}")
        self.objectives_view.setPlainText(
            "\n".join(lines) or "No objectives have been set for this source.")

    def _add_objective(self) -> None:
        run_id = self._selected_run()
        if not run_id or self._unavailable():
            return
        text, ok = QInputDialog.getText(self, "Add an objective",
                                        "What should it try to learn?")
        if ok and text.strip():
            self.training.add_objective(run_id, text.strip())
            self._queue_selected()

    # -- knowledge -----------------------------------------------------
    def _search_knowledge(self) -> None:
        if self._unavailable():
            return
        entries = self.training.search_knowledge(
            keyword=self.knowledge_query.text().strip())
        self._knowledge = entries
        table = self.knowledge_table
        table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = (entry.normalized_statement, entry.category, entry.raga,
                      f"{entry.confidence:.2f}", entry.source_title,
                      entry.status + (" (disputed)" if entry.contradicted
                                      else ""))
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        self.status.setText(f"{len(entries)} knowledge item(s).")

    def _selected_knowledge(self) -> Optional[Any]:
        row = self.knowledge_table.currentRow()
        entries = getattr(self, "_knowledge", [])
        return entries[row] if 0 <= row < len(entries) else None

    def _show_provenance(self) -> None:
        entry = self._selected_knowledge()
        if entry is None or self.training is None:
            return
        record = self.training.provenance(entry.knowledge_id)
        lines = [f"{key.replace('_', ' ')}: {value}"
                 for key, value in record.items() if key != "audit"]
        self.provenance_view.setPlainText("\n".join(lines))

    def _mark_incorrect(self) -> None:
        entry = self._selected_knowledge()
        if entry is None or self._unavailable():
            return
        self.training.mark_knowledge_incorrect(entry.knowledge_id,
                                               "marked incorrect in the tab")
        self._search_knowledge()

    def _approve(self) -> None:
        entry = self._selected_knowledge()
        if entry is None or self._unavailable():
            return
        self.training.approve_knowledge(entry.knowledge_id)
        self._search_knowledge()

    def _open_result(self, item) -> None:
        source = self._selected_source()
        if source is not None:
            self.status.setText(f"{source.title} - {source.url}")

    # ==================================================================
    def refresh(self) -> None:
        if self.training is None:
            return
        self._queue_rows = self.training.queue_snapshot()
        table = self.queue_table
        table.setRowCount(len(self._queue_rows))
        current = 0.0
        for row, entry in enumerate(self._queue_rows):
            if entry["is_current"]:
                current = float(entry["progress"])
            values = (entry["position"], entry["title"], entry["objective"],
                      entry["status_label"],
                      f"{float(entry['progress']) * 100:.0f}%",
                      _stamp(float(entry["started"])),
                      _stamp(float(entry["completed"])),
                      entry["result"] or entry["detail"])
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        self.queue_progress.setValue(int(current * 100))

        self._history_rows = self.training.training_history(
            topic=self.history_filter.text().strip())
        history = self.history_table
        history.setRowCount(len(self._history_rows))
        for row, entry in enumerate(self._history_rows):
            values = (entry["title"], entry["when"], entry["search_phrase"],
                      f"{entry['objectives_met']}/{entry['objectives']}",
                      entry["knowledge_added"], entry["conflicts"],
                      f"{entry['confidence']:.2f}", entry["status_label"])
            for column, value in enumerate(values):
                history.setItem(row, column, QTableWidgetItem(str(value)))
        history.resizeColumnsToContents()

        totals = self.training.totals()
        self.history_totals.setText(
            f"{totals['sources_seen']} source(s) seen, "
            f"{totals['completed']} completed, {totals['failed']} failed, "
            f"{totals['queued']} waiting; {totals['knowledge_items']} "
            f"knowledge item(s), {totals['open_conflicts']} open conflict(s)")
