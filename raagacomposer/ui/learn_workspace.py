"""The LEARN workspace (spec v0.3 sections 4, 4.2, 25, 45, 56; TEST I).

Section 4 makes MAIN and LEARN two top-level workspaces rather than treating
learning as one more tab beside Tune and Lyrics.  This module builds LEARN's
six areas - Dashboard, Curriculum, Training Sources, Practice / Quiz,
Knowledge, History / Evaluation - by RE-HOMING the widgets that already exist
in :class:`AgentPanel` and :class:`TrainingPanel` rather than rewriting them.

Only two things are genuinely new here: the Dashboard's grid of labelled
values (built from ``agent.status()`` and ``curriculum.stage_summary()``) and
the Practice / Quiz area's "Run one exercise" button, which runs one practice
step for the current unit on the shared :class:`JobManager` so the GUI thread
is never blocked by synthesis + analysis.

Everything else is an existing widget, taken out of ``AgentPanel``'s or
``TrainingPanel``'s own tab bar and placed under a new heading.  The panel
instances themselves stay alive as attributes of the main window (their
``refresh()`` methods, click handlers and internal state are untouched); only
where their pieces are drawn on screen changes.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QPushButton, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from .panels.agent_panel import AgentPanel
from .panels.training_panel import TrainingPanel

AREA_NAMES = ["Dashboard", "Curriculum", "Training Sources", "Practice / Quiz",
             "Knowledge", "History / Evaluation"]

DASHBOARD_FIELDS = [
    ("Agent status", "activity_state"),
    ("Current curriculum stage", "stage"),
    ("Current raaga", "current_raaga"),
    ("Current lesson / unit", "lesson"),
    ("Current learning objective", "objective"),
    ("Mastery score", "mastery"),
    ("Overall progress", "overall_percent"),
    ("Current activity", "activity"),
    ("Last learned item", "last_step"),
    ("Errors / warnings", "errors"),
    ("Knowledge repository size", "repository_size"),
    ("Sources analysed", "sources_analysed"),
    ("Learned phrases", "phrases"),
]

EXERCISE_COLUMNS = ("Exercise", "Score", "Pass threshold", "Result", "Detail")
LESSON_COLUMNS = ("Mistake", "Times", "Last seen", "Correction")
FINDINGS_COLUMNS = ("When", "Unit / task", "Mistake", "Correction")


def provider_summary_line(rows: List[Any]) -> str:
    """One line: "Claude: <state> - Local model: <state> - Built-in: Ready".

    Only Claude is a cloud provider today (spec 42), so every ``kind ==
    "cloud"`` row is treated as Claude's state; a local *language-model*
    backend (as opposed to a local speech backend, which is also reported as
    "local") is anything not named "vosk" or "whisper". Never displays a
    secret - only the state word ("Configured", "Not configured", ...).
    """
    def best_state(matched: List[Any]) -> str:
        if not matched:
            return "Unknown"
        for row in matched:
            if row.state not in ("Off", "Not installed", "Not configured"):
                return row.state
        return matched[0].state

    claude_rows = [r for r in rows if r.kind == "cloud"]
    local_llm_rows = [r for r in rows if r.kind == "local"
                      and r.name not in ("vosk", "whisper")]
    return (f"Claude: {best_state(claude_rows)}  ·  "
            f"Local model: {best_state(local_llm_rows)}  ·  "
            f"Built-in engines: Ready")


def _take_tab(tabs, label: str) -> QWidget:
    """Detach the page named *label* from a QTabWidget and return it.

    ``removeTab`` does not delete the page widget; it only drops the tab bar
    entry, so the widget (and everything inside it) is intact and ready to be
    added to a different layout, which reparents it.
    """
    for i in range(tabs.count()):
        if tabs.tabText(i) == label:
            widget = tabs.widget(i)
            tabs.removeTab(i)
            return widget
    raise KeyError(f"no tab named {label!r}")


class LearnWorkspace(QWidget):
    """LEARN: the six areas of spec section 4.2, built over the existing panels."""

    changed = Signal()

    def __init__(self, app, agent_panel: AgentPanel, training_panel: TrainingPanel,
                open_settings=None, parent=None) -> None:
        super().__init__(parent)
        self.app = app
        self.agent_panel = agent_panel
        self.training_panel = training_panel
        self._open_settings = open_settings
        self._dash_values: Dict[str, QLabel] = {}

        self.nav = QListWidget()
        self.nav.addItems(AREA_NAMES)
        self.nav.setMaximumWidth(190)
        self.nav.setMinimumWidth(140)

        self.stack = QStackedWidget()
        self.dashboard = self._build_dashboard()
        self.curriculum_page = self._build_curriculum_page()
        self.sources_page = self._build_sources_page()
        self.practice_page = self._build_practice_page()
        self.knowledge_page = self._build_knowledge_page()
        self.history_page = self._build_history_page()
        for page in (self.dashboard, self.curriculum_page, self.sources_page,
                     self.practice_page, self.knowledge_page, self.history_page):
            self.stack.addWidget(page)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)

        # Now that every tab page has been taken out of it, agent_panel's
        # inner tab widget has nothing left to show.
        self.agent_panel.tabs.hide()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.nav)
        layout.addWidget(self.stack, 1)

        self._timer = QTimer(self)
        self._timer.setInterval(800)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    # ==================================================================
    # navigation
    # ==================================================================
    def show_area(self, name: str) -> None:
        if name in AREA_NAMES:
            self.nav.setCurrentRow(AREA_NAMES.index(name))

    def area_names(self) -> List[str]:
        return list(AREA_NAMES)

    # ==================================================================
    # A. Dashboard (new widget - spec 4.2A)
    # ==================================================================
    def _build_dashboard(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        grid_box = QGroupBox("Learning dashboard")
        grid = QGridLayout(grid_box)
        for row, (label, key) in enumerate(DASHBOARD_FIELDS):
            grid.addWidget(QLabel(label + ":"), row, 0)
            value = QLabel("-")
            value.setWordWrap(True)
            value.setObjectName("hint")
            grid.addWidget(value, row, 1)
            self._dash_values[key] = value
        layout.addWidget(grid_box)

        controls_row = QHBoxLayout()
        self.dash_start_btn = QPushButton("Start")
        self.dash_start_btn.setObjectName("primary")
        self.dash_start_btn.clicked.connect(
            lambda: self._run_control(self.app.start_learning))
        self.dash_pause_btn = QPushButton("Pause")
        self.dash_pause_btn.clicked.connect(
            lambda: self._run_control(self.app.pause_learning))
        self.dash_resume_btn = QPushButton("Resume")
        self.dash_resume_btn.clicked.connect(
            lambda: self._run_control(self.app.resume_learning))
        self.dash_stop_btn = QPushButton("Stop")
        self.dash_stop_btn.clicked.connect(
            lambda: self._run_control(self.app.stop_learning))
        for button in (self.dash_start_btn, self.dash_pause_btn,
                       self.dash_resume_btn, self.dash_stop_btn):
            controls_row.addWidget(button)
        controls_row.addStretch(1)
        layout.addLayout(controls_row)

        # The existing "what raaga to study" / "one lesson" / "choose a
        # learning folder" controls, and the live status readout they sit
        # beside - re-homed, not rebuilt.
        layout.addWidget(self.agent_panel.status_box)
        layout.addWidget(self.agent_panel.controls)

        provider_row = QHBoxLayout()
        self.provider_label = QLabel("-")
        self.provider_label.setObjectName("hint")
        self.provider_label.setWordWrap(True)
        settings_btn = QPushButton("Settings...")
        settings_btn.clicked.connect(self._on_settings_clicked)
        provider_row.addWidget(self.provider_label, 1)
        provider_row.addWidget(settings_btn)
        layout.addLayout(provider_row)

        layout.addStretch(1)
        return page

    def _run_control(self, fn) -> None:
        fn()
        self.refresh()

    def _on_settings_clicked(self) -> None:
        if self._open_settings is not None:
            self._open_settings()

    def set_provider_line(self, text: str) -> None:
        """Pushed in by the main window on its own 30s cadence (spec 41)."""
        self.provider_label.setText(text)

    def _refresh_dashboard(self) -> None:
        status = self.app.agent_status()
        activity_state = ("learning" if status["learning"]
                          else ("paused" if status["paused"] else "idle"))
        errors = status.get("errors") or []
        values = {
            "activity_state": activity_state,
            "stage": str(status["stage"]),
            "current_raaga": status["current_raaga"],
            "lesson": status["next_unit"] or "(none)",
            "objective": status["next_goal"],
            "mastery": f"{status['mastery']:.2f}",
            "overall_percent": f"{status['overall_percent']:.0f}%",
            "activity": status.get("activity") or activity_state,
            "last_step": status["last_step"] or "nothing yet this session",
            "errors": "; ".join(errors) if errors else "none",
            "repository_size": (f"{status['facts']} fact(s), "
                                f"{status['phrases']} phrase(s) "
                                f"({status['repository_bytes'] / 1024:.0f} KB)"),
            "sources_analysed": f"{status['sources_analysed']}/{status['sources']}",
            "phrases": str(status["phrases"]),
        }
        for key, label in self._dash_values.items():
            label.setText(str(values.get(key, "-")))

    # ==================================================================
    # B. Curriculum
    # ==================================================================
    def _build_curriculum_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(_take_tab(self.agent_panel.tabs, "Curriculum"), 1)
        self.next_unit_label = QLabel("-")
        self.next_unit_label.setWordWrap(True)
        self.next_unit_label.setObjectName("hint")
        layout.addWidget(self.next_unit_label)
        return page

    def _refresh_curriculum(self) -> None:
        raaga = self.app.agent.curriculum.current_raaga()
        summary = self.app.agent.curriculum.stage_summary(raaga)
        self.next_unit_label.setText(
            f"Next recommended unit: {summary['next_unit'] or '(none)'} "
            f"— {summary['next_goal']}")

    # ==================================================================
    # C. Training Sources - the whole TrainingPanel, minus the two tabs
    # that belong under Knowledge and History instead.
    # ==================================================================
    def _build_sources_page(self) -> QWidget:
        return self.training_panel

    # ==================================================================
    # D. Practice / Quiz (new widget - spec 4.2D, 25)
    # ==================================================================
    def _build_practice_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        run_row = QHBoxLayout()
        self.run_exercise_btn = QPushButton("Run one exercise")
        self.run_exercise_btn.setObjectName("primary")
        self.run_exercise_btn.clicked.connect(self._run_exercise)
        run_row.addWidget(self.run_exercise_btn)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.practice_status = QLabel("Press “Run one exercise” to test "
                                      "the agent on its current unit.")
        self.practice_status.setWordWrap(True)
        self.practice_status.setObjectName("hint")
        layout.addWidget(self.practice_status)

        self.exercise_table = QTableWidget(0, len(EXERCISE_COLUMNS))
        self.exercise_table.setHorizontalHeaderLabels(list(EXERCISE_COLUMNS))
        self.exercise_table.verticalHeader().setVisible(False)
        self.exercise_table.setMinimumHeight(140)
        layout.addWidget(self.exercise_table, 1)

        remediation_box = QGroupBox("Retry / remediation")
        remediation_layout = QVBoxLayout(remediation_box)
        self.lesson_table = QTableWidget(0, len(LESSON_COLUMNS))
        self.lesson_table.setHorizontalHeaderLabels(list(LESSON_COLUMNS))
        self.lesson_table.verticalHeader().setVisible(False)
        self.lesson_table.setMinimumHeight(100)
        remediation_layout.addWidget(self.lesson_table)
        self.lesson_hint = QLabel("No lessons yet for this unit.")
        self.lesson_hint.setWordWrap(True)
        self.lesson_hint.setObjectName("hint")
        remediation_layout.addWidget(self.lesson_hint)
        layout.addWidget(remediation_box)

        layout.addWidget(self.agent_panel.critique_btn)
        layout.addWidget(_take_tab(self.agent_panel.tabs, "Its own critique"))

        self.readiness_label = QLabel("-")
        self.readiness_label.setWordWrap(True)
        self.readiness_label.setObjectName("hint")
        layout.addWidget(self.readiness_label)
        return page

    def _run_exercise(self) -> None:
        raaga = self.app.agent.curriculum.current_raaga()
        unit = self.app.agent.curriculum.next_unit(raaga)
        if unit is None:
            self.practice_status.setText(
                "Curriculum complete for this raaga - nothing left to practice.")
            return
        threshold = unit.minimum_pass_score
        self.practice_status.setText(f"Running {unit.id}...")
        self.run_exercise_btn.setEnabled(False)

        def work(ctx):
            return self.app.agent.practice.run(unit, raaga_name=raaga)

        def done(report) -> None:
            self.run_exercise_btn.setEnabled(True)
            self._fill_exercise_table(report, threshold)
            self.practice_status.setText(report.summary())
            self.refresh()
            self.changed.emit()

        def failed(exc: BaseException) -> None:
            self.run_exercise_btn.setEnabled(True)
            self.practice_status.setText(f"Exercise failed: {exc}")

        self.app.jobs.submit("practice.run", "practice", work, on_done=done,
                             on_error=failed, description=f"Practice: {unit.id}")

    def _fill_exercise_table(self, report, threshold: float) -> None:
        table = self.exercise_table
        table.setRowCount(len(report.exercises))
        for row, exercise in enumerate(report.exercises):
            values = (exercise.name, f"{exercise.score:.2f}", f"{threshold:.2f}",
                      "pass" if exercise.passed else "retry", exercise.detail)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()

    def _fill_lesson_table(self, lessons: List[Any]) -> None:
        table = self.lesson_table
        if not lessons:
            table.hide()
            self.lesson_hint.setText("No lessons yet for this unit.")
            self.lesson_hint.show()
            return
        table.show()
        self.lesson_hint.hide()
        table.setRowCount(len(lessons))
        for row, lesson in enumerate(lessons):
            last_seen = time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(lesson.last_at))
            values = (f"{lesson.kind}: {lesson.failure_reason}",
                     str(lesson.recurrences), last_seen, lesson.correction)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()

    def _refresh_practice(self) -> None:
        raaga = self.app.agent.curriculum.current_raaga()
        summary = self.app.agent.curriculum.stage_summary(raaga)
        mastered = summary["mastered_raagas"]
        # The summary already asked the curriculum for the next unit; asking
        # again from a timer would repeat next_unit's revisit bookkeeping.
        next_unit_id = summary.get("next_unit", "")
        if next_unit_id:
            lessons = self.app.agent.repo.lessons(raaga=raaga,
                                                  unit_id=next_unit_id)
        else:
            lessons = self.app.agent.repo.lessons(raaga=raaga, limit=8)
        self._fill_lesson_table(lessons)
        cross_units = self.app.agent.curriculum.cross_units()
        required = 0
        if cross_units:
            required = int(cross_units[0].params.get("requires_mastered_raagas", 0))
        if required:
            met = len(mastered) >= required
            self.readiness_label.setText(
                f"Readiness for the next level (cross-raaga stage): "
                f"{len(mastered)}/{required} raagas mastered"
                + (" - threshold met, ready to advance." if met else "."))
        else:
            self.readiness_label.setText(
                f"Stage {summary['stage']}, {summary['overall_percent']:.0f}% "
                f"complete for {raaga}.")

    # ==================================================================
    # E. Knowledge
    # ==================================================================
    def _build_knowledge_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(_take_tab(self.agent_panel.tabs, "What it knows"))
        layout.addWidget(_take_tab(self.agent_panel.tabs, "Where it learned it"))

        gaps_box = QGroupBox("Knowledge gaps")
        gaps_layout = QVBoxLayout(gaps_box)
        self.gaps_label = QLabel("No recurring mistakes recorded.")
        self.gaps_label.setWordWrap(True)
        self.gaps_label.setObjectName("hint")
        gaps_layout.addWidget(self.gaps_label)
        layout.addWidget(gaps_box)

        layout.addWidget(self.agent_panel.ask)
        layout.addWidget(self.agent_panel.answer)
        layout.addWidget(_take_tab(self.training_panel._tabs, "Knowledge base"), 1)
        return page

    def _refresh_knowledge_gaps(self) -> None:
        raaga = self.app.agent.curriculum.current_raaga()
        counts = self.app.agent.repo.lesson_counts(raaga)
        recurring = sorted(((kind, n) for kind, n in counts.items() if n >= 2),
                           key=lambda kv: -kv[1])
        if not recurring:
            self.gaps_label.setText("No recurring mistakes recorded.")
            return
        self.gaps_label.setText(
            ", ".join(f"{kind} (x{n})" for kind, n in recurring))

    # ==================================================================
    # F. History / Evaluation
    # ==================================================================
    def _build_history_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(_take_tab(self.training_panel._tabs, "History"), 1)
        layout.addWidget(_take_tab(self.agent_panel.tabs, "Recent activity"), 1)

        findings_box = QGroupBox("Evaluation findings")
        findings_layout = QVBoxLayout(findings_box)
        self.findings_table = QTableWidget(0, len(FINDINGS_COLUMNS))
        self.findings_table.setHorizontalHeaderLabels(list(FINDINGS_COLUMNS))
        self.findings_table.verticalHeader().setVisible(False)
        self.findings_table.setMinimumHeight(120)
        findings_layout.addWidget(self.findings_table)
        layout.addWidget(findings_box)
        return page

    def _refresh_findings(self) -> None:
        raaga = self.app.agent.curriculum.current_raaga()
        lessons = sorted(self.app.agent.repo.lessons(raaga=raaga, limit=30),
                         key=lambda l: -l.last_at)
        table = self.findings_table
        table.setRowCount(len(lessons))
        for row, lesson in enumerate(lessons):
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(lesson.last_at))
            unit_task = lesson.unit_id or lesson.task
            values = (when, unit_task,
                     f"{lesson.kind}: {lesson.failure_reason}", lesson.correction)
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                table.setItem(row, column, item)
        table.resizeColumnsToContents()

    # ==================================================================
    def refresh(self) -> None:
        self._refresh_dashboard()
        self._refresh_curriculum()
        self._refresh_practice()
        self._refresh_knowledge_gaps()
        self._refresh_findings()
        self.agent_panel.refresh()
        if (self.training_panel.training is not None
                and not self.training_panel.training.store.closed):
            self.training_panel.refresh()
