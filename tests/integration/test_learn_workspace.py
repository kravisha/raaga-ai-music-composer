"""Integration: the LEARN workspace's six areas, driven offscreen.

Builds the real MainWindow (same pattern as test_ui_window.py) and exercises
LEARN specifically: the Dashboard reads the real agent, Start/Stop reach the
real background learner, Practice/Quiz runs one real exercise on the shared
JobManager, and the Knowledge area's search+provenance widgets are the very
ones TrainingPanel already had - just re-homed.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication            # noqa: E402

from raagacomposer.agent.knowledge import Lesson       # noqa: E402
from raagacomposer.app import AppController           # noqa: E402
from raagacomposer.ui import theme                    # noqa: E402
from raagacomposer.ui.main_window import MainWindow    # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui, pytest.mark.slow]

SHOTS = Path(os.environ.get("RAAGA_SHOT_DIR", tempfile.gettempdir())) / "raaga-shots"

PHRASE = "Kamboji raga beginner lesson"


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(theme.STYLESHEET)
    return application


@pytest.fixture(scope="module")
def window(qt_app, tmp_path_factory):
    from raagacomposer.core.settings import Settings

    directory = tmp_path_factory.mktemp("learn-ui")
    settings = Settings.load()
    settings.projects_dir = str(directory / "projects")
    settings.stt_provider = "none"

    controller = AppController(settings)
    win = MainWindow(controller)
    win.resize(1500, 950)
    win.show()
    win.set_workspace("LEARN")
    _pump(qt_app, 0.3)
    try:
        yield win
    finally:
        win._timer.stop()
        win._provider_timer.stop()
        controller.close()


def _pump(qt_app: QApplication, seconds: float = 0.2) -> None:
    end = time.time() + seconds
    while time.time() < end:
        qt_app.processEvents()
        time.sleep(0.01)


def _wait_for(qt_app: QApplication, controller: AppController, predicate,
             timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        qt_app.processEvents()
        controller.pump()
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError("condition was never met")


def _shot(window: MainWindow, name: str) -> Path:
    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / f"{name}.png"
    window.grab().save(str(path))
    return path


# --------------------------------------------------------------------------
# A. Dashboard
# --------------------------------------------------------------------------
def test_dashboard_shows_the_current_raaga_and_stage(window, qt_app):
    app = window.app
    window.learn_workspace.refresh()
    _pump(qt_app, 0.1)

    status = app.agent_status()
    dash = window.learn_workspace._dash_values
    assert dash["current_raaga"].text() == status["current_raaga"]
    assert dash["stage"].text() == str(status["stage"])
    assert dash["current_raaga"].text() == "Keeravani"
    _shot(window, "09-learn-dashboard")


def test_start_then_stop_learning_updates_the_dashboard(window, qt_app):
    app = window.app
    dash = window.learn_workspace

    dash.dash_start_btn.click()
    _wait_for(qt_app, app, lambda: app.agent_status()["learning"], timeout=10.0)
    _pump(qt_app, 0.2)
    assert dash._dash_values["activity_state"].text() == "learning"

    dash.dash_stop_btn.click()
    _wait_for(qt_app, app, lambda: not app.agent_status()["learning"], timeout=15.0)
    _pump(qt_app, 0.2)
    assert dash._dash_values["activity_state"].text() in ("idle", "paused")


# --------------------------------------------------------------------------
# B. Curriculum
# --------------------------------------------------------------------------
def test_curriculum_area_shows_the_next_recommended_unit(window, qt_app):
    window.learn_workspace.show_area("Curriculum")
    window.learn_workspace.refresh()
    _pump(qt_app, 0.1)
    assert "Next recommended unit" in window.learn_workspace.next_unit_label.text()


# --------------------------------------------------------------------------
# D. Practice / Quiz
# --------------------------------------------------------------------------
def test_running_one_exercise_lists_a_scored_row(window, qt_app):
    app = window.app
    window.learn_workspace.show_area("Practice / Quiz")
    practice = window.learn_workspace

    practice.run_exercise_btn.click()
    _wait_for(qt_app, app,
             lambda: practice.exercise_table.rowCount() > 0, timeout=90.0)
    _pump(qt_app, 0.2)

    assert practice.exercise_table.rowCount() >= 1
    score_text = practice.exercise_table.item(0, 1).text()
    assert float(score_text) >= 0.0
    _shot(window, "10-learn-practice")


def test_practice_area_lists_lessons_for_the_current_unit(window, qt_app):
    app = window.app
    raaga = app.agent.curriculum.current_raaga()
    next_unit = app.agent.curriculum.next_unit(raaga)
    assert next_unit is not None
    app.agent.repo.add_lesson(Lesson(
        raaga=raaga, unit_id=next_unit.id, kind="outside_swara",
        dimension="swara_correctness",
        failure_reason="1 note(s) outside Keeravani: G3", evidence="G3",
        correction=f"stay inside {raaga}"))

    window.learn_workspace.show_area("Practice / Quiz")
    window.learn_workspace.refresh()
    _pump(qt_app, 0.2)

    assert window.learn_workspace.lesson_table.rowCount() >= 1

    window.learn_workspace.show_area("History / Evaluation")
    _pump(qt_app, 0.1)
    assert window.learn_workspace.findings_table.rowCount() >= 1


# --------------------------------------------------------------------------
# E. Knowledge
# --------------------------------------------------------------------------
def test_knowledge_area_can_search_and_show_provenance(window, qt_app):
    app = window.app
    training = app.training
    assert training is not None, "the training system should be available"

    results = training.search(PHRASE)
    assert results
    training.add_to_queue([r.source_id for r in results[:1]])
    report = training.learn_one_now()
    assert report is not None
    assert training.search_knowledge()

    window.learn_workspace.show_area("Knowledge")
    _pump(qt_app, 0.1)

    window.training_panel.knowledge_query.setText("")
    window.training_panel._search_knowledge()
    _pump(qt_app, 0.1)
    assert window.training_panel.knowledge_table.rowCount() >= 1

    window.training_panel.knowledge_table.setCurrentCell(0, 0)
    window.training_panel._show_provenance()
    _pump(qt_app, 0.1)
    assert window.training_panel.provenance_view.toPlainText().strip()
    _shot(window, "11-learn-knowledge")


# --------------------------------------------------------------------------
# status bar (spec 41)
# --------------------------------------------------------------------------
def test_the_status_bar_provider_label_names_claude(window, qt_app):
    window._refresh_provider_status()
    _pump(qt_app, 0.1)
    assert "Claude" in window.provider_status_label.text()
