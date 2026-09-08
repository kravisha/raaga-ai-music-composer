"""Integration: the production team through the real window.

The controller path is covered in test_production_flow.py.  This drives
the same production from the menu, offscreen, because a Compose action
with no caller is exactly the gap a controller test cannot see.
"""
from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction                     # noqa: E402
from PySide6.QtWidgets import QApplication, QPlainTextEdit  # noqa: E402

from raagacomposer.app import AppController           # noqa: E402
from raagacomposer.production.contracts import CodexCritic  # noqa: E402
from raagacomposer.ui import theme                    # noqa: E402
from raagacomposer.ui.main_window import MainWindow   # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui, pytest.mark.slow]


class FakeCodex:
    provider = "codex"
    model = "test-double"

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, schema, *, cancelled=None):
        packet = json.loads(prompt.split("Evidence packet:\n", 1)[1])
        self.calls.append(packet["stage"])
        return dict(request_id=packet["request_id"], stage=packet["stage"],
                    accept=True, blocked=False,
                    reason="The supplied evidence supports this stage",
                    revisions=[], strengths=[], lessons=[], uncertainty=[],
                    evidence=[])


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(theme.STYLESHEET)
    return application


def build_window(tmp_dir):
    from raagacomposer.core.settings import Settings

    settings = Settings.load()
    settings.projects_dir = str(tmp_dir / "projects")
    settings.stt_provider = "none"
    controller = AppController(settings)
    win = MainWindow(controller)
    win.resize(1500, 950)
    win.show()
    return win


def close_window(win):
    win._timer.stop()
    win._provider_timer.stop()
    win.app.close()


def action_named(win, text: str) -> QAction:
    matches = [a for a in win.findChildren(QAction) if a.text() == text]
    assert len(matches) == 1, (text, [a.text() for a in win.findChildren(QAction)])
    return matches[0]


def spin(qt_app, app, until, seconds: float, what: str) -> None:
    end = time.time() + seconds
    while time.time() < end:
        qt_app.processEvents()
        app.pump()
        if until():
            return
        time.sleep(0.02)
    raise AssertionError(f"{what}: {app.status_text}")


def test_the_window_says_who_the_critic_is_at_startup(qt_app, tmp_path):
    win = build_window(tmp_path)
    try:
        message = win.statusBar().currentMessage()
        assert message.startswith("Ready - Codex reviewer"), message
        assert win.app.production_status() in message
        for text in ("Produce a whole song", "Stop the production",
                     "Production report..."):
            action_named(win, text)
    finally:
        close_window(win)


def test_compose_produces_a_whole_song_and_reports_it(qt_app, tmp_path):
    win = build_window(tmp_path)
    app = win.app
    try:
        app.new_project("From the menu", write=False)
        app.update_brief(duration_target=60, language="Tamil", tempo_preference=108,
                         situation="A hopeful reunion after a long separation",
                         notes="Include Prelude, Pallavi, Anupallavi, Interlude, "
                               "Charanam and Ending.")
        app.select_raaga("Hamsadhwani")
        codex = FakeCodex()
        app.critic = CodexCritic(codex)

        action_named(win, "Produce a whole song").trigger()
        assert app.producer is not None and not app.producer.finished
        spin(qt_app, app, lambda: app.producer.finished, 600,
             "the production never finished")
        producer = app.producer
        assert producer.phase == "done", producer.report()
        assert producer.accepted == 7 and len(codex.calls) == 7
        assert "Whole song produced" in win.statusBar().currentMessage()

        action_named(win, "Production report...").trigger()
        qt_app.processEvents()
        dialog = win._production_report_dialog
        assert dialog.isVisible()
        text = dialog.findChild(QPlainTextEdit)
        assert "Codex accepted 7 of 7 stages" in text.toPlainText()
        dialog.close()
    finally:
        close_window(win)


def test_stop_the_production_cancels_it(qt_app, tmp_path):
    win = build_window(tmp_path)
    app = win.app
    try:
        app.new_project("Stopped from the menu", write=False)
        app.update_brief(duration_target=60, situation="A hopeful reunion")
        app.select_raaga("Hamsadhwani")
        import threading

        class HoldTheTune(FakeCodex):
            """Holds the tune review open so Stop lands while the tune's
            follow-on hum render is still in flight."""

            def __init__(self):
                super().__init__()
                self.entered, self.release = threading.Event(), threading.Event()

            def __call__(self, prompt, schema, *, cancelled=None):
                packet = json.loads(prompt.split("Evidence packet:\n", 1)[1])
                if packet["stage"] == "tune":
                    self.entered.set()
                    assert self.release.wait(30), "the test must release the reviewer"
                return super().__call__(prompt, schema, cancelled=cancelled)

        gate = HoldTheTune()
        app.critic = CodexCritic(gate)
        try:
            action_named(win, "Produce a whole song").trigger()
            spin(qt_app, app, gate.entered.is_set, 120, "the tune review never began")
            assert app.producer.phase == "reviewing" and app.producer.stage == "tune"
            action_named(win, "Stop the production").trigger()
            assert app.producer.phase == "cancelled"
            assert "stopped" in win.statusBar().currentMessage().lower()
        finally:
            gate.release.set()
        spin(qt_app, app, lambda: not app.jobs.active_jobs(), 60,
             "jobs kept running after Stop")
        assert app.producer.phase == "cancelled"
        # Nothing lands after Stop: the tune it had is the tune it keeps, no
        # words are written for it, and the held review's late answer is not
        # a verdict.  (The tune's hum render precedes its review, so it has
        # already happened by the time anyone can press Stop.)
        assert len(app.project.melodies) == 1 and not app.project.lyrics
        assert app.producer.accepted == 1, "only the brief was reviewed before Stop"
        assert not any(r.stage == "tune" for r in app.producer.state.records)
    finally:
        close_window(win)
