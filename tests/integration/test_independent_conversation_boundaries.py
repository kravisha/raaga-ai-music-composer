"""Independent acceptance probes, only in the immutable review snapshot."""
import threading
import time

import pytest

from raagacomposer.speech.intent import Command


def finish(app):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        app.pump()
        if not app.jobs.active_jobs() and app._interpreting is None:
            app.pump()
            return
        time.sleep(0.01)
    raise AssertionError("interpretation failed to settle")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_full_typed_queue_preserves_unaccepted_entry(app, qt_app):
    from raagacomposer.ui.panels.conversation_panel import ConversationPanel
    panel = ConversationPanel(app)
    for i in range(64):
        assert app.say(f"queued instruction {i}")
    requested = "compose a tune in Hamsadhwani"
    panel.entry.setText(requested)
    panel.entry.returnPressed.emit()
    observed = panel.entry.text()
    assert app._typed_queue.qsize() == 64
    panel.close()
    assert observed == requested, "full queue rejected the instruction but the input box erased it"


def test_queued_instruction_does_not_cross_to_new_song(app):
    app.new_project("Original", write=False)
    assert app.say("set raga Hamsadhwani")
    app.new_project("Replacement", write=False)
    app.select_raaga("Keeravani")
    finish(app)
    assert app.project.raaga.selected == "Keeravani", "an instruction queued for Original changed Replacement"


def _hold_interpretation(app, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def held(text, *args, **kwargs):
        started.set()
        assert release.wait(5), "probe did not release the interpreter"
        return Command(intent="raaga.set", raaga="Hamsadhwani", text=text,
                       confidence=1.0)

    monkeypatch.setattr("raagacomposer.app.interpret", held)
    assert app.say("set raga Hamsadhwani")
    app._drain_utterances()
    assert started.wait(3), "background interpretation did not start"
    return release


def test_stop_listening_preserves_inflight_typed_instruction(app, monkeypatch):
    app.select_raaga("Keeravani")
    release = _hold_interpretation(app, monkeypatch)
    try:
        app.stop_listening()
    finally:
        release.set()
    finish(app)
    assert app.project.raaga.selected == "Hamsadhwani", "Stop Listening discarded a typed instruction already being interpreted"


def test_inflight_instruction_does_not_cross_to_new_song(app, monkeypatch):
    app.new_project("Original", write=False)
    release = _hold_interpretation(app, monkeypatch)
    try:
        app.new_project("Replacement", write=False)
        app.select_raaga("Keeravani")
    finally:
        release.set()
    finish(app)
    assert app.project.raaga.selected == "Keeravani", "an old interpretation completion changed the replacement song"


def test_ordinary_typed_instruction_still_applies(app):
    app.select_raaga("Keeravani")
    assert app.say("set raga Hamsadhwani")
    assert app.project.raaga.selected == "Keeravani"
    finish(app)
    assert app.project.raaga.selected == "Hamsadhwani"
