"""Integration: the Voice panel's Record, Stop and Cancel, offscreen.

The state of the input must be unmistakable on screen: a red RECORDING
line with the section and the seconds while a take runs, Stop and Cancel
live only then, and the take on the list the moment Stop is pressed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication            # noqa: E402

from raagacomposer.ui import theme                    # noqa: E402
from raagacomposer.ui.panels.voice_panel import VoicePanel  # noqa: E402
from raagacomposer.voice.recorder import TakeRecorder  # noqa: E402

from test_take_recording import FakeInput, _a_tune, _pallavi  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui]


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(theme.STYLESHEET)
    return application


def test_the_panel_shows_recording_and_lists_the_take(app, qt_app):
    _a_tune(app, "Panel take")
    mic = FakeInput()
    app.recorder = TakeRecorder(open_stream=mic, sample_rate=app.sample_rate)
    app.selection = (_pallavi(app).start + 0.1, _pallavi(app).end)
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        assert panel.record_btn.isEnabled()
        assert not panel.stop_take_btn.isEnabled() and not panel.cancel_take_btn.isEnabled()
        assert not panel.play_take_btn.isEnabled()
        assert panel.recording_state.text() == "Not recording"

        panel.record_btn.click()
        qt_app.processEvents()
        assert app.recorder.recording
        state = panel.recording_state.text()
        assert "RECORDING" in state and "PALLAVI" in state, state
        assert "d02020" in panel.recording_state.styleSheet()
        assert not panel.record_btn.isEnabled()
        assert panel.stop_take_btn.isEnabled() and panel.cancel_take_btn.isEnabled()

        mic.last.feed(0.5)
        panel.stop_take_btn.click()
        qt_app.processEvents()
        assert not app.recorder.recording and mic.last.released
        assert len(app.project.recordings) == 1
        assert panel.recording_state.text() == "Not recording"
        rows = [panel.takes_list.item(i).text() for i in range(panel.takes_list.count())]
        assert rows and "Take 1 - Pallavi" in rows[0] and "recorded by you" in rows[0], rows
        # Playing is by choice, not "the latest": nothing chosen, nothing to play.
        assert not panel.play_take_btn.isEnabled()
        panel.takes_list.item(0).setSelected(True)
        qt_app.processEvents()
        assert panel.play_take_btn.isEnabled()

        # Cancel keeps nothing, and says so on the list.
        panel.record_btn.click()
        qt_app.processEvents()
        mic.last.feed(0.2)
        panel.cancel_take_btn.click()
        qt_app.processEvents()
        assert len(app.project.recordings) == 1 and mic.last.released
        assert panel.recording_state.text() == "Not recording"
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()


def test_the_panel_says_when_there_is_nothing_to_record_with(app, qt_app):
    _a_tune(app, "Panel without input")
    app.recorder = TakeRecorder(backend_present=False, sample_rate=app.sample_rate)
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        assert not panel.record_btn.isEnabled()
        assert panel.recording_state.text().startswith("Input unavailable")
        assert "sounddevice" in panel.record_btn.toolTip()
    finally:
        panel.close()
        panel.deleteLater()
        qt_app.processEvents()
