"""Integration: the real Qt window, driven offscreen.

Builds the actual MainWindow and every panel, runs the creative workflow
through it, and checks the widgets reflect the project.  Screenshots are
written so a layout change can be inspected; set RAAGA_SHOT_DIR to choose
where.
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication            # noqa: E402

from raagacomposer.app import AppController           # noqa: E402
from raagacomposer.ui import theme                    # noqa: E402
from raagacomposer.ui.main_window import MainWindow   # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.ui, pytest.mark.slow]

SHOTS = Path(os.environ.get("RAAGA_SHOT_DIR", tempfile.gettempdir())) / "raaga-shots"


@pytest.fixture(scope="module")
def qt_app():
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    application.setStyleSheet(theme.STYLESHEET)
    return application


@pytest.fixture(scope="module")
def window(qt_app, tmp_path_factory):
    from raagacomposer.core.settings import Settings

    directory = tmp_path_factory.mktemp("ui")
    settings = Settings.load()
    settings.projects_dir = str(directory / "projects")
    settings.stt_provider = "none"

    controller = AppController(settings)
    win = MainWindow(controller)
    win.resize(1500, 950)
    win.show()
    _pump(qt_app, 0.3)
    try:
        yield win
    finally:
        win._timer.stop()
        controller.close()


def _pump(qt_app: QApplication, seconds: float = 0.2) -> None:
    end = time.time() + seconds
    while time.time() < end:
        qt_app.processEvents()
        time.sleep(0.01)


def _settle(qt_app: QApplication, controller: AppController,
            timeout: float = 240.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        qt_app.processEvents()
        controller.pump()
        if not controller.jobs.active_jobs():
            _pump(qt_app, 0.1)
            if not controller.jobs.active_jobs():
                return
        time.sleep(0.02)
    raise TimeoutError("jobs did not finish")


def _shot(window: MainWindow, name: str) -> Path:
    SHOTS.mkdir(parents=True, exist_ok=True)
    path = SHOTS / f"{name}.png"
    window.grab().save(str(path))
    return path


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------
def test_the_window_builds_with_every_panel(window):
    assert window.windowTitle().startswith("Raaga AI Music Composer")
    for panel in (window.project_panel, window.brief_panel, window.raaga_panel,
                  window.tune, window.lyrics, window.voice, window.output,
                  window.arrangement, window.conversation, window.agent_panel):
        assert panel is not None
    assert window.tabs.count() == 5
    assert [window.tabs.tabText(i) for i in range(5)] == \
        ["Tune", "Lyrics", "Voice", "Output", "Learning"]


def test_the_window_fits_an_ordinary_screen(window):
    hint = window.minimumSizeHint()
    assert hint.width() <= 1280, "the window demands too much width"
    assert hint.height() <= 800, "the window demands too much height"


def test_the_transport_and_menus_exist(window):
    assert window.play_btn.text() == "Play"
    titles = [a.text() for a in window.menuBar().actions()]
    assert "&File" in titles and "&Compose" in titles


def test_an_empty_project_renders(window, qt_app):
    _pump(qt_app, 0.2)
    assert _shot(window, "01-empty").exists()


# --------------------------------------------------------------------------
# the workflow through the UI
# --------------------------------------------------------------------------
def test_the_brief_and_raaga_panels_drive_the_project(window, qt_app):
    app = window.app
    app.new_project("Terrace at Midnight")
    window.brief_panel.situation.setText("a man alone on a terrace after midnight")
    window.brief_panel.mood.setCurrentText("longing")
    window.brief_panel.feel.setPlainText("lonely, late at night, but still warm")
    window.brief_panel.language.setCurrentText("Tamil")
    window.brief_panel.duration.setValue(60.0)
    window.brief_panel.apply()

    assert app.project.brief.feel.startswith("lonely")

    window.raaga_panel.suggest()
    assert window.raaga_panel.suggestions.count() >= 2
    window.raaga_panel.accept_selected()
    assert app.project.raaga.selected
    window.refresh()
    _shot(window, "02-brief-and-raagas")


def test_the_tune_panel_generates_and_lists_sections(window, qt_app):
    app = window.app
    window.tune.generate_btn.click()
    _settle(qt_app, app)
    window.refresh()

    assert app.project.melody() is not None
    assert window.tune.sections.rowCount() == len(app.project.melody().sections)
    assert window.tune.versions.count() >= 1
    assert "fidelity" in window.tune.report.toPlainText()
    _shot(window, "03-tune")


def test_locking_the_tune_updates_the_panel(window, qt_app):
    app = window.app
    window.tune.accept_btn.click()
    window.refresh()
    assert app.project.melody().state.value == "locked"
    assert "locked" in window.tune.versions.currentText()


def test_the_lyrics_panel_shows_the_fit(window, qt_app):
    app = window.app
    window.tabs.setCurrentIndex(1)
    window.lyrics.generate_btn.click()
    _settle(qt_app, app)
    window.refresh()

    lyrics = app.project.lyrics_version()
    assert lyrics is not None
    assert window.lyrics.table.rowCount() == len(lyrics.lines)
    assert "MISFIT" not in window.lyrics.alignment.toPlainText()
    _shot(window, "04-lyrics")


def test_editing_a_line_in_the_table_refits_it(window, qt_app):
    app = window.app
    window.lyrics.table.item(0, 3).setText("puthiya vaanam")
    _pump(qt_app, 0.2)
    line = app.project.lyrics_version().lines[0]
    assert line.text == "puthiya vaanam"
    assert len(line.syllables) == len(line.note_indices)


def test_the_voice_panel_produces_the_vocal_only_master(window, qt_app):
    app = window.app
    window.tabs.setCurrentIndex(2)
    window.voice.style_box.setCurrentText("sad")
    window.voice._apply_direction()
    window.voice.master_btn.click()
    _settle(qt_app, app)
    window.refresh()

    assert app.project.vocal_master is not None
    assert Path(app.project.vocal_master.audio_path).exists()
    assert "master" in window.voice.info.toPlainText()
    _shot(window, "05-voice")


def test_the_arrangement_panel_adds_an_instrument(window, qt_app):
    app = window.app
    window.tabs.setCurrentIndex(0)
    index = window.arrangement.instrument_box.findData("veena")
    window.arrangement.instrument_box.setCurrentIndex(index)
    window.arrangement.whole_box.setChecked(True)
    window.arrangement._add()
    _settle(qt_app, app)
    window.refresh()

    assert app.project.arrangement().tracks_for_instrument("veena")
    assert window.arrangement.tracks.rowCount() >= 1


def test_the_timeline_reflects_the_project(window, qt_app):
    app = window.app
    window.arrangement._auto = None
    timeline = window.arrangement.timeline
    assert timeline.melody is app.project.melody()
    assert timeline.arrangement is app.project.arrangement()
    assert timeline.duration >= app.project.melody().duration - 1

    timeline.zoom_to_fit(window.arrangement.scroll.viewport().width())
    assert timeline.pps > 0
    timeline.set_playhead(12.0)
    assert timeline.playhead == pytest.approx(12.0)
    timeline.set_selection((10.0, 20.0))
    assert timeline.selection == (10.0, 20.0)


def test_a_timeline_selection_reaches_the_controller(window, qt_app):
    app = window.app
    window.arrangement._selection_changed((5.0, 15.0))
    assert app.selection == (5.0, 15.0)
    assert window.arrangement.start_spin.value() == pytest.approx(5.0)
    window.arrangement._selection_changed(None)
    assert app.selection is None


def test_the_conversation_panel_runs_a_typed_instruction(window, qt_app):
    app = window.app
    window.conversation.entry.setText("Add flute.")
    window.conversation._submit()
    _settle(qt_app, app)
    window.refresh()

    assert app.project.arrangement().tracks_for_instrument("flute")
    assert window.conversation.history.count() >= 1
    assert "Flute" in window.conversation.interpretation.toPlainText()


def test_the_full_mix_and_output_panel(window, qt_app):
    app = window.app
    window.tabs.setCurrentIndex(3)
    window.output.refresh()
    app.render("full", autoplay=False)
    _settle(qt_app, app)
    window.refresh()

    assert app.project.latest_mix("full") is not None
    text = window.output.info.toPlainText()
    assert "full" in text and "Providers" in text
    _shot(window, "06-arrangement")
    window.tabs.setCurrentIndex(0)
    window.refresh()
    _shot(window, "07-output")


def test_undo_and_redo_from_the_menu(window, qt_app):
    app = window.app
    before = len(app.project.arrangement().tracks)
    window.undo_action.trigger()
    _pump(qt_app, 0.1)
    assert len(app.project.arrangement().tracks) <= before
    window.redo_action.trigger()
    _pump(qt_app, 0.1)
    assert len(app.project.arrangement().tracks) == before


def test_saving_and_reopening_through_the_window(window, qt_app):
    app = window.app
    window.save_project()
    directory = app.project_dir
    assert (directory / "project.json").exists()

    window._open_path(str(directory))
    _pump(qt_app, 0.2)
    assert app.project.melody() is not None
    assert app.project.vocal_master is not None
    assert window.tune.sections.rowCount() > 0


def test_the_help_text_lists_the_spoken_commands(window):
    from raagacomposer.ui.main_window import HELP_TEXT
    for phrase in ("Play the first minute",
                   "from the second minute to the third minute",
                   "Add veena here", "without instruments"):
        assert phrase in HELP_TEXT
