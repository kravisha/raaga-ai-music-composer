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

from PySide6.QtCore import Qt                         # noqa: E402
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
        win._provider_timer.stop()
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
    # The song's name leads the title bar, the way a document's does.
    assert window.windowTitle().endswith("Raaga AI Music Composer")
    assert window.app.project.title in window.windowTitle()
    for panel in (window.brief_panel, window.raaga_panel,
                  window.tune, window.lyrics, window.voice, window.output,
                  window.arrangement, window.conversation, window.agent_panel,
                  window.training_panel, window.learn_workspace):
        assert panel is not None
    # v0.3 section 4 / TEST I: Learning left the composer's own tab bar.
    assert window.tabs.count() == 4
    assert [window.tabs.tabText(i) for i in range(4)] == \
        ["Tune", "Lyrics", "Voice", "Output"]


def test_the_window_fits_an_ordinary_screen(window):
    hint = window.minimumSizeHint()
    assert hint.width() <= 1280, "the window demands too much width"
    assert hint.height() <= 800, "the window demands too much height"


def test_a_song_can_be_more_than_one_mood(window):
    """Specification 9.1: mood is multi-select.

    A song is rarely one feeling - "hopeful and romantic" is an ordinary
    brief - and a single-choice box threw half of it away.
    """
    panel = window.brief_panel
    panel.mood.setCurrentText("hopeful, romantic")
    assert panel.mood.checked() == ["romantic", "hopeful"] or \
        set(panel.mood.checked()) == {"hopeful", "romantic"}
    assert "hopeful" in panel.mood.currentText()
    assert "romantic" in panel.mood.currentText()

    panel.apply()
    assert "hopeful" in window.app.project.brief.mood
    assert "romantic" in window.app.project.brief.mood

    # Both feelings must reach the emotion vector, not just the first.
    from raagacomposer.raaga import emotion
    vector = emotion.target_vector(window.app.project.brief)
    assert vector["romance"] > 0.2, "the romantic half was lost"
    assert vector["brightness"] > 0.2, "the hopeful half was lost"

    # A mood the list does not know is kept rather than dropped.
    panel.mood.setCurrentText("wistful")
    assert panel.mood.currentText() == "wistful"


def test_a_new_brief_opens_on_a_real_situation(window):
    """An empty box asks the creator to invent a starting point; a
    sentence asks them to edit one."""
    from raagacomposer.core.models import CreativeBrief

    fresh = CreativeBrief()
    assert "novice musician" in fresh.situation
    assert "," in fresh.mood, "the default should show that moods are plural"


def test_the_voice_pipeline_is_visible_stage_by_stage(window, qt_app):
    """Spec: no silent "ready" state with no output.

    A creator has to be able to tell a misheard phrase from one heard
    correctly and not understood, and from one understood and refused.
    """
    app = window.app
    panel = window.conversation

    app.handle_utterance("hello how are you doing")
    _pump(qt_app, 0.1)
    panel.refresh()
    assert panel.heard_label.text() == "hello how are you doing"
    assert panel.intent_label.text() in ("unknown", "not recognised")
    assert "Not understood" in panel.result_label.text()
    assert "could not tell" in panel.result_label.text()

    app.handle_utterance("add a theremin")
    _pump(qt_app, 0.1)
    panel.refresh()
    assert "Failed" in panel.result_label.text()
    assert "theremin" in panel.result_label.text()
    # An instrument it does have takes a different path entirely.
    assert panel.action_label.text() != "-"


def test_the_transport_and_menus_exist(window):
    assert window.play_btn.text() == "Play"
    titles = [a.text() for a in window.menuBar().actions()]
    assert "&File" in titles and "&Compose" in titles


def _visible_menus(window):
    return [a.text() for a in window.menuBar().actions() if a.isVisible()]


def _live_shortcuts(window):
    """Every shortcut that would actually fire in the current workspace."""
    live = set()
    for menu_action in window.menuBar().actions():
        menu = menu_action.menu()
        if menu is None or not menu_action.isVisible():
            continue
        for action in menu.actions():
            if action.isEnabled() and action.shortcut().toString():
                live.add(action.shortcut().toString())
    return live


def test_each_workspace_gets_only_its_own_menus(window):
    """MAIN and LEARN are separate top-level workspaces.

    File is a *project's* file, Edit is a project's undo stack, Compose
    makes a project's music - and LEARN is not part of any project, so
    none of those belong to it.  What LEARN does benefits the agent and
    touches only the knowledge store.
    """
    window.set_workspace("MAIN")
    assert _visible_menus(window) == \
        ["&File", "&Edit", "&View", "&Compose", "&Voice control", "&Help"]

    window.set_workspace("LEARN")
    assert _visible_menus(window) == \
        ["&Learn", "&View", "&Voice control", "&Help"]

    window.set_workspace("MAIN")
    assert "&File" in _visible_menus(window)


def test_a_hidden_menus_shortcuts_do_not_still_fire(window):
    """Hiding a menu leaves its shortcuts armed - so they are disabled too.

    Otherwise Ctrl+T would generate a tune while you were reading the
    agent's curriculum.
    """
    window.set_workspace("LEARN")
    live = _live_shortcuts(window)
    for dead in ("Ctrl+T", "Ctrl+S", "Ctrl+M", "Ctrl+N", "Ctrl+L"):
        assert dead not in live, f"{dead} still fires from the LEARN screen"
    # What the application owns stays reachable from both.
    for alive in ("Ctrl+,", "Ctrl+1", "Ctrl+2", "Ctrl+Space"):
        assert alive in live, f"{alive} should still work in LEARN"
    assert "Ctrl+Shift+L" in live

    window.set_workspace("MAIN")
    back = _live_shortcuts(window)
    assert "Ctrl+T" in back and "Ctrl+S" in back


def test_undo_does_not_come_back_on_the_next_timer_tick(window, qt_app):
    """refresh() runs on a timer and used to re-enable Undo unconditionally."""
    window.app.update_brief(situation="a terrace after midnight")
    window.set_workspace("MAIN")
    window.refresh()
    assert window.app.undo.can_undo
    assert window.undo_action.isEnabled()

    window.set_workspace("LEARN")
    window.refresh()
    _pump(qt_app, 0.1)
    assert not window.undo_action.isEnabled(), "Ctrl+Z came back in LEARN"

    window.set_workspace("MAIN")
    window.refresh()
    assert window.undo_action.isEnabled()


def test_the_project_readout_is_hidden_where_there_is_no_project(window):
    window.set_workspace("MAIN")
    assert window.project_status_label.isVisible()
    window.set_workspace("LEARN")
    assert not window.project_status_label.isVisible()
    # The transport stays in both - an ear test may need to play something.
    assert window.play_btn.isVisible()
    window.set_workspace("MAIN")


def test_project_management_lives_in_the_file_menu(window):
    """The Project panel was removed from the left column on purpose.

    It was the tallest thing there, and it pushed the creative controls
    below the fold.  New / Open / Save / Save As / Recent belong in File,
    where every other application keeps them, so the space they used goes
    back to the work.
    """
    assert not hasattr(window, "project_panel")
    file_menu = next(a.menu() for a in window.menuBar().actions()
                     if a.text() == "&File")
    labels = [a.text() for a in file_menu.actions()]
    for wanted in ("New project", "Open project...", "Save", "Save As..."):
        assert wanted in labels, f"{wanted} is not in the File menu"
    assert any(a.menu() is window.recent_menu for a in file_menu.actions())


def test_the_left_column_fits_the_window(window):
    """The point of the reorganisation: no controls cut off below the fold."""
    left = window.brief_panel.parentWidget()
    tall = sum(w.sizeHint().height()
               for w in (window.brief_panel, window.raaga_panel))
    assert left is window.raaga_panel.parentWidget()
    assert tall <= 900, f"the left column wants {tall}px and will be cut off"


def test_the_song_name_is_not_editable_on_screen(window):
    """Renaming happens through Save As, so no stray name field on screen."""
    assert window.brief_panel.title.parentWidget() is None
    assert not window.brief_panel.title.isVisible()


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


def test_apply_brief_shows_progress_and_populates_suggestions(window, qt_app):
    """v0.3 TEST A / section 6.1: clicking Apply must visibly rank raagas
    from the brief - not merely save it - and never do nothing."""
    app = window.app
    window.brief_panel.situation.setText("a long journey home, alone")
    window.brief_panel.mood.setCurrentText("hopeful")
    window.brief_panel.feel.setPlainText("tired, but still hopeful")
    window.brief_panel.apply()
    _settle(qt_app, app)
    window.refresh()

    assert window.raaga_panel.suggestions.count() >= 1
    assert app.last_suggestions
    assert "suggested" in window.brief_panel.status_label.text().lower()


# --------------------------------------------------------------------------
# TEST I (v0.3 section 63): LEARN is a separate top-level workspace, not a
# tab squeezed in beside Tune and Lyrics.
# --------------------------------------------------------------------------
def test_learn_is_a_top_level_workspace_not_a_composition_tab(window, qt_app):
    app = window.app
    assert window.workspaces.count() == 2

    main_tab_texts = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert "Learning" not in main_tab_texts
    assert "Training" not in main_tab_texts

    # switch to LEARN via the toolbar button
    window.learn_ws_btn.click()
    _pump(qt_app, 0.1)
    assert window.workspaces.currentWidget() is window.learn_workspace
    area_names = [window.learn_workspace.nav.item(i).text()
                  for i in range(window.learn_workspace.nav.count())]
    assert area_names == ["Dashboard", "Curriculum", "Training Sources",
                          "Practice / Quiz", "Knowledge", "History / Evaluation"]
    _shot(window, "08-learn-workspace")

    # back to MAIN via the toolbar
    window.main_ws_btn.click()
    _pump(qt_app, 0.1)
    assert window.workspaces.currentWidget() is window.main_page

    # switch to LEARN via the View menu action too
    window.learn_ws_action.trigger()
    _pump(qt_app, 0.1)
    assert window.workspaces.currentWidget() is window.learn_workspace
    assert window.learn_ws_btn.isChecked()

    window.main_ws_action.trigger()
    _pump(qt_app, 0.1)
    assert window.workspaces.currentWidget() is window.main_page
    assert window.main_ws_btn.isChecked()

    # switching back and forth did not disturb MAIN's own state
    assert window.raaga_panel.suggestions.count() >= 1
    assert app.last_suggestions


def test_the_settings_action_exists_and_builds_the_dialog(window, monkeypatch):
    edit_menu = next(a.menu() for a in window.menuBar().actions()
                     if a.text() == "&Edit")
    assert "Settings..." in [a.text() for a in edit_menu.actions()]

    from raagacomposer.ui.settings_dialog import SettingsDialog
    monkeypatch.setattr(SettingsDialog, "exec", lambda self: 0)
    window._open_settings()
    assert "Claude" in window.provider_status_label.text()


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


def test_the_tala_picker_offers_the_cycles_and_their_shape(window):
    """Tala is chosen, not inferred from a beat count (specification 11.4)."""
    panel = window.tune
    labels = [panel.tala.itemText(i) for i in range(panel.tala.count())]
    # The first entry is the one that leaves the choice to the application.
    # It reads "Automatic" rather than "From the tune" because with no tune
    # there is nothing to take it from: the cycle comes from the brief.
    assert panel.tala.itemData(0) == "", "there must be a way not to choose"
    assert labels[0] == "Automatic"
    assert any("Misra Chapu" in l and "3+2+2" in l for l in labels), \
        "the picker should show each cycle's shape, not only its name"
    assert any("Adi" in l and "4+2+2" in l for l in labels)

    # Choosing one records it on the brief, where the tune reads it too.
    index = next(i for i in range(panel.tala.count())
                 if panel.tala.itemData(i) == "Misra Chapu")
    panel.tala.setCurrentIndex(index)
    panel._tala_chosen(index)
    assert window.app.project.brief.tala == "Misra Chapu"
    assert window.app.current_tala().aksharas == 7


# --------------------------------------------------------------------------
# Choosing a raaga the brief did not suggest (Arya, 2026-09-06 20:16:42)
# --------------------------------------------------------------------------
def test_a_raaga_the_brief_did_not_suggest_can_still_be_heard(window):
    """Krish's case: the raaga he wants is not suggested, so pick it and listen.

    The panel read the suggestions list first and fell back to the box only
    when nothing was highlighted.  Applying a brief always highlights a row,
    so from then on the box was unreachable - choosing a raaga there and
    pressing "Hear the scale" played whichever suggestion was selected.
    """
    panel = window.raaga_panel
    heard = []
    panel.app.audition_raaga = lambda name, play=True: heard.append(name)

    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    suggested = [panel.suggestions.item(i).data(Qt.UserRole)
                 for i in range(panel.suggestions.count())]
    assert suggested, "the brief suggested nothing, so the case is untestable"

    # Whichever raaga the ranking left out.  Naming one outright makes the
    # test depend on how the brief happens to rank today - the first version
    # asserted Keeravani was absent, and it was suggested first.
    unsuggested = next(n for n in window.app.raagas.names()
                       if n not in suggested)

    panel.suggestions.setCurrentRow(0)          # as applying a brief leaves it
    panel.all_raagas.setCurrentText(unsuggested)
    panel.all_raagas.activated.emit(panel.all_raagas.currentIndex())

    panel.audition_selected()
    assert heard == [unsuggested], \
        f"heard {heard} - the highlighted suggestion won over the creator's pick"

    heard.clear()
    panel.preview_chosen()
    assert heard == [unsuggested]


def test_the_scale_preview_never_plays_a_suggestion(window):
    """`Play this scale` always means the box, whatever is highlighted."""
    panel = window.raaga_panel
    heard = []
    panel.app.audition_raaga = lambda name, play=True: heard.append(name)

    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.refresh()
    panel.suggestions.setCurrentRow(0)
    panel._chose("suggestions")                 # creator last touched the list
    panel.all_raagas.setCurrentText("Keeravani")

    panel.preview_chosen()
    assert heard == ["Keeravani"]


def test_clicking_a_suggestion_still_wins_afterwards(window):
    """The fix must not strand the suggestions list."""
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    assert panel.suggestions.count(), "nothing was suggested to click"

    panel.all_raagas.setCurrentText("Keeravani")
    panel.all_raagas.activated.emit(panel.all_raagas.currentIndex())
    assert panel._current_name() == "Keeravani"

    panel.suggestions.setCurrentRow(0)
    panel.suggestions.itemClicked.emit(panel.suggestions.item(0))
    assert panel._current_name() == panel.suggestions.item(0).data(Qt.UserRole)


def test_every_library_raaga_is_offered(window):
    """Ninety-three raagas, not just the ones a brief happened to rank."""
    panel = window.raaga_panel
    offered = {panel.all_raagas.itemText(i)
               for i in range(panel.all_raagas.count())}
    assert "Keeravani" in offered
    assert offered == set(window.app.raagas.names())


def test_the_reason_this_brief_gave_survives_into_the_details(window):
    """Arya's finding: the rationale existed only in the clipped list row.

    _render_suggestions stored just the name, and the details pane and the
    comparison look a raaga up in the catalogue - so they could say what a
    raaga is but never why it had been suggested.
    """
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    assert panel.suggestions.count(), "nothing was suggested"

    first = panel.suggestions.item(0)
    name = first.data(Qt.UserRole)
    reason = first.data(Qt.UserRole + 1)
    assert reason, "the reason was not kept beside the name"

    panel.suggestions.setCurrentRow(0)
    panel.suggestions.itemClicked.emit(first)
    shown = panel.details.toPlainText()
    assert f"Why this brief suggested {name}" in shown
    assert reason[:24] in shown, "the actual rationale is not shown"
    assert "not the catalogue" in shown, "its basis is not labelled"


def test_the_comparison_carries_the_reason_too(window):
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    name = panel.suggestions.item(0).data(Qt.UserRole)
    window.app.select_raaga(name, "for the test")

    other = next(n for n in window.app.raagas.names() if n != name)
    panel.all_raagas.setCurrentText(other)
    panel.all_raagas.activated.emit(panel.all_raagas.currentIndex())
    panel.compare_with_current()

    shown = panel.details.toPlainText()
    assert name in shown and other in shown
    assert f"Why this brief suggested {name}" in shown


def test_looking_at_a_raaga_does_not_re_rank(window):
    """Displaying a reason must not re-run the recommendation."""
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    calls = []
    original = window.app.raaga_suggestions
    window.app.raaga_suggestions = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        panel.suggestions.setCurrentRow(0)
        panel._show_named(panel.suggestions.item(0).data(Qt.UserRole))
        panel.compare_with_current()
        assert not calls, "looking at a raaga re-ran the ranking"
    finally:
        window.app.raaga_suggestions = original


def test_each_rationale_says_which_raaga_it_is_about(window):
    """Arya's finding: two explanations under one generic heading.

    Comparing two raagas the brief had both suggested printed "Why this
    brief suggested it" twice in a row, and nothing said which sentence
    belonged to which raaga.
    """
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    assert panel.suggestions.count() >= 2, "need two suggestions to compare"

    first = str(panel.suggestions.item(0).data(Qt.UserRole))
    second = str(panel.suggestions.item(1).data(Qt.UserRole))
    window.app.select_raaga(first, "for the test")
    panel.all_raagas.setCurrentText(second)
    panel.all_raagas.activated.emit(panel.all_raagas.currentIndex())
    panel.compare_with_current()

    shown = panel.details.toPlainText()
    assert f"Why this brief suggested {first}" in shown
    assert f"Why this brief suggested {second}" in shown
    assert "Why this brief suggested it" not in shown, "still unattributed"


def test_a_new_project_does_not_inherit_the_last_one_s_reasons(window):
    """Arya's finding: a prior context's recommendation shown as evidence.

    The rationale cache was keyed by raaga name and nothing cleared it, so
    after starting a fresh song the previous song's ranking still explained
    itself as "the current brief's ranking".
    """
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    name = str(panel.suggestions.item(0).data(Qt.UserRole))
    assert panel._recommendation_note(name), "nothing was cached to begin with"

    window.app.new_project("Second Song", write=False)
    window.refresh()

    assert panel._rationales == {}, "the previous ranking's reasons survived"
    assert panel.suggestions.count() == 0, "the previous ranking's rows survived"
    panel._show_named(name)
    assert "Why this brief suggested" not in panel.details.toPlainText()


def test_a_brief_that_moved_on_is_not_called_the_current_one(window):
    """Editing the brief without applying it does not re-rank.

    The reasons on screen are still why these raagas are listed, so they
    stay; but they answer the question as it was asked, not as it now
    stands, and the label has to say so.
    """
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    name = str(panel.suggestions.item(0).data(Qt.UserRole))
    assert "from the current brief's ranking" in panel._recommendation_note(name)

    window.app.project.brief.mood = "grieving, heavy"
    assert "from an earlier brief's ranking" in panel._recommendation_note(name)


def test_the_visible_comparison_is_relabelled_when_the_brief_moves_on(window):
    """Arya's finding: the helper answered, the screen did not.

    _recommendation_note started saying "an earlier brief" the moment the
    brief was edited, but the words already in the pane had been written
    before that and nothing rewrote them.  The label was only right if the
    creator happened to click something.
    """
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    assert panel.suggestions.count() >= 2, "need two suggestions to compare"
    first = str(panel.suggestions.item(0).data(Qt.UserRole))
    second = str(panel.suggestions.item(1).data(Qt.UserRole))
    window.app.select_raaga(first, "for the test")
    panel.all_raagas.setCurrentText(second)
    panel.all_raagas.activated.emit(panel.all_raagas.currentIndex())
    panel.compare_with_current()

    before = panel.details.toPlainText()
    assert "from the current brief's ranking" in before

    # The brief changes without being applied: no new ranking exists.
    window.app.update_brief(mood="grieving, heavy")
    window.refresh()

    shown = panel.details.toPlainText()
    assert "from an earlier brief's ranking" in shown, "the pane was not relabelled"
    assert "from the current brief's ranking" not in shown
    # The creator keeps the comparison they were reading.
    assert first in shown and second in shown
    assert f"Why this brief suggested {first}" in shown
    assert f"Why this brief suggested {second}" in shown


def test_relabelling_keeps_the_creator_s_place_in_the_text(window):
    """Rewriting the label must not scroll them back to the top."""
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    first = str(panel.suggestions.item(0).data(Qt.UserRole))
    second = str(panel.suggestions.item(1).data(Qt.UserRole))
    window.app.select_raaga(first, "for the test")
    panel.all_raagas.setCurrentText(second)
    panel.all_raagas.activated.emit(panel.all_raagas.currentIndex())
    panel.compare_with_current()

    bar = panel.details.verticalScrollBar()
    panel.details.document().setTextWidth(panel.details.viewport().width())
    if bar.maximum() <= 0:
        pytest.skip("the comparison fits without scrolling at this size")
    bar.setValue(bar.maximum())
    was = bar.value()

    window.app.update_brief(mood="grieving, heavy")
    window.refresh()

    assert "from an earlier brief's ranking" in panel.details.toPlainText()
    assert bar.value() == min(was, bar.maximum()), "scrolled away from their place"


def test_relabelling_does_not_re_rank(window):
    """A label correction is not a reason to recompute the recommendation."""
    panel = window.raaga_panel
    window.app.apply_brief_sync(mood="hopeful, romantic", feel="")
    panel.suggest()
    panel._show_named(str(panel.suggestions.item(0).data(Qt.UserRole)))

    calls = []
    original = window.app.raaga_suggestions
    window.app.raaga_suggestions = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        window.app.update_brief(mood="grieving, heavy")
        window.refresh()
        assert "from an earlier brief's ranking" in panel.details.toPlainText()
        assert not calls, "relabelling re-ran the ranking"
    finally:
        window.app.raaga_suggestions = original


def test_typing_in_the_conversation_box_does_not_act_inline(window):
    """The box used to call handle_utterance directly, so interpreting a
    typed phrase happened on the thread that had to redraw the window."""
    app = window.app
    panel = window.conversation
    before = len(app.project.conversation)

    panel.entry.setText("compose a tune")
    panel.entry.returnPressed.emit()

    assert panel.entry.text() == "", "the box did not clear"
    assert len(app.project.conversation) == before,         "the typed phrase was interpreted on the interface thread"
    assert app._typed_queue.qsize() == 1, "it was not queued either"


def test_the_chosen_cycle_and_its_reason_are_on_screen(window):
    """A choice made for the creator is shown as one.

    Every new tune was composed in Adi because Adi is what the lookup falls
    back to, and nothing said a decision had been made or offered another.
    """
    app = window.app
    # The window fixture is module-scoped, so an earlier test may have left
    # a tala on the brief.  This test is about the automatic choice, so it
    # says so rather than depending on the order it happens to run in.
    app.update_brief(tala="")
    app.apply_brief_sync(mood="tense, nervous", feel="",
                         situation="a chase through a city at night")
    window.refresh()

    note = window.tune.tala_note.text()
    assert "Khanda Chapu" in note, note
    assert "chase" in note, "the reason does not say what decided it"
    # Stated as a suggestion, because that is what it is: no lullaby has to
    # be in three.  It still owns the choice and says what decided it.
    assert note.startswith("I suggest"), note
    assert "Change it with the picker" in note, "no way offered to disagree"

    # And it is overridable: choosing one makes it the creator's.
    index = next(i for i in range(window.tune.tala.count())
                 if window.tune.tala.itemData(i) == "Rupaka")
    window.tune._tala_chosen(index)
    window.refresh()
    note = window.tune.tala_note.text()
    assert note.startswith("you chose"), note
    assert "Rupaka" in note
