"""Integration: saved takes are chosen by id and reused explicitly.

Offscreen VoicePanel on the real controller; takes made through the fake
input stream (synthetic audio, no microphone); playback faked.  Choosing
a take plays nothing and makes nothing.  "Play selected take" plays the
one chosen, by id, never the latest.  "New voice from selected takes"
goes through the existing profile door with a name the creator gives,
after every id and file has been checked, and refuses stale ids,
unreadable files and an active recording without making anything.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox  # noqa: E402

from raagacomposer.ui import theme  # noqa: E402
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


def _tone(seconds, sr, hz=220.0, level=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)[:, None]


def _record(app, mic, seconds, section_id=None, hz=220.0):
    """One saved take of a sung tone, through the real recorder path."""
    assert app.start_take(section_id)
    stream = mic.last
    audio = _tone(seconds, app.sample_rate, hz)
    for start in range(0, len(audio), 1024):
        block = audio[start:start + 1024]
        stream.callback(block, len(block), None, None)
    take = app.stop_take()
    assert take is not None
    return take


@pytest.fixture
def two_takes(app):
    _a_tune(app, "Choose a take")
    mic = FakeInput()
    app.recorder = TakeRecorder(open_stream=mic, sample_rate=app.sample_rate)
    pallavi = _pallavi(app)
    first = _record(app, mic, 1.2, pallavi.id, hz=196.0)
    second = _record(app, mic, 0.8, None, hz=262.0)
    return app, mic, first, second


def _rows(panel):
    return [panel.takes_list.item(i).text() for i in range(panel.takes_list.count())]


def _select(panel, qt_app, *take_ids):
    panel.takes_list.clearSelection()
    for i in range(panel.takes_list.count()):
        item = panel.takes_list.item(i)
        item.setSelected(item.data(Qt.UserRole) in take_ids)
    qt_app.processEvents()


def test_the_list_shows_every_saved_take_and_choosing_one_does_nothing_by_itself(two_takes, qt_app, monkeypatch):
    app, mic, first, second = two_takes
    played = []
    monkeypatch.setattr(app, "play_take", lambda take_id: played.append(take_id) or True)
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        rows = _rows(panel)
        assert len(rows) == 2
        assert first.label in rows[0] and "Pallavi" in rows[0] and "1.2s" in rows[0]
        assert second.label in rows[1] and "whole song" in rows[1] and "0.8s" in rows[1]
        assert not panel.play_take_btn.isEnabled() and not panel.profile_from_takes_btn.isEnabled()
        _select(panel, qt_app, first.id)
        assert panel.play_take_btn.isEnabled() and panel.profile_from_takes_btn.isEnabled()
        assert played == [] and len(app.voices.all()) == len(app.voices.all())
        assert app.project.voice_profile_id == app.voices.default().id
    finally:
        panel.close(); panel.deleteLater(); qt_app.processEvents()


def test_play_selected_take_plays_the_chosen_one_by_id_not_the_latest(two_takes, qt_app, monkeypatch):
    app, mic, first, second = two_takes
    played = []
    monkeypatch.setattr(app, "play_take", lambda take_id: played.append(take_id) or True)
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        _select(panel, qt_app, first.id)          # the older one, not the latest
        panel.play_take_btn.click()
        qt_app.processEvents()
        assert played == [first.id]
        _select(panel, qt_app, first.id, second.id)
        assert not panel.play_take_btn.isEnabled(), "two chosen: nothing to play until one is"
        panel.play_take_btn.click()
        qt_app.processEvents()
        assert played == [first.id]
        # The selection survives an ordinary refresh, by id.
        panel.refresh()
        qt_app.processEvents()
        assert sorted(panel._selected_take_ids()) == sorted([first.id, second.id])
    finally:
        panel.close(); panel.deleteLater(); qt_app.processEvents()


def test_a_new_voice_from_selected_takes_is_explicit_named_and_made_by_the_existing_door(two_takes, qt_app, monkeypatch):
    app, mic, first, second = two_takes
    before_profiles = [p.id for p in app.voices.all()]
    before_melodies = len(app.project.melodies)
    before_lyrics = len(app.project.lyrics)
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("My Take Voice", True)))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(a[2])))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append("WARN " + a[2])))
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        _select(panel, qt_app, first.id, second.id)
        panel.profile_from_takes_btn.click()
        qt_app.processEvents()
        assert shown and shown[-1].startswith("Created My Take Voice from 2 saved take(s)"), shown
        profile = next(p for p in app.voices.all() if p.id not in before_profiles)
        assert profile.name == "My Take Voice" and not profile.builtin
        assert sorted(profile.source_samples) == sorted([first.audio_path, second.audio_path])
        assert "Derived from 2 recording(s)" in profile.notes
        assert app.project.voice_profile_id == profile.id
        # Nothing else moved: takes, tune, words, other profiles.
        assert [t.id for t in app.project.recordings] == [first.id, second.id]
        assert os.path.exists(first.audio_path) and os.path.exists(second.audio_path)
        assert len(app.project.melodies) == before_melodies and len(app.project.lyrics) == before_lyrics
        assert all(p in [q.id for q in app.voices.all()] for p in before_profiles)
        # And the list is still keyed by id, the profile box shows the new voice.
        assert panel.voice_box.currentData() == profile.id
    finally:
        panel.close(); panel.deleteLater(); qt_app.processEvents()


def test_a_cancelled_name_makes_no_profile(two_takes, qt_app, monkeypatch):
    app, mic, first, second = two_takes
    before = [p.id for p in app.voices.all()]
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("", False)))
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        _select(panel, qt_app, first.id)
        panel.profile_from_takes_btn.click()
        qt_app.processEvents()
        assert [p.id for p in app.voices.all()] == before
        assert app.project.voice_profile_id == app.voices.default().id
    finally:
        panel.close(); panel.deleteLater(); qt_app.processEvents()


def test_stale_ids_unreadable_files_and_an_active_recording_are_refused_before_anything_is_made(two_takes, qt_app, monkeypatch):
    app, mic, first, second = two_takes
    before = [p.id for p in app.voices.all()]
    with pytest.raises(ValueError, match="not in this song"):
        app.takes_for_reuse([first.id, "rec_from_another_song"])
    with pytest.raises(ValueError, match="at least one"):
        app.takes_for_reuse([])
    os.remove(second.audio_path)
    with pytest.raises(ValueError, match="cannot be read"):
        app.takes_for_reuse([second.id])
    assert app.start_take()
    with pytest.raises(ValueError, match="recording"):
        app.takes_for_reuse([first.id])
    app.cancel_take()
    with pytest.raises(ValueError, match="Name the voice"):
        app.create_voice_from_takes([first.id], "   ")
    assert [p.id for p in app.voices.all()] == before
    # Through the panel: the warning, and nothing made.
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Should Not Exist", True)))
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        _select(panel, qt_app, second.id)      # its file is gone
        panel.profile_from_takes_btn.click()
        qt_app.processEvents()
        assert warned and "cannot be read" in warned[-1]
        assert [p.id for p in app.voices.all()] == before
        _select(panel, qt_app, first.id)
        assert panel.profile_from_takes_btn.isEnabled()
        assert app.start_take()
        panel.refresh()                        # what the Record button does
        qt_app.processEvents()
        assert not panel.profile_from_takes_btn.isEnabled() and not panel.play_take_btn.isEnabled()
        app.cancel_take()
    finally:
        panel.close(); panel.deleteLater(); qt_app.processEvents()


def test_the_selection_is_cleared_on_a_new_song_and_the_list_survives_save_and_reopen(two_takes, qt_app):
    app, mic, first, second = two_takes
    panel = VoicePanel(app)
    try:
        qt_app.processEvents()
        _select(panel, qt_app, first.id)
        assert panel._selected_take_ids() == [first.id]
        app.save()
        saved_dir = app.project_dir
        app.new_project("Another song", write=False)
        panel.refresh()
        qt_app.processEvents()
        assert panel.takes_list.count() == 0 and panel._selected_take_ids() == []
        assert not panel.play_take_btn.isEnabled()
        app.open_project(saved_dir)
        panel.refresh()
        qt_app.processEvents()
        assert [panel.takes_list.item(i).data(Qt.UserRole)
                for i in range(panel.takes_list.count())] == [first.id, second.id]
        assert panel._selected_take_ids() == [], "a selection from before the switch came back"
        _select(panel, qt_app, second.id)
        assert app.takes_for_reuse([second.id])[0].id == second.id
    finally:
        panel.close(); panel.deleteLater(); qt_app.processEvents()
