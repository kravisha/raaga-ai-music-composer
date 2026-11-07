"""The song team as the producer meets it: through the controller, as real
audio, and in the Song Team tab (docs/SONG_TEAM.md).

The rough cut is heard back by the agent's own ears (agent/analysis.py), so
"the alapana is in the raaga" is checked on the sound and not only on the
plan that made it.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from raagacomposer.agent import analysis
from raagacomposer.audio import dsp
from raagacomposer.core.models import (
    MelodyVersion,
    Section,
    SectionKind,
    VocalDirection,
)
from raagacomposer.core.settings import config_dir
from raagacomposer.providers.local import LocalVoiceProvider
from raagacomposer.studio import SongTeam, rough_cut
from raagacomposer.voice.profiles import BUILTIN

pytestmark = pytest.mark.integration

SR = 22050
LONELY = "lonely, late at night, but still warm"


@pytest.fixture(scope="module")
def draft():
    return SongTeam().run(LONELY, "Tamil", seed=11)


@pytest.fixture(scope="module")
def cut(draft):
    return rough_cut.render(draft, SR, seed=3)


def _collapse(seq):
    out = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


# --------------------------------------------------------------------------
# the sound
# --------------------------------------------------------------------------
def test_the_rough_cut_is_alapana_then_two_lines_with_every_layer(draft, cut):
    audio, layers = cut
    assert audio.ndim == 2 and audio.shape[1] == 2
    expected = (draft.tune.alapana_seconds + rough_cut.BREATH
                + draft.tune.melody.duration + rough_cut.TAIL)
    assert abs(len(audio) / SR - expected) < 0.5
    assert len(layers) == 4
    assert any("alapana" in l for l in layers)
    assert any("mridangam" in l for l in layers)
    assert any("tanpura" in l for l in layers)
    assert dsp.peak_db(audio) <= -0.5


def test_both_halves_of_the_rough_cut_are_heard(draft, cut):
    audio, _ = cut
    split = int((draft.tune.alapana_seconds + rough_cut.BREATH) * SR)
    assert dsp.rms_db(audio[:split]) > -35.0, "the alapana is silent"
    assert dsp.rms_db(audio[split:]) > -35.0, "the two lines are silent"


def test_the_sung_alapana_is_heard_back_as_the_raaga_it_was_written_in(
        draft, raagas):
    tune = draft.tune
    raaga = raagas.require(tune.raaga)
    alapana = MelodyVersion(
        raaga=tune.raaga, tonic_midi=tune.tonic_midi, notes=list(tune.alapana),
        sections=[Section(id="alapana", kind=SectionKind.PALLAVI, start=0.0,
                          end=tune.alapana_seconds)])
    sung = LocalVoiceProvider().render_vocal(
        alapana, None, BUILTIN[0], VocalDirection(), SR,
        tune.alapana_seconds + 0.5, vocal_sections_only=False, vowel="a")
    heard = analysis.analyse(dsp.as_mono(sung), SR, raaga,
                             fixed_tonic_midi=float(tune.tonic_midi))
    written = _collapse([n.swara for n in tune.alapana])
    assert _collapse(heard.swara_sequence()) == written


# --------------------------------------------------------------------------
# through the controller
# --------------------------------------------------------------------------
def test_a_draft_is_written_beside_its_cards_and_leaves_the_song_alone(app):
    before = (len(app.project.melodies), len(app.project.lyrics),
              len(app.project.beats), app.project.raaga.selected, app.dirty)
    draft = app.make_song_draft_sync(LONELY, "Tamil", seed=11)

    assert draft.rough is not None
    wav = Path(draft.rough.path)
    assert wav.exists() and wav.parent == config_dir() / "song_team"
    card = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
    assert card["tune"]["raaga"] == draft.tune.raaga
    assert card["rough"]["layers"] == draft.rough.layers
    assert card["rough"]["missing"]
    assert app.song_draft is draft
    assert app.rendered("song_draft") is not None
    # A draft is something to listen to, not an edit to the open song.
    after = (len(app.project.melodies), len(app.project.lyrics),
             len(app.project.beats), app.project.raaga.selected, app.dirty)
    assert after == before


def test_the_producers_direction_reaches_the_agents(app):
    draft = app.make_song_draft_sync(LONELY, "Tamil", raaga="Mohanam",
                                     tala="Rupaka", seed=4)
    assert draft.tune.raaga == "Mohanam" and draft.tune.chosen_by == "producer"
    assert draft.beat.tala == "Rupaka" and draft.beat.chosen_by == "producer"


def test_the_background_path_reports_back_through_pump(app, settle):
    got = []
    app.on_song_draft = got.append
    assert app.make_song_draft(LONELY, "English", seed=8)
    settle(120)
    assert got and got[-1] is app.song_draft
    assert got[-1].mood.language == "English"


def test_an_empty_mood_starts_nothing(app):
    assert app.make_song_draft("   ") is False
    assert "mood" in app.status_text.lower()


def test_a_failed_run_hands_the_panel_its_button_back(app, settle,
                                                      monkeypatch):
    got = []
    app.on_song_draft = got.append

    def broken(*args, **kwargs):
        raise RuntimeError("the tunesmith is unwell")

    monkeypatch.setattr(app, "song_team", broken)
    assert app.make_song_draft(LONELY, seed=1)
    settle(60)
    assert got == [None]


# --------------------------------------------------------------------------
# the Song Team tab
# --------------------------------------------------------------------------
@pytest.mark.ui
@pytest.mark.slow
def test_the_song_team_tab_from_mood_to_a_playable_rough_song(app):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from raagacomposer.ui.panels.song_team_panel import (
        AUTO_RAAGA,
        AUTO_TALA,
        SongTeamPanel,
    )

    qt = QApplication.instance() or QApplication([])
    panel = SongTeamPanel(app)
    panel.show()
    assert panel.raaga_box.currentText() == AUTO_RAAGA
    assert panel.tala_box.currentText() == AUTO_TALA
    assert not panel.play_btn.isEnabled()
    assert "Prelude" in panel.todo_text.toPlainText()

    panel.mood_edit.setText("a lullaby for my daughter, gentle and sleepy")
    panel.make_btn.click()
    assert not panel.make_btn.isEnabled()
    deadline = time.time() + 120
    while time.time() < deadline and not panel.play_btn.isEnabled():
        app.pump()
        qt.processEvents()
        time.sleep(0.02)

    assert panel.play_btn.isEnabled() and panel.make_btn.isEnabled()
    assert "Tisra Eka" in panel.beat_text.toPlainText()
    tune = panel.tune_text.toPlainText()
    assert app.song_draft.tune.raaga in tune and "Alapana" in tune
    assert "As sung" in panel.lyrics_text.toPlainText()
    assert "Mood listener" in panel.log_text.toPlainText()
    assert "Rough song" in panel.status_label.text()
    panel.close()
