"""Integration: controller-level workflow, jobs, undo and error handling."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from raagacomposer.core.models import Stage

pytestmark = pytest.mark.integration


@pytest.fixture
def ready(app, settle, brief):
    """A project with a brief, a raaga and a short generated tune."""
    app.new_project("Flow Test")
    brief.duration_target = 45.0
    app.update_brief(**{f: getattr(brief, f) for f in
                        ("situation", "mood", "feel", "language",
                         "duration_target")})
    app.select_raaga("Keeravani")
    app.generate_tune(seed=3)
    settle()
    return app


# --------------------------------------------------------------------------
# stage progression
# --------------------------------------------------------------------------
def test_a_new_project_starts_at_the_brief(app):
    assert app.project.current_stage is Stage.BRIEF
    assert app.project_dir is None or app.project_dir.exists()
    assert app.project.voice_profile_id


def test_the_workflow_advances_stage_by_stage(ready, settle):
    app = ready
    assert app.project.current_stage is Stage.TUNE
    app.accept_tune(lock=True)
    assert app.project.current_stage is Stage.LYRICS
    app.generate_lyrics(seed=2)
    settle()
    assert app.project.current_stage is Stage.VOICE
    app.render_vocal("preview", autoplay=False)
    settle()
    assert app.project.current_stage is Stage.ARRANGEMENT
    app.render("full", autoplay=False)
    settle()
    assert app.project.current_stage is Stage.MIX


def test_the_brief_drives_the_raaga_and_the_tempo(app, settle):
    app.new_project("Brief Driven")
    app.update_brief(mood="celebration", feel="festive and bright",
                     duration_target=40.0, tempo_preference=120)
    suggestions = app.raaga_suggestions()
    assert suggestions
    app.select_raaga(suggestions[0].name)
    app.generate_tune(seed=1)
    settle()
    assert app.project.melody().tempo_bpm == 120


def test_a_raaga_is_chosen_automatically_when_the_creator_does_not(app, settle):
    app.new_project("Auto Raaga")
    app.update_brief(mood="devotional", duration_target=40.0)
    app.generate_tune(seed=1)
    settle()
    assert app.project.raaga.selected
    assert app.project.melody().raaga == app.project.raaga.selected


def test_a_locked_raaga_cannot_be_changed(ready):
    from raagacomposer.core.versioning import LockedContentError
    ready.set_raaga_lock(True)
    with pytest.raises(LockedContentError):
        ready.select_raaga("Kalyani")


# --------------------------------------------------------------------------
# tune versions
# --------------------------------------------------------------------------
def test_versions_accumulate_and_can_be_reselected(ready, settle):
    app = ready
    first = app.project.melody().version
    app.make_variation()
    settle()
    second = app.project.melody().version
    assert second > first
    assert len(app.project.melodies) == 2

    app.select_melody_version(first)
    settle()
    assert app.project.melody().version == first


def test_regenerating_a_section_keeps_the_others(ready, settle):
    app = ready
    melody = app.project.melody()
    target = melody.sections[1]
    before = [(n.start, n.midi) for n in melody.notes
              if n.section_id != target.id]
    app.regenerate_tune_section(target.id)
    settle()
    after = [(n.start, n.midi) for n in app.project.melody().notes
             if n.section_id != target.id]
    assert before == after


def test_a_locked_section_is_protected_at_the_controller(ready):
    from raagacomposer.core.versioning import LockedContentError
    app = ready
    section = app.project.melody().sections[1]
    app.set_section_lock(section.id, True)
    with pytest.raises(LockedContentError):
        app.regenerate_tune_section(section.id)


def test_tempo_change_produces_a_new_version(ready, settle):
    app = ready
    original = app.project.melody()
    app.set_tempo(original.tempo_bpm + 20)
    settle()
    assert app.project.melody().tempo_bpm == original.tempo_bpm + 20
    assert app.project.melody().version > original.version


def test_the_validation_report_is_available(ready):
    report = ready.validation_report()
    assert "fidelity" in report


# --------------------------------------------------------------------------
# lyrics and voice
# --------------------------------------------------------------------------
def test_lyrics_need_a_tune_first(app):
    app.new_project("No Tune")
    app.generate_lyrics()
    assert app.project.lyrics_version() is None
    assert "tune first" in app.status_text


def test_editing_a_line_refits_it(ready, settle):
    app = ready
    app.generate_lyrics(seed=2)
    settle()
    lyrics = app.project.lyrics_version()
    line = lyrics.lines[0]
    app.edit_lyric_line(line.id, "puthiya vaanam ondru")
    assert lyrics.lines[0].text == "puthiya vaanam ondru"
    assert len(lyrics.lines[0].syllables) == len(lyrics.lines[0].note_indices)


def test_changing_singer_does_not_change_the_tune(ready, settle):
    app = ready
    app.generate_lyrics(seed=2)
    settle()
    before = [(n.start, n.midi) for n in app.project.melody().notes]
    male = next(v for v in app.voices.all() if v.gender == "male")
    app.set_voice(male.id)
    app.render_vocal("preview", autoplay=False)
    settle()
    assert [(n.start, n.midi) for n in app.project.melody().notes] == before
    assert app.project.latest_vocal.voice_profile_id == male.id


def test_the_vocal_only_master_is_written_and_marked(ready, settle):
    app = ready
    app.render_vocal("master", autoplay=False)
    settle()
    master = app.project.vocal_master
    assert master is not None and master.kind == "master"
    assert Path(master.audio_path).exists()
    assert app.project.vocal_master_id == master.id
    assert app.rendered("vocal_master") is not None


# --------------------------------------------------------------------------
# arrangement and mix
# --------------------------------------------------------------------------
def test_add_replace_and_remove_through_the_controller(ready, settle):
    app = ready
    duration = app.project.duration
    app.add_instrument("violin", 0.0, duration, role="lead")
    settle()
    assert app.project.arrangement().tracks_for_instrument("violin")

    app.replace_instrument("violin", "veena")
    settle()
    assert not app.project.arrangement().tracks_for_instrument("violin")
    assert app.project.arrangement().tracks_for_instrument("veena")

    app.remove_instrument("veena")
    settle()
    assert not app.project.arrangement().tracks_for_instrument("veena")


def test_an_unavailable_instrument_is_reported_with_alternatives(ready):
    app = ready
    before = len(app.project.errors)
    app.add_instrument("theremin", 0.0, 20.0)
    assert len(app.project.errors) == before + 1
    message = app.project.errors[-1].message
    assert "theremin" in message
    assert "Closest available" in message


def test_track_flags_and_levels(ready, settle):
    app = ready
    app.add_instrument("veena", 0.0, app.project.duration)
    settle()
    track = app.project.arrangement().tracks[0]
    app.set_track_flag(track.id, mute=True)
    settle()
    assert track.mute
    app.change_level("veena", 1.5)
    settle()
    assert track.gain > 1.0


def test_auto_arrange_then_full_mix(ready, settle):
    app = ready
    app.auto_arrange()
    settle()
    assert len(app.project.arrangement().tracks) >= 4
    app.render("full", autoplay=False)
    settle()
    mix = app.project.latest_mix("full")
    assert mix and Path(mix.audio_path).exists()
    assert app.rendered("full") is not None


def test_instrumental_and_full_are_separate_products(ready, settle):
    app = ready
    app.auto_arrange()
    settle()
    app.render("instrumental", autoplay=False)
    settle()
    app.render("full", autoplay=False)
    settle()
    assert app.project.latest_mix("instrumental") is not None
    assert app.project.latest_mix("full") is not None
    assert app.rendered("instrumental").path != app.rendered("full").path


def test_feel_based_suggestions_come_back_ranked(ready):
    ranked = ready.suggest_instruments(["lonely", "night", "warm"])
    assert ranked
    assert all(hasattr(inst, "key") for inst, _ in ranked)


# --------------------------------------------------------------------------
# jobs, undo, autosave, diagnostics
# --------------------------------------------------------------------------
def test_long_work_runs_off_the_calling_thread(ready):
    app = ready
    app.render("full", autoplay=False)
    assert app.jobs.active_jobs() or app.rendered("full") is not None
    app.jobs.cancel_all("test")
    for _ in range(200):
        app.pump()
        if not app.jobs.active_jobs():
            break
        time.sleep(0.02)
    assert not app.jobs.active_jobs()


def test_a_barge_in_pauses_playback_and_cancels_work(ready):
    app = ready
    app.render("full", autoplay=False)
    app._on_barge_in()
    for _ in range(200):
        app.pump()
        if not app.jobs.active_jobs():
            break
        time.sleep(0.02)
    assert not app.jobs.active_jobs()


def test_undo_and_redo_walk_the_project(ready, settle):
    app = ready
    app.add_instrument("veena", 0.0, app.project.duration)
    settle()
    assert app.project.arrangement().tracks_for_instrument("veena")

    assert app.undo_action()
    arrangement = app.project.arrangement()
    assert arrangement is None or not arrangement.tracks_for_instrument("veena")

    assert app.redo_action()
    assert app.project.arrangement().tracks_for_instrument("veena")


def test_autosave_writes_after_the_interval(ready):
    app = ready
    app.settings.autosave_seconds = 0
    app.dirty = True
    app._last_autosave = 0.0
    app.maybe_autosave()
    assert not app.dirty
    assert (app.project_dir / "project.json").exists()


def test_history_and_conversation_are_recorded(ready):
    app = ready
    app.handle_utterance("Play the first minute.")
    assert app.project.history
    assert app.project.conversation
    assert app.project.conversation[-1].intent == "transport.play"


def test_diagnostics_export_bundles_the_logs(ready, tmp_path):
    out = ready.export_diagnostics(tmp_path / "diag.zip")
    import zipfile
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "environment.json" in names
    assert any(n.startswith("logs/") or n == "session.log" for n in names)


def test_the_summary_describes_the_project(ready):
    text = ready.summary()
    assert "Project:" in text and "Raaga:" in text and "Tune:" in text


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------
def test_exports_write_real_files(ready, settle, tmp_path):
    app = ready
    app.generate_lyrics(seed=2)
    settle()
    app.auto_arrange()
    settle()
    app.render("full", autoplay=False)
    settle()

    assert app.export(tmp_path / "mix.wav", "full").exists()
    assert app.export_midi(tmp_path / "tune.mid").exists()
    assert app.export_musicxml(tmp_path / "tune.musicxml").exists()
    assert app.export_lyrics(tmp_path / "lyrics.txt").exists()
    assert app.export_stems(tmp_path / "stems")
    assert app.archive(tmp_path / "project.zip").exists()


def test_exporting_something_unrendered_is_reported_not_crashed(ready, tmp_path):
    assert ready.export(tmp_path / "nothing.wav", "instrumental") is None
    assert "Render" in ready.status_text


# --------------------------------------------------------------------------
# the brief chooses the instrument, and Save As names the song
# --------------------------------------------------------------------------
def test_the_brief_decides_which_instrument_the_tune_is_heard_on(app):
    """The brief's "Prefer" field was written and never read.

    The tune and the audition were rendered on a hardcoded veena whatever
    the creator asked for, which is why a violinist kept hearing a veena.
    """
    assert app.tune_instrument().name.lower() == "veena"   # the old default
    app.update_brief(instruments_preferred=["violin"])
    assert app.tune_instrument().name.lower() == "violin"


def test_a_percussion_preference_does_not_take_over_the_melody(app):
    """"Prefer mridangam" is about the arrangement, not the lead line."""
    app.update_brief(instruments_preferred=["mridangam"])
    assert "lead" in app.tune_instrument().roles


def test_save_as_is_how_a_song_is_renamed(app, tmp_path):
    """There is no name field on screen; the folder you choose is the name."""
    app.new_project("Untitled Song")
    app.save_as(tmp_path / "Kaadhal Tholvi")
    assert app.project.title == "Kaadhal Tholvi"
    assert app.project.brief.title == "Kaadhal Tholvi"
    assert not app.dirty

    # and it survives the round trip to disk
    reopened = app.store.open(app.project_dir)
    assert reopened.title == "Kaadhal Tholvi"
