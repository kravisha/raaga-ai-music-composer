"""Integration: controller-level workflow, jobs, undo and error handling."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
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


def test_the_audition_and_the_arrangement_cast_the_same_lead(ready, settle):
    """No drift: one brief, one lead, whichever path asks.

    These decided separately - the audition fell back to the veena and
    ignored the feel of the brief, the arrangement fell back to the flute
    and weighed it - so the same brief could be auditioned on one
    instrument and arranged around another.
    """
    from raagacomposer.music import arrangement as arranger

    app = ready
    for preferred in ([], ["violin"], ["mridangam"], ["flute"]):
        app.update_brief(instruments_preferred=preferred)
        auditioned = app.cast_lead()

        # A fresh arrangement, so earlier versions' tracks are not carried
        # forward and confused with this one's casting.
        built = arranger.auto_arrange(
            app.project.melody(), app.require_raaga(), app.project.brief,
            previous=None, lead=auditioned.instrument)
        leads = {t.instrument for t in built.tracks if t.role == "lead"}
        assert leads == {auditioned.instrument.key}, (
            f"preferred={preferred}: auditioned "
            f"{auditioned.instrument.key!r} but arranged {sorted(leads)}")

    # And the controller hands that same casting down rather than letting
    # the arranger decide again from less information.
    app.update_brief(instruments_preferred=["violin"])
    app.auto_arrange()
    settle()
    fresh = app.project.arrangement()
    assert any(t.instrument == "violin" and t.role == "lead"
               for t in fresh.tracks), \
        "the controller's casting did not reach the arrangement"


def test_the_choice_of_lead_can_explain_itself(app):
    """Specification 13: show which instrument, and why it was chosen."""
    app.update_brief(instruments_preferred=["violin"])
    chosen = app.cast_lead()
    assert chosen.instrument.name.lower() == "violin"
    assert chosen.chosen_by_the_creator
    assert "asked for" in chosen.reason

    app.update_brief(instruments_preferred=[])
    fallback = app.cast_lead()
    assert fallback.reason and not fallback.chosen_by_the_creator


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


# --------------------------------------------------------------------------
# the tune is hummed, not played
# --------------------------------------------------------------------------
def test_the_tune_is_hummed_rather_than_played_on_an_instrument(ready, settle):
    """Specification 10.1-10.3: judge the line before the instrument.

    Previewing a tune on an instrument asked the creator to judge two
    things at once, and every instrument here is additive synthesis - a
    "violin" is seven harmonics and an envelope, which is why it sounded
    like a keyboard.  A singer with no words sings on "aa".
    """
    app = ready
    app.render("tune", autoplay=False)
    settle()
    rendered = app.rendered("tune")
    assert rendered is not None, "no tune render"
    assert len(rendered.audio), "the hum is empty"


def test_the_hum_covers_the_instrumental_sections_too(ready, settle):
    """A sung take skips the interlude; hearing the tune must not.

    ``plan_segments`` drops instrumental sections by default, which left
    20 of 53 notes silent - holes exactly where the prelude, interlude and
    outro are.
    """
    from raagacomposer.voice import renderer

    melody = ready.project.melody()
    instrumental = [s for s in melody.sections if s.kind.instrumental]
    assert instrumental, "this tune has no instrumental section to check"

    sung_only = renderer.plan_segments(melody, None)
    everything = renderer.plan_segments(melody, None, vocal_sections_only=False)
    assert len(everything) > len(sung_only), \
        "the fixture no longer exercises the case this guards"
    assert len(everything) == len(melody.notes)

    ready.render("tune", autoplay=False)
    settle()
    audio = ready.rendered("tune").audio
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio
    sr = ready.sample_rate
    for section in instrumental:
        start, end = int(section.start * sr), int(min(section.end, melody.duration) * sr)
        block = mono[start:end]
        if len(block) < sr // 4:
            continue
        level = float(np.sqrt(np.mean(block ** 2)))
        assert level > 0.005, \
            f"{section.kind.name} is silent in the hum (rms {level:.4f})"


# --------------------------------------------------------------------------
# playing back the render you just made, and not the one before it
# --------------------------------------------------------------------------
def _loaded(app):
    """The audio actually sitting in the playback engine."""
    return app.playback._buffer.copy()


def test_auditioning_a_second_raaga_plays_the_second_raaga(app):
    """Reported from the app: pick Hamsadhwani, hear it; pick Mohanam, hear
    Hamsadhwani again.

    ``play_render`` skipped loading when the engine's ``source_name`` already
    matched - but that name is the *kind*, and a second audition is still
    called "audition".  Two different scales, one name, and the first one
    kept playing.
    """
    first = app.audition_raaga("Hamsadhwani", play=True)
    assert first is not None
    hamsadhwani = _loaded(app)
    assert len(hamsadhwani), "nothing was loaded for the first audition"

    second = app.audition_raaga("Mohanam", play=True)
    assert second is not None
    mohanam = _loaded(app)

    assert mohanam.shape != hamsadhwani.shape or \
        not np.array_equal(mohanam, hamsadhwani), \
        "the second audition replayed the first raaga's scale"


def test_replaying_the_same_audition_does_not_reload_it(app):
    """The guard being fixed is worth keeping: ``load`` stops and rewinds,
    so replaying what is already loaded must not restart it."""
    app.audition_raaga("Hamsadhwani", play=True)
    loaded = app._loaded_render
    app.play_render("audition")
    assert app._loaded_render is loaded, "the same render was reloaded"


def test_a_re_rendered_tune_is_the_one_you_hear(ready, settle):
    """The same fault, on the path that matters more than the audition.

    This used to ask for the instrumental mix, which is built from the
    arrangement - and this fixture has none, so both renders were silence.
    It passed only because every render used to come out half a second
    longer than the last, so two silences compared unequal by their
    length.  Fixing that growth made it fail, which is how the accident
    came to light.

    It renders the tune now, which is the thing that actually follows the
    melody, and compares the sound rather than the shape.
    """
    app = ready
    app.render("tune", autoplay=True)
    settle()
    first = _loaded(app)
    assert len(first) and np.abs(first).max() > 0, "the tune rendered silent"

    app.generate_tune(seed=99)
    settle()
    app.render("tune", autoplay=True)
    settle()
    second = _loaded(app)

    assert not np.array_equal(second, first), \
        "the re-rendered tune played the previous version"


def test_the_arrangement_is_not_silently_rebuilt_by_a_new_tune(ready, settle):
    """The other half of what that test was accidentally covering.

    An arrangement is the creator's work.  Composing a new tune does not
    regenerate it, so the instrumental mix still plays what was arranged -
    which is right, and is why the instrumental render could never have
    been the thing that test was checking.
    """
    app = ready
    app.auto_arrange()
    settle()
    before = [(t.instrument, len(t.regions)) for t in
              app.project.arrangement().tracks]

    app.generate_tune(seed=101)
    settle()

    after = [(t.instrument, len(t.regions)) for t in
             app.project.arrangement().tracks]
    assert after == before, "composing a tune quietly rebuilt the arrangement"


# --------------------------------------------------------------------------
# the beat, as its own layer
# --------------------------------------------------------------------------
def test_a_beat_can_be_made_and_varied_without_touching_the_tune(ready, settle):
    """Specification 11.9: regenerate the beat without changing the melody.

    The beat is written against the tala rather than the melody's notes,
    so the two cannot disturb each other.
    """
    app = ready
    before = [(n.start, n.midi, n.velocity) for n in app.project.melody().notes]

    app.generate_beat()
    settle()
    first = app.project.beat()
    assert first is not None and first.notes, "no beat was made"
    assert first.tala, "the beat does not know its tala"
    assert app.rendered("beat") is not None, "the beat was never sounded"

    app.beat_variation("moderate")
    settle()
    second = app.project.beat()
    assert second.version == first.version + 1
    assert second.tala == first.tala, "a variation must not change the tala"
    assert len(app.project.beats) == 2, "earlier takes must be kept"

    after = [(n.start, n.midi, n.velocity) for n in app.project.melody().notes]
    assert after == before, "making a beat changed the tune"


def test_the_beat_and_the_tune_share_a_tempo(ready, settle):
    app = ready
    app.generate_beat()
    settle()
    assert app.project.beat().tempo_bpm == app.project.melody().tempo_bpm


def test_a_south_indian_song_keeps_its_mridangam(ready):
    """Ranking on feel alone put a tambourine under a Carnatic tune,
    because "celebration" scores well on one."""
    app = ready
    app.update_brief(mood="celebration", language="Tamil")
    assert app.beat_instrument().key == "mridangam"

    app.update_brief(instruments_preferred=["tabla"])
    assert app.beat_instrument().key == "tabla", "an explicit request must win"


def test_the_arrangement_uses_the_beat_you_made(ready, settle):
    """Not a second one generated behind your back (specification 11.5).

    ``auto_arrange`` wrote its own rhythm part, so a creator who made and
    approved a beat got a different one in the mix.
    """
    app = ready
    app.update_brief(language="Tamil")
    app.generate_beat("busy")
    settle()
    beat = app.project.beat()

    app.auto_arrange()
    settle()
    rhythm = [t for t in app.project.arrangement().tracks if t.role == "rhythm"]
    assert rhythm, "no rhythm track in the arrangement"
    assert all(t.created_by == "beat" for t in rhythm), \
        "the arrangement generated its own percussion instead"
    meta = [r.meta for t in rhythm for r in t.regions]
    assert any(m.get("beat_version") == str(beat.version) for m in meta), \
        "the arranged rhythm is not the beat that was approved"


def test_without_a_beat_the_arrangement_still_plays_percussion(ready, settle):
    """The old behaviour is intact for a song nobody made a beat for."""
    app = ready
    assert app.project.beat() is None
    app.auto_arrange()
    settle()
    rhythm = [t for t in app.project.arrangement().tracks if t.role == "rhythm"]
    assert rhythm, "a song with no beat lost its percussion entirely"
    assert all(t.created_by == "auto" for t in rhythm)


def test_choosing_a_tala_moves_the_tune_and_the_beat_together(app, settle):
    """Specification 11.3/11.4: the cycle is chosen, and one cycle governs
    both.  A tune in 8 with a beat in 7 would drift apart."""
    app.new_project("Tala")
    app.update_brief(mood="devotional", language="Tamil", duration_target=20.0)
    app.select_raaga("Hamsadhwani")

    for name, aksharas in (("Misra Chapu", 7), ("Khanda Chapu", 5), ("Adi", 8)):
        app.update_brief(tala=name)
        app.generate_tune(seed=4)
        settle()
        app.generate_beat()
        settle()
        assert app.current_tala().name == name
        assert app.project.melody().beats_per_cycle == aksharas, \
            f"{name}: the tune is not in {aksharas}"
        assert app.project.beat().tala == name, \
            f"{name}: the beat is in a different cycle from the tune"


def test_an_unset_tala_keeps_the_cycle_the_tune_is_already_in(app, settle):
    app.new_project("Inferred")
    app.update_brief(mood="devotional", duration_target=20.0)
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=4)
    settle()
    assert app.project.brief.tala == ""
    assert app.current_tala().aksharas == app.project.melody().beats_per_cycle


# --------------------------------------------------------------------------
# Stop is for stopping, not a prerequisite
# --------------------------------------------------------------------------
def test_a_creative_action_stops_what_is_playing(ready, settle):
    """Specification 12.2/12.4: pressing Stop first should not be a habit.

    A creative action used to start while the previous audio kept going
    underneath it, so the way out of the last thing was always Stop.
    """
    app = ready
    stopped = []
    real_stop = app.playback.stop
    app.playback.stop = lambda: (stopped.append(True), real_stop())[1]

    actions = (("generate tune", lambda: app.generate_tune(seed=3)),
               ("tune variation", lambda: app.make_variation()),
               ("generate beat", lambda: app.generate_beat()),
               ("beat variation", lambda: app.beat_variation()),
               ("write lyrics", lambda: app.generate_lyrics(seed=1)),
               ("arrange", lambda: app.auto_arrange()),
               ("change the tempo", lambda: app.set_tempo(96)))
    try:
        for name, run in actions:
            # No audio device under test, so assert the action reaches the
            # transport rather than that a sound card obeyed.
            app.playback._playing, app.playback._paused = True, False
            stopped.clear()
            run()
            settle()
            assert stopped, f"{name} started while the last thing was playing"
    finally:
        app.playback.stop = real_stop


def test_taking_the_floor_says_whether_anything_was_playing(ready):
    app = ready
    app.playback._playing, app.playback._paused = False, False
    assert app.take_the_floor("nothing") is False
    app.playback._playing, app.playback._paused = True, False
    assert app.take_the_floor("something") is True


def test_playing_something_does_not_refuse_the_next_thing(ready, settle):
    """``play_render`` takes no floor - playing *is* the action there.

    What has to hold is that asking for audio while other audio is going
    swaps to the new material rather than being refused; the transport's
    own ``load`` does the stopping, so nothing is left sounding underneath.
    """
    app = ready
    app.generate_tune(seed=5)
    settle()
    app.render("tune", autoplay=False)
    app.generate_beat()
    settle()
    app.render_beat(autoplay=False)
    settle()

    app.play_render("tune")
    assert app.playback.source_name == "tune"
    tune_render = app._loaded_render

    app.playback._playing, app.playback._paused = True, False
    app.play_render("beat")
    assert app.playback.source_name == "beat"
    assert app._loaded_render is not tune_render
    assert "Nothing has been rendered" not in app.playback.last_error


def test_stop_says_which_of_the_two_things_happened(ready):
    app = ready
    app.playback._playing, app.playback._paused = False, False
    app.stop()
    assert app.status_text == "Nothing was playing"
    app.playback._playing, app.playback._paused = True, False
    app.stop()
    assert app.status_text == "Stopped"


# --------------------------------------------------------------------------
# Unreadable mood words (Arya's specification, 2026-09-06 13:16)
# --------------------------------------------------------------------------
def test_an_unfamiliar_mood_word_neither_blocks_nor_vanishes(ready, settle):
    """The brief is answered on what was understood, and the rest is kept."""
    from raagacomposer.raaga import vocabulary

    app = ready
    status = app.apply_brief_sync(mood="nervy, hopeful, skittish")
    settle()

    assert status.state.name == "COMPLETED", status.message
    assert app.raaga_suggestions(4), "the understood words produced nothing"

    kept = {t.term for t in app.agent.repo.unresolved_terms()}
    assert {"nervy", "skittish"} <= kept
    assert "hopeful" not in kept


def test_the_brief_says_which_words_it_could_not_use(ready, settle):
    """Section: do not claim every mood influenced the result."""
    app = ready
    status = app.apply_brief_sync(mood="nervy, hopeful")
    settle()
    assert "nervy" in status.message, status.message
    assert "not used yet" in status.message, status.message


def test_a_brief_of_known_words_says_nothing_about_deferrals(ready, settle):
    """Both fields are cleared deliberately.

    The fixture's feel is "lonely, late at night, but still warm", and
    *late* is a word the engine genuinely cannot read - so setting only the
    mood does not make a brief the engine fully understands.  The first
    version of this test asserted otherwise and was wrong about its own
    fixture, not about the code.
    """
    app = ready
    status = app.apply_brief_sync(mood="hopeful, romantic", feel="")
    settle()
    assert "not used yet" not in status.message, status.message


def test_a_resolved_word_reaches_the_next_search(ready, settle):
    app = ready
    app.apply_brief_sync(mood="nervy", feel="")
    settle()
    assert app.agent.repo.unresolved_term("nervy") is not None

    app.agent.repo.record_investigation("nervy", ["nervous", "tense"], 0.8,
                                        "on edge")

    readable = app.readable_brief(app.project.brief)
    assert "nervous" in readable.mood and "nervy" not in readable.mood

    status = app.apply_brief_sync(mood="nervy", feel="")
    settle()
    assert "not used yet" not in status.message, status.message


def test_the_brief_says_when_a_guess_is_shaping_the_result(ready, settle):
    """A resolution used to succeed silently: the deferral line vanished and
    nothing replaced it, so a guess did the ranking unannounced."""
    app = ready
    app.apply_brief_sync(mood="nervy", feel="")
    settle()
    app.agent.repo.record_investigation("nervy", ["nervous", "tense"], 0.8,
                                        "on edge")

    status = app.apply_brief_sync(mood="nervy", feel="")
    settle()
    assert "Reading nervy as nervous and tense (unconfirmed)" in status.message, \
        status.message
    assert "not used yet" not in status.message, status.message


def test_a_guess_about_a_word_you_did_not_use_is_not_mentioned(ready, settle):
    app = ready
    app.apply_brief_sync(mood="nervy", feel="")
    settle()
    app.agent.repo.record_investigation("nervy", ["nervous"], 0.8, "on edge")

    status = app.apply_brief_sync(mood="hopeful", feel="")
    settle()
    assert "Reading nervy" not in status.message, status.message


def test_a_dismissed_reading_stops_shaping_the_brief(ready, settle):
    app = ready
    app.apply_brief_sync(mood="nervy", feel="")
    settle()
    app.agent.repo.record_investigation("nervy", ["nervous", "tense"], 0.8, "x")
    assert app.readings_in_use(app.project.brief)

    app.agent.repo.dismiss_term("nervy", "wrong")

    assert app.readings_in_use(app.project.brief) == []
    readable = app.readable_brief(app.project.brief)
    assert "nervy" in readable.mood, "the word was still being rewritten"
    status = app.apply_brief_sync(mood="nervy", feel="")
    settle()
    assert "Reading nervy" not in status.message, status.message


# --------------------------------------------------------------------------
# A live microphone must not be able to freeze the window.
# Crash of 2026-09-06 20:25: a conversation held near the machine filled the
# utterance queue faster than it drained, each phrase was interpreted on the
# interface thread, and the application died with Stop Listening unserviced.
# --------------------------------------------------------------------------
def test_a_flood_of_speech_does_not_block_the_interface(ready, settle):
    """104 phrases arrived and 89 were answered.  The gap was the crash."""
    import time as _time

    app = ready
    slow = []

    def crawl(text, *args, **kwargs):
        slow.append(text)
        _time.sleep(0.05)          # stands in for a language model
        raise RuntimeError("unintelligible")

    import raagacomposer.app as module
    original = module.interpret
    module.interpret = crawl
    try:
        for i in range(60):
            app._on_transcript_final(f"phrase number {i}")

        # The queue is bounded, so the room cannot make unbounded work.
        assert app._utterance_queue.qsize() <= 8
        assert app._utterances_dropped > 0, "nothing was dropped; the queue grew"

        # One pump must start at most one interpretation and return at once.
        started = _time.time()
        app.pump()
        assert _time.time() - started < 0.05, "pump blocked on interpretation"
        assert app._interpreting is not None
    finally:
        module.interpret = original
        app.stop_listening()
        settle()


def test_stop_listening_discards_the_backlog(ready, settle):
    """Before this, being told to stop meant working through everything
    already heard - the button could not end what it was for."""
    app = ready
    for i in range(6):
        app._on_transcript_final(f"something overheard {i}")
    assert app._utterance_queue.qsize() > 0

    app.stop_listening()

    assert app._utterance_queue.qsize() == 0
    assert app._interpreting is None
    assert "Microphone off" in app.status_text


def test_the_queue_drops_rather_than_growing(ready):
    """A microphone is an unbounded source of work; the queue is not."""
    app = ready
    for i in range(200):
        app._on_transcript_final(f"talking {i}")
    assert app._utterance_queue.qsize() <= 8
    assert app._utterances_dropped >= 190
    app._clear_utterances()


def test_an_interpreted_phrase_is_still_acted_on(ready, settle):
    """The fix must not break the feature it protects."""
    app = ready
    acted = []
    app.execute = lambda cmd: acted.append(cmd.intent)

    app._on_transcript_final("stop")
    app.pump()
    settle()
    for _ in range(200):
        app.pump()
        if app._interpreting is None:
            break
    assert acted or "did not understand" in app.status_text.lower(), \
        f"nothing happened and nothing was said: {app.status_text!r}"


# --------------------------------------------------------------------------
# Asking about what it knows (Krish's acceptance sequence, 2026-09-07 08:50)
# --------------------------------------------------------------------------
def _teach(app, raaga="Keeravani"):
    from raagacomposer.agent.knowledge import Phrase, Source
    from raagacomposer.core import provenance
    source, _ = app.agent.repo.add_source(Source(
        locator="rec://one", title="a real recording", raaga=raaga,
        origin=provenance.HUMAN, status="analysed"))
    for swaras in (["S", "R2", "G2"], ["G2", "M1", "P"], ["P", "D1", "N3"]):
        app.agent.repo.add_phrase(Phrase(raaga=raaga, swaras=swaras,
                                         source_id=source.id, confidence=0.7))
    return source


def test_a_follow_up_question_keeps_its_subject(ready):
    """Ask about a raaga, then ask a follow-up without naming it.

    Every question used to re-derive the subject from its own words, so
    "what did it learn?" answered about whatever the curriculum was
    studying rather than about the raaga just asked about.
    """
    app = ready
    _teach(app, "Keeravani")

    first = app.ask_agent("has Keeravani been trained?")
    assert "Keeravani" in first
    assert app.question_subject() == "Keeravani"

    for follow_up in ("what did it learn?", "which recordings support that?",
                      "what is missing?"):
        answer = app.ask_agent(follow_up)
        assert "Keeravani" in answer, \
            f"{follow_up!r} lost the subject: {answer[:80]!r}"


def test_the_training_question_is_answered_from_records(ready):
    app = ready
    assert "no training" in app.ask_agent("has Keeravani been trained?").lower()

    _teach(app, "Keeravani")
    answer = app.ask_agent("has Keeravani been trained?")
    assert answer.lower().startswith("yes")
    assert "3 phrase(s) heard in 1 recording(s)" in answer
    assert "person's recording" in answer


# --------------------------------------------------------------------------
# Arya's four reproduced answer errors (2026-09-07 11:08)
# --------------------------------------------------------------------------
def _register(app, raaga="Keeravani", status="pending", origin="", n=1,
              phrases=0):
    """Put sources on file without pretending they were learned from."""
    from raagacomposer.agent.knowledge import Phrase, Source
    from raagacomposer.core import provenance
    made = []
    for i in range(n):
        source, _ = app.agent.repo.add_source(Source(
            locator=f"rec://{status}-{i}", title=f"fixture {status} {i}",
            raaga=raaga, origin=origin or provenance.HUMAN, status=status))
        made.append(source)
        for k in range(phrases):
            app.agent.repo.add_phrase(Phrase(
                raaga=raaga,
                swaras=["S", "R2", "G2"] + ["M1"] * (i + k + 1),
                source_id=source.id, confidence=0.8,
                origin=origin or provenance.HUMAN))
    return made


def test_registering_a_recording_is_not_learning_from_it(ready):
    """Arya's finding: a source alone answered "yes, trained".

    A recording that is queued, or one whose analysis failed, is something
    the agent has.  It is not something it has heard.
    """
    app = ready
    _register(app, status="pending")
    answer = app.ask_agent("has Keeravani been trained?")
    assert answer.lower().startswith("no"), answer
    assert "1 waiting to be analysed" in answer
    assert "not the same as having heard it" in answer


def test_a_failed_analysis_is_reported_without_claiming_training(ready):
    app = ready
    _register(app, status="failed")
    answer = app.ask_agent("has Keeravani been trained?")
    assert answer.lower().startswith("no"), answer
    assert "whose most recent analysis failed" in answer


def test_a_later_failure_does_not_discount_earlier_learning(ready):
    """The other half of the same rule: retained evidence decides."""
    app = ready
    _teach(app, "Keeravani")
    _register(app, status="failed")
    answer = app.ask_agent("has Keeravani been trained?")
    assert answer.lower().startswith("yes"), answer
    assert "3 phrase(s) heard in 1 recording(s)" in answer
    assert "most recent analysis failed" in answer, "the failure was hidden"


def test_reference_practice_is_not_described_as_a_performance(ready):
    """Arya's finding: reference material reported as heard in recordings.

    Practising against the library's own rendered material is legitimate
    and is not listening to a singer.  Both facts have to survive.
    """
    from raagacomposer.core import provenance
    app = ready
    _register(app, status="analysed", origin=provenance.REFERENCE, phrases=1)

    learned = app.ask_agent("what did it learn about Keeravani?")
    assert "heard in real recordings" not in learned, learned
    assert "reference pack" in learned
    assert "not a performance" in learned

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("yes"), "reference practice still counts"
    assert "Nothing heard from a recording yet" in trained
    assert "1 phrase(s) heard in" not in trained


def test_a_second_recording_is_not_missing_when_two_are_on_file(ready):
    """Arya's finding: a sentence written for one case, printed for all."""
    app = ready
    _register(app, status="analysed", n=2, phrases=1)
    answer = app.ask_agent("what is missing for Keeravani?")
    assert "a second recording would" not in answer, answer
    assert "2 recordings" in answer, answer
    assert "not assessed" in answer, "it must say what it has not checked"


def test_totals_are_totals_and_not_the_size_of_the_page(ready):
    """Arya's finding: twenty-one recordings answered as twenty."""
    app = ready
    _register(app, status="analysed", n=21, phrases=1)

    trained = app.ask_agent("has Keeravani been trained?")
    assert "21 phrase(s) heard in 21 recording(s)" in trained, trained

    listing = app.ask_agent("which recordings support that?")
    assert "a listing limit, not the total" in listing, listing
    # The stated total is whatever is on file - a fresh install also seeds a
    # reference pack - and the point is only that it exceeds the page shown.
    stated = int(re.search(r"the 20 most recent of (\d+)", listing).group(1))
    assert stated >= 21, f"the page size was reported as the total: {stated}"


def test_a_question_about_a_raaga_is_not_answered_about_the_tune(ready, settle):
    """Arya's finding: any question containing "why" or "phrase" was routed
    to the tune explainer as soon as a tune existed, which stole exactly the
    questions this feature is for."""
    app = ready
    _teach(app, "Keeravani")
    app.generate_tune(seed=3)
    settle()
    assert app.project.melody() is not None, "no tune, so the case is untested"

    answer = app.ask_agent("what phrases has Keeravani learned?")
    assert "Keeravani" in answer
    # the tune explainer talks about the line it wrote, not about records
    assert "learned" in answer.lower() or "heard" in answer.lower()


def test_a_question_about_the_tune_still_reaches_the_tune(ready, settle):
    """The fix must not strand the tune explanation."""
    app = ready
    app.generate_tune(seed=3)
    settle()
    assert app.project.melody() is not None

    asked = []
    app.agent.explain_choice = lambda melody, raaga: asked.append(raaga) or "ok"
    app.ask_agent("why did you write this tune that way?")
    assert asked, "a question about the tune no longer reaches explain_choice"


def test_asking_never_changes_the_song(ready, settle):
    """Informational questions must not compose, train, or alter anything."""
    app = ready
    _teach(app, "Keeravani")
    app.generate_tune(seed=3)
    settle()
    before = app.project.melody().version
    phrases_before = app.agent.repo.count_phrases("Keeravani")

    for q in ("has Keeravani been trained?", "what did it learn?",
              "what is missing?", "which recordings support that?"):
        app.ask_agent(q)

    assert app.project.melody().version == before
    assert app.agent.repo.count_phrases("Keeravani") == phrases_before


def _fail_the_source(app, source):
    """The same source re-run, its second attempt failing, findings kept.

    add_source returns an existing row untouched, so the state has to be
    reached the way research reaches it: by updating the source in place.
    """
    app.agent.repo.update_source(source.id, status="failed",
                                 error="fixture: reprocessing failed")
    return app.agent.repo.source(source.id)


def test_a_failure_on_the_same_source_does_not_erase_its_findings(ready):
    """Arya's finding: my earlier test failed a *different* source.

    The state that contradicts itself is one source whose findings were
    kept and whose most recent attempt then failed.  Status is the history
    of the last attempt; it is not a statement about what is still held.
    """
    app = ready
    source = _teach(app, "Keeravani")
    refreshed = _fail_the_source(app, source)
    assert refreshed.status == "failed", "the fixture did not reach the state"

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("yes"), trained
    assert "3 phrase(s) heard in 1 recording(s)" in trained
    assert "0 recording(s)" not in trained
    assert "most recent analysis failed" in trained

    listing = app.ask_agent("which recordings support that?")
    assert "on file" in listing, listing
    assert "failed, kept" in listing, "the row denied its own retained finding"

    gaps = app.ask_agent("what is missing for Keeravani?")
    assert "Nothing has been heard from a recording" not in gaps, gaps


def test_a_fact_learned_from_a_recording_is_not_called_the_library(ready):
    """Arya's finding: every fact was attributed to the built-in reference
    whenever no analysed source or learned phrase remained."""
    from raagacomposer.agent.knowledge import Fact, Source
    from raagacomposer.core import provenance
    app = ready
    source, _ = app.agent.repo.add_source(Source(
        locator="rec://facts-only", title="a real recording", raaga="Keeravani",
        origin=provenance.HUMAN, status="analysed"))
    app.agent.repo.add_fact(Fact(raaga="Keeravani", key="fixture_fact",
                                 value="heard in the recording",
                                 confidence=0.8, source_id=source.id))
    _fail_the_source(app, source)

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("yes"), trained
    assert "fact(s) learned from a recording" in trained
    assert "no training" not in trained.lower()


def test_an_analysis_that_kept_nothing_is_not_training(ready):
    """Arya's finding: "analysed" only means the attempt finished.

    research marks a source analysed whether or not anything was retained,
    so reading the status as learning claimed training while the content
    answer correctly said nothing had been learned.
    """
    app = ready
    _register(app, status="analysed", n=1, phrases=0)

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("no"), trained
    assert "analysed with nothing kept from it" in trained

    # The library's own structural facts are seeded on a fresh install, so
    # the honest answer is not "nothing" - it is that nothing was learned
    # from a recording, which is what the training answer just said.
    learned = app.ask_agent("what did it learn about Keeravani?")
    assert "Nothing has been learned from a recording" in learned, learned
    assert "What I have learned about" not in learned


def test_a_source_listing_states_what_a_source_is_not_what_it_taught(ready):
    """Arya's finding: a queued recording listed as "learned from"."""
    app = ready
    _register(app, status="pending")
    listing = app.ask_agent("which recordings support that?")
    assert "a person's recording" in listing, listing
    assert "learned from a person's recording" not in listing
    assert "nothing kept" in listing


def _fact_from(app, origin, raaga="Keeravani", with_source=True):
    """A fact whose source has a given origin - or no source at all."""
    from raagacomposer.agent.knowledge import Fact, Source
    source_id = ""
    if with_source:
        source, _ = app.agent.repo.add_source(Source(
            locator=f"fixture://{origin}", title=f"fixture {origin}",
            raaga=raaga, origin=origin, status="analysed"))
        source_id = source.id
    app.agent.repo.add_fact(Fact(raaga=raaga, key="fixture_fact",
                                 value="review only", confidence=0.8,
                                 source_id=source_id))


def test_a_fact_with_no_identified_source_is_not_called_the_library(ready):
    """Arya's finding: the remainder was labelled the shipped library.

    reference_facts was len(facts) minus the learned ones, so a fact with no
    source, an unknown source or a generated one was reported as knowledge
    the application had been given - a provenance arrived at by subtraction.
    """
    from raagacomposer.core import provenance
    app = ready
    _fact_from(app, provenance.UNKNOWN)

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("no"), trained
    assert "1 fact(s) on file with no identified source" in trained, trained
    # The seeded library facts stay fifteen: the unattributed one was named
    # separately rather than absorbed into them.
    assert "library's built-in reference (15 fact(s))" in trained


def test_a_fact_this_system_wrote_is_named_as_its_own(ready):
    from raagacomposer.core import provenance
    app = ready
    _fact_from(app, provenance.GENERATED)

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("no"), trained
    assert "1 fact(s) this system wrote itself" in trained, trained
    assert "library's built-in reference (15 fact(s))" in trained


def test_a_fact_with_no_source_at_all_is_reported_as_such(ready):
    from raagacomposer.core import provenance
    app = ready
    _fact_from(app, provenance.HUMAN, with_source=False)

    trained = app.ask_agent("has Keeravani been trained?")
    assert trained.lower().startswith("no"), trained
    assert "1 fact(s) on file with no identified source" in trained, trained
    assert "library's built-in reference (15 fact(s))" in trained


def test_the_library_keeps_its_reference_attribution(ready):
    """The control: a fact that really does come from the shipped library."""
    from raagacomposer.core import provenance
    app = ready
    _fact_from(app, provenance.REFERENCE)

    trained = app.ask_agent("has Keeravani been trained?")
    assert "library's built-in reference" in trained, trained

    learned = app.ask_agent("what did it learn about Keeravani?")
    assert "the library ships with" in learned, learned


def test_a_later_failure_is_reported_as_history_not_as_denial(ready):
    """Arya's point: "and not learning" read as a denial of what was kept."""
    app = ready
    source = _teach(app, "Keeravani")
    _fail_the_source(app, source)

    trained = app.ask_agent("has Keeravani been trained?")
    assert "Latest analysis status" in trained, trained
    assert "not learning" not in trained
    assert "3 phrase(s) heard in 1 recording(s)" in trained


def _fact_keyed(app, key, origin, raaga="Keeravani", with_source=True):
    """A fact under a chosen key, so it sorts into the displayed page.

    facts() orders by key, and a fresh install seeds fifteen; a fixture key
    beginning with "aa_" is therefore visible rather than paged out.
    """
    from raagacomposer.agent.knowledge import Fact, Source
    source_id = ""
    if with_source:
        source, _ = app.agent.repo.add_source(Source(
            locator=f"fixture://{key}", title=f"fixture {key}", raaga=raaga,
            origin=origin, status="analysed"))
        source_id = source.id
    app.agent.repo.add_fact(Fact(raaga=raaga, key=key, value="review only",
                                 confidence=0.8, source_id=source_id))


def _fact_rows(answer):
    """The displayed fact lines, excluding headings and phrase lines."""
    return [line for line in answer.split("\n")
            if line.startswith("  aa_") or line.startswith("  fixture")]


def test_each_displayed_fact_says_where_it_came_from(ready):
    """Arya's finding: the heading assigned every row to the library.

    A store holding one library fact, one this system wrote and one with no
    source at all listed all three under whichever heading the first of them
    earned.  A heading cannot carry per-row provenance; the rows must.
    """
    from raagacomposer.core import provenance
    app = ready
    _fact_keyed(app, "aa_reference", provenance.REFERENCE)
    _fact_keyed(app, "aa_generated", provenance.GENERATED)
    _fact_keyed(app, "aa_unknown", provenance.UNKNOWN)

    learned = app.ask_agent("what did it learn about Keeravani?")
    assert "the library ships with" not in learned, learned
    assert "What is on file" in learned
    assert "aa_generated" in learned and "written by this system" in learned
    assert "aa_unknown" in learned and "no identified source" in learned
    assert "aa_reference" in learned
    assert "rendered for practice" in learned
    for row in _fact_rows(learned):
        assert " - " in row, f"a displayed fact carried no source: {row!r}"


def test_learned_facts_do_not_lend_their_standing_to_the_others(ready):
    """The second half: once anything is genuinely learned, the heading
    becomes "What I have learned" and the unlearned rows must not inherit
    that claim."""
    from raagacomposer.core import provenance
    app = ready
    _teach(app, "Keeravani")
    _fact_keyed(app, "aa_generated", provenance.GENERATED)
    _fact_keyed(app, "aa_unknown", provenance.UNKNOWN)

    learned = app.ask_agent("what did it learn about Keeravani?")
    assert learned.startswith("What I have learned about Keeravani:"), learned
    assert "aa_generated" in learned and "written by this system" in learned
    assert "aa_unknown" in learned and "no identified source" in learned
    for row in _fact_rows(learned):
        assert " - " in row, f"a displayed fact carried no source: {row!r}"


def test_the_fact_listing_says_when_it_is_only_a_page(ready):
    """Fifteen seeded facts, six shown: the same limit-as-total error."""
    app = ready
    _teach(app, "Keeravani")
    learned = app.ask_agent("what did it learn about Keeravani?")
    assert "a display limit, not the total" in learned, learned


# --------------------------------------------------------------------------
# Jam workspace: the raaga a creator names is the raaga they hear
# (Arya's first named path, 2026-09-07 12:31)
# --------------------------------------------------------------------------
def _say(app, text):
    """Interpret and act, the way the conversation panel does."""
    from raagacomposer.speech.intent import interpret
    cmd = interpret(text, app.context.time_context())
    app.apply_utterance(text, cmd)
    return cmd


def test_composing_in_a_named_raaga_uses_that_raaga(ready, settle):
    """Krish's own Jam example: ask for Hamsadhwani, get Hamsadhwani.

    The interpreter parsed the name correctly all along; the dispatch called
    generate_tune() and never read it, so the tune came back in whatever was
    already selected.
    """
    app = ready
    app.select_raaga("Keeravani", "the project starts here")
    cmd = _say(app, "compose a tune in Hamsadhwani")
    assert cmd.intent == "tune.generate" and cmd.raaga == "Hamsadhwani", cmd
    settle()

    melody = app.project.melody()
    assert melody is not None, "no tune was composed"
    assert melody.raaga == "Hamsadhwani", f"composed in {melody.raaga}"
    assert app.project.raaga.selected == "Hamsadhwani", \
        "the tune and the panel disagree about the raaga"


def test_a_locked_raaga_is_not_silently_overridden_or_ignored(ready, settle):
    """Neither answer is acceptable on its own: composing in the locked
    raaga answers a question they did not ask, and switching anyway undoes
    a decision they did.  So it says so and composes nothing."""
    app = ready
    app.select_raaga("Keeravani", "chosen deliberately")
    app.set_raaga_lock(True)
    # The workflow fixture has already composed once, so what has to stay
    # unchanged is the number of versions, not their absence.
    before = len(app.project.melodies)

    _say(app, "compose a tune in Hamsadhwani")
    settle()

    assert app.project.raaga.selected == "Keeravani", "the lock was overridden"
    assert len(app.project.melodies) == before, "it composed anyway"
    assert "locked" in app.status_text.lower(), app.status_text


def test_an_unknown_raaga_is_named_rather_than_substituted(ready, settle):
    """A name the library does not have is said aloud, not swapped out.

    The rule interpreter only extracts raaga names it recognises, so it
    cannot deliver this case - "compose a tune in Nonexistentraaga" parses
    with an empty raaga and is an ordinary compose request, which is also
    what keeps "compose a tune in the morning" working.  The LLM
    interpretation can return any string, so the guard is reached there,
    and that is the path this exercises.
    """
    from raagacomposer.speech.intent import Command
    app = ready
    app.select_raaga("Keeravani", "the project starts here")
    before = len(app.project.melodies)

    app.apply_utterance("compose a tune in Nonexistentraaga",
                        Command(intent="tune.generate",
                                text="compose a tune in Nonexistentraaga",
                                raaga="Nonexistentraaga", confidence=0.9))
    settle()
    assert len(app.project.melodies) == before, "it composed something anyway"
    assert "do not know" in app.status_text.lower(), app.status_text


def test_an_unrecognised_word_is_not_treated_as_a_raaga(ready, settle):
    """The other side of it: "in the morning" is not a raaga, and neither
    is an unknown word, so both stay ordinary compose requests."""
    from raagacomposer.speech.intent import interpret
    app = ready
    for text in ("compose a tune in the morning",
                 "compose a tune in Nonexistentraaga"):
        cmd = interpret(text, app.context.time_context())
        assert cmd.intent == "tune.generate", cmd
        assert cmd.raaga == "", f"{text!r} invented a raaga: {cmd.raaga!r}"


def test_composing_without_naming_a_raaga_still_works(ready, settle):
    """The guard must not stand between the creator and the ordinary case."""
    app = ready
    app.select_raaga("Keeravani", "the project starts here")
    before = len(app.project.melodies)
    _say(app, "compose a tune")
    settle()
    melody = app.project.melody()
    assert len(app.project.melodies) > before, "nothing was composed"
    assert melody is not None and melody.raaga == "Keeravani", melody


# --------------------------------------------------------------------------
# One door into interpretation (Arya's third named path, 2026-09-07 12:31)
# --------------------------------------------------------------------------
def test_typed_text_is_not_interpreted_on_the_calling_thread(ready):
    """The conversation box used to call handle_utterance inline.

    That is the fault the microphone had: interpreting may wait on a
    language model, and doing it on the thread that redraws the window
    freezes the window.  say() queues instead.
    """
    app = ready
    before = len(app.project.conversation)
    assert app.say("compose a tune") is True
    assert len(app.project.conversation) == before, \
        "the instruction was acted on before returning"

    for _ in range(400):
        app.pump()
        if len(app.project.conversation) > before:
            break
        time.sleep(0.01)
    assert len(app.project.conversation) > before, "it was never acted on"


def test_a_typed_instruction_is_served_before_overheard_speech(ready):
    """Deliberate text should not queue behind a talkative room."""
    app = ready
    for i in range(5):
        app._on_transcript_final(f"overheard phrase {i}")
    app.say("play the first minute")

    app._drain_utterances()
    assert app._interpreting == "play the first minute", app._interpreting


def test_a_typed_instruction_is_never_dropped_silently(ready):
    """Overheard speech may be discarded when the room outruns us; what
    the creator typed may not be."""
    app = ready
    for i in range(20):
        app._on_transcript_final(f"overheard {i}")
    assert app._utterances_dropped > 0, "the speech queue did not overflow"

    assert app.say("compose a tune") is True, "typed text was refused"
    assert app._typed_queue.qsize() == 1


def test_stop_listening_does_not_discard_what_was_typed(ready):
    """Stop Listening is about the microphone.  Silently throwing away a
    typed instruction would be a different act wearing the same button."""
    app = ready
    app._on_transcript_final("something overheard")
    app.say("compose a tune")

    discarded = app._clear_utterances()
    assert discarded == 1, discarded
    assert app._typed_queue.qsize() == 1, "the typed instruction was discarded"


# --------------------------------------------------------------------------
# Selected-section song production (Krish's priority before Jam, 12:48)
# --------------------------------------------------------------------------
def _sections(app):
    melody = app.project.melody()
    assert melody is not None, "no tune to work from"
    return melody, {s.name: s for s in melody.sections}


def _written(app, section_id):
    lyrics = app.project.lyrics_version()
    return [l.text for l in lyrics.lines
            if l.section_id == section_id and l.text.strip()]


def test_lyrics_can_be_written_for_one_section(ready, settle):
    """Krish settles the Pallavi first, against the tune it already has."""
    app = ready
    melody, by_name = _sections(app)
    pallavi = by_name.get("Pallavi")
    assert pallavi is not None, f"no Pallavi in {list(by_name)}"

    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    settle()

    lyrics = app.project.lyrics_version()
    assert lyrics is not None, "no lyrics were produced"
    assert _written(app, pallavi.id), "the Pallavi got no words"
    elsewhere = [s.name for s in melody.sections
                 if s.id != pallavi.id and _written(app, s.id)]
    assert not elsewhere, f"words were written outside the selection: {elsewhere}"
    assert "Pallavi" in app.status_text, app.status_text


def test_writing_a_second_section_leaves_the_first_alone(ready, settle):
    """The whole point of choosing: settling one part must not cost another."""
    app = ready
    melody, by_name = _sections(app)
    pallavi, charanam = by_name.get("Pallavi"), by_name.get("Charanam 1")
    assert pallavi is not None and charanam is not None, list(by_name)

    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    settle()
    settled = _written(app, pallavi.id)
    assert settled, "the Pallavi got no words to begin with"

    app.generate_lyrics(seed=9, section_ids=[charanam.id])
    settle()

    assert _written(app, pallavi.id) == settled, "the Pallavi words changed"
    assert _written(app, charanam.id), "the Charanam got no words"


def test_an_instrumental_section_is_told_it_is_instrumental(ready, settle):
    """Instrumental regions need music, not compulsory lyrics."""
    app = ready
    melody, _ = _sections(app)
    instrumental = next(s for s in melody.sections if s.kind.instrumental)
    before = len(app.project.lyrics)

    app.generate_lyrics(seed=3, section_ids=[instrumental.id])
    settle()

    assert len(app.project.lyrics) == before, "it wrote words anyway"
    assert "instrumental" in app.status_text.lower(), app.status_text


def test_a_locked_section_is_not_rewritten(ready, settle):
    app = ready
    melody, by_name = _sections(app)
    pallavi = by_name["Pallavi"]
    pallavi.locked = True
    before = len(app.project.lyrics)

    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    settle()

    assert len(app.project.lyrics) == before, "a locked section was rewritten"
    assert "locked" in app.status_text.lower(), app.status_text


def test_a_section_this_tune_does_not_have_is_refused(ready, settle):
    app = ready
    before = len(app.project.lyrics)
    app.generate_lyrics(seed=3, section_ids=["sec_not_in_this_tune"])
    settle()
    assert len(app.project.lyrics) == before, "it wrote something anyway"
    assert "not part of this tune" in app.status_text.lower(), app.status_text


def test_asking_for_no_section_still_writes_the_whole_song(ready, settle):
    """Every existing caller passes nothing, and must keep working."""
    app = ready
    melody, _ = _sections(app)
    app.generate_lyrics(seed=3)
    settle()
    sung = [s for s in melody.sections if not s.kind.instrumental]
    with_words = [s.name for s in sung if _written(app, s.id)]
    assert len(with_words) > 1, f"only {with_words} got words"


# --------------------------------------------------------------------------
# An explicit tempo survives Generate Tune (Krish's walkthrough, 14:35)
# --------------------------------------------------------------------------
def test_an_explicit_tempo_survives_generating_again(ready, settle):
    """Krish set 200 bpm, pressed Generate Tune, and got 80 back.

    melody_options carried the tune's tempo and generate_tune overwrote it
    on the next line, and set_tempo only recorded the creator's choice when
    there was no tune to apply it to - so on an existing tune the choice
    lived in that melody alone and the next generation never saw it.
    """
    app = ready
    app.set_tempo(200)
    settle()
    assert app.project.brief.tempo_preference == 200, \
        "the choice was not remembered"
    assert app.project.melody().tempo_bpm == 200

    app.generate_tune(seed=6)
    settle()
    assert app.project.melody().tempo_bpm == 200, "the tempo was reset"


def test_generating_again_still_makes_a_song_of_the_asked_for_length(ready,
                                                                     settle):
    """The other half: a time-scaled tune is shorter, and that shortness is
    not a new request.  The duration target comes from the brief."""
    app = ready
    app.project.brief.duration_target = 60
    app.set_tempo(200)
    settle()
    shortened = app.project.melody().duration

    app.generate_tune(seed=6)
    settle()
    fresh = app.project.melody()
    assert fresh.tempo_bpm == 200
    assert fresh.duration > shortened + 5, \
        f"the song shrank to the time-scaled length: {fresh.duration:.0f}s"


def test_the_tempo_is_still_inferred_when_nobody_has_chosen_one(ready, settle):
    """The automatic path has to keep working: only an explicit choice
    takes precedence over it."""
    app = ready
    assert not app.project.brief.tempo_preference, "the fixture chose one"
    before = app.project.melody().tempo_bpm
    app.generate_tune(seed=7)
    settle()
    assert app.project.melody().tempo_bpm == before, \
        "an inferred tempo should stay stable across regeneration"


# --------------------------------------------------------------------------
# The cycle is chosen, not defaulted (Krish's component list, 14:35)
# --------------------------------------------------------------------------
def test_a_new_tune_is_composed_in_a_cycle_chosen_from_the_brief(ready, settle):
    """Every new tune used to be Adi because Adi is the fallback.

    A default is not a choice: nothing told the creator a decision had been
    made for them, and nothing offered another.
    """
    app = ready
    # A fresh song: with a tune in hand the cycle rightly comes from the
    # tune, which is a different rule and has its own test below.
    app.new_project("Cycle probe", write=False)
    app.update_brief(tala="")
    app.apply_brief_sync(mood="tense, nervous", feel="",
                         situation="a chase through a city at night")
    app.select_raaga("Keeravani", "for the test")
    choice = app.tala_choice()
    assert choice.tala.name == "Khanda Chapu", choice.describe()
    assert not choice.chosen_by_creator
    assert "chase" in choice.reason

    app.generate_tune(seed=11)
    settle()
    melody = app.project.melody()
    assert melody.beats_per_cycle == choice.tala.aksharas, \
        "the tune was not composed in the cycle that was chosen"


def test_an_explicitly_chosen_cycle_wins(ready, settle):
    app = ready
    app.new_project("Explicit cycle", write=False)
    app.update_brief(tala="Rupaka")
    app.select_raaga("Keeravani", "for the test")
    choice = app.tala_choice()
    assert choice.tala.name == "Rupaka"
    assert choice.chosen_by_creator, "an explicit choice must read as theirs"

    app.generate_tune(seed=12)
    settle()
    assert app.project.melody().beats_per_cycle == 6


def test_adding_a_beat_keeps_the_tune_in_its_own_cycle(ready, settle):
    """Krish: adding tala to an existing tune applies percussion to that
    tune on its existing timing - not a disconnected beat, and not a
    silent rewrite of the melody."""
    from raagacomposer.music import tala as tala_module
    app = ready
    app.update_brief(tala="")
    melody = app.project.melody()
    before_notes = [(n.midi, round(n.start, 4)) for n in melody.notes]
    cycle = melody.beats_per_cycle

    app.generate_beat()
    settle()
    beat = app.project.beat()
    assert beat is not None, "no beat was produced"
    assert tala_module.require(beat.tala).aksharas == cycle, \
        "the beat is in a different cycle from the tune"

    after = app.project.melody()
    assert [(n.midi, round(n.start, 4)) for n in after.notes] == before_notes, \
        "adding a beat rewrote the tune"


def test_the_reason_names_what_decided_it(ready):
    """A reason the creator cannot check is not a reason."""
    app = ready
    for situation, expected in (("a lullaby for a child", "Tisra Eka"),
                                ("a village folk festival", "Misra Chapu"),
                                ("two friends talking quietly", "Adi")):
        app.new_project(f"Reason probe {expected}", write=False)
        app.update_brief(tala="", situation=situation, mood="")
        choice = app.tala_choice()
        assert choice.tala.name == expected, \
            f"{situation!r} chose {choice.tala.name}: {choice.reason}"
        assert choice.reason, "no reason given"


def test_a_good_section_preview_still_mixes_and_plays(ready, settle):
    """The other half of Arya's lifecycle finding.

    Scoping the follow-on to its own render must not quietly disable it:
    a preview that succeeds still asks for the mix and plays that span.
    """
    app = ready
    melody, by_name = _sections(app)
    pallavi = by_name.get("Pallavi")
    assert pallavi is not None, list(by_name)
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    settle()

    asked = []
    original = app.render

    def watch(kind="full", autoplay=False, play_range=None):
        asked.append((kind, autoplay, play_range))
        return original(kind=kind, autoplay=False, play_range=play_range)

    app.render = watch
    try:
        app.preview_section(pallavi.id, autoplay=True)
        settle()
    finally:
        app.render = original

    assert asked, "the preview never asked for a mix"
    kind, autoplay, play_range = asked[-1]
    assert kind == "full", kind
    assert autoplay is True, "a preview the creator asked to hear did not play"
    assert play_range is not None
    start, end = play_range
    assert abs(start - pallavi.start) < 0.01 and abs(end - pallavi.end) < 0.01,         f"played {play_range}, not the Pallavi at {pallavi.start}-{pallavi.end}"


def test_feedback_on_a_new_song_is_not_judged_by_the_old_ones_evaluation(
        ready, settle):
    """The agent's verdict on one tune must not be filed against another.

    last_evaluation is set when a tune is composed and read when the
    creator says something about one.  It survived new_project, so praise
    or criticism of the new song recorded lessons derived from the
    previous song's evaluation under the new song's project id.
    """
    app = ready
    assert app.last_evaluation is not None, "the fixture composed nothing"

    app.new_project("A different song", write=False)
    assert app.last_evaluation is None, \
        "the previous song's evaluation followed us into the new one"
    assert app.playhead == 0.0, "the playhead came from the previous song"


def test_opening_a_song_does_not_inherit_the_last_ones_judgement(ready, settle,
                                                                 tmp_path):
    app = ready
    where = app.save()
    assert where is not None, "the fixture project was not saved"
    assert app.last_evaluation is not None

    app.new_project("Something else", write=False)
    app.open_project(where)
    assert app.last_evaluation is None, \
        "opening a song brought the other song's evaluation with it"


# --------------------------------------------------------------------------
# Adding a tala puts percussion on the tune (Arya, 2026-09-07 18:15)
# --------------------------------------------------------------------------
def _rhythm_versions(app):
    arrangement = app.project.arrangement()
    if arrangement is None:
        return None
    return [r.meta.get("beat_version") for t in arrangement.tracks
            if t.role == "rhythm" for r in t.regions]


def test_a_beat_reaches_the_song_without_finding_auto_arrange(ready, settle):
    """Krish asked for percussion on the tune, and got a beat that only
    played on its own: the full mix takes its parts from the arrangement,
    and there was none, so the beat was simply absent from the song."""
    app = ready
    app.generate_beat()
    settle()

    versions = _rhythm_versions(app)
    assert versions, "the beat never became part of the song"
    assert versions == [str(app.project.beat().version)]


def test_only_one_percussion_layer_after_auto_arrange(ready, settle):
    """Two code paths built a rhythm track, so the song ended up with two
    percussionists playing the same part."""
    app = ready
    app.generate_beat()
    settle()
    app.auto_arrange()
    settle()

    versions = _rhythm_versions(app)
    assert len(versions) == 1, f"{len(versions)} rhythm regions: {versions}"


def test_a_newer_beat_replaces_the_one_in_the_song(ready, settle):
    """The selected beat was v2 while the song still played v1."""
    app = ready
    app.generate_beat()
    settle()
    app.auto_arrange()
    settle()
    others = [(t.instrument, t.role, t.locked, len(t.regions))
              for t in app.project.arrangement().tracks if t.role != "rhythm"]

    app.beat_variation(strength="busier")
    settle()

    selected = app.project.beat()
    assert _rhythm_versions(app) == [str(selected.version)], \
        "the song still plays an older beat than the selected one"
    after = [(t.instrument, t.role, t.locked, len(t.regions))
             for t in app.project.arrangement().tracks if t.role != "rhythm"]
    assert after == others, "changing the beat rebuilt the accompaniment"


def test_a_locked_percussion_track_is_not_replaced(ready, settle):
    """A lock is a decision, and a new beat is not a reason to undo it."""
    app = ready
    app.generate_beat()
    settle()
    track = next(t for t in app.project.arrangement().tracks
                 if t.role == "rhythm")
    track.locked = True
    before = [r.meta.get("beat_version") for r in track.regions]

    # status_text holds only the latest line, and rendering the beat speaks
    # after this does, so what was said is collected rather than sampled.
    said = []
    original = app.status
    app.status = lambda text, *a, **k: (said.append(text), original(text))[1]
    try:
        app.beat_variation(strength="busier")
        settle()
    finally:
        app.status = original

    assert [r.meta.get("beat_version") for r in track.regions] == before, \
        "a locked percussion track was replaced"
    assert any("locked" in line.lower() for line in said), said


def test_a_superseded_vocal_completion_neither_lands_nor_plays(ready, settle):
    """Arya's timing case: a worker finishes, its completion waits in the
    queue for the interface timer, and in that window the creator asks for
    another render.  The ticket cannot tell the two apart - same song, same
    words, same voice - so the older one still wrote its take and played
    its follow-on.  What separates them is which was asked for last.
    """
    app = ready
    melody, by_name = _sections(app)
    pallavi = by_name.get("Pallavi")
    assert pallavi is not None, list(by_name)
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    settle()

    asked = []
    original_render = app.render
    app.render = lambda **kw: asked.append(kw)

    said = []
    original_status = app.status
    app.status = lambda text, *a, **k: (said.append(text),
                                        original_status(text))[1]
    takes_before = len(app.project.vocal_renders)
    try:
        # The first request is a section preview, which wants to play.
        app.preview_section(pallavi.id, autoplay=True)
        # Before its completion is drained, the creator asks for another.
        app.render_vocal(kind="preview", autoplay=False)
        settle()
    finally:
        app.render = original_render
        app.status = original_status

    played = [kw for kw in asked if kw.get("autoplay")]
    assert not played, f"a superseded preview still played: {played}"
    # Only one take is kept.  Which mechanism stopped the other - the job
    # manager superseding it on the shared target, or the ownership check
    # in its completion - depends on how far it got before the second
    # request arrived.  The creator does not care which; what matters is
    # that one render was asked for last and one take exists.
    assert len(app.project.vocal_renders) == takes_before + 1, \
        f"{len(app.project.vocal_renders) - takes_before} takes kept: {said}"


def test_the_newest_vocal_request_is_still_kept_and_heard(ready, settle):
    """The guard must not make the last request disappear along with the
    superseded one."""
    app = ready
    melody, by_name = _sections(app)
    pallavi = by_name["Pallavi"]
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    settle()

    asked = []
    original = app.render
    app.render = lambda **kw: asked.append(kw)
    try:
        app.preview_section(pallavi.id, autoplay=True)
        settle()
    finally:
        app.render = original

    assert any(kw.get("autoplay") for kw in asked), \
        "the only preview asked for did not play"


# --------------------------------------------------------------------------
# Play Vocal plays the current take (Krish's fifth walkthrough item)
# --------------------------------------------------------------------------
def _cache_take(app, kind, when, voice_id="", version=1):
    """Put a rendered take in the cache, as a render would have."""
    import numpy as np
    from raagacomposer.app import RenderedAudio
    from raagacomposer.core.models import VocalRender
    app._renders[kind] = RenderedAudio(
        kind=kind, audio=np.zeros(1000, dtype=np.float32),
        sample_rate=app.sample_rate, created_at=when)
    app.project.vocal_renders.append(VocalRender(
        version=version, kind="master" if kind == "vocal_master" else "preview",
        voice_profile_id=voice_id or app.current_voice().id))


def test_play_vocal_plays_the_newest_take_not_the_master(ready):
    """A master rendered an hour ago outranked a preview rendered a moment
    ago, because the window asked by kind rather than by recency."""
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    _cache_take(app, "vocal_master", when=1000.0, version=1)
    _cache_take(app, "vocal_preview", when=2000.0, version=2)

    played = []
    original = app.play_render
    app.play_render = lambda kind=None, **kw: played.append(kind) or True
    try:
        assert app.play_vocal() is True
    finally:
        app.play_render = original
    assert played == ["vocal_preview"], played
    assert "preview" in app.status_text.lower(), app.status_text


def test_play_vocal_still_prefers_a_master_when_it_is_the_newer_one(ready):
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    _cache_take(app, "vocal_preview", when=1000.0, version=1)
    _cache_take(app, "vocal_master", when=2000.0, version=2)

    played = []
    original = app.play_render
    app.play_render = lambda kind=None, **kw: played.append(kind) or True
    try:
        app.play_vocal()
    finally:
        app.play_render = original
    assert played == ["vocal_master"], played


def test_play_vocal_says_when_the_take_is_in_another_voice(ready):
    """A listening test on the wrong voice looks like an answer."""
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    other = next(v for v in app.voices.all() if v.id != app.current_voice().id)
    _cache_take(app, "vocal_preview", when=2000.0, voice_id=other.id, version=3)

    original = app.play_render
    app.play_render = lambda kind=None, **kw: True
    try:
        app.play_vocal()
    finally:
        app.play_render = original
    assert other.name in app.status_text, app.status_text
    assert "render again" in app.status_text.lower(), app.status_text


def test_play_vocal_with_nothing_rendered_says_so(ready):
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    assert app.play_vocal() is False
    assert "render a vocal" in app.status_text.lower(), app.status_text


def test_a_locked_percussion_region_is_not_replaced(ready, settle):
    """Arya: guarding the track is not guarding what is in it.

    apply_beat refused when the track was locked and then replaced its
    regions wholesale, so a lock on a region was thrown away along with
    the region.
    """
    app = ready
    app.generate_beat()
    settle()
    track = next(t for t in app.project.arrangement().tracks
                 if t.role == "rhythm")
    assert track.regions, "no percussion region to lock"
    track.regions[0].locked = True
    before = [(r.meta.get("beat_version"), r.locked, len(r.notes))
              for r in track.regions]

    said = []
    original = app.status
    app.status = lambda text, *a, **k: (said.append(text), original(text))[1]
    try:
        app.beat_variation(strength="busier")
        settle()
    finally:
        app.status = original

    after = [(r.meta.get("beat_version"), r.locked, len(r.notes))
             for r in track.regions]
    assert after == before, "a locked percussion passage was replaced"
    assert any("locked" in line.lower() for line in said), said


def test_add_beat_is_one_undo_step(ready, settle):
    """One thing the creator did should take one Undo to undo.

    The beat was committed to the undo stack when it was stored and again
    when it reached the arrangement, so a single Undo left the new beat
    selected while the song still played the old one.
    """
    app = ready
    app.generate_beat()
    settle()
    first = app.project.beat().version
    first_in_mix = _rhythm_versions(app)

    app.beat_variation(strength="busier")
    settle()
    second = app.project.beat().version
    assert second != first, "the variation produced no new version"
    assert _rhythm_versions(app) == [str(second)]

    app.undo_action()
    settle()

    assert app.project.beat().version == first, \
        f"one Undo left beat v{app.project.beat().version} selected"
    assert _rhythm_versions(app) == first_in_mix, \
        "the selected beat and the one in the mix disagree after Undo"


def test_the_full_mix_sings_the_take_play_vocal_would_play(ready, settle):
    """Arya: fixing the button and leaving the mix reading "master first".

    The creator could hear the newer take on its own and the older one
    inside the song - the wrong voice or the wrong words, in the thing they
    were actually judging.
    """
    import numpy as np
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    _cache_take(app, "vocal_master", when=1000.0, version=1)
    _cache_take(app, "vocal_preview", when=2000.0, version=2)
    # Distinguishable audio, so the assertion is about which one arrived.
    app._renders["vocal_master"].audio = np.full(1000, 0.25, dtype=np.float32)
    app._renders["vocal_preview"].audio = np.full(1000, 0.75, dtype=np.float32)

    seen = {}
    from raagacomposer.music import mixer
    original = mixer.mix

    def watch(arrangement, vocal, *a, **kw):
        seen["vocal"] = None if vocal is None else float(np.mean(vocal))
        return original(arrangement, vocal, *a, **kw)

    mixer.mix = watch
    try:
        app.render(kind="full", autoplay=False)
        settle()
    finally:
        mixer.mix = original

    assert "vocal" in seen, "the mixer was never asked for a full mix"
    assert seen["vocal"] is not None, "the mix went out with no vocal at all"
    assert abs(seen["vocal"] - 0.75) < 1e-3, \
        f"the mix used the older master (mean {seen['vocal']:.2f})"


def test_playing_a_take_leaves_its_name_on_screen(ready):
    """play_render writes its own line, so saying which take it was before
    starting playback meant the creator only ever saw "Playing vocal
    preview" and never learned whose voice it was."""
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    other = next(v for v in app.voices.all() if v.id != app.current_voice().id)
    _cache_take(app, "vocal_preview", when=2000.0, voice_id=other.id, version=4)

    assert app.play_vocal() is True
    assert other.name in app.status_text, app.status_text
    assert "render again" in app.status_text.lower(), app.status_text


def test_a_playback_failure_keeps_its_own_message(ready):
    """When playback fails, the reason it failed is the useful line."""
    app = ready
    app._renders.pop("vocal_master", None)
    app._renders.pop("vocal_preview", None)
    _cache_take(app, "vocal_preview", when=2000.0, version=5)

    original = app.play_render

    def refuse(kind=None, **kw):
        app.status("The audio device is not available.")
        return False

    app.play_render = refuse
    try:
        assert app.play_vocal() is False
    finally:
        app.play_render = original
    assert "device" in app.status_text.lower(), app.status_text


# --------------------------------------------------------------------------
# Adjustable mix: balance, room, and a comparison that changes nothing else
# (Arya's pre-Jam gap, 2026-09-07 20:27)
# --------------------------------------------------------------------------
def _mix_audio(app, settle):
    """Render the full mix and hand back what the mixer produced."""
    seen = {}
    from raagacomposer.music import mixer
    original = mixer.mix

    def watch(arrangement, vocal, sr, total, **kw):
        result = original(arrangement, vocal, sr, total, **kw)
        seen["audio"] = result.audio
        seen["kw"] = kw
        return result

    mixer.mix = watch
    try:
        app.render(kind="full", autoplay=False)
        settle()
    finally:
        mixer.mix = original
    return seen


def test_the_voice_can_be_set_against_the_instruments(ready, settle):
    """mix() took a vocal_gain and nothing ever passed one."""
    app = ready
    app.auto_arrange()
    settle()
    app.render_vocal(kind="preview", autoplay=False)
    settle()

    app.set_mix(vocal_gain=0.4)
    quiet = _mix_audio(app, settle)
    assert quiet["kw"].get("vocal_gain") == pytest.approx(0.4), quiet["kw"]

    app.set_mix(vocal_gain=1.6)
    loud = _mix_audio(app, settle)
    assert loud["kw"].get("vocal_gain") == pytest.approx(1.6)


def test_the_room_can_be_opened_and_closed(ready, settle):
    """A real difference in the rendered audio, not only in a setting."""
    import numpy as np
    app = ready
    app.auto_arrange()
    settle()

    app.set_mix(reverb=0.0)
    dry = _mix_audio(app, settle)["audio"].copy()
    app.set_mix(reverb=1.8)
    wet = _mix_audio(app, settle)["audio"].copy()

    assert dry.shape == wet.shape, "the room changed the length of the song"
    assert not np.allclose(dry, wet), "opening the room changed nothing"


def test_a_dry_comparison_changes_only_the_room(ready, settle):
    """The comparison means nothing unless everything else is held."""
    import numpy as np
    app = ready
    app.auto_arrange()
    settle()
    melody_before = [(n.midi, round(n.start, 5)) for n in
                     app.project.melody().notes]
    takes_before = len(app.project.vocal_renders)
    melodies_before = len(app.project.melodies)

    effected = _mix_audio(app, settle)["audio"].copy()
    app.compare_dry()
    settle()
    dry = app._renders["full"].audio

    assert app.project.mix_settings.effects is False
    assert dry.shape == effected.shape, "the dry version is a different length"
    assert not np.allclose(dry, effected), "nothing was taken away"
    assert [(n.midi, round(n.start, 5)) for n in
            app.project.melody().notes] == melody_before, \
        "comparing dry rewrote the tune"
    assert len(app.project.vocal_renders) == takes_before, \
        "comparing dry sang the song again"
    assert len(app.project.melodies) == melodies_before


def test_the_mix_stays_within_its_limits(ready, settle):
    """Turning everything up must not send the mix past full scale."""
    import numpy as np
    app = ready
    app.auto_arrange()
    settle()
    app.render_vocal(kind="preview", autoplay=False)
    settle()
    app.set_mix(vocal_gain=2.0, reverb=2.0, room=0.95)
    audio = _mix_audio(app, settle)["audio"]
    assert float(np.abs(audio).max()) <= 1.0, "the mix clipped"


def test_mix_settings_are_the_songs_and_come_back_with_it(ready, settle,
                                                          tmp_path):
    app = ready
    app.set_mix(vocal_gain=0.7, reverb=1.5, room=0.6)
    where = app.save()
    assert where is not None

    app.new_project("Something else", write=False)
    assert app.project.mix_settings.vocal_gain == pytest.approx(1.0), \
        "a new song inherited the last one's balance"

    app.open_project(where)
    settings = app.project.mix_settings
    assert settings.vocal_gain == pytest.approx(0.7)
    assert settings.reverb == pytest.approx(1.5)
    assert settings.room == pytest.approx(0.6)


def test_changing_the_mix_can_be_undone(ready, settle):
    app = ready
    app.set_mix(vocal_gain=0.5)
    assert app.project.mix_settings.vocal_gain == pytest.approx(0.5)
    app.undo_action()
    settle()
    assert app.project.mix_settings.vocal_gain == pytest.approx(1.0), \
        "the balance change could not be undone"


def test_a_track_can_be_placed_across_the_stereo_field(ready, settle):
    """Pan was supported by the controller and had no way in."""
    import numpy as np
    app = ready
    app.auto_arrange()
    settle()
    track = next(t for t in app.project.arrangement().tracks
                 if t.role != "drone")

    app.set_track_flag(track.id, pan=-0.9)
    left = _mix_audio(app, settle)["audio"].copy()
    app.set_track_flag(track.id, pan=0.9)
    right = _mix_audio(app, settle)["audio"].copy()

    assert not np.allclose(left, right), "moving a track across did nothing"


def test_rendering_the_same_song_twice_gives_the_same_length(ready, settle):
    """Found by a mix test that could not tell reverb from arithmetic.

    Project.duration counted the mixes, and render() asks for that plus
    half a second of tail - so every render stored a mix half a second
    longer than the last, and the next render started from that.  Measured
    at 149.86s growing to 151.86s over four renders of an unchanged song,
    the difference being silence.  A mix's length is a consequence of the
    song's length and cannot be part of its definition.
    """
    app = ready
    app.auto_arrange()
    settle()
    lengths = []
    for _ in range(3):
        app.render(kind="full", autoplay=False)
        settle()
        lengths.append(len(app._renders["full"].audio))
    assert len(set(lengths)) == 1, f"the song grew: {lengths}"


def test_a_comparison_stays_on_the_section_being_auditioned(ready, settle):
    """Krish auditions the Pallavi, then asks to hear it without the room.
    Sending him back to the top of the song is not a comparison of what he
    was listening to."""
    app = ready
    app.auto_arrange()
    settle()
    melody = app.project.melody()
    section = next(sec for sec in melody.sections if not sec.kind.instrumental)

    assert app.audition_scope() == "the whole song"
    app.preview_section(section.id, autoplay=True)
    settle()
    assert app.audition_scope() == section.name, \
        f"the app forgot what was being heard: {app.audition_scope()}"

    played = []
    original = app.play_render
    app.play_render = lambda kind, span=None: played.append((kind, span))
    try:
        app.compare_dry()
        # Read this now: the render sets its own status again when it
        # finishes, and by then this one is gone.
        asked = app.status_text
        settle()
    finally:
        app.play_render = original

    assert section.name in asked, asked
    assert played, "the comparison played nothing"
    kind, span = played[-1]
    assert span is not None, "the comparison jumped back to the whole song"
    assert abs(span[0] - section.start) < 0.01, span
    assert abs(span[1] - section.end) < 0.01, span


def test_hearing_the_whole_song_widens_the_comparison_again(ready, settle):
    """The scope is whatever was last actually heard, not a mode to escape."""
    app = ready
    app.auto_arrange()
    settle()
    melody = app.project.melody()
    section = next(sec for sec in melody.sections if not sec.kind.instrumental)
    app.preview_section(section.id, autoplay=True)
    settle()
    assert app.audition_scope() == section.name

    app.render(kind="full", autoplay=True)
    settle()
    assert app.audition_scope() == "the whole song", \
        "playing the whole song left the comparison on one section"


def test_the_scope_follows_what_was_actually_heard(ready, settle):
    """Arya's three failures, one cause: the scope was recorded when a
    render finished rather than when something was played, so playing a
    render already in hand - the ordinary case - never reached it."""
    app = ready
    app.auto_arrange()
    settle()
    melody = app.project.melody()
    section = next(sec for sec in melody.sections if not sec.kind.instrumental)

    app.render(kind="full", autoplay=False)
    settle()
    assert "full" in app._renders, "nothing to play"

    # Cached playback of a section: no render runs at all.
    assert app.play_render("full", (section.start, section.end))
    assert app.audition_scope() == section.name, \
        f"playing a section did not become the scope: {app.audition_scope()}"

    # Cached playback of the whole song widens it again.
    assert app.play_render("full")
    assert app.audition_scope() == "the whole song", \
        "playing the whole song left the comparison on one section"


def test_a_playback_that_failed_leaves_the_scope_alone(ready, settle):
    """Nothing was heard, so the creator is still listening to whatever
    they were listening to before."""
    app = ready
    app.auto_arrange()
    settle()
    melody = app.project.melody()
    section = next(sec for sec in melody.sections if not sec.kind.instrumental)
    app.render(kind="full", autoplay=False)
    settle()

    assert app.play_render("full", (section.start, section.end))
    assert app.audition_scope() == section.name

    original = app.playback.play
    app.playback.play = lambda *a, **k: False
    try:
        assert not app.play_render("full")
    finally:
        app.playback.play = original

    assert app.audition_scope() == section.name, \
        "a playback nobody heard moved the comparison"
