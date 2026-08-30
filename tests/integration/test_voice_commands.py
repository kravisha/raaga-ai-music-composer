"""Integration: spoken instructions driving real changes to the project.

Every instruction here goes through the same path a microphone utterance takes:
interpret -> resolve against conversation state -> execute -> record the turn.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def song(app, settle):
    app.new_project("Voice Control")
    app.update_brief(mood="longing", feel="lonely, late at night, but still warm",
                     language="Tamil", duration_target=45.0)
    app.select_raaga("Keeravani")
    app.generate_tune(seed=4)
    settle()
    app.accept_tune(lock=True)
    return app


def say(app, settle, text):
    cmd = app.handle_utterance(text)
    settle()
    return cmd


# --------------------------------------------------------------------------
# arrangement by voice
# --------------------------------------------------------------------------
def test_add_an_instrument_at_the_playhead(song, settle):
    app = song
    section = app.project.sections[1]
    app.seek(section.start + 1.0)
    cmd = say(app, settle, "Add veena here.")
    assert cmd.intent == "arrange.add"
    tracks = app.project.arrangement().tracks_for_instrument("veena")
    assert tracks
    region = tracks[0].regions[0]
    # "here" starts at the playhead and runs to the end of that section.
    assert region.start == pytest.approx(section.start + 1.0, abs=0.5)
    assert region.end == pytest.approx(section.end, abs=0.5)


def test_use_an_instrument_for_a_named_section(song, settle):
    app = song
    interlude = next(s for s in app.project.sections if s.kind.value == "interlude")
    say(app, settle, "Use saxophone for this interlude.")
    region = app.project.arrangement().tracks_for_instrument("saxophone")[0].regions[0]
    assert region.start == pytest.approx(interlude.start, abs=0.5)
    assert region.end == pytest.approx(interlude.end, abs=0.5)


def test_only_one_instrument_for_the_first_fifteen_seconds(song, settle):
    app = song
    say(app, settle, "Use only piano for the first 15 seconds.")
    region = app.project.arrangement().tracks_for_instrument("piano")[0].regions[0]
    assert (region.start, region.end) == (0.0, 15.0)


def test_take_an_instrument_out(song, settle):
    app = song
    say(app, settle, "Add mridangam.")
    assert app.project.arrangement().tracks_for_instrument("mridangam")
    say(app, settle, "Take the mridangam out.")
    assert not app.project.arrangement().tracks_for_instrument("mridangam")


def test_replace_one_instrument_with_another(song, settle):
    app = song
    say(app, settle, "Add violin.")
    say(app, settle, "Replace violin with veena.")
    arrangement = app.project.arrangement()
    assert not arrangement.tracks_for_instrument("violin")
    assert arrangement.tracks_for_instrument("veena")


def test_a_correction_mid_flow_supersedes_the_previous_choice(song, settle):
    app = song
    say(app, settle, "Use saxophone for this interlude.")
    say(app, settle, "No, change that saxophone to veena.")
    arrangement = app.project.arrangement()
    assert not arrangement.tracks_for_instrument("saxophone")
    assert arrangement.tracks_for_instrument("veena")


def test_make_this_part_lighter_lowers_the_level(song, settle):
    app = song
    say(app, settle, "Add veena.")
    track = app.project.arrangement().tracks_for_instrument("veena")[0]
    before = track.gain
    app.set_selection(0.0, app.project.duration)
    say(app, settle, "Make this part lighter.")
    after = app.project.arrangement().tracks_for_instrument("veena")[0]
    assert after.gain < before or any(r.gain < 1.0 for r in after.regions)


def test_a_described_feel_produces_an_instrument(song, settle):
    app = song
    before = set()
    cmd = say(app, settle,
              "I want this to feel lonely, late at night, but still warm.")
    assert cmd.intent == "arrange.suggest"
    after = {t.instrument for t in app.project.arrangement().tracks}
    assert after - before


def test_an_unavailable_instrument_is_named_not_substituted(song, settle):
    app = song
    before = len(app.project.errors)
    cmd = app.handle_utterance("Add a theremin here.")
    assert cmd.intent in ("unknown", "arrange.add")
    assert len(app.project.errors) > before
    assert "theremin" in app.project.errors[-1].message
    assert not app.project.arrangement() or \
        not any(t.instrument == "theremin"
                for t in app.project.arrangement().tracks)


# --------------------------------------------------------------------------
# playback by voice
# --------------------------------------------------------------------------
@pytest.mark.parametrize("text,start,end", [
    ("Play the first minute.", 0.0, 60.0),
    ("Play the last 30 seconds.", None, None),
    ("Play from the chorus.", None, None),
])
def test_playback_commands_are_understood(song, text, start, end):
    cmd = song.handle_utterance(text)
    assert cmd.intent == "transport.play"
    assert cmd.time is not None
    if start is not None:
        assert cmd.time.start == pytest.approx(start)


def test_the_playhead_moves_on_a_seek_command(song):
    app = song
    app.seek(30.0)
    app.handle_utterance("Go back 10 seconds.")
    assert app.playhead == pytest.approx(20.0, abs=0.5)


def test_stop_and_pause_are_understood(song):
    assert song.handle_utterance("Stop.").intent == "transport.stop"
    assert song.handle_utterance("Pause.").intent == "transport.pause"


# --------------------------------------------------------------------------
# composition by voice
# --------------------------------------------------------------------------
def test_ask_for_the_vocal_only_master(song, settle):
    app = song
    say(app, settle, "Give me the song without instruments.")
    assert app.project.vocal_master is not None
    assert app.rendered("vocal_master") is not None


def test_ask_for_lyrics_and_a_mix(song, settle):
    app = song
    say(app, settle, "Write the lyrics.")
    assert app.project.lyrics_version() is not None
    say(app, settle, "Mix the song.")
    assert app.project.latest_mix("full") is not None


def test_change_the_raaga_by_name(song, settle):
    app = song
    say(app, settle, "Use raaga Kalyani.")
    assert app.project.raaga.selected == "Kalyani"


def test_set_the_tempo_by_voice(song, settle):
    app = song
    say(app, settle, "Set the tempo to 96 bpm.")
    assert app.project.melody().tempo_bpm == 96


def test_lock_a_named_section_by_voice(song, settle):
    app = song
    pallavi = next(s for s in app.project.sections if s.kind.value == "pallavi")
    say(app, settle, "Add veena.")
    say(app, settle, "Lock the pallavi.")
    locked = [r for t in app.project.arrangement().tracks
              for r in t.regions if r.locked]
    assert locked
    assert any(s.locked for s in app.project.sections if s.id == pallavi.id)


def test_undo_by_voice(song, settle):
    app = song
    say(app, settle, "Add veena.")
    assert app.project.arrangement().tracks_for_instrument("veena")
    say(app, settle, "Undo.")
    arrangement = app.project.arrangement()
    assert arrangement is None or not arrangement.tracks_for_instrument("veena")


def test_cancel_stops_work_in_flight(song):
    app = song
    app.render("full", autoplay=False)
    app.handle_utterance("Cancel.")
    for _ in range(200):
        app.pump()
        if not app.jobs.active_jobs():
            break
    assert not app.jobs.active_jobs()


# --------------------------------------------------------------------------
# conversation bookkeeping
# --------------------------------------------------------------------------
def test_each_turn_is_recorded_with_its_outcome(song, settle):
    app = song
    say(app, settle, "Add veena.")
    turn = app.project.conversation[-1]
    assert turn.speaker == "creator"
    assert turn.intent == "arrange.add"
    assert turn.status == "applied"
    assert turn.interpretation


def test_an_unknown_instruction_is_marked_ignored(song):
    app = song
    app.handle_utterance("the quick brown fox jumps")
    assert app.project.conversation[-1].status == "ignored"


def test_a_refused_instruction_is_marked_failed(song, settle):
    app = song
    say(app, settle, "Add veena.")
    say(app, settle, "Lock the pallavi.")
    pallavi = next(s for s in app.project.sections if s.kind.value == "pallavi")
    app.set_selection(pallavi.start, pallavi.end)
    app.handle_utterance("Add veena here.")
    statuses = [t.status for t in app.project.conversation]
    assert "failed" in statuses or app.project.errors


def test_context_carries_between_utterances(song, settle):
    app = song
    say(app, settle, "Add violin.")
    assert app.context.last_instrument == "violin"
    say(app, settle, "Make that softer.")
    assert app.context.last_instrument == "violin"


def test_typed_input_uses_the_same_pipeline(song, settle):
    app = song
    app.voice_input.submit_text("Add veena.")
    settle()
    assert app.project.arrangement().tracks_for_instrument("veena")
    assert app.project.conversation[-1].text == "Add veena."
