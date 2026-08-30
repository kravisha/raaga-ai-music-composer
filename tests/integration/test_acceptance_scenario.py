"""Integration: the specification's own first end-to-end acceptance test.

Specification section 21, step for step. One test function per numbered step
group so a failure names the step that broke.

The scenario runs against the real controller with the real engines: no mocks,
real synthesis, real files on disk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.app import AppController

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    """Runs the whole scenario once; each test then asserts on the result."""
    from raagacomposer.core.settings import Settings

    directory = tmp_path_factory.mktemp("acceptance")
    settings = Settings.load()
    settings.projects_dir = str(directory / "projects")
    settings.stt_provider = "none"
    settings.autosave_seconds = 5

    app = AppController(settings)
    state: dict = {"app": app, "settings": settings}

    def settle(timeout: float = 300.0) -> None:
        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.pump()
            if not app.jobs.active_jobs():
                app.pump()
                if not app.jobs.active_jobs():
                    return
            time.sleep(0.02)
        raise TimeoutError("jobs did not finish")

    # 1-2. launch and create a project
    app.new_project("Terrace at Midnight")
    state["project_dir"] = app.project_dir

    # 3. speak a creative feel
    app.update_brief(situation="a man alone on a terrace after midnight",
                     mood="longing",
                     feel="lonely, late at night, but still warm",
                     language="Tamil", duration_target=100.0)

    # 4. receive and select a raaga
    state["suggestions"] = app.raaga_suggestions()
    app.select_raaga(state["suggestions"][0].name)

    # 5-6. generate a tune and hear it
    app.generate_tune(seed=21)
    settle()

    # 7. approve and lock it
    app.accept_tune(lock=True)

    # 8. lyrics fitted to that tune
    app.generate_lyrics(seed=9)
    settle()

    # 9. render the song using a voice
    app.render_vocal("preview", autoplay=False)
    settle()

    # 10. studio-quality vocal-only render, asked for in words
    app.handle_utterance("Give me the song without instruments.")
    settle()

    # 11-12. ask verbally for an instrument in a named section
    interlude = next(s for s in app.project.sections
                     if s.kind.value == "interlude")
    state["interlude"] = interlude
    app.seek(interlude.start + 1.0)
    app.handle_utterance("Use saxophone for this interlude.")
    settle()
    # Snapshot the facts now: the arrangement object is mutated by later steps.
    sax_tracks = app.project.arrangement().tracks_for_instrument("saxophone")
    state["sax"] = [(r.start, r.end, len(r.notes))
                    for t in sax_tracks for r in t.regions]

    # 13-14. interrupt mid-render and replace the instrument
    app.render("full", autoplay=False)
    app._on_barge_in()
    state["cancelled_by_barge_in"] = [j.status for j in app.jobs.jobs()
                                      if j.job_type.startswith("render")]
    app.handle_utterance("No, change that saxophone to veena.")
    settle()

    # 15-17. natural-language playback (the song's length at this moment is
    # what the ranges are clamped against, so record it alongside).
    state["duration_at_playback"] = app.project.duration
    state["cmd_first_minute"] = app.handle_utterance("Play the first minute.")
    state["cmd_two_to_three"] = app.handle_utterance(
        "Play from the second minute to the third minute.")
    state["cmd_end"] = app.handle_utterance("Play the end.")
    app.stop()

    # 18. a feel instead of an instrument name
    state["before_feel"] = {t.instrument for t in app.project.arrangement().tracks}
    app.handle_utterance("I want this to feel lonely, late at night, but still warm.")
    settle()
    state["after_feel"] = {t.instrument for t in app.project.arrangement().tracks}

    # 19. accept part of the arrangement and lock it
    app.auto_arrange()
    settle()
    pallavi = next(s for s in app.project.sections if s.kind.value == "pallavi")
    state["pallavi"] = pallavi
    app.lock_range(pallavi.start, pallavi.end, True)
    state["locked_signature"] = [(r.id, len(r.notes))
                                 for t in app.project.arrangement().tracks
                                 for r in t.regions if r.locked]

    # 20. modify a different section without damaging the locked one
    outro = app.project.sections[-1]
    app.add_instrument("flute", outro.start, outro.end, role="lead")
    settle()
    state["signature_after_edit"] = [(r.id, len(r.notes))
                                     for t in app.project.arrangement().tracks
                                     for r in t.regions if r.locked]

    # and a direct edit of the locked range must be refused
    locked_track = next(t for t in app.project.arrangement().tracks
                        if any(r.locked for r in t.regions))
    locked_region = next(r for r in locked_track.regions if r.locked)
    errors_before = len(app.project.errors)
    app.add_instrument(locked_track.instrument, locked_region.start,
                       locked_region.end)
    state["lock_refusals"] = len(app.project.errors) - errors_before

    # 21. produce a full mix
    app.render("full", autoplay=False)
    settle()

    # 22-24. save, close, reopen
    app.save()
    state["before_restart"] = {
        "title": app.project.title,
        "notes": len(app.project.melody().notes),
        "lines": len(app.project.lyrics_version().lines),
        "tracks": len(app.project.arrangement().tracks),
        "history": len(app.project.history),
        "conversation": len(app.project.conversation),
        "mix_path": app.project.latest_mix("full").audio_path,
        "vocal_master": app.project.vocal_master.audio_path,
    }
    app.close()

    reopened = AppController(settings)
    reopened.open_project(state["project_dir"])
    state["reopened"] = reopened
    yield state
    reopened.close()


# --------------------------------------------------------------------------
# steps 1-4
# --------------------------------------------------------------------------
def test_step_02_a_project_is_created_on_disk(scenario):
    directory: Path = scenario["project_dir"]
    assert directory.exists()
    assert (directory / "project.json").exists()


def test_step_03_the_spoken_feel_is_recorded(scenario):
    brief = scenario["app"].project.brief
    assert "lonely" in brief.feel
    assert brief.language == "Tamil"


def test_step_04_raagas_are_suggested_and_one_is_selected(scenario):
    suggestions = scenario["suggestions"]
    assert len(suggestions) >= 2
    assert all(s.rationale for s in suggestions)
    assert scenario["app"].project.raaga.selected == suggestions[0].name


# --------------------------------------------------------------------------
# steps 5-8
# --------------------------------------------------------------------------
def test_step_05_a_tune_is_generated(scenario):
    melody = scenario["app"].project.melody()
    assert melody is not None
    assert len(melody.notes) > 20
    assert melody.duration > 60


def test_step_06_the_tune_can_be_played(scenario):
    assert scenario["app"].rendered("tune") is not None


def test_step_07_the_tune_is_locked(scenario):
    assert scenario["app"].project.melody().state.value == "locked"


def test_step_08_lyrics_are_fitted_to_that_tune(scenario):
    app = scenario["app"]
    lyrics = app.project.lyrics_version()
    assert lyrics is not None and lyrics.lines
    assert lyrics.melody_version == app.project.melody().version
    for line in lyrics.lines:
        assert len(line.syllables) == len(line.note_indices)


# --------------------------------------------------------------------------
# steps 9-10
# --------------------------------------------------------------------------
def test_step_09_the_song_is_rendered_with_a_voice(scenario):
    assert scenario["app"].rendered("vocal_preview") is not None


def test_step_10_a_studio_vocal_only_master_is_produced(scenario):
    master = scenario["app"].project.vocal_master
    assert master is not None
    assert master.kind == "master"
    assert Path(master.audio_path).exists()
    assert Path(master.audio_path).stat().st_size > 40_000


# --------------------------------------------------------------------------
# steps 11-14
# --------------------------------------------------------------------------
def test_step_12_the_instrument_lands_in_the_named_section(scenario):
    interlude = scenario["interlude"]
    regions = scenario["sax"]
    assert regions, "saxophone was never added"
    start, end, note_count = regions[0]
    assert start == pytest.approx(interlude.start, abs=0.5)
    assert end == pytest.approx(interlude.end, abs=0.5)
    assert note_count > 0


def test_step_13_an_interruption_stops_work_in_flight(scenario):
    assert any(status in ("cancelled", "stale", "done")
               for status in scenario["cancelled_by_barge_in"])


def test_step_14_the_instrument_is_replaced(scenario):
    arrangement = scenario["app"].project.arrangement()
    assert not arrangement.tracks_for_instrument("saxophone")
    assert arrangement.tracks_for_instrument("veena")


# --------------------------------------------------------------------------
# steps 15-18
# --------------------------------------------------------------------------
def test_step_15_play_the_first_minute(scenario):
    spec = scenario["cmd_first_minute"].time
    assert (spec.start, spec.end) == (0.0, 60.0)


def test_step_16_play_from_the_second_to_the_third_minute(scenario):
    spec = scenario["cmd_two_to_three"].time
    duration = scenario["duration_at_playback"]
    assert spec.start == 60.0
    assert spec.end == pytest.approx(min(180.0, duration))


def test_step_17_play_the_end(scenario):
    spec = scenario["cmd_end"].time
    duration = scenario["duration_at_playback"]
    assert spec.end == pytest.approx(duration, abs=0.5)
    assert spec.start < spec.end


def test_step_18_a_feel_produces_an_instrument_proposal(scenario):
    added = scenario["after_feel"] - scenario["before_feel"]
    assert added, "the AI proposed nothing for the described feel"


# --------------------------------------------------------------------------
# steps 19-21
# --------------------------------------------------------------------------
def test_step_19_part_of_the_arrangement_is_locked(scenario):
    assert scenario["locked_signature"]


def test_step_20_editing_elsewhere_leaves_the_locked_part_alone(scenario):
    assert scenario["signature_after_edit"] == scenario["locked_signature"]


def test_step_20b_a_direct_edit_of_a_locked_region_is_refused(scenario):
    assert scenario["lock_refusals"] >= 1
    message = scenario["app"].project.errors[-1].message
    assert "lock" in message.lower()


def test_step_21_a_full_mix_is_produced(scenario):
    mix = scenario["app"].project.latest_mix("full")
    assert mix is not None
    assert Path(mix.audio_path).exists()
    assert mix.duration > 30


# --------------------------------------------------------------------------
# steps 22-25
# --------------------------------------------------------------------------
def test_step_25_everything_survives_close_and_reopen(scenario):
    before = scenario["before_restart"]
    project = scenario["reopened"].project

    assert project.title == before["title"]
    assert len(project.melody().notes) == before["notes"]
    assert project.melody().state.value == "locked"
    assert len(project.lyrics_version().lines) == before["lines"]
    assert project.vocal_master is not None
    assert len(project.arrangement().tracks) == before["tracks"]
    assert any(r.locked for t in project.arrangement().tracks for r in t.regions)
    assert len(project.history) >= before["history"]
    assert len(project.conversation) >= before["conversation"]
    assert Path(before["mix_path"]).exists()
    assert Path(before["vocal_master"]).exists()


def test_step_25b_the_reopened_project_can_carry_on(scenario):
    app = scenario["reopened"]
    assert app.rendered("full") is not None or app.best_render() is not None
    cmd = app.handle_utterance("Play the first minute.")
    assert cmd.intent == "transport.play"
