"""Review-only probes for the selected-section workflow and request ownership."""
import threading
import time

from raagacomposer.speech.intent import Command


def pump_until(app, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.pump()
        if predicate():
            app.pump()
            return
        time.sleep(0.01)
    raise AssertionError("review operation did not settle")


def finish(app):
    pump_until(app, lambda: not app.jobs.active_jobs() and app._interpreting is None)


def prepare_song(app):
    app.new_project("Original", write=False)
    app.update_brief(duration_target=45.0)
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=8)
    finish(app)
    return next(s for s in app.project.melody().sections if s.name == "Pallavi")


def hold_lyrics(app, monkeypatch, section):
    import raagacomposer.app as module
    real_generate = module.lyric_generator.generate
    started = threading.Event()
    release = threading.Event()

    def held(*args, **kwargs):
        result = real_generate(*args, **kwargs)
        started.set()
        assert release.wait(5), "review did not release lyric generation"
        return result

    monkeypatch.setattr(module.lyric_generator, "generate", held)
    app.generate_lyrics(seed=4, section_ids=[section.id])
    assert started.wait(3), "lyric generation did not start"
    return release


def test_selected_lyrics_cannot_land_in_a_replacement_song(app, monkeypatch):
    pallavi = prepare_song(app)
    release = hold_lyrics(app, monkeypatch, pallavi)
    try:
        app.new_project("Replacement", write=False)
    finally:
        release.set()
    finish(app)
    assert not app.project.lyrics, "Pallavi lyrics from Original were appended to the empty Replacement song"


def test_locking_selected_section_rejects_late_lyrics(app, monkeypatch):
    pallavi = prepare_song(app)
    release = hold_lyrics(app, monkeypatch, pallavi)
    before = len(app.project.lyrics)
    try:
        app.set_section_lock(pallavi.id, True)
    finally:
        release.set()
    finish(app)
    assert pallavi.locked
    assert len(app.project.lyrics) == before, "lyrics were committed after their target section was locked"


def test_old_cancel_callback_cannot_supersede_a_new_song_request(app, monkeypatch):
    first_started, first_release = threading.Event(), threading.Event()
    second_started, second_release = threading.Event(), threading.Event()
    first_text, second_text, third_text = "old request", "set raga Hamsadhwani", "a follow-up question"

    def held(text, *args, **kwargs):
        if text == first_text:
            first_started.set()
            assert first_release.wait(5)
            return Command(intent="unknown", text=text)
        if text == second_text:
            second_started.set()
            assert second_release.wait(5)
            return Command(intent="raaga.set", text=text, raaga="Hamsadhwani", confidence=1.0)
        return Command(intent="unknown", text=text)

    monkeypatch.setattr("raagacomposer.app.interpret", held)
    app.new_project("Original", write=False)
    app.say(first_text)
    app._drain_utterances()
    assert first_started.wait(3)
    old_job = app.jobs.jobs()[-1]
    app.new_project("Replacement", write=False)
    app.select_raaga("Keeravani")
    app.say(second_text)
    app._drain_utterances()
    assert second_started.wait(3)
    new_job = app.jobs.jobs()[-1]
    try:
        first_release.set()
        deadline = time.monotonic() + 3
        while old_job.active and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not old_job.active
        app.jobs.drain()
        app.say(third_text)
        app.pump()
        cancelled = new_job.cancel_event.is_set()
    finally:
        first_release.set()
        second_release.set()
    finish(app)
    assert not cancelled, "an old song's cancellation callback cleared the current request flag, letting the next request cancel it"
    assert app.project.raaga.selected == "Hamsadhwani", "the valid new-song typed request was lost"


def test_open_project_discards_old_queued_input(app):
    app.new_project("Saved replacement")
    app.select_raaga("Keeravani")
    app.save()
    replacement = app.project_dir
    app.new_project("Original", write=False)
    app.say("set raga Hamsadhwani")
    app.open_project(replacement)
    finish(app)
    assert app.project.raaga.selected == "Keeravani"


def test_open_project_discards_old_inflight_input(app, monkeypatch):
    app.new_project("Saved replacement")
    app.select_raaga("Keeravani")
    app.save()
    replacement = app.project_dir
    app.new_project("Original", write=False)
    started, release = threading.Event(), threading.Event()

    def held(text, *args, **kwargs):
        started.set()
        assert release.wait(5)
        return Command(intent="raaga.set", text=text, raaga="Hamsadhwani", confidence=1.0)

    monkeypatch.setattr("raagacomposer.app.interpret", held)
    app.say("set raga Hamsadhwani")
    app._drain_utterances()
    assert started.wait(3)
    try:
        app.open_project(replacement)
    finally:
        release.set()
    finish(app)
    assert app.project.raaga.selected == "Keeravani"


def test_late_charanam_draft_preserves_new_pallavi_edit(app, monkeypatch):
    pallavi = prepare_song(app)
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    finish(app)
    line = next(l for l in app.project.lyrics_version().lines
                if l.section_id == pallavi.id and l.text)
    charanam = next(s for s in app.project.melody().sections if s.name == "Charanam 1")
    release = hold_lyrics(app, monkeypatch, charanam)
    edited = "nenjam paadum puthiya varigal"
    try:
        app.edit_lyric_line(line.id, edited)
    finally:
        release.set()
    finish(app)
    current = [l.text for l in app.project.lyrics_version().lines if l.section_id == pallavi.id]
    assert edited in current, "a late Charanam draft erased a newer edit to the unselected Pallavi"


def test_late_vocal_cannot_land_in_a_replacement_song(app, monkeypatch):
    import numpy as np
    pallavi = prepare_song(app)
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    finish(app)
    started, release = threading.Event(), threading.Event()

    def held(melody, lyrics, profile, direction, sample_rate, duration, **kwargs):
        started.set()
        assert release.wait(5)
        return np.zeros(int(sample_rate * 0.2), dtype=np.float32)

    monkeypatch.setattr(app.providers.voice, "render_vocal", held)
    app.render_vocal("preview", autoplay=False)
    assert started.wait(3)
    try:
        app.new_project("Replacement for vocal", write=False)
    finally:
        release.set()
    finish(app)
    assert not app.project.vocal_renders, "the previous song's vocal take was appended to the replacement song"
