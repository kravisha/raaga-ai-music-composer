"""Additional independent preview ownership cases for a completed callback queue."""
import threading
import time

import numpy as np


def finish(app):
    end = time.monotonic() + 8
    while time.monotonic() < end:
        app.pump()
        if not app.jobs.active_jobs():
            app.pump()
            return
        time.sleep(0.01)
    raise AssertionError("review job did not settle")


def silent_voice(melody, lyrics, profile, direction, sample_rate, duration=None, **kwargs):
    return np.zeros(int(sample_rate * 0.2), dtype=np.float32)


def prepare(app, monkeypatch):
    monkeypatch.setattr(app.providers.voice, "render_vocal", silent_voice)
    monkeypatch.setattr(app, "render", lambda *a, **kw: None)
    app.new_project("Preview ownership", write=False)
    app.update_brief(duration_target=45)
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=8)
    finish(app)
    pallavi = next(s for s in app.project.melody().sections if s.name == "Pallavi")
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    finish(app)
    requests = []
    monkeypatch.setattr(app, "render", lambda *a, **kw: requests.append(kw))
    return pallavi, requests


def test_current_section_preview_keeps_correct_play_range(app, monkeypatch):
    section, requests = prepare(app, monkeypatch)
    app.preview_section(section.id, autoplay=True)
    finish(app)
    assert requests == [dict(kind="full", autoplay=True,
                            play_range=(section.start, section.end))]


def test_new_silent_render_supersedes_a_still_running_preview(app, monkeypatch):
    section, requests = prepare(app, monkeypatch)
    started, release = threading.Event(), threading.Event()
    calls = 0
    def held_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(5)
        return silent_voice(*args, **kwargs)
    monkeypatch.setattr(app.providers.voice, "render_vocal", held_first)
    app.preview_section(section.id, autoplay=True)
    assert started.wait(3)
    try:
        app.render_vocal("preview", autoplay=False)
    finally:
        release.set()
    finish(app)
    assert not requests, "superseded running preview retained playback intent"


def test_new_silent_render_supersedes_an_already_queued_preview_callback(app, monkeypatch):
    section, requests = prepare(app, monkeypatch)
    app.preview_section(section.id, autoplay=True)
    # The worker can finish between GUI timer ticks. Leave its callback queued,
    # then accept a new request before the next GUI pump drains that callback.
    end = time.monotonic() + 5
    while app.jobs.active_jobs() and time.monotonic() < end:
        time.sleep(0.005)
    assert not app.jobs.active_jobs(), "vocal worker never completed"
    assert not requests, "callback ran without pumping the controller"
    app.render_vocal("preview", autoplay=False)
    finish(app)
    assert not requests, "completed but superseded preview callback still starts a full mix with autoplay=True"
