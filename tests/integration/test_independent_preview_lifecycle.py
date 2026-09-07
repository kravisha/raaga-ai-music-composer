"""User-visible playback intent must not leak from a failed/abandoned preview."""
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


def song(app, title):
    app.new_project(title, write=False)
    app.update_brief(duration_target=45)
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=8)
    finish(app)
    pallavi = next(s for s in app.project.melody().sections if s.name == "Pallavi")
    app.generate_lyrics(seed=3, section_ids=[pallavi.id])
    finish(app)
    return pallavi


def silent_voice(melody, lyrics, profile, direction, sample_rate, duration=None, **kwargs):
    return np.zeros(int(sample_rate * 0.2), dtype=np.float32)


def test_failed_section_preview_does_not_play_a_later_silent_render(app, monkeypatch):
    pallavi = song(app, "First")
    requests = []
    monkeypatch.setattr(app, "render", lambda *a, **kw: requests.append(kw))
    def failed(*args, **kwargs):
        raise RuntimeError("synthetic vocal failure")
    monkeypatch.setattr(app.providers.voice, "render_vocal", failed)
    app.preview_section(pallavi.id, autoplay=True)
    finish(app)
    assert not requests
    monkeypatch.setattr(app.providers.voice, "render_vocal", silent_voice)
    app.render_vocal("preview", autoplay=False)
    finish(app)
    assert not requests, "a failed earlier section preview made a later silent Render Vocal start a full mix and play"


def test_switching_songs_drops_an_abandoned_section_preview(app, monkeypatch):
    pallavi = song(app, "First")
    started, release = threading.Event(), threading.Event()
    def held(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return silent_voice(*args, **kwargs)
    monkeypatch.setattr(app.providers.voice, "render_vocal", held)
    app.preview_section(pallavi.id, autoplay=True)
    assert started.wait(3)
    try:
        app.new_project("Replacement", write=False)
    finally:
        release.set()
    finish(app)
    monkeypatch.setattr(app.providers.voice, "render_vocal", silent_voice)
    song(app, "Replacement")
    requests = []
    monkeypatch.setattr(app, "render", lambda *a, **kw: requests.append(kw))
    monkeypatch.setattr(app.providers.voice, "render_vocal", silent_voice)
    app.render_vocal("preview", autoplay=False)
    finish(app)
    assert not requests, "the previous song's preview request triggered playback while silently rendering the replacement song"
