"""Compare real Play Vocal dispatch and full-mix input after a newer preview."""
from types import SimpleNamespace

import numpy as np
import pytest

from raagacomposer.app import RenderedAudio
from raagacomposer.core.models import VocalRender


@pytest.fixture
def cached_takes(app, settle, monkeypatch):
    real_render = app.render
    monkeypatch.setattr(app, "render", lambda *a, **kw: None)
    app.new_project("Vocal cache consistency", write=False)
    app.update_brief(duration_target=20)
    app.select_raaga("Hamsadhwani")
    app.generate_tune(seed=4)
    settle()
    singer = app.current_voice()
    for kind, when, value, version in (("vocal_master", 1000.0, 0.1, 1),
                                       ("vocal_preview", 2000.0, 0.2, 2)):
        app._renders[kind] = RenderedAudio(kind=kind,
            audio=np.full(1000, value, dtype=np.float32), sample_rate=app.sample_rate,
            created_at=when)
        app.project.vocal_renders.append(VocalRender(version=version,
            kind="master" if kind == "vocal_master" else "preview",
            melody_version=app.project.melody().version,
            voice_profile_id=singer.id))
    # Keep the real controller/play_render path; replace only device playback.
    monkeypatch.setattr(app.playback, "play", lambda *a, **kw: True)
    return app, real_render, singer


def test_real_play_dispatch_uses_newer_preview(cached_takes):
    app, _, _ = cached_takes
    assert app.play_vocal()
    assert app.playback.source_name == "vocal_preview"
    assert app._loaded_render is app._renders["vocal_preview"]


def test_visible_play_status_keeps_take_and_singer(cached_takes):
    app, _, singer = cached_takes
    assert app.play_vocal()
    assert singer.name in app.status_text and "v2" in app.status_text, (
        "The real play_render path overwrote the take and singer explanation: " + app.status_text)


def test_full_mix_uses_same_current_take_as_play_vocal(cached_takes, settle, monkeypatch):
    from raagacomposer import app as app_module
    app, real_render, _ = cached_takes
    expected = app._renders["vocal_preview"].audio
    calls = []
    def capture(arrangement, vocal_audio, sr, total, **kwargs):
        calls.append(vocal_audio)
        return SimpleNamespace(audio=np.zeros((32, 2), dtype=np.float32),
            notes=[], loudness_db=-20, track_count=0, summary=lambda: "silent mix observation")
    monkeypatch.setattr(app_module.mixer, "mix", capture)
    assert app.current_vocal_take()[0] == "vocal_preview"
    real_render("full", autoplay=False)
    settle()
    assert calls
    assert calls[-1] is expected, "Full mix still uses the older master while Play Vocal selects the newer preview"
