"""Integration: reliability - restart, recovery and no silent data loss.

Specification section 18. Accepted work must survive a restart, a crash
mid-save, a provider failure and a missing audio device.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from raagacomposer.app import AppController
from raagacomposer.core.models import ApprovalState

pytestmark = pytest.mark.integration


def pump_until_idle(controller, timeout: float = 120.0) -> None:
    """Drain jobs on this thread until the controller has nothing running."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        controller.pump()
        if not controller.jobs.active_jobs():
            controller.pump()
            if not controller.jobs.active_jobs():
                return
        time.sleep(0.02)
    raise TimeoutError("jobs did not finish")


def build(app, settle, title="Recovery"):
    app.new_project(title)
    app.update_brief(mood="longing", feel="lonely night", language="Tamil",
                     duration_target=40.0)
    app.select_raaga("Keeravani")
    app.generate_tune(seed=6)
    settle()
    app.accept_tune(lock=True)
    app.generate_lyrics(seed=1)
    settle()
    app.render_vocal("master", autoplay=False)
    settle()
    app.add_instrument("veena", 0.0, app.project.duration, role="lead")
    settle()
    app.save()
    return app.project_dir


def test_everything_accepted_survives_a_restart(app, settle, settings):
    directory = build(app, settle)
    before = {
        "raaga": app.project.raaga.selected,
        "notes": [(n.start, n.midi, n.swara) for n in app.project.melody().notes],
        "lines": [l.text for l in app.project.lyrics_version().lines],
        "master": app.project.vocal_master.audio_path,
        "tracks": [t.instrument for t in app.project.arrangement().tracks],
        "history": len(app.project.history),
    }
    app.close()

    reopened = AppController(settings)
    try:
        reopened.open_project(directory)
        project = reopened.project
        assert project.raaga.selected == before["raaga"]
        assert [(n.start, n.midi, n.swara)
                for n in project.melody().notes] == before["notes"]
        assert project.melody().state is ApprovalState.LOCKED
        assert [l.text for l in project.lyrics_version().lines] == before["lines"]
        assert project.vocal_master.audio_path == before["master"]
        assert [t.instrument
                for t in project.arrangement().tracks] == before["tracks"]
        assert len(project.history) >= before["history"]
    finally:
        reopened.close()


def test_rendered_audio_is_reloaded_after_a_restart(app, settle, settings):
    directory = build(app, settle, "Reload Audio")
    app.render("full", autoplay=False)
    settle()
    app.save()
    app.close()

    reopened = AppController(settings)
    try:
        reopened.open_project(directory)
        assert reopened.rendered("vocal_master") is not None
        assert reopened.rendered("full") is not None
        assert reopened.best_render() is not None
    finally:
        reopened.close()


def test_a_crash_mid_save_is_recovered_from_the_backup(app, settle, settings):
    directory = build(app, settle, "Crash Recovery")
    app.project.title = "Crash Recovery v2"
    app.save()
    app.close()

    # Simulate a process killed while writing project.json.
    (directory / "project.json").write_text('{"title": "hal', encoding="utf-8")

    reopened = AppController(settings)
    try:
        project = reopened.open_project(directory)
        assert project.melody() is not None
        assert project.melody().state is ApprovalState.LOCKED
        assert project.title.startswith("Crash Recovery")
    finally:
        reopened.close()


def test_a_project_with_a_missing_audio_file_still_opens(app, settle, settings):
    directory = build(app, settle, "Missing Audio")
    master = Path(app.project.vocal_master.audio_path)
    app.close()
    master.unlink()

    reopened = AppController(settings)
    try:
        project = reopened.open_project(directory)
        assert project.vocal_master is not None      # the decision is preserved
        assert reopened.rendered("vocal_master") is None
        # and it can be produced again
        reopened.render_vocal("master", autoplay=False)
        pump_until_idle(reopened)
        assert Path(reopened.project.vocal_master.audio_path).exists()
    finally:
        reopened.close()


def test_autosave_keeps_the_project_current(app, settle):
    build(app, settle, "Autosave")
    app.settings.autosave_seconds = 0
    app.add_instrument("flute", 0.0, 20.0, role="counter")
    settle()
    app._last_autosave = 0.0
    app.maybe_autosave()
    saved = json.loads((app.project_dir / "project.json").read_text(encoding="utf-8"))
    instruments = [t["instrument"] for a in saved["arrangements"]
                   for t in a["tracks"]]
    assert "flute" in instruments


def test_a_failing_job_does_not_lose_accepted_work(app, settle):
    build(app, settle, "Job Failure")
    notes_before = [(n.start, n.midi) for n in app.project.melody().notes]
    errors_before = len(app.project.errors)

    def boom(ctx):
        raise RuntimeError("provider exploded")

    app.jobs.submit("render.full", "render:full", boom,
                    on_error=lambda e: app.error("render", f"Render failed: {e}"))
    pump_until_idle(app)

    assert len(app.project.errors) > errors_before
    assert [(n.start, n.midi) for n in app.project.melody().notes] == notes_before
    assert app.project.melody().state is ApprovalState.LOCKED
    assert app.project.vocal_master is not None


def test_playback_without_an_audio_device_is_reported_not_raised(app, settle,
                                                                 monkeypatch):
    build(app, settle, "No Device")

    def fail(*args, **kwargs):
        raise OSError("no output device")

    monkeypatch.setattr(app.playback, "_open_stream", lambda: False)
    monkeypatch.setattr(app.playback, "last_error", "Audio device error: none")
    assert app.play_render("vocal_master") is False
    assert app.project.errors[-1].where == "playback"


def test_the_registry_lists_the_project_after_a_restart(app, settle, settings):
    directory = build(app, settle, "Registry")
    app.close()
    reopened = AppController(settings)
    try:
        entries = [Path(e["directory"]) for e in reopened.recent_projects()]
        assert directory in entries
    finally:
        reopened.close()


def test_reopening_repairs_a_missing_voice_profile(app, settle, settings):
    directory = build(app, settle, "Voice Repair")
    app.project.voice_profile_id = "voice_that_no_longer_exists"
    app.save()
    app.close()

    reopened = AppController(settings)
    try:
        reopened.open_project(directory)
        assert reopened.voices.get(reopened.project.voice_profile_id) is not None
    finally:
        reopened.close()
