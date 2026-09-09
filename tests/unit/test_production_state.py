"""A production journal must survive revisions without losing earlier evidence."""
from dataclasses import asdict
import json
import os
import subprocess
import sys

import pytest

from raagacomposer.production.state import JournalLockError, ProjectState, StageRecord


def accepted(**changes):
    values = dict(stage="tune", role="critic", artifact_ref="tune-v1",
                  rationale="The supplied note evidence fits the requested phrases",
                  verdict="accept", round=1, provider="codex", evidence=["score:v1"])
    return StageRecord(**{**values, **changes})


def test_reopen_retains_original_history_and_later_revision(tmp_path):
    state = ProjectState(project_id="song-1", brief={"situation": "hopeful reunion", "language": "Tamil"},
                         tune_ref="tune-v1", lyric_ref="words-v1", vocal_takes=["voice-v1"], mix="mix-v1")
    state.add_record(accepted())
    target = state.save(tmp_path / "song" / "production.json")
    reopened = ProjectState.load(target)
    assert reopened.is_accepted("tune", "tune-v1")
    first = asdict(reopened.records[0])
    reopened.invalidate_from("lyrics", "Change the second line's emotional emphasis")
    assert reopened.tune_ref == "tune-v1"
    assert reopened.lyric_ref == "" and not reopened.vocal_takes and reopened.mix == ""
    assert not reopened.is_accepted("tune", "tune-v1")
    reopened.add_record(accepted(revision=1, round=1))
    reopened.save(target)
    final = ProjectState.load(target)
    assert asdict(final.records[0]) == first
    assert final.is_accepted("tune", "tune-v1") and final.revision == 1
    assert len(final.records) == 2 and final.feedback


def test_refuses_to_discard_another_writers_preserved_history(tmp_path):
    target = tmp_path / "production.json"
    state = ProjectState(project_id="song")
    state.save(target)
    stale = ProjectState.load(target)
    state.add_record(accepted())
    state.save(target)
    original = target.read_bytes()
    with pytest.raises(ValueError, match="history"):
        stale.save(target)
    assert target.read_bytes() == original
    assert not target.with_name(target.name + ".lock").exists()


def test_another_writer_lock_is_preserved(tmp_path):
    target = tmp_path / "production.json"
    lock = tmp_path / "production.json.lock"
    lock.write_text("other writer", encoding="utf-8")
    with pytest.raises(JournalLockError) as caught:
        ProjectState(project_id="song").save(target)
    assert str(lock.resolve()) in str(caught.value)
    assert caught.value.owner_status == "unknown" and "age" in str(caught.value)
    assert lock.read_text() == "other writer" and not target.exists()


def test_live_writer_lock_names_owner_and_age_without_reclaiming(tmp_path):
    from raagacomposer.production.state import utc_now
    target = tmp_path / "production.json"
    lock = target.with_name(target.name + ".lock")
    content = json.dumps({"pid": os.getpid(), "created_at": utc_now()})
    lock.write_text(content)
    with pytest.raises(JournalLockError) as caught:
        ProjectState(project_id="song").save(target)
    assert caught.value.pid == os.getpid()
    assert caught.value.owner_status == "running" and caught.value.age_seconds >= 0
    assert lock.read_text() == content and not target.exists()


def test_crashed_writer_leaves_diagnosable_lock_and_original_journal(tmp_path):
    target = ProjectState(project_id="song").save(tmp_path / "production.json")
    original = target.read_bytes()
    # Simulate a real abrupt process exit, which skips Python finally blocks.
    script = """
import os, sys
from raagacomposer.production.state import ProjectState
from raagacomposer.production import state as module
song = ProjectState.load(sys.argv[1])
song.brief['mood'] = 'unsaved change'
module.os.replace = lambda *args: os._exit(23)
song.save(sys.argv[1])
"""
    process = subprocess.run([sys.executable, "-c", script, str(target)],
                             capture_output=True, timeout=15)
    assert process.returncode == 23, process.stderr
    lock = target.with_name(target.name + ".lock")
    metadata = json.loads(lock.read_text())
    assert metadata["pid"] > 0 and metadata["created_at"]
    with pytest.raises(JournalLockError) as caught:
        ProjectState.load(target).save(target)
    assert caught.value.pid == metadata["pid"]
    assert caught.value.owner_status == "not running"
    assert "preserve" in str(caught.value) and str(lock.resolve()) in str(caught.value)
    assert target.read_bytes() == original and lock.exists()


def test_failed_lock_metadata_write_releases_only_owned_lock(tmp_path, monkeypatch):
    from raagacomposer.production import state as module
    monkeypatch.setattr(module.os, "fsync", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        ProjectState(project_id="song").save(tmp_path / "production.json")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("content", ['{"pid": -1}', '{"pid": true}', '{"pid": 999999999999999}',
                                    '{"pid":', '["not an object"]'])
def test_malformed_lock_still_reports_safe_recovery_context(tmp_path, content):
    lock = tmp_path / "production.json.lock"
    lock.write_text(content)
    with pytest.raises(JournalLockError) as caught:
        ProjectState(project_id="song").save(tmp_path / "production.json")
    assert caught.value.owner_status == "unknown"
    assert str(lock.resolve()) in str(caught.value)
    assert lock.read_text() == content


def test_failed_atomic_replace_keeps_original_and_cleans_owned_temporary(tmp_path, monkeypatch):
    from raagacomposer.production import state as module
    target = ProjectState(project_id="song").save(tmp_path / "production.json")
    original = target.read_bytes()
    monkeypatch.setattr(module.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    with pytest.raises(OSError):
        ProjectState.load(target).save(target)
    assert target.read_bytes() == original
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("data", [
    {"schema_version": 2}, {"revision": True}, {"revision": -1},
    {"unexpected": "silent loss would hide this"}, {"vocal_takes": "path"},
    {"brief": []}, {"brief": {"duration": float("nan")}},
])
def test_malformed_state_is_rejected(data):
    with pytest.raises((ValueError, TypeError)):
        ProjectState.from_dict(data)


def test_loaded_metadata_does_not_alias_callers_mutable_data():
    incoming = ProjectState(project_id="song", records=[accepted()], brief={"moods": ["hopeful"]}).to_dict()
    loaded = ProjectState.from_dict(incoming)
    incoming["records"][0]["evidence"].append("unreviewed")
    incoming["brief"]["moods"].append("angry")
    assert loaded.records[0].evidence == ["score:v1"]
    assert loaded.brief == {"moods": ["hopeful"]}


def test_acceptance_requires_exact_artifact_revision_and_actual_codex_identity():
    state = ProjectState(project_id="song")
    state.add_record(accepted(provider="local rules"))
    assert not state.is_accepted("tune", "tune-v1")
    state.add_record(accepted(round=2))
    assert state.is_accepted("tune", "tune-v1")
    assert not state.is_accepted("tune", "tune-v2")
    state.add_record(accepted(verdict="blocked", round=3, rationale="new measurements missing"))
    assert not state.is_accepted("tune", "tune-v1")


def test_fingerprint_changes_for_music_or_feedback_but_not_appended_review():
    state = ProjectState(project_id="song", tune_ref="v1")
    baseline = state.fingerprint()
    state.add_record(accepted())
    assert state.fingerprint() == baseline
    state.tune_ref = "v2"
    assert state.fingerprint() != baseline
    second = state.fingerprint()
    state.feedback.append("More longing")
    assert state.fingerprint() != second


def test_bad_existing_journal_is_not_replaced(tmp_path):
    target = tmp_path / "production.json"
    target.write_text('{"truncated":', encoding="utf-8")
    original = target.read_bytes()
    with pytest.raises(json.JSONDecodeError):
        ProjectState(project_id="song").save(target)
    assert target.read_bytes() == original


def test_stale_or_future_acceptance_cannot_be_added():
    state = ProjectState(project_id="song", revision=1)
    with pytest.raises(ValueError):
        state.add_record(accepted(revision=0))
    with pytest.raises(ValueError):
        state.add_record(accepted(revision=2))


def test_stale_writer_cannot_overwrite_brief_even_before_any_reviews(tmp_path):
    state = ProjectState(project_id="song", brief={"mood": "joy"})
    target = state.save(tmp_path / "production.json")
    stale = ProjectState.load(target)
    state.brief["mood"] = "longing"
    state.save(target)
    with pytest.raises(ValueError, match="stale"):
        stale.save(target)
    assert ProjectState.load(target).brief["mood"] == "longing"


def test_new_object_cannot_silently_replace_existing_journal(tmp_path):
    target = ProjectState(project_id="song", brief={"mood": "hopeful"}).save(tmp_path / "production.json")
    with pytest.raises(ValueError, match="reload"):
        ProjectState(project_id="song").save(target)
    assert ProjectState.load(target).brief["mood"] == "hopeful"
