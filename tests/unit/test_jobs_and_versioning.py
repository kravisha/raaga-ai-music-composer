"""Unit tests: background jobs, cancellation, undo/redo and lock protection."""
from __future__ import annotations

import threading
import time

import pytest

from raagacomposer.core.jobs import JobCancelled, JobManager
from raagacomposer.core.models import (ApprovalState, MelodyVersion, Note,
                                       Project, Region, Section, Track)
from raagacomposer.core.versioning import (LockedContentError, UndoManager,
                                           assert_melody_editable,
                                           assert_unlocked_track,
                                           locked_regions_in,
                                           locked_sections_in)

pytestmark = pytest.mark.unit


def drain_until(jobs: JobManager, predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        jobs.drain()
        if predicate():
            return True
        time.sleep(0.01)
    jobs.drain()
    return predicate()


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------
def test_a_job_runs_and_delivers_its_result():
    jobs = JobManager(max_workers=2)
    got = []
    jobs.submit("test", "target", lambda ctx: 21 * 2, on_done=got.append)
    assert drain_until(jobs, lambda: got == [42])
    jobs.shutdown()


def test_results_are_delivered_on_the_draining_thread():
    jobs = JobManager(max_workers=2)
    where = {}
    jobs.submit("test", "target", lambda ctx: None,
                on_done=lambda _: where.setdefault("thread",
                                                   threading.current_thread().name))
    assert drain_until(jobs, lambda: "thread" in where)
    assert where["thread"] == threading.current_thread().name
    jobs.shutdown()


def test_progress_is_reported():
    jobs = JobManager(max_workers=1)

    def work(ctx):
        ctx.progress(0.5, "halfway")
        time.sleep(0.05)
        return "ok"

    job = jobs.submit("test", "target", work)
    assert drain_until(jobs, lambda: job.status == "done")
    assert job.progress == 1.0
    jobs.shutdown()


def test_failures_are_reported_not_swallowed():
    jobs = JobManager(max_workers=1)
    errors = []

    def boom(ctx):
        raise ValueError("no good")

    job = jobs.submit("test", "target", boom, on_error=errors.append)
    assert drain_until(jobs, lambda: bool(errors))
    assert isinstance(errors[0], ValueError)
    assert job.status == "failed"
    assert "no good" in job.error
    jobs.shutdown()


def test_cancelling_stops_a_running_job():
    jobs = JobManager(max_workers=2)
    cancelled = []
    started = threading.Event()

    def work(ctx):
        started.set()
        for _ in range(300):
            ctx.check()
            time.sleep(0.01)
        return "finished"

    job = jobs.submit("test", "target", work, on_cancelled=lambda: cancelled.append(1))
    assert started.wait(2.0)
    assert jobs.cancel(job.id)
    assert drain_until(jobs, lambda: bool(cancelled))
    assert job.status == "cancelled"
    jobs.shutdown()


def test_a_newer_job_supersedes_the_older_one_for_the_same_target():
    jobs = JobManager(max_workers=3)
    delivered = []

    def slow(ctx):
        for _ in range(200):
            if ctx.cancelled:
                return "old"
            time.sleep(0.01)
        return "old"

    jobs.submit("test", "same", slow, on_done=delivered.append)
    time.sleep(0.05)
    jobs.submit("test", "same", lambda ctx: "new", on_done=delivered.append)
    assert drain_until(jobs, lambda: "new" in delivered)
    time.sleep(0.3)
    jobs.drain()
    assert "old" not in delivered
    jobs.shutdown()


def test_a_result_from_a_superseded_epoch_is_rejected_as_stale():
    """A provider call that cannot be cancelled must not overwrite newer intent."""
    jobs = JobManager(max_workers=3)
    delivered = []
    release = threading.Event()

    def uncancellable(ctx):
        release.wait(3.0)          # deliberately ignores the cancel flag
        return "stale"

    job = jobs.submit("test", "same", uncancellable, on_done=delivered.append,
                      cancellable=False)
    time.sleep(0.05)
    jobs.submit("test", "same", lambda ctx: "fresh", on_done=delivered.append)
    assert drain_until(jobs, lambda: "fresh" in delivered)
    release.set()
    assert drain_until(jobs, lambda: job.status == "stale")
    assert "stale" not in delivered
    jobs.shutdown()


def test_jobs_on_different_targets_do_not_interfere():
    jobs = JobManager(max_workers=3)
    delivered = []
    jobs.submit("test", "a", lambda ctx: "a", on_done=delivered.append)
    jobs.submit("test", "b", lambda ctx: "b", on_done=delivered.append)
    assert drain_until(jobs, lambda: sorted(delivered) == ["a", "b"])
    jobs.shutdown()


def test_cancel_all_stops_everything_in_flight():
    jobs = JobManager(max_workers=3)

    def slow(ctx):
        for _ in range(200):
            ctx.check()
            time.sleep(0.01)

    for target in ("a", "b", "c"):
        jobs.submit("test", target, slow)
    time.sleep(0.1)
    assert jobs.cancel_all("interrupted") >= 1
    assert drain_until(jobs, lambda: not jobs.active_jobs(), timeout=5)
    jobs.shutdown()


def test_cancel_target_only_touches_that_target():
    jobs = JobManager(max_workers=3)

    def slow(ctx):
        for _ in range(200):
            ctx.check()
            time.sleep(0.01)

    keep = jobs.submit("test", "keep", slow)
    jobs.submit("test", "drop", slow)
    time.sleep(0.1)
    assert jobs.cancel_target("drop") == 1
    time.sleep(0.1)
    assert keep.active
    jobs.cancel_all()
    jobs.shutdown()


def test_job_context_check_raises_when_cancelled():
    event = threading.Event()
    from raagacomposer.core.jobs import JobContext
    ctx = JobContext("j", "t", event, lambda *a: None)
    ctx.check()
    event.set()
    assert ctx.cancelled
    with pytest.raises(JobCancelled):
        ctx.check()


def test_prune_keeps_the_recent_history():
    jobs = JobManager(max_workers=2)
    for i in range(10):
        jobs.submit("test", f"t{i}", lambda ctx: i)
    drain_until(jobs, lambda: not jobs.active_jobs())
    jobs.prune(keep=3)
    assert len(jobs.jobs()) <= 4
    jobs.shutdown()


def test_shutdown_refuses_new_work():
    jobs = JobManager(max_workers=1)
    jobs.shutdown()
    with pytest.raises(RuntimeError):
        jobs.submit("test", "t", lambda ctx: None)


# --------------------------------------------------------------------------
# undo / redo
# --------------------------------------------------------------------------
def test_undo_and_redo_walk_the_stack():
    undo = UndoManager()
    project = Project(title="One")
    undo.reset(project)
    assert not undo.can_undo and not undo.can_redo

    project.title = "Two"
    undo.commit(project, "rename to Two")
    project.title = "Three"
    undo.commit(project, "rename to Three")

    assert undo.can_undo
    restored, label = undo.undo()
    assert restored.title == "Two" and label == "rename to Three"
    restored, _ = undo.undo()
    assert restored.title == "One"
    assert not undo.can_undo

    forward, _ = undo.redo()
    assert forward.title == "Two"
    forward, _ = undo.redo()
    assert forward.title == "Three"
    assert not undo.can_redo


def test_a_new_commit_drops_the_redo_tail():
    undo = UndoManager()
    project = Project(title="One")
    undo.reset(project)
    project.title = "Two"
    undo.commit(project, "two")
    undo.undo()
    project.title = "Different"
    undo.commit(project, "different")
    assert not undo.can_redo
    assert undo.undo()[0].title == "One"


def test_undo_restores_deep_structure():
    undo = UndoManager()
    project = Project()
    melody = MelodyVersion(version=1, notes=[Note(swara="G2", midi=63)],
                           state=ApprovalState.LOCKED)
    project.melodies = [melody]
    project.approved_melody = 1
    undo.reset(project)
    project.melodies[0].notes.append(Note(swara="P", midi=67))
    undo.commit(project, "added a note")
    restored, _ = undo.undo()
    assert len(restored.melody().notes) == 1
    assert restored.melody().state is ApprovalState.LOCKED


def test_the_stack_is_bounded():
    undo = UndoManager(depth=5)
    project = Project()
    undo.reset(project)
    for i in range(20):
        project.title = f"v{i}"
        undo.commit(project, f"step {i}")
    steps = 0
    while undo.can_undo:
        undo.undo()
        steps += 1
    assert steps <= 5


def test_undo_labels_are_reported():
    undo = UndoManager()
    project = Project()
    undo.reset(project)
    undo.commit(project, "add veena")
    assert undo.undo_label() == "add veena"
    undo.undo()
    assert undo.redo_label() == "add veena"


# --------------------------------------------------------------------------
# lock protection
# --------------------------------------------------------------------------
def _project_with_locked_section():
    project = Project()
    section = Section(name="Pallavi", start=10.0, end=30.0, locked=True)
    melody = MelodyVersion(version=1, sections=[section],
                           notes=[Note(start=10.0, duration=1.0,
                                       section_id=section.id)])
    project.melodies = [melody]
    project.approved_melody = 1
    return project, section


def test_locked_sections_are_found_by_overlap():
    project, section = _project_with_locked_section()
    assert locked_sections_in(project, 20.0, 25.0) == [section]
    assert locked_sections_in(project, 0.0, 5.0) == []


def test_editing_a_locked_section_is_refused():
    project, _ = _project_with_locked_section()
    with pytest.raises(LockedContentError) as err:
        assert_melody_editable(project, 15.0, 25.0)
    assert "Locked section" in str(err.value)
    assert_melody_editable(project, 0.0, 5.0)      # elsewhere is fine


def test_a_locked_tune_refuses_partial_edits():
    project = Project()
    melody = MelodyVersion(version=1, state=ApprovalState.LOCKED,
                           notes=[Note(start=0.0, duration=30.0)])
    project.melodies = [melody]
    project.approved_melody = 1
    with pytest.raises(LockedContentError):
        assert_melody_editable(project, 5.0, 10.0)


def test_locked_track_and_region_protection():
    region = Region(start=10.0, end=20.0, locked=True)
    track = Track(instrument="veena", regions=[region])
    assert locked_regions_in(track, 15.0, 18.0) == [region]
    with pytest.raises(LockedContentError) as err:
        assert_unlocked_track(track, 12.0, 18.0)
    assert "locked regions" in str(err.value)
    assert_unlocked_track(track, 25.0, 30.0)       # a clear range is fine

    track.locked = True
    with pytest.raises(LockedContentError):
        assert_unlocked_track(track, 25.0, 30.0)
