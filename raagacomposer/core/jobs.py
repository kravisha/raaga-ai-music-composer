"""Background job manager, cancellation and stale-result rejection.

Spec sections 12.29/12.30 and 17.  Two rules drive the design:

* the GUI thread never blocks on generation or rendering; and
* the newest explicit creator instruction wins -- when a new job is submitted
  for the same *target*, the epoch for that target is bumped and any result
  arriving later from an older epoch is discarded rather than allowed to
  overwrite newer creator intent.

Results are handed back through a queue and dispatched by :meth:`drain`, which
the UI calls on its own thread.  Worker threads therefore never touch Qt.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .logging_setup import get_logger

log = get_logger("jobs")


class JobCancelled(Exception):
    """Raised inside a worker when the creator interrupts."""


@dataclass
class JobContext:
    job_id: str
    target: str
    _cancel: threading.Event
    _progress_cb: Callable[[str, float, str], None]
    # The epoch this run was submitted under, so a worker can label what it
    # reports (status text, partial results) and the UI thread can tell a
    # superseded run's words from the current one's.
    epoch: int = 0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise JobCancelled(self.job_id)

    def progress(self, value: float, message: str = "") -> None:
        self.check()
        self._progress_cb(self.job_id, max(0.0, min(1.0, value)), message)


@dataclass
class Job:
    id: str
    job_type: str
    target: str
    epoch: int
    description: str = ""
    provider: str = "local"
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    error: str = ""
    cancellable: bool = True
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


class JobManager:
    """Thread pool + epoch bookkeeping."""

    def __init__(self, max_workers: int = 3) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="raaga-job")
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._epochs: Dict[str, int] = {}
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._counter = 0
        self._shutdown = False
        self.on_change: Optional[Callable[[], None]] = None

    # -- introspection -----------------------------------------------------
    def jobs(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def active_jobs(self) -> List[Job]:
        return [j for j in self.jobs() if j.active]

    def job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def current_epoch(self, target: str) -> int:
        with self._lock:
            return self._epochs.get(target, 0)

    # -- submission --------------------------------------------------------
    def submit(self,
               job_type: str,
               target: str,
               fn: Callable[[JobContext], object],
               on_done: Optional[Callable[[object], None]] = None,
               on_error: Optional[Callable[[BaseException], None]] = None,
               on_cancelled: Optional[Callable[[], None]] = None,
               description: str = "",
               provider: str = "local",
               supersede: bool = True,
               cancellable: bool = True) -> Job:
        """Queue *fn* on a worker thread.

        With ``supersede`` (the default) any in-flight job for the same target
        is cancelled and its eventual output rejected as stale.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("job manager is shut down")
            if supersede:
                for j in self._jobs.values():
                    if j.target == target and j.active:
                        self._cancel_locked(j, "superseded")
            self._counter += 1
            epoch = self._epochs.get(target, 0) + 1
            self._epochs[target] = epoch
            job = Job(id=f"job_{self._counter:05d}", job_type=job_type,
                      target=target, epoch=epoch,
                      description=description or job_type, provider=provider,
                      cancellable=cancellable)
            self._jobs[job.id] = job

        log.info("submit %s type=%s target=%s epoch=%d", job.id, job_type, target, epoch)
        self._emit("submitted", job.id, None)
        self._pool.submit(self._run, job, fn, on_done, on_error, on_cancelled)
        return job

    # -- cancellation ------------------------------------------------------
    def cancel(self, job_id: str, reason: str = "cancelled") -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or not job.active:
                return False
            return self._cancel_locked(job, reason)

    def cancel_target(self, target: str, reason: str = "superseded") -> int:
        with self._lock:
            n = 0
            for j in self._jobs.values():
                if j.target == target and j.active:
                    n += int(self._cancel_locked(j, reason))
            return n

    def cancel_all(self, reason: str = "cancelled") -> int:
        with self._lock:
            return sum(int(self._cancel_locked(j, reason))
                       for j in self._jobs.values() if j.active)

    def _cancel_locked(self, job: Job, reason: str) -> bool:
        # Bump the epoch first: even a provider call that cannot be interrupted
        # then lands as stale rather than overwriting newer creator intent.
        self._epochs[job.target] = self._epochs.get(job.target, 0) + 1
        if not job.cancellable:
            job.message = f"{reason} (uninterruptible - its output is discarded)"
            log.info("job %s cannot be interrupted; result will be stale", job.id)
            return False
        job.cancel_event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.message = reason
            job.finished_at = time.time()
            self._emit("cancelled", job.id, None)
        else:
            job.message = reason
        log.info("cancel %s (%s)", job.id, reason)
        return True

    # -- worker ------------------------------------------------------------
    def _run(self, job: Job, fn, on_done, on_error, on_cancelled) -> None:
        if job.cancel_event.is_set():
            self._finish(job, "cancelled", on_cancelled=on_cancelled)
            return
        job.status = "running"
        job.started_at = time.time()
        self._emit("started", job.id, None)
        ctx = JobContext(job.id, job.target, job.cancel_event, self._progress,
                         epoch=job.epoch)
        try:
            result = fn(ctx)
        except JobCancelled:
            self._finish(job, "cancelled", on_cancelled=on_cancelled)
            return
        except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
            job.error = f"{type(exc).__name__}: {exc}"
            log.error("job %s failed: %s\n%s", job.id, job.error, traceback.format_exc())
            self._finish(job, "failed", error=exc, on_error=on_error)
            return

        if job.cancel_event.is_set():
            self._finish(job, "cancelled", on_cancelled=on_cancelled)
            return
        with self._lock:
            stale = self._epochs.get(job.target, 0) != job.epoch
        if stale:
            log.info("job %s result rejected as stale (target=%s)", job.id, job.target)
            self._finish(job, "stale", on_cancelled=on_cancelled)
            return
        self._finish(job, "done", result=result, on_done=on_done)

    def _finish(self, job: Job, status: str, result=None, error=None,
                on_done=None, on_error=None, on_cancelled=None) -> None:
        job.status = status
        job.finished_at = time.time()
        job.progress = 1.0 if status == "done" else job.progress
        if status == "done" and on_done:
            self._events.put(("callback", job.id, (on_done, (result,))))
        elif status == "failed" and on_error:
            self._events.put(("callback", job.id, (on_error, (error,))))
        elif status in ("cancelled", "stale") and on_cancelled:
            self._events.put(("callback", job.id, (on_cancelled, ())))
        self._emit(status, job.id, None)

    def _progress(self, job_id: str, value: float, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.progress = value
                if message:
                    job.message = message
        self._emit("progress", job_id, None)

    def _emit(self, kind: str, job_id: str, payload) -> None:
        self._events.put((kind, job_id, payload))
        if self.on_change:
            try:
                self.on_change()
            except Exception:
                pass

    # -- main-thread pump --------------------------------------------------
    def drain(self, limit: int = 64) -> List[tuple]:
        """Run queued completion callbacks on the calling (UI) thread."""
        seen: List[tuple] = []
        for _ in range(limit):
            try:
                kind, job_id, payload = self._events.get_nowait()
            except queue.Empty:
                break
            seen.append((kind, job_id))
            if kind == "callback" and payload:
                fn, args = payload
                try:
                    fn(*args)
                except Exception:
                    log.error("completion callback failed for %s\n%s",
                              job_id, traceback.format_exc())
        return seen

    def prune(self, keep: int = 200) -> None:
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if not j.active),
                              key=lambda j: j.finished_at)
            for j in finished[:-keep] if len(finished) > keep else []:
                self._jobs.pop(j.id, None)

    def shutdown(self, wait: bool = False) -> None:
        self.cancel_all("shutdown")
        with self._lock:
            self._shutdown = True
        self._pool.shutdown(wait=wait, cancel_futures=True)
