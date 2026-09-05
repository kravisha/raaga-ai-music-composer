"""The learning queue - specification sections 3.3, 13, 14 and 15.

One source at a time, on one worker thread, with the whole of its state on
disk rather than in memory.  That last point is what makes section 15 true: a
creator who closes the application mid-run finds the queue where they left it,
because the queue *is* the ``runs`` table and was never anything else.

The architecture allows more than one worker later - the queue hands out work
by claiming the next queued run - but one is what ships, as section 3.3 asks.

Two behaviours are worth naming because they are easy to get wrong:

*Interrupted runs are recovered, not lost.*  A run left mid-flight by a crash
is in a working status that nothing is working on.  On startup those are put
back to Queued with their attempt count intact, so they are retried rather
than sitting for ever in "Analyzing" with nobody analysing.

*Cancelling is cooperative.*  The pipeline is asked between phases whether it
should stop, so a cancel takes effect in a moment without killing a thread
half-way through a database write.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..raaga.library import RaagaLibrary
from .models import (LearningReport, LearningRun, LearningSource, RunStatus,
                     new_id)
from .pipeline import LearningPipeline
from .store import TrainingStore

log = get_logger("training.queue")

#: Section 14 - a network failure is retried with bounded backoff, not for ever.
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = (2.0, 4.0, 8.0)


class TrainingQueueService:
    """Works the queue one source at a time, and survives being closed."""

    def __init__(self, store: TrainingStore, pipeline: LearningPipeline,
                 on_change: Optional[Callable[[], None]] = None,
                 on_report: Optional[Callable[[LearningReport], None]] = None
                 ) -> None:
        self.store = store
        self.pipeline = pipeline
        self.on_change = on_change
        #: Called with each completed report, so what a source taught can be
        #: turned into lessons the agent is examined on
        #: (``training/lessons.py``).  Never allowed to fail a run: a source
        #: was still studied even if nothing downstream could use it.
        self.on_report = on_report

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._cancel_current = threading.Event()
        self._current_run_id = ""
        self._lock = threading.RLock()
        self.last_error = ""

        self.recover()

    # ==================================================================
    # lifecycle
    # ==================================================================
    def recover(self) -> int:
        """Section 14: put back anything a crash left mid-flight."""
        stranded = self.store.runs(statuses=RunStatus.IN_FLIGHT)
        for run in stranded:
            self.store.update_run(
                run.run_id, status=RunStatus.QUEUED, progress=0.0,
                detail="returned to the queue after the application closed")
            self.store.audit("run.recovered",
                             f"was {run.status} with nobody working on it",
                             run_id=run.run_id)
        if stranded:
            log.info("recovered %d interrupted run(s)", len(stranded))
        return len(stranded)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    @property
    def current_run_id(self) -> str:
        return self._current_run_id

    # ==================================================================
    # queueing
    # ==================================================================
    def enqueue(self, sources: Sequence[LearningSource],
                search_phrase: str = "",
                objectives_for=None) -> List[LearningRun]:
        """Add approved sources. Nothing is queued that the user did not pick."""
        created: List[LearningRun] = []
        for source in sources:
            self.store.save_candidate(source)
            run = self.store.add_run(LearningRun(
                source_id=source.source_id, search_phrase=search_phrase))
            if objectives_for is not None:
                objectives = objectives_for(source)
                if objectives:
                    self.store.save_objectives(run.run_id, objectives)
            created.append(run)
        self._changed()
        return created

    def relearn(self, source: LearningSource, search_phrase: str = "",
                previous_run_id: str = "") -> LearningRun:
        """Section 10: a new run, and the old report is left standing."""
        run = self.store.add_run(LearningRun(
            source_id=source.source_id, search_phrase=search_phrase,
            supersedes=previous_run_id))
        self.store.audit("run.relearn",
                         f"relearning; the earlier report is kept",
                         run_id=run.run_id, source_id=source.source_id)
        self._changed()
        return run

    def remove(self, run_id: str) -> None:
        run = self.store.run(run_id)
        if run is None:
            return
        if run_id == self._current_run_id:
            self.cancel_current()
        self.store.delete_run(run_id)
        self._changed()

    def retry(self, run_id: str) -> None:
        run = self.store.run(run_id)
        if run is None:
            return
        self.store.update_run(run_id, status=RunStatus.QUEUED, progress=0.0,
                              error="", detail="queued again by the user",
                              completed_at=0.0)
        self.store.audit("run.retry", "queued again", run_id=run_id)
        self._changed()

    def skip(self, run_id: str) -> None:
        self.store.update_run(run_id, status=RunStatus.SKIPPED, progress=0.0,
                              detail="skipped by the user",
                              result="skipped", completed_at=time.time())
        self._changed()

    # ==================================================================
    # working
    # ==================================================================
    def start(self) -> None:
        with self._lock:
            self._pause.clear()
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._work,
                                            name="training-queue", daemon=True)
            self._thread.start()
            log.info("training queue started")

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()
        if not self.running:
            self.start()

    def cancel_current(self) -> None:
        self._cancel_current.set()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._cancel_current.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------
    def pending(self) -> List[LearningRun]:
        return self.store.runs(statuses=(RunStatus.QUEUED,))

    def _next(self) -> Optional[LearningRun]:
        queued = self.pending()
        return queued[0] if queued else None

    def _work(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.05)
                continue
            run = self._next()
            if run is None:
                break
            self._process(run)
        self._changed()

    def _process(self, run: LearningRun) -> Optional[LearningReport]:
        source = self.store.candidate(run.source_id)
        if source is None:
            self.store.update_run(
                run.run_id, status=RunStatus.FAILED,
                error="the source is no longer in the store",
                result="failed", completed_at=time.time())
            self._changed()
            return None

        self._current_run_id = run.run_id
        self._cancel_current.clear()
        try:
            report = self.pipeline.run(
                run, source,
                on_progress=lambda *_: self._changed(),
                cancelled=lambda: (self._cancel_current.is_set()
                                   or self._stop.is_set()))
            if report is not None and self.on_report is not None:
                try:
                    self.on_report(report)
                except Exception as exc:                     # noqa: BLE001
                    log.warning("could not file lessons for run %s: %s",
                                run.run_id, exc)
            return report
        except Exception as exc:  # noqa: BLE001 - one bad source is not fatal
            self.last_error = str(exc)
            attempts = run.attempts + 1
            log.warning("run %s failed on attempt %d: %s", run.run_id,
                        attempts, exc)
            if attempts < MAX_ATTEMPTS:
                delay = BACKOFF_SECONDS[min(attempts - 1,
                                            len(BACKOFF_SECONDS) - 1)]
                self.store.update_run(
                    run.run_id, status=RunStatus.QUEUED, attempts=attempts,
                    error=str(exc),
                    detail=f"attempt {attempts} failed; retrying in "
                           f"{delay:.0f}s")
                self.store.audit("run.retrying", str(exc), run_id=run.run_id)
                # Wait, but stay responsive to a stop.
                self._stop.wait(delay)
            else:
                self.store.update_run(
                    run.run_id, status=RunStatus.FAILED, attempts=attempts,
                    error=str(exc), result=f"failed: {exc}",
                    detail="gave up after repeated failures",
                    completed_at=time.time())
                self.store.audit("run.gave_up", str(exc), run_id=run.run_id)
            return None
        finally:
            self._current_run_id = ""
            self._changed()

    def process_next(self) -> Optional[LearningReport]:
        """Work one source on the calling thread. Used by tests and the CLI."""
        run = self._next()
        if run is None:
            return None
        return self._process(run)

    # ==================================================================
    def snapshot(self) -> List[Dict[str, object]]:
        """What the queue view shows - section 3.3."""
        rows: List[Dict[str, object]] = []
        for run in self.store.runs(limit=500):
            source = self.store.candidate(run.source_id)
            objectives = self.store.objectives(run.run_id)
            current = next((o.description for o in objectives
                            if not o.met), objectives[0].description
                           if objectives else "")
            rows.append({
                "run_id": run.run_id,
                "position": run.position,
                "title": source.title if source else "(missing source)",
                "objective": current,
                "objectives": len(objectives),
                "status": run.status,
                "status_label": RunStatus.LABELS.get(run.status, run.status),
                "progress": run.progress,
                "detail": run.detail,
                "started": run.started_at,
                "completed": run.completed_at,
                "result": run.result,
                "attempts": run.attempts,
                "is_current": run.run_id == self._current_run_id,
            })
        return rows

    def _changed(self) -> None:
        if self.on_change:
            try:
                self.on_change()
            except Exception:  # noqa: BLE001
                log.debug("queue change hook failed", exc_info=True)
