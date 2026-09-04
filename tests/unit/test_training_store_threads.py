"""Unit test: TrainingStore is safe to use from several threads at once.

The training store's connection is shared between the queue worker and the
UI thread (``check_same_thread=False``), exactly as the agent's
KnowledgeRepository is - and the sqlite3 module does not make concurrent
statements on one connection safe on its own.  Two threads running
statements at the same time corrupt the connection's state rather than wait
for each other: ``sqlite3.InterfaceError: bad parameter or other API
misuse``, a SELECT that another thread reset half-way through, or a worker
that never returns.  ``TrainingStore._lock`` serialises every method that
touches ``self._conn`` so that cannot happen.

This is the shape that found the flake in
``test_the_queue_worker_processes_everything_it_is_given``: the worker
selecting the next queued run and updating its progress while the test's
main thread polled ``store.runs(statuses=('queued',))`` - the *same* SQL,
so the two threads fought over one cached statement.  Without the lock this
test fails within milliseconds; with it, the interleaving is harmless.
"""
from __future__ import annotations

import threading

import pytest

from raagacomposer.training.models import (KnowledgeEntry, LearningRun,
                                           LearningSource, Objective,
                                           RunStatus)
from raagacomposer.training.store import TrainingStore

pytestmark = pytest.mark.unit

ROUNDS = 300
WRITER_THREADS = 3


def test_concurrent_reads_and_writes_do_not_corrupt_the_connection(tmp_path):
    store = TrainingStore(tmp_path / "threaded.db")
    errors: list = []
    lock = threading.Lock()          # guards the shared `errors` list itself

    def record(exc: BaseException) -> None:
        with lock:
            errors.append(exc)

    sources = [store.save_candidate(LearningSource(
        title=f"lesson {n}", url=f"raaga-exercise://Keeravani/unit{n}",
        metadata={"raaga": "Keeravani"})) for n in range(WRITER_THREADS)]
    runs = [store.add_run(LearningRun(source_id=s.source_id,
                                      search_phrase="Keeravani"))
            for s in sources]

    def writer(n: int) -> None:
        """What the queue worker does to its run while it is being worked."""
        run = runs[n]
        try:
            for i in range(ROUNDS):
                store.runs(statuses=(RunStatus.QUEUED,))      # _next()
                store.update_run(run.run_id, status=RunStatus.ANALYZING,
                                 progress=i / ROUNDS, detail=f"step {i}")
                store.audit("run.progress", f"writer {n} round {i}",
                            run_id=run.run_id)
                store.save_objectives(run.run_id, [Objective(
                    run_id=run.run_id, description=f"objective {i}")])
                store.add_knowledge(KnowledgeEntry(
                    subject=f"claim {n}-{i}", concept="Keeravani",
                    normalized_statement=f"writer {n} learned {i}",
                    category="fact", raga="Keeravani",
                    source_id=run.source_id, run_id=run.run_id,
                    evidence=str(i)))
            store.update_run(run.run_id, status=RunStatus.COMPLETED,
                             progress=1.0)
        except Exception as exc:  # noqa: BLE001
            record(exc)

    def reader() -> None:
        """What the UI (and the test's polling loop) does meanwhile."""
        try:
            for _ in range(ROUNDS):
                store.runs(statuses=(RunStatus.QUEUED,))      # pending()
                store.runs(limit=500)                          # snapshot()
                store.run(runs[0].run_id)
                store.objectives(runs[1].run_id)
                store.audit_trail(run_id=runs[2].run_id, limit=5)
                store.search_knowledge(raga="Keeravani", limit=50)
        except Exception as exc:  # noqa: BLE001
            record(exc)

    threads = [threading.Thread(target=writer, args=(n,), name=f"writer-{n}")
               for n in range(WRITER_THREADS)]
    try:
        for thread in threads:
            thread.start()
        reader()                     # the main thread reads while they write
    finally:
        for thread in threads:
            thread.join(timeout=60.0)

    try:
        assert not any(t.is_alive() for t in threads), \
            "a writer never finished"
        assert not errors, [repr(e) for e in errors]

        # Every writer got to the end and nothing it wrote was lost.
        assert all(store.run(r.run_id).status == RunStatus.COMPLETED
                   for r in runs)
        assert store.knowledge_count() == WRITER_THREADS * ROUNDS
        for run in runs:
            assert len(store.objectives(run.run_id)) == 1
            # The queued audit, then per round one progress audit and the
            # knowledge.added audit that add_knowledge writes itself.
            assert len(store.audit_trail(run_id=run.run_id, limit=1000)) \
                == 2 * ROUNDS + 1
    finally:
        store.close()
