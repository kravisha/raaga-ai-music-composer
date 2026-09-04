"""Unit test: KnowledgeRepository is safe to use from several threads at once.

Learning specification section 8's connection is shared between the UI
thread and the background learning thread (``check_same_thread=False``).
sqlite3 does not make concurrent statements on one connection safe on its
own - two threads issuing statements at the same time can corrupt the
connection's internal state and raise things like
``sqlite3.InterfaceError: bad parameter or other API misuse`` or
``sqlite3.OperationalError: cannot commit - no transaction is active``,
rather than merely block on each other.  ``KnowledgeRepository._lock``
serialises every method that touches ``self._conn`` so that cannot happen.

A single writer thread racing the main thread's reads was not enough to
reproduce this reliably on this machine - the interleaving window is narrow.
Three writer threads racing the main thread's reads reproduced it in 5 out of
5 runs before the lock was added (see the session notes); that is the shape
kept here.
"""
from __future__ import annotations

import threading

import pytest

from raagacomposer.agent.knowledge import (KnowledgeRepository, Lesson,
                                           UnitProgress)

pytestmark = pytest.mark.unit

ROUNDS = 300
WRITER_THREADS = 3
KINDS = 5


def test_concurrent_reads_and_writes_do_not_corrupt_the_connection(tmp_path):
    repo = KnowledgeRepository(tmp_path / "threaded.db")
    errors: list = []
    lock = threading.Lock()          # guards the shared `errors` list itself

    def record(exc: BaseException) -> None:
        with lock:
            errors.append(exc)

    def writer(n: int) -> None:
        try:
            for i in range(ROUNDS):
                repo.add_lesson(Lesson(raaga="Keeravani", unit_id="a01.sound",
                                       kind=f"kind{i % KINDS}",
                                       failure_reason=f"writer {n} round {i}",
                                       evidence=str(i)))
                repo.log_event("test.write", f"{n}-{i}")
                repo.save_progress(UnitProgress(unit_id="a01.sound",
                                                raaga="Keeravani", attempts=i))
        except Exception as exc:  # noqa: BLE001
            record(exc)

    def reader() -> None:
        try:
            for _ in range(ROUNDS):
                repo.lessons(raaga="Keeravani")
                repo.progress("a01.sound")
                repo.stats()
                repo.events(limit=5)
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
        assert not any(t.is_alive() for t in threads), "a writer never finished"
        assert not errors, [repr(e) for e in errors]

        # Five kinds, WRITER_THREADS * ROUNDS writes: each kind recurs that
        # many times over 5 and nothing was lost or duplicated by the race.
        stored = repo.lessons(raaga="Keeravani", unit_id="a01.sound",
                              include_applied=True, limit=100)
        assert len(stored) == KINDS
        assert sum(l.recurrences for l in stored) == WRITER_THREADS * ROUNDS
        assert repo.stats()["lessons"] == KINDS
    finally:
        repo.close()
