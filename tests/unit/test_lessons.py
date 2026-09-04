"""Unit tests: lessons are knowledge (learning specification section 38).

A lesson is the Failure/Lesson object the spec asks for - and the point of
storing it is that the same mistake, made twice, does not get written down
twice: it strengthens one row instead, so the agent can be asked "what keeps
going wrong" and get an answer.
"""
from __future__ import annotations

import pytest

from raagacomposer.agent.knowledge import (SCHEMA_VERSION, KnowledgeRepository,
                                           Lesson)

pytestmark = pytest.mark.unit


@pytest.fixture
def repo(tmp_path) -> KnowledgeRepository:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    yield repository
    repository.close()


def _lesson(raaga="Keeravani", unit_id="b13.short_phrase:Keeravani",
           kind="outside_swara", **kw) -> Lesson:
    return Lesson(raaga=raaga, unit_id=unit_id, kind=kind,
                 dimension=kw.pop("dimension", "swara_correctness"),
                 failure_reason=kw.pop("failure_reason", "1 note(s) outside"),
                 evidence=kw.pop("evidence", "G3"), **kw)


# --------------------------------------------------------------------------
def test_a_lesson_is_inserted(repo):
    lesson, is_new = repo.add_lesson(_lesson())
    assert is_new
    stored = repo.lessons(raaga="Keeravani")
    assert len(stored) == 1
    assert stored[0].id == lesson.id
    assert stored[0].recurrences == 1


def test_the_same_mistake_recurs_rather_than_duplicates(repo):
    first, is_new = repo.add_lesson(_lesson(evidence="G3"))
    assert is_new
    second, is_new_again = repo.add_lesson(_lesson(evidence="G3", result=0.4))
    assert not is_new_again
    assert second.id == first.id
    assert second.recurrences == 2
    rows = repo.lessons(raaga="Keeravani", unit_id="b13.short_phrase:Keeravani",
                        kind="outside_swara")
    assert len(rows) == 1
    assert rows[0].recurrences == 2


def test_a_different_kind_does_not_merge(repo):
    repo.add_lesson(_lesson(kind="outside_swara"))
    repo.add_lesson(_lesson(kind="no_cadence"))
    rows = repo.lessons(raaga="Keeravani", unit_id="b13.short_phrase:Keeravani")
    assert len(rows) == 2
    assert {r.kind for r in rows} == {"outside_swara", "no_cadence"}


def test_a_different_unit_does_not_merge(repo):
    repo.add_lesson(_lesson(unit_id="a01.sound"))
    repo.add_lesson(_lesson(unit_id="a02.pitch"))
    rows = repo.lessons(raaga="Keeravani")
    assert len(rows) == 2


def test_a_different_raaga_does_not_merge(repo):
    repo.add_lesson(_lesson(raaga="Keeravani"))
    repo.add_lesson(_lesson(raaga="Kalyani"))
    assert len(repo.lessons(raaga="Keeravani")) == 1
    assert len(repo.lessons(raaga="Kalyani")) == 1


def test_lessons_filters_by_raaga_unit_kind_and_min_recurrences(repo):
    for _ in range(3):
        repo.add_lesson(_lesson(unit_id="b13.short_phrase:Keeravani",
                                kind="outside_swara"))
    repo.add_lesson(_lesson(unit_id="b14.chains:Keeravani", kind="no_cadence"))

    assert len(repo.lessons(raaga="Keeravani")) == 2
    assert len(repo.lessons(raaga="Keeravani",
                            unit_id="b13.short_phrase:Keeravani")) == 1
    assert len(repo.lessons(raaga="Keeravani", kind="no_cadence")) == 1
    assert len(repo.lessons(raaga="Keeravani", min_recurrences=2)) == 1
    assert len(repo.lessons(raaga="Keeravani", min_recurrences=4)) == 0


def test_lesson_counts_sum_recurrences_by_kind(repo):
    for _ in range(2):
        repo.add_lesson(_lesson(kind="outside_swara"))
    repo.add_lesson(_lesson(kind="no_cadence", unit_id="b14.chains:Keeravani"))
    counts = repo.lesson_counts("Keeravani")
    assert counts["outside_swara"] == 2
    assert counts["no_cadence"] == 1


def test_marking_a_lesson_applied_removes_it_from_lessons_and_counts(repo):
    lesson, _ = repo.add_lesson(_lesson())
    assert repo.lessons(raaga="Keeravani")
    assert repo.lesson_counts("Keeravani")

    repo.mark_lesson_applied(lesson.id)
    assert repo.lessons(raaga="Keeravani") == []
    assert repo.lesson_counts("Keeravani") == {}
    # It is still there when explicitly asked for the applied ones.
    assert repo.lessons(raaga="Keeravani", include_applied=True)


def test_lessons_survive_close_and_reopen(tmp_path):
    """Section 27: learn, restart, retrieve."""
    path = tmp_path / "persist.db"
    first = KnowledgeRepository(path)
    first.add_lesson(Lesson(raaga="Keeravani", unit_id="a01.sound",
                            kind="off_beat", dimension="rhythm",
                            failure_reason="the note lengths do not sit on "
                                          "the beat",
                            evidence="0.6", correction="line it up with the "
                                                       "beat"))
    first.close()

    second = KnowledgeRepository(path)
    try:
        rows = second.lessons(raaga="Keeravani")
        assert len(rows) == 1
        assert rows[0].kind == "off_beat"
        assert rows[0].correction == "line it up with the beat"
    finally:
        second.close()


def test_a_schema_version_1_database_migrates_and_gains_lessons(tmp_path):
    path = tmp_path / "v1.db"
    repository = KnowledgeRepository(path)
    repository._conn.execute(
        "UPDATE meta SET value='1' WHERE key='schema_version'")
    repository._conn.execute("DROP TABLE lessons")
    repository._conn.commit()
    repository.close()

    reopened = KnowledgeRepository(path)
    try:
        # Whatever the current schema is, not the number it happened to be
        # when this test was written: the point is that an old database is
        # brought up to date and keeps what it had, and pinning the literal
        # made every later schema change look like a regression.
        assert reopened.schema_version == SCHEMA_VERSION
        tables = {r["name"] for r in reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"lessons", "selection_weights"} <= tables
        # And the tables actually work, not just exist.
        reopened.add_lesson(Lesson(raaga="Keeravani", unit_id="a01.sound",
                                   kind="off_beat"))
        assert reopened.lessons(raaga="Keeravani")
        reopened.record_selection_feedback("Keeravani", -0.7, {"sadness": 1.0})
        assert reopened.selection_weights("Keeravani")
    finally:
        reopened.close()


def test_stats_reports_lesson_count(repo):
    assert repo.stats()["lessons"] == 0
    repo.add_lesson(_lesson(kind="outside_swara"))
    repo.add_lesson(_lesson(kind="no_cadence"))
    assert repo.stats()["lessons"] == 2
    lesson, _ = repo.add_lesson(_lesson(kind="no_cadence"))
    assert repo.stats()["lessons"] == 2         # recurrence, not a new row
    repo.mark_lesson_applied(lesson.id)
    assert repo.stats()["lessons"] == 1
