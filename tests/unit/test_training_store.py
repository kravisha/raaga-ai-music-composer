"""The training store - specification sections 9, 10, 15 and 16.

Persistence is the whole point of this module, so most of these tests close a
store and open it again rather than trusting what is still in memory.
"""
from __future__ import annotations

import time

import pytest

from raagacomposer.training.models import (Conflict, KnowledgeEntry,
                                           KnowledgeStatus, LearningRun,
                                           LearningSource, Objective,
                                           RunStatus, SearchQuery)
from raagacomposer.training.store import TrainingStore, identity_of

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# identity (section 10)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://youtu.be/DoTQ3ZYK-Qw",
    "https://youtu.be/DoTQ3ZYK-Qw?is=xyz123",
    "https://www.youtube.com/watch?v=DoTQ3ZYK-Qw",
    "https://m.youtube.com/watch?v=DoTQ3ZYK-Qw&t=42s",
    "http://youtube.com/watch?feature=share&v=DoTQ3ZYK-Qw",
])
def test_the_same_lesson_is_recognised_through_any_of_its_links(url: str):
    """Section 10 asks for more than exact-URL matching: a share link, a
    tracking parameter and the mobile host all name one lesson."""
    assert identity_of(url) == "youtube:DoTQ3ZYK-Qw".lower()


def test_different_videos_are_not_confused():
    assert identity_of("https://youtu.be/AAAAAAAAAAA") != \
        identity_of("https://youtu.be/BBBBBBBBBBB")


def test_an_ordinary_url_ignores_query_and_fragment():
    assert identity_of("https://example.org/a/b/?x=1#top") == \
        identity_of("https://www.example.org/a/b")


def test_a_source_with_no_url_still_has_an_identity():
    assert identity_of("", "Kambhoji lesson 1").startswith("title:")


# --------------------------------------------------------------------------
# persistence (section 15)
# --------------------------------------------------------------------------
def test_everything_survives_being_closed_and_reopened(tmp_path):
    path = tmp_path / "training.db"
    store = TrainingStore(path)
    search_id = store.record_search(SearchQuery(phrase="Kambhoji"), 1)
    source = store.save_candidate(LearningSource(
        title="A lesson", url="https://example.org/one", search_id=search_id))
    run = store.add_run(LearningRun(source_id=source.source_id,
                                    search_phrase="Kambhoji"))
    store.save_objectives(run.run_id, [Objective(description="Find the raaga")])
    store.add_knowledge(KnowledgeEntry(
        subject="Kambhoji", concept="arohanam",
        normalized_statement="S R2 G3 M1 P D2 S+", category="scale",
        raga="Kambhoji", source_id=source.source_id, run_id=run.run_id))
    store.close()

    reopened = TrainingStore(path)
    try:
        assert reopened.schema_version == 1
        assert len(reopened.runs()) == 1
        assert len(reopened.objectives(run.run_id)) == 1
        assert reopened.knowledge_count() == 1
        assert reopened.candidate(source.source_id).title == "A lesson"
        assert reopened.searches()[0]["phrase"] == "Kambhoji"
    finally:
        reopened.close()


def test_a_store_written_by_a_newer_version_is_refused(tmp_path, monkeypatch):
    """Opening it read-anyway would silently misread columns that moved."""
    path = tmp_path / "future.db"
    TrainingStore(path).close()
    import raagacomposer.training.store as module
    monkeypatch.setattr(module, "SCHEMA_VERSION", 0)
    with pytest.raises(RuntimeError, match="newer version"):
        TrainingStore(path)


# --------------------------------------------------------------------------
# knowledge (section 9)
# --------------------------------------------------------------------------
def _entry(store, **kwargs) -> KnowledgeEntry:
    defaults = dict(subject="Kambhoji", concept="phrase",
                    normalized_statement="A phrase: S R2 G3 M1.",
                    category="phrase", raga="Kambhoji", source_id="src_1",
                    run_id="run_1", confidence=0.6)
    defaults.update(kwargs)
    return store.add_knowledge(KnowledgeEntry(**defaults))


def test_knowledge_is_searchable_on_every_axis_the_spec_names(training_store):
    _entry(training_store, raga="Kambhoji", category="phrase",
           tags=["heard"], difficulty="beginner")
    _entry(training_store, raga="Keeravani", category="tala", tala="Adi",
           normalized_statement="The lesson refers to Adi tala.",
           concept="tala")

    assert len(training_store.search_knowledge(raga="Kambhoji")) == 1
    assert len(training_store.search_knowledge(tala="Adi")) == 1
    assert len(training_store.search_knowledge(category="phrase")) == 1
    assert len(training_store.search_knowledge(source_id="src_1")) == 2
    assert len(training_store.search_knowledge(difficulty="beginner")) == 1
    assert len(training_store.search_knowledge(tag="heard")) == 1
    assert len(training_store.search_knowledge(keyword="Adi")) == 1


def test_the_same_claim_about_the_same_thing_is_recognised(training_store):
    """Identity is subject plus category plus raaga, so a second source making
    the same claim collides rather than piling up a duplicate."""
    first = _entry(training_store, concept="arohanam", category="scale")
    twin = KnowledgeEntry(
        subject="Kambhoji", concept="arohanam",
        normalized_statement="A different wording of the scale.",
        category="scale", raga="Kambhoji", source_id="src_2", run_id="run_2")
    found = training_store.existing_knowledge(twin)
    assert found is not None and found.knowledge_id == first.knowledge_id


def test_a_claim_about_another_raaga_is_not_a_collision(training_store):
    _entry(training_store, category="scale", raga="Kambhoji")
    other = KnowledgeEntry(subject="Keeravani", concept="arohanam",
                           normalized_statement="something", category="scale",
                           raga="Keeravani")
    assert training_store.existing_knowledge(other) is None


# --------------------------------------------------------------------------
# audit (section 16)
# --------------------------------------------------------------------------
def test_every_stored_item_leaves_an_audit_trail(training_store):
    entry = _entry(training_store)
    trail = training_store.audit_trail(knowledge_id=entry.knowledge_id)
    assert any(row["kind"] == "knowledge.added" for row in trail)

    training_store.update_knowledge(entry.knowledge_id, user_approved=True)
    trail = training_store.audit_trail(knowledge_id=entry.knowledge_id)
    assert any(row["kind"] == "knowledge.updated" for row in trail)


def test_a_conflict_is_recorded_rather_than_resolved(training_store):
    entry = _entry(training_store)
    conflict = training_store.add_conflict(Conflict(
        run_id="run_2", knowledge_id=entry.knowledge_id,
        existing_claim="the old claim", new_claim="the new claim",
        recommendation="a person should decide"))
    open_ones = training_store.conflicts(unresolved_only=True)
    assert [c.conflict_id for c in open_ones] == [conflict.conflict_id]
    # ... and the original is still there, unchanged.
    assert training_store.knowledge(entry.knowledge_id).normalized_statement \
        == "A phrase: S R2 G3 M1."


def test_resolving_a_conflict_records_who_decided_what(training_store):
    entry = _entry(training_store)
    conflict = training_store.add_conflict(Conflict(
        knowledge_id=entry.knowledge_id, existing_claim="a", new_claim="b"))
    training_store.resolve_conflict(conflict.conflict_id, "kept the original")
    assert not training_store.conflicts(unresolved_only=True)
    assert training_store.conflicts()[0].resolution == "kept the original"


# --------------------------------------------------------------------------
# runs and history
# --------------------------------------------------------------------------
def test_runs_keep_their_queue_order(training_store):
    for index in range(3):
        source = training_store.save_candidate(
            LearningSource(title=f"s{index}", url=f"https://e.org/{index}"))
        training_store.add_run(LearningRun(source_id=source.source_id))
    positions = [r.position for r in training_store.runs()]
    assert positions == sorted(positions) == [1, 2, 3]


def test_a_completed_run_makes_its_lesson_already_learned(training_store):
    source = training_store.save_candidate(LearningSource(
        title="A lesson", url="https://youtu.be/DoTQ3ZYK-Qw"))
    run = training_store.add_run(LearningRun(source_id=source.source_id))
    assert training_store.completed_run_for(source) is None

    training_store.update_run(run.run_id, status=RunStatus.COMPLETED,
                              completed_at=time.time())
    # The same lesson reached by a different link is still already learned.
    same = LearningSource(title="A lesson",
                          url="https://www.youtube.com/watch?v=DoTQ3ZYK-Qw")
    found = training_store.completed_run_for(same)
    assert found is not None and found.run_id == run.run_id


def test_deleting_a_run_takes_its_objectives_and_report_with_it(training_store):
    source = training_store.save_candidate(LearningSource(title="x", url="u"))
    run = training_store.add_run(LearningRun(source_id=source.source_id))
    training_store.save_objectives(run.run_id, [Objective(description="a")])
    training_store.delete_run(run.run_id)
    assert training_store.run(run.run_id) is None
    assert training_store.objectives(run.run_id) == []
