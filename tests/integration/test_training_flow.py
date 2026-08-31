"""The Training tab working - sections 4, 10, 13, 14 and 15.

The acceptance suite proves the happy path.  These cover what the
specification spends most of its words on: what happens when a source cannot
be reached, when the creator changes their mind, when something breaks, and
when the application is closed in the middle.
"""
from __future__ import annotations

import time

import pytest
import soundfile as sf

from raagacomposer.training.models import (Accessibility, LearningRun,
                                           LearningSource, RunStatus)
from raagacomposer.training.pipeline import METADATA_ONLY_NOTICE
from tests.conftest import ANALYSIS_SR, lesson_signal, sung_signal

pytestmark = pytest.mark.integration


def _queue_web_source(training, url="https://youtu.be/DoTQ3ZYK-Qw",
                      title="A Carnatic lesson on video", **metadata):
    """A source of the kind we may look at but not take."""
    source = training.store.save_candidate(LearningSource(
        title=title, url=url, source_type="lead",
        description="A lesson somebody published.",
        accessibility_status=Accessibility.METADATA_ONLY,
        provider="web", metadata=metadata))
    training.queue.enqueue([source], "Carnatic lesson")
    return source


# --------------------------------------------------------------------------
# honesty about what was not reached (section 4, rule 9)
# --------------------------------------------------------------------------
def test_an_unfetched_source_says_so_in_the_specifications_own_words(training):
    _queue_web_source(training)
    report = training.learn_one_now()

    assert METADATA_ONLY_NOTICE in report.summary
    assert report.learned == []
    assert report.analysed_representation == "none"
    assert report.confidence == 0.0


def test_an_unfetched_source_adds_no_knowledge(training):
    """The failure that would matter most: a source nobody read producing
    entries that later look like findings."""
    _queue_web_source(training)
    training.learn_one_now()
    assert training.search_knowledge() == []


def test_an_unfetched_source_offers_the_two_ways_forward(training):
    _queue_web_source(training)
    report = training.learn_one_now()
    offers = " ".join(report.next_learning)
    assert "Upload the source file manually" in offers
    assert "Provide transcript" in offers


def test_supplying_the_file_turns_a_lead_into_a_lesson(training, tmp_path):
    """Section 4's escape hatch, end to end: the creator hands over audio they
    are entitled to use, and the same source becomes learnable."""
    source = _queue_web_source(training, metadata={"raaga": "Keeravani"})
    report = training.learn_one_now()
    assert report.analysed_representation == "none"

    path = tmp_path / "Keeravani-alapana.wav"
    audio, _, _ = lesson_signal(talk_seconds=3.0, sung_seconds=8.0)
    sf.write(path, audio, ANALYSIS_SR)
    assert training.supply_file(source.source_id, str(path))

    run = training.queue.relearn(training.store.candidate(source.source_id))
    second = training.learn_one_now()
    assert second.analysed_representation != "none"
    assert second.learned
    assert training.store.run(run.run_id).status == RunStatus.COMPLETED


def test_supplying_a_transcript_is_read_but_never_claimed_to_be_heard(
        training):
    source = _queue_web_source(training, metadata={"raaga": "Kambhoji"})
    training.learn_one_now()
    training.supply_transcript(
        source.source_id,
        "Today we learn Kambhoji. The arohanam is S R2 G3 M1 P D2 S. "
        "This lesson is set to Adi tala and uses kampita gamaka throughout.")
    training.queue.relearn(training.store.candidate(source.source_id))
    report = training.learn_one_now()

    assert report.analysed_representation.startswith("transcript")
    assert report.learned
    assert any("no audio" in limit for limit in report.honest_limits)
    assert any("stated rather than" in text or "not been verified" in text
               for text in report.practical_application + report.honest_limits)


def test_a_stated_phrase_is_not_given_to_the_composer(training, agent_repo):
    """The line that keeps 'we read that this is a phrase' from becoming
    'this is a phrase' in the music."""
    source = _queue_web_source(training, metadata={"raaga": "Kambhoji"})
    training.learn_one_now()
    training.supply_transcript(
        source.source_id,
        "In Kambhoji we sing S R2 G3 M1 P D2 S and then P D2 S R2 G3 M1.")
    training.queue.relearn(training.store.candidate(source.source_id))
    training.learn_one_now()

    assert training.search_knowledge(category="phrase")
    assert not agent_repo.phrases(raaga="Kambhoji", limit=50)


# --------------------------------------------------------------------------
# duplicates (section 10)
# --------------------------------------------------------------------------
def test_a_lesson_already_learned_is_flagged_on_the_next_search(training):
    results = training.search("Keeravani characteristic phrases")
    training.add_to_queue([results[0].source_id])
    training.learn_one_now()

    again = training.search("Keeravani characteristic phrases")
    same = next(r for r in again if r.title == results[0].title)
    assert same.previously_learned


def test_relearning_keeps_the_earlier_report(training):
    """Section 10: a new run rather than a destroyed one."""
    results = training.search("Keeravani characteristic phrases")
    first_run = training.add_to_queue([results[0].source_id])[0]
    training.learn_one_now()
    first_report = training.render_report(first_run.run_id)

    second_run = training.relearn(results[0].source_id)
    training.learn_one_now()

    assert second_run.run_id != first_run.run_id
    assert second_run.supersedes == first_run.run_id
    assert training.render_report(first_run.run_id) == first_report
    assert training.render_report(second_run.run_id)


# --------------------------------------------------------------------------
# the creator's controls (section 13)
# --------------------------------------------------------------------------
def test_a_source_can_be_removed_from_the_queue(training):
    results = training.search("Kambhoji lesson")
    runs = training.add_to_queue([r.source_id for r in results[:2]])
    training.remove_from_queue(runs[0].run_id)
    remaining = [r["run_id"] for r in training.queue_snapshot()]
    assert runs[0].run_id not in remaining
    assert runs[1].run_id in remaining


def test_a_failed_source_can_be_retried(training):
    source = training.store.save_candidate(LearningSource(
        title="Broken exercise", url="raaga-exercise://NotARaaga/identity",
        metadata={"raaga": "NotARaaga", "topic": "identity"}))
    training.queue.enqueue([source], "x")
    training.learn_one_now()
    run = training.store.runs()[0]
    assert run.status == RunStatus.FAILED

    training.retry(run.run_id)
    assert training.store.run(run.run_id).status == RunStatus.QUEUED


def test_knowledge_can_be_marked_incorrect_and_leaves_a_trace(training):
    """Section 13's last line: nothing important changes silently."""
    results = training.search("Keeravani characteristic phrases")
    training.add_to_queue([results[0].source_id])
    training.learn_one_now()
    entry = training.search_knowledge()[0]

    training.mark_knowledge_incorrect(entry.knowledge_id, "I disagree")
    record = training.provenance(entry.knowledge_id)
    assert record["status"] == "rejected"
    assert any(row["kind"] == "knowledge.rejected"
               for row in record["audit"])


def test_knowledge_can_be_approved(training):
    results = training.search("Keeravani characteristic phrases")
    training.add_to_queue([results[0].source_id])
    training.learn_one_now()
    entry = training.search_knowledge()[0]
    training.approve_knowledge(entry.knowledge_id)
    assert training.provenance(entry.knowledge_id)["user_approved"]


# --------------------------------------------------------------------------
# failure and restart (sections 14, 15)
# --------------------------------------------------------------------------
def test_a_source_that_cannot_be_read_fails_without_taking_the_queue_with_it(
        training):
    broken = training.store.save_candidate(LearningSource(
        title="Broken", url="raaga-exercise://NotARaaga/identity",
        metadata={"raaga": "NotARaaga", "topic": "identity"}))
    good = training.search("Kambhoji lesson")[0]
    training.queue.enqueue([broken], "x")
    training.add_to_queue([good.source_id])

    training.learn_one_now()
    training.learn_one_now()

    statuses = {r["title"]: r["status"] for r in training.queue_snapshot()}
    assert statuses["Broken"] == RunStatus.FAILED
    assert statuses[good.title] == RunStatus.COMPLETED


def test_a_failed_source_still_produces_a_report(training):
    """Rule 4 has no exception for failure: a source that taught us nothing
    still has to say what happened."""
    broken = training.store.save_candidate(LearningSource(
        title="Broken", url="raaga-exercise://NotARaaga/identity",
        metadata={"raaga": "NotARaaga", "topic": "identity"}))
    training.queue.enqueue([broken], "x")
    training.learn_one_now()
    run = training.store.runs()[0]
    rendered = training.render_report(run.run_id)
    assert "could not be processed" in rendered
    assert "What I learned" in rendered


def test_a_run_interrupted_by_a_crash_is_returned_to_the_queue(
        training_settings, agent_repo):
    """Section 14's last line.  A run left mid-flight with nobody working on
    it must not sit in 'Analyzing' for ever."""
    from raagacomposer.training.controller import TrainingController

    first = TrainingController(training_settings, agent_repo=agent_repo)
    results = first.search("Kambhoji lesson")
    run = first.add_to_queue([results[0].source_id])[0]
    # Simulate the application being killed mid-analysis.
    first.store.update_run(run.run_id, status=RunStatus.ANALYZING,
                           progress=0.5)
    first.store.close()

    second = TrainingController(training_settings, agent_repo=agent_repo)
    try:
        recovered = second.store.run(run.run_id)
        assert recovered.status == RunStatus.QUEUED
        assert "after the application closed" in recovered.detail
        assert second.learn_one_now() is not None
    finally:
        second.close()


def test_the_queue_worker_processes_everything_it_is_given(training):
    results = training.search("Kambhoji lesson")
    training.add_to_queue([r.source_id for r in results[:2]])
    training.start_learning()

    deadline = time.time() + 120
    while time.time() < deadline:
        if not training.queue.pending() and not training.queue.running:
            break
        time.sleep(0.1)
    training.queue.stop()

    statuses = {r["status"] for r in training.queue_snapshot()}
    assert RunStatus.QUEUED not in statuses
    assert RunStatus.COMPLETED in statuses


# --------------------------------------------------------------------------
# history (section 12)
# --------------------------------------------------------------------------
def test_the_history_records_what_was_learned_and_from_where(training):
    results = training.search("Keeravani characteristic phrases")
    training.add_to_queue([results[0].source_id])
    training.learn_one_now()

    rows = training.training_history()
    assert len(rows) == 1
    row = rows[0]
    assert row["title"] == results[0].title
    assert row["search_phrase"] == "Keeravani characteristic phrases"
    assert row["knowledge_added"] > 0
    assert row["objectives"] > 0
    assert row["when"]


def test_the_history_can_be_filtered_by_topic(training):
    for phrase in ("Keeravani characteristic phrases", "Kambhoji gamaka"):
        results = training.search(phrase)
        training.add_to_queue([results[0].source_id])
        training.learn_one_now()

    assert len(training.training_history()) == 2
    assert len(training.training_history(topic="Kambhoji")) == 1


def test_the_totals_add_up(training):
    results = training.search("Kambhoji lesson")
    training.add_to_queue([r.source_id for r in results[:3]])
    training.learn_one_now()

    totals = training.totals()
    assert totals["sources_seen"] == 3
    assert totals["completed"] == 1
    assert totals["queued"] == 2
    assert totals["knowledge_items"] > 0
