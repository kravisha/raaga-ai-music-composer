"""Specification section 19, step by step, on the real controller.

One test per numbered step of the acceptance demonstration, plus the section
20 rules that the demonstration is meant to prove.  Everything here goes
through :class:`TrainingController` with a real store on disk, real audio
rendered and really analysed - nothing is mocked, because what is being
checked is that the parts work when they meet.
"""
from __future__ import annotations

import pytest

from raagacomposer.training.models import (Accessibility, ObjectiveStatus,
                                           RunStatus)

pytestmark = pytest.mark.integration

PHRASE = "Kamboji raga beginner lesson"


@pytest.fixture
def searched(training):
    """Steps 1 to 4: open the tab, type the phrase, search."""
    return training, training.search(PHRASE)


# --------------------------------------------------------------------------
# steps 1 to 6
# --------------------------------------------------------------------------
def test_step_04_the_search_returns_about_ten_candidates(searched):
    _, results = searched
    assert 8 <= len(results) <= 10


def test_step_05_every_candidate_carries_what_the_table_needs(searched):
    """Section 3.2 names the columns; a row that cannot fill them is no use."""
    _, results = searched
    for source in results:
        assert source.title and source.url
        assert source.accessibility_status in Accessibility.LABELS
        assert 0.0 <= source.relevance_score <= 1.0
        assert source.source_type
        assert source.duration_label


def test_step_06_selecting_three_queues_exactly_three(searched):
    training, results = searched
    runs = training.add_to_queue([r.source_id for r in results[:3]])
    assert len(runs) == 3
    assert len(training.queue_snapshot()) == 3


def test_nothing_is_learned_that_was_not_chosen(searched):
    """Section 20 rule 1.  Searching is not approving."""
    training, results = searched
    assert training.queue_snapshot() == []
    training.add_to_queue([results[0].source_id])
    assert len(training.queue_snapshot()) == 1


# --------------------------------------------------------------------------
# steps 7 to 12
# --------------------------------------------------------------------------
@pytest.fixture
def learned(searched):
    """Steps 7 to 12: queue three, learn the first."""
    training, results = searched
    runs = training.add_to_queue([r.source_id for r in results[:3]])
    report = training.learn_one_now()
    return training, runs, report


def test_step_09_objectives_are_identified_for_every_source(learned):
    training, runs, _ = learned
    for run in runs:
        assert training.objectives(run.run_id)


def test_step_11_a_report_is_produced(learned):
    _, _, report = learned
    assert report is not None


def test_step_12_the_report_contains_everything_section_8_requires(learned):
    training, runs, _ = learned
    rendered = training.render_report(runs[0].run_id)
    for heading in ("Source", "Learning objectives", "Summary",
                    "What I understood", "What I learned",
                    "Existing knowledge confirmed",
                    "Conflicts and disagreements", "Practical application",
                    "Confidence", "Recommended next learning"):
        assert heading in rendered, f"the report has no '{heading}' section"


def test_understood_and_learned_are_separate_sections(learned):
    """Section 20 rule 5 - one narrative would collapse the distinction."""
    training, runs, _ = learned
    rendered = training.render_report(runs[0].run_id)
    assert rendered.index("What I understood") < rendered.index("What I learned")
    report = training.report(runs[0].run_id)
    assert report.understood
    assert isinstance(report.learned, list)


def test_every_objective_gets_a_verdict_including_a_negative_one(learned):
    """An objective the source did not cover must say so rather than being
    left looking unfinished."""
    training, runs, _ = learned
    statuses = {o.status for o in training.objectives(runs[0].run_id)}
    assert statuses
    assert ObjectiveStatus.NOT_STARTED not in statuses
    assert statuses & {ObjectiveStatus.LEARNED, ObjectiveStatus.PARTIAL}


# --------------------------------------------------------------------------
# steps 13 to 15
# --------------------------------------------------------------------------
def test_step_13_new_knowledge_reaches_the_knowledge_base(learned):
    training, _, _ = learned
    assert training.search_knowledge()


def test_step_13_every_learned_item_can_be_traced_to_its_source(learned):
    """Section 16 asks nine questions of every item; these are answered."""
    training, runs, _ = learned
    for entry in training.search_knowledge():
        record = training.provenance(entry.knowledge_id)
        assert record["where_from"]
        assert record["source_id"]
        assert record["learning_run"]
        assert record["evidence"]
        assert record["search_phrase"] == PHRASE
        assert record["confidence"] > 0
        assert record["status"]
        assert "ever_contradicted" in record
        assert "user_approved" in record


def test_step_14_the_first_source_is_completed(learned):
    training, runs, _ = learned
    assert training.store.run(runs[0].run_id).status == RunStatus.COMPLETED


def test_step_15_the_second_source_then_begins(learned):
    training, runs, _ = learned
    assert training.store.run(runs[1].run_id).status == RunStatus.QUEUED
    training.learn_one_now()
    assert training.store.run(runs[1].run_id).status == RunStatus.COMPLETED


def test_sources_are_processed_one_at_a_time(learned):
    """Section 3.3 and rule 2 - the first implementation is serial."""
    training, runs, _ = learned
    working = [r for r in training.queue_snapshot()
               if r["status"] not in (RunStatus.COMPLETED, RunStatus.QUEUED,
                                      RunStatus.SKIPPED, RunStatus.FAILED)]
    assert len(working) <= 1


# --------------------------------------------------------------------------
# steps 16 and 17
# --------------------------------------------------------------------------
def test_steps_16_and_17_everything_survives_close_and_reopen(
        learned, training_settings, agent_repo):
    from raagacomposer.training.controller import TrainingController

    training, runs, _ = learned
    before_history = len(training.training_history())
    before_knowledge = len(training.search_knowledge())
    before_queue = len(training.queue_snapshot())
    report_before = training.render_report(runs[0].run_id)
    training.close()

    reopened = TrainingController(training_settings, agent_repo=agent_repo)
    try:
        assert len(reopened.training_history()) == before_history
        assert len(reopened.search_knowledge()) == before_knowledge
        assert len(reopened.queue_snapshot()) == before_queue
        assert reopened.render_report(runs[0].run_id) == report_before
        assert reopened.objectives(runs[0].run_id)
    finally:
        reopened.close()


# --------------------------------------------------------------------------
# rule 12 - the point of the whole exercise
# --------------------------------------------------------------------------
def test_what_is_learned_reaches_the_music(learned, agent_repo):
    """Section 20 rule 12.  A training system whose findings never change a
    note the application plays has taught nobody anything."""
    training, _, _ = learned
    phrases = agent_repo.phrases(raaga="Kambhoji", limit=100)
    assert phrases, "nothing learned reached the composer's memory"
    assert any("training" in (agent_repo.source(p.source_id).provider or "")
               for p in phrases if p.source_id)
