"""Learning objectives - specification section 6.

Objectives are what turn "the system watched a lesson" into a claim that can
be checked, so what matters here is that they follow the source rather than
being the same list every time, and that the creator can always change them.
"""
from __future__ import annotations

import pytest

from raagacomposer.training.models import (LearningRun, LearningSource,
                                           Objective, ObjectiveStatus)
from raagacomposer.training.objectives import LearningObjectiveService

pytestmark = pytest.mark.unit


@pytest.fixture
def objectives(training_store, raagas) -> LearningObjectiveService:
    return LearningObjectiveService(training_store, raagas)


def _source(topic: str = "identity", raaga: str = "Kambhoji",
            title: str = "") -> LearningSource:
    return LearningSource(
        title=title or f"{raaga}: {topic}",
        url=f"raaga-exercise://{raaga}/{topic}",
        metadata={"raaga": raaga, "topic": topic})


def test_every_source_gets_objectives(objectives):
    assert objectives.suggest(_source(), "Kamboji raga beginner lesson")


def test_objectives_follow_what_the_source_is_about(objectives):
    """A gamaka lesson should not be handed the same list as a scale lesson."""
    gamaka = objectives.suggest(_source("gamaka"), "Kambhoji gamaka technique")
    scale = objectives.suggest(_source("identity"), "Kambhoji arohanam")
    assert "gamakas" in gamaka[0].description.lower()
    assert gamaka[0].description != scale[0].description


def test_the_raaga_is_named_in_the_objectives(objectives):
    suggested = objectives.suggest(_source(), "Kamboji lesson")
    assert any("Kambhoji" in o.description for o in suggested)


def test_a_source_that_says_nothing_still_gets_sensible_objectives(objectives):
    """Falling back to first principles beats setting none at all."""
    bare = LearningSource(title="untitled", url="https://example.org/x")
    suggested = objectives.suggest(bare, "")
    assert suggested
    assert any("raaga" in o.description.lower() for o in suggested)


def test_objectives_start_not_started(objectives):
    assert all(o.status == ObjectiveStatus.NOT_STARTED
               for o in objectives.suggest(_source(), "Kambhoji"))


def test_a_knowledge_gap_raises_an_objective_up_the_list(training_store,
                                                         raagas, agent_repo):
    """Section 11: what the agent is missing is the best reason to ask for it.
    With nothing known about the raaga, the core facts should be asked for
    ahead of an incidental keyword match."""
    service = LearningObjectiveService(training_store, raagas, agent_repo)
    suggested = service.suggest(_source("mood"), "Kambhoji mood and usage")
    categories = [o.category for o in suggested[:3]]
    assert "scale" in categories or "raaga" in categories


# --------------------------------------------------------------------------
# the creator's edits (section 6, section 13)
# --------------------------------------------------------------------------
def test_objectives_can_be_replaced_added_and_removed(objectives,
                                                      training_store):
    source = training_store.save_candidate(_source())
    run = training_store.add_run(LearningRun(source_id=source.source_id))

    saved = objectives.set_objectives(
        run.run_id, [Objective(description="Identify the raaga.")])
    assert len(saved) == 1

    saved = objectives.add_objective(run.run_id, "Learn the gamakas.",
                                     "ornament")
    assert len(saved) == 2
    assert saved[1].user_defined

    saved = objectives.remove_objective(run.run_id, saved[0].objective_id)
    assert len(saved) == 1
    assert saved[0].description == "Learn the gamakas."


def test_edited_objectives_survive_a_restart(objectives, training_store,
                                             tmp_path):
    source = training_store.save_candidate(_source())
    run = training_store.add_run(LearningRun(source_id=source.source_id))
    objectives.add_objective(run.run_id, "Something I care about.")
    path = training_store.path
    training_store.close()

    from raagacomposer.training.store import TrainingStore
    reopened = TrainingStore(path)
    try:
        kept = reopened.objectives(run.run_id)
        assert [o.description for o in kept] == ["Something I care about."]
    finally:
        reopened.close()


def test_an_empty_objective_is_not_stored(objectives, training_store):
    source = training_store.save_candidate(_source())
    run = training_store.add_run(LearningRun(source_id=source.source_id))
    saved = objectives.set_objectives(run.run_id, [
        Objective(description="A real one."), Objective(description="   ")])
    assert len(saved) == 1
