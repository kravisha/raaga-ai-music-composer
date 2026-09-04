"""A full lesson cycle on the toy domain writes reiteration, tests, results
and mastery."""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.factory.cycle import LearningCycle
from raagacomposer.factory.models import MasteryLevel
from raagacomposer.factory.store import FactoryStore

from tests.unit.factory.toy_domain import HardRule, ToyStudent, ToyTrainer

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path):
    s = FactoryStore(tmp_path / "factory.db")
    try:
        yield s
    finally:
        s.close()


def test_lesson_cycle_writes_everything(store: FactoryStore):
    student = ToyStudent()
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    assert lesson is not None
    assert lesson.concept == "add_s"

    outcome = cycle.run(lesson, max_tests=2)

    assert outcome.lesson_id == lesson.id
    assert outcome.reiteration_accepted is True
    assert len(outcome.results) == 2
    assert outcome.mastery_after >= MasteryLevel.L3_CAN_EXPLAIN

    # reiteration persisted
    pairs = store.reiterations(student.profile.id, lesson_id=lesson.id)
    assert len(pairs) == 1
    assert pairs[0][0].restate == "add s"
    assert pairs[0][1].accepted

    # tests and results persisted
    saved_results = store.results(student.profile.id, capability="add_s")
    assert len(saved_results) == 2
    for result in saved_results:
        assert store.test(result.test_id) is not None

    # mastery persisted and reflects at least the reiteration evidence
    mastery = store.mastery(student.profile.id, "add_s")
    assert mastery.level >= MasteryLevel.L3_CAN_EXPLAIN
    assert mastery.evidence  # something was recorded


def test_lesson_cycle_advances_mastery_on_passes(store: FactoryStore):
    student = ToyStudent()
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    outcome = cycle.run(lesson, max_tests=1)

    assert outcome.mastery_after >= outcome.mastery_before
    assert outcome.advanced is True


def test_run_until_stops_when_curriculum_exhausted(store: FactoryStore):
    student = ToyStudent()
    trainer = ToyTrainer()  # 3 concepts
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    outcomes = cycle.run_until(max_cycles=10)
    assert len(outcomes) == 3
    concepts = {store.lesson(o.lesson_id).concept for o in outcomes}
    assert concepts == {"add_s", "sibilant_es", "y_to_ies"}
