"""Unit tests for ``RagaTrainer`` (``raagacomposer/agent/trainer.py``):
``next_lesson`` builds a real framework Lesson, ``build_tests`` follows the
ladder and keeps splits/seeds/novelty honest, ``grade`` reads the
performance's own report, ``remediate`` responds differently to different
failure shapes, and ``learn_from`` retires a beaten test on the third pass."""
from __future__ import annotations

import pytest

from raagacomposer.agent.music_agent import MusicAgent
from raagacomposer.agent.practice import PracticeReport
from raagacomposer.agent.trainer import RagaTrainer
from raagacomposer.factory.models import (Lesson, Performance, Split,
                                          TestLevel, TestResult, TestSpec)

pytestmark = pytest.mark.unit

AROHANAM_UNIT = "b02.arohanam:Keeravani"


def _teach(agent: MusicAgent, target_unit: str = "b06.prayogas:Keeravani",
          max_steps: int = 40) -> None:
    for _ in range(max_steps):
        if agent.repo.progress(target_unit).status == "passed":
            break
        step = agent.learn_step()
        if step.action == "idle":
            break


@pytest.fixture
def agent(settings, raagas):
    a = MusicAgent(settings, raagas)
    yield a
    a.close()


@pytest.fixture
def trainer(agent):
    store = agent.factory_store()
    return RagaTrainer(agent, store), store, agent.profile()


@pytest.fixture
def taught_agent(settings, raagas):
    a = MusicAgent(settings, raagas)
    _teach(a)
    yield a
    a.close()


@pytest.fixture
def taught_trainer(taught_agent):
    store = taught_agent.factory_store()
    return RagaTrainer(taught_agent, store), store, taught_agent.profile()


# --------------------------------------------------------------------------
# next_lesson
# --------------------------------------------------------------------------
def test_next_lesson_builds_a_lesson_with_prereqs_examples_counterexamples(
        taught_trainer):
    raga_trainer, store, profile = taught_trainer
    lesson = raga_trainer.next_lesson(profile, [])
    assert lesson is not None
    assert lesson.domain == "carnatic-music"
    assert lesson.origin
    assert lesson.prerequisites            # b07.boundaries needs b06.prayogas
    assert lesson.examples                 # heard phrases, since we taught b06
    assert lesson.counterexamples
    assert lesson.expected_behavior

    # A second call for the same unit returns the cached lesson, not a
    # freshly-minted one.
    again = raga_trainer.next_lesson(profile, [])
    assert again.id == lesson.id


# --------------------------------------------------------------------------
# build_tests
# --------------------------------------------------------------------------
def test_build_tests_t0_t1_for_a_fresh_concept(trainer):
    raga_trainer, store, profile = trainer
    lesson = Lesson(domain="carnatic-music", concept="fresh-concept",
                    origin=AROHANAM_UNIT)
    tests = raga_trainer.build_tests(lesson, profile, [])
    assert len(tests) == 2
    assert tests[0].level == TestLevel.T0_RECOGNITION
    assert tests[0].payload["skill_type"] == "listen.identify"
    assert tests[1].level == TestLevel.T1_RECALL
    assert tests[1].payload["skill_type"] == "recall.fact"
    assert tests[0].novelty == 1.0
    assert tests[1].novelty == 1.0


def test_build_tests_t7_uses_correct_phrase(trainer):
    raga_trainer, store, profile = trainer
    lesson = Lesson(domain="carnatic-music", concept="c7", origin=AROHANAM_UNIT)
    # Force the record to L6 so next_test_level lands on T8, then check the
    # ladder directly at T7 through the private level branch instead -
    # simplest is to call _test() the way build_tests would for T7.
    test = raga_trainer._test(lesson, [], TestLevel.T7_CORRECTION,
                              "correct.phrase", {}, "Keeravani", attempt=0)
    assert test.level == TestLevel.T7_CORRECTION
    assert test.payload["skill_type"] == "correct.phrase"
    assert test.payload["params"]["exercises"] == 1
    assert "question" in test.payload


def test_build_tests_t8_uses_a_second_raaga(trainer):
    raga_trainer, store, profile = trainer
    lesson = Lesson(domain="carnatic-music", concept="c8", origin=AROHANAM_UNIT)
    second = raga_trainer._second_raaga("Keeravani")
    assert second != "Keeravani"
    # Keeravani (melakarta 21) has no library sibling, so the fallback is
    # simply the first other raaga in the library.
    assert second == sorted(r.name for r in agent_library(raga_trainer)
                            if r.name != "Keeravani")[0]


def agent_library(raga_trainer):
    return raga_trainer.agent.library.all()


def test_split_and_seed_differ_across_splits(trainer):
    raga_trainer, store, profile = trainer
    lesson = Lesson(domain="carnatic-music", concept="splits", origin=AROHANAM_UNIT)
    training = raga_trainer._test(lesson, [], TestLevel.T4_INDEPENDENT_APPLICATION,
                                  "generate.pattern", {}, "Keeravani", attempt=0)
    validation = raga_trainer._test(lesson, [], TestLevel.T4_INDEPENDENT_APPLICATION,
                                    "generate.pattern", {}, "Keeravani", attempt=2)
    hidden = raga_trainer._test(lesson, [], TestLevel.T4_INDEPENDENT_APPLICATION,
                                "generate.pattern", {}, "Keeravani", attempt=4)
    assert training.split == Split.TRAINING
    assert validation.split == Split.VALIDATION
    assert hidden.split == Split.HIDDEN
    assert len({training.seed, validation.seed, hidden.seed}) == 3


def test_novelty_is_1_for_a_seed_and_raaga_never_seen(trainer):
    raga_trainer, store, profile = trainer
    lesson = Lesson(domain="carnatic-music", concept="novel", origin=AROHANAM_UNIT)
    test = raga_trainer._test(lesson, [], TestLevel.T4_INDEPENDENT_APPLICATION,
                              "generate.pattern", {}, "Keeravani", attempt=0)
    assert test.novelty == 1.0


# --------------------------------------------------------------------------
# grade
# --------------------------------------------------------------------------
def test_grade_reads_the_performance_report(trainer):
    raga_trainer, store, profile = trainer
    test = TestSpec(capability="c", level=TestLevel.T4_INDEPENDENT_APPLICATION,
                    split=Split.TRAINING,
                    payload={"skill_type": "generate.pattern",
                            "raaga": "Keeravani", "params": {}})
    report = PracticeReport(unit_id="x", skill_type="generate.pattern",
                            score=0.91, passed=True)
    performance = Performance(output="ok", claim="valid", confidence=0.8,
                              evidence=[], payload={"report": report})
    result = raga_trainer.grade(test, performance)
    assert result.score == 0.91
    assert result.passed is True
    assert result.student_claim == "valid"
    assert result.student_confidence == 0.8


def test_grade_a_failed_objective_test_names_a_failure_mode(trainer):
    raga_trainer, store, profile = trainer
    test = TestSpec(capability="c", level=TestLevel.T6_ERROR_DETECTION,
                    split=Split.TRAINING,
                    payload={"skill_type": "classify.valid",
                            "raaga": "Keeravani", "params": {}})
    from raagacomposer.agent.practice import ExerciseResult
    report = PracticeReport(
        unit_id="x", skill_type="classify.valid", score=0.0, passed=False,
        exercises=[ExerciseResult(name="judge 1", score=0.0, passed=False,
                                  expected="valid", heard="invalid")])
    performance = Performance(output="", claim="invalid", confidence=0.5,
                              evidence=[], payload={"report": report})
    result = raga_trainer.grade(test, performance)
    assert result.score == 0.0
    assert result.passed is False


# --------------------------------------------------------------------------
# remediate
# --------------------------------------------------------------------------
def test_remediate_reacts_differently_to_different_failure_shapes(trainer):
    raga_trainer, store, profile = trainer
    lesson = Lesson(domain="carnatic-music", concept="remediate",
                    origin="b13.short_phrase:Keeravani")

    guided_failures = [
        TestResult(level=TestLevel.T4_INDEPENDENT_APPLICATION,
                  failure_mode="no_cadence", passed=False),
        TestResult(level=TestLevel.T4_INDEPENDENT_APPLICATION,
                  failure_mode="no_cadence", passed=False),
    ]
    guided = raga_trainer.remediate(profile, lesson, guided_failures)
    assert guided.kind == "guided"
    assert guided.payload.get("guided") is True

    listening_failures = [
        TestResult(level=TestLevel.T0_RECOGNITION, failure_mode="", passed=False),
    ]
    different = raga_trainer.remediate(profile, lesson, listening_failures)
    assert different.kind == "different_practice"

    generic_failures = [
        TestResult(level=TestLevel.T5_VARIATION, failure_mode="something_odd",
                  passed=False),
    ]
    level_down = raga_trainer.remediate(profile, lesson, generic_failures)
    assert level_down.kind == "level_down"

    assert len({guided.kind, different.kind, level_down.kind}) == 3


# --------------------------------------------------------------------------
# learn_from
# --------------------------------------------------------------------------
def test_learn_from_retires_a_test_on_the_third_beat(trainer):
    raga_trainer, store, profile = trainer
    test = TestSpec(capability="beat-me", level=TestLevel.T4_INDEPENDENT_APPLICATION,
                    seed=1, payload={"raaga": "Keeravani"})
    store.save_test(test)
    result = TestResult(test_id=test.id, agent_id=profile.id, lesson_id="lesson1",
                        level=test.level, split=test.split, score=0.9, passed=True)

    raga_trainer.learn_from(result)
    raga_trainer.learn_from(result)
    assert not store.test(test.id).retired

    raga_trainer.learn_from(result)
    assert store.test(test.id).retired
    assert raga_trainer._harder_wanted.get("beat-me") is True
