"""AdaptiveTrainer: choose_test never returns retired or hidden, prefers
novelty; should_raise_difficulty needs calibration; should_remediate on a
repeated failure mode and on a confident failure; a second identical
remediation escalates to level_down."""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.factory.models import (AgentProfile, MasteryRecord,
                                          Split, TestLevel, TestResult,
                                          TestSpec)
from raagacomposer.factory.store import FactoryStore
from raagacomposer.factory.trainer import AdaptiveTrainer, TrainerPolicy

from tests.unit.factory.toy_domain import ToyTrainer

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path):
    s = FactoryStore(tmp_path / "factory.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def profile(store: FactoryStore) -> AgentProfile:
    p = AgentProfile(name="toy", domain="plural-rules", capabilities=["add_s"])
    store.save_profile(p)
    return p


@pytest.fixture
def adaptive(store: FactoryStore) -> AdaptiveTrainer:
    return AdaptiveTrainer(store, ToyTrainer())


def _result(test: TestSpec, agent_id: str, *, passed: bool, confidence: float,
           lesson_id: str = "lesson_1", failure_mode: str = "",
           at: float = 0.0) -> TestResult:
    r = TestResult(test_id=test.id, agent_id=agent_id, lesson_id=lesson_id,
                   level=test.level, split=test.split, passed=passed,
                   score=1.0 if passed else 0.0, student_confidence=confidence,
                   trainer_confidence=0.9, failure_mode=failure_mode)
    r.at = at
    return r


# -- choose_test --------------------------------------------------------------
def test_choose_test_never_returns_retired_or_hidden(store, profile, adaptive):
    training = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                        split=Split.TRAINING, novelty=0.5)
    retired = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                       split=Split.TRAINING, novelty=1.0, retired=True)
    hidden = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                      split=Split.HIDDEN, novelty=1.0)
    store.save_test(training)
    store.save_test(retired)
    store.save_test(hidden)

    from raagacomposer.factory.models import Lesson
    lesson = Lesson(domain="plural-rules", concept="add_s")

    chosen = adaptive.choose_test(profile, lesson, [training, retired, hidden], [])
    assert chosen is not None
    assert chosen.id == training.id


def test_choose_test_never_returns_a_test_retired_in_the_store(store, profile,
                                                                adaptive):
    from raagacomposer.factory.models import Lesson
    lesson = Lesson(domain="plural-rules", concept="add_s")
    a = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                split=Split.TRAINING, novelty=0.9)
    b = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                split=Split.TRAINING, novelty=0.1)
    store.save_test(a)
    store.save_test(b)
    store.retire_test(a.id)  # a is retired in the store, but the candidate
                              # object handed in still says retired=False

    chosen = adaptive.choose_test(profile, lesson, [a, b], [])
    assert chosen is not None
    assert chosen.id == b.id


def test_choose_test_prefers_novelty(store, profile, adaptive):
    from raagacomposer.factory.models import Lesson
    lesson = Lesson(domain="plural-rules", concept="add_s")
    low = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                   split=Split.TRAINING, novelty=0.2)
    high = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                    split=Split.TRAINING, novelty=0.9)
    store.save_test(low)
    store.save_test(high)

    chosen = adaptive.choose_test(profile, lesson, [low, high], [])
    assert chosen is not None
    assert chosen.id == high.id


def test_choose_test_returns_none_with_no_candidates(store, profile, adaptive):
    from raagacomposer.factory.models import Lesson
    lesson = Lesson(domain="plural-rules", concept="add_s")
    assert adaptive.choose_test(profile, lesson, [], []) is None


# -- should_raise_difficulty ---------------------------------------------------
def test_should_raise_difficulty_needs_calibration(adaptive):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION)
    calibrated = [
        _result(test, "agent_1", passed=True, confidence=0.9, at=i)
        for i in range(3)]
    assert adaptive.should_raise_difficulty(calibrated) is True

    uncalibrated = [
        _result(test, "agent_1", passed=True, confidence=0.1, at=i)
        for i in range(3)]
    assert adaptive.should_raise_difficulty(uncalibrated) is False

    too_few = calibrated[:2]
    assert adaptive.should_raise_difficulty(too_few) is False

    mixed = calibrated[:2] + [_result(test, "agent_1", passed=False,
                                      confidence=0.9, at=5)]
    assert adaptive.should_raise_difficulty(mixed) is False


# -- should_remediate -----------------------------------------------------------
def test_should_remediate_on_repeated_failure_mode(adaptive):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION)
    recent = [
        _result(test, "agent_1", passed=False, confidence=0.3,
               failure_mode="wrong_suffix", at=0),
        _result(test, "agent_1", passed=False, confidence=0.3,
               failure_mode="wrong_suffix", at=1),
    ]
    assert adaptive.should_remediate(recent) is True


def test_should_remediate_on_confident_failure(adaptive):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION)
    recent = [_result(test, "agent_1", passed=False, confidence=0.9, at=0)]
    assert adaptive.should_remediate(recent) is True


def test_should_not_remediate_on_low_confidence_single_failure(adaptive):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION)
    recent = [_result(test, "agent_1", passed=False, confidence=0.3, at=0)]
    assert adaptive.should_remediate(recent) is False


def test_should_not_remediate_on_pass(adaptive):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION)
    recent = [_result(test, "agent_1", passed=True, confidence=0.9, at=0)]
    assert adaptive.should_remediate(recent) is False


# -- after_result: remediation escalates on repeat -----------------------------
def test_repeated_remediation_escalates_to_level_down(store, profile):
    from raagacomposer.factory.models import Lesson

    class AlwaysSameTrainer(ToyTrainer):
        def remediate(self, profile, lesson, failures):
            return __import__("raagacomposer.factory.models",
                              fromlist=["Remediation"]).Remediation(
                kind="guided", lesson_id=lesson.id, detail="same every time")

        def learn_from(self, result):
            pass

    domain_trainer = AlwaysSameTrainer()
    adaptive = AdaptiveTrainer(store, domain_trainer,
                               TrainerPolicy(remediate_after_repeats=1))
    lesson = Lesson(domain="plural-rules", concept="add_s")
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                    split=Split.TRAINING)
    store.save_test(test)

    fail1 = _result(test, profile.id, passed=False, confidence=0.9,
                    lesson_id=lesson.id, failure_mode="wrong_suffix", at=1)
    store.save_result(fail1)
    remediation1 = adaptive.after_result(profile, lesson, test, fail1)
    assert remediation1 is not None
    assert remediation1.kind == "guided"

    fail2 = _result(test, profile.id, passed=False, confidence=0.9,
                    lesson_id=lesson.id, failure_mode="wrong_suffix", at=2)
    store.save_result(fail2)
    remediation2 = adaptive.after_result(profile, lesson, test, fail2)
    assert remediation2 is not None
    assert remediation2.kind == "level_down"
