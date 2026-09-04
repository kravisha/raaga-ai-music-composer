"""Round trips for every table in FactoryStore, schema versioning, and the
reusable-lesson promotion rule."""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.factory.models import (AgentProfile, AgentSpec, Dispute,
                                          DisputeStatus, KnowledgeClass,
                                          Lesson, MasteryLevel, MasteryRecord,
                                          Maturity, Promotion, Reiteration,
                                          ReiterationCheck, ReusableLesson,
                                          Ruling, Split, TestLevel, TestResult,
                                          TestSpec, ValidationStatus)
from raagacomposer.factory.store import FactoryStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path: Path):
    s = FactoryStore(tmp_path / "factory.db")
    try:
        yield s
    finally:
        s.close()


def test_profile_round_trip(store: FactoryStore):
    spec = AgentSpec(name="toy", role="student", domain="plural-rules",
                     capabilities=["add_s"], rollback="revert to last profile",
                     permissions=["read"], monitoring="log every test",
                     escalation="page a human")
    profile = AgentProfile(name="toy", role="student", domain="plural-rules",
                           capabilities=["add_s"], spec=spec,
                           strengths=["preloaded:add_s"])
    store.save_profile(profile)

    back = store.profile(profile.id)
    assert back is not None
    assert back.id == profile.id
    assert back.capabilities == ["add_s"]
    assert back.strengths == ["preloaded:add_s"]
    assert back.spec is not None
    assert back.spec.rollback == "revert to last profile"
    assert back.spec.permissions == ["read"]

    store.profiles()  # does not raise
    assert [p.id for p in store.profiles(domain="plural-rules")] == [profile.id]
    assert store.profiles(domain="nothing-here") == []


def test_lesson_round_trip(store: FactoryStore):
    lesson = Lesson(domain="plural-rules", concept="add_s", objective="learn it",
                    prerequisites=["basics"], source_knowledge=["rule book"],
                    explanation="add s", examples=["cat -> cats"],
                    counterexamples=["cat -> cat"], practice_tasks=["try cat"],
                    test_tasks=["test cat"], expected_behavior="says cats",
                    common_errors=["forgets the s"], remediation=["retry"],
                    knowledge_class=KnowledgeClass.HARD, confidence=0.9)
    store.save_lesson(lesson)

    back = store.lesson(lesson.id)
    assert back is not None
    assert back.explanation == "add s"
    assert back.examples == ["cat -> cats"]
    assert back.knowledge_class == KnowledgeClass.HARD
    assert back.confidence == 0.9

    assert [l.id for l in store.lessons(domain="plural-rules")] == [lesson.id]
    assert [l.id for l in store.lessons(concept="add_s")] == [lesson.id]
    assert store.lessons(concept="nope") == []


def test_reiteration_round_trip(store: FactoryStore):
    r = Reiteration(lesson_id="lesson_1", agent_id="agent_1", restate="add s",
                    explain="because plural", connect="basics", example="cats",
                    counterexample="cat", apply_summary="used it",
                    apply_score=1.0, self_check="none")
    check = ReiterationCheck(restate_ok=True, explain_ok=True, connect_ok=True,
                             example_ok=True, counterexample_ok=True,
                             notes=["fine"])
    row_id = store.save_reiteration(r, check)
    assert isinstance(row_id, str) and row_id

    pairs = store.reiterations("agent_1")
    assert len(pairs) == 1
    got_r, got_check = pairs[0]
    assert got_r.restate == "add s"
    assert got_r.lesson_id == "lesson_1"
    assert got_check.accepted
    assert got_check.notes == ["fine"]

    assert store.reiterations("agent_1", lesson_id="lesson_1")
    assert store.reiterations("agent_1", lesson_id="nope") == []
    assert store.reiterations("nobody") == []


def test_test_and_result_round_trip(store: FactoryStore):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                    novelty=1.0, difficulty=0.4, split=Split.TRAINING,
                    payload={"word": "cat"})
    store.save_test(test)

    back = store.test(test.id)
    assert back is not None
    assert back.capability == "add_s"
    assert back.level == TestLevel.T3_CONTROLLED_APPLICATION
    assert back.payload == {"word": "cat"}

    found = store.tests(capability="add_s")
    assert [t.id for t in found] == [test.id]
    assert store.tests(level=TestLevel.T3_CONTROLLED_APPLICATION)
    assert store.tests(split=Split.TRAINING)
    assert store.tests(split=Split.HIDDEN) == []

    result = TestResult(test_id=test.id, agent_id="agent_1", lesson_id="lesson_1",
                        level=test.level, split=test.split, score=1.0,
                        passed=True, student_claim="cats",
                        student_confidence=0.9, trainer_claim="cats",
                        trainer_confidence=0.95, evidence=["cats is right"])
    store.save_result(result)

    results = store.results("agent_1")
    assert len(results) == 1
    assert results[0].student_claim == "cats"
    assert results[0].evidence == ["cats is right"]

    assert store.results("agent_1", capability="add_s")
    assert store.results("agent_1", capability="nope") == []
    assert store.results("agent_1", level=test.level)
    assert store.results("agent_1", split=Split.TRAINING)


def test_retire_test_moves_to_regression(store: FactoryStore):
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                    split=Split.TRAINING)
    store.save_test(test)

    store.retire_test(test.id)

    back = store.test(test.id)
    assert back is not None
    assert back.retired is True
    assert back.split == Split.REGRESSION

    assert store.tests(split=Split.TRAINING) == []
    assert store.tests(include_retired=True)
    assert store.tests() == []  # retired tests hidden by default


def test_dispute_and_ruling_round_trip(store: FactoryStore):
    dispute = Dispute(agent_id="agent_1", test_id="test_1", lesson_id="lesson_1",
                      question="plural of cat", student_claim="cats",
                      trainer_claim="cat", student_confidence=0.8,
                      trainer_confidence=0.6, critical=True)
    store.save_dispute(dispute)

    back = store.dispute(dispute.id)
    assert back is not None
    assert back.status == DisputeStatus.OPEN
    assert back.critical is True

    assert [d.id for d in store.disputes(agent_id="agent_1")] == [dispute.id]
    assert store.disputes(status=DisputeStatus.OPEN)
    assert store.disputes(status=DisputeStatus.RESOLVED) == []

    ruling = Ruling(dispute_id=dispute.id, ruling="cats", accepted_claim="student",
                    rejected_claim="trainer", rationale="hard rule",
                    confidence=1.0, decided_by="plural_hard_rule")
    store.save_ruling(ruling)

    got_ruling = store.ruling(ruling.id)
    assert got_ruling is not None
    assert got_ruling.ruling == "cats"

    updated_dispute = store.dispute(dispute.id)
    assert updated_dispute is not None
    assert updated_dispute.status == DisputeStatus.RESOLVED
    assert updated_dispute.ruling_id == ruling.id


def test_dispute_unresolved_status(store: FactoryStore):
    dispute = Dispute(agent_id="agent_1", question="plural of xyzzy")
    store.save_dispute(dispute)
    ruling = Ruling(dispute_id=dispute.id, ruling="unresolved",
                    needs_external_evidence=True)
    store.save_ruling(ruling)

    back = store.dispute(dispute.id)
    assert back is not None
    assert back.status == DisputeStatus.UNRESOLVED


def test_reusable_lesson_promotion_rule(store: FactoryStore):
    reusable = ReusableLesson(
        source_event="ruling_1", rule_or_procedure="add s after most words",
        knowledge_class=KnowledgeClass.DISPUTE_LESSON, confidence=0.7,
        scope_domain="plural-rules", scope_concept="add_s",
        source_agent_id="agent_1")
    store.save_reusable(reusable)

    back = store.reusable(reusable.id)
    assert back is not None
    assert back.validation_status == ValidationStatus.CANDIDATE
    assert back.validations == 0

    # The source agent's own validation never counts.
    self_validated = store.validate_reusable(reusable.id, "agent_1")
    assert self_validated.validations == 0
    assert self_validated.validation_status == ValidationStatus.CANDIDATE

    once = store.validate_reusable(reusable.id, "agent_2")
    assert once.validations == 1
    assert once.validation_status == ValidationStatus.CANDIDATE

    twice = store.validate_reusable(reusable.id, "agent_3")
    assert twice.validations == 2
    assert twice.validation_status == ValidationStatus.VALIDATED

    thrice = store.validate_reusable(reusable.id, "agent_4")
    assert thrice.validations == 3
    assert thrice.validation_status == ValidationStatus.SHARED

    assert store.reusable_lessons(status=ValidationStatus.SHARED)
    assert [r.id for r in store.reusable_lessons(domain="plural-rules",
                                                 concept="add_s")] == [reusable.id]

    store.deprecate_reusable(reusable.id, superseded_by="reuse_new")
    deprecated = store.reusable(reusable.id)
    assert deprecated is not None
    assert deprecated.deprecated is True
    assert deprecated.superseded_by == "reuse_new"
    assert deprecated.validation_status == ValidationStatus.DEPRECATED
    assert store.reusable_lessons() == []
    assert store.reusable_lessons(include_deprecated=True)


def test_mastery_round_trip(store: FactoryStore):
    absent = store.mastery("agent_1", "add_s")
    assert absent.level == MasteryLevel.L0_UNKNOWN

    record = MasteryRecord(agent_id="agent_1", concept="add_s",
                           level=MasteryLevel.L4_APPLY_WITH_GUIDANCE,
                           evidence=["result_1", "result_2"],
                           failures_at_level=1)
    store.save_mastery(record)

    back = store.mastery("agent_1", "add_s")
    assert back.level == MasteryLevel.L4_APPLY_WITH_GUIDANCE
    assert back.evidence == ["result_1", "result_2"]
    assert back.failures_at_level == 1

    table = store.mastery_table("agent_1")
    assert set(table.keys()) == {"add_s"}

    record.level = MasteryLevel.L5_APPLY_INDEPENDENTLY
    store.save_mastery(record)
    assert store.mastery("agent_1", "add_s").level == MasteryLevel.L5_APPLY_INDEPENDENTLY


def test_promotion_round_trip(store: FactoryStore):
    promo = Promotion(agent_id="agent_1", from_maturity=Maturity.S0_CREATED,
                      to_maturity=Maturity.S1_EFFECTIVE, evidence=["result_1"])
    store.save_promotion(promo)

    found = store.promotions("agent_1")
    assert len(found) == 1
    assert found[0].to_maturity == Maturity.S1_EFFECTIVE
    assert found[0].evidence == ["result_1"]
    assert store.promotions("nobody") == []


def test_metrics_and_stats(store: FactoryStore):
    store.record_metric("test.retired", 1.0, agent_id="agent_1", detail="test_1")
    store.record_metric("test.retired", 1.0, agent_id="agent_2", detail="test_2")

    all_metrics = store.metrics()
    assert len(all_metrics) == 2
    named = store.metrics("test.retired")
    assert len(named) == 2
    assert store.metrics("nope") == []

    stats = store.stats()
    assert stats["metrics"] == 2


def test_reopen_survives(tmp_path: Path):
    path = tmp_path / "factory.db"
    store = FactoryStore(path)
    profile = AgentProfile(name="toy", domain="plural-rules")
    store.save_profile(profile)
    store.close()

    reopened = FactoryStore(path)
    try:
        back = reopened.profile(profile.id)
        assert back is not None
        assert back.name == "toy"
    finally:
        reopened.close()


def test_newer_schema_refused(tmp_path: Path):
    path = tmp_path / "factory.db"
    store = FactoryStore(path)
    store.close()

    import sqlite3
    conn = sqlite3.connect(str(path))
    conn.execute("UPDATE meta SET value=? WHERE key='schema_version'",
                (str(FactoryStore.SCHEMA_VERSION + 1),))
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError):
        FactoryStore(path)


def test_close_is_idempotent(tmp_path: Path):
    store = FactoryStore(tmp_path / "factory.db")
    assert store.closed is False
    store.close()
    assert store.closed is True
    store.close()  # no error
