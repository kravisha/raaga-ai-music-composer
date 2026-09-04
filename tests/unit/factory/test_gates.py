"""Each promotion-gate and release-gate check individually fails, then
passes."""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.factory.gates import (GateThresholds, promotion_gate,
                                         release_gate)
from raagacomposer.factory.models import (AgentProfile, AgentSpec, Dispute,
                                          DisputeStatus, MasteryLevel,
                                          MasteryRecord, Split, TestResult)
from raagacomposer.factory.store import FactoryStore

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


# -- promotion_gate: each check individually --------------------------------
def test_promotion_gate_fails_fresh(store: FactoryStore, profile: AgentProfile):
    report = promotion_gate(store, profile, "add_s")
    assert report.passed is False
    # no_critical_unresolved passes trivially - there are no disputes yet.
    assert set(report.failed_checks()) == {
        "mastery", "unseen_success", "stable", "calibrated"}


def test_promotion_gate_mastery_check(store: FactoryStore, profile: AgentProfile):
    record = MasteryRecord(agent_id=profile.id, concept="add_s",
                           level=MasteryLevel.L5_APPLY_INDEPENDENTLY)
    store.save_mastery(record)
    report = promotion_gate(store, profile, "add_s")
    assert report.checks["mastery"] is True
    assert "mastery" not in report.failed_checks()


def test_promotion_gate_no_critical_unresolved_check(store: FactoryStore,
                                                      profile: AgentProfile):
    from raagacomposer.factory.models import Lesson
    lesson = Lesson(domain="plural-rules", concept="add_s")
    store.save_lesson(lesson)
    dispute = Dispute(agent_id=profile.id, lesson_id=lesson.id, critical=True,
                      status=DisputeStatus.OPEN)
    store.save_dispute(dispute)
    report = promotion_gate(store, profile, "add_s")
    assert report.checks["no_critical_unresolved"] is False

    dispute.status = DisputeStatus.RESOLVED
    store.save_dispute(dispute)
    report = promotion_gate(store, profile, "add_s")
    assert report.checks["no_critical_unresolved"] is True


def test_promotion_gate_unseen_success_check(store: FactoryStore,
                                             profile: AgentProfile):
    from raagacomposer.factory.models import TestLevel, TestSpec
    test = TestSpec(capability="add_s", level=TestLevel.T4_INDEPENDENT_APPLICATION,
                    split=Split.VALIDATION)
    store.save_test(test)
    result = TestResult(test_id=test.id, agent_id=profile.id, level=test.level,
                        split=Split.VALIDATION, passed=True, score=1.0,
                        student_confidence=0.9)
    store.save_result(result)
    report = promotion_gate(store, profile, "add_s")
    assert report.checks["unseen_success"] is True


def test_promotion_gate_stable_and_calibrated_checks(store: FactoryStore,
                                                      profile: AgentProfile):
    from raagacomposer.factory.models import TestLevel, TestSpec
    for i in range(3):
        test = TestSpec(capability="add_s",
                        level=TestLevel.T4_INDEPENDENT_APPLICATION,
                        split=Split.TRAINING)
        store.save_test(test)
        result = TestResult(test_id=test.id, agent_id=profile.id,
                            level=test.level, split=Split.TRAINING,
                            passed=True, score=0.95, student_confidence=0.9)
        store.save_result(result)
    report = promotion_gate(store, profile, "add_s")
    assert report.checks["stable"] is True
    assert report.checks["calibrated"] is True


def test_promotion_gate_passes_fully(store: FactoryStore, profile: AgentProfile):
    from raagacomposer.factory.models import TestLevel, TestSpec
    record = MasteryRecord(agent_id=profile.id, concept="add_s",
                           level=MasteryLevel.L5_APPLY_INDEPENDENTLY)
    store.save_mastery(record)
    validation_test = TestSpec(capability="add_s",
                               level=TestLevel.T4_INDEPENDENT_APPLICATION,
                               split=Split.VALIDATION)
    store.save_test(validation_test)
    store.save_result(TestResult(test_id=validation_test.id, agent_id=profile.id,
                                 level=validation_test.level,
                                 split=Split.VALIDATION, passed=True, score=0.95,
                                 student_confidence=0.9))
    for i in range(3):
        test = TestSpec(capability="add_s",
                        level=TestLevel.T4_INDEPENDENT_APPLICATION,
                        split=Split.TRAINING)
        store.save_test(test)
        store.save_result(TestResult(test_id=test.id, agent_id=profile.id,
                                     level=test.level, split=Split.TRAINING,
                                     passed=True, score=0.95,
                                     student_confidence=0.9))
    report = promotion_gate(store, profile, "add_s")
    assert report.passed is True
    assert report.failed_checks() == []


# -- release_gate: each check individually ------------------------------------
def test_release_gate_fails_fresh_naming_every_missing_item(store: FactoryStore,
                                                             profile: AgentProfile):
    report = release_gate(store, profile)
    assert report.passed is False
    assert report.checks["capabilities_pass"] is False
    assert report.checks["rollback_stated"] is False
    assert report.checks["permissions_bounded"] is False
    assert report.checks["monitoring_stated"] is False
    assert report.checks["escalation_stated"] is False
    assert report.checks["knowledge_version_set"] is False
    assert len(report.reasons) >= 5


def test_release_gate_no_capabilities(store: FactoryStore):
    empty = AgentProfile(name="empty", domain="plural-rules", capabilities=[])
    store.save_profile(empty)
    report = release_gate(store, empty)
    assert report.checks["capabilities_pass"] is False
    assert any("no capabilities" in r for r in report.reasons)


def test_release_gate_passes_when_everything_stated_and_proven(
       store: FactoryStore):
    from raagacomposer.factory.models import TestLevel, TestSpec

    spec = AgentSpec(name="toy", role="student", domain="plural-rules",
                     capabilities=["add_s"], rollback="revert to prior profile",
                     permissions=["read_knowledge"], monitoring="log results",
                     escalation="page a human on repeated failure")
    profile = AgentProfile(name="toy", domain="plural-rules",
                           capabilities=["add_s"], spec=spec,
                           knowledge_version="v1")
    store.save_profile(profile)

    record = MasteryRecord(agent_id=profile.id, concept="add_s",
                           level=MasteryLevel.L5_APPLY_INDEPENDENTLY)
    store.save_mastery(record)

    hidden_test = TestSpec(capability="add_s",
                           level=TestLevel.T4_INDEPENDENT_APPLICATION,
                           split=Split.HIDDEN)
    store.save_test(hidden_test)
    store.save_result(TestResult(test_id=hidden_test.id, agent_id=profile.id,
                                 level=hidden_test.level, split=Split.HIDDEN,
                                 passed=True, score=0.95, student_confidence=0.9))
    for i in range(3):
        test = TestSpec(capability="add_s",
                        level=TestLevel.T4_INDEPENDENT_APPLICATION,
                        split=Split.TRAINING)
        store.save_test(test)
        store.save_result(TestResult(test_id=test.id, agent_id=profile.id,
                                     level=test.level, split=Split.TRAINING,
                                     passed=True, score=0.95,
                                     student_confidence=0.9))

    report = release_gate(store, profile)
    assert report.passed is True, report.reasons
    assert report.failed_checks() == []


def test_release_gate_requires_hidden_pass_not_just_validation(
       store: FactoryStore):
    from raagacomposer.factory.models import TestLevel, TestSpec

    spec = AgentSpec(name="toy", role="student", domain="plural-rules",
                     capabilities=["add_s"], rollback="revert",
                     permissions=["read"], monitoring="log",
                     escalation="escalate")
    profile = AgentProfile(name="toy", domain="plural-rules",
                           capabilities=["add_s"], spec=spec,
                           knowledge_version="v1")
    store.save_profile(profile)
    record = MasteryRecord(agent_id=profile.id, concept="add_s",
                           level=MasteryLevel.L5_APPLY_INDEPENDENTLY)
    store.save_mastery(record)

    # Only a VALIDATION pass, no HIDDEN pass.
    validation_test = TestSpec(capability="add_s",
                               level=TestLevel.T4_INDEPENDENT_APPLICATION,
                               split=Split.VALIDATION)
    store.save_test(validation_test)
    store.save_result(TestResult(test_id=validation_test.id, agent_id=profile.id,
                                 level=validation_test.level,
                                 split=Split.VALIDATION, passed=True, score=0.95,
                                 student_confidence=0.9))
    for i in range(3):
        test = TestSpec(capability="add_s",
                        level=TestLevel.T4_INDEPENDENT_APPLICATION,
                        split=Split.TRAINING)
        store.save_test(test)
        store.save_result(TestResult(test_id=test.id, agent_id=profile.id,
                                     level=test.level, split=Split.TRAINING,
                                     passed=True, score=0.95,
                                     student_confidence=0.9))

    report = release_gate(store, profile,
                          GateThresholds(require_hidden_pass=True))
    assert report.checks["capabilities_pass"] is False
    assert any("no passed hidden test" in r for r in report.reasons)
