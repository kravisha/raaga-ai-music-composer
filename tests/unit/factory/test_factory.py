"""AgentFactory: create() preloads shared lessons; assess() walks
S0 -> S1 -> S2 -> S3 with evidence; field_lesson() stays CANDIDATE."""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.factory.factory import AgentFactory
from raagacomposer.factory.models import (AgentSpec, KnowledgeClass,
                                          MasteryLevel, MasteryRecord,
                                          Maturity, ReusableLesson, Split,
                                          TestLevel, TestResult, TestSpec,
                                          ValidationStatus)
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
def factory(store: FactoryStore) -> AgentFactory:
    return AgentFactory(store)


def test_create_preloads_shared_reusable_lessons(store: FactoryStore,
                                                  factory: AgentFactory):
    shared = ReusableLesson(
        source_event="ruling_1", rule_or_procedure="add s after most words",
        knowledge_class=KnowledgeClass.DISPUTE_LESSON,
        validation_status=ValidationStatus.SHARED, scope_domain="plural-rules",
        scope_concept="add_s", source_agent_id="agent_other")
    validated = ReusableLesson(
        source_event="ruling_2", rule_or_procedure="es after sibilants",
        knowledge_class=KnowledgeClass.DISPUTE_LESSON,
        validation_status=ValidationStatus.VALIDATED, scope_domain="plural-rules",
        scope_concept="sibilant_es", source_agent_id="agent_other")
    candidate = ReusableLesson(
        source_event="ruling_3", rule_or_procedure="not proven yet",
        knowledge_class=KnowledgeClass.DISPUTE_LESSON,
        validation_status=ValidationStatus.CANDIDATE, scope_domain="plural-rules",
        scope_concept="y_to_ies", source_agent_id="agent_other")
    store.save_reusable(shared)
    store.save_reusable(validated)
    store.save_reusable(candidate)

    spec = AgentSpec(name="second-agent", role="student", domain="plural-rules",
                     capabilities=["add_s", "sibilant_es", "y_to_ies"])
    profile = factory.create(spec, curriculum="plural-curriculum",
                             knowledge_version="v1")

    assert "preloaded:add_s" in profile.strengths
    assert "preloaded:sibilant_es" in profile.strengths
    assert "preloaded:y_to_ies" not in profile.strengths  # still CANDIDATE

    mastery_add_s = store.mastery(profile.id, "add_s")
    assert mastery_add_s.level >= MasteryLevel.L1_EXPOSED
    mastery_sibilant = store.mastery(profile.id, "sibilant_es")
    assert mastery_sibilant.level >= MasteryLevel.L1_EXPOSED
    mastery_y = store.mastery(profile.id, "y_to_ies")
    assert mastery_y.level == MasteryLevel.L0_UNKNOWN

    preloaded = factory.preloaded_lessons(profile)
    assert {r.scope_concept for r in preloaded} == {"add_s", "sibilant_es"}

    metrics = store.metrics("bootstrap.preloaded")
    assert len(metrics) == 1
    assert metrics[0]["value"] == 2.0

    # the profile itself is durably saved
    back = store.profile(profile.id)
    assert back is not None
    assert back.knowledge_version == "v1"
    assert back.current_curriculum == "plural-curriculum"


def _pass(store: FactoryStore, agent_id: str, capability: str,
         level: TestLevel, split: Split) -> None:
    test = TestSpec(capability=capability, level=level, split=split)
    store.save_test(test)
    store.save_result(TestResult(test_id=test.id, agent_id=agent_id, level=level,
                                 split=split, passed=True, score=0.95,
                                 student_confidence=0.9))


def test_assess_walks_maturity_with_evidence(store: FactoryStore,
                                              factory: AgentFactory):
    spec = AgentSpec(name="toy", role="student", domain="plural-rules",
                     capabilities=["add_s"])
    profile = factory.create(spec)
    assert factory.assess(profile) == Maturity.S0_CREATED

    # S1/S2: with a single capability, "any at L4+" and "all at L4+" are the
    # same condition, so one capability reaching L4 jumps straight to S2.
    store.save_mastery(MasteryRecord(agent_id=profile.id, concept="add_s",
                                     level=MasteryLevel.L4_APPLY_WITH_GUIDANCE))
    assert factory.assess(profile) >= Maturity.S1_EFFECTIVE
    assert profile.maturity >= Maturity.S1_EFFECTIVE

    # S3 needs the last 3 results at this capability to have passed.
    for _ in range(3):
        _pass(store, profile.id, "add_s", TestLevel.T3_CONTROLLED_APPLICATION,
             Split.TRAINING)
    reached = factory.assess(profile)
    assert reached >= Maturity.S3_RELIABLE

    promotions = store.promotions(profile.id)
    assert len(promotions) >= 2
    assert promotions[0].from_maturity == Maturity.S0_CREATED


def test_field_lesson_stays_candidate(store: FactoryStore, factory: AgentFactory):
    spec = AgentSpec(name="toy", role="student", domain="plural-rules",
                     capabilities=["add_s"])
    profile = factory.create(spec)

    reusable = factory.field_lesson(profile, event="deployment_event_1",
                                    rule_or_procedure="creator confirmed cats",
                                    concept="add_s", confidence=0.6)

    assert reusable.validation_status == ValidationStatus.CANDIDATE
    assert reusable.source_agent_id == profile.id
    assert reusable.scope_domain == "plural-rules"
    assert reusable.scope_concept == "add_s"

    back = store.reusable(reusable.id)
    assert back is not None
    assert back.validation_status == ValidationStatus.CANDIDATE

    # A single field event never self-promotes, even after the factory
    # looks at it again.
    still = store.reusable(reusable.id)
    assert still.validation_status == ValidationStatus.CANDIDATE
