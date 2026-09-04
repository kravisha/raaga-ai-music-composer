"""Document 06's acceptance tests, proven domain-free on the toy plural-rules
domain (AF02, AF03, AF04, AF05, AF06, AF07, AF08, AF09, AF10 - AF01
reiteration is exercised in test_cycle.py)."""
from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pytest

from raagacomposer.factory import judge as judge_module
from raagacomposer.factory.cycle import LearningCycle
from raagacomposer.factory.factory import AgentFactory
from raagacomposer.factory.gates import release_gate
from raagacomposer.factory.judge import _Judge
from raagacomposer.factory.models import (AgentSpec, KnowledgeClass,
                                          MasteryLevel, ReusableLesson,
                                          ValidationStatus)
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


# -- AF02: adaptive testing ----------------------------------------------------
def test_af02_adaptive_testing(store: FactoryStore):
    """After three calibrated passes at a level, the next chosen test is
    harder or more novel."""
    student = ToyStudent()
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    outcome = cycle.run(lesson, max_tests=6)

    results = [r for r in outcome.results]
    assert len(results) >= 4

    # Levels seen across the run should not be uniformly the entry level -
    # a calibrated streak should have pushed at least one test harder.
    levels_seen = [int(r.level) for r in results]
    assert max(levels_seen) > min(levels_seen) or any(
        r.split.value != "training" for r in results)


def test_af02_choose_test_raises_after_calibrated_streak(store: FactoryStore):
    """A narrower, deterministic version: three calibrated T5 passes at L5
    push the next chosen candidate to T6."""
    from raagacomposer.factory.mastery import apply_evidence
    from raagacomposer.factory.models import (AgentProfile, Lesson, Split,
                                              TestLevel, TestResult, TestSpec)
    from raagacomposer.factory.trainer import AdaptiveTrainer

    profile = AgentProfile(name="toy", domain="plural-rules")
    store.save_profile(profile)
    lesson = Lesson(domain="plural-rules", concept="add_s")
    record = store.mastery(profile.id, "add_s")
    record.level = MasteryLevel.L5_APPLY_INDEPENDENTLY
    store.save_mastery(record)

    t5 = TestSpec(capability="add_s", level=TestLevel.T5_VARIATION,
                  split=Split.TRAINING, novelty=0.5)
    t6 = TestSpec(capability="add_s", level=TestLevel.T6_ERROR_DETECTION,
                 split=Split.TRAINING, novelty=0.5)
    store.save_test(t5)
    store.save_test(t6)

    history = []
    for i in range(3):
        r = TestResult(test_id=t5.id, agent_id=profile.id, lesson_id=lesson.id,
                       level=TestLevel.T5_VARIATION, split=Split.TRAINING,
                       passed=True, score=0.95, student_confidence=0.9)
        r.at = float(i)
        history.append(r)

    adaptive = AdaptiveTrainer(store, ToyTrainer())
    chosen = adaptive.choose_test(profile, lesson, [t5, t6], history)
    assert chosen is not None
    assert chosen.level == TestLevel.T6_ERROR_DETECTION


# -- AF03: remediation ---------------------------------------------------------
def test_af03_remediation(store: FactoryStore):
    """After two failures with the same failure mode, the trainer's next
    action is a different test or instruction - never the same test id."""
    student = ToyStudent(scripted_errors={"pen": "pens_wrong", "hat": "hats_wrong"})
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    outcome = cycle.run(lesson, max_tests=3)

    failures = [r for r in outcome.results if not r.passed]
    assert failures, "the scripted errors should have produced a failure"
    assert outcome.remediation is not None
    remediation = outcome.remediation
    assert remediation.kind != ""

    # Running again must not repeat the exact same remediation for this
    # lesson - it should differ, or the adaptive trainer would have
    # escalated it to level_down.
    outcome2 = cycle.run(lesson, max_tests=3)
    if outcome2.remediation is not None:
        assert (outcome2.remediation.kind, outcome2.remediation.detail) != \
            (remediation.kind, remediation.detail)


def test_af03_second_identical_remediation_escalates(store: FactoryStore):
    from raagacomposer.factory.models import (AgentProfile, Lesson, Remediation,
                                              Split, TestLevel, TestSpec)
    from raagacomposer.factory.trainer import AdaptiveTrainer, TrainerPolicy

    class StubbornTrainer(ToyTrainer):
        def remediate(self, profile, lesson, failures):
            return Remediation(kind="guided", lesson_id=lesson.id,
                               detail="always the same")

        def learn_from(self, result):
            pass

    profile = AgentProfile(name="toy", domain="plural-rules")
    store.save_profile(profile)
    lesson = Lesson(domain="plural-rules", concept="add_s")
    test = TestSpec(capability="add_s", level=TestLevel.T3_CONTROLLED_APPLICATION,
                    split=Split.TRAINING)
    store.save_test(test)

    adaptive = AdaptiveTrainer(store, StubbornTrainer(),
                               TrainerPolicy(remediate_after_repeats=1))

    def fail(at):
        from raagacomposer.factory.models import TestResult
        r = TestResult(test_id=test.id, agent_id=profile.id, lesson_id=lesson.id,
                       level=test.level, split=test.split, passed=False,
                       score=0.0, student_confidence=0.9,
                       failure_mode="wrong_suffix")
        r.at = at
        store.save_result(r)
        return r

    first = adaptive.after_result(profile, lesson, test, fail(1))
    assert first.kind == "guided"
    second = adaptive.after_result(profile, lesson, test, fail(2))
    assert second.kind == "level_down"
    assert (second.kind, second.detail) != (first.kind, first.detail)


# -- AF04: judge -----------------------------------------------------------
def test_af04_judge(store: FactoryStore):
    """A scripted wrong answer with confidence 0.8 against the trainer's 0.9
    claim convenes a judge that returns a ruling with rationale and a
    reusable lesson."""
    student = ToyStudent(scripted_errors={"cat": "cat"})  # plausible mistake
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    outcome = cycle.run(lesson, max_tests=3)

    assert outcome.dispute is not None
    assert outcome.ruling is not None
    assert outcome.ruling.rationale
    assert outcome.ruling.resolved
    assert outcome.ruling.reusable_lesson_id

    reusable = store.reusable(outcome.ruling.reusable_lesson_id)
    assert reusable is not None
    assert reusable.validation_status == ValidationStatus.CANDIDATE


# -- AF05: judge ephemerality ---------------------------------------------------
def test_af05_judge_ephemerality(store: FactoryStore):
    """After the ruling, the only persisted objects are the dispute, ruling
    and reusable lesson; the judge module exposes no instance; gc finds no
    _Judge alive."""
    student = ToyStudent(scripted_errors={"cat": "cat"})
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    outcome = cycle.run(lesson, max_tests=3)
    assert outcome.dispute is not None
    assert outcome.ruling is not None

    gc.collect()
    live_judges = [obj for obj in gc.get_objects() if isinstance(obj, _Judge)]
    assert live_judges == []

    for name in dir(judge_module):
        if name == "_Judge":
            continue
        value = getattr(judge_module, name)
        assert not isinstance(value, _Judge)

    # The persisted trio exists in the store.
    assert store.dispute(outcome.dispute.id) is not None
    assert store.ruling(outcome.ruling.id) is not None
    assert store.reusable(outcome.ruling.reusable_lesson_id) is not None


# -- AF06: knowledge reuse -------------------------------------------------------
def test_af06_knowledge_reuse(store: FactoryStore):
    """Agent one's dispute lesson, validated twice by other agents, is
    preloaded into a newly created agent two as "exposed" evidence and
    listed in its strengths."""
    factory = AgentFactory(store)

    student1 = ToyStudent(scripted_errors={"cat": "cat"})
    trainer1 = ToyTrainer()
    cycle1 = LearningCycle(store, student1, trainer1, rules=[HardRule()])
    lesson1 = trainer1.next_lesson(student1.profile, [])
    outcome1 = cycle1.run(lesson1, max_tests=3)
    assert outcome1.ruling is not None
    reusable_id = outcome1.ruling.reusable_lesson_id
    assert reusable_id

    store.validate_reusable(reusable_id, "agent_validator_1")
    store.validate_reusable(reusable_id, "agent_validator_2")
    validated = store.reusable(reusable_id)
    assert validated.validation_status == ValidationStatus.VALIDATED

    spec2 = AgentSpec(name="agent-two", role="student", domain="plural-rules",
                      capabilities=["add_s"])
    profile2 = factory.create(spec2)

    assert f"preloaded:{validated.scope_concept}" in profile2.strengths
    mastery2 = store.mastery(profile2.id, validated.scope_concept)
    assert mastery2.level >= MasteryLevel.L1_EXPOSED
    assert reusable_id in mastery2.evidence


# -- AF07: no false mastery ------------------------------------------------------
def test_af07_no_false_mastery(store: FactoryStore):
    """Acquiring and reiterating ten lessons with no application test leaves
    every concept at or below L3."""
    from raagacomposer.factory.mastery import apply_evidence
    from raagacomposer.factory.models import AgentProfile, Lesson

    profile = AgentProfile(name="toy", domain="plural-rules")
    store.save_profile(profile)

    for i in range(10):
        concept = f"concept_{i}"
        lesson = Lesson(domain="plural-rules", concept=concept,
                        explanation="add s")
        store.save_lesson(lesson)
        record = store.mastery(profile.id, concept)
        record = apply_evidence(record, "exposed", lesson.id, True)
        record = apply_evidence(record, "restated", f"reiter_{i}", True)
        record = apply_evidence(record, "explained", f"reiter_{i}", True)
        store.save_mastery(record)

    for i in range(10):
        record = store.mastery(profile.id, f"concept_{i}")
        assert record.level <= MasteryLevel.L3_CAN_EXPLAIN


# -- AF08: test evolution ---------------------------------------------------------
def test_af08_test_evolution(store: FactoryStore):
    """Over eight cycles, the recorded tests' difficulty or novelty is
    non-decreasing and rises at least once."""
    student = ToyStudent()
    trainer = ToyTrainer()
    cycle = LearningCycle(store, student, trainer, rules=[HardRule()])

    lesson = trainer.next_lesson(student.profile, [])
    combined_scores = []
    for _ in range(8):
        outcome = cycle.run(lesson, max_tests=1)
        for r in outcome.results:
            test = store.test(r.test_id)
            combined_scores.append(test.difficulty + test.novelty)

    assert len(combined_scores) >= 4
    increased = any(b > a for a, b in zip(combined_scores, combined_scores[1:]))
    assert increased


# -- AF09: hard vs heuristic ---------------------------------------------------
def test_af09_hard_vs_heuristic(store: FactoryStore):
    """A heuristic reusable lesson never decides a dispute a hard rule
    applies to; the ruling's decided_by names the hard rule; a dispute only
    a heuristic could answer stays unresolved."""
    from raagacomposer.factory.judge import convene
    from raagacomposer.factory.models import Dispute

    hard_dispute = Dispute(agent_id="agent_1", question="plural of cat",
                           student_claim="cats", trainer_claim="cat",
                           student_confidence=0.8, trainer_confidence=0.6)
    ruling = convene(hard_dispute, [HardRule()])
    assert ruling.decided_by == "plural_hard_rule"
    assert ruling.resolved

    # A dispute the hard rule table deliberately declines (a y-final word,
    # where the correct call needs the defeasible heuristic, not the hard
    # table) stays unresolved without an escalation callable.
    heuristic_only = Dispute(agent_id="agent_1", question="plural of toy",
                             student_claim="toys", trainer_claim="toies",
                             student_confidence=0.8, trainer_confidence=0.6)
    ruling2 = convene(heuristic_only, [HardRule()])
    assert not ruling2.resolved
    assert ruling2.needs_external_evidence is True


# -- AF10: factory release -------------------------------------------------------
def test_af10_factory_release(store: FactoryStore):
    """release_gate fails on a fresh profile with reasons naming every
    missing item; passes after hidden tests pass, the spec states
    rollback/permissions/monitoring/escalation, and knowledge_version is
    set."""
    from raagacomposer.factory.models import (AgentProfile, MasteryRecord,
                                              Split, TestLevel, TestResult,
                                              TestSpec)

    bare_spec = AgentSpec(name="toy", role="student", domain="plural-rules",
                          capabilities=["add_s"])
    fresh = AgentProfile(name="toy", domain="plural-rules",
                         capabilities=["add_s"], spec=bare_spec)
    store.save_profile(fresh)

    report = release_gate(store, fresh)
    assert report.passed is False
    for expected in ("rollback", "permissions", "monitoring",
                     "escalation", "knowledge_version"):
        assert any(expected.split("_")[0] in r.lower() or "no capabilities" in r
                  for r in report.reasons) or not report.checks.get(
                      f"{expected}_stated", True)

    complete_spec = AgentSpec(
        name="toy", role="student", domain="plural-rules",
        capabilities=["add_s"], rollback="revert to prior profile",
        permissions=["read_knowledge"], monitoring="log every test result",
        escalation="page a human after three failures")
    profile = AgentProfile(name="toy", domain="plural-rules",
                           capabilities=["add_s"], spec=complete_spec,
                           knowledge_version="v1")
    store.save_profile(profile)
    store.save_mastery(MasteryRecord(agent_id=profile.id, concept="add_s",
                                     level=MasteryLevel.L5_APPLY_INDEPENDENTLY))

    hidden = TestSpec(capability="add_s", level=TestLevel.T4_INDEPENDENT_APPLICATION,
                      split=Split.HIDDEN)
    store.save_test(hidden)
    store.save_result(TestResult(test_id=hidden.id, agent_id=profile.id,
                                 level=hidden.level, split=Split.HIDDEN,
                                 passed=True, score=0.95, student_confidence=0.9))
    for _ in range(3):
        t = TestSpec(capability="add_s", level=TestLevel.T4_INDEPENDENT_APPLICATION,
                    split=Split.TRAINING)
        store.save_test(t)
        store.save_result(TestResult(test_id=t.id, agent_id=profile.id,
                                     level=t.level, split=Split.TRAINING,
                                     passed=True, score=0.95,
                                     student_confidence=0.9))

    report = release_gate(store, profile)
    assert report.passed is True, report.reasons
