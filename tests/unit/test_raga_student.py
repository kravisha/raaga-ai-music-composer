"""Unit tests for ``RagaStudent`` (``raagacomposer/agent/student.py``):
reiteration is built from the knowledge base and is true to it, a
counterexample really is invalid, ``perform`` answers the ladder's early
levels with a real claim and confidence, and a correction is written back as
knowledge."""
from __future__ import annotations

import pytest

from raagacomposer.agent.music_agent import MusicAgent
from raagacomposer.agent.student import RagaStudent
from raagacomposer.factory.models import Lesson, TestLevel, TestSpec
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.unit

UNIT_ID = "b02.arohanam:Keeravani"


def _teach(agent: MusicAgent, target_unit: str = "b06.prayogas:Keeravani",
          max_steps: int = 40) -> None:
    """Same path as ``tests/regression/test_agent_regressions.py``'s
    ``_teach_through_prayogas``: real research, real practice, no mocks."""
    for _ in range(max_steps):
        if agent.repo.progress(target_unit).status == "passed":
            break
        step = agent.learn_step()
        if step.action == "idle":
            break


@pytest.fixture
def taught_agent(settings, raagas):
    agent = MusicAgent(settings, raagas)
    _teach(agent)
    yield agent
    agent.close()


@pytest.fixture
def student(taught_agent):
    profile = taught_agent.profile()
    return RagaStudent(taught_agent, profile), profile


def _arohanam_lesson(taught_agent) -> Lesson:
    unit = taught_agent.curriculum.unit(UNIT_ID)
    assert unit is not None
    return Lesson(domain="carnatic-music", concept=UNIT_ID,
                 objective=unit.learning_goal, origin=UNIT_ID)


# --------------------------------------------------------------------------
# reiterate
# --------------------------------------------------------------------------
def test_reiteration_is_non_empty_and_true_to_the_knowledge_base(
        taught_agent, student):
    raga_student, profile = student
    lesson = _arohanam_lesson(taught_agent)
    reiteration = raga_student.reiterate(lesson)

    fact = taught_agent.repo.best_fact("Keeravani", "arohanam")
    assert fact is not None
    assert reiteration.restate
    assert fact.value in reiteration.restate

    assert reiteration.explain
    assert "permitted way up" in reiteration.explain

    assert reiteration.connect

    assert reiteration.example
    assert reiteration.apply_summary
    assert reiteration.self_check
    assert reiteration.retest_due_at > 0

    # The counterexample really is invalid: it uses a swara Keeravani does
    # not have, and it says so.
    assert reiteration.counterexample
    phrase_text = reiteration.counterexample.split("(")[0].strip()
    bases = {parse_swara(t)[0] for t in phrase_text.split()}
    assert not bases.issubset(set(raagas_allowed(taught_agent)))
    assert "does not have" in reiteration.counterexample or \
        "does not stay inside" in reiteration.counterexample


def raagas_allowed(agent):
    return agent.library.require("Keeravani").allowed


# --------------------------------------------------------------------------
# perform
# --------------------------------------------------------------------------
def _spec(level: TestLevel, skill_type: str, params: dict, raaga="Keeravani",
         seed: int = 11, guided=None, capability: str = UNIT_ID) -> TestSpec:
    payload = {"skill_type": skill_type, "params": dict(params),
              "raaga": raaga, "origin_unit": UNIT_ID}
    if guided is not None:
        payload["guided"] = guided
    return TestSpec(capability=capability, level=level, seed=seed,
                    payload=payload)


def test_perform_t1_recall_returns_a_claim_and_confidence(taught_agent, student):
    raga_student, _ = student
    test = _spec(TestLevel.T1_RECALL, "recall.fact", {"facts": ["arohanam"]})
    performance = raga_student.perform(test)
    assert performance.claim
    assert 0.0 < performance.confidence <= 1.0
    assert any(line.startswith("phrase:") or "arohanam" in line
              for line in performance.evidence) or performance.evidence
    assert performance.payload.get("report") is not None


def test_perform_t4_generation_returns_a_valid_invalid_claim(taught_agent, student):
    raga_student, _ = student
    test = _spec(TestLevel.T4_INDEPENDENT_APPLICATION, "generate.pattern",
                 {"length": 5}, guided=False)
    performance = raga_student.perform(test)
    assert performance.claim in ("valid", "invalid")
    assert 0.0 < performance.confidence <= 1.0
    assert performance.payload.get("report") is not None
    assert any(line.startswith("raaga:") for line in performance.evidence)
    assert any(line.startswith("score:") for line in performance.evidence)


def test_perform_t6_classification_returns_a_claim(taught_agent, student):
    raga_student, _ = student
    test = _spec(TestLevel.T6_ERROR_DETECTION, "classify.valid",
                 {"mode": "in_raaga_vs_out", "exercises": 1})
    performance = raga_student.perform(test)
    assert performance.claim
    assert 0.0 < performance.confidence <= 1.0
    assert performance.payload.get("report") is not None


def test_perform_explain_test_uses_knowledge_only(taught_agent, student):
    raga_student, _ = student
    test = _spec(TestLevel.T2_EXPLANATION, "explain", {"facts": ["arohanam"]})
    performance = raga_student.perform(test)
    assert performance.output
    assert performance.claim == performance.output
    assert 0.0 < performance.confidence <= 1.0


# --------------------------------------------------------------------------
# apply_correction
# --------------------------------------------------------------------------
def test_apply_correction_writes_a_fact_and_a_lesson(taught_agent, student):
    raga_student, _ = student
    lesson = _arohanam_lesson(taught_agent)
    before = len(taught_agent.repo.lessons(unit_id=UNIT_ID, include_applied=True))

    # A key that is not already seeded, so ``add_fact`` inserts a fresh row
    # rather than merging into an existing one at whatever confidence it
    # already had - the exact 0.9 the brief asks for is only guaranteed on
    # a first insert.
    correction = ("Keeravani's disputed_cadence_note is P by the library "
                  "(hard knowledge)")
    raga_student.apply_correction(correction, lesson)

    after = taught_agent.repo.lessons(unit_id=UNIT_ID, include_applied=True)
    assert len(after) == before + 1
    assert after[0].kind == "judge_correction"

    facts = taught_agent.repo.facts("Keeravani", "disputed_cadence_note")
    assert len(facts) == 1
    assert facts[0].value == "P"
    assert facts[0].confidence == 0.9
    assert "ruling" in facts[0].notes
