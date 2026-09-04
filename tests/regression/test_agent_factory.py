"""Handoff acceptance tests (docs/spec/agent_factory/06) on the Raga agent,
the F2 increment of docs/PLAN_agent_factory.md.

Every test here drives the real ``MusicAgent``, ``RagaStudent`` and
``RagaTrainer`` - no mocked knowledge, no model calls.  ``test_af04_judge``
is the one exception worth explaining: ``LearningCycle.run`` only convenes a
dispute when *no* hard rule can already decide the question (its
``hard_rule_settles`` check short-circuits ``should_convene`` before
``judge.convene`` is ever called), so a disagreement a hard rule such as
``RestingNoteRule`` *can* settle never reaches the Judge through the full
training-cycle pipeline - only a disagreement nothing can settle does, and
that always comes back unresolved.  This test therefore exercises the Judge
integration directly: a real ``Dispute`` built the way ``LearningCycle``
builds one, real ``hard_rules``, real ``judge.convene``, and a real
``RagaStudent.apply_correction``.  See this file's report for the same note
addressed to the coordinator.
"""
from __future__ import annotations

import pytest

from raagacomposer.agent.knowledge import Fact
from raagacomposer.agent.music_agent import MusicAgent
from raagacomposer.agent.practice import PracticeReport
from raagacomposer.agent.rules import hard_rules
from raagacomposer.agent.student import RagaStudent
from raagacomposer.agent.trainer import RagaTrainer
from raagacomposer.core.settings import Settings

# The factory core (raagacomposer/factory/{cycle,judge}.py) is being built by
# another agent at the same time as this file; imported defensively so this
# module still *collects* (and every test in it is skipped, not errored) on
# a checkout where the core has not landed yet.
try:
    from raagacomposer.factory.cycle import LearningCycle
    from raagacomposer.factory.judge import convene
    from raagacomposer.factory.models import Dispute
    FACTORY_CORE_PRESENT = True
except Exception:  # noqa: BLE001
    FACTORY_CORE_PRESENT = False

pytestmark = [pytest.mark.regression,
             pytest.mark.skipif(not FACTORY_CORE_PRESENT,
                                reason="factory core not built yet")]


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


def _cycle(agent: MusicAgent):
    profile = agent.profile()
    store = agent.factory_store()
    trainer = RagaTrainer(agent, store)
    student = RagaStudent(agent, profile)
    rules = hard_rules(agent.library, agent.phrase_index(
        agent.curriculum.current_raaga()))
    return LearningCycle(store, student, trainer, rules=rules), trainer, \
        student, store, profile


# --------------------------------------------------------------------------
# TEST 1 - reiteration
# --------------------------------------------------------------------------
def test_af01_reiteration(taught_agent):
    cycle, trainer, student, store, profile = _cycle(taught_agent)
    lesson = trainer.next_lesson(profile, store.results(profile.id))
    assert lesson is not None

    reiteration = student.reiterate(lesson)
    assert reiteration.restate
    assert reiteration.explain
    assert reiteration.example
    assert reiteration.apply_summary
    assert isinstance(reiteration.apply_score, float)

    check = trainer.check_reiteration(lesson, reiteration)
    assert check.accepted, check.notes


# --------------------------------------------------------------------------
# TEST 2 - adaptive testing
# --------------------------------------------------------------------------
def test_af02_adaptive_testing(taught_agent):
    """Repeated success on the same lesson earns a higher test level: the
    ladder the agent is offered climbs rather than repeating T0/T1 forever."""
    cycle, trainer, student, store, profile = _cycle(taught_agent)
    lesson = trainer.next_lesson(profile, store.results(profile.id))
    assert lesson is not None

    levels = []
    for _ in range(6):
        outcome = cycle.run(lesson)
        levels.extend(int(r.level) for r in outcome.results)
        if not outcome.results:
            break

    assert levels, "no tests were ever offered"
    assert max(levels) > min(levels), \
        f"the test ladder never climbed: {levels}"


# --------------------------------------------------------------------------
# TEST 3 - remediation
# --------------------------------------------------------------------------
def test_af03_remediation(taught_agent, monkeypatch):
    """Repeated failure changes what the trainer does next rather than
    repeating the same test forever."""
    def always_fails(unit, raaga_name="", seed=0, guidance=None):
        return PracticeReport(unit_id=unit.id, skill_type=unit.skill_type,
                              score=0.1, passed=False, detail="failed")
    monkeypatch.setattr(taught_agent.practice, "run", always_fails)

    cycle, trainer, student, store, profile = _cycle(taught_agent)
    lesson = trainer.next_lesson(profile, store.results(profile.id))
    assert lesson is not None

    remediation = None
    for _ in range(6):
        outcome = cycle.run(lesson)
        if outcome.remediation is not None:
            remediation = outcome.remediation
            break

    assert remediation is not None
    assert remediation.kind in ("guided", "different_practice", "level_down")


# --------------------------------------------------------------------------
# TEST 4 - judge
# --------------------------------------------------------------------------
def test_af04_judge(taught_agent):
    """A learned fact for "nyasa" that contradicts the library at high
    confidence makes the student's claim about a phrase's resting note
    differ from the trainer's; the ruling names RestingNoteRule and the
    student's correction is recorded (see the module docstring for why this
    calls the Judge directly rather than through a train_step)."""
    taught_agent.repo.add_fact(Fact(raaga="Keeravani", key="nyasa",
                                    value="S R2 N3", confidence=0.9))
    disputed = taught_agent.repo.best_fact("Keeravani", "nyasa")
    assert disputed is not None and disputed.disputed

    # The phrase ends on D1: not one of Keeravani's real resting notes
    # (S, P, G2), but the student, going by the disputed learned fact, would
    # call it valid.
    dispute = Dispute(
        agent_id=taught_agent.profile().id,
        question="does the phrase rest on a resting note?",
        student_claim="valid", trainer_claim="invalid",
        evidence_student=["phrase: S R2 G2 M1 D1", "raaga: Keeravani"],
        evidence_trainer=["phrase: S R2 G2 M1 D1", "raaga: Keeravani"],
        student_confidence=0.7, trainer_confidence=0.8)

    rules = hard_rules(taught_agent.library,
                       taught_agent.phrase_index("Keeravani"))
    ruling = convene(dispute, rules)

    assert ruling.decided_by == "RestingNoteRule"
    assert ruling.ruling == "invalid"
    assert ruling.accepted_claim == "trainer"
    assert ruling.correction_student

    student = RagaStudent(taught_agent, taught_agent.profile())
    student.apply_correction(ruling.correction_student, None)
    corrections = taught_agent.repo.lessons(raaga="Keeravani",
                                            kind="judge_correction",
                                            include_applied=True)
    assert corrections


# --------------------------------------------------------------------------
# TEST 8 - test evolution
# --------------------------------------------------------------------------
def test_af08_test_evolution(settings, raagas):
    """Over several training cycles the recorded tests' level climbs."""
    agent = MusicAgent(settings, raagas)
    try:
        for _ in range(8):
            agent.train_step()
        store = agent.factory_store()
        profile = agent.profile()
        results = sorted(store.results(profile.id, limit=1000),
                         key=lambda r: r.at)
        assert len(results) >= 2
        levels = [int(r.level) for r in results]
        assert max(levels) > levels[0], \
            f"the ladder never rose across 8 train_steps: {levels}"
    finally:
        agent.close()


# --------------------------------------------------------------------------
# REG-095 re-verified: train_step must not perturb learn_step
# --------------------------------------------------------------------------
def test_train_step_does_not_change_learn_step(tmp_path, settings, raagas):
    """REG-095's own harness (does the learning loop ever stall, does it
    reach good completion) gives the exact same step sequence whether or not
    a ``train_step`` was ever called on the agent - the new path is
    additive, not a change to ``learn_step``."""
    baseline_agent = MusicAgent(settings, raagas)
    try:
        seen_baseline = []
        for _ in range(60):
            step = baseline_agent.learn_step()
            seen_baseline.append((step.action, step.unit_id,
                                  round(step.score, 4), step.passed))
            if step.action == "idle":
                break
    finally:
        baseline_agent.close()

    other_settings = Settings.load()
    other_settings.projects_dir = settings.projects_dir
    other_settings.knowledge_db = str(tmp_path / "k_other.db")
    other_settings.factory_db = str(tmp_path / "f_other.db")
    other_settings.stt_provider = "none"
    other_settings.learning_corpus_dir = ""
    other_settings.learning_allow_web = False
    other_settings.learning_autostart = False

    trained_agent = MusicAgent(other_settings, raagas)
    try:
        trained_agent.train_step()
        seen_after_train = []
        for _ in range(60):
            step = trained_agent.learn_step()
            seen_after_train.append((step.action, step.unit_id,
                                     round(step.score, 4), step.passed))
            if step.action == "idle":
                break
    finally:
        trained_agent.close()

    assert seen_baseline == seen_after_train
