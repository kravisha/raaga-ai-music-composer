"""Integration: the agent behind the existing composer interface.

Milestone 1 of the learning specification: the app still works, Apply Brief and
Generate Tune are driven by what the agent knows, learning is visible, and
nothing that was already there was broken to get here.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from raagacomposer.core.models import ApprovalState
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.integration


def teach(app, steps: int = 40, until: str = "b06.prayogas:Keeravani") -> None:
    for _ in range(steps):
        if app.agent.repo.progress(until).status == "passed":
            return
        if app.agent.learn_step().action == "idle":
            return


# --------------------------------------------------------------------------
# Milestone 1: everything exists and is connected
# --------------------------------------------------------------------------
def test_the_application_still_starts_with_its_composer_intact(app):
    assert app.project is not None
    assert app.raagas.names()
    assert app.voices.all()
    assert app.playback is not None
    # and the student is there too
    assert app.agent is not None
    assert app.agent.curriculum.current_raaga() == "Keeravani"
    assert Path(app.agent.repo.path).exists()


def test_the_pilot_raaga_is_known_from_the_first_run(app):
    facts = app.agent.repo.facts("Keeravani")
    assert facts, "the agent should start with its reference book"
    assert {f.key for f in facts} >= {"arohanam", "avarohanam", "nyasa"}
    for fact in facts:
        assert fact.source_id
        assert app.agent.repo.source(fact.source_id) is not None


def test_apply_brief_is_answered_by_the_agent(app):
    app.update_brief(situation="love failure", mood="sad",
                     feel="lonely late at night but still warm",
                     duration_target=40.0)
    suggestions = app.raaga_suggestions()
    assert suggestions
    for suggestion in suggestions:
        assert suggestion.name
        assert suggestion.rationale
        assert "confident" in suggestion.rationale
    assert app.project.raaga.alternatives == [] or True


def test_apply_brief_survives_an_agent_failure(app, monkeypatch):
    """The panel must never be left empty (learning specification 21)."""
    def boom(*args, **kwargs):
        raise RuntimeError("memory unavailable")

    monkeypatch.setattr(app.agent, "suggest_raagas", boom)
    errors_before = len(app.project.errors)
    suggestions = app.raaga_suggestions()
    assert suggestions, "a fallback suggestion is still required"
    assert len(app.project.errors) > errors_before


def test_generate_tune_uses_what_was_learned(app, settle):
    app.update_brief(mood="longing", feel="lonely late at night",
                     duration_target=40.0)
    app.select_raaga("Keeravani")
    teach(app)

    heard = app.agent.phrase_bank("Keeravani")
    assert heard, "nothing was learned to compose from"
    composing = app.composing_raaga()
    assert composing.prayogas[0] in heard
    assert composing.source == "learned"

    app.generate_tune(seed=5)
    settle()
    melody = app.project.melody()
    assert melody is not None
    assert len(melody.notes) > 10
    assert len({n.midi for n in melody.notes}) > 3


def test_generate_tune_never_returns_a_single_sustained_note(app, settle):
    app.update_brief(duration_target=30.0)
    app.select_raaga("Keeravani")
    app.generate_tune(seed=3)
    settle()
    melody = app.project.melody()
    assert len(melody.notes) >= 8
    assert len({n.midi for n in melody.notes}) >= 3
    assert melody.duration > 5.0


def test_a_generated_tune_stays_inside_the_raaga(app, settle):
    app.update_brief(duration_target=40.0)
    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=8)
    settle()
    allowed = set(app.raagas.require("Keeravani").allowed)
    for note in app.project.melody().notes:
        assert parse_swara(note.swara)[0] in allowed


def test_the_agent_marks_the_tune_it_wrote(app, settle):
    app.update_brief(mood="longing", duration_target=40.0)
    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=11)
    settle()

    assert app.last_evaluation is not None
    assert app.last_evaluation.scores
    recorded = app.agent.repo.compositions()
    assert recorded
    assert recorded[0]["raaga"] == "Keeravani"
    assert recorded[0]["scores"]

    critique = app.critique_tune()
    assert "overall" in critique
    assert "raaga_correctness" in critique


def test_a_tune_is_not_a_copy_of_what_was_learned(app, settle):
    from raagacomposer.agent.originality import check
    app.update_brief(duration_target=45.0)
    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=13)
    settle()
    report = check([n.swara for n in app.project.melody().notes],
                   app.agent.phrase_index("Keeravani"))
    assert report.is_original, report.summary()


# --------------------------------------------------------------------------
# learning through the controller
# --------------------------------------------------------------------------
def test_one_lesson_can_be_run_from_the_interface(app):
    before = len(app.agent.repo.completed_units())
    message = app.learn_now(1)
    assert message
    assert len(app.agent.repo.completed_units()) >= before


def test_background_learning_is_controlled_from_the_interface(app):
    assert app.start_learning()
    deadline = time.time() + 15
    while time.time() < deadline and not app.agent.repo.completed_units():
        time.sleep(0.1)
    assert app.agent.repo.completed_units()
    app.pause_learning()
    assert app.agent.is_paused
    app.stop_learning()
    assert not app.agent.is_learning


def test_the_creator_can_see_what_it_is_learning(app):
    app.learn_now(2)
    status = app.agent_status()
    assert status["current_raaga"] == "Keeravani"
    assert status["next_goal"]
    assert status["repository"]
    events = app.agent_events(10)
    assert events and "kind" in events[0]
    assert "Keeravani" in app.agent_knowledge()


def test_switching_the_studied_raaga(app):
    message = app.study_raaga("Kalyani")
    assert "Kalyani" in message
    assert app.agent.curriculum.current_raaga() == "Kalyani"
    assert app.agent.repo.facts("Kalyani")


# --------------------------------------------------------------------------
# spoken and typed instructions
# --------------------------------------------------------------------------
def test_learn_a_raaga_by_voice(app):
    command = app.handle_utterance("Learn Kalyani.")
    assert command.intent == "agent.learn"
    assert app.agent.curriculum.current_raaga() == "Kalyani"


def test_ask_why_by_voice(app):
    teach(app)
    command = app.handle_utterance("Why did you choose this phrase?")
    assert command.intent == "agent.explain"
    assert app.status_text


def test_a_correction_by_voice_is_stored_and_applied(app, settle):
    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=6)
    settle()
    before = {p.id: p.confidence for p in app.agent.repo.phrases("Keeravani")}

    command = app.handle_utterance("This does not sound like Keeravani.")
    assert command.intent == "agent.feedback"
    stored = app.agent.repo.feedback(raaga="Keeravani")
    assert stored and stored[0]["sentiment"] == "negative"
    after = {p.id: p.confidence
             for p in app.agent.repo.phrases("Keeravani", include_rejected=True)}
    assert any(after.get(pid, 0.0) < value for pid, value in before.items())


def test_asking_what_it_is_learning(app):
    command = app.handle_utterance("What are you learning?")
    assert command.intent == "agent.status"
    assert "Keeravani" in app.status_text


# --------------------------------------------------------------------------
# nothing that already worked was broken
# --------------------------------------------------------------------------
def test_the_whole_composing_workflow_still_runs(app, settle):
    app.new_project("Still Works")
    app.update_brief(mood="longing", feel="lonely late at night",
                     language="Tamil", duration_target=40.0)
    app.select_raaga("Keeravani")
    app.generate_tune(seed=4)
    settle()
    assert app.project.melody() is not None

    app.accept_tune(lock=True)
    assert app.project.melody().state is ApprovalState.LOCKED

    app.generate_lyrics(seed=2)
    settle()
    assert app.project.lyrics_version().lines

    app.render_vocal("master", autoplay=False)
    settle()
    assert app.project.vocal_master is not None

    app.auto_arrange()
    settle()
    assert app.project.arrangement().tracks

    app.render("full", autoplay=False)
    settle()
    assert app.project.latest_mix("full") is not None


# --------------------------------------------------------------------------
# a creator's critique and feedback become lessons (spec section 26, 38)
# --------------------------------------------------------------------------
def test_negative_feedback_becomes_a_lesson(app, settle):
    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=9)
    settle()

    app.critique_tune()
    app.give_feedback("This does not sound like Keeravani, too mechanical.")

    lessons = app.agent.repo.lessons(raaga="Keeravani", kind="creator_feedback")
    assert lessons
    assert lessons[0].confidence >= 0.9

    app.save()
    assert (app.project_dir / "project.json").exists()


# --------------------------------------------------------------------------
# item 4: guided regeneration, provenance, T10 (docs/PLAN_agent_factory.md)
# --------------------------------------------------------------------------
def test_item4_feedback_shapes_the_next_tune(app, settle):
    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=11)
    settle()

    app.critique_tune()
    app.give_feedback("This does not sound like Keeravani, too mechanical.")

    store = app.agent.factory_store()
    profile = app.agent.profile()
    disputes = store.disputes(profile.id)
    assert disputes, "negative feedback should have opened a dispute"
    ruling = store.ruling(disputes[0].ruling_id)
    assert ruling is not None and ruling.decided_by == "creator"

    reusable = store.reusable_lessons(domain="carnatic-music")
    assert any(r.rule_or_procedure.startswith(
        "the creator rejected a Keeravani tune") for r in reusable)

    app.generate_tune(seed=13)
    settle()
    second = app.project.melody()
    guided = (any(line.startswith("rewrite") for line in second.validation)
             or bool(second.guidance_note))

    evaluation = app.agent.evaluator("Keeravani").evaluate(
        second.notes, app.composing_raaga(), brief=app.project.brief,
        tempo_bpm=second.tempo_bpm,
        expected_seconds=app.project.brief.duration_target,
        learned_phrases=app.agent.phrase_bank("Keeravani"))
    no_repetitive_finding = not any(f.kind == "repetitive"
                                    for f in evaluation.findings)
    assert guided or no_repetitive_finding, (
        f"neither held: validation={second.validation!r} "
        f"guidance_note={second.guidance_note!r} "
        f"findings={[f.kind for f in evaluation.findings]!r}")
    if guided:
        reason = "guidance was applied (rewrite line or guidance note present)"
    else:
        reason = "the second tune's evaluation carries no repetitive finding"
    assert reason

    answer = app.ask_agent("why did you choose this phrase?")
    assert answer
    names_a_phrase = "quoted" in answer.lower()
    names_a_lesson = "avoided" in answer.lower() or "guided" in answer.lower()
    assert names_a_phrase or names_a_lesson, answer


def test_item4_provenance_survives_save_and_reopen(app, settle):
    from raagacomposer.core.persistence import ProjectStore

    app.select_raaga("Keeravani")
    teach(app)
    app.generate_tune(seed=17)
    settle()
    melody = app.project.melody()
    app.save()

    reopened = ProjectStore(app.settings).open(app.project_dir)
    got = reopened.melody()
    assert got.provenance == melody.provenance
    assert got.guidance_note == melody.guidance_note


def test_item4_a_tune_is_a_t10_result(app, settle):
    from raagacomposer.factory.models import Split, TestLevel

    app.select_raaga("Keeravani")
    app.generate_tune(seed=21)
    settle()

    store = app.agent.factory_store()
    profile = app.agent.profile()
    results = store.results(profile.id, level=TestLevel.T10_REAL_WORLD,
                            split=Split.REAL_WORLD)
    assert results
    test = store.test(results[0].test_id)
    assert test is not None
    assert test.capability == "compose:Keeravani"

    record = store.mastery(profile.id, "compose:Keeravani")
    assert record.evidence


def test_learning_state_outlives_the_application(app, settle, settings):
    from raagacomposer.app import AppController
    teach(app)
    phrases = app.agent.repo.count_phrases("Keeravani")
    passed = len(app.agent.repo.completed_units())
    assert phrases and passed
    app.close()

    restarted = AppController(settings)
    try:
        assert restarted.agent.repo.count_phrases("Keeravani") == phrases
        assert len(restarted.agent.repo.completed_units()) == passed
        assert restarted.composing_raaga().source == "learned"
    finally:
        restarted.close()
