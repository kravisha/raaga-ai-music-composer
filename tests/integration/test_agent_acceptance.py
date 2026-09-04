"""Integration: the learning specification's own acceptance tests.

Section 27, tests A to F, plus the seventeen items of Milestone 1 and the
things that must remain true afterwards: the agent learns, remembers across a
restart, and the music it writes changes because of what it learned.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.agent.music_agent import MusicAgent
from raagacomposer.core.models import CreativeBrief
from raagacomposer.raaga.library import parse_swara

pytestmark = pytest.mark.integration


@pytest.fixture
def agent(tmp_path, settings, raagas):
    settings.knowledge_db = str(tmp_path / "knowledge.db")
    settings.learning_corpus_dir = ""
    settings.learning_allow_web = False
    musician = MusicAgent(settings, raagas)
    try:
        yield musician
    finally:
        musician.close()


def study(agent: MusicAgent, steps: int = 40, until: str = "") -> list:
    """Run the loop, stopping early when a unit passes or nothing is left."""
    done = []
    for _ in range(steps):
        if until and agent.repo.progress(until).status == "passed":
            break
        step = agent.learn_step()
        done.append(step)
        if step.action == "idle":
            break
    return done


# ==========================================================================
# Test A - persistence
# ==========================================================================
def test_a_persistence_across_a_restart(tmp_path, settings, raagas):
    settings.knowledge_db = str(tmp_path / "knowledge.db")

    first = MusicAgent(settings, raagas)
    study(first, 30, until="b06.prayogas:Keeravani")
    facts_before = {f.key: f.value for f in first.repo.facts("Keeravani")}
    phrases_before = [tuple(p.swaras) for p in first.repo.phrases("Keeravani")]
    passed_before = set(first.repo.completed_units())
    assert facts_before and phrases_before and passed_before
    first.close()

    second = MusicAgent(settings, raagas)
    try:
        facts_after = {f.key: f.value for f in second.repo.facts("Keeravani")}
        phrases_after = [tuple(p.swaras) for p in second.repo.phrases("Keeravani")]
        assert facts_after == facts_before
        assert phrases_after == phrases_before
        assert set(second.repo.completed_units()) == passed_before
        # And it carries on from where it stopped rather than starting again.
        assert second.curriculum.next_unit() is not None
        assert second.curriculum.next_unit().id not in passed_before
    finally:
        second.close()


def test_a2_the_repository_is_a_real_file_that_survives(agent):
    study(agent, 10)
    path = Path(agent.repo.path)
    assert path.exists() and path.stat().st_size > 0
    assert agent.repo.schema_version >= 1
    reopened = KnowledgeRepository(path)
    try:
        assert reopened.count_phrases() >= 0
        assert reopened.state("current_raaga") == "Keeravani"
    finally:
        reopened.close()


# ==========================================================================
# Test B - Apply Brief
# ==========================================================================
def test_b_apply_brief_returns_ranked_suggestions_with_reasons(agent):
    brief = CreativeBrief(situation="love failure", mood="sad",
                          feel="lonely late at night but still warm")
    suggestions = agent.suggest_raagas(brief, limit=4)

    assert len(suggestions) >= 2
    for suggestion in suggestions:
        assert suggestion.reason.strip()
        assert 0.0 < suggestion.confidence <= 0.95
        assert agent.library.get(suggestion.name) is not None
    scores = [s.score for s in suggestions]
    assert scores == sorted(scores, reverse=True)
    assert any(n in [s.name for s in suggestions]
               for n in ("Keeravani", "Shivaranjani", "Charukesi", "Hindolam"))


def test_b2_the_brief_does_not_pretend_to_know_what_it_has_not_studied(agent):
    brief = CreativeBrief(mood="sad", feel="lonely late at night but still warm")
    suggestions = agent.suggest_raagas(brief, limit=6)
    studied = [s for s in suggestions if s.learned]
    unstudied = [s for s in suggestions if not s.learned]
    assert unstudied, "everything cannot be claimed as studied"
    for suggestion in unstudied:
        assert "not studied" in suggestion.rationale
    if studied:
        assert max(s.confidence for s in studied) >= \
            max(s.confidence for s in unstudied)


def test_b3_studying_a_raaga_raises_the_confidence_it_is_offered_with(agent):
    brief = CreativeBrief(mood="sad", feel="lonely late at night but still warm")
    before = next(s for s in agent.suggest_raagas(brief, 8)
                  if s.name == "Keeravani")
    study(agent, 30, until="b06.prayogas:Keeravani")
    after = next(s for s in agent.suggest_raagas(brief, 8)
                 if s.name == "Keeravani")
    assert after.confidence > before.confidence
    assert after.evidence


def test_b3b_a_bare_scale_is_offered_less_surely_than_a_curated_raaga(agent):
    """Score says how well it fits; confidence says how much is known.

    A melakarta the Stage 1 pack supplied and nobody curated can fit a brief
    perfectly - that is what the block model is for - but there is a parent
    scale and a starter tag behind it and nothing else, and the confidence it
    is offered with has to say so.
    """
    brief = CreativeBrief(mood="sad", feel="lonely late at night but warm")
    suggestions = agent.suggest_raagas(brief, limit=8)
    scales = [s for s in suggestions
              if agent.library.require(s.name).scale_only]
    curated = [s for s in suggestions
               if not agent.library.require(s.name).scale_only]
    assert scales and curated, [s.name for s in suggestions]
    assert max(s.confidence for s in curated) > \
        max(s.confidence for s in scales)


def test_b4_an_explicit_request_still_wins(agent):
    brief = CreativeBrief(mood="celebration", raaga_preference="Kalyani")
    assert agent.suggest_raagas(brief, 3)[0].name == "Kalyani"


# ==========================================================================
# Test C - raaga basics
# ==========================================================================
def test_c_the_agent_can_state_what_it_learned_about_keeravani(agent):
    study(agent, 30, until="b06.prayogas:Keeravani")

    arohanam = agent.repo.best_fact("Keeravani", "arohanam")
    avarohanam = agent.repo.best_fact("Keeravani", "avarohanam")
    assert arohanam and avarohanam
    assert arohanam.value.split()[0] == "S"
    assert arohanam.source_id and avarohanam.source_id

    answer = agent.explain("what is the arohanam of Keeravani?")
    assert "arohanam" in answer and "S R2 G2 M1 P D1 N3" in answer
    assert "confidence" in answer

    phrases = agent.repo.phrases("Keeravani")
    assert phrases, "no characteristic phrases were retained"
    assert "phrases" in agent.explain("why did you choose this phrase?")


def test_c2_the_learned_view_is_used_rather_than_the_book(agent):
    study(agent, 30, until="b06.prayogas:Keeravani")
    view, completeness = agent.raaga_for_composition("Keeravani")
    assert view is not None
    assert completeness > 0.5
    heard = agent.phrase_bank("Keeravani")
    assert heard
    assert view.prayogas[0] in heard      # what it heard leads what it was told


# ==========================================================================
# Test D - tune generation
# ==========================================================================
def test_d_a_requested_prelude_is_real_music(agent, raagas):
    from raagacomposer.agent.practice import PracticeEngine
    from raagacomposer.music.validator import validate
    from raagacomposer.core.models import MelodyVersion

    study(agent, 30, until="b06.prayogas:Keeravani")
    unit = agent.curriculum.unit("b19.alapana:Keeravani")
    unit.params = dict(unit.params)
    unit.params["seconds"] = 12
    unit.exercises = 1
    report = PracticeEngine(agent.repo, raagas).run(unit, "Keeravani", seed=5)

    assert report.artifacts, "nothing was produced"
    notes = report.artifacts[0]
    assert len(notes) >= 4, "a prelude must be more than a sustained note"
    assert len({n.midi for n in notes}) >= 3
    assert max(n.start + n.duration for n in notes) > 2.0

    raaga = raagas.require("Keeravani")
    melody = MelodyVersion(version=1, raaga="Keeravani", notes=list(notes))
    assert validate(melody, raaga).stats["out_of_raaga"] == 0


def test_d2_generated_lines_stay_inside_the_raaga(agent, raagas):
    from raagacomposer.agent.practice import PracticeEngine
    study(agent, 30, until="b06.prayogas:Keeravani")
    engine = PracticeEngine(agent.repo, raagas)
    allowed = set(raagas.require("Keeravani").allowed)
    for unit_id in ("b12.micro:Keeravani", "b13.short_phrase:Keeravani"):
        report = engine.run(agent.curriculum.unit(unit_id), "Keeravani", seed=2)
        assert report.artifacts
        for notes in report.artifacts:
            for note in notes:
                assert parse_swara(note.swara)[0] in allowed


# ==========================================================================
# Test E - learning
# ==========================================================================
def test_e_learn_more_keeravani_alapana(agent):
    """The whole loop: curriculum -> research -> analysis -> memory -> progress."""
    assert agent.study_raaga("Keeravani").startswith("Now studying Keeravani")

    sources_before = len(agent.repo.sources())
    phrases_before = agent.repo.count_phrases("Keeravani")
    passed_before = len(agent.repo.completed_units())

    steps = study(agent, 40, until="b06.prayogas:Keeravani")
    research_steps = [s for s in steps if s.action == "research"]
    practice_steps = [s for s in steps if s.action == "practice"]

    assert research_steps, "the agent never went looking for material"
    assert practice_steps, "the agent never practised"
    assert len(agent.repo.sources()) > sources_before
    assert agent.repo.count_phrases("Keeravani") > phrases_before
    assert len(agent.repo.completed_units()) > passed_before

    for source in agent.repo.sources("Keeravani"):
        assert source.rights_status in ("internally-generated", "user-supplied",
                                        "own-output", "external-unverified")
        assert source.provider
        assert source.ingested_at > 0

    kinds = {e["kind"] for e in agent.repo.events(200)}
    assert {"source.added", "source.analysed", "curriculum.attempt"} <= kinds


def test_e2_nothing_is_fetched_without_permission(agent):
    """No provider reaches outside without the creator turning it on."""
    for candidate in agent.research.find_sources("Keeravani", "phrases", 10):
        assert candidate.provider in ("reference", "corpus", "project")
        assert candidate.rights_status != "external-unverified"


def test_e3_running_out_of_material_is_said_plainly(agent):
    study(agent, 60)
    for _ in range(6):
        step = agent.learn_step()
        if step.action == "research" and not step.passed:
            assert "no new permitted material" in step.detail
            assert "learning folder" in step.detail
            return
    # Reaching the end of the curriculum is also an acceptable outcome.
    assert agent.curriculum.stage_summary()["overall_percent"] > 50


def test_e4_the_whole_curriculum_can_be_completed(agent):
    study(agent, 120)
    summary = agent.curriculum.stage_summary("Keeravani")
    assert summary["overall_percent"] >= 90.0
    assert "Keeravani" in agent.curriculum.mastered_raagas()
    assert agent.status()["mastery"] > 0.6


# ==========================================================================
# Test F - user correction
# ==========================================================================
def test_f_a_correction_lowers_confidence_and_changes_behaviour(agent):
    study(agent, 30, until="b06.prayogas:Keeravani")
    phrases = agent.repo.phrases("Keeravani")
    assert phrases
    target = phrases[0]
    before = {p.id: p.confidence for p in phrases}

    answer = agent.record_feedback("This phrase does not sound like Keeravani.",
                                   raaga="Keeravani", swaras=target.swaras)
    assert "lowered" in answer.lower() or "noted" in answer.lower()

    stored = agent.repo.feedback(raaga="Keeravani")
    assert stored and stored[0]["sentiment"] == "negative"
    assert stored[0]["applied"] == 1

    after = {p.id: p.confidence
             for p in agent.repo.phrases("Keeravani", include_rejected=True)}
    assert any(after.get(pid, 0.0) < value for pid, value in before.items())


def test_f2_a_discredited_phrase_stops_being_used(agent):
    study(agent, 30, until="b06.prayogas:Keeravani")
    target = agent.repo.phrases("Keeravani")[0]
    assert list(target.swaras) in agent.phrase_bank("Keeravani")

    for _ in range(4):                       # keep saying it is wrong
        agent.record_feedback("This does not sound like Keeravani.",
                              raaga="Keeravani", swaras=target.swaras)
    refreshed = agent.repo.phrase(target.id)
    assert refreshed.rejected or refreshed.confidence < target.confidence
    if refreshed.rejected:
        assert list(target.swaras) not in agent.phrase_bank("Keeravani")


def test_f3_praise_is_recorded_too(agent):
    study(agent, 30, until="b06.prayogas:Keeravani")
    target = agent.repo.phrases("Keeravani")[0]
    before = target.confidence
    agent.record_feedback("I like that phrase, keep it.", raaga="Keeravani",
                          swaras=target.swaras)
    assert agent.repo.phrase(target.id).confidence >= before
    assert agent.repo.feedback()[0]["sentiment"] == "positive"


# ==========================================================================
# background learning and status
# ==========================================================================
def test_background_learning_starts_pauses_and_stops(agent):
    assert agent.start_learning()
    deadline = time.time() + 20
    while time.time() < deadline and not agent.repo.completed_units():
        time.sleep(0.1)
    assert agent.repo.completed_units(), "nothing was learned in the background"

    agent.pause_learning()
    assert agent.is_paused
    settled = len(agent.repo.completed_units())
    time.sleep(0.5)
    assert len(agent.repo.completed_units()) == settled

    agent.resume_learning()
    assert not agent.is_paused
    agent.stop_learning()
    assert not agent.is_learning


def test_the_status_tells_the_creator_what_is_happening(agent):
    study(agent, 12)
    status = agent.status()
    for key in ("stage", "current_raaga", "next_goal", "phrases", "sources",
                "facts", "overall_percent", "mastery", "repository"):
        assert key in status
    assert status["current_raaga"] == "Keeravani"
    assert Path(status["repository"]).exists()
    assert agent.recent_events(5)


def test_the_agent_answers_questions_about_itself(agent):
    study(agent, 30, until="b06.prayogas:Keeravani")
    assert "Keeravani" in agent.explain("what do you know?")
    assert "stage" in agent.explain("how is your learning going?").lower()
    assert "arohanam" in agent.explain("what is the scale?")
