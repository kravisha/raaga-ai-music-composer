"""Unit tests: permanent memory, the curriculum engine, and the learned view."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from raagacomposer.agent.curriculum import CROSS_RAAGA_GATE, CurriculumEngine
from raagacomposer.agent.knowledge import (Fact, KnowledgeRepository, Phrase,
                                           Source, UnitProgress, fingerprint)
from raagacomposer.agent.learned import (describe_knowledge, knowledge_confidence,
                                         learned_phrase_bank, learned_raaga)

pytestmark = pytest.mark.unit


@pytest.fixture
def repo(tmp_path) -> KnowledgeRepository:
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    yield repository
    repository.close()


def _phrase(raaga="Keeravani", swaras=("S", "R2", "G2"), **kw) -> Phrase:
    return Phrase(raaga=raaga, swaras=list(swaras),
                  midi=[60, 62, 63][:len(swaras)],
                  durations=[0.5] * len(swaras), **kw)


# --------------------------------------------------------------------------
# repository
# --------------------------------------------------------------------------
def test_a_new_repository_is_created_with_its_schema(tmp_path):
    path = tmp_path / "fresh.db"
    repository = KnowledgeRepository(path)
    try:
        assert path.exists()
        assert repository.schema_version >= 1
        assert repository.stats()["phrases"] == 0
    finally:
        repository.close()


def test_a_repository_from_the_future_is_refused(tmp_path):
    path = tmp_path / "future.db"
    repository = KnowledgeRepository(path)
    repository._conn.execute(
        "UPDATE meta SET value='999' WHERE key='schema_version'")
    repository._conn.commit()
    repository.close()
    with pytest.raises(RuntimeError, match="newer version"):
        KnowledgeRepository(path)


def test_sources_are_stored_once(repo):
    source = Source(locator="reference://Keeravani/arohanam", title="ascent",
                    provider="reference", raaga="Keeravani")
    stored, is_new = repo.add_source(source)
    assert is_new
    again, is_new_again = repo.add_source(
        Source(locator="reference://Keeravani/arohanam", provider="reference"))
    assert not is_new_again
    assert again.id == stored.id
    assert repo.has_source("reference", "reference://Keeravani/arohanam")
    assert not repo.has_source("reference", "reference://Kalyani/arohanam")


def test_source_status_can_be_updated(repo):
    stored, _ = repo.add_source(Source(locator="x", provider="p"))
    repo.update_source(stored.id, status="analysed", confidence=0.8)
    assert repo.source(stored.id).status == "analysed"
    assert repo.source(stored.id).confidence == pytest.approx(0.8)


def test_an_identical_phrase_strengthens_rather_than_duplicates(repo):
    first, is_new = repo.add_phrase(_phrase(confidence=0.5))
    assert is_new
    second, is_new_again = repo.add_phrase(_phrase(confidence=0.5))
    assert not is_new_again
    assert second.votes == 2
    assert second.confidence > first.confidence
    assert repo.count_phrases("Keeravani") == 1


def test_phrases_are_filtered_by_confidence_and_rejection(repo):
    repo.add_phrase(_phrase(swaras=("S", "R2", "G2"), confidence=0.8))
    repo.add_phrase(_phrase(swaras=("P", "D1", "N3"), confidence=0.2))
    assert len(repo.phrases("Keeravani")) == 2
    assert len(repo.phrases("Keeravani", min_confidence=0.5)) == 1

    weak = repo.phrases("Keeravani", min_confidence=0.0)[-1]
    repo.reject_phrase(weak.id, "the creator said it was wrong")
    assert len(repo.phrases("Keeravani")) == 1
    assert len(repo.phrases("Keeravani", include_rejected=True)) == 2


def test_agreeing_facts_gain_confidence(repo):
    repo.add_fact(Fact(raaga="Keeravani", key="arohanam",
                       value="S R2 G2 M1 P D1 N3 S+", confidence=0.6))
    repo.add_fact(Fact(raaga="Keeravani", key="arohanam",
                       value="S R2 G2 M1 P D1 N3 S+", confidence=0.6))
    facts = repo.facts("Keeravani", "arohanam")
    assert len(facts) == 1
    assert facts[0].confidence > 0.6


def test_disagreeing_facts_are_kept_and_flagged(repo):
    repo.add_fact(Fact(raaga="Keeravani", key="arohanam", value="S R2 G2",
                       confidence=0.7, source_id="a"))
    repo.add_fact(Fact(raaga="Keeravani", key="arohanam", value="S R1 G2",
                       confidence=0.5, source_id="b"))
    facts = repo.facts("Keeravani", "arohanam")
    assert len(facts) == 2                       # nothing was overwritten
    assert all(f.disputed for f in facts)
    assert repo.best_fact("Keeravani", "arohanam").value == "S R2 G2"
    assert repo.stats()["disputed_facts"] == 2


def test_curriculum_progress_round_trips(repo):
    progress = UnitProgress(unit_id="a01.sound", status="passed", mastery=0.9,
                            attempts=2, completed_at=time.time())
    repo.save_progress(progress)
    stored = repo.progress("a01.sound")
    assert stored.status == "passed"
    assert stored.mastery == pytest.approx(0.9)
    assert repo.completed_units() == ["a01.sound"]


def test_compositions_and_feedback_are_recorded(repo):
    composition = repo.record_composition(
        project_id="p1", title="Night", raaga="Keeravani",
        brief={"mood": "longing"}, structure={"sections": ["Pallavi"]},
        scores={"raaga_correctness": 0.9}, final_score=0.82)
    assert composition
    stored = repo.compositions()[0]
    assert stored["brief"]["mood"] == "longing"
    assert stored["scores"]["raaga_correctness"] == pytest.approx(0.9)

    feedback_id = repo.add_feedback(target_kind="composition", target_id="p1",
                                    text="does not sound like Keeravani",
                                    sentiment="negative", raaga="Keeravani")
    repo.mark_feedback_applied(feedback_id)
    assert repo.feedback()[0]["applied"] == 1


def test_state_tasks_and_events(repo):
    repo.set_state("current_raaga", "Kalyani")
    assert repo.state("current_raaga") == "Kalyani"
    assert repo.state("missing", "fallback") == "fallback"

    task = repo.queue_task("ingest", {"raaga": "Kalyani"})
    assert repo.pending_tasks()[0]["payload"]["raaga"] == "Kalyani"
    repo.finish_task(task)
    assert repo.pending_tasks() == []

    repo.log_event("test.event", "something happened", raaga="Kalyani")
    assert repo.events()[0]["kind"] == "test.event"


def test_everything_survives_reopening(tmp_path):
    path = tmp_path / "persist.db"
    first = KnowledgeRepository(path)
    first.add_phrase(_phrase(swaras=("S", "R2", "G2", "M1"), confidence=0.7))
    first.add_fact(Fact(raaga="Keeravani", key="arohanam", value="S R2 G2",
                        confidence=0.8))
    first.save_progress(UnitProgress(unit_id="a01.sound", status="passed"))
    first.set_state("current_raaga", "Keeravani")
    first.close()

    second = KnowledgeRepository(path)
    try:
        assert second.count_phrases("Keeravani") == 1
        assert second.best_fact("Keeravani", "arohanam").value == "S R2 G2"
        assert second.completed_units() == ["a01.sound"]
        assert second.state("current_raaga") == "Keeravani"
    finally:
        second.close()


def test_fingerprints_are_stable():
    assert fingerprint(["Keeravani", "S", "R2"]) == \
        fingerprint(["Keeravani", "S", "R2"])
    assert fingerprint(["Keeravani", "S", "R2"]) != \
        fingerprint(["Keeravani", "S", "G2"])


# --------------------------------------------------------------------------
# curriculum
# --------------------------------------------------------------------------
@pytest.fixture
def curriculum(repo) -> CurriculumEngine:
    return CurriculumEngine(repo, pilot_raaga="Keeravani")


def test_the_curriculum_loads_all_three_stages(curriculum):
    assert len(curriculum.universal_units()) >= 10
    assert len(curriculum.raaga_units("Keeravani")) >= 20
    assert len(curriculum.cross_units()) >= 2
    assert curriculum.version >= 1


def test_every_unit_names_a_handler_and_a_goal(curriculum):
    handlers = {"listen.compare", "listen.identify", "listen.transcribe",
                "generate.pattern", "generate.section", "recall.fact",
                "recall.phrases", "classify.valid"}
    for unit in curriculum.all_units(["Keeravani"]):
        assert unit.skill_type in handlers, unit.id
        assert unit.learning_goal
        assert 0 < unit.minimum_pass_score <= 1.0


def test_stage_b_units_are_instantiated_per_raaga(curriculum):
    keeravani = curriculum.raaga_units("Keeravani")
    kalyani = curriculum.raaga_units("Kalyani")
    assert keeravani[0].id.endswith(":Keeravani")
    assert kalyani[0].id.endswith(":Kalyani")
    assert keeravani[0].id != kalyani[0].id
    assert all(p.endswith(":Keeravani") or p.startswith("a")
               for p in keeravani[-1].prerequisites())


def test_the_first_unit_has_no_prerequisites(curriculum):
    first = curriculum.next_unit("Keeravani")
    assert first is not None
    assert curriculum.is_available(first)
    assert first.level == 1


def test_a_unit_is_blocked_until_its_prerequisites_pass(curriculum, repo):
    units = {u.curriculum_unit_id: u for u in curriculum.universal_units()}
    second = units["a02.pitch"]
    assert curriculum.blocked_by(second) == ["a01.sound"]
    repo.save_progress(UnitProgress(unit_id="a01.sound", status="passed"))
    assert curriculum.is_available(second)


def test_passing_a_unit_advances_the_schedule(curriculum):
    first = curriculum.next_unit()
    curriculum.record_attempt(first, 0.95, True)
    assert curriculum.is_passed(first)
    following = curriculum.next_unit()
    assert following is not None and following.id != first.id


def test_a_unit_is_retried_then_rested(curriculum):
    unit = curriculum.next_unit()
    for _ in range(unit.max_failures()):
        curriculum.record_attempt(unit, 0.2, False)
    progress = curriculum.repo.progress(unit.id)
    assert progress.status == "failed"
    assert progress.failures == unit.max_failures()
    # It is rested, but with nothing else available the agent comes back to it
    # rather than sitting idle - up to a hard limit on total attempts.
    revisited = curriculum.next_unit()
    assert revisited is not None and revisited.id == unit.id
    for _ in range(curriculum.max_attempts_per_unit):
        offered = curriculum.next_unit()
        if offered is None:
            break
        curriculum.record_attempt(offered, 0.2, False)
    assert curriculum.next_unit() is None


def test_a_rested_unit_is_revisited_later(repo):
    """A lesson that beat the student today is offered again, not abandoned."""
    curriculum = CurriculumEngine(repo, pilot_raaga="Keeravani",
                                  revisit_after=0.0)
    unit = curriculum.next_unit()
    for _ in range(unit.max_failures()):
        curriculum.record_attempt(unit, 0.4, False)
    assert curriculum.repo.progress(unit.id).status == "failed"

    revisited = curriculum.next_unit()
    assert revisited is not None and revisited.id == unit.id
    assert curriculum.repo.progress(unit.id).failures == 0
    assert curriculum.repo.progress(unit.id).mastery == pytest.approx(0.4)


def test_cross_raaga_work_waits_for_two_raagas(curriculum, repo):
    cross = curriculum.cross_units()[0]
    assert any("mastered" in reason for reason in curriculum.blocked_by(cross))
    for raaga in ("Keeravani", "Kalyani"):
        repo.save_progress(UnitProgress(unit_id=f"{CROSS_RAAGA_GATE}:{raaga}",
                                        status="passed", raaga=raaga))
    assert curriculum.mastered_raagas() == ["Kalyani", "Keeravani"]
    assert not [r for r in curriculum.blocked_by(cross) if "mastered" in r]


def test_the_studied_raaga_is_remembered(curriculum, repo):
    assert curriculum.current_raaga() == "Keeravani"
    curriculum.set_current_raaga("Kalyani")
    assert curriculum.current_raaga() == "Kalyani"
    assert CurriculumEngine(repo).current_raaga() == "Kalyani"


def test_the_summary_describes_where_it_is(curriculum):
    summary = curriculum.stage_summary("Keeravani")
    assert summary["stage"] == "A"
    assert "/" in summary["foundations"]
    assert summary["next_goal"]
    rows = curriculum.progress_table("Keeravani")
    assert rows and {"unit", "status", "goal"} <= set(rows[0])


# --------------------------------------------------------------------------
# the learned view
# --------------------------------------------------------------------------
def test_with_no_memory_the_reference_library_is_used(repo, raagas):
    view, completeness = learned_raaga(repo, raagas, "Keeravani")
    assert view is not None
    assert completeness == 0.0
    assert view.arohanam == raagas.require("Keeravani").arohanam


def test_learned_facts_replace_the_reference(repo, raagas):
    repo.add_fact(Fact(raaga="Keeravani", key="arohanam",
                       value="S R2 G2 M1 P D1 N3 S+", confidence=0.8))
    repo.add_fact(Fact(raaga="Keeravani", key="nyasa", value="S P",
                       confidence=0.8))
    view, completeness = learned_raaga(repo, raagas, "Keeravani")
    assert view.nyasa == ["S", "P"]
    assert completeness > 0.0
    assert view.source == "learned"


def test_heard_phrases_lead_the_prayogas(repo, raagas):
    repo.add_phrase(_phrase(swaras=("P", "D1", "N3", "S+"), confidence=0.9))
    view, _ = learned_raaga(repo, raagas, "Keeravani")
    assert view.prayogas[0] == ["P", "D1", "N3", "S+"]
    assert learned_phrase_bank(repo, "Keeravani") == [["P", "D1", "N3", "S+"]]


def test_low_confidence_phrases_are_left_out(repo, raagas):
    repo.add_phrase(_phrase(swaras=("P", "D1", "N3"), confidence=0.1))
    assert learned_phrase_bank(repo, "Keeravani", min_confidence=0.4) == []


def test_knowledge_confidence_grows_with_evidence(repo, raagas):
    empty = knowledge_confidence(repo, "Keeravani")
    assert empty["overall"] == 0.0
    for key, value in (("arohanam", "S R2 G2"), ("avarohanam", "G2 R2 S"),
                       ("swaras", "S R2 G2"), ("jeeva", "G2"), ("nyasa", "S"),
                       ("gamaka", "G2:kampita")):
        repo.add_fact(Fact(raaga="Keeravani", key=key, value=value,
                           confidence=0.8))
    for i in range(6):
        repo.add_phrase(_phrase(swaras=("S", "R2", f"G2{'+' * (i % 2)}"),
                                confidence=0.7))
    filled = knowledge_confidence(repo, "Keeravani")
    assert filled["core_facts"] == pytest.approx(1.0)
    assert filled["overall"] > empty["overall"]


def test_the_agent_can_say_what_it_knows_and_where_from(repo, raagas):
    stored, _ = repo.add_source(Source(locator="library://Keeravani",
                                       title="reference book",
                                       provider="library", raaga="Keeravani"))
    repo.add_fact(Fact(raaga="Keeravani", key="arohanam", value="S R2 G2",
                       confidence=0.9, source_id=stored.id))
    text = describe_knowledge(repo, "Keeravani")
    assert "arohanam" in text
    assert "reference book" in text
    assert "confidence" in text
    assert "have not learned" in describe_knowledge(repo, "Todi")


# --------------------------------------------------------------------------
# unlearning, so a changed ear can be rebuilt from the audio
# --------------------------------------------------------------------------
def test_forgetting_a_source_removes_what_came_from_it(repo):
    """A phrase is not knowledge about a raaga - it is knowledge about a
    raaga *as heard by a particular version of the ears*.  When those
    change it has to be re-derived, and that starts with removing what the
    old version left behind."""
    kept, _ = repo.add_source(Source(locator="keep://one", provider="corpus",
                                     raaga="Keeravani"))
    doomed, _ = repo.add_source(Source(locator="stale://two", provider="corpus",
                                       raaga="Keeravani"))
    for source, swaras in ((kept, ["S", "R2", "G2"]),
                           (doomed, ["G2", "M1", "P"])):
        repo.add_phrase(Phrase(raaga="Keeravani", swaras=swaras,
                               source_id=source.id, confidence=0.7))
    repo.add_fact(Fact(raaga="Keeravani", key="observed_tempo", value="88",
                       confidence=0.6, source_id=doomed.id))

    removed = repo.forget_source(doomed.id)
    assert removed >= 3, "the phrase, the fact and the source row"

    left = repo.phrases(raaga="Keeravani", limit=50)
    assert [p.source_id for p in left] == [kept.id], \
        "forgetting one source took another's phrases with it"
    assert all(s.id != doomed.id for s in repo.sources())
    assert not [f for f in repo.facts("Keeravani") if f.source_id == doomed.id]


def test_forgetting_something_that_is_not_there_is_harmless(repo):
    before = len(repo.phrases(limit=50))
    assert repo.forget_source("src_nonexistent") == 0
    assert len(repo.phrases(limit=50)) == before
