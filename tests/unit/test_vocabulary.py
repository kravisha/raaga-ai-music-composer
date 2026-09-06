"""A mood word the engine cannot read is kept, not dropped.

Arya's specification, 2026-09-06 13:16:07 -04:00, in the shared conversation
record.  Each acceptance criterion has a test here, named for it.
"""
from __future__ import annotations

import pytest

from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.core import provenance
from raagacomposer.core.models import CreativeBrief
from raagacomposer.raaga import emotion, selection, vocabulary


@pytest.fixture()
def repo(tmp_path):
    r = KnowledgeRepository(tmp_path / "knowledge.db")
    yield r
    r.close()


class FakeLLM:
    """A backend whose answers the test controls."""

    available = True

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.asked = []

    def map_mood_word(self, term, catalog):
        self.asked.append((term, tuple(catalog)))
        return self.answers.get(
            term, {"means": [], "confidence": 0.0, "reason": "no idea"})


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------
def test_only_genuinely_unreadable_words_are_reported():
    found = vocabulary.unknown_terms("nervy, upbeat, skittish",
                                     "a very tense chase")
    assert "nervy" in found and "skittish" in found
    # known vocabulary is not "unknown"
    assert "upbeat" not in found and "tense" not in found
    # grammar the scorer handles itself is not unknown either
    assert "very" not in found
    # nor is filler
    assert "the" not in found and "a" not in found


def test_the_same_word_is_reported_once_in_the_order_it_was_typed():
    assert vocabulary.unknown_terms("nervy and nervy and zesty") == \
        ["nervy", "zesty"]


# --------------------------------------------------------------------------
# "results from the understood portion without waiting for research"
# --------------------------------------------------------------------------
def test_a_brief_still_ranks_on_the_words_that_were_understood():
    brief = CreativeBrief(mood="nervy, upbeat, skittish")
    weights = emotion.target_vector(brief).weights
    assert any(v > 0 for v in weights.values()), "the known words did nothing"
    assert selection.suggest(brief, limit=3), "no raaga was suggested"


# --------------------------------------------------------------------------
# "each unresolved term is retained ... repeated occurrences can share one"
# --------------------------------------------------------------------------
def test_unknown_words_are_kept_and_repeats_share_one_investigation(repo):
    first = repo.note_unknown_terms(["nervy", "skittish"])
    second = repo.note_unknown_terms(["nervy"])
    assert first == ["nervy", "skittish"]
    assert second == ["nervy"]
    row = repo.unresolved_term("nervy")
    assert row.occurrences == 2
    assert row.status == vocabulary.PENDING
    assert len(repo.unresolved_terms()) == 2


def test_a_resolved_word_is_no_longer_outstanding(repo):
    repo.note_unknown_terms(["nervy"])
    repo.record_investigation("nervy", ["nervous"], 0.7, "jittery")
    assert repo.note_unknown_terms(["nervy"]) == []


# --------------------------------------------------------------------------
# "do not invent associations"
# --------------------------------------------------------------------------
def test_a_word_the_engine_does_not_have_is_refused():
    """The safeguard: a model may answer with anything; only real vocabulary
    survives, so a resolution can never conjure an emotion weight."""
    assert vocabulary.validate_mapping(["frazzlish", "nervous"]) == ["nervous"]
    assert vocabulary.validate_mapping(["frazzlish"]) == []

    liar = FakeLLM({"skittish": {"means": ["frazzlish"], "confidence": 1.0,
                                 "reason": "obviously"}})
    mapped, confidence, _ = vocabulary.investigate("skittish", liar)
    assert mapped == []
    assert confidence == 0.0


def test_the_model_is_only_ever_offered_words_the_engine_knows():
    llm = FakeLLM()
    vocabulary.investigate("nervy", llm)
    _, catalog = llm.asked[0]
    assert catalog, "no vocabulary was offered"
    assert all(word in emotion.LEXICON for word in catalog)


def test_a_models_confidence_cannot_outrank_curated_vocabulary():
    llm = FakeLLM({"nervy": {"means": ["nervous"], "confidence": 1.0,
                             "reason": "certain"}})
    _, confidence, _ = vocabulary.investigate("nervy", llm)
    assert confidence <= 0.8


def test_a_resolution_records_that_a_machine_proposed_it(repo):
    repo.note_unknown_terms(["nervy"])
    repo.record_investigation("nervy", ["nervous"], 0.7, "jittery, on edge")
    row = repo.unresolved_term("nervy")
    assert row.origin == provenance.GENERATED
    assert not provenance.may_be_learned_from(row.origin)
    assert "jittery" in row.evidence, "the reasoning was not kept"


# --------------------------------------------------------------------------
# "observable pending / investigating / resolved / needs-user-input status"
# --------------------------------------------------------------------------
def test_every_status_is_reachable_and_visible(repo):
    repo.note_unknown_terms(["nervy", "skittish"])
    assert repo.unresolved_term("nervy").status == vocabulary.PENDING

    repo.set_term_status("nervy", vocabulary.INVESTIGATING)
    assert repo.unresolved_term("nervy").status == vocabulary.INVESTIGATING

    repo.record_investigation("nervy", ["nervous"], 0.7, "jittery")
    assert repo.unresolved_term("nervy").status == vocabulary.RESOLVED

    for _ in range(vocabulary.MAX_ATTEMPTS):
        repo.record_investigation("skittish", [], 0.0, "nothing usable")
    assert repo.unresolved_term("skittish").status == vocabulary.NEEDS_USER_INPUT


def test_a_person_is_asked_only_after_investigation_has_failed(repo):
    """Not up front, and not for ever - the specification says both."""
    repo.note_unknown_terms(["skittish"])
    for attempt in range(1, vocabulary.MAX_ATTEMPTS):
        repo.record_investigation("skittish", [], 0.0, "nothing usable")
        assert repo.unresolved_term("skittish").status == vocabulary.PENDING, \
            f"asked a person after only {attempt} attempt(s)"
    repo.record_investigation("skittish", [], 0.0, "nothing usable")
    assert repo.unresolved_term("skittish").status == vocabulary.NEEDS_USER_INPUT


# --------------------------------------------------------------------------
# "supported resolutions become available to later raga searches"
# --------------------------------------------------------------------------
def test_a_resolution_changes_what_later_searches_see(repo):
    repo.note_unknown_terms(["nervy"])
    before = emotion.target_vector(CreativeBrief(mood="nervy")).weights
    assert not any(v > 0 for v in before.values()), \
        "the word was understood before it was resolved"

    repo.record_investigation("nervy", ["nervous", "tense"], 0.8, "on edge")
    table = repo.resolutions()
    assert table == {"nervy": ["nervous", "tense"]}

    rewritten = vocabulary.resolve_text("nervy", table)
    after = emotion.target_vector(CreativeBrief(mood=rewritten)).weights
    assert after["tension"] > 0.5, "the resolution did not reach the vector"


def test_resolution_leaves_words_it_does_not_know_alone():
    table = {"nervy": ["nervous"]}
    assert vocabulary.resolve_text("nervy and bright", table) == \
        "nervous and bright"
    assert vocabulary.resolve_text("", table) == ""
    assert vocabulary.resolve_text("bright", {}) == "bright"


# --------------------------------------------------------------------------
# no backend at all
# --------------------------------------------------------------------------
def test_with_no_language_backend_nothing_is_claimed():
    mapped, confidence, why = vocabulary.investigate("nervy", None)
    assert mapped == [] and confidence == 0.0
    assert "backend" in why


def test_narrative_prose_is_not_treated_as_unreadable_feeling(tmp_path):
    """Situation is a story, not a mood list.

    Scanning it reported "terrace", "man" and "late" as unreadable
    feelings - noise, and a promise to investigate words that were never
    moods.  Only mood and feel are scanned.
    """
    from raagacomposer.app import AppController
    from raagacomposer.core.settings import Settings

    settings = Settings()
    settings.projects_dir = str(tmp_path / "projects")
    settings.knowledge_db = str(tmp_path / "knowledge.db")
    settings.stt_provider = "none"
    app = AppController(settings)
    try:
        app.update_brief(mood="nervy",
                         situation="a man on a terrace late at night",
                         notes="the second reel, after the interval")
        deferred = app.note_unreadable_words()
        assert "nervy" in deferred
        for noise in ("terrace", "man", "late", "interval", "reel"):
            assert noise not in deferred, f"{noise!r} is not a feeling"
    finally:
        app.close()


# --------------------------------------------------------------------------
# (a) an unconfirmed reading says so; (b) a dismissed one stops acting
# Approved by Arya 2026-09-06 14:35:42 under Krish's delegated authority.
# --------------------------------------------------------------------------
def test_an_unconfirmed_reading_is_offered_for_telling_the_creator(repo):
    repo.note_unknown_terms(["nervy"])
    repo.record_investigation("nervy", ["nervous", "tense"], 0.8, "on edge")
    assert repo.unconfirmed_readings() == [("nervy", ["nervous", "tense"])]


def test_a_confirmed_reading_is_not_flagged_as_a_guess(repo):
    """Only what a machine proposed needs the caveat."""
    repo.note_unknown_terms(["nervy"])
    repo.record_investigation("nervy", ["nervous"], 0.8, "on edge",
                              origin=provenance.HUMAN)
    assert repo.resolutions() == {"nervy": ["nervous"]}
    assert repo.unconfirmed_readings() == []


def test_a_dismissed_reading_stops_influencing_anything(repo):
    repo.note_unknown_terms(["zesty"])
    repo.record_investigation("zesty", ["playful"], 0.6, "lively")
    assert repo.resolutions() == {"zesty": ["playful"]}

    repo.dismiss_term("zesty", "wrong - zesty is not playful")

    assert repo.resolutions() == {}
    assert repo.unconfirmed_readings() == []
    row = repo.unresolved_term("zesty")
    assert row.status == vocabulary.DISMISSED
    assert row.mapped_to == []
    assert "wrong" in row.note, "the reason was not kept"


def test_a_dismissed_reading_is_not_proposed_again(repo):
    """Without this the next sweep would quietly undo the rejection."""
    repo.note_unknown_terms(["zesty"])
    repo.dismiss_term("zesty", "no")

    assert repo.record_investigation("zesty", ["playful"], 0.9, "trying again") \
        == vocabulary.DISMISSED
    assert repo.resolutions() == {}
    assert [t.term for t in repo.unresolved_terms(status=vocabulary.PENDING)] == []


def test_a_dismissed_word_is_not_reported_as_work_in_progress(repo):
    """Dismissed is settled, not outstanding - saying "still working out"
    would promise work that is deliberately not happening."""
    repo.note_unknown_terms(["zesty"])
    repo.dismiss_term("zesty", "no")
    assert repo.note_unknown_terms(["zesty", "fresh"]) == ["fresh"]
