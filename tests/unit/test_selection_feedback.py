"""Selection feedback as learned weights - Stage 1 pack document 05 section 6,
plan item S3.

The pack's rule has two halves and both are tested here: a rejected
suggestion must rank lower next time, and none of it may reach the grammar.
"Do not rewrite Arohanam/Avarohanam from preference feedback."
"""
from __future__ import annotations

from pathlib import Path

import pytest

from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.core.models import CreativeBrief
from raagacomposer.raaga import emotion
from raagacomposer.raaga.emotion import (feedback_bias, read_correction,
                                         score_raaga, target_vector)
from raagacomposer.raaga.library import RaagaLibrary

pytestmark = pytest.mark.unit

NO_USER_FILE = Path("no-such-user-raagas.json")

SAD = CreativeBrief(situation="love failure", mood="sad romantic",
                    feel="lonely late at night but still warm")
JOYFUL = CreativeBrief(situation="a wedding", mood="joyful",
                       feel="bright and festive")


@pytest.fixture(scope="module")
def lib() -> RaagaLibrary:
    return RaagaLibrary(extra_path=NO_USER_FILE)


@pytest.fixture
def repo(tmp_path) -> KnowledgeRepository:
    store = KnowledgeRepository(tmp_path / "knowledge.db")
    try:
        yield store
    finally:
        store.close()


# -- reading a correction ---------------------------------------------------
@pytest.mark.parametrize("text, dimension, sign", [
    ("too sad", "sadness", -1),
    ("too bright", "brightness", -1),
    ("more devotional", "devotion", +1),
    ("warmer", "warmth", +1),
    ("more intense", "power", +1),
])
def test_the_packs_own_correction_examples(text, dimension, sign):
    """Document 05 section 6 lists exactly these five."""
    correction = read_correction(text)
    assert correction.get(dimension, 0.0) * sign > 0, correction


def test_a_correction_generalises_beyond_the_five(text=None):
    assert read_correction("less tense")["tension"] < 0
    assert read_correction("a bit more mysterious")["mystery"] > 0
    assert read_correction("not warm enough")["warmth"] > 0


def test_a_comment_that_is_not_a_correction_changes_nothing():
    """Better to understand nothing than to invent a number."""
    for text in ("I like it", "keep going", "", "the second one"):
        assert read_correction(text) == {}


# -- the store --------------------------------------------------------------
def test_feedback_is_stored_against_the_dimensions_it_was_given_for(repo):
    repo.record_selection_feedback("Keeravani", -0.7,
                                   {"sadness": 1.0, "warmth": 0.5},
                                   source="rejected")
    stored = {w.dimension: w for w in repo.selection_weights("Keeravani")}
    assert stored["*"].weight < 0
    assert stored["sadness"].weight < stored["warmth"].weight < 0
    assert all(w.observations == 1 for w in stored.values())


def test_repeated_feedback_strengthens_rather_than_duplicates(repo):
    for _ in range(3):
        repo.record_selection_feedback("Keeravani", -0.7, {"sadness": 1.0},
                                       source="rejected")
    weights = [w for w in repo.selection_weights("Keeravani")
               if w.dimension == "sadness"]
    assert len(weights) == 1
    assert weights[0].observations == 3


def test_a_weight_cannot_run_away(repo):
    for _ in range(50):
        repo.record_selection_feedback("Keeravani", -0.7, {"sadness": 1.0},
                                       source="rejected")
    for weight in repo.selection_weights("Keeravani"):
        assert abs(weight.weight) <= 3.0


def test_preferences_can_be_reviewed_and_withdrawn(repo):
    repo.record_selection_feedback("Keeravani", 1.0, {"sadness": 1.0},
                                   source="accepted")
    assert repo.selection_weights("Keeravani")
    assert "Keeravani" in repo.selection_weights()[0].describe()
    assert repo.reset_selection_weights("Keeravani") >= 1
    assert repo.selection_weights("Keeravani") == []


def test_withdrawing_keeps_the_record(repo):
    """Framework document 04 section 6: deprecate, do not delete - a
    preference that was held and withdrawn is not the same as one never
    held, and a creator should be able to see which happened."""
    repo.record_selection_feedback("Keeravani", 1.0, {"sadness": 1.0})
    repo.reset_selection_weights("Keeravani")
    rows = repo._conn.execute(
        "SELECT deprecated FROM selection_weights WHERE raaga='Keeravani'"
    ).fetchall()
    assert rows and all(r["deprecated"] == 1 for r in rows)


# -- how it reaches the ranking --------------------------------------------
def test_the_bias_is_bounded_however_much_evidence_there_is(lib):
    target = target_vector(SAD)
    huge = {d: -3.0 for d in emotion.DIMENSIONS}
    huge["*"] = -3.0
    assert abs(feedback_bias(huge, target)) <= emotion.MAX_FEEDBACK
    assert abs(feedback_bias({d: 3.0 for d in emotion.DIMENSIONS}, target)) \
        <= emotion.MAX_FEEDBACK


def test_no_feedback_scores_exactly_as_before(lib):
    raaga = lib.require("Keeravani")
    assert score_raaga(SAD, raaga).score == score_raaga(SAD, raaga, weights={}).score


def test_feedback_is_spent_only_on_the_dimensions_the_brief_asks_for(lib):
    """A rejection in a joyful brief must not sink the raaga in a sad one."""
    raaga = lib.require("Keeravani")
    weights = {"*": 0.0, "joy": -2.0, "brightness": -2.0}
    assert feedback_bias(weights, target_vector(SAD)) == pytest.approx(0.0, abs=0.01)
    assert feedback_bias(weights, target_vector(JOYFUL)) < -0.02


def test_the_reason_says_when_preference_moved_it(lib):
    raaga = lib.require("Keeravani")
    disliked = score_raaga(SAD, raaga, weights={"*": -2.0, "sadness": -2.0})
    assert any("passed this over" in p for p in disliked.penalties)
    liked = score_raaga(SAD, raaga, weights={"*": 2.0, "sadness": 2.0})
    assert any("chosen this before" in b for b in liked.bonuses)


# -- what S3 exists to prove -----------------------------------------------
def test_a_rejected_suggestion_ranks_lower_next_time(lib, repo):
    """The plan's acceptance criterion for S3, end to end."""
    before = emotion.rank(SAD, lib.all(), limit=5)
    top = before[0]

    repo.record_selection_feedback(top.name, -0.7, target_vector(SAD).weights,
                                   source="rejected")
    after = emotion.rank(SAD, lib.all(), limit=5,
                         weights=repo.selection_weight_map())

    scores_after = {s.name: s.score for s in
                    emotion.rank(SAD, lib.all(), limit=80,
                                 weights=repo.selection_weight_map())}
    assert scores_after[top.name] < top.score
    assert after[0].name != top.name


def test_an_accepted_suggestion_ranks_higher_next_time(lib, repo):
    everything = {s.name: s.score for s in emotion.rank(SAD, lib.all(), limit=80)}
    chosen = sorted(everything, key=everything.get)[len(everything) // 2]

    repo.record_selection_feedback(chosen, 1.0, target_vector(SAD).weights,
                                   source="accepted")
    after = {s.name: s.score for s in
             emotion.rank(SAD, lib.all(), limit=80,
                          weights=repo.selection_weight_map())}
    assert after[chosen] > everything[chosen]


def test_learning_for_one_feeling_leaves_another_alone(lib, repo):
    """The whole reason weights are stored per dimension."""
    joyful_before = [s.name for s in emotion.rank(JOYFUL, lib.all(), limit=5)]
    top = emotion.rank(SAD, lib.all(), limit=5)[0].name
    repo.record_selection_feedback(top, -0.7, target_vector(SAD).weights,
                                   source="rejected")
    joyful_after = [s.name for s in
                    emotion.rank(JOYFUL, lib.all(), limit=5,
                                 weights=repo.selection_weight_map())]
    assert joyful_after == joyful_before


def test_preference_never_touches_the_grammar(lib, repo):
    """Pack document 05 section 6: "do not rewrite Arohanam/Avarohanam from
    preference feedback".  Structurally true - nothing in the feedback path
    can reach a raaga's notes - and asserted here so it stays true."""
    raaga = lib.require("Keeravani")
    before = (list(raaga.arohanam), list(raaga.avarohanam), list(raaga.jeeva),
              list(raaga.nyasa), dict(raaga.gamaka), [list(p) for p in raaga.prayogas])
    for _ in range(10):
        repo.record_selection_feedback("Keeravani", -0.7,
                                       target_vector(SAD).weights,
                                       source="rejected")
    after = (list(raaga.arohanam), list(raaga.avarohanam), list(raaga.jeeva),
             list(raaga.nyasa), dict(raaga.gamaka),
             [list(p) for p in raaga.prayogas])
    assert after == before
    # ... and nothing was written to the tables that hold what it knows.
    assert repo.facts("Keeravani") == []
    assert repo.phrases(raaga="Keeravani") == []
