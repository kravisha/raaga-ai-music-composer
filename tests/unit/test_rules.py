"""Unit tests for the Agent Factory pilot's hard rules
(``raagacomposer/agent/rules.py``): each rule decides its case from a
hand-built ``Dispute`` and stays silent outside its competence, and a
learned fact that contradicts the library loses to the library (document 04,
"hard versus heuristic")."""
from __future__ import annotations

import pytest

from raagacomposer.agent.knowledge import Phrase
from raagacomposer.agent.originality import PhraseIndex
from raagacomposer.agent.rules import (AllowedSwarasRule, DirectionRule,
                                       FactRule, OriginalityRule,
                                       RestingNoteRule, hard_rules)
from raagacomposer.factory.models import Dispute

pytestmark = pytest.mark.unit


def _dispute(question: str, student_claim: str = "valid",
            trainer_claim: str = "invalid", phrase: str = "",
            raaga: str = "Keeravani", **extra) -> Dispute:
    evidence = []
    if phrase:
        evidence.append(f"phrase: {phrase}")
    if raaga:
        evidence.append(f"raaga: {raaga}")
    return Dispute(question=question, student_claim=student_claim,
                   trainer_claim=trainer_claim, evidence_student=evidence,
                   student_confidence=0.7, trainer_confidence=0.7, **extra)


# --------------------------------------------------------------------------
# AllowedSwarasRule
# --------------------------------------------------------------------------
def test_allowed_swaras_rule_valid_phrase(raagas):
    rule = AllowedSwarasRule(raagas)
    dispute = _dispute("is the phrase valid in Keeravani?",
                       student_claim="valid", trainer_claim="invalid",
                       phrase="S R2 G2 M1 P")
    assert rule.applies(dispute)
    ruling = rule.decide(dispute)
    assert ruling is not None
    assert ruling.ruling == "valid"
    assert ruling.accepted_claim == "student"
    assert ruling.rejected_claim == "trainer"
    assert ruling.decided_by == "AllowedSwarasRule"


def test_allowed_swaras_rule_invalid_phrase(raagas):
    rule = AllowedSwarasRule(raagas)
    dispute = _dispute("valid in Keeravani", student_claim="valid",
                       trainer_claim="invalid", phrase="S R2 G3 M1 P")
    ruling = rule.decide(dispute)
    assert ruling.ruling == "invalid"
    assert ruling.accepted_claim == "trainer"
    assert "G3" in ruling.rationale


def test_allowed_swaras_rule_outside_competence(raagas):
    rule = AllowedSwarasRule(raagas)
    mood_dispute = _dispute("does this phrase sound sad?",
                            phrase="S R2 G2", raaga="Keeravani")
    assert not rule.applies(mood_dispute)
    assert rule.decide(mood_dispute) is None
    # No phrase/raaga evidence at all: still outside competence.
    bare = Dispute(question="is the phrase valid in Keeravani?")
    assert not rule.applies(bare)


# --------------------------------------------------------------------------
# DirectionRule
# --------------------------------------------------------------------------
def test_direction_rule_valid_ascent(raagas):
    rule = DirectionRule(raagas)
    # Abheri's arohanam omits R and D; this climb uses only ascending swaras.
    dispute = _dispute("does this respect the arohanam?",
                       phrase="S G2 M1 P N2", raaga="Abheri")
    assert rule.applies(dispute)
    ruling = rule.decide(dispute)
    assert ruling.ruling == "valid"


def test_direction_rule_forbidden_ascent(raagas):
    rule = DirectionRule(raagas)
    # R2 only belongs to Abheri's avarohanam; using it going up is forbidden.
    dispute = _dispute("does this respect the arohanam?",
                       student_claim="valid", trainer_claim="invalid",
                       phrase="S R2 G2", raaga="Abheri")
    ruling = rule.decide(dispute)
    assert ruling.ruling == "invalid"
    assert ruling.accepted_claim == "trainer"


def test_direction_rule_outside_competence(raagas):
    rule = DirectionRule(raagas)
    dispute = _dispute("is this an original phrase?", phrase="S G2 M1",
                       raaga="Abheri")
    assert not rule.applies(dispute)
    assert rule.decide(dispute) is None


# --------------------------------------------------------------------------
# RestingNoteRule
# --------------------------------------------------------------------------
def test_resting_note_rule_valid_cadence(raagas):
    rule = RestingNoteRule(raagas)
    dispute = _dispute("does the phrase rest on a resting note?",
                       phrase="S R2 G2 M1 P", raaga="Keeravani")
    assert rule.applies(dispute)
    ruling = rule.decide(dispute)
    assert ruling.ruling == "valid"


def test_resting_note_rule_invalid_cadence(raagas):
    rule = RestingNoteRule(raagas)
    dispute = _dispute("does the phrase rest on a resting note?",
                       student_claim="valid", trainer_claim="invalid",
                       phrase="S R2 G2 M1 D1", raaga="Keeravani")
    ruling = rule.decide(dispute)
    assert ruling.ruling == "invalid"
    assert ruling.accepted_claim == "trainer"
    assert "D1" in ruling.rationale


def test_resting_note_rule_outside_competence(raagas):
    rule = RestingNoteRule(raagas)
    dispute = _dispute("is the phrase valid in Keeravani?",
                       phrase="S R2 G2", raaga="Keeravani")
    assert not rule.applies(dispute)


# --------------------------------------------------------------------------
# FactRule: hard knowledge beats a learned fact that disagrees with it
# --------------------------------------------------------------------------
def test_fact_rule_student_correct_trainer_wrong_loses(raagas):
    rule = FactRule(raagas)
    correct = "S R2 G2 M1 P D1 N3 S+"
    dispute = _dispute("what is the arohanam of Keeravani",
                       student_claim=correct,
                       trainer_claim="S R2 G2 M1 P D2 N2 S+")
    assert rule.applies(dispute)
    ruling = rule.decide(dispute)
    assert ruling.ruling == correct
    assert ruling.accepted_claim == "student"
    assert ruling.rejected_claim == "trainer"
    assert ruling.correction_trainer


def test_fact_rule_a_learned_value_that_disagrees_with_the_library_loses(raagas):
    """A fact learned from a recording, differing from the library, is
    heuristic; the library's canonical value is hard and wins regardless of
    which side (student or trainer) is the one holding the learned value."""
    rule = FactRule(raagas)
    correct = "S P G2"
    dispute = _dispute("what is the nyasa of Keeravani",
                       student_claim="S P N3",     # the disputed, learned value
                       trainer_claim=correct)
    ruling = rule.decide(dispute)
    assert ruling.ruling == correct
    assert ruling.accepted_claim == "trainer"
    assert ruling.correction_student


def test_fact_rule_outside_competence(raagas):
    rule = FactRule(raagas)
    dispute = _dispute("do you like Keeravani?")
    assert not rule.applies(dispute)
    assert rule.decide(dispute) is None


# --------------------------------------------------------------------------
# OriginalityRule
# --------------------------------------------------------------------------
def test_originality_rule_needs_an_index():
    rule = OriginalityRule(None, phrase_index=None)  # type: ignore[arg-type]
    dispute = _dispute("is this an original phrase?", phrase="S R2 G2 M1 P")
    assert not rule.applies(dispute)


def test_originality_rule_flags_a_copied_phrase(raagas):
    index = PhraseIndex(n=3)
    index.add(Phrase(raaga="Keeravani",
                     swaras=["S", "R2", "G2", "M1", "P", "D1", "N3"],
                     confidence=0.9))
    rule = OriginalityRule(raagas, index)
    dispute = _dispute("is this an original phrase?",
                       student_claim="valid", trainer_claim="invalid",
                       phrase="S R2 G2 M1 P D1 N3", raaga="Keeravani")
    assert rule.applies(dispute)
    ruling = rule.decide(dispute)
    assert ruling.ruling == "invalid"
    assert ruling.accepted_claim == "trainer"


def test_originality_rule_accepts_a_fresh_phrase(raagas):
    index = PhraseIndex(n=3)
    index.add(Phrase(raaga="Keeravani",
                     swaras=["S", "R2", "G2", "M1", "P", "D1", "N3"],
                     confidence=0.9))
    rule = OriginalityRule(raagas, index)
    dispute = _dispute("is this an original phrase?",
                       student_claim="valid", trainer_claim="invalid",
                       phrase="P D1 N3 S+ N3 D1 P", raaga="Keeravani")
    ruling = rule.decide(dispute)
    assert ruling.ruling == "valid"
    assert ruling.accepted_claim == "student"


# --------------------------------------------------------------------------
# hard_rules()
# --------------------------------------------------------------------------
def test_hard_rules_returns_all_five(raagas):
    rules = hard_rules(raagas)
    names = {r.name for r in rules}
    assert names == {"AllowedSwarasRule", "DirectionRule", "RestingNoteRule",
                     "FactRule", "OriginalityRule"}
    assert all(r.knowledge_class.value == "hard" for r in rules)
