"""Deciding what may be believed - sections 7F, 8.7 and 20 rule 8.

The rules under test are the ones a learning system gets wrong in the same two
ways every time: it launders a guess into a fact, and it lets the newest source
quietly overwrite the oldest.  Neither is allowed here.
"""
from __future__ import annotations

import pytest

from raagacomposer.training.models import (KnowledgeEntry, KnowledgeStatus,
                                           LearningSource)
from raagacomposer.training.semantics import Observation
from raagacomposer.training.validation import (CONFIDENCE_FLOOR,
                                               KnowledgeValidationService)

pytestmark = pytest.mark.unit


@pytest.fixture
def validator(training_store) -> KnowledgeValidationService:
    return KnowledgeValidationService(training_store)


@pytest.fixture
def source() -> LearningSource:
    return LearningSource(source_id="src_1", title="A lesson",
                          url="https://example.org/one")


def _observation(**kwargs) -> Observation:
    defaults = dict(statement="Kambhoji ascends S R2 G3 M1 P D2 S+.",
                    subject="Kambhoji", concept="arohanam", category="scale",
                    raga="Kambhoji", evidence="heard at 0.0s-8.0s",
                    confidence=0.8, reference="0.0s-8.0s",
                    objective_category="scale", tags=["heard"])
    defaults.update(kwargs)
    return Observation(**defaults)


# --------------------------------------------------------------------------
# the confidence floor
# --------------------------------------------------------------------------
def test_a_guess_is_reported_but_not_stored(validator, source):
    """Section 7F: uncertain inference does not become fact.  It is not
    silently dropped either - it comes back as uncertain so the report can
    say what was noticed and not believed."""
    outcome = validator.validate([_observation(confidence=0.05)], source, "r1")
    assert not outcome.accepted
    assert len(outcome.uncertain) == 1


def test_a_well_supported_observation_is_accepted(validator, source):
    outcome = validator.validate([_observation(confidence=0.8)], source, "r1")
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].confidence >= CONFIDENCE_FLOOR


def test_something_merely_stated_counts_for_less_than_something_heard(
        validator, source):
    heard = validator.validate([_observation(tags=["heard"])], source, "r1")
    stated = validator.validate([_observation(tags=["stated"],
                                              category="tala")], source, "r2")
    assert stated.accepted[0].confidence < heard.accepted[0].confidence


# --------------------------------------------------------------------------
# provenance is not optional
# --------------------------------------------------------------------------
def test_every_accepted_entry_can_say_where_it_came_from(validator, source):
    entry = validator.validate([_observation()], source, "run_9").accepted[0]
    assert entry.source_id == source.source_id
    assert entry.source_url == source.url
    assert entry.source_title == source.title
    assert entry.run_id == "run_9"
    assert entry.source_timestamp
    assert entry.evidence


# --------------------------------------------------------------------------
# conflicts (section 8.7, rule 8)
# --------------------------------------------------------------------------
def test_a_disagreeing_source_never_overwrites_what_is_held(validator, source,
                                                            training_store):
    first = validator.validate([_observation(
        statement="Kambhoji ascends S R2 G3 M1 P D2 S+.")], source, "r1")
    training_store.add_knowledge(first.accepted[0])

    other = LearningSource(source_id="src_2", title="Another teacher",
                           url="https://example.org/two")
    second = validator.validate([_observation(
        statement="Kambhoji ascends S R2 G3 M1 P D2 N2 S+.")], other, "r2")

    assert len(second.conflicts) == 1
    conflict = second.conflicts[0]
    assert conflict.existing_claim != conflict.new_claim
    assert conflict.recommendation

    # The original is still exactly as it was, flagged but not rewritten.
    kept = training_store.knowledge(first.accepted[0].knowledge_id)
    assert kept.normalized_statement == \
        "Kambhoji ascends S R2 G3 M1 P D2 S+."
    assert kept.contradicted
    # ... and the new claim is kept beside it rather than thrown away.
    assert second.accepted[0].status == KnowledgeStatus.DISPUTED


def test_the_same_claim_from_a_second_source_is_confirmation(validator, source,
                                                             training_store):
    """Section 8.6 - independent reinforcement is worth recording, and it
    raises confidence rather than creating a duplicate."""
    first = validator.validate([_observation(confidence=0.6)], source, "r1")
    stored = training_store.add_knowledge(first.accepted[0])
    before = stored.confidence

    other = LearningSource(source_id="src_2", title="Another teacher")
    second = validator.validate([_observation(confidence=0.6)], other, "r2")

    assert not second.accepted
    assert len(second.confirmed) == 1
    assert training_store.knowledge(stored.knowledge_id).confidence > before


def test_confidence_cannot_be_inflated_past_certainty(validator, source,
                                                      training_store):
    entry = validator.validate([_observation(confidence=0.95)],
                               source, "r1").accepted[0]
    entry.confidence = 0.98
    training_store.add_knowledge(entry)
    for index in range(30):
        other = LearningSource(source_id=f"s{index}", title="another")
        validator.validate([_observation(confidence=0.95)], other, f"r{index}")
    assert training_store.knowledge(entry.knowledge_id).confidence <= 0.99


# --------------------------------------------------------------------------
# internal consistency
# --------------------------------------------------------------------------
def test_a_source_that_contradicts_itself_keeps_only_its_best_answer(
        validator, source):
    outcome = validator.validate([
        _observation(statement="The tonic is at 261.6 Hz.", concept="tonic",
                     category="tonic", confidence=0.9),
        _observation(statement="The tonic is at 196.0 Hz.", concept="tonic",
                     category="tonic", confidence=0.4),
    ], source, "r1")
    assert len(outcome.rejected) == 1
    assert "more than one thing" in outcome.rejected[0][1]
    assert len(outcome.accepted) == 1
    assert "261.6" in outcome.accepted[0].normalized_statement


def test_many_phrases_from_one_source_are_not_a_contradiction(validator,
                                                              source):
    """A lesson containing several phrases is a lesson, not an inconsistency."""
    outcome = validator.validate([
        _observation(statement="A phrase of the raaga: S R2 G3 M1.",
                     concept="phrase", category="phrase"),
        _observation(statement="A phrase of the raaga: P D2 S+ N2.",
                     concept="phrase", category="phrase"),
    ], source, "r1")
    assert not outcome.rejected
    assert len(outcome.accepted) == 2
