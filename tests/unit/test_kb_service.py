"""The Knowledge Base service - specification sections 9, 12, 13, 15, 28, 38.

These cover the four things that separate a Knowledge Base from a table of
notes: a fact learned twice gains evidence rather than a second row, a
disagreement is recorded rather than resolved by whoever wrote last, a change
keeps the old reading, and nothing gets in without saying where it came from.
"""
from __future__ import annotations

import pytest

from raagacomposer.kb.models import (ConflictState, Evidence, ExtractionMethod,
                                     KnowledgeItem, KnowledgeType, Relation,
                                     Source, Status)
from raagacomposer.kb.service import CommitOutcome

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# provenance is the price of entry (section 9)
# --------------------------------------------------------------------------
def test_knowledge_with_no_evidence_is_refused(kb, kb_claim):
    """Section 9's rule, enforced at the door rather than lamented later."""
    result = kb.commit_knowledge(kb_claim(), [])
    assert result.outcome == CommitOutcome.REJECTED
    assert "evidence" in result.reason
    assert kb.store.count("evidence") == 0


def test_structural_knowledge_may_be_stored_without_a_source(kb, kb_claim):
    """The shipped taxonomy is not a claim anybody made, so it is the one
    thing allowed through - and it has to be asked for explicitly."""
    result = kb.commit_knowledge(kb_claim(), [], allow_without_evidence=True)
    assert result.outcome == CommitOutcome.NEW


def test_every_stored_item_can_say_where_it_came_from(kb, kb_claim,
                                                      kb_evidence, kb_source):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    record = kb.provenance(item.knowledge_id)
    assert record["where_from"][0]["title"] == kb_source.title
    assert record["where_in_the_source"][0]["from"] == 10.0
    assert record["audio_transcript_or_inferred"] == ["audio_derived"]
    assert record["supporting_sources"] == 1
    assert record["confidence_explained"]


# --------------------------------------------------------------------------
# duplicate control (section 15)
# --------------------------------------------------------------------------
def test_the_same_fact_from_two_sources_is_one_item_with_two_evidences(
        kb, kb_claim, kb_evidence):
    """Section 15's headline, and specification test 3."""
    first = kb.commit_knowledge(kb_claim(), [kb_evidence()])
    other = kb.add_source(Source(title="Another teacher",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(
        kb_claim(statement="The arohanam is S R2 G3 M1 P D2 S+."),
        [kb_evidence(other)])

    assert second.outcome == CommitOutcome.DUPLICATE
    assert second.item.knowledge_id == first.item.knowledge_id
    assert len(kb.evidence_for(first.item.knowledge_id)) == 2
    assert len(kb.sources_for(first.item.knowledge_id)) == 2


def test_a_second_independent_source_raises_confidence(kb, kb_claim,
                                                       kb_evidence):
    first = kb.commit_knowledge(kb_claim(), [kb_evidence()])
    before = first.item.confidence
    other = kb.add_source(Source(title="Another",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(kb_claim(), [kb_evidence(other)])
    assert second.item.confidence > before


def test_many_phrases_of_one_raga_coexist(kb, kb_claim, kb_evidence):
    """A raga has one arohanam but many characteristic phrases.  Treating the
    second phrase as a rival to the first would fill the store with conflicts
    between things never in competition."""
    for phrase in ("S R2 G3 M1 P", "P D2 S+ N2", "G3 M1 P D2", "M1 G3 R2 S"):
        result = kb.commit_knowledge(
            kb_claim("prayoga", phrase,
                     f"A characteristic phrase: {phrase}.",
                     knowledge_type=KnowledgeType.PATTERN), [kb_evidence()])
        assert result.outcome == CommitOutcome.NEW, phrase
    assert not kb.conflicts()
    assert len(kb.items_about("Kambhoji")) >= 4


def test_the_same_phrase_twice_is_still_one_phrase(kb, kb_claim, kb_evidence):
    kb.commit_knowledge(kb_claim("prayoga", "P D2 S+ N2",
                                 knowledge_type=KnowledgeType.PATTERN),
                        [kb_evidence()])
    other = kb.add_source(Source(title="Another",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    again = kb.commit_knowledge(
        kb_claim("prayoga", "P D2 S+ N2", knowledge_type=KnowledgeType.PATTERN),
        [kb_evidence(other)])
    assert again.outcome == CommitOutcome.DUPLICATE


def test_a_reworded_claim_becomes_a_version_not_a_row(kb, kb_evidence):
    from raagacomposer.kb import normalize

    def free_text(statement):
        return KnowledgeItem(
            canonical_name="Kambhoji", subject="Kambhoji", predicate="rasa",
            object_value="heroic", statement=statement, raga="Kambhoji",
            structured_value=normalize.structured_for("rasa", "heroic"))

    first = kb.commit_knowledge(
        free_text("Kambhoji carries a heroic and majestic feeling."),
        [kb_evidence()])
    other = kb.add_source(Source(title="Another",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(
        free_text("Kambhoji carries a heroic, majestic feeling."),
        [kb_evidence(other)])
    assert second.item.knowledge_id == first.item.knowledge_id
    assert second.outcome in (CommitOutcome.DUPLICATE, CommitOutcome.REFINEMENT)


# --------------------------------------------------------------------------
# contradiction (section 12, rule: never silently overwrite)
# --------------------------------------------------------------------------
@pytest.fixture
def disagreement(kb, kb_claim, kb_evidence):
    first = kb.commit_knowledge(
        kb_claim(value="S R2 G3 M1 P D2 S+",
                 statement="Kambhoji ascends S R2 G3 M1 P D2 S+."),
        [kb_evidence()])
    other = kb.add_source(Source(title="A second teacher",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(
        kb_claim(value="S R2 G3 M1 P D2 N2 S+",
                 statement="Kambhoji ascends S R2 G3 M1 P D2 N2 S+."),
        [kb_evidence(other)])
    return first, second


def test_a_contradiction_keeps_both_claims(kb, disagreement):
    first, second = disagreement
    assert second.outcome == CommitOutcome.CONTRADICTION
    assert kb.get_by_id(first.item.knowledge_id) is not None
    assert kb.get_by_id(second.item.knowledge_id) is not None


def test_a_contradiction_does_not_alter_the_original_claim(kb, disagreement):
    first, _ = disagreement
    kept = kb.get_by_id(first.item.knowledge_id)
    assert kept.object_value == "S R2 G3 M1 P D2 S+"
    assert kept.statement == "Kambhoji ascends S R2 G3 M1 P D2 S+."


def test_a_contradiction_is_recorded_with_a_recommendation(kb, disagreement):
    conflicts = kb.conflicts(open_only=True)
    assert len(conflicts) == 1
    assert conflicts[0].resolution_status == ConflictState.UNRESOLVED
    assert conflicts[0].notes


def test_the_two_claims_are_linked_to_one_another(kb, disagreement):
    first, second = disagreement
    neighbours = kb.graph_neighbors(first.item.knowledge_id)
    assert any(rel.relation_type == Relation.CONFLICTS_WITH
               and item.knowledge_id == second.item.knowledge_id
               for rel, item in neighbours)


def test_resolving_supersedes_rather_than_deletes(kb, disagreement):
    """Section 13: the losing reading is kept, marked, and still traceable."""
    first, second = disagreement
    conflict = kb.conflicts()[0]
    assert kb.resolve_conflict(conflict.conflict_id,
                               ConflictState.RESOLVED_A, reviewer="a person")
    assert kb.get_by_id(first.item.knowledge_id).status == Status.ACCEPTED
    loser = kb.get_by_id(second.item.knowledge_id)
    assert loser is not None, "the losing claim was deleted"
    assert loser.status == Status.SUPERSEDED
    assert not kb.conflicts(open_only=True)


def test_both_can_be_valid_in_different_contexts(kb, disagreement):
    """Section 12: context-dependent disagreement must not be flattened."""
    conflict = kb.conflicts()[0]
    kb.resolve_conflict(conflict.conflict_id, ConflictState.BOTH_VALID,
                        notes="one is the sampurna form")
    first, second = disagreement
    assert kb.get_by_id(first.item.knowledge_id).status == Status.ACCEPTED
    assert kb.get_by_id(second.item.knowledge_id).status == Status.ACCEPTED


def test_an_unknown_resolution_is_refused(kb, disagreement):
    conflict = kb.conflicts()[0]
    with pytest.raises(ValueError):
        kb.resolve_conflict(conflict.conflict_id, "whatever")


# --------------------------------------------------------------------------
# versions and corrections (sections 13, 28)
# --------------------------------------------------------------------------
def test_a_correction_keeps_the_previous_reading(kb, kb_claim, kb_evidence):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    kb.correct_knowledge(item.knowledge_id, action="mark_incorrect",
                         explanation="that is not the arohanam")

    updated = kb.get_by_id(item.knowledge_id)
    assert updated.status == Status.REJECTED
    assert updated.confidence <= 0.1
    versions = kb.versions_of(item.knowledge_id)
    assert len(versions) >= 2
    assert any("S R2 G3 M1 P D2 S+" in str(v.snapshot) for v in versions)


def test_a_correction_leaves_an_audit_record(kb, kb_claim, kb_evidence):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    kb.correct_knowledge(item.knowledge_id, action="mark_correct",
                         explanation="my teacher confirms this")
    record = kb.provenance(item.knowledge_id)
    assert record["corrections"]
    assert record["corrections"][0]["action"] == "mark_correct"
    assert record["corrections"][0]["explanation"]


@pytest.mark.parametrize("action,expected", [
    ("mark_correct", Status.ACCEPTED),
    ("mark_incorrect", Status.REJECTED),
    ("needs_review", Status.NEEDS_REVIEW),
])
def test_the_user_verdicts_the_spec_lists(kb, kb_claim, kb_evidence, action,
                                          expected):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    kb.correct_knowledge(item.knowledge_id, action=action)
    assert kb.get_by_id(item.knowledge_id).status == expected


def test_confidence_can_be_nudged_either_way(kb, kb_claim, kb_evidence):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    before = item.confidence
    kb.correct_knowledge(item.knowledge_id, action="reduce_confidence")
    assert kb.get_by_id(item.knowledge_id).confidence < before
    kb.correct_knowledge(item.knowledge_id, action="increase_confidence")
    kb.correct_knowledge(item.knowledge_id, action="increase_confidence")
    assert kb.get_by_id(item.knowledge_id).confidence > before - 0.01


# --------------------------------------------------------------------------
# the network (section 3)
# --------------------------------------------------------------------------
def test_a_claim_is_attached_to_the_thing_it_is_about(kb, kb_claim,
                                                      kb_evidence):
    """Section 3.  A claim nothing points at cannot be found by asking about
    the raga, which is how nearly every real query starts."""
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    entity = kb.find_entity("Kambhoji")
    assert entity is not None
    assert any(other.knowledge_id == item.knowledge_id
               for _, other in kb.graph_neighbors(entity.knowledge_id))


def test_an_entity_is_found_through_any_of_its_spellings(kb, kb_claim,
                                                         kb_evidence):
    kb.commit_knowledge(kb_claim(), [kb_evidence()])
    for spelling in ("Kambhoji", "Kamboji", "kambhoji", "KAMBOJI"):
        assert kb.find_entity(spelling) is not None, spelling


def test_a_relationship_is_not_duplicated(kb, kb_claim, kb_evidence):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    entity = kb.find_entity("Kambhoji")
    before = kb.store.count("relationships")
    kb.add_relationship(item.knowledge_id, Relation.BELONGS_TO_RAGA,
                        entity.knowledge_id)
    kb.add_relationship(item.knowledge_id, Relation.BELONGS_TO_RAGA,
                        entity.knowledge_id)
    assert kb.store.count("relationships") == before


def test_an_item_cannot_relate_to_itself(kb, kb_claim, kb_evidence):
    item = kb.commit_knowledge(kb_claim(), [kb_evidence()]).item
    assert kb.add_relationship(item.knowledge_id, Relation.RELATED_TO,
                               item.knowledge_id) is None


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def test_the_same_source_through_a_different_link_is_one_source(kb):
    first = kb.add_source(Source(title="A lesson",
                                 reference="https://youtu.be/DoTQ3ZYK-Qw"))
    second = kb.add_source(Source(
        title="A lesson",
        reference="https://www.youtube.com/watch?v=DoTQ3ZYK-Qw&t=30"))
    assert second.source_id == first.source_id
    assert kb.store.count("sources") == 1
