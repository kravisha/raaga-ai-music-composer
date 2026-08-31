"""Retrieval, context assembly and the librarian - sections 18, 19, 30, 39, 40.

Section 18 is explicit that retrieval must not "merely return text chunks with
similar words", so the tests that matter here are the ones a keyword search
would fail: reaching a raga's constraints without anybody having said the word
"constraint", and reaching anything at all through a spelling nobody stored.
"""
from __future__ import annotations

import pytest

from raagacomposer.kb.context import PROFILES, KnowledgeContextBuilder
from raagacomposer.kb.librarian import Librarian
from raagacomposer.kb.models import (Evidence, ExtractionMethod, KnowledgeGap,
                                     KnowledgeType, Source, Status)
from raagacomposer.kb.retrieval import HybridRetriever, Query

pytestmark = pytest.mark.unit


@pytest.fixture
def populated(kb, kb_claim, kb_evidence):
    """A small but connected picture of one raga."""
    kb.commit_knowledge(
        kb_claim("arohanam", "S R2 G3 M1 P D2 S+",
                 "Kambhoji ascends S R2 G3 M1 P D2 S+."), [kb_evidence()])
    kb.commit_knowledge(
        kb_claim("avarohanam", "S+ N2 D2 P M1 G3 R2 S",
                 "Kambhoji descends S+ N2 D2 P M1 G3 R2 S."), [kb_evidence()])
    kb.commit_knowledge(
        kb_claim("prayoga", "P D2 S+ N2", "A characteristic phrase: P D2 S+ N2.",
                 knowledge_type=KnowledgeType.PATTERN, tags=["heard"]),
        [kb_evidence()])
    kb.commit_knowledge(
        kb_claim("avoid", "N3", "Kambhoji does not use N3.",
                 knowledge_type=KnowledgeType.CONSTRAINT), [kb_evidence()])
    kb.commit_knowledge(
        kb_claim("gamaka", "kampita on G3", "G3 carries a kampita.",
                 tags=["stated"]), [kb_evidence()])
    return kb


# --------------------------------------------------------------------------
# retrieval (section 18)
# --------------------------------------------------------------------------
def test_an_exact_lookup_finds_the_entity(populated):
    hits = HybridRetriever(populated).search(Query(subject="Kambhoji"))
    assert hits
    assert any("exact" in h.routes or "subject" in h.routes for h in hits)


def test_a_raga_is_reached_through_a_spelling_nobody_stored(populated):
    """Section 8.  The creator should not have to guess our transliteration."""
    hits = HybridRetriever(populated).search(Query(subject="Kamboji"))
    assert hits, "a differently spelled name found nothing"


def test_the_constraints_arrive_without_being_asked_for(populated):
    """This is the test a keyword search fails.  Nothing in "Kambhoji" says
    "avoid", but the constraint hangs off the raga in the graph."""
    hits = HybridRetriever(populated).search(
        Query(subject="Kambhoji", include_graph=True))
    found = [h for h in hits
             if h.item.knowledge_type == KnowledgeType.CONSTRAINT]
    assert found, "the raga's constraints were not reached"


def test_a_confidence_floor_excludes_weak_knowledge(populated, kb_claim,
                                                    kb_evidence):
    weak = populated.commit_knowledge(
        kb_claim("tempo", "very slow", "Kambhoji is taken very slowly."),
        [kb_evidence(strength=0.05, method=ExtractionMethod.INFERRED)])
    hits = HybridRetriever(populated).search(
        Query(subject="Kambhoji", min_confidence=0.5))
    assert all(h.item.confidence >= 0.5 for h in hits)


def test_rejected_knowledge_is_not_served(populated, kb_claim, kb_evidence):
    item = populated.commit_knowledge(
        kb_claim("tempo", "very slow"), [kb_evidence()]).item
    populated.correct_knowledge(item.knowledge_id, action="mark_incorrect")
    hits = HybridRetriever(populated).search(Query(subject="Kambhoji"))
    assert all(h.item.knowledge_id != item.knowledge_id for h in hits)


def test_semantic_search_says_it_is_not_semantic(populated, caplog):
    """Section 18 lists it; there is no embedding model, and pretending
    otherwise would be a lie about how the answer was found."""
    retriever = HybridRetriever(populated)
    assert retriever.semantic_available is False
    with caplog.at_level("INFO"):
        results = retriever.semantic_search("Kambhoji phrases")
    assert results
    assert any("semantic search is unavailable" in r.message
               for r in caplog.records)


def test_what_was_served_is_recorded(populated):
    """Section 41 - "which knowledge did you use?" needs an answer later."""
    context = KnowledgeContextBuilder(populated).build("compose",
                                                       raga="Kambhoji")
    rows = populated.store.query("SELECT * FROM retrieval_usage")
    assert rows
    assert rows[0]["item_count"] == len(context.knowledge_ids)


# --------------------------------------------------------------------------
# context assembly (section 19)
# --------------------------------------------------------------------------
def test_a_compose_context_has_what_composing_needs(populated):
    """Specification test 5, in miniature: structure, phrases, constraints -
    not text chunks with similar words."""
    context = KnowledgeContextBuilder(populated).build("compose",
                                                       raga="Kambhoji")
    assert context.items
    assert context.constraints, "no constraints were carried"
    text = context.render()
    assert "Must not" in text
    predicates = {i.predicate for i in context.items}
    assert "arohanam" in predicates
    assert "prayoga" in predicates


def test_a_context_is_bounded_rather_than_everything(populated, kb_claim,
                                                     kb_evidence):
    """Section 19 exists to stop prompt bloat, so the bound has to hold even
    when the Knowledge Base holds far more than the bound."""
    for index in range(60):
        populated.commit_knowledge(
            kb_claim("prayoga", f"S R2 G3 M1 P D2 N2 S+ {index}",
                     f"A characteristic phrase number {index}.",
                     knowledge_type=KnowledgeType.PATTERN), [kb_evidence()])
    profile = PROFILES["compose"]
    context = KnowledgeContextBuilder(populated).build("compose",
                                                       raga="Kambhoji")
    assert len(context.items) <= profile.total


def test_a_context_repeats_nothing(populated):
    context = KnowledgeContextBuilder(populated).build("compose",
                                                       raga="Kambhoji")
    statements = [i.display() for i in context.items]
    assert len(statements) == len(set(statements))


def test_a_context_carries_the_disagreements(kb, kb_claim, kb_evidence):
    """Never hand over the more confident of two answers as though settled."""
    kb.commit_knowledge(kb_claim(value="S R2 G3 M1 P D2 S+"), [kb_evidence()])
    other = kb.add_source(Source(title="Another",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    kb.commit_knowledge(kb_claim(value="S R2 G3 M1 P D2 N2 S+",
                                 statement="Kambhoji ascends with N2."),
                        [kb_evidence(other)])
    context = KnowledgeContextBuilder(kb).build("compose", raga="Kambhoji")
    assert context.disagreements
    assert "Sources disagree about" in context.render()


def test_a_context_says_what_is_weakly_supported(populated, kb_claim,
                                                 kb_evidence):
    populated.commit_knowledge(
        kb_claim("tempo", "unclear", "The tempo is unclear."),
        [kb_evidence(strength=0.1, method=ExtractionMethod.INFERRED)])
    context = KnowledgeContextBuilder(populated).build(
        "teach", raga="Kambhoji")
    assert any("weakly supported" in w or "provisional" in w
               for w in context.warnings)


def test_a_context_for_nothing_known_says_so_rather_than_inventing(kb):
    context = KnowledgeContextBuilder(kb).build("compose", raga="Nonesuchraga")
    assert context.is_empty
    assert context.unknowns
    assert "Nothing is held about this yet" in context.render()


def test_a_context_reports_the_gaps_for_its_subject(populated):
    populated.add_gap(KnowledgeGap(
        subject="Kambhoji", missing_information="no tala material",
        reason="no source has covered it"))
    context = KnowledgeContextBuilder(populated).build("compose",
                                                       raga="Kambhoji")
    assert any("tala" in u for u in context.unknowns)


def test_explaining_carries_provenance_and_composing_does_not(populated):
    """Section 19 step 6 - provenance where it is needed, not everywhere."""
    explain = KnowledgeContextBuilder(populated).build("explain",
                                                       raga="Kambhoji")
    compose = KnowledgeContextBuilder(populated).build("compose",
                                                       raga="Kambhoji")
    assert explain.provenance
    assert not compose.provenance


# --------------------------------------------------------------------------
# the librarian (sections 30, 39, 40)
# --------------------------------------------------------------------------
def test_the_health_report_covers_what_section_40_asks_for(populated):
    report = Librarian(populated).health()
    text = Librarian(populated).render_health(report)
    assert report.total_items > 0
    assert report.sources >= 1
    assert report.evidence >= 1
    for heading in ("Knowledge items", "Sources", "Evidence records",
                    "Relationships", "Orphan items", "Duplicate candidates",
                    "Unresolved conflicts", "Knowledge gaps",
                    "Schema version", "Integrity"):
        assert heading in text, heading


def test_gaps_are_detected_for_a_subject_we_have_begun(populated):
    gaps = Librarian(populated).detect_gaps(["Kambhoji"])
    missing = " ".join(g.missing_information for g in gaps)
    assert "jeeva" in missing
    assert all(g.suggested_search for g in gaps)


def test_no_gap_is_invented_for_a_subject_we_hold_nothing_about(populated):
    gaps = Librarian(populated).detect_gaps(["Nonesuchraga"], record=False)
    assert not gaps


def test_compaction_never_touches_a_recorded_disagreement(kb, kb_claim,
                                                          kb_evidence):
    """Section 16's limit: compaction must not erase meaningful disagreement."""
    kb.commit_knowledge(kb_claim(value="S R2 G3 M1 P D2 S+"), [kb_evidence()])
    other = kb.add_source(Source(title="Another",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    kb.commit_knowledge(kb_claim(value="S R2 G3 M1 P D2 N2 S+",
                                 statement="Kambhoji ascends S R2 G3 M1 P D2 S+ mostly."),
                        [kb_evidence(other)])
    before = kb.store.count("knowledge_items")
    report = Librarian(kb).compact(dry_run=False)
    assert kb.store.count("knowledge_items") == before
    assert len(kb.conflicts(open_only=True)) == 1


def test_a_dry_run_changes_nothing(populated):
    before = populated.store.count("knowledge_items")
    report = Librarian(populated).compact(dry_run=True)
    assert report["dry_run"]
    assert report["merged"] == 0
    assert populated.store.count("knowledge_items") == before
