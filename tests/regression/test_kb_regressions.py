"""Defects found while building the Knowledge Base, each named.

Four of these were caught by the code's own smoke tests before any of it
shipped.  Two of them would have been slow poisons rather than loud failures,
which is why they are guarded here rather than merely fixed.
"""
from __future__ import annotations

import pytest

from raagacomposer.kb import normalize
from raagacomposer.kb.librarian import Librarian
from raagacomposer.kb.models import (Evidence, ExtractionMethod, KnowledgeType,
                                     Source, Status)
from raagacomposer.kb.service import CommitOutcome
from raagacomposer.kb.store import KnowledgeStore

pytestmark = pytest.mark.regression


def test_reg_kb_a_different_scale_was_filed_as_a_reworded_one(kb, kb_claim,
                                                              kb_evidence):
    """Two arohanams differing by one swara read as very similar *sentences*,
    and the refinement path compared only the shape of the structured value
    rather than its contents.  One teacher's scale therefore overwrote
    another's under the name of tidying up the wording - the exact silent
    overwrite section 12 forbids.
    """
    first = kb.commit_knowledge(
        kb_claim(value="S R2 G3 M1 P D2 S+",
                 statement="Kambhoji ascends S R2 G3 M1 P D2 S+."),
        [kb_evidence()])
    other = kb.add_source(Source(title="Another teacher",
                                 reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(
        kb_claim(value="S R2 G3 M1 P D2 N2 S+",
                 statement="Kambhoji ascends S R2 G3 M1 P D2 N2 S+."),
        [kb_evidence(other)])

    assert second.outcome == CommitOutcome.CONTRADICTION
    assert kb.get_by_id(first.item.knowledge_id).object_value == \
        "S R2 G3 M1 P D2 S+"
    assert kb.conflicts(open_only=True)


def test_reg_kb_every_phrase_of_a_raga_contradicted_the_last(kb, kb_claim,
                                                             kb_evidence):
    """Identity was subject plus predicate, so the second characteristic
    phrase of a raga collided with the first and was recorded as disagreeing
    with it.  Seeding the shipped library produced a conflict for every
    prayoga after the first, and buried the real disagreements among them.

    A raga has one arohanam but many phrases: the distinction is whether a
    predicate is set-valued.
    """
    phrases = ("S R2 G3 M1 P", "P D2 S+ N2", "G3 M1 P D2", "M1 G3 R2 S")
    for phrase in phrases:
        result = kb.commit_knowledge(
            kb_claim("prayoga", phrase, f"A characteristic phrase: {phrase}.",
                     knowledge_type=KnowledgeType.PATTERN), [kb_evidence()])
        assert result.outcome == CommitOutcome.NEW, phrase
    assert not kb.conflicts(), "phrases were recorded as contradicting"

    # ... while a single-valued predicate still collides as it should.
    assert normalize.is_set_valued("prayoga")
    assert not normalize.is_set_valued("arohanam")


def test_reg_kb_seeding_the_library_twice_doubled_it(kb, raagas):
    """The same defect seen from the other side: because each prayoga
    contradicted the last, a second migration created a fresh disputed row for
    every one of them, and the Knowledge Base grew every time the application
    started.  Migration must be idempotent.
    """
    from raagacomposer.kb.migrate import migrate_all

    migrate_all(kb, raagas=raagas)
    after_first = kb.store.count("knowledge_items")
    migrate_all(kb, raagas=raagas)
    after_second = kb.store.count("knowledge_items")

    assert after_second == after_first, "a second migration added rows"
    assert not kb.conflicts(), "a second migration invented disagreements"


def test_reg_kb_a_plain_write_blocked_the_next_transaction(kb_path):
    """Python's sqlite3 driver opens a transaction of its own on any write
    unless told not to, so an ordinary INSERT left one open and the next
    explicit BEGIN failed with "cannot start a transaction within a
    transaction" - which made the all-or-nothing write path of section 34
    unusable after the first write.
    """
    store = KnowledgeStore(kb_path)
    try:
        store.audit("test", "a plain write outside any transaction")
        with store.transaction():
            store.execute(
                "INSERT INTO knowledge_items(knowledge_id, knowledge_type, "
                "canonical_name, statement, created_at, updated_at, identity) "
                "VALUES ('kn_x','fact','Kambhoji','a thing',1,1,'x')")
        store.set_meta("something", "else")
        with store.transaction():
            store.execute("UPDATE knowledge_items SET statement='changed'")
        assert store.count("knowledge_items") == 1
    finally:
        store.close()


def test_reg_kb_an_entity_looked_like_knowledge_that_had_lost_its_source(
        kb, kb_claim, kb_evidence):
    """The librarian reports items with no evidence, because section 9 says no
    learned knowledge may lose its source.  Entity nodes have no evidence by
    design - they record that a name denotes a thing, which nobody taught -
    and were being reported as defective for ever, and as knowledge gaps.
    """
    kb.commit_knowledge(kb_claim(), [kb_evidence()])
    entity = kb.find_entity("Kambhoji")
    assert entity is not None

    librarian = Librarian(kb)
    assert all(i.knowledge_id != entity.knowledge_id
               for i in librarian.unsupported())
    assert all("Kambhoji is a raga" not in g.missing_information
               for g in librarian.detect_gaps(["Kambhoji"], record=False))

    # ... and asking where it came from gets an honest answer, not a blank.
    record = kb.provenance(entity.knowledge_id)
    assert record["structural"]
    assert record["why_no_source"]


def test_reg_kb_a_store_too_damaged_to_open_was_not_preserved(kb_path):
    """Integrity handling ran after the connection was configured, so a file
    corrupt enough that SQLite would not open it at all raised a raw driver
    error - skipping the preserve-and-report path section 36 requires.
    """
    from raagacomposer.kb.store import KnowledgeBaseCorrupt

    KnowledgeStore(kb_path).close()
    with open(kb_path, "r+b") as handle:
        handle.write(b"not a database, not even a little bit")

    with pytest.raises(KnowledgeBaseCorrupt):
        KnowledgeStore(kb_path)
    assert kb_path.exists()
    assert list(kb_path.parent.glob("*.damaged-*.db"))
