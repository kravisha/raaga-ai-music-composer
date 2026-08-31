"""The knowledge-base specification's own acceptance tests - section 45.

Ten tests, numbered as the specification numbers them, run against the real
service with a real file on disk.  Several of them close the application and
open it again, because the thing being checked is that nothing was recreated.
"""
from __future__ import annotations

import pytest

from raagacomposer.kb.context import KnowledgeContextBuilder
from raagacomposer.kb.librarian import Librarian
from raagacomposer.kb.models import (ConflictState, Evidence, ExtractionMethod,
                                     KnowledgeItem, KnowledgeType, Source,
                                     Status)
from raagacomposer.kb.service import CommitOutcome, KnowledgeBaseService
from raagacomposer.kb.store import KnowledgeStore

pytestmark = pytest.mark.integration


def _reopen(path) -> KnowledgeBaseService:
    """What the application does on its next start."""
    return KnowledgeBaseService.initialize_if_needed(path)


# ==========================================================================
# TEST 1: PERSISTENCE
# ==========================================================================
def test_45_1_knowledge_survives_a_restart(kb_path, kb_claim, kb_evidence):
    kb = _reopen(kb_path)
    source = kb.add_source(Source(title="A lesson",
                                  reference="https://youtu.be/AAAAAAAAAAA"))
    item = kb.commit_knowledge(
        kb_claim(), [Evidence(source_id=source.source_id, strength=0.8,
                              extraction_method=ExtractionMethod.AUDIO)]).item
    knowledge_id = item.knowledge_id
    kb.close()

    reopened = _reopen(kb_path)
    try:
        found = reopened.get_by_id(knowledge_id)
        assert found is not None, "the item disappeared"
        assert found.raga == "Kambhoji"
        assert reopened.find_entity("Kambhoji") is not None
    finally:
        reopened.close()


# ==========================================================================
# TEST 2: SOURCE PROVENANCE
# ==========================================================================
def test_45_2_a_learned_fact_can_name_its_source_run_and_evidence(
        kb, kb_claim, kb_evidence, kb_source):
    item = kb.commit_knowledge(kb_claim(),
                               [kb_evidence(run_id="run_42")],
                               run_id="run_42").item
    record = kb.provenance(item.knowledge_id)

    assert record["where_from"], "no source"
    assert record["where_from"][0]["title"] == kb_source.title
    assert record["where_in_the_source"], "no evidence"
    assert record["where_in_the_source"][0]["run"] == "run_42"
    assert record["where_in_the_source"][0]["segment"]
    assert record["audio_transcript_or_inferred"]


# ==========================================================================
# TEST 3: DUPLICATE LEARNING
# ==========================================================================
def test_45_3_one_canonical_fact_with_two_evidence_records(kb, kb_claim,
                                                           kb_evidence):
    first = kb.commit_knowledge(kb_claim(), [kb_evidence()])
    second_source = kb.add_source(Source(
        title="A second teacher", reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(
        kb_claim(statement="The ascent is S R2 G3 M1 P D2 S+."),
        [kb_evidence(second_source)])

    assert second.item.knowledge_id == first.item.knowledge_id
    evidence = kb.evidence_for(first.item.knowledge_id)
    assert len(evidence) == 2
    assert len({e.source_id for e in evidence}) == 2


# ==========================================================================
# TEST 4: CONTRADICTION
# ==========================================================================
def test_45_4_a_conflict_is_recorded_and_nothing_is_overwritten(
        kb, kb_claim, kb_evidence):
    first = kb.commit_knowledge(
        kb_claim(value="S R2 G3 M1 P D2 S+",
                 statement="Kambhoji ascends S R2 G3 M1 P D2 S+."),
        [kb_evidence()])
    disagreeing = kb.add_source(Source(
        title="A second teacher", reference="https://youtu.be/BBBBBBBBBBB"))
    second = kb.commit_knowledge(
        kb_claim(value="S R2 G3 M1 P D2 N2 S+",
                 statement="Kambhoji ascends S R2 G3 M1 P D2 N2 S+."),
        [kb_evidence(disagreeing)])

    assert second.outcome == CommitOutcome.CONTRADICTION
    assert kb.conflicts(open_only=True)
    kept = kb.get_by_id(first.item.knowledge_id)
    assert kept.object_value == "S R2 G3 M1 P D2 S+", "it was overwritten"
    assert kb.get_by_id(second.item.knowledge_id) is not None


# ==========================================================================
# TEST 5: RETRIEVAL
# ==========================================================================
def test_45_5_retrieval_answers_what_is_needed_before_generating(
        kb, kb_claim, kb_evidence):
    """"What do I need to know before generating a Kambhoji melody?" - the
    answer must include structure, phrases, constraints and cautions, not
    only text that shares words with the question."""
    kb.commit_knowledge(kb_claim("arohanam", "S R2 G3 M1 P D2 S+"),
                        [kb_evidence()])
    kb.commit_knowledge(kb_claim("avarohanam", "S+ N2 D2 P M1 G3 R2 S"),
                        [kb_evidence()])
    kb.commit_knowledge(
        kb_claim("prayoga", "P D2 S+ N2", "A characteristic phrase: P D2 S+ N2.",
                 knowledge_type=KnowledgeType.PATTERN), [kb_evidence()])
    kb.commit_knowledge(
        kb_claim("avoid", "N3", "Kambhoji does not use N3.",
                 knowledge_type=KnowledgeType.CONSTRAINT), [kb_evidence()])
    weak = kb.add_source(Source(title="An uncertain source",
                                reference="https://youtu.be/CCCCCCCCCCC"))
    kb.commit_knowledge(
        kb_claim("tempo", "unclear", "The tempo is unclear."),
        [kb_evidence(weak, strength=0.1,
                     method=ExtractionMethod.INFERRED)])

    context = KnowledgeContextBuilder(kb).build("melody", raga="Kambhoji")
    predicates = {i.predicate for i in context.items}
    assert "arohanam" in predicates
    assert "avarohanam" in predicates
    assert "prayoga" in predicates
    assert context.constraints, "the constraints were not included"
    rendered = context.render()
    assert "Must not" in rendered


# ==========================================================================
# TEST 6: USER CORRECTION
# ==========================================================================
def test_45_6_a_correction_and_its_history_survive_a_restart(
        kb_path, kb_claim, kb_evidence):
    kb = _reopen(kb_path)
    source = kb.add_source(Source(title="A lesson",
                                  reference="https://youtu.be/AAAAAAAAAAA"))
    item = kb.commit_knowledge(
        kb_claim(), [Evidence(source_id=source.source_id, strength=0.8,
                              extraction_method=ExtractionMethod.AUDIO)]).item
    kb.correct_knowledge(item.knowledge_id, action="mark_incorrect",
                         explanation="my teacher says otherwise")
    kb.close()

    reopened = _reopen(kb_path)
    try:
        updated = reopened.get_by_id(item.knowledge_id)
        assert updated.status == Status.REJECTED
        record = reopened.provenance(item.knowledge_id)
        assert record["corrections"], "the correction was lost"
        assert record["corrections"][0]["explanation"] == \
            "my teacher says otherwise"
        assert len(reopened.versions_of(item.knowledge_id)) >= 2
    finally:
        reopened.close()


# ==========================================================================
# TEST 7: CRASH RECOVERY
# ==========================================================================
def test_45_7_committed_knowledge_survives_an_abrupt_termination(
        kb_path, kb_claim):
    """No clean close at all: the store object is abandoned the way a killed
    process abandons it."""
    kb = _reopen(kb_path)
    source = kb.add_source(Source(title="A lesson",
                                  reference="https://youtu.be/AAAAAAAAAAA"))
    item = kb.commit_knowledge(
        kb_claim(), [Evidence(source_id=source.source_id, strength=0.8,
                              extraction_method=ExtractionMethod.AUDIO)]).item
    del kb                     # no close(), no checkpoint, nothing

    reopened = _reopen(kb_path)
    try:
        assert reopened.get_by_id(item.knowledge_id) is not None
        assert reopened.evidence_for(item.knowledge_id)
    finally:
        reopened.close()


# ==========================================================================
# TEST 8: TRAINING INTEGRATION
# ==========================================================================
def test_45_8_a_training_run_integrates_into_the_knowledge_base(
        training_settings, agent_repo, kb_path):
    """One approved source, learned, and its concepts must end up in the KB
    linked to that source - not filed as a video summary."""
    from raagacomposer.training.controller import TrainingController

    kb = _reopen(kb_path)
    training = TrainingController(training_settings, agent_repo=agent_repo,
                                  kb=kb)
    try:
        results = training.search("Keeravani characteristic phrases")
        training.add_to_queue([results[0].source_id])
        report = training.learn_one_now()
        assert report is not None and report.learned

        items = kb.items_about("Keeravani")
        assert items, "nothing from the run reached the Knowledge Base"

        # Every *claim* carries its evidence.  The entity the claims hang off
        # is structure rather than a claim and correctly has none.
        claims = [i for i in items if i.knowledge_type != KnowledgeType.ENTITY]
        assert claims, "the run produced no claims, only an entity"
        for item in claims:
            assert kb.evidence_for(item.knowledge_id), \
                f"a claim arrived without evidence: {item.display()}"
        # ... and it is linked to the source the run used.
        sources = kb.sources_for(claims[0].knowledge_id)
        assert sources
        assert sources[0].training_source_id == results[0].source_id
    finally:
        training.close()
        kb.close()


def test_45_8b_a_second_run_of_the_same_source_does_not_double_the_knowledge(
        training_settings, agent_repo, kb_path):
    """Section 26's distinction, checked: the Learning Report is per run, the
    Knowledge Base accumulates across runs without piling up copies."""
    from raagacomposer.training.controller import TrainingController

    kb = _reopen(kb_path)
    training = TrainingController(training_settings, agent_repo=agent_repo,
                                  kb=kb)
    try:
        results = training.search("Keeravani characteristic phrases")
        training.add_to_queue([results[0].source_id])
        training.learn_one_now()
        after_first = kb.store.count("knowledge_items")

        training.relearn(results[0].source_id)
        training.learn_one_now()
        after_second = kb.store.count("knowledge_items")

        assert after_second == after_first, "relearning duplicated knowledge"
        assert kb.store.count("evidence") >= 1
    finally:
        training.close()
        kb.close()


# ==========================================================================
# TEST 9: COMPOSITION INTEGRATION
# ==========================================================================
def test_45_9_compose_reads_the_knowledge_base_through_the_service(
        training_settings, tmp_path):
    """Compose must go through KnowledgeBaseService, and the retrieval must
    leave a trace that can be inspected afterwards."""
    from raagacomposer.app import AppController

    app = AppController(training_settings)
    try:
        assert app.kb is not None, "the application has no Knowledge Base"
        context = app.knowledge_for("compose", raaga="Kambhoji")
        assert context is not None
        assert context.items, "composition was given nothing to work from"

        trace = app.kb.store.query(
            "SELECT * FROM retrieval_usage ORDER BY id DESC LIMIT 1")
        assert trace, "no retrieval trace was recorded"
        assert trace[0]["task"] == "compose"
        assert trace[0]["item_count"] == len(context.knowledge_ids)
    finally:
        app.close()


def test_45_9b_a_composition_choice_can_be_explained(training_settings):
    """Section 41 - which knowledge was used, and where it came from."""
    from raagacomposer.app import AppController

    app = AppController(training_settings)
    try:
        context = app.knowledge_for("compose", raaga="Kambhoji")
        # A learned claim, not the entity node the claims hang off: the entity
        # is structure and has no source by design.
        chosen = next(i for i in context.items
                      if i.knowledge_type != KnowledgeType.ENTITY)
        record = app.explain_knowledge(chosen.knowledge_id)
        assert record["statement"]
        assert record["where_from"], "the knowledge cannot name its source"
        assert record["confidence_explained"]
        assert not record["structural"]

        # ... and asking the same of a structural node gets an honest answer
        # rather than an empty list that would read like lost provenance.
        entity = app.kb.find_entity("Kambhoji")
        structural = app.explain_knowledge(entity.knowledge_id)
        assert structural["structural"]
        assert structural["why_no_source"]
    finally:
        app.close()


# ==========================================================================
# TEST 10: NO DESTRUCTIVE REINITIALIZATION
# ==========================================================================
def test_45_10_restarting_the_application_keeps_what_was_learned(
        training_settings):
    """The rule the whole specification turns on, through the real
    application: build a Knowledge Base, restart, find it intact."""
    from raagacomposer.app import AppController

    first = AppController(training_settings)
    try:
        source = first.kb.add_source(Source(
            title="Something the user taught it",
            reference="https://example.org/a-lesson"))
        item = first.kb.commit_knowledge(
            KnowledgeItem(
                canonical_name="Kambhoji", subject="Kambhoji",
                predicate="gamaka", object_value="kampita on G3",
                statement="G3 carries a kampita in Kambhoji.",
                raga="Kambhoji", knowledge_type=KnowledgeType.FACT),
            [Evidence(source_id=source.source_id, strength=0.9,
                      extraction_method=ExtractionMethod.AUDIO)]).item
        knowledge_id = item.knowledge_id
        before = first.kb.store.count("knowledge_items")
        stamp = first.kb.store.initialized_at
    finally:
        first.close()

    second = AppController(training_settings)
    try:
        assert second.kb.store.initialized_at == stamp, "it was reinitialized"
        assert second.kb.store.count("knowledge_items") == before
        assert second.kb.get_by_id(knowledge_id) is not None
    finally:
        second.close()


def test_45_10b_a_schema_migration_does_not_clear_knowledge(kb_path,
                                                            kb_claim):
    """"...and schema migration does not clear it"."""
    kb = _reopen(kb_path)
    source = kb.add_source(Source(title="A lesson",
                                  reference="https://youtu.be/AAAAAAAAAAA"))
    item = kb.commit_knowledge(
        kb_claim(), [Evidence(source_id=source.source_id, strength=0.8,
                              extraction_method=ExtractionMethod.AUDIO)]).item
    before = kb.store.count("knowledge_items")
    # Pretend the file was written by an older schema.
    kb.store.set_meta("schema_version", "0")
    kb.close()

    migrated = _reopen(kb_path)
    try:
        assert migrated.store.count("knowledge_items") == before
        assert migrated.get_by_id(item.knowledge_id) is not None
        assert migrated.store.count("migrations") == 1
    finally:
        migrated.close()
