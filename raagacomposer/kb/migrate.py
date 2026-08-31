"""Bringing what already exists into the Knowledge Base - section 47.

The specification is explicit: inspect the project first, do not create a
parallel Knowledge Base, extend rather than duplicate, and do not silently
discard previously learned data.  Three stores were already here when this was
written, and this module is the answer to what happens to each.

``training.db``     the Training tab's record.  Its ``knowledge`` and
                    ``conflicts`` tables were a flat list with provenance but
                    no network, no versions and no evidence objects, and they
                    are migrated into the Knowledge Base here.  The rest of
                    that file - searches, candidates, runs, objectives,
                    reports - *stays where it is*, because section 26 draws
                    exactly that line: a Learning Report is what happened
                    during one run, and the Knowledge Base is the durable
                    integrated knowledge across all runs.  They are different
                    things and merging them would lose the distinction the
                    specification asks for.

``knowledge.db``    the agent's own memory.  Its ``raaga_facts`` are claims
                    and are projected in as such.  Its phrases, curriculum
                    progress and event log stay: the phrase index is on the
                    composer's hot path and is already proven, and curriculum
                    progress is run state rather than knowledge.

``raagas.json``     the shipped structural library.  Seeded as accepted
                    knowledge with a source of its own, so that a claim
                    learned from a teacher can be compared against what the
                    application already asserted - which is where most real
                    contradictions will come from.

Migration is idempotent.  Running it twice attaches no second copy of
anything, because everything goes through the same duplicate control as any
other write.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..core.logging_setup import get_logger
from . import normalize
from .models import (Conflict, ConflictState, Evidence, ExtractionMethod,
                     KnowledgeItem, KnowledgeType, Relation, Scope, Source,
                     Status)
from .service import CommitOutcome, KnowledgeBaseService

log = get_logger("kb.migrate")

#: The source every claim taken from the shipped library is attributed to.
LIBRARY_REFERENCE = "raagacomposer://raaga/data/raagas.json"

#: Which raaga-library fields become claims, and what each one is called.
LIBRARY_FIELDS = (
    ("arohanam", "arohanam", "ascends"),
    ("avarohanam", "avarohanam", "descends"),
    ("jeeva", "jeeva", "leans on"),
    ("nyasa", "nyasa", "comes to rest on"),
    ("graha", "graha", "may begin from"),
)


class MigrationReport:
    def __init__(self) -> None:
        self.library_claims = 0
        self.training_items = 0
        self.training_conflicts = 0
        self.agent_facts = 0
        self.skipped = 0
        self.errors: List[str] = []

    @property
    def total(self) -> int:
        return (self.library_claims + self.training_items + self.agent_facts)

    def summary(self) -> str:
        return (f"{self.library_claims} from the shipped library, "
                f"{self.training_items} from training, "
                f"{self.agent_facts} from the agent's memory, "
                f"{self.training_conflicts} conflict(s) carried over"
                + (f"; {len(self.errors)} error(s)" if self.errors else ""))


# --------------------------------------------------------------------------
def seed_from_library(kb: KnowledgeBaseService, raagas) -> int:
    """Bring the shipped raaga definitions in as attributed claims.

    These are not learned and should not pretend to be: they carry a source of
    their own naming the file they came from, marked ``structural_library``,
    so a teacher later disagreeing with one produces an honest conflict
    between "what we shipped" and "what a source taught" rather than a mystery.
    """
    source = kb.add_source(Source(
        source_type="structural_library",
        title="the application's shipped raaga library",
        author_or_channel="Raaga AI Music Composer",
        reference=LIBRARY_REFERENCE,
        license_or_access_notes="ships with the application",
        language="swara notation"))

    written = 0
    for raaga in raagas.all():
        entity = kb.ensure_entity(raaga.name, kind="Raga",
                                  aliases=list(raaga.aliases))
        for field_name, predicate, verb in LIBRARY_FIELDS:
            values = list(getattr(raaga, field_name, ()) or ())
            if not values:
                continue
            text = " ".join(values)
            item = KnowledgeItem(
                canonical_name=raaga.name, knowledge_type=KnowledgeType.FACT,
                subject=raaga.name, predicate=predicate, object_value=text,
                statement=f"{raaga.name} {verb} {text}.",
                structured_value=normalize.structured_for(predicate, values),
                scope=[Scope.CARNATIC, Scope.RAGA], raga=raaga.name,
                importance=0.85 if predicate in ("arohanam", "avarohanam")
                else 0.6,
                learned_by="shipped library", language="swara notation")
            evidence = Evidence(
                source_id=source.source_id, source_segment=field_name,
                strength=0.85,
                extraction_method=ExtractionMethod.STRUCTURAL,
                transcript_excerpt=text)
            if kb.commit_knowledge(item, [evidence],
                                   source_quality=0.85).outcome == \
                    CommitOutcome.NEW:
                written += 1

        for index, prayoga in enumerate(raaga.prayogas):
            text = " ".join(prayoga)
            item = KnowledgeItem(
                canonical_name=raaga.name, knowledge_type=KnowledgeType.PATTERN,
                subject=raaga.name, predicate="prayoga", object_value=text,
                statement=f"A characteristic phrase of {raaga.name}: {text}.",
                structured_value=normalize.structured_for("prayoga", prayoga),
                scope=[Scope.CARNATIC, Scope.RAGA], raga=raaga.name,
                importance=0.75, learned_by="shipped library",
                tags=["prayoga"])
            evidence = Evidence(
                source_id=source.source_id,
                source_segment=f"prayoga {index + 1}", strength=0.8,
                extraction_method=ExtractionMethod.STRUCTURAL,
                transcript_excerpt=text)
            if kb.commit_knowledge(item, [evidence],
                                   source_quality=0.85).outcome == \
                    CommitOutcome.NEW:
                written += 1

        for swara in getattr(raaga, "forbidden_swaras", ()) or ():
            item = KnowledgeItem(
                canonical_name=raaga.name,
                knowledge_type=KnowledgeType.CONSTRAINT,
                subject=raaga.name, predicate="avoid", object_value=swara,
                statement=f"{raaga.name} does not use {swara}.",
                structured_value={"kind": "swaras", "swaras": [swara]},
                scope=[Scope.CARNATIC, Scope.RAGA], raga=raaga.name,
                importance=0.8, learned_by="shipped library")
            evidence = Evidence(
                source_id=source.source_id, source_segment="forbidden_swaras",
                strength=0.85, extraction_method=ExtractionMethod.STRUCTURAL)
            if kb.commit_knowledge(item, [evidence],
                                   source_quality=0.85).outcome == \
                    CommitOutcome.NEW:
                written += 1

        # Ragas that share swaras are the ones a listener confuses, and
        # section 21 asks for the connection explicitly.
        for alias in raaga.aliases:
            kb.add_alias(entity.knowledge_id, alias) if entity else None

    log.info("seeded %d claim(s) from the shipped raaga library", written)
    return written


# --------------------------------------------------------------------------
def migrate_training_store(kb: KnowledgeBaseService,
                           training_db: Path) -> MigrationReport:
    """Carry the Training tab's flat knowledge table into the network."""
    report = MigrationReport()
    path = Path(training_db)
    if not path.exists():
        return report

    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    try:
        try:
            rows = connection.execute("SELECT * FROM knowledge").fetchall()
        except sqlite3.Error:
            return report                       # nothing of that shape here

        source_cache: Dict[str, Source] = {}
        id_map: Dict[str, str] = {}
        for row in rows:
            try:
                source = source_cache.get(row["source_id"])
                if source is None:
                    source = kb.add_source(Source(
                        source_type="training",
                        title=row["source_title"] or "a training source",
                        reference=row["source_url"] or "",
                        training_source_id=row["source_id"] or "",
                        license_or_access_notes="migrated from training.db"))
                    source_cache[row["source_id"]] = source

                predicate = normalize.normalise_predicate(
                    row["concept"] or row["category"] or "note")
                item = KnowledgeItem(
                    canonical_name=row["subject"] or row["raga"] or "",
                    knowledge_type=_type_for(row["category"] or ""),
                    subject=row["subject"] or row["raga"] or "",
                    predicate=predicate,
                    object_value=row["normalized_statement"] or "",
                    statement=row["normalized_statement"] or "",
                    structured_value=normalize.structured_for(
                        predicate, row["normalized_statement"] or ""),
                    scope=[Scope.CARNATIC, Scope.TRAINING],
                    raga=row["raga"] or "", tala=row["tala"] or "",
                    difficulty=row["difficulty"] or "",
                    learned_by="training", tags=_tags(row),
                    notes="migrated from the training store")
                evidence = Evidence(
                    source_id=source.source_id,
                    source_segment=row["source_timestamp"] or "",
                    transcript_excerpt=row["evidence"] or "",
                    strength=float(row["confidence"] or 0.5),
                    extraction_method=_method(row),
                    run_id=row["run_id"] or "")
                result = kb.commit_knowledge(item, [evidence],
                                             source_quality=0.7)
                if result.stored and result.item is not None:
                    id_map[row["knowledge_id"]] = result.item.knowledge_id
                    report.training_items += 1
                else:
                    report.skipped += 1
            except Exception as exc:  # noqa: BLE001 - one bad row is not fatal
                report.errors.append(f"{row['knowledge_id']}: {exc}")

        try:
            conflicts = connection.execute("SELECT * FROM conflicts").fetchall()
        except sqlite3.Error:
            conflicts = []
        for row in conflicts:
            mapped = id_map.get(row["knowledge_id"], "")
            if not mapped:
                continue
            kb.add_conflict(Conflict(
                claim_a=mapped, claim_b="",
                confidence_a=float(row["existing_confidence"] or 0.0),
                confidence_b=float(row["new_confidence"] or 0.0),
                resolution_status=(ConflictState.RESOLVED_A
                                   if row["resolved"]
                                   else ConflictState.UNRESOLVED),
                notes=(f"migrated from training.db. "
                       f"Existing: {row['existing_claim']}. "
                       f"New: {row['new_claim']}. "
                       f"{row['recommendation'] or ''}").strip()))
            report.training_conflicts += 1
    finally:
        connection.close()

    log.info("migrated %d item(s) from %s", report.training_items, path)
    return report


def _type_for(category: str) -> str:
    return {
        "phrase": KnowledgeType.PATTERN,
        "grammar": KnowledgeType.CONSTRAINT,
        "practice": KnowledgeType.PROCEDURE,
        "self-assessment": KnowledgeType.META,
    }.get(category, KnowledgeType.FACT)


def _tags(row: sqlite3.Row) -> List[str]:
    import json
    try:
        return json.loads(row["tags"]) or []
    except (TypeError, ValueError, IndexError):
        return []


def _method(row: sqlite3.Row) -> str:
    tags = _tags(row)
    if "heard" in tags or "measured" in tags:
        return ExtractionMethod.AUDIO
    if "stated" in tags:
        return ExtractionMethod.TRANSCRIPT
    return ExtractionMethod.INFERRED


# --------------------------------------------------------------------------
def migrate_agent_facts(kb: KnowledgeBaseService, agent_repo) -> int:
    """Project the agent's learned raaga facts in as claims.

    The agent's store keeps them too - its curriculum and its phrase index
    read from there on the composer's hot path, and breaking that to satisfy a
    tidiness argument would be the wrong trade.  What the Knowledge Base adds
    is the network, the evidence and the history around the same facts.
    """
    if agent_repo is None:
        return 0
    source = kb.add_source(Source(
        source_type="agent_memory",
        title="the agent's own learning",
        reference="raagacomposer://agent/knowledge.db",
        license_or_access_notes="the application's own study record"))

    written = 0
    try:
        raagas = {s.raaga for s in agent_repo.sources(limit=500) if s.raaga}
    except Exception:  # noqa: BLE001
        raagas = set()
    for name in sorted(raagas):
        try:
            facts = agent_repo.facts(name)
        except Exception:  # noqa: BLE001
            continue
        for fact in facts:
            predicate = normalize.normalise_predicate(fact.key)
            item = KnowledgeItem(
                canonical_name=name, knowledge_type=KnowledgeType.FACT,
                subject=name, predicate=predicate, object_value=fact.value,
                statement=f"{name}: {fact.key} is {fact.value}.",
                structured_value=normalize.structured_for(predicate,
                                                          fact.value),
                scope=[Scope.CARNATIC, Scope.RAGA], raga=name,
                learned_by="the agent's own study",
                notes=fact.notes or "")
            evidence = Evidence(
                source_id=source.source_id, source_segment=fact.key,
                strength=float(fact.confidence or 0.5),
                extraction_method=ExtractionMethod.AUDIO,
                transcript_excerpt=fact.value)
            if kb.commit_knowledge(item, [evidence],
                                   source_quality=0.65).stored:
                written += 1
    log.info("projected %d agent fact(s) into the Knowledge Base", written)
    return written


# --------------------------------------------------------------------------
def migrate_all(kb: KnowledgeBaseService, *, raagas=None,
                training_db: Optional[Path] = None,
                agent_repo=None) -> MigrationReport:
    """Everything, once, idempotently.  Section 47's "do not discard"."""
    report = MigrationReport()
    already = kb.store.get_meta("migrated_existing_stores")

    if raagas is not None:
        try:
            report.library_claims = seed_from_library(kb, raagas)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"library: {exc}")

    if training_db is not None:
        try:
            training = migrate_training_store(kb, training_db)
            report.training_items = training.training_items
            report.training_conflicts = training.training_conflicts
            report.errors.extend(training.errors)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"training: {exc}")

    if agent_repo is not None:
        try:
            report.agent_facts = migrate_agent_facts(kb, agent_repo)
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"agent: {exc}")

    kb.store.set_meta("migrated_existing_stores", str(time.time()))
    kb.store.audit("kb.migration", report.summary())
    if already:
        log.info("migration ran again; duplicate control kept it idempotent")
    log.info("migration complete: %s", report.summary())
    return report
