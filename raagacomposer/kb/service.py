"""KnowledgeBaseService - the way everything else talks to the KB.

Specification section 38.  Other subsystems use this rather than the tables,
so the rules the specification cares about live in one place and cannot be
sidestepped by a caller that writes SQL of its own.

The rules, and where they are:

* nothing is stored without provenance - :meth:`commit_knowledge` refuses an
  item with no evidence and no source (section 9).
* a duplicate attaches evidence instead of making another row; a near
  duplicate is offered for merging; a refinement makes a version
  (section 15).
* a contradiction records a conflict and leaves both claims standing
  (section 12), and nothing is ever overwritten in place without the old
  reading being kept (section 13).
* a candidate becomes durable knowledge by going through validation, not by
  being written (section 11).

The write path is one transaction, so a failed learning run leaves no
half-created canonical knowledge (section 34).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from . import confidence as confidence_model
from . import normalize
from .models import (Conflict, ConflictState, Evidence, ExtractionMethod,
                     ExampleDetail, FailureLesson, HealthReport, KnowledgeGap,
                     KnowledgeItem, KnowledgeType, ProcedureDetail, Relation,
                     Relationship, Scope, Source, Status, UserCorrection,
                     Version, new_id)
from .store import (KnowledgeBaseCorrupt, KnowledgeBaseError, KnowledgeStore,
                    dumps, loads)

log = get_logger("kb.service")

#: How alike two statements must be before one is offered as a near duplicate.
NEAR_DUPLICATE = 0.72


class CommitOutcome:
    """What happened to one candidate.  Section 15's four answers."""

    NEW = "new"
    DUPLICATE = "duplicate"           # same claim, evidence attached
    REFINEMENT = "refinement"         # better wording, new version
    CONTRADICTION = "contradiction"   # both kept, conflict recorded
    REJECTED = "rejected"             # not enough to store


class CommitResult:
    def __init__(self, outcome: str, item: Optional[KnowledgeItem] = None,
                 conflict: Optional[Conflict] = None, reason: str = "") -> None:
        self.outcome = outcome
        self.item = item
        self.conflict = conflict
        self.reason = reason

    @property
    def stored(self) -> bool:
        return self.item is not None and self.outcome != CommitOutcome.REJECTED

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CommitResult({self.outcome}, {self.reason!r})"


class KnowledgeBaseService:
    """Section 38.  Open it once, write through it, read through it."""

    def __init__(self, store: Optional[KnowledgeStore] = None,
                 path: Optional[Path] = None, *, create: bool = True) -> None:
        self.store = store or KnowledgeStore(path, create=create)

    # ==================================================================
    # lifecycle
    # ==================================================================
    @classmethod
    def initialize_if_needed(cls, path: Optional[Path] = None
                             ) -> "KnowledgeBaseService":
        """Open the existing Knowledge Base; create one only on a first run."""
        service = cls(path=path, create=True)
        if service.store.count("knowledge_items") == 0:
            service.seed_taxonomy()
        return service

    def seed_taxonomy(self) -> int:
        """Section 31 step 3.  The categories, not anybody's claims.

        These are structural: they say that a Raga is a kind of thing this
        application knows about.  They carry no evidence because no source
        taught them, and they are marked accepted because there is nothing to
        verify.
        """
        from .schema import CORE_TAXONOMY

        written = 0
        for name, kind, description in CORE_TAXONOMY:
            existing = self.find_entity(name)
            if existing is not None:
                continue
            item = KnowledgeItem(
                canonical_name=name, knowledge_type=KnowledgeType.ENTITY,
                subject=name, predicate="is_a", object_value="concept",
                statement=f"{name}: {description}.",
                structured_value={"kind": "taxonomy"},
                scope=[Scope.CARNATIC, Scope.GLOBAL],
                status=Status.ACCEPTED, confidence=1.0, importance=0.9,
                learned_by="core taxonomy",
                notes="part of the shipped taxonomy, not learned from a source")
            self._insert(item)
            written += 1
        if written:
            self.store.audit("kb.taxonomy", f"{written} core concept(s) seeded")
        return written

    def ensure_entity(self, name: str, *, kind: str = "Raga",
                      aliases: Sequence[str] = ()) -> Optional[KnowledgeItem]:
        """Find or create the node a claim is about - section 3.

        This is what makes the Knowledge Base a network rather than a pile.  A
        claim that Kambhoji ascends a certain way is only reachable from
        "Kambhoji" if there is a Kambhoji to reach it from, so committing such
        a claim brings the entity into being if it is not already there.

        The entity carries no evidence and is not a claim about anything: it
        asserts only that this name denotes a thing of this kind, which is
        identity rather than musical knowledge.  Everything that *is* a claim
        hangs off it.
        """
        if not name.strip():
            return None
        existing = self.find_entity(name)
        if existing is not None:
            for alias in aliases:
                self.add_alias(existing.knowledge_id, alias)
            return existing
        item = KnowledgeItem(
            canonical_name=name, knowledge_type=KnowledgeType.ENTITY,
            subject=name, predicate="is_a", object_value=kind,
            statement=f"{name} is a {kind.lower()}.",
            structured_value={"kind": "entity", "of": kind},
            scope=[Scope.CARNATIC, Scope.RAGA if kind == "Raga" else Scope.MUSIC],
            status=Status.ACCEPTED, confidence=0.95, importance=0.85,
            raga=name if kind == "Raga" else "",
            tala=name if kind == "Tala" else "",
            learned_by="entity resolution",
            notes="an identity node: what claims about this name attach to")
        self._insert(item)
        for alias in aliases:
            self.add_alias(item.knowledge_id, alias)
        parent = self.find_entity(kind)
        if parent is not None:
            self.add_relationship(item.knowledge_id, Relation.IS_A,
                                  parent.knowledge_id, confidence=0.95,
                                  evidence="core taxonomy")
        self.store.audit("entity.created", f"{name} ({kind})",
                         knowledge_id=item.knowledge_id)
        return item

    def health_check(self) -> HealthReport:
        """Section 40."""
        report = HealthReport()
        report.total_items = self.store.count("knowledge_items")
        report.sources = self.store.count("sources")
        report.evidence = self.store.count("evidence")
        report.relationships = self.store.count("relationships")
        report.schema_version = self.store.schema_version
        report.initialized_at = self.store.initialized_at

        for row in self.store.query(
                "SELECT status, COUNT(*) AS n FROM knowledge_items "
                "GROUP BY status"):
            report.by_status[row["status"]] = int(row["n"])
        for row in self.store.query(
                "SELECT knowledge_type, COUNT(*) AS n FROM knowledge_items "
                "GROUP BY knowledge_type"):
            report.by_type[row["knowledge_type"]] = int(row["n"])

        row = self.store.one(
            "SELECT COUNT(*) AS n FROM conflicts WHERE resolution_status IN "
            "('unresolved','context_dependent')")
        report.unresolved_conflicts = int(row["n"]) if row else 0
        row = self.store.one(
            "SELECT COUNT(*) AS n FROM knowledge_gaps WHERE status='open'")
        report.knowledge_gaps = int(row["n"]) if row else 0
        row = self.store.one(
            "SELECT COUNT(*) AS n FROM knowledge_items WHERE confidence < 0.3 "
            "AND status NOT IN ('rejected','superseded','deprecated')")
        report.low_confidence_items = int(row["n"]) if row else 0
        row = self.store.one(
            "SELECT COUNT(*) AS n FROM knowledge_items k WHERE NOT EXISTS ("
            "SELECT 1 FROM relationships r WHERE r.source_knowledge_id="
            "k.knowledge_id OR r.target_knowledge_id=k.knowledge_id)")
        report.orphan_items = int(row["n"]) if row else 0

        report.last_backup = float(self.store.get_meta("last_backup") or 0.0)
        report.integrity_ok = self.store.integrity_ok
        report.last_integrity_check = self.store.last_integrity_check
        return report

    def close(self) -> None:
        self.store.close()

    # ==================================================================
    # writing (sections 14, 15, 34)
    # ==================================================================
    def add_candidate(self, item: KnowledgeItem,
                      evidence: Sequence[Evidence] = ()) -> KnowledgeItem:
        """Stage a candidate.  Nothing durable happens yet - section 34."""
        item.status = Status.CANDIDATE
        item.updated_at = time.time()
        for record in evidence:
            record.knowledge_id = item.knowledge_id
        return item

    def commit_knowledge(self, item: KnowledgeItem,
                         evidence: Sequence[Evidence] = (),
                         *, source_quality: float = 0.6,
                         relationships: Sequence[Relationship] = (),
                         run_id: str = "",
                         allow_without_evidence: bool = False) -> CommitResult:
        """Validate, compare, and store - the whole of section 14 for one item.

        One transaction.  Either the item, its evidence, its relationships and
        its version row are all there, or none of them is.
        """
        evidence = list(evidence)

        # Section 9: no important learned knowledge may lose its source.
        if not evidence and not allow_without_evidence:
            return CommitResult(
                CommitOutcome.REJECTED, reason=(
                    "refused: nothing supports this. Learned knowledge must "
                    "carry its evidence."))

        item.subject = item.subject or item.canonical_name
        item.canonical_name = item.canonical_name or item.subject
        identity = normalize.identity_of(item.subject, item.predicate,
                                         item.raga, item.tala,
                                         value=item.object_value)
        existing = self._by_identity(identity)

        scored = confidence_model.score(
            evidence, source_quality=source_quality,
            agrees_with_existing=None if existing is None
            else self._says_the_same(existing, item))
        item.confidence = scored.value
        item.confidence_parts = scored.parts

        if existing is None:
            return self._commit_new(item, identity, evidence, relationships,
                                    run_id)
        if self._says_the_same(existing, item):
            return self._commit_duplicate(existing, item, evidence,
                                          source_quality, run_id)
        # A refinement is a better *wording* of the same claim.  Where the
        # structured values disagree the substance differs, however alike the
        # sentences read, and calling that a refinement would overwrite one
        # teacher's scale with another's under the name of tidying up.
        similar = normalize.similarity(existing.statement, item.statement)
        if similar >= NEAR_DUPLICATE and not self._structured_disagree(
                existing, item):
            return self._commit_refinement(existing, item, evidence,
                                           source_quality, run_id)
        return self._commit_contradiction(existing, item, evidence,
                                          source_quality, run_id)

    # -- the four outcomes ---------------------------------------------
    def _commit_new(self, item: KnowledgeItem, identity: str,
                    evidence: Sequence[Evidence],
                    relationships: Sequence[Relationship],
                    run_id: str) -> CommitResult:
        item.status = Status.LEARNED
        item.source_count = len({e.source_id for e in evidence if e.source_id})
        # Section 3: a claim has to hang off the thing it is about, or nothing
        # can ever reach it by asking about that thing.
        anchor = None
        if item.raga:
            anchor = self.ensure_entity(item.raga, kind="Raga")
        elif item.tala:
            anchor = self.ensure_entity(item.tala, kind="Tala")
        elif item.subject and item.knowledge_type != KnowledgeType.ENTITY:
            anchor = self.find_entity(item.subject)
        with self.store.transaction():
            self._insert(item, identity=identity, in_transaction=True)
            for record in evidence:
                record.knowledge_id = item.knowledge_id
                record.run_id = record.run_id or run_id
                self._insert_evidence(record)
            for relationship in relationships:
                relationship.source_knowledge_id = (
                    relationship.source_knowledge_id or item.knowledge_id)
                self._insert_relationship(relationship)
            self._insert_version(item, "first recorded", run_id=run_id)
            if anchor is not None and anchor.knowledge_id != item.knowledge_id:
                relation = (Relation.BELONGS_TO_RAGA if item.raga
                            else Relation.BELONGS_TO_TALA if item.tala
                            else Relation.RELATED_TO)
                self._insert_relationship(Relationship(
                    source_knowledge_id=item.knowledge_id,
                    relation_type=relation,
                    target_knowledge_id=anchor.knowledge_id,
                    confidence=0.9,
                    evidence="attached to the entity the claim is about"))
        self.store.audit("knowledge.new", item.display()[:200],
                         knowledge_id=item.knowledge_id, run_id=run_id)
        return CommitResult(CommitOutcome.NEW, item)

    def _commit_duplicate(self, existing: KnowledgeItem, incoming: KnowledgeItem,
                          evidence: Sequence[Evidence], source_quality: float,
                          run_id: str) -> CommitResult:
        """Section 15: the same fact from another source gains evidence.

        This is the behaviour that stops ten teachers producing ten rows.  The
        canonical item stays; what grows is what stands behind it.
        """
        with self.store.transaction():
            for record in evidence:
                record.knowledge_id = existing.knowledge_id
                record.run_id = record.run_id or run_id
                self._insert_evidence(record)
        refreshed = self.recompute_confidence(existing.knowledge_id,
                                              source_quality=source_quality)
        self.store.audit(
            "knowledge.confirmed",
            f"another source supports this ({len(evidence)} record(s))",
            knowledge_id=existing.knowledge_id, run_id=run_id)
        return CommitResult(CommitOutcome.DUPLICATE, refreshed or existing)

    def _commit_refinement(self, existing: KnowledgeItem,
                           incoming: KnowledgeItem,
                           evidence: Sequence[Evidence], source_quality: float,
                           run_id: str) -> CommitResult:
        """Section 13: a better wording of the same claim becomes a version."""
        with self.store.transaction():
            self._insert_version(existing, "refined by a later source",
                                 run_id=run_id)
            self.store.execute(
                "UPDATE knowledge_items SET statement=?, structured_value=?, "
                "version=version+1, updated_at=? WHERE knowledge_id=?",
                (incoming.statement, dumps(incoming.structured_value),
                 time.time(), existing.knowledge_id))
            for record in evidence:
                record.knowledge_id = existing.knowledge_id
                record.run_id = record.run_id or run_id
                self._insert_evidence(record)
        refreshed = self.recompute_confidence(existing.knowledge_id,
                                              source_quality=source_quality)
        self.store.audit("knowledge.refined", incoming.statement[:200],
                         knowledge_id=existing.knowledge_id, run_id=run_id)
        return CommitResult(CommitOutcome.REFINEMENT, refreshed or existing)

    def _commit_contradiction(self, existing: KnowledgeItem,
                              incoming: KnowledgeItem,
                              evidence: Sequence[Evidence],
                              source_quality: float,
                              run_id: str) -> CommitResult:
        """Section 12: both survive, a conflict is recorded, a person decides.

        The existing claim is not touched beyond being marked disputed.  The
        incoming one is stored as disputed too, because at this point we do
        not know which is right - and saying so is the honest answer.
        """
        identity = normalize.identity_of(incoming.subject, incoming.predicate,
                                         incoming.raga, incoming.tala,
                                         value=incoming.object_value)
        incoming.status = Status.DISPUTED
        incoming.source_count = len({e.source_id for e in evidence
                                     if e.source_id})
        conflict = Conflict(
            claim_a=existing.knowledge_id, claim_b=incoming.knowledge_id,
            source_a=self._first_source(existing.knowledge_id),
            source_b=(evidence[0].source_id if evidence else ""),
            confidence_a=existing.confidence, confidence_b=incoming.confidence,
            resolution_status=ConflictState.UNRESOLVED,
            notes=self._recommend(existing, incoming))

        with self.store.transaction():
            # Identity is deliberately suffixed so the disputing claim can
            # coexist: two rows, one conflict, neither overwritten.
            self._insert(incoming, identity=f"{identity}:{incoming.knowledge_id[-8:]}",
                         in_transaction=True)
            for record in evidence:
                record.knowledge_id = incoming.knowledge_id
                record.run_id = record.run_id or run_id
                self._insert_evidence(record)
            self._insert_version(incoming, "recorded as a disputing claim",
                                 run_id=run_id)
            self.store.execute(
                "UPDATE knowledge_items SET status=?, updated_at=? "
                "WHERE knowledge_id=? AND status NOT IN ('rejected')",
                (Status.DISPUTED, time.time(), existing.knowledge_id))
            self._insert_conflict(conflict)
            for pair in ((existing.knowledge_id, incoming.knowledge_id),
                         (incoming.knowledge_id, existing.knowledge_id)):
                self._insert_relationship(Relationship(
                    source_knowledge_id=pair[0],
                    relation_type=Relation.CONFLICTS_WITH,
                    target_knowledge_id=pair[1],
                    evidence="recorded automatically when the two met"))
        self.store.audit("knowledge.conflict",
                         f"{existing.display()[:100]} vs {incoming.display()[:100]}",
                         knowledge_id=incoming.knowledge_id, run_id=run_id)
        return CommitResult(CommitOutcome.CONTRADICTION, incoming, conflict)

    @staticmethod
    def _recommend(existing: KnowledgeItem, incoming: KnowledgeItem) -> str:
        difference = incoming.confidence - existing.confidence
        if difference > 0.25:
            return ("the new claim is markedly better supported; consider "
                    "accepting it and superseding what is held")
        if difference < -0.25:
            return ("what is already held is better supported; consider "
                    "keeping it and recording the new claim as an alternative")
        return ("about equally supported - teachers do differ here, and this "
                "may be context-dependent rather than one of them being wrong")

    # ==================================================================
    # reading
    # ==================================================================
    def get_by_id(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        row = self.store.one(
            "SELECT * FROM knowledge_items WHERE knowledge_id=?",
            (knowledge_id,))
        return self._row_to_item(row) if row else None

    def find_entity(self, name: str,
                    knowledge_type: str = KnowledgeType.ENTITY
                    ) -> Optional[KnowledgeItem]:
        """By canonical name or any alias, across spellings - section 8."""
        normalised = normalize.normalise_name(name)
        row = self.store.one(
            "SELECT k.* FROM knowledge_items k JOIN aliases a "
            "ON a.knowledge_id = k.knowledge_id "
            "WHERE a.normalised=? AND k.knowledge_type=? LIMIT 1",
            (normalised, knowledge_type))
        if row is not None:
            return self._row_to_item(row)
        for candidate in self.store.query(
                "SELECT * FROM knowledge_items WHERE knowledge_type=?",
                (knowledge_type,)):
            if normalize.normalise_name(candidate["canonical_name"]) == normalised:
                return self._row_to_item(candidate)
        return None

    def items_about(self, subject: str, *, usable_only: bool = True,
                    limit: int = 500) -> List[KnowledgeItem]:
        """Everything held about one thing, however it is spelled."""
        normalised = normalize.normalise_name(subject)
        rows = self.store.query(
            "SELECT * FROM knowledge_items ORDER BY importance DESC, "
            "confidence DESC LIMIT ?", (limit * 4,))
        out: List[KnowledgeItem] = []
        for row in rows:
            if normalize.normalise_name(row["subject"]) == normalised or \
                    normalize.normalise_name(row["raga"]) == normalised:
                item = self._row_to_item(row)
                if usable_only and not item.usable:
                    continue
                out.append(item)
            if len(out) >= limit:
                break
        return out

    def evidence_for(self, knowledge_id: str) -> List[Evidence]:
        rows = self.store.query(
            "SELECT * FROM evidence WHERE knowledge_id=? ORDER BY created_at",
            (knowledge_id,))
        return [self._row_to_evidence(r) for r in rows]

    def sources_for(self, knowledge_id: str) -> List[Source]:
        rows = self.store.query(
            "SELECT DISTINCT s.* FROM sources s JOIN evidence e "
            "ON e.source_id = s.source_id WHERE e.knowledge_id=?",
            (knowledge_id,))
        return [self._row_to_source(r) for r in rows]

    def versions_of(self, knowledge_id: str) -> List[Version]:
        rows = self.store.query(
            "SELECT * FROM knowledge_versions WHERE knowledge_id=? "
            "ORDER BY version", (knowledge_id,))
        return [Version(
            version_id=r["version_id"], knowledge_id=r["knowledge_id"],
            version=int(r["version"]), snapshot=loads(r["snapshot"], {}),
            changed_at=float(r["changed_at"] or 0.0), reason=r["reason"] or "",
            caused_by_source_id=r["caused_by_source_id"] or "",
            caused_by_run_id=r["caused_by_run_id"] or "",
            changed_by=r["changed_by"] or "system") for r in rows]

    def conflicts(self, *, open_only: bool = True,
                  limit: int = 200) -> List[Conflict]:
        clause = ("WHERE resolution_status IN ('unresolved','context_dependent')"
                  if open_only else "")
        rows = self.store.query(
            f"SELECT * FROM conflicts {clause} ORDER BY created_at DESC "
            f"LIMIT ?", (limit,))
        return [self._row_to_conflict(r) for r in rows]

    def provenance(self, knowledge_id: str) -> Dict[str, Any]:
        """Section 9's six questions and section 41's three, answered."""
        item = self.get_by_id(knowledge_id)
        if item is None:
            return {}
        evidence = self.evidence_for(knowledge_id)
        supporting = [e for e in evidence if e.supports]
        against = [e for e in evidence if not e.supports]
        sources = {s.source_id: s for s in self.sources_for(knowledge_id)}
        scored = confidence_model.score(
            evidence, human_confirmed=item.review_state == "confirmed")
        # An entity or a taxonomy row asserts only that a name denotes a thing
        # of a kind.  Nobody taught it, so it has no evidence - and saying
        # that plainly is different from an empty list, which would read like
        # a learned claim that had lost its provenance.
        structural = (item.knowledge_type == KnowledgeType.ENTITY
                      or item.learned_by in ("core taxonomy",
                                             "entity resolution"))
        return {
            "knowledge_id": knowledge_id,
            "statement": item.display(),
            "status": item.status,
            "confidence": item.confidence,
            "structural": structural,
            "why_no_source": (
                "this is structure rather than a claim: it records that the "
                "name denotes a thing of this kind, which no source taught"
                if structural and not evidence else ""),
            "confidence_explained": (
                confidence_model.ConfidenceResult(
                    item.confidence, item.confidence_parts).explain()),
            "where_from": [
                {"source_id": s.source_id, "title": s.title,
                 "author": s.author_or_channel, "reference": s.reference,
                 "type": s.source_type}
                for s in sources.values()],
            "where_in_the_source": [
                {"source": sources[e.source_id].title
                 if e.source_id in sources else e.source_id,
                 "segment": e.source_segment,
                 "from": e.timestamp_start, "to": e.timestamp_end,
                 "how": e.extraction_method, "run": e.run_id}
                for e in supporting],
            "audio_transcript_or_inferred": sorted(
                {e.extraction_method for e in supporting}),
            "supporting_sources": len({e.source_id for e in supporting}),
            "disagreeing_sources": len({e.source_id for e in against}),
            "conflicts": [
                {"conflict_id": c.conflict_id, "with": (
                    c.claim_b if c.claim_a == knowledge_id else c.claim_a),
                 "status": c.resolution_status, "notes": c.notes}
                for c in self._conflicts_touching(knowledge_id)],
            "versions": len(self.versions_of(knowledge_id)),
            "corrections": [
                dict(r) for r in self.store.query(
                    "SELECT * FROM user_corrections WHERE knowledge_id=? "
                    "ORDER BY created_at", (knowledge_id,))],
            "audit": self.store.audit_trail(knowledge_id=knowledge_id),
        }

    def _conflicts_touching(self, knowledge_id: str) -> List[Conflict]:
        rows = self.store.query(
            "SELECT * FROM conflicts WHERE claim_a=? OR claim_b=?",
            (knowledge_id, knowledge_id))
        return [self._row_to_conflict(r) for r in rows]

    # ==================================================================
    # relationships and the graph (section 7)
    # ==================================================================
    def add_relationship(self, source_id: str, relation_type: str,
                         target_id: str, *, confidence: float = 0.7,
                         evidence: str = "") -> Optional[Relationship]:
        if source_id == target_id:
            return None
        relationship = Relationship(
            source_knowledge_id=source_id, relation_type=relation_type,
            target_knowledge_id=target_id, confidence=confidence,
            evidence=evidence)
        try:
            with self.store.transaction():
                self._insert_relationship(relationship)
        except Exception as exc:  # noqa: BLE001 - a duplicate edge is fine
            log.debug("relationship not added: %s", exc)
            return None
        return relationship

    def graph_neighbors(self, knowledge_id: str, *,
                        relation_types: Sequence[str] = (),
                        depth: int = 1, limit: int = 100
                        ) -> List[Tuple[Relationship, KnowledgeItem]]:
        """Walk outward, following edges in both directions.

        Symmetric relations have no direction worth respecting, and for the
        rest a caller asking "what is connected to Kambhoji" wants the things
        that point *at* it as much as the things it points to.
        """
        seen = {knowledge_id}
        frontier = [knowledge_id]
        out: List[Tuple[Relationship, KnowledgeItem]] = []
        for _ in range(max(1, depth)):
            if not frontier or len(out) >= limit:
                break
            marks = ",".join("?" for _ in frontier)
            params: List[Any] = list(frontier) + list(frontier)
            clause = ""
            if relation_types:
                clause = ("AND relation_type IN ("
                          + ",".join("?" for _ in relation_types) + ")")
                params += list(relation_types)
            rows = self.store.query(
                f"SELECT * FROM relationships WHERE (source_knowledge_id IN "
                f"({marks}) OR target_knowledge_id IN ({marks})) {clause} "
                f"AND status <> 'rejected' LIMIT ?", params + [limit])
            next_frontier: List[str] = []
            for row in rows:
                relationship = self._row_to_relationship(row)
                other = (relationship.target_knowledge_id
                         if relationship.source_knowledge_id in seen
                         else relationship.source_knowledge_id)
                if other in seen:
                    continue
                item = self.get_by_id(other)
                if item is None:
                    continue
                seen.add(other)
                next_frontier.append(other)
                out.append((relationship, item))
                if len(out) >= limit:
                    break
            frontier = next_frontier
        return out

    # ==================================================================
    # sources and evidence (section 9)
    # ==================================================================
    def add_source(self, source: Source) -> Source:
        identity = normalize.source_identity(source.reference, source.title)
        existing = self.store.one(
            "SELECT * FROM sources WHERE identity=?", (identity,))
        if existing is not None:
            return self._row_to_source(existing)
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO sources(source_id, source_type, title, "
                "author_or_channel, reference, published_date, acquired_date, "
                "license_or_access_notes, language, checksum, metadata, "
                "training_source_id, identity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (source.source_id, source.source_type, source.title,
                 source.author_or_channel, source.reference,
                 source.published_date, source.acquired_date,
                 source.license_or_access_notes, source.language,
                 source.checksum, dumps(source.metadata),
                 source.training_source_id, identity))
        self.store.audit("source.added", source.title[:200],
                         source_id=source.source_id)
        return source

    def get_source(self, source_id: str) -> Optional[Source]:
        row = self.store.one("SELECT * FROM sources WHERE source_id=?",
                             (source_id,))
        return self._row_to_source(row) if row else None

    def add_evidence(self, evidence: Evidence) -> Evidence:
        with self.store.transaction():
            self._insert_evidence(evidence)
        self.recompute_confidence(evidence.knowledge_id)
        return evidence

    def add_conflict(self, conflict: Conflict) -> Conflict:
        with self.store.transaction():
            self._insert_conflict(conflict)
        return conflict

    def resolve_conflict(self, conflict_id: str, resolution: str,
                         reviewer: str = "user", notes: str = "") -> bool:
        """Section 12.  Resolving is a decision, and it is recorded as one."""
        if resolution not in ConflictState.ALL:
            raise ValueError(f"'{resolution}' is not a resolution state")
        row = self.store.one("SELECT * FROM conflicts WHERE conflict_id=?",
                             (conflict_id,))
        if row is None:
            return False
        conflict = self._row_to_conflict(row)
        with self.store.transaction():
            self.store.execute(
                "UPDATE conflicts SET resolution_status=?, reviewer=?, "
                "notes=?, resolved_at=? WHERE conflict_id=?",
                (resolution, reviewer,
                 (conflict.notes + " | " + notes).strip(" |"),
                 time.time(), conflict_id))
            if resolution == ConflictState.RESOLVED_A:
                self._settle(conflict.claim_a, conflict.claim_b)
            elif resolution == ConflictState.RESOLVED_B:
                self._settle(conflict.claim_b, conflict.claim_a)
            elif resolution == ConflictState.REJECTED_BOTH:
                for claim in (conflict.claim_a, conflict.claim_b):
                    self.store.execute(
                        "UPDATE knowledge_items SET status=?, updated_at=? "
                        "WHERE knowledge_id=?",
                        (Status.REJECTED, time.time(), claim))
            elif resolution in (ConflictState.BOTH_VALID,
                                ConflictState.CONTEXT_DEPENDENT):
                for claim in (conflict.claim_a, conflict.claim_b):
                    self.store.execute(
                        "UPDATE knowledge_items SET status=?, updated_at=? "
                        "WHERE knowledge_id=?",
                        (Status.ACCEPTED, time.time(), claim))
        self.store.audit("conflict.resolved", f"{conflict_id}: {resolution}",
                         actor=reviewer)
        return True

    def _settle(self, winner: str, loser: str) -> None:
        """The loser is superseded, never deleted - section 13."""
        self.store.execute(
            "UPDATE knowledge_items SET status=?, updated_at=? "
            "WHERE knowledge_id=?", (Status.ACCEPTED, time.time(), winner))
        self.store.execute(
            "UPDATE knowledge_items SET status=?, updated_at=? "
            "WHERE knowledge_id=?", (Status.SUPERSEDED, time.time(), loser))
        self._insert_relationship(Relationship(
            source_knowledge_id=winner, relation_type=Relation.SUPERSEDES,
            target_knowledge_id=loser, evidence="a person resolved a conflict"))

    # ==================================================================
    # correction (sections 13, 28)
    # ==================================================================
    def create_version(self, knowledge_id: str, reason: str,
                       run_id: str = "", source_id: str = "") -> Optional[Version]:
        item = self.get_by_id(knowledge_id)
        if item is None:
            return None
        with self.store.transaction():
            return self._insert_version(item, reason, run_id=run_id,
                                        source_id=source_id)

    def correct_knowledge(self, knowledge_id: str, *, action: str,
                          explanation: str = "",
                          new_statement: Optional[str] = None,
                          new_confidence: Optional[float] = None
                          ) -> Optional[KnowledgeItem]:
        """Section 28.  A person's verdict, kept rather than applied and lost."""
        item = self.get_by_id(knowledge_id)
        if item is None:
            return None

        correction = UserCorrection(
            knowledge_id=knowledge_id, action=action, explanation=explanation,
            previous_value=item.statement,
            new_value=new_statement if new_statement is not None else "",
            previous_confidence=item.confidence,
            new_confidence=new_confidence if new_confidence is not None
            else item.confidence)

        status = item.status
        confidence = item.confidence
        review_state = item.review_state
        if action == "mark_incorrect":
            status, review_state = Status.REJECTED, "rejected"
            confidence = min(confidence, 0.1)
        elif action == "mark_correct":
            status, review_state = Status.ACCEPTED, "confirmed"
            confidence = max(confidence, 0.9)
        elif action == "reduce_confidence":
            confidence = max(0.0, confidence - 0.2)
        elif action == "increase_confidence":
            confidence = min(1.0, confidence + 0.2)
        elif action == "needs_review":
            status, review_state = Status.NEEDS_REVIEW, "flagged"
        if new_confidence is not None:
            confidence = max(0.0, min(1.0, float(new_confidence)))

        with self.store.transaction():
            self._insert_version(item, f"user correction: {action}. "
                                       f"{explanation}".strip())
            fields = ["status=?", "confidence=?", "review_state=?",
                      "version=version+1", "updated_at=?"]
            params: List[Any] = [status, confidence, review_state, time.time()]
            if new_statement is not None:
                fields.insert(0, "statement=?")
                params.insert(0, new_statement)
            params.append(knowledge_id)
            self.store.execute(
                f"UPDATE knowledge_items SET {', '.join(fields)} "
                f"WHERE knowledge_id=?", params)
            self.store.execute(
                "INSERT INTO user_corrections(correction_id, knowledge_id, "
                "action, explanation, previous_value, new_value, "
                "previous_confidence, new_confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (correction.correction_id, knowledge_id, action, explanation,
                 correction.previous_value, correction.new_value,
                 correction.previous_confidence, confidence,
                 correction.created_at))
        self.store.audit(f"knowledge.{action}", explanation[:200],
                         knowledge_id=knowledge_id, actor="user")
        return self.get_by_id(knowledge_id)

    def mark_rejected(self, knowledge_id: str, reason: str = ""
                      ) -> Optional[KnowledgeItem]:
        return self.correct_knowledge(knowledge_id, action="mark_incorrect",
                                      explanation=reason)

    def recompute_confidence(self, knowledge_id: str,
                             source_quality: float = 0.6
                             ) -> Optional[KnowledgeItem]:
        item = self.get_by_id(knowledge_id)
        if item is None:
            return None
        evidence = self.evidence_for(knowledge_id)
        scored = confidence_model.score(
            evidence, source_quality=source_quality,
            human_confirmed=item.review_state == "confirmed",
            human_rejected=item.review_state == "rejected")
        sources = len({e.source_id for e in evidence
                       if e.supports and e.source_id})
        with self.store.transaction():
            self.store.execute(
                "UPDATE knowledge_items SET confidence=?, confidence_parts=?, "
                "source_count=?, updated_at=?, last_verified_at=? "
                "WHERE knowledge_id=?",
                (scored.value, dumps(scored.parts), sources, time.time(),
                 time.time(), knowledge_id))
        return self.get_by_id(knowledge_id)

    # ==================================================================
    # gaps, lessons, details
    # ==================================================================
    def add_gap(self, gap: KnowledgeGap) -> KnowledgeGap:
        identity = normalize.identity_of(gap.subject, gap.missing_information)
        try:
            with self.store.transaction():
                self.store.execute(
                    "INSERT INTO knowledge_gaps(gap_id, subject, "
                    "missing_information, importance, reason, "
                    "suggested_search, curriculum_dependency, status, "
                    "created_at, identity) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (gap.gap_id, gap.subject, gap.missing_information,
                     gap.importance, gap.reason, gap.suggested_search,
                     gap.curriculum_dependency, gap.status, gap.created_at,
                     identity))
        except Exception:  # noqa: BLE001 - already recorded
            row = self.store.one(
                "SELECT * FROM knowledge_gaps WHERE identity=?", (identity,))
            if row is not None:
                return self._row_to_gap(row)
        return gap

    def gaps(self, *, open_only: bool = True, limit: int = 200
             ) -> List[KnowledgeGap]:
        clause = "WHERE status='open'" if open_only else ""
        rows = self.store.query(
            f"SELECT * FROM knowledge_gaps {clause} "
            f"ORDER BY importance DESC LIMIT ?", (limit,))
        return [self._row_to_gap(r) for r in rows]

    def close_gap(self, gap_id: str) -> None:
        with self.store.transaction():
            self.store.execute(
                "UPDATE knowledge_gaps SET status='closed', closed_at=? "
                "WHERE gap_id=?", (time.time(), gap_id))

    def add_failure_lesson(self, lesson: FailureLesson) -> FailureLesson:
        """Section 25.  A mistake, remembered so it is not made twice."""
        with self.store.transaction():
            self.store.execute(
                "INSERT INTO failure_lessons(lesson_id, task, "
                "attempted_method, result, failure_reason, correction, "
                "related_knowledge, source_or_run, confidence, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (lesson.lesson_id, lesson.task, lesson.attempted_method,
                 lesson.result, lesson.failure_reason, lesson.correction,
                 dumps(lesson.related_knowledge), lesson.source_or_run,
                 lesson.confidence, lesson.created_at))
        self.store.audit("lesson.recorded", lesson.failure_reason[:200])
        return lesson

    def failure_lessons(self, task: str = "", limit: int = 100
                        ) -> List[FailureLesson]:
        if task:
            rows = self.store.query(
                "SELECT * FROM failure_lessons WHERE task LIKE ? "
                "ORDER BY created_at DESC LIMIT ?", (f"%{task}%", limit))
        else:
            rows = self.store.query(
                "SELECT * FROM failure_lessons ORDER BY created_at DESC "
                "LIMIT ?", (limit,))
        return [FailureLesson(
            lesson_id=r["lesson_id"], task=r["task"] or "",
            attempted_method=r["attempted_method"] or "",
            result=r["result"] or "", failure_reason=r["failure_reason"] or "",
            correction=r["correction"] or "",
            related_knowledge=loads(r["related_knowledge"], []),
            source_or_run=r["source_or_run"] or "",
            confidence=float(r["confidence"] or 0.0),
            created_at=float(r["created_at"] or 0.0)) for r in rows]

    def set_procedure_detail(self, detail: ProcedureDetail) -> None:
        with self.store.transaction():
            self.store.execute(
                "INSERT OR REPLACE INTO procedure_details(knowledge_id, goal, "
                "prerequisites, inputs, steps, optional_branches, constraints,"
                " failure_modes, evaluation_criteria) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (detail.knowledge_id, detail.goal, dumps(detail.prerequisites),
                 dumps(detail.inputs), dumps(detail.steps),
                 dumps(detail.optional_branches), dumps(detail.constraints),
                 dumps(detail.failure_modes),
                 dumps(detail.evaluation_criteria)))

    def set_example_detail(self, detail: ExampleDetail) -> None:
        with self.store.transaction():
            self.store.execute(
                "INSERT OR REPLACE INTO example_details(knowledge_id, "
                "concept_demonstrated, notation, swaras, features, source_id, "
                "timestamp_start, timestamp_end, quality, curriculum_stage) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (detail.knowledge_id, detail.concept_demonstrated,
                 detail.notation, dumps(detail.swaras), dumps(detail.features),
                 detail.source_id, detail.timestamp_start,
                 detail.timestamp_end, detail.quality, detail.curriculum_stage))

    def example_detail(self, knowledge_id: str) -> Optional[ExampleDetail]:
        row = self.store.one(
            "SELECT * FROM example_details WHERE knowledge_id=?",
            (knowledge_id,))
        if row is None:
            return None
        return ExampleDetail(
            knowledge_id=row["knowledge_id"],
            concept_demonstrated=row["concept_demonstrated"] or "",
            notation=row["notation"] or "", swaras=loads(row["swaras"], []),
            features=loads(row["features"], {}),
            source_id=row["source_id"] or "",
            timestamp_start=float(row["timestamp_start"] or 0.0),
            timestamp_end=float(row["timestamp_end"] or 0.0),
            quality=float(row["quality"] or 0.0),
            curriculum_stage=row["curriculum_stage"] or "")

    # ==================================================================
    # aliases and tags (section 8)
    # ==================================================================
    def add_alias(self, knowledge_id: str, alias: str,
                  kind: str = "alias") -> None:
        normalised = normalize.normalise_name(alias)
        if not normalised:
            return
        try:
            with self.store.transaction():
                self.store.execute(
                    "INSERT OR IGNORE INTO aliases(knowledge_id, alias, "
                    "normalised, kind) VALUES (?,?,?,?)",
                    (knowledge_id, alias, normalised, kind))
        except Exception as exc:  # noqa: BLE001
            log.debug("alias not added: %s", exc)

    def add_tags(self, knowledge_id: str, tags: Iterable[str]) -> None:
        with self.store.transaction():
            for tag in tags:
                tag = tag.strip()
                if not tag:
                    continue
                self.store.execute(
                    "INSERT OR IGNORE INTO tags(tag, created_at) VALUES (?,?)",
                    (tag, time.time()))
                self.store.execute(
                    "INSERT OR IGNORE INTO knowledge_tags(knowledge_id, tag) "
                    "VALUES (?,?)", (knowledge_id, tag))

    # ==================================================================
    # low-level writes, all inside a caller's transaction
    # ==================================================================
    def _insert(self, item: KnowledgeItem, identity: str = "",
                in_transaction: bool = False) -> KnowledgeItem:
        """Write an item with its aliases and tags.

        When the caller has not already opened a transaction, the whole lot -
        row, aliases, tags - goes in one, so a half-inserted item with no way
        to find it by name cannot exist.
        """
        if not in_transaction:
            with self.store.transaction():
                return self._insert(item, identity, in_transaction=True)
        identity = identity or normalize.identity_of(
            item.subject, item.predicate, item.raga, item.tala,
            value=item.object_value)
        sql = (
            "INSERT INTO knowledge_items(knowledge_id, canonical_name, "
            "knowledge_type, subject, predicate, object_value, statement, "
            "structured_value, scope, status, confidence, confidence_parts, "
            "importance, source_count, created_at, updated_at, "
            "last_verified_at, version, tags, valid_from, valid_until, "
            "language, difficulty, curriculum_level, owner_or_creator, "
            "review_state, usage_count, last_used_at, learned_by, notes, "
            "raga, tala, identity) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
        params = (
            item.knowledge_id, item.canonical_name, item.knowledge_type,
            item.subject, item.predicate, item.object_value, item.statement,
            dumps(item.structured_value), dumps(item.scope), item.status,
            item.confidence, dumps(item.confidence_parts), item.importance,
            item.source_count, item.created_at, item.updated_at,
            item.last_verified_at, item.version, dumps(item.tags),
            item.valid_from, item.valid_until, item.language, item.difficulty,
            item.curriculum_level, item.owner_or_creator, item.review_state,
            item.usage_count, item.last_used_at, item.learned_by, item.notes,
            item.raga, item.tala, identity)
        self.store.execute(sql, params)
        for name in {item.canonical_name, item.subject}:
            if name:
                for variant in normalize.alias_variants(name):
                    try:
                        self.store.execute(
                            "INSERT OR IGNORE INTO aliases(knowledge_id, "
                            "alias, normalised, kind) VALUES (?,?,?,?)",
                            (item.knowledge_id, variant,
                             normalize.normalise_name(variant), "spelling"))
                    except Exception:  # noqa: BLE001
                        pass
        for tag in item.tags:
            try:
                self.store.execute(
                    "INSERT OR IGNORE INTO tags(tag, created_at) VALUES (?,?)",
                    (tag, time.time()))
                self.store.execute(
                    "INSERT OR IGNORE INTO knowledge_tags(knowledge_id, tag) "
                    "VALUES (?,?)", (item.knowledge_id, tag))
            except Exception:  # noqa: BLE001
                pass
        return item

    def _insert_evidence(self, record: Evidence) -> Evidence:
        self.store.execute(
            "INSERT OR REPLACE INTO evidence(evidence_id, source_id, "
            "knowledge_id, source_segment, timestamp_start, timestamp_end, "
            "transcript_excerpt, feature_reference, strength, "
            "extraction_method, run_id, created_at, supports) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (record.evidence_id, record.source_id, record.knowledge_id,
             record.source_segment, record.timestamp_start,
             record.timestamp_end, record.transcript_excerpt,
             record.feature_reference, record.strength,
             record.extraction_method, record.run_id, record.created_at,
             int(record.supports)))
        return record

    def _insert_relationship(self, relationship: Relationship) -> Relationship:
        self.store.execute(
            "INSERT OR IGNORE INTO relationships(relationship_id, "
            "source_knowledge_id, relation_type, target_knowledge_id, "
            "confidence, evidence, created_at, status) VALUES (?,?,?,?,?,?,?,?)",
            (relationship.relationship_id, relationship.source_knowledge_id,
             relationship.relation_type, relationship.target_knowledge_id,
             relationship.confidence, relationship.evidence,
             relationship.created_at, relationship.status))
        return relationship

    def _insert_conflict(self, conflict: Conflict) -> Conflict:
        self.store.execute(
            "INSERT OR REPLACE INTO conflicts(conflict_id, claim_a, claim_b, "
            "source_a, source_b, confidence_a, confidence_b, "
            "resolution_status, reviewer, notes, created_at, resolved_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (conflict.conflict_id, conflict.claim_a, conflict.claim_b,
             conflict.source_a, conflict.source_b, conflict.confidence_a,
             conflict.confidence_b, conflict.resolution_status,
             conflict.reviewer, conflict.notes, conflict.created_at,
             conflict.resolved_at))
        return conflict

    def _insert_version(self, item: KnowledgeItem, reason: str,
                        run_id: str = "", source_id: str = "") -> Version:
        version = Version(
            knowledge_id=item.knowledge_id, version=item.version,
            snapshot={
                "statement": item.statement, "status": item.status,
                "confidence": item.confidence,
                "structured_value": item.structured_value,
                "object_value": item.object_value},
            reason=reason, caused_by_run_id=run_id,
            caused_by_source_id=source_id)
        self.store.execute(
            "INSERT INTO knowledge_versions(version_id, knowledge_id, "
            "version, snapshot, changed_at, reason, caused_by_source_id, "
            "caused_by_run_id, changed_by) VALUES (?,?,?,?,?,?,?,?,?)",
            (version.version_id, version.knowledge_id, version.version,
             dumps(version.snapshot), version.changed_at, version.reason,
             version.caused_by_source_id, version.caused_by_run_id,
             version.changed_by))
        return version

    # ==================================================================
    # helpers
    # ==================================================================
    def _by_identity(self, identity: str) -> Optional[KnowledgeItem]:
        row = self.store.one(
            "SELECT * FROM knowledge_items WHERE identity=? AND status "
            "NOT IN ('rejected','superseded') ORDER BY confidence DESC LIMIT 1",
            (identity,))
        return self._row_to_item(row) if row else None

    @staticmethod
    def _says_the_same(existing: KnowledgeItem, incoming: KnowledgeItem) -> bool:
        """Same claim, allowing for wording and for how a scale is written."""
        if existing.structured_value and incoming.structured_value:
            a, b = existing.structured_value, incoming.structured_value
            if a.get("kind") == b.get("kind") == "swaras":
                return a.get("swaras") == b.get("swaras")
            if a.get("kind") == b.get("kind") == "number":
                return abs(float(a.get("value", 0)) -
                           float(b.get("value", 0))) < 1e-6
        if existing.object_value and incoming.object_value:
            return (normalize.normalise_statement(existing.object_value) ==
                    normalize.normalise_statement(incoming.object_value))
        return (normalize.normalise_statement(existing.statement) ==
                normalize.normalise_statement(incoming.statement))

    @staticmethod
    def _structured_disagree(existing: KnowledgeItem,
                             incoming: KnowledgeItem) -> bool:
        """Do the machine-readable values actually say different things?

        Only a value we can compare counts.  Two claims whose structured form
        is free text are not in disagreement merely for being different text -
        that is what the wording similarity is for.
        """
        a = existing.structured_value or {}
        b = incoming.structured_value or {}
        kind = a.get("kind")
        if kind != b.get("kind"):
            return True
        if kind == "swaras":
            return a.get("swaras") != b.get("swaras")
        if kind == "number":
            return abs(float(a.get("value", 0)) - float(b.get("value", 0))) > 1e-6
        if kind == "list":
            return sorted(a.get("items", [])) != sorted(b.get("items", []))
        return False

    def _first_source(self, knowledge_id: str) -> str:
        row = self.store.one(
            "SELECT source_id FROM evidence WHERE knowledge_id=? "
            "ORDER BY created_at LIMIT 1", (knowledge_id,))
        return row["source_id"] if row else ""

    # -- row mapping ----------------------------------------------------
    @staticmethod
    def _row_to_item(row) -> KnowledgeItem:
        return KnowledgeItem(
            knowledge_id=row["knowledge_id"],
            canonical_name=row["canonical_name"] or "",
            knowledge_type=row["knowledge_type"] or "",
            subject=row["subject"] or "", predicate=row["predicate"] or "",
            object_value=row["object_value"] or "",
            statement=row["statement"] or "",
            structured_value=loads(row["structured_value"], {}),
            scope=loads(row["scope"], []), status=row["status"] or "",
            confidence=float(row["confidence"] or 0.0),
            confidence_parts=loads(row["confidence_parts"], {}),
            importance=float(row["importance"] or 0.0),
            source_count=int(row["source_count"] or 0),
            created_at=float(row["created_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
            last_verified_at=float(row["last_verified_at"] or 0.0),
            version=int(row["version"] or 1), tags=loads(row["tags"], []),
            valid_from=row["valid_from"] or "",
            valid_until=row["valid_until"] or "",
            language=row["language"] or "", difficulty=row["difficulty"] or "",
            curriculum_level=row["curriculum_level"] or "",
            owner_or_creator=row["owner_or_creator"] or "",
            review_state=row["review_state"] or "",
            usage_count=int(row["usage_count"] or 0),
            last_used_at=float(row["last_used_at"] or 0.0),
            learned_by=row["learned_by"] or "", notes=row["notes"] or "",
            raga=row["raga"] or "", tala=row["tala"] or "")

    @staticmethod
    def _row_to_evidence(row) -> Evidence:
        return Evidence(
            evidence_id=row["evidence_id"], source_id=row["source_id"] or "",
            knowledge_id=row["knowledge_id"] or "",
            source_segment=row["source_segment"] or "",
            timestamp_start=float(row["timestamp_start"] or 0.0),
            timestamp_end=float(row["timestamp_end"] or 0.0),
            transcript_excerpt=row["transcript_excerpt"] or "",
            feature_reference=row["feature_reference"] or "",
            strength=float(row["strength"] or 0.0),
            extraction_method=row["extraction_method"] or "",
            run_id=row["run_id"] or "",
            created_at=float(row["created_at"] or 0.0),
            supports=bool(row["supports"]))

    @staticmethod
    def _row_to_source(row) -> Source:
        return Source(
            source_id=row["source_id"], source_type=row["source_type"] or "",
            title=row["title"] or "",
            author_or_channel=row["author_or_channel"] or "",
            reference=row["reference"] or "",
            published_date=row["published_date"] or "",
            acquired_date=float(row["acquired_date"] or 0.0),
            license_or_access_notes=row["license_or_access_notes"] or "",
            language=row["language"] or "", checksum=row["checksum"] or "",
            metadata=loads(row["metadata"], {}),
            training_source_id=row["training_source_id"] or "")

    @staticmethod
    def _row_to_relationship(row) -> Relationship:
        return Relationship(
            relationship_id=row["relationship_id"],
            source_knowledge_id=row["source_knowledge_id"],
            relation_type=row["relation_type"],
            target_knowledge_id=row["target_knowledge_id"],
            confidence=float(row["confidence"] or 0.0),
            evidence=row["evidence"] or "",
            created_at=float(row["created_at"] or 0.0),
            status=row["status"] or "")

    @staticmethod
    def _row_to_conflict(row) -> Conflict:
        return Conflict(
            conflict_id=row["conflict_id"], claim_a=row["claim_a"],
            claim_b=row["claim_b"], source_a=row["source_a"] or "",
            source_b=row["source_b"] or "",
            confidence_a=float(row["confidence_a"] or 0.0),
            confidence_b=float(row["confidence_b"] or 0.0),
            resolution_status=row["resolution_status"] or "",
            reviewer=row["reviewer"] or "", notes=row["notes"] or "",
            created_at=float(row["created_at"] or 0.0),
            resolved_at=float(row["resolved_at"] or 0.0))

    @staticmethod
    def _row_to_gap(row) -> KnowledgeGap:
        return KnowledgeGap(
            gap_id=row["gap_id"], subject=row["subject"] or "",
            missing_information=row["missing_information"] or "",
            importance=float(row["importance"] or 0.0),
            reason=row["reason"] or "",
            suggested_search=row["suggested_search"] or "",
            curriculum_dependency=row["curriculum_dependency"] or "",
            status=row["status"] or "open",
            created_at=float(row["created_at"] or 0.0),
            closed_at=float(row["closed_at"] or 0.0))
