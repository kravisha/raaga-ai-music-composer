"""Keeping the Knowledge Base in order - specification sections 30, 39, 40.

The Librarian is ordinary service logic, deliberately: section 39 says not to
block a first working version on a sophisticated autonomous agent, and none of
what it does needs one.  It looks over the store and reports what a careful
person would notice - the same fact recorded twice under different wordings, a
claim nothing is attached to, a subject with a hole in it, a rule that ten
sources support and one canonical item could carry.

Everything here *reports*.  The only thing it changes on its own is
compaction, which merges duplicates that are provably the same claim, and even
that keeps every evidence record and refuses to touch a disagreement: section
16 is explicit that compaction must never erase meaningful disagreement.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from . import normalize
from .models import (ConflictState, HealthReport, KnowledgeGap, KnowledgeItem,
                     KnowledgeType, Relation, Status)
from .service import KnowledgeBaseService

log = get_logger("kb.librarian")

#: The facts a raga needs before the system can be said to know it.
RAGA_ESSENTIALS = ("arohanam", "avarohanam", "prayoga", "gamaka", "jeeva")
#: ... and a tala.
TALA_ESSENTIALS = ("anga", "beat_count", "nadai")

NEAR_DUPLICATE = 0.85


@dataclass
class DuplicateCandidate:
    keep: KnowledgeItem
    merge: KnowledgeItem
    similarity: float

    def describe(self) -> str:
        return (f"\"{self.merge.display()}\" looks like "
                f"\"{self.keep.display()}\" ({self.similarity:.0%})")


class Librarian:
    """Section 39.  Notices things; changes almost nothing."""

    def __init__(self, service: KnowledgeBaseService) -> None:
        self.service = service
        self.store = service.store

    # ==================================================================
    # noticing
    # ==================================================================
    def duplicate_candidates(self, limit: int = 100
                             ) -> List[DuplicateCandidate]:
        """Near duplicates, offered rather than merged - section 15.

        Exact duplicates never get this far: they are caught on the way in by
        the identity key.  What is left is the harder case - two rows that say
        the same thing in different words - and that is a judgement, so it is
        reported.
        """
        rows = self.store.query(
            "SELECT * FROM knowledge_items WHERE status NOT IN "
            "('rejected','superseded') ORDER BY subject, predicate LIMIT ?",
            (limit * 8,))
        items = [self.service._row_to_item(r) for r in rows]
        by_subject: Dict[Tuple[str, str], List[KnowledgeItem]] = {}
        for item in items:
            key = (normalize.normalise_name(item.subject),
                   normalize.normalise_predicate(item.predicate))
            by_subject.setdefault(key, []).append(item)

        out: List[DuplicateCandidate] = []
        for group in by_subject.values():
            if len(group) < 2:
                continue
            for index, first in enumerate(group):
                for second in group[index + 1:]:
                    # A recorded disagreement is not a duplicate; it is the
                    # point.  Leave it alone.
                    if self._in_conflict(first.knowledge_id,
                                         second.knowledge_id):
                        continue
                    score = normalize.similarity(first.display(),
                                                 second.display())
                    if score >= NEAR_DUPLICATE:
                        keep, merge = ((first, second)
                                       if first.confidence >= second.confidence
                                       else (second, first))
                        out.append(DuplicateCandidate(keep, merge, score))
                    if len(out) >= limit:
                        return out
        return out

    def orphans(self, limit: int = 100) -> List[KnowledgeItem]:
        """Knowledge connected to nothing - section 39.

        Not wrong, but a sign that linking did not happen, and unreachable by
        graph traversal, which is how most retrieval finds anything.
        """
        rows = self.store.query(
            "SELECT k.* FROM knowledge_items k WHERE k.status NOT IN "
            "('rejected','superseded') AND NOT EXISTS (SELECT 1 FROM "
            "relationships r WHERE r.source_knowledge_id = k.knowledge_id "
            "OR r.target_knowledge_id = k.knowledge_id) LIMIT ?", (limit,))
        return [self.service._row_to_item(r) for r in rows]

    def weakly_supported(self, threshold: float = 0.35,
                         limit: int = 100) -> List[KnowledgeItem]:
        rows = self.store.query(
            "SELECT * FROM knowledge_items WHERE confidence < ? AND status "
            "NOT IN ('rejected','superseded','deprecated') "
            "ORDER BY confidence LIMIT ?", (threshold, limit))
        return [self.service._row_to_item(r) for r in rows]

    #: Nodes that are structure rather than claims.  An entity says only that
    #: a name denotes a thing of a kind - it is identity, not something a
    #: source taught - so having no evidence is correct for it rather than a
    #: loss of provenance.
    STRUCTURAL = ("core taxonomy", "entity resolution")

    def unsupported(self, limit: int = 100,
                    subjects: Sequence[str] = ()) -> List[KnowledgeItem]:
        """Learned claims with no evidence at all - section 9's line, checked.

        Structural nodes are excluded: the taxonomy and the entities claims
        hang off were never claimed by anybody, which is a different thing
        from a learned claim that has lost its provenance.
        """
        marks = ",".join("?" for _ in self.STRUCTURAL)
        rows = self.store.query(
            f"SELECT k.* FROM knowledge_items k WHERE k.learned_by NOT IN "
            f"({marks}) AND k.knowledge_type <> 'entity' AND NOT EXISTS ("
            f"SELECT 1 FROM evidence e WHERE e.knowledge_id = k.knowledge_id) "
            f"LIMIT ?", list(self.STRUCTURAL) + [limit])
        items = [self.service._row_to_item(r) for r in rows]
        if not subjects:
            return items
        wanted = {normalize.normalise_name(s) for s in subjects}
        return [i for i in items
                if normalize.normalise_name(i.subject) in wanted
                or normalize.normalise_name(i.raga) in wanted]

    def stale(self, older_than_days: float = 180.0,
              limit: int = 100) -> List[KnowledgeItem]:
        cutoff = time.time() - older_than_days * 86400
        rows = self.store.query(
            "SELECT * FROM knowledge_items WHERE last_verified_at > 0 AND "
            "last_verified_at < ? AND status IN ('learned','validated',"
            "'accepted') ORDER BY last_verified_at LIMIT ?", (cutoff, limit))
        return [self.service._row_to_item(r) for r in rows]

    # ==================================================================
    # gaps (section 30)
    # ==================================================================
    def detect_gaps(self, subjects: Sequence[str] = (),
                    record: bool = True) -> List[KnowledgeGap]:
        """What the Knowledge Base does not know, said out loud.

        A gap is only worth recording where the subject is one we have reason
        to care about - a raga the system has begun studying, or one a caller
        named.  Enumerating every fact absent about every raga in existence
        would be true and useless.
        """
        found: List[KnowledgeGap] = []
        targets = list(subjects) or self._subjects_in_play()

        for subject in targets:
            held = {normalize.normalise_predicate(i.predicate)
                    for i in self.service.items_about(subject)}
            if not held:
                continue                # nothing at all: not a gap, a blank
            for essential in RAGA_ESSENTIALS:
                if essential in held:
                    continue
                gap = KnowledgeGap(
                    subject=subject,
                    missing_information=f"no {essential} is recorded",
                    importance=0.8 if essential in ("arohanam", "avarohanam")
                    else 0.5,
                    reason="no source processed so far has supplied it",
                    suggested_search=f"{subject} {essential} lesson")
                found.append(self.service.add_gap(gap) if record else gap)

        # A claim nothing supports is a gap of a different kind.  Scoped to
        # the subjects asked about, so a question about one raga does not come
        # back with everything wrong anywhere.
        for item in self.unsupported(limit=20, subjects=targets):
            gap = KnowledgeGap(
                subject=item.subject or item.canonical_name,
                missing_information=f"\"{item.display()}\" has no evidence",
                importance=0.7,
                reason="it is held but nothing was recorded to support it",
                suggested_search=f"{item.subject} {item.predicate}")
            found.append(self.service.add_gap(gap) if record else gap)
        return found

    def _subjects_in_play(self) -> List[str]:
        rows = self.store.query(
            "SELECT raga, COUNT(*) AS n FROM knowledge_items WHERE raga <> '' "
            "GROUP BY raga ORDER BY n DESC LIMIT 20")
        return [r["raga"] for r in rows]

    # ==================================================================
    # linking (section 39)
    # ==================================================================
    def link_orphans(self, limit: int = 200) -> int:
        """Attach loose knowledge to the entity it is obviously about.

        Nothing clever: an item whose raga names an entity we hold gets a
        ``belongs_to_raga`` edge to it.  That is what makes graph traversal
        from "Kambhoji" reach the phrases and constraints filed under it.
        """
        linked = 0
        for item in self.orphans(limit=limit):
            target_name = item.raga or item.subject
            if not target_name:
                continue
            entity = self.service.find_entity(target_name)
            if entity is None or entity.knowledge_id == item.knowledge_id:
                continue
            relation = (Relation.BELONGS_TO_RAGA if item.raga
                        else Relation.RELATED_TO)
            if self.service.add_relationship(
                    item.knowledge_id, relation, entity.knowledge_id,
                    confidence=0.8,
                    evidence="linked by the librarian from the item's subject"):
                linked += 1
        if linked:
            self.store.audit("librarian.linked",
                             f"{linked} loose item(s) connected")
        return linked

    # ==================================================================
    # compaction (section 16)
    # ==================================================================
    def compact(self, dry_run: bool = True) -> Dict[str, Any]:
        """Merge provable duplicates, keeping every evidence record.

        Section 16's rule is the constraint that shapes this: compaction must
        never erase meaningful disagreement.  So anything with a conflict
        recorded against it is skipped outright, and a merge moves evidence
        rather than dropping it - ten sources teaching one rule end as one
        canonical rule with ten evidence records, which is exactly what the
        specification describes.
        """
        candidates = [c for c in self.duplicate_candidates(limit=200)
                      if c.similarity >= 0.95]
        report: Dict[str, Any] = {
            "examined": len(candidates), "merged": 0, "skipped": 0,
            "dry_run": dry_run, "details": []}

        for candidate in candidates:
            if self._in_conflict(candidate.keep.knowledge_id,
                                 candidate.merge.knowledge_id):
                report["skipped"] += 1
                continue
            report["details"].append(candidate.describe())
            if dry_run:
                continue
            self._merge(candidate.keep, candidate.merge)
            report["merged"] += 1

        if not dry_run and report["merged"]:
            self.store.audit("librarian.compacted",
                             f"{report['merged']} duplicate(s) merged")
        return report

    def _merge(self, keep: KnowledgeItem, merge: KnowledgeItem) -> None:
        """Move evidence and edges onto the survivor; retire the other."""
        with self.store.transaction():
            self.store.execute(
                "UPDATE OR IGNORE evidence SET knowledge_id=? "
                "WHERE knowledge_id=?", (keep.knowledge_id,
                                         merge.knowledge_id))
            for column in ("source_knowledge_id", "target_knowledge_id"):
                self.store.execute(
                    f"UPDATE OR IGNORE relationships SET {column}=? "
                    f"WHERE {column}=?",
                    (keep.knowledge_id, merge.knowledge_id))
            self.store.execute(
                "UPDATE knowledge_items SET status=?, notes=?, updated_at=? "
                "WHERE knowledge_id=?",
                (Status.SUPERSEDED,
                 f"merged into {keep.knowledge_id} by compaction",
                 time.time(), merge.knowledge_id))
            self.service._insert_relationship(_supersede(keep, merge))
        self.service.recompute_confidence(keep.knowledge_id)

    def _in_conflict(self, first: str, second: str) -> bool:
        row = self.store.one(
            "SELECT 1 FROM conflicts WHERE ((claim_a=? AND claim_b=?) OR "
            "(claim_a=? AND claim_b=?)) AND resolution_status IN "
            "('unresolved','context_dependent','both_valid_under_conditions')",
            (first, second, second, first))
        return row is not None

    # ==================================================================
    # the health report (section 40)
    # ==================================================================
    def health(self) -> HealthReport:
        report = self.service.health_check()
        report.duplicate_candidates = len(self.duplicate_candidates(limit=200))
        return report

    def render_health(self, report: Optional[HealthReport] = None) -> str:
        report = report or self.health()
        lines = ["KNOWLEDGE BASE HEALTH", "=" * 21, ""]
        lines.append(f"  Schema version        {report.schema_version}")
        lines.append(f"  Initialized           "
                     f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(report.initialized_at)) if report.initialized_at else 'unknown'}")
        lines.append(f"  Integrity             "
                     f"{'ok' if report.integrity_ok else 'FAILED'}")
        lines.append(f"  Last integrity check  "
                     f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(report.last_integrity_check)) if report.last_integrity_check else 'not run'}")
        lines.append(f"  Last backup           "
                     f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(report.last_backup)) if report.last_backup else 'never'}")
        lines.extend(["", f"  Knowledge items       {report.total_items}"])
        for status, count in sorted(report.by_status.items()):
            lines.append(f"      {status:<18} {count}")
        lines.append("")
        for kind, count in sorted(report.by_type.items()):
            lines.append(f"      {kind:<18} {count}")
        lines.extend([
            "",
            f"  Sources               {report.sources}",
            f"  Evidence records      {report.evidence}",
            f"  Relationships         {report.relationships}",
            "",
            f"  Orphan items          {report.orphan_items}",
            f"  Duplicate candidates  {report.duplicate_candidates}",
            f"  Unresolved conflicts  {report.unresolved_conflicts}",
            f"  Knowledge gaps        {report.knowledge_gaps}",
            f"  Weakly supported      {report.low_confidence_items}",
        ])
        return "\n".join(lines)


def _supersede(keep: KnowledgeItem, merge: KnowledgeItem):
    from .models import Relationship

    return Relationship(
        source_knowledge_id=keep.knowledge_id,
        relation_type=Relation.SUPERSEDES,
        target_knowledge_id=merge.knowledge_id,
        evidence="merged by compaction as the same claim")
