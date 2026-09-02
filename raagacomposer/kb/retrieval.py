"""Getting knowledge back out - specification section 18.

Retrieval is hybrid, and the specification is specific about why: the answer
to "what do I need to know to compose a Kambhoji alapana" is not the text
chunks whose words look most like the question.  It is the raga's structure,
its phrases, its gamaka rules, the constraints it must not break, examples
worth hearing, and the cautions attached to any of that.  Those are reached by
different routes, so several routes run and their results are merged.

The routes:

``exact``        an entity by canonical name or any alias, across spellings
``keyword``      full-text where the build has FTS5, substring where it does not
``tag``          curated labels
``graph``        outward from whatever the exact lookup found, which is how
                 the phrases and constraints attached to a raga arrive without
                 anybody having mentioned them
``structured``   type, raga, tala, difficulty, curriculum level
``filters``      source, confidence floor, status

One route is deliberately *not* implemented and not faked: semantic search
needs an embedding model, this application has none, and a lexical match
dressed up under that name would be a lie about how the answer was found.
:meth:`HybridRetriever.semantic_search` says so and falls back to keyword,
and :attr:`semantic_available` is what a caller should ask.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from ..core.logging_setup import get_logger
from . import normalize
from .models import KnowledgeItem, KnowledgeType, Relation, Status
from .service import KnowledgeBaseService

log = get_logger("kb.retrieval")


@dataclass
class Query:
    """What is being asked for - section 18's filters, all optional."""

    text: str = ""
    subject: str = ""
    raga: str = ""
    tala: str = ""
    knowledge_types: Sequence[str] = ()
    tags: Sequence[str] = ()
    scopes: Sequence[str] = ()
    source_id: str = ""
    min_confidence: float = 0.0
    curriculum_level: str = ""
    difficulty: str = ""
    statuses: Sequence[str] = Status.USABLE
    include_graph: bool = True
    graph_depth: int = 1
    limit: int = 40


@dataclass
class Hit:
    """One result, and how it was reached - which matters for explaining."""

    item: KnowledgeItem
    score: float = 0.0
    routes: Set[str] = field(default_factory=set)
    via: str = ""

    def reason(self) -> str:
        return f"{', '.join(sorted(self.routes))}{f' via {self.via}' if self.via else ''}"


class HybridRetriever:
    """Section 18.  Several routes, merged and ranked."""

    def __init__(self, service: KnowledgeBaseService) -> None:
        self.service = service
        self.store = service.store

    #: There is no embedding model in this application.  Saying so is the
    #: point: a caller can check rather than assume it got a semantic match.
    semantic_available = False

    # ==================================================================
    def search(self, query: Query) -> List[Hit]:
        hits: Dict[str, Hit] = {}

        def add(item: KnowledgeItem, weight: float, route: str,
                via: str = "") -> None:
            hit = hits.get(item.knowledge_id)
            if hit is None:
                hit = Hit(item=item, via=via)
                hits[item.knowledge_id] = hit
            hit.score += weight
            hit.routes.add(route)
            if via and not hit.via:
                hit.via = via

        # -- exact -------------------------------------------------------
        anchor: Optional[KnowledgeItem] = None
        for name in (query.subject, query.raga, query.text):
            if not name:
                continue
            found = self.service.find_entity(name)
            if found is None:
                found = self._exact_by_name(name)
            if found is not None:
                anchor = found
                add(found, 1.0, "exact")
                break

        # -- everything held about the subject ---------------------------
        for name in {query.subject, query.raga}:
            if not name:
                continue
            for item in self.service.items_about(name, limit=query.limit * 2):
                add(item, 0.75, "subject")

        # -- keyword ------------------------------------------------------
        if query.text:
            for item, rank in self._keyword(query.text, query.limit * 2):
                add(item, 0.55 * rank, "keyword")

        # -- tags ---------------------------------------------------------
        for tag in query.tags:
            for item in self._by_tag(tag, query.limit):
                add(item, 0.4, "tag")

        # -- structured ----------------------------------------------------
        for item in self._structured(query):
            add(item, 0.35, "structured")

        # -- graph ---------------------------------------------------------
        if query.include_graph and anchor is not None:
            for relationship, item in self.service.graph_neighbors(
                    anchor.knowledge_id, depth=query.graph_depth,
                    limit=query.limit * 2):
                # A constraint or a phrase reached from the raga is exactly
                # what a keyword search would have missed.
                weight = 0.6 if relationship.relation_type in (
                    Relation.BELONGS_TO_RAGA, Relation.PART_OF,
                    Relation.EXAMPLE_OF, Relation.DEMONSTRATES) else 0.45
                add(item, weight * max(0.2, relationship.confidence),
                    "graph", via=relationship.relation_type)

        # -- filters, applied once at the end ------------------------------
        results = [h for h in hits.values() if self._passes(h.item, query)]
        for hit in results:
            hit.score *= self._quality_multiplier(hit.item)
        results.sort(key=lambda h: (-h.score, -h.item.importance,
                                    -h.item.confidence))
        return results[:query.limit]

    # ==================================================================
    def semantic_search(self, text: str, limit: int = 20) -> List[Hit]:
        """Not implemented, and not pretended.

        Section 18 lists semantic search as one route of a hybrid retriever and
        section 32 makes embeddings an optional later addition.  Until there is
        an embedding model this falls back to the keyword route and says so, so
        that no caller believes it received a semantic match.
        """
        log.info("semantic search is unavailable - no embedding model is "
                 "configured; falling back to keyword matching")
        return self.search(Query(text=text, limit=limit, include_graph=False))

    # ==================================================================
    def _exact_by_name(self, name: str) -> Optional[KnowledgeItem]:
        normalised = normalize.normalise_name(name)
        row = self.store.one(
            "SELECT k.* FROM knowledge_items k JOIN aliases a "
            "ON a.knowledge_id = k.knowledge_id WHERE a.normalised=? "
            "ORDER BY k.importance DESC LIMIT 1", (normalised,))
        return self.service._row_to_item(row) if row else None

    def _keyword(self, text: str, limit: int) -> List[tuple]:
        """FTS where available, LIKE where not.  Same shape either way."""
        terms = [t for t in re.split(r"[^\w+]+", text) if len(t) > 1]
        if not terms:
            return []
        if self.store.fts_available:
            try:
                match = " OR ".join(f'"{t}"' for t in terms)
                rows = self.store.query(
                    "SELECT k.*, bm25(knowledge_fts) AS rank "
                    "FROM knowledge_fts f JOIN knowledge_items k "
                    "ON k.rowid = f.rowid WHERE knowledge_fts MATCH ? "
                    "ORDER BY rank LIMIT ?", (match, limit))
                out = []
                for position, row in enumerate(rows):
                    # bm25 is negative and unbounded; position is enough to
                    # rank within this route and keeps the scale comparable.
                    out.append((self.service._row_to_item(row),
                                1.0 / (1 + position * 0.25)))
                return out
            except sqlite3.Error as exc:
                log.debug("full-text query failed (%s); using substring", exc)

        clauses = " OR ".join(
            ["canonical_name LIKE ?", "subject LIKE ?", "statement LIKE ?",
             "object_value LIKE ?"])
        out = []
        seen: Set[str] = set()
        for term in terms[:4]:
            pattern = f"%{term}%"
            rows = self.store.query(
                f"SELECT * FROM knowledge_items WHERE {clauses} "
                f"ORDER BY confidence DESC LIMIT ?",
                (pattern, pattern, pattern, pattern, limit))
            for row in rows:
                if row["knowledge_id"] in seen:
                    continue
                seen.add(row["knowledge_id"])
                out.append((self.service._row_to_item(row), 0.8))
        return out[:limit]

    def _by_tag(self, tag: str, limit: int) -> List[KnowledgeItem]:
        rows = self.store.query(
            "SELECT k.* FROM knowledge_items k JOIN knowledge_tags t "
            "ON t.knowledge_id = k.knowledge_id WHERE t.tag=? LIMIT ?",
            (tag, limit))
        return [self.service._row_to_item(r) for r in rows]

    def _structured(self, query: Query) -> List[KnowledgeItem]:
        clauses: List[str] = []
        params: List[Any] = []
        if query.knowledge_types:
            clauses.append("knowledge_type IN (" +
                           ",".join("?" for _ in query.knowledge_types) + ")")
            params.extend(query.knowledge_types)
        if query.raga:
            clauses.append("raga=?")
            params.append(query.raga)
        if query.tala:
            clauses.append("tala=?")
            params.append(query.tala)
        if query.curriculum_level:
            clauses.append("curriculum_level=?")
            params.append(query.curriculum_level)
        if query.difficulty:
            clauses.append("difficulty=?")
            params.append(query.difficulty)
        if not clauses:
            return []
        rows = self.store.query(
            f"SELECT * FROM knowledge_items WHERE {' AND '.join(clauses)} "
            f"ORDER BY importance DESC, confidence DESC LIMIT ?",
            params + [query.limit * 2])
        return [self.service._row_to_item(r) for r in rows]

    # ==================================================================
    def _passes(self, item: KnowledgeItem, query: Query) -> bool:
        if query.statuses and item.status not in query.statuses:
            return False
        if item.confidence < query.min_confidence:
            return False
        if query.knowledge_types and item.knowledge_type not in query.knowledge_types:
            return False
        if query.scopes and not set(query.scopes) & set(item.scope):
            return False
        if query.source_id:
            row = self.store.one(
                "SELECT 1 FROM evidence WHERE knowledge_id=? AND source_id=? "
                "LIMIT 1", (item.knowledge_id, query.source_id))
            if row is None:
                return False
        return True

    @staticmethod
    def _quality_multiplier(item: KnowledgeItem) -> float:
        """Well-supported knowledge should surface first, but a disputed item
        must not vanish: a caller composing in this raga needs to know the
        disagreement exists."""
        multiplier = 0.55 + 0.45 * max(0.0, min(1.0, item.confidence))
        if item.status == Status.DISPUTED:
            multiplier *= 0.85
        if item.status == Status.ACCEPTED:
            multiplier *= 1.1
        return multiplier

    # ==================================================================
    def record_usage(self, task: str, query_text: str,
                     knowledge_ids: Sequence[str], context_id: str = "") -> None:
        """Section 41 - what was actually served, so it can be asked about."""
        from .store import dumps

        with self.store.transaction():
            self.store.execute(
                "INSERT INTO retrieval_usage(at, task, query, knowledge_ids, "
                "context_id, item_count) VALUES (?,?,?,?,?,?)",
                (time.time(), task, query_text, dumps(list(knowledge_ids)),
                 context_id, len(knowledge_ids)))
            for knowledge_id in knowledge_ids:
                self.store.execute(
                    "UPDATE knowledge_items SET usage_count=usage_count+1, "
                    "last_used_at=? WHERE knowledge_id=?",
                    (time.time(), knowledge_id))
