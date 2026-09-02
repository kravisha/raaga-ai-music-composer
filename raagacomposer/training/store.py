"""Durable storage for everything the Training tab does.

Training specification sections 15 and 16.  A creator who closes the
application mid-queue must find the queue, the objectives, the reports and the
learned knowledge exactly as they left them, and every learned item must still
be able to say where it came from.

This is a second SQLite file rather than more tables in ``knowledge.db``.  The
two stores answer different questions - that one holds what the agent knows
about music, this one holds the record of how training was conducted - and
keeping them apart means a creator can throw a training history away without
touching what the agent learned from their own recordings.  The link between
them is the knowledge entry's provenance, which names the source and the run.

Written with WAL, opened without a thread check, and every write wrapped in a
transaction, so the queue worker and the UI thread can both touch it.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.settings import config_dir
from .models import (Accessibility, Conflict, KnowledgeEntry, KnowledgeStatus,
                     LearningReport, LearningRun, LearningSource, Objective,
                     ObjectiveStatus, RunStatus, SearchQuery, new_id)

log = get_logger("training.store")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS searches (
    id TEXT PRIMARY KEY,
    at REAL,
    phrase TEXT,
    query TEXT,
    result_count INTEGER DEFAULT 0);

CREATE TABLE IF NOT EXISTS candidates (
    source_id TEXT PRIMARY KEY,
    search_id TEXT,
    source_type TEXT,
    title TEXT,
    url TEXT,
    author TEXT,
    description TEXT,
    duration REAL DEFAULT 0,
    published_date TEXT DEFAULT '',
    language TEXT DEFAULT '',
    relevance_score REAL DEFAULT 0,
    accessibility_status TEXT,
    transcript_available INTEGER DEFAULT 0,
    previously_learned INTEGER DEFAULT 0,
    provider TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    local_path TEXT DEFAULT '',
    found_at REAL,
    identity TEXT);
CREATE INDEX IF NOT EXISTS candidates_identity ON candidates(identity);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    source_id TEXT,
    search_phrase TEXT DEFAULT '',
    position INTEGER DEFAULT 0,
    status TEXT,
    progress REAL DEFAULT 0,
    detail TEXT DEFAULT '',
    result TEXT DEFAULT '',
    error TEXT DEFAULT '',
    attempts INTEGER DEFAULT 0,
    queued_at REAL,
    started_at REAL DEFAULT 0,
    completed_at REAL DEFAULT 0,
    supersedes TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);

CREATE TABLE IF NOT EXISTS objectives (
    objective_id TEXT PRIMARY KEY,
    run_id TEXT,
    description TEXT,
    category TEXT DEFAULT 'general',
    priority INTEGER DEFAULT 2,
    status TEXT,
    confidence REAL DEFAULT 0,
    evidence TEXT DEFAULT '',
    outcome TEXT DEFAULT '',
    user_defined INTEGER DEFAULT 0,
    position INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS objectives_run ON objectives(run_id);

CREATE TABLE IF NOT EXISTS reports (
    run_id TEXT PRIMARY KEY,
    generated_at REAL,
    payload TEXT);

CREATE TABLE IF NOT EXISTS knowledge (
    knowledge_id TEXT PRIMARY KEY,
    subject TEXT,
    concept TEXT,
    normalized_statement TEXT,
    category TEXT DEFAULT '',
    raga TEXT DEFAULT '',
    tala TEXT DEFAULT '',
    difficulty TEXT DEFAULT '',
    source_id TEXT,
    source_url TEXT DEFAULT '',
    source_title TEXT DEFAULT '',
    source_timestamp TEXT DEFAULT '',
    evidence TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    date_learned REAL,
    related_knowledge TEXT DEFAULT '[]',
    tags TEXT DEFAULT '[]',
    version INTEGER DEFAULT 1,
    status TEXT DEFAULT 'active',
    run_id TEXT DEFAULT '',
    objective_id TEXT DEFAULT '',
    user_approved INTEGER DEFAULT 0,
    contradicted INTEGER DEFAULT 0,
    identity TEXT);
CREATE INDEX IF NOT EXISTS knowledge_identity ON knowledge(identity);
CREATE INDEX IF NOT EXISTS knowledge_raga ON knowledge(raga);
CREATE INDEX IF NOT EXISTS knowledge_source ON knowledge(source_id);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    run_id TEXT,
    knowledge_id TEXT,
    existing_claim TEXT,
    new_claim TEXT,
    source_evidence TEXT DEFAULT '',
    existing_confidence REAL DEFAULT 0,
    new_confidence REAL DEFAULT 0,
    recommendation TEXT DEFAULT '',
    resolved INTEGER DEFAULT 0,
    resolution TEXT DEFAULT '',
    at REAL);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL,
    kind TEXT,
    detail TEXT,
    run_id TEXT DEFAULT '',
    source_id TEXT DEFAULT '',
    knowledge_id TEXT DEFAULT '');
"""


def identity_of(url: str, title: str = "") -> str:
    """A stable identity for a source, so the same lesson is recognised again.

    Section 10 asks for more than exact-URL matching: the same video reached by
    a share link, a tracking parameter or the mobile host is the same lesson.
    The host is normalised, the query string dropped except for a video id, and
    a title is used only when there is no usable URL at all.
    """
    text = (url or "").strip().lower()
    if text:
        text = re.sub(r"^https?://", "", text)
        text = re.sub(r"^(www|m)\.", "", text)
        # youtu.be/ID and youtube.com/watch?v=ID are the same lesson.
        short = re.match(r"youtu\.be/([\w-]+)", text)
        if short:
            return f"youtube:{short.group(1)}"
        watch = re.match(r"youtube\.com/watch\?.*\bv=([\w-]+)", text)
        if watch:
            return f"youtube:{watch.group(1)}"
        text = text.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        return f"url:{text}"
    return "title:" + hashlib.sha1(
        (title or "").strip().lower().encode("utf-8")).hexdigest()[:16]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(text: str, fallback: Any) -> Any:
    try:
        return json.loads(text) if text else fallback
    except (ValueError, TypeError):
        return fallback


class TrainingStore:
    """Everything the Training tab needs to survive a restart."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else config_dir() / "training.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._closed = False
        self._migrate()

    # -- lifecycle ---------------------------------------------------------
    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA)
            found = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if found is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),))
                log.info("training store created at %s", self.path)
                return
            stored = int(found["value"])
            if stored > SCHEMA_VERSION:
                raise RuntimeError(
                    f"training.db was written by a newer version "
                    f"(schema {stored} > {SCHEMA_VERSION})")
            if stored < SCHEMA_VERSION:
                log.info("migrating training schema %d -> %d", stored,
                         SCHEMA_VERSION)
                self._conn.execute(
                    "UPDATE meta SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION),))

    @property
    def schema_version(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        return int(row["value"]) if row else 0

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._conn.close()

    # -- audit (section 16) ------------------------------------------------
    def audit(self, kind: str, detail: str = "", *, run_id: str = "",
              source_id: str = "", knowledge_id: str = "") -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO audit(at, kind, detail, run_id, source_id, "
                "knowledge_id) VALUES (?,?,?,?,?,?)",
                (time.time(), kind, detail, run_id, source_id, knowledge_id))

    def audit_trail(self, *, run_id: str = "", knowledge_id: str = "",
                    limit: int = 200) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if knowledge_id:
            clauses.append("knowledge_id=?")
            params.append(knowledge_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM audit {where} ORDER BY id DESC LIMIT ?",
            params + [limit]).fetchall()
        return [dict(r) for r in rows]

    # -- searches ----------------------------------------------------------
    def record_search(self, query: SearchQuery, result_count: int) -> str:
        search_id = new_id("sch")
        with self._conn:
            self._conn.execute(
                "INSERT INTO searches(id, at, phrase, query, result_count) "
                "VALUES (?,?,?,?,?)",
                (search_id, time.time(), query.phrase, _json(query.as_dict()),
                 result_count))
        return search_id

    def searches(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM searches ORDER BY at DESC LIMIT ?",
            (limit,)).fetchall()
        return [{**dict(r), "query": _loads(r["query"], {})} for r in rows]

    # -- candidates --------------------------------------------------------
    def save_candidate(self, source: LearningSource) -> LearningSource:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO candidates(source_id, search_id, "
                "source_type, title, url, author, description, duration, "
                "published_date, language, relevance_score, "
                "accessibility_status, transcript_available, "
                "previously_learned, provider, metadata, local_path, found_at,"
                " identity) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (source.source_id, source.search_id, source.source_type,
                 source.title, source.url, source.author, source.description,
                 source.duration, source.published_date, source.language,
                 source.relevance_score, source.accessibility_status,
                 int(source.transcript_available),
                 int(source.previously_learned), source.provider,
                 _json(source.metadata), source.local_path, source.found_at,
                 identity_of(source.url, source.title)))
        return source

    def candidate(self, source_id: str) -> Optional[LearningSource]:
        row = self._conn.execute(
            "SELECT * FROM candidates WHERE source_id=?",
            (source_id,)).fetchone()
        return self._row_to_source(row) if row else None

    def candidates_for_search(self, search_id: str) -> List[LearningSource]:
        rows = self._conn.execute(
            "SELECT * FROM candidates WHERE search_id=? "
            "ORDER BY relevance_score DESC", (search_id,)).fetchall()
        return [self._row_to_source(r) for r in rows]

    def update_candidate(self, source_id: str, **fields) -> None:
        allowed = {"accessibility_status", "transcript_available",
                   "previously_learned", "local_path", "duration", "language",
                   "description", "title", "author", "metadata"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        if "metadata" in sets and not isinstance(sets["metadata"], str):
            sets["metadata"] = _json(sets["metadata"])
        for flag in ("transcript_available", "previously_learned"):
            if flag in sets:
                sets[flag] = int(bool(sets[flag]))
        assignments = ", ".join(f"{k}=?" for k in sets)
        with self._conn:
            self._conn.execute(
                f"UPDATE candidates SET {assignments} WHERE source_id=?",
                list(sets.values()) + [source_id])

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> LearningSource:
        return LearningSource(
            source_id=row["source_id"], search_id=row["search_id"] or "",
            source_type=row["source_type"] or "", title=row["title"] or "",
            url=row["url"] or "", author=row["author"] or "",
            description=row["description"] or "",
            duration=float(row["duration"] or 0.0),
            published_date=row["published_date"] or "",
            language=row["language"] or "",
            relevance_score=float(row["relevance_score"] or 0.0),
            accessibility_status=row["accessibility_status"] or "",
            transcript_available=bool(row["transcript_available"]),
            previously_learned=bool(row["previously_learned"]),
            provider=row["provider"] or "",
            metadata=_loads(row["metadata"], {}),
            local_path=row["local_path"] or "",
            found_at=float(row["found_at"] or 0.0))

    # -- duplicate detection (section 10) ---------------------------------
    def completed_run_for(self, source: LearningSource) -> Optional[LearningRun]:
        """Has this lesson already been learned, by any URL that names it?"""
        identity = identity_of(source.url, source.title)
        row = self._conn.execute(
            "SELECT r.* FROM runs r JOIN candidates c "
            "ON c.source_id = r.source_id "
            "WHERE c.identity=? AND r.status=? "
            "ORDER BY r.completed_at DESC LIMIT 1",
            (identity, RunStatus.COMPLETED)).fetchone()
        return self._row_to_run(row) if row else None

    # -- runs --------------------------------------------------------------
    def add_run(self, run: LearningRun) -> LearningRun:
        if not run.position:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(position), 0) AS p FROM runs").fetchone()
            run.position = int(row["p"]) + 1
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs(run_id, source_id, search_phrase, position, "
                "status, progress, detail, result, error, attempts, queued_at,"
                " started_at, completed_at, supersedes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run.run_id, run.source_id, run.search_phrase, run.position,
                 run.status, run.progress, run.detail, run.result, run.error,
                 run.attempts, run.queued_at, run.started_at,
                 run.completed_at, run.supersedes))
        self.audit("run.queued", f"queued {run.source_id}", run_id=run.run_id,
                   source_id=run.source_id)
        return run

    def update_run(self, run_id: str, **fields) -> None:
        allowed = {"status", "progress", "detail", "result", "error",
                   "attempts", "started_at", "completed_at", "position"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        assignments = ", ".join(f"{k}=?" for k in sets)
        with self._conn:
            self._conn.execute(f"UPDATE runs SET {assignments} WHERE run_id=?",
                               list(sets.values()) + [run_id])

    def run(self, run_id: str) -> Optional[LearningRun]:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id=?",
                                 (run_id,)).fetchone()
        return self._row_to_run(row) if row else None

    def runs(self, statuses: Sequence[str] = (), limit: int = 200
             ) -> List[LearningRun]:
        if statuses:
            marks = ",".join("?" for _ in statuses)
            rows = self._conn.execute(
                f"SELECT * FROM runs WHERE status IN ({marks}) "
                f"ORDER BY position LIMIT ?",
                list(statuses) + [limit]).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY position LIMIT ?",
                (limit,)).fetchall()
        return [self._row_to_run(r) for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM objectives WHERE run_id=?",
                               (run_id,))
            self._conn.execute("DELETE FROM reports WHERE run_id=?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        self.audit("run.deleted", "removed from the queue", run_id=run_id)

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> LearningRun:
        return LearningRun(
            run_id=row["run_id"], source_id=row["source_id"] or "",
            search_phrase=row["search_phrase"] or "",
            position=int(row["position"] or 0), status=row["status"] or "",
            progress=float(row["progress"] or 0.0),
            detail=row["detail"] or "", result=row["result"] or "",
            error=row["error"] or "", attempts=int(row["attempts"] or 0),
            queued_at=float(row["queued_at"] or 0.0),
            started_at=float(row["started_at"] or 0.0),
            completed_at=float(row["completed_at"] or 0.0),
            supersedes=row["supersedes"] or "")

    # -- objectives --------------------------------------------------------
    def save_objectives(self, run_id: str,
                        objectives: Sequence[Objective]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM objectives WHERE run_id=?",
                               (run_id,))
            for position, objective in enumerate(objectives):
                objective.run_id = run_id
                objective.position = position
                self._conn.execute(
                    "INSERT INTO objectives(objective_id, run_id, description,"
                    " category, priority, status, confidence, evidence, "
                    "outcome, user_defined, position) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (objective.objective_id, run_id, objective.description,
                     objective.category, objective.priority, objective.status,
                     objective.confidence, objective.evidence,
                     objective.outcome, int(objective.user_defined), position))

    def objectives(self, run_id: str) -> List[Objective]:
        rows = self._conn.execute(
            "SELECT * FROM objectives WHERE run_id=? ORDER BY position",
            (run_id,)).fetchall()
        return [Objective(
            objective_id=r["objective_id"], run_id=r["run_id"],
            description=r["description"] or "", category=r["category"] or "",
            priority=int(r["priority"] or 2), status=r["status"] or "",
            confidence=float(r["confidence"] or 0.0),
            evidence=r["evidence"] or "", outcome=r["outcome"] or "",
            user_defined=bool(r["user_defined"]),
            position=int(r["position"] or 0)) for r in rows]

    # -- reports -----------------------------------------------------------
    def save_report(self, report: LearningReport) -> None:
        payload = {
            "summary": report.summary,
            "understood": report.understood,
            "learned": report.learned,
            "confirmed": report.confirmed,
            "practical_application": report.practical_application,
            "confidence": report.confidence,
            "next_learning": report.next_learning,
            "knowledge_ids": report.knowledge_ids,
            "analysed_representation": report.analysed_representation,
            "honest_limits": report.honest_limits,
            "conflict_ids": [c.conflict_id for c in report.conflicts],
        }
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO reports(run_id, generated_at, payload)"
                " VALUES (?,?,?)",
                (report.run_id, report.generated_at, _json(payload)))
        self.audit("report.saved", f"{len(report.learned)} item(s) learned",
                   run_id=report.run_id)

    def report(self, run_id: str) -> Optional[LearningReport]:
        row = self._conn.execute("SELECT * FROM reports WHERE run_id=?",
                                 (run_id,)).fetchone()
        if row is None:
            return None
        payload = _loads(row["payload"], {})
        run = self.run(run_id)
        report = LearningReport(
            run_id=run_id, generated_at=float(row["generated_at"] or 0.0),
            objectives=self.objectives(run_id),
            summary=payload.get("summary", ""),
            understood=payload.get("understood", ""),
            learned=payload.get("learned", []),
            confirmed=payload.get("confirmed", []),
            practical_application=payload.get("practical_application", []),
            confidence=float(payload.get("confidence", 0.0)),
            next_learning=payload.get("next_learning", []),
            knowledge_ids=payload.get("knowledge_ids", []),
            analysed_representation=payload.get("analysed_representation", ""),
            honest_limits=payload.get("honest_limits", []))
        if run is not None:
            report.source = self.candidate(run.source_id)
        report.conflicts = [c for c in self.conflicts(run_id=run_id)]
        return report

    # -- knowledge base (section 9) ---------------------------------------
    def existing_knowledge(self, entry: KnowledgeEntry
                           ) -> Optional[KnowledgeEntry]:
        """The active entry this one would collide with, if any."""
        identity = hashlib.sha1(
            "|".join(entry.fingerprint_values()[:3]).encode("utf-8")
        ).hexdigest()[:20]
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE identity=? AND status=? "
            "ORDER BY version DESC LIMIT 1",
            (identity, KnowledgeStatus.ACTIVE)).fetchone()
        return self._row_to_knowledge(row) if row else None

    def add_knowledge(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        # Identity is subject+category+raga: the same claim about the same
        # thing, whatever words a particular teacher used for it.
        identity = hashlib.sha1(
            "|".join(entry.fingerprint_values()[:3]).encode("utf-8")
        ).hexdigest()[:20]
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO knowledge(knowledge_id, subject, "
                "concept, normalized_statement, category, raga, tala, "
                "difficulty, source_id, source_url, source_title, "
                "source_timestamp, evidence, confidence, date_learned, "
                "related_knowledge, tags, version, status, run_id, "
                "objective_id, user_approved, contradicted, identity) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (entry.knowledge_id, entry.subject, entry.concept,
                 entry.normalized_statement, entry.category, entry.raga,
                 entry.tala, entry.difficulty, entry.source_id,
                 entry.source_url, entry.source_title, entry.source_timestamp,
                 entry.evidence, entry.confidence, entry.date_learned,
                 _json(entry.related_knowledge), _json(entry.tags),
                 entry.version, entry.status, entry.run_id,
                 entry.objective_id, int(entry.user_approved),
                 int(entry.contradicted), identity))
        self.audit("knowledge.added", entry.normalized_statement[:200],
                   run_id=entry.run_id, source_id=entry.source_id,
                   knowledge_id=entry.knowledge_id)
        return entry

    def update_knowledge(self, knowledge_id: str, **fields) -> None:
        allowed = {"status", "confidence", "user_approved", "contradicted",
                   "version", "normalized_statement", "evidence", "tags"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        if "tags" in sets and not isinstance(sets["tags"], str):
            sets["tags"] = _json(sets["tags"])
        for flag in ("user_approved", "contradicted"):
            if flag in sets:
                sets[flag] = int(bool(sets[flag]))
        assignments = ", ".join(f"{k}=?" for k in sets)
        with self._conn:
            self._conn.execute(
                f"UPDATE knowledge SET {assignments} WHERE knowledge_id=?",
                list(sets.values()) + [knowledge_id])
        self.audit("knowledge.updated", _json(sets)[:200],
                   knowledge_id=knowledge_id)

    def knowledge(self, knowledge_id: str) -> Optional[KnowledgeEntry]:
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE knowledge_id=?",
            (knowledge_id,)).fetchone()
        return self._row_to_knowledge(row) if row else None

    def search_knowledge(self, *, keyword: str = "", raga: str = "",
                         tala: str = "", category: str = "",
                         source_id: str = "", objective_id: str = "",
                         difficulty: str = "", tag: str = "",
                         status: str = "", limit: int = 200
                         ) -> List[KnowledgeEntry]:
        """Section 9's required search axes, all optional and combinable."""
        clauses: List[str] = []
        params: List[Any] = []
        if keyword:
            clauses.append("(subject LIKE ? OR concept LIKE ? OR "
                           "normalized_statement LIKE ? OR evidence LIKE ?)")
            params.extend([f"%{keyword}%"] * 4)
        for column, value in (("raga", raga), ("tala", tala),
                              ("category", category), ("source_id", source_id),
                              ("objective_id", objective_id),
                              ("difficulty", difficulty), ("status", status)):
            if value:
                clauses.append(f"{column}=?")
                params.append(value)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM knowledge {where} ORDER BY date_learned DESC "
            f"LIMIT ?", params + [limit]).fetchall()
        return [self._row_to_knowledge(r) for r in rows]

    def knowledge_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM knowledge WHERE status=?",
            (KnowledgeStatus.ACTIVE,)).fetchone()
        return int(row["n"]) if row else 0

    @staticmethod
    def _row_to_knowledge(row: sqlite3.Row) -> KnowledgeEntry:
        return KnowledgeEntry(
            knowledge_id=row["knowledge_id"], subject=row["subject"] or "",
            concept=row["concept"] or "",
            normalized_statement=row["normalized_statement"] or "",
            category=row["category"] or "", raga=row["raga"] or "",
            tala=row["tala"] or "", difficulty=row["difficulty"] or "",
            source_id=row["source_id"] or "",
            source_url=row["source_url"] or "",
            source_title=row["source_title"] or "",
            source_timestamp=row["source_timestamp"] or "",
            evidence=row["evidence"] or "",
            confidence=float(row["confidence"] or 0.0),
            date_learned=float(row["date_learned"] or 0.0),
            related_knowledge=_loads(row["related_knowledge"], []),
            tags=_loads(row["tags"], []), version=int(row["version"] or 1),
            status=row["status"] or KnowledgeStatus.ACTIVE,
            run_id=row["run_id"] or "", objective_id=row["objective_id"] or "",
            user_approved=bool(row["user_approved"]),
            contradicted=bool(row["contradicted"]))

    # -- conflicts (section 8.7) ------------------------------------------
    def add_conflict(self, conflict: Conflict) -> Conflict:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO conflicts(conflict_id, run_id, "
                "knowledge_id, existing_claim, new_claim, source_evidence, "
                "existing_confidence, new_confidence, recommendation, "
                "resolved, resolution, at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (conflict.conflict_id, conflict.run_id, conflict.knowledge_id,
                 conflict.existing_claim, conflict.new_claim,
                 conflict.source_evidence, conflict.existing_confidence,
                 conflict.new_confidence, conflict.recommendation,
                 int(conflict.resolved), conflict.resolution, conflict.at))
        self.audit("conflict.recorded", conflict.new_claim[:200],
                   run_id=conflict.run_id, knowledge_id=conflict.knowledge_id)
        return conflict

    def conflicts(self, *, run_id: str = "", unresolved_only: bool = False,
                  limit: int = 200) -> List[Conflict]:
        clauses, params = [], []
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if unresolved_only:
            clauses.append("resolved=0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM conflicts {where} ORDER BY at DESC LIMIT ?",
            params + [limit]).fetchall()
        return [Conflict(
            conflict_id=r["conflict_id"], run_id=r["run_id"] or "",
            knowledge_id=r["knowledge_id"] or "",
            existing_claim=r["existing_claim"] or "",
            new_claim=r["new_claim"] or "",
            source_evidence=r["source_evidence"] or "",
            existing_confidence=float(r["existing_confidence"] or 0.0),
            new_confidence=float(r["new_confidence"] or 0.0),
            recommendation=r["recommendation"] or "",
            resolved=bool(r["resolved"]), resolution=r["resolution"] or "",
            at=float(r["at"] or 0.0)) for r in rows]

    def resolve_conflict(self, conflict_id: str, resolution: str) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE conflicts SET resolved=1, resolution=? "
                "WHERE conflict_id=?", (resolution, conflict_id))
        self.audit("conflict.resolved", resolution[:200])

    # -- history (section 12) ---------------------------------------------
    def history(self, *, raga: str = "", status: str = "",
                min_confidence: float = 0.0, since: float = 0.0,
                phrase: str = "", limit: int = 200) -> List[Dict[str, Any]]:
        clauses = ["r.status NOT IN ('queued')"]
        params: List[Any] = []
        if status:
            clauses = ["r.status=?"]
            params.append(status)
        if since:
            clauses.append("r.completed_at >= ?")
            params.append(since)
        if phrase:
            clauses.append("r.search_phrase LIKE ?")
            params.append(f"%{phrase}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT r.*, c.title, c.url, c.author FROM runs r "
            f"LEFT JOIN candidates c ON c.source_id = r.source_id "
            f"{where} ORDER BY r.completed_at DESC, r.position LIMIT ?",
            params + [limit]).fetchall()

        out: List[Dict[str, Any]] = []
        for row in rows:
            entries = self.search_knowledge(source_id=row["source_id"],
                                            limit=500)
            if raga and not any(e.raga.lower() == raga.lower()
                                for e in entries):
                continue
            confidence = (sum(e.confidence for e in entries) / len(entries)
                          if entries else 0.0)
            if confidence < min_confidence:
                continue
            objectives = self.objectives(row["run_id"])
            out.append({
                "run_id": row["run_id"],
                "source_id": row["source_id"],
                "title": row["title"] or "(unknown source)",
                "url": row["url"] or "",
                "author": row["author"] or "",
                "search_phrase": row["search_phrase"] or "",
                "status": row["status"],
                "result": row["result"] or "",
                "completed_at": float(row["completed_at"] or 0.0),
                "objectives": len(objectives),
                "objectives_met": sum(1 for o in objectives if o.met),
                "knowledge_added": len(entries),
                "conflicts": len(self.conflicts(run_id=row["run_id"])),
                "confidence": round(confidence, 3),
            })
        return out
