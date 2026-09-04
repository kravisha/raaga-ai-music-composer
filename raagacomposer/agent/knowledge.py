"""Permanent knowledge repository - the agent's long-term musical memory.

Learning specification section 8.  This is the source of truth: it survives
application restart, machine restart and software upgrade.  Everything the
agent learns, every source it learned it from, its curriculum progress, the
compositions it has made and the feedback it was given all live here.

SQLite, one file, written with WAL so a crash mid-write cannot corrupt it.
Nothing is ever silently overwritten: facts carry confidence and provenance,
disagreement is recorded rather than resolved by the last writer, and every
change is appended to an event log (section 20).

The connection is shared between the UI thread and the background learning
thread (``check_same_thread=False``), and the sqlite3 module does not make
that safe on its own: two threads issuing statements on the same connection
at the same time can corrupt its internal state rather than merely block on
each other.  ``self._lock`` serialises every method that touches
``self._conn``, so only one statement runs at a time no matter which thread
asked for it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.logging_setup import get_logger
from ..core.settings import config_dir

log = get_logger("agent.knowledge")

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    locator TEXT,
    title TEXT,
    performer TEXT,
    raaga TEXT,
    content_type TEXT,
    rights_status TEXT,
    provider TEXT,
    quality REAL DEFAULT 0.5,
    ingested_at REAL,
    extraction_version TEXT,
    confidence REAL DEFAULT 0.5,
    status TEXT DEFAULT 'pending',
    error TEXT DEFAULT '',
    fingerprint TEXT UNIQUE,
    notes TEXT DEFAULT '');

CREATE TABLE IF NOT EXISTS phrases (
    id TEXT PRIMARY KEY,
    raaga TEXT,
    swaras TEXT,
    midi TEXT,
    durations TEXT,
    function TEXT DEFAULT 'phrase',
    source_id TEXT,
    confidence REAL DEFAULT 0.5,
    fingerprint TEXT,
    contour TEXT DEFAULT '',
    tempo REAL DEFAULT 0,
    votes INTEGER DEFAULT 1,
    rejected INTEGER DEFAULT 0,
    learned_at REAL,
    notes TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS phrases_by_raaga ON phrases (raaga, rejected);
CREATE INDEX IF NOT EXISTS phrases_by_fingerprint ON phrases (fingerprint);

CREATE TABLE IF NOT EXISTS raaga_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raaga TEXT,
    key TEXT,
    value TEXT,
    confidence REAL DEFAULT 0.5,
    source_id TEXT,
    disputed INTEGER DEFAULT 0,
    learned_at REAL,
    notes TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS facts_by_raaga ON raaga_facts (raaga, key);

CREATE TABLE IF NOT EXISTS curriculum_progress (
    unit_id TEXT PRIMARY KEY,
    raaga TEXT,
    status TEXT DEFAULT 'not_started',
    mastery REAL DEFAULT 0.0,
    attempts INTEGER DEFAULT 0,
    failures INTEGER DEFAULT 0,
    last_attempted_at REAL DEFAULT 0,
    completed_at REAL DEFAULT 0,
    notes TEXT DEFAULT '');

CREATE TABLE IF NOT EXISTS compositions (
    id TEXT PRIMARY KEY,
    at REAL,
    project_id TEXT,
    title TEXT,
    raaga TEXT,
    brief TEXT,
    structure TEXT,
    scores TEXT,
    final_score REAL DEFAULT 0,
    notes TEXT DEFAULT '');

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    at REAL,
    target_kind TEXT,
    target_id TEXT,
    raaga TEXT,
    text TEXT,
    sentiment TEXT,
    applied INTEGER DEFAULT 0);

CREATE TABLE IF NOT EXISTS agent_state (
    key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    at REAL,
    kind TEXT,
    payload TEXT,
    status TEXT DEFAULT 'queued');

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL,
    kind TEXT,
    unit_id TEXT DEFAULT '',
    raaga TEXT DEFAULT '',
    source_id TEXT DEFAULT '',
    detail TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS events_by_time ON events (at DESC);

CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    at REAL,
    last_at REAL,
    raaga TEXT,
    unit_id TEXT,
    attempt INTEGER,
    task TEXT,
    method TEXT,
    result REAL,
    kind TEXT,
    dimension TEXT,
    failure_reason TEXT,
    evidence TEXT,
    correction TEXT,
    related TEXT,
    source_run TEXT,
    confidence REAL,
    recurrences INTEGER DEFAULT 1,
    applied INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS lessons_by_unit ON lessons (raaga, unit_id, kind);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def fingerprint(values: Sequence[Any]) -> str:
    """Stable fingerprint of a phrase, used for de-duplication."""
    payload = "|".join(str(v) for v in values)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------
@dataclass
class Source:
    id: str = field(default_factory=lambda: new_id("src"))
    locator: str = ""
    title: str = ""
    performer: str = ""
    raaga: str = ""
    content_type: str = "audio"
    rights_status: str = "unknown"
    provider: str = ""
    quality: float = 0.5
    ingested_at: float = field(default_factory=time.time)
    extraction_version: str = ""
    confidence: float = 0.5
    status: str = "pending"
    error: str = ""
    fingerprint: str = ""
    notes: str = ""


@dataclass
class Phrase:
    id: str = field(default_factory=lambda: new_id("phr"))
    raaga: str = ""
    swaras: List[str] = field(default_factory=list)
    midi: List[int] = field(default_factory=list)
    durations: List[float] = field(default_factory=list)
    function: str = "phrase"
    source_id: str = ""
    confidence: float = 0.5
    fingerprint: str = ""
    contour: str = ""
    tempo: float = 0.0
    votes: int = 1
    rejected: bool = False
    learned_at: float = field(default_factory=time.time)
    notes: str = ""

    def compute_fingerprint(self) -> str:
        return fingerprint([self.raaga] + list(self.swaras))

    @property
    def length(self) -> int:
        return len(self.swaras)


@dataclass
class Fact:
    raaga: str
    key: str
    value: str
    confidence: float = 0.5
    source_id: str = ""
    disputed: bool = False
    learned_at: float = field(default_factory=time.time)
    notes: str = ""


@dataclass
class Lesson:
    """The spec's Failure/Lesson object (section 38): a task, the method that
    was tried, its result, why it failed, the correction offered, related
    knowledge, the source/run it came from, a confidence and a date - so a
    mistake made once does not have to be rediscovered."""
    id: str = field(default_factory=lambda: new_id("les"))
    at: float = field(default_factory=time.time)
    last_at: float = field(default_factory=time.time)
    raaga: str = ""
    unit_id: str = ""
    attempt: int = 0
    task: str = ""
    method: str = ""
    result: float = 0.0
    kind: str = ""
    dimension: str = ""
    failure_reason: str = ""
    evidence: str = ""
    correction: str = ""
    related: List[str] = field(default_factory=list)
    source_run: str = ""
    confidence: float = 0.5
    recurrences: int = 1
    applied: bool = False


@dataclass
class UnitProgress:
    unit_id: str
    raaga: str = ""
    status: str = "not_started"        # not_started | in_progress | passed | failed
    mastery: float = 0.0
    attempts: int = 0
    failures: int = 0
    last_attempted_at: float = 0.0
    completed_at: float = 0.0
    notes: str = ""


# --------------------------------------------------------------------------
# repository
# --------------------------------------------------------------------------
class KnowledgeRepository:
    """Durable store. Every method is safe to call from the agent thread.

    Every public method that touches ``self._conn`` holds ``self._lock`` for
    its whole body: read and write alike, so the background learner and the
    UI never run a statement on the connection at the same time.  Rows are
    always materialised (``fetchone``/``fetchall``) before the lock is
    released - nobody is handed a live cursor to read after the fact.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else config_dir() / "knowledge.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._closed = False
        self._migrate()

    # -- lifecycle ---------------------------------------------------------
    def _migrate(self) -> None:
        # Called once from __init__, before this object can have reached any
        # other thread - nothing to serialise against yet.
        with self._conn:
            self._conn.executescript(_SCHEMA)
            found = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if found is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),))
                log.info("knowledge repository created at %s", self.path)
            else:
                stored = int(found["value"])
                if stored > SCHEMA_VERSION:
                    raise RuntimeError(
                        f"knowledge.db was written by a newer version "
                        f"(schema {stored} > {SCHEMA_VERSION})")
                if stored < SCHEMA_VERSION:
                    log.info("migrating knowledge schema %d -> %d", stored,
                             SCHEMA_VERSION)
                    self._conn.execute(
                        "UPDATE meta SET value=? WHERE key='schema_version'",
                        (str(SCHEMA_VERSION),))

    @property
    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
            return int(row["value"]) if row else 0

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    # -- events ------------------------------------------------------------
    def log_event(self, kind: str, detail: str = "", unit_id: str = "",
                  raaga: str = "", source_id: str = "") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO events(at, kind, unit_id, raaga, source_id,"
                    " detail) VALUES (?,?,?,?,?,?)",
                    (time.time(), kind, unit_id, raaga, source_id, detail))

    def events(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events ORDER BY at DESC LIMIT ?",
                (limit,)).fetchall()
            return [dict(r) for r in rows]

    # -- sources -----------------------------------------------------------
    def add_source(self, source: Source) -> Tuple[Source, bool]:
        """Insert a source. Returns (stored, is_new); duplicates are not re-added."""
        with self._lock:
            source.fingerprint = source.fingerprint or fingerprint(
                [source.provider, source.locator])
            existing = self._conn.execute(
                "SELECT * FROM sources WHERE fingerprint=?",
                (source.fingerprint,)).fetchone()
            if existing is not None:
                return self._row_to_source(existing), False
            with self._conn:
                self._conn.execute(
                    "INSERT INTO sources(id, locator, title, performer, raaga,"
                    " content_type, rights_status, provider, quality, ingested_at,"
                    " extraction_version, confidence, status, error, fingerprint,"
                    " notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (source.id, source.locator, source.title, source.performer,
                     source.raaga, source.content_type, source.rights_status,
                     source.provider, source.quality, source.ingested_at,
                     source.extraction_version, source.confidence, source.status,
                     source.error, source.fingerprint, source.notes))
            self.log_event("source.added", source.title or source.locator,
                           raaga=source.raaga, source_id=source.id)
            return source, True

    def update_source(self, source_id: str, **fields) -> None:
        with self._lock:
            if not fields:
                return
            allowed = {"status", "error", "confidence", "quality", "raaga",
                       "extraction_version", "notes", "rights_status", "title"}
            sets = {k: v for k, v in fields.items() if k in allowed}
            if not sets:
                return
            assignments = ", ".join(f"{k}=?" for k in sets)
            with self._conn:
                self._conn.execute(f"UPDATE sources SET {assignments} WHERE id=?",
                                   list(sets.values()) + [source_id])

    def source(self, source_id: str) -> Optional[Source]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM sources WHERE id=?",
                                     (source_id,)).fetchone()
            return self._row_to_source(row) if row else None

    def sources(self, raaga: str = "", limit: int = 200) -> List[Source]:
        with self._lock:
            if raaga:
                rows = self._conn.execute(
                    "SELECT * FROM sources WHERE raaga=? ORDER BY ingested_at DESC"
                    " LIMIT ?", (raaga, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM sources ORDER BY ingested_at DESC LIMIT ?",
                    (limit,)).fetchall()
            return [self._row_to_source(r) for r in rows]

    def has_source(self, provider: str, locator: str) -> bool:
        with self._lock:
            fp = fingerprint([provider, locator])
            return self._conn.execute(
                "SELECT 1 FROM sources WHERE fingerprint=?",
                (fp,)).fetchone() is not None

    @staticmethod
    def _row_to_source(row: sqlite3.Row) -> Source:
        return Source(**{k: row[k] for k in row.keys()})

    # -- phrases -----------------------------------------------------------
    def add_phrase(self, phrase: Phrase) -> Tuple[Phrase, bool]:
        """Store a learned phrase; an identical one strengthens the existing entry."""
        with self._lock:
            phrase.fingerprint = phrase.fingerprint or phrase.compute_fingerprint()
            existing = self._conn.execute(
                "SELECT * FROM phrases WHERE fingerprint=?",
                (phrase.fingerprint,)).fetchone()
            if existing is not None:
                votes = existing["votes"] + 1
                confidence = min(0.99, max(existing["confidence"],
                                           phrase.confidence) + 0.05)
                with self._conn:
                    self._conn.execute(
                        "UPDATE phrases SET votes=?, confidence=? WHERE id=?",
                        (votes, confidence, existing["id"]))
                stored = self._row_to_phrase(existing)
                stored.votes = votes
                stored.confidence = confidence
                return stored, False

            with self._conn:
                self._conn.execute(
                    "INSERT INTO phrases(id, raaga, swaras, midi, durations,"
                    " function, source_id, confidence, fingerprint, contour, tempo,"
                    " votes, rejected, learned_at, notes)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (phrase.id, phrase.raaga, json.dumps(phrase.swaras),
                     json.dumps(phrase.midi), json.dumps(phrase.durations),
                     phrase.function, phrase.source_id, phrase.confidence,
                     phrase.fingerprint, phrase.contour, phrase.tempo, phrase.votes,
                     int(phrase.rejected), phrase.learned_at, phrase.notes))
            return phrase, True

    def phrases(self, raaga: str = "", min_confidence: float = 0.0,
                include_rejected: bool = False, limit: int = 500,
                function: str = "") -> List[Phrase]:
        with self._lock:
            clauses = ["confidence >= ?"]
            params: List[Any] = [min_confidence]
            if raaga:
                clauses.append("raaga = ?")
                params.append(raaga)
            if function:
                clauses.append("function = ?")
                params.append(function)
            if not include_rejected:
                clauses.append("rejected = 0")
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT * FROM phrases WHERE {' AND '.join(clauses)}"
                f" ORDER BY confidence DESC, votes DESC LIMIT ?", params).fetchall()
            return [self._row_to_phrase(r) for r in rows]

    def phrase(self, phrase_id: str) -> Optional[Phrase]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM phrases WHERE id=?",
                                     (phrase_id,)).fetchone()
            return self._row_to_phrase(row) if row else None

    def set_phrase_confidence(self, phrase_id: str, confidence: float,
                              note: str = "") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE phrases SET confidence=?, notes=? WHERE id=?",
                    (max(0.0, min(1.0, confidence)), note, phrase_id))

    def reject_phrase(self, phrase_id: str, reason: str = "") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE phrases SET rejected=1, confidence=0.0, notes=?"
                    " WHERE id=?",
                    (reason, phrase_id))
            self.log_event("phrase.rejected", reason)

    def count_phrases(self, raaga: str = "") -> int:
        with self._lock:
            if raaga:
                row = self._conn.execute(
                    "SELECT count(*) AS n FROM phrases WHERE raaga=? AND rejected=0",
                    (raaga,)).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT count(*) AS n FROM phrases WHERE rejected=0").fetchone()
            return int(row["n"])

    @staticmethod
    def _row_to_phrase(row: sqlite3.Row) -> Phrase:
        return Phrase(
            id=row["id"], raaga=row["raaga"], swaras=json.loads(row["swaras"]),
            midi=json.loads(row["midi"]), durations=json.loads(row["durations"]),
            function=row["function"], source_id=row["source_id"],
            confidence=row["confidence"], fingerprint=row["fingerprint"],
            contour=row["contour"], tempo=row["tempo"], votes=row["votes"],
            rejected=bool(row["rejected"]), learned_at=row["learned_at"],
            notes=row["notes"])

    # -- facts -------------------------------------------------------------
    def add_fact(self, fact: Fact) -> None:
        """Record a claim. A conflicting claim is flagged, never overwritten."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM raaga_facts WHERE raaga=? AND key=?",
                (fact.raaga, fact.key)).fetchall()
            for row in rows:
                if row["value"] == fact.value:
                    with self._conn:
                        self._conn.execute(
                            "UPDATE raaga_facts SET confidence=? WHERE id=?",
                            (min(0.99, max(row["confidence"], fact.confidence)
                                 + 0.05), row["id"]))
                    return
            # An "observed_*" entry is evidence from one source, not a claim
            # about the raaga: two recordings showing different notes do not
            # contradict each other. Only canonical claims can be in dispute.
            disputed = bool(rows) and not fact.key.startswith("observed_")
            if disputed:
                with self._conn:
                    self._conn.execute(
                        "UPDATE raaga_facts SET disputed=1 WHERE raaga=? AND key=?",
                        (fact.raaga, fact.key))
                self.log_event("fact.disputed", f"{fact.key}={fact.value}",
                               raaga=fact.raaga)
            with self._conn:
                self._conn.execute(
                    "INSERT INTO raaga_facts(raaga, key, value, confidence,"
                    " source_id, disputed, learned_at, notes)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (fact.raaga, fact.key, fact.value, fact.confidence,
                     fact.source_id, int(disputed), fact.learned_at, fact.notes))

    def facts(self, raaga: str, key: str = "") -> List[Fact]:
        with self._lock:
            if key:
                rows = self._conn.execute(
                    "SELECT * FROM raaga_facts WHERE raaga=? AND key=?"
                    " ORDER BY confidence DESC", (raaga, key)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM raaga_facts WHERE raaga=? ORDER BY key,"
                    " confidence DESC", (raaga,)).fetchall()
            return [Fact(raaga=r["raaga"], key=r["key"], value=r["value"],
                         confidence=r["confidence"], source_id=r["source_id"],
                         disputed=bool(r["disputed"]), learned_at=r["learned_at"],
                         notes=r["notes"]) for r in rows]

    def best_fact(self, raaga: str, key: str) -> Optional[Fact]:
        with self._lock:
            found = self.facts(raaga, key)
            return found[0] if found else None

    def known_raagas(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT raaga FROM raaga_facts WHERE raaga <> ''"
                " ORDER BY raaga").fetchall()
            return [r["raaga"] for r in rows]

    # -- curriculum progress ----------------------------------------------
    def progress(self, unit_id: str) -> UnitProgress:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM curriculum_progress WHERE unit_id=?",
                (unit_id,)).fetchone()
            if row is None:
                return UnitProgress(unit_id=unit_id)
            return UnitProgress(**{k: row[k] for k in row.keys()})

    def save_progress(self, progress: UnitProgress) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO curriculum_progress(unit_id, raaga, status,"
                    " mastery, attempts, failures, last_attempted_at,"
                    " completed_at, notes)"
                    " VALUES (?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(unit_id) DO UPDATE SET raaga=excluded.raaga,"
                    " status=excluded.status, mastery=excluded.mastery,"
                    " attempts=excluded.attempts, failures=excluded.failures,"
                    " last_attempted_at=excluded.last_attempted_at,"
                    " completed_at=excluded.completed_at, notes=excluded.notes",
                    (progress.unit_id, progress.raaga, progress.status,
                     progress.mastery, progress.attempts, progress.failures,
                     progress.last_attempted_at, progress.completed_at,
                     progress.notes))

    def all_progress(self) -> List[UnitProgress]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM curriculum_progress ORDER BY unit_id").fetchall()
            return [UnitProgress(**{k: r[k] for k in r.keys()}) for r in rows]

    def completed_units(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT unit_id FROM curriculum_progress WHERE status='passed'"
            ).fetchall()
            return [r["unit_id"] for r in rows]

    # -- compositions and feedback ----------------------------------------
    def record_composition(self, *, project_id: str, title: str, raaga: str,
                           brief: Dict[str, Any], structure: Dict[str, Any],
                           scores: Dict[str, float], final_score: float,
                           notes: str = "") -> str:
        with self._lock:
            composition_id = new_id("comp")
            with self._conn:
                self._conn.execute(
                    "INSERT INTO compositions(id, at, project_id, title, raaga,"
                    " brief, structure, scores, final_score, notes)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (composition_id, time.time(), project_id, title, raaga,
                     json.dumps(brief), json.dumps(structure), json.dumps(scores),
                     final_score, notes))
            self.log_event("composition.recorded", title, raaga=raaga)
            return composition_id

    def compositions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM compositions ORDER BY at DESC LIMIT ?",
                (limit,)).fetchall()
            out = []
            for row in rows:
                item = dict(row)
                for key in ("brief", "structure", "scores"):
                    try:
                        item[key] = json.loads(item[key])
                    except Exception:  # noqa: BLE001
                        item[key] = {}
                out.append(item)
            return out

    def add_feedback(self, *, target_kind: str, target_id: str, text: str,
                     sentiment: str, raaga: str = "") -> str:
        with self._lock:
            feedback_id = new_id("fb")
            with self._conn:
                self._conn.execute(
                    "INSERT INTO feedback(id, at, target_kind, target_id, raaga,"
                    " text, sentiment, applied) VALUES (?,?,?,?,?,?,?,0)",
                    (feedback_id, time.time(), target_kind, target_id, raaga, text,
                     sentiment))
            self.log_event("feedback.received", f"{sentiment}: {text[:80]}",
                           raaga=raaga)
            return feedback_id

    def feedback(self, limit: int = 50, raaga: str = "") -> List[Dict[str, Any]]:
        with self._lock:
            if raaga:
                rows = self._conn.execute(
                    "SELECT * FROM feedback WHERE raaga=? ORDER BY at DESC LIMIT ?",
                    (raaga, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM feedback ORDER BY at DESC LIMIT ?",
                    (limit,)).fetchall()
            return [dict(r) for r in rows]

    def mark_feedback_applied(self, feedback_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("UPDATE feedback SET applied=1 WHERE id=?",
                                   (feedback_id,))

    # -- lessons (section 38) ----------------------------------------------
    def add_lesson(self, lesson: Lesson) -> Tuple[Lesson, bool]:
        """Store a lesson; the same mistake recurring strengthens it rather
        than duplicating it - that is what stops it being rediscovered."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT * FROM lessons WHERE raaga=? AND unit_id=? AND kind=?"
                " AND applied=0",
                (lesson.raaga, lesson.unit_id, lesson.kind)).fetchone()
            if existing is not None:
                recurrences = existing["recurrences"] + 1
                confidence = max(existing["confidence"], lesson.confidence)
                with self._conn:
                    self._conn.execute(
                        "UPDATE lessons SET last_at=?, attempt=?, result=?,"
                        " failure_reason=?, evidence=?, correction=?,"
                        " source_run=?, confidence=?, recurrences=? WHERE id=?",
                        (lesson.last_at, lesson.attempt, lesson.result,
                         lesson.failure_reason, lesson.evidence, lesson.correction,
                         lesson.source_run, confidence, recurrences,
                         existing["id"]))
                stored = self._row_to_lesson(existing)
                stored.last_at = lesson.last_at
                stored.attempt = lesson.attempt
                stored.result = lesson.result
                stored.failure_reason = lesson.failure_reason
                stored.evidence = lesson.evidence
                stored.correction = lesson.correction
                stored.source_run = lesson.source_run
                stored.confidence = confidence
                stored.recurrences = recurrences
                self.log_event("lesson.recurred",
                               f"{lesson.kind}: {lesson.failure_reason[:80]}",
                               unit_id=lesson.unit_id, raaga=lesson.raaga)
                return stored, False

            with self._conn:
                self._conn.execute(
                    "INSERT INTO lessons(id, at, last_at, raaga, unit_id, attempt,"
                    " task, method, result, kind, dimension, failure_reason,"
                    " evidence, correction, related, source_run, confidence,"
                    " recurrences, applied)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (lesson.id, lesson.at, lesson.last_at, lesson.raaga,
                     lesson.unit_id, lesson.attempt, lesson.task, lesson.method,
                     lesson.result, lesson.kind, lesson.dimension,
                     lesson.failure_reason, lesson.evidence, lesson.correction,
                     json.dumps(lesson.related), lesson.source_run,
                     lesson.confidence, lesson.recurrences, int(lesson.applied)))
            self.log_event("lesson.recorded",
                           f"{lesson.kind}: {lesson.failure_reason[:80]}",
                           unit_id=lesson.unit_id, raaga=lesson.raaga)
            return lesson, True

    def lessons(self, raaga: str = "", unit_id: str = "", kind: str = "",
               min_recurrences: int = 1, include_applied: bool = False,
               limit: int = 200) -> List[Lesson]:
        with self._lock:
            clauses = ["recurrences >= ?"]
            params: List[Any] = [min_recurrences]
            if raaga:
                clauses.append("raaga = ?")
                params.append(raaga)
            if unit_id:
                clauses.append("unit_id = ?")
                params.append(unit_id)
            if kind:
                clauses.append("kind = ?")
                params.append(kind)
            if not include_applied:
                clauses.append("applied = 0")
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT * FROM lessons WHERE {' AND '.join(clauses)}"
                f" ORDER BY recurrences DESC, last_at DESC LIMIT ?",
                params).fetchall()
            return [self._row_to_lesson(r) for r in rows]

    def lesson_counts(self, raaga: str = "") -> Dict[str, int]:
        """Kind -> total recurrences, unapplied lessons only."""
        with self._lock:
            if raaga:
                rows = self._conn.execute(
                    "SELECT kind, SUM(recurrences) AS n FROM lessons"
                    " WHERE applied=0 AND raaga=? GROUP BY kind",
                    (raaga,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT kind, SUM(recurrences) AS n FROM lessons"
                    " WHERE applied=0 GROUP BY kind").fetchall()
            return {r["kind"]: int(r["n"]) for r in rows}

    def mark_lesson_applied(self, lesson_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("UPDATE lessons SET applied=1 WHERE id=?",
                                   (lesson_id,))

    @staticmethod
    def _row_to_lesson(row: sqlite3.Row) -> Lesson:
        return Lesson(
            id=row["id"], at=row["at"], last_at=row["last_at"],
            raaga=row["raaga"], unit_id=row["unit_id"], attempt=row["attempt"],
            task=row["task"], method=row["method"], result=row["result"],
            kind=row["kind"], dimension=row["dimension"],
            failure_reason=row["failure_reason"], evidence=row["evidence"],
            correction=row["correction"],
            related=json.loads(row["related"]) if row["related"] else [],
            source_run=row["source_run"], confidence=row["confidence"],
            recurrences=row["recurrences"], applied=bool(row["applied"]))

    # -- agent state and tasks --------------------------------------------
    def set_state(self, key: str, value: Any) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO agent_state(key, value) VALUES (?, ?)"
                    " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value)))

    def state(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM agent_state WHERE key=?",
                                     (key,)).fetchone()
            if row is None:
                return default
            try:
                return json.loads(row["value"])
            except Exception:  # noqa: BLE001
                return default

    def queue_task(self, kind: str, payload: Dict[str, Any]) -> str:
        with self._lock:
            task_id = new_id("task")
            with self._conn:
                self._conn.execute(
                    "INSERT INTO tasks(id, at, kind, payload, status)"
                    " VALUES (?,?,?,?, 'queued')",
                    (task_id, time.time(), kind, json.dumps(payload)))
            return task_id

    def pending_tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE status='queued' ORDER BY at").fetchall()
            out = []
            for row in rows:
                item = dict(row)
                try:
                    item["payload"] = json.loads(item["payload"])
                except Exception:  # noqa: BLE001
                    item["payload"] = {}
                out.append(item)
            return out

    def finish_task(self, task_id: str, status: str = "done") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("UPDATE tasks SET status=? WHERE id=?",
                                   (status, task_id))

    # -- reporting ---------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            def count(table: str, where: str = "") -> int:
                sql = f"SELECT count(*) AS n FROM {table}"
                if where:
                    sql += f" WHERE {where}"
                return int(self._conn.execute(sql).fetchone()["n"])

            return {
                "schema_version": self.schema_version,
                "path": str(self.path),
                "size_bytes": self.path.stat().st_size if self.path.exists() else 0,
                "sources": count("sources"),
                "sources_analysed": count("sources", "status='analysed'"),
                "phrases": count("phrases", "rejected=0"),
                "phrases_rejected": count("phrases", "rejected=1"),
                "facts": count("raaga_facts"),
                "disputed_facts": count("raaga_facts", "disputed=1"),
                "units_passed": count("curriculum_progress", "status='passed'"),
                "units_attempted": count("curriculum_progress", "attempts>0"),
                "compositions": count("compositions"),
                "feedback": count("feedback"),
                "raagas_known": len(self.known_raagas()),
                "lessons": count("lessons", "applied=0"),
            }
