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

from ..core import provenance
from ..raaga import vocabulary
from ..core.logging_setup import get_logger
from ..core.settings import config_dir

log = get_logger("agent.knowledge")

#: 3 adds ``selection_weights`` (Stage 1 pack document 05 section 6).  The
#: table is created by the same ``IF NOT EXISTS`` script an older database
#: already ran, so an existing knowledge.db gains it and keeps everything.
SCHEMA_VERSION = 5

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
    notes TEXT DEFAULT '',
    origin TEXT DEFAULT 'human');

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
    notes TEXT DEFAULT '',
    -- Who produced this (training specification 2.5).  See core.provenance:
    -- generated material is composable but is never evidence of learning.
    origin TEXT DEFAULT 'human');
CREATE INDEX IF NOT EXISTS phrases_by_origin ON phrases (raaga, origin, rejected);
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

-- Mood words the engine could not read (Arya's specification of
-- 2026-09-06 13:16).  A brief is never held up for one of these; it is
-- recorded here and worked on afterwards.
CREATE TABLE IF NOT EXISTS unresolved_terms (
    term TEXT PRIMARY KEY,
    status TEXT DEFAULT 'pending',
    mapped_to TEXT DEFAULT '[]',
    confidence REAL DEFAULT 0,
    evidence TEXT DEFAULT '',
    origin TEXT DEFAULT 'generated',
    occurrences INTEGER DEFAULT 1,
    attempts INTEGER DEFAULT 0,
    first_seen REAL,
    last_seen REAL,
    note TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS unresolved_by_status
    ON unresolved_terms (status, last_seen DESC);

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

-- What the creator's choices have taught us about which raaga suits which
-- feeling (Stage 1 pack document 05 section 6).  Deliberately a table of its
-- own and deliberately not touching raaga_facts: the pack's rule is "save
-- feedback separately from hard grammar; do not rewrite Arohanam/Avarohanam
-- from preference feedback", and nothing here can, because nothing here is
-- read by anything but the ranking.  ``dimension`` is one of the fourteen
-- emotion dimensions, or '*' for the raaga overall.
CREATE TABLE IF NOT EXISTS selection_weights (
    id TEXT PRIMARY KEY,
    raaga TEXT NOT NULL,
    dimension TEXT NOT NULL,
    weight REAL DEFAULT 0.0,
    observations INTEGER DEFAULT 0,
    at REAL,
    last_at REAL,
    source TEXT DEFAULT '',
    deprecated INTEGER DEFAULT 0);
CREATE UNIQUE INDEX IF NOT EXISTS selection_weights_key
    ON selection_weights (raaga, dimension);
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
    origin: str = provenance.HUMAN


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
    #: Where this came from - see ``core.provenance``.  Defaulting to HUMAN
    #: is safe only because ``add_phrase`` derives the truth from whether a
    #: source was named; nothing else may assume it.
    origin: str = provenance.HUMAN

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


#: A single signal cannot swing a ranking, and no amount of them can turn a
#: raaga into something it is not: the pack's own weights are +1.0 for an
#: acceptance and -0.7 for a rejection, and this is where they saturate.
WEIGHT_LIMIT = 3.0


@dataclass
class SelectionWeight:
    """What the creator's choices taught us about one raaga and one feeling.

    Stage 1 pack document 05 section 6.  Heuristic knowledge in the Agent
    Factory's sense (framework document 04 section 1): defeasible, evidenced
    by a count of observations, reviewable and resettable, and structurally
    incapable of reaching a raaga's notes - the ranking is the only thing
    that reads it.
    """
    id: str = field(default_factory=lambda: new_id("selw"))
    raaga: str = ""
    #: One of the fourteen emotion dimensions, or ``"*"`` for the raaga overall.
    dimension: str = "*"
    weight: float = 0.0
    observations: int = 0
    at: float = field(default_factory=time.time)
    last_at: float = field(default_factory=time.time)
    source: str = ""
    deprecated: bool = False

    def describe(self) -> str:
        direction = "prefers" if self.weight > 0 else "avoids"
        where = "generally" if self.dimension == "*" else f"for {self.dimension}"
        return (f"{direction} {self.raaga} {where} "
                f"({self.weight:+.2f} from {self.observations} "
                f"observation{'s' if self.observations != 1 else ''})")


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
            # Before the schema script, not after.  The script now creates an
            # index over ``phrases.origin``, and on a database written before
            # that column existed the index cannot be built - CREATE INDEX IF
            # NOT EXISTS still has to resolve the column it names.
            self._add_origin_columns()
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
                    if stored < 4:
                        self._backfill_origins()
                    # Schema 5 adds unresolved_terms, which the schema script
                    # creates on its own.  Nothing to migrate; said out loud
                    # because an empty branch here otherwise looks forgotten.
                    self._conn.execute(
                        "UPDATE meta SET value=? WHERE key='schema_version'",
                        (str(SCHEMA_VERSION),))

    def _add_origin_columns(self) -> None:
        """Schema 4: provenance becomes a column instead of a convention.

        ``executescript(_SCHEMA)`` cannot do this - every statement there is
        CREATE TABLE IF NOT EXISTS, which does nothing at all to a table that
        already exists, so a new column in the schema text reaches new
        databases only.  Existing ones need the ALTER.

        An empty ``table_info`` means the table is not there yet, which is a
        fresh database: the schema script is about to create it with the
        column already in place, so there is nothing to alter.
        """
        for table in ("sources", "phrases"):
            columns = {r["name"] for r in
                       self._conn.execute(f"PRAGMA table_info({table})")}
            if columns and "origin" not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN origin TEXT DEFAULT 'human'")
                log.info("added %s.origin", table)

    def _backfill_origins(self) -> None:
        """Classify what was stored before anything recorded an origin.

        Derived, not guessed: a phrase learned from a source carries that
        source's id, so a phrase with no source was not learned from one.
        That is exactly the material the agent wrote itself - practice
        output kept by ``_keep_best_artifact`` - and it is the only thing
        this reclassifies.
        """
        moved = self._conn.execute(
            "UPDATE phrases SET origin=? WHERE source_id IS NULL OR source_id=''",
            (provenance.GENERATED,)).rowcount
        if moved:
            log.info("provenance backfill: %d phrase(s) named no source and "
                     "are recorded as generated", moved)

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
                    " notes, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (source.id, source.locator, source.title, source.performer,
                     source.raaga, source.content_type, source.rights_status,
                     source.provider, source.quality, source.ingested_at,
                     source.extraction_version, source.confidence, source.status,
                     source.error, source.fingerprint, source.notes,
                     source.origin))
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

    def forget_source(self, source_id: str) -> int:
        """Remove a source and everything derived from it.

        The agent could learn but never unlearn, and that is untenable once
        the code that does the hearing changes: a phrase is not knowledge
        about a raaga, it is knowledge about a raaga *as heard by a
        particular version of the ears*.  When those change, what they
        produced has to be re-derived, and re-deriving means first removing
        what the old version left behind.

        Only what came from this recording goes.  The recording itself is
        untouched - it is on disk, it is the thing of record, and it can be
        listened to again.  Returns how many rows were removed, the source
        row included.
        """
        with self._lock:
            removed = 0
            for table in ("phrases", "raaga_facts"):
                cur = self._conn.execute(
                    f"DELETE FROM {table} WHERE source_id=?", (source_id,))
                removed += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cur = self._conn.execute("DELETE FROM sources WHERE id=?",
                                     (source_id,))
            removed += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            self._conn.commit()
        log.info("forgot source %s and %d derived row(s)", source_id, removed)
        return removed

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
        """Store a phrase; an identical one strengthens the existing entry.

        Two provenance rules live here (training specification 2.4).

        The origin is settled rather than trusted: a phrase that names no
        source cannot have been learned from one, so it is recorded as
        generated whatever it claimed.

        And a generated phrase may not strengthen a learned one.  Repeating
        yourself is not a second witness; letting it through was a quiet
        route by which the agent's own output raised the confidence of the
        material it was supposed to be judged against.
        """
        with self._lock:
            phrase.origin = provenance.coerce(phrase.origin,
                                              source_id=phrase.source_id)
            phrase.fingerprint = phrase.fingerprint or phrase.compute_fingerprint()
            existing = self._conn.execute(
                "SELECT * FROM phrases WHERE fingerprint=?",
                (phrase.fingerprint,)).fetchone()
            if existing is not None:
                stored = self._row_to_phrase(existing)
                # Any origin that cannot be learned from must not be able to
                # corroborate one that can.  This named GENERATED
                # specifically, which was exactly wide enough until UNKNOWN
                # arrived: an unmapped provider's phrase then matched a real
                # recording by fingerprint and strengthened it, so
                # non-trainable material raised trainable evidence by the
                # side door.  Asking may_be_learned_from covers every
                # non-trainable origin, including any added later.
                if not provenance.may_be_learned_from(phrase.origin) \
                        and provenance.may_be_learned_from(stored.origin):
                    log.debug("%s material matched learned %s; not "
                              "strengthening it", phrase.origin, stored.id)
                    return stored, False
                votes = existing["votes"] + 1
                confidence = min(0.99, max(existing["confidence"],
                                           phrase.confidence) + 0.05)
                # The promotion the rule implies in the other direction: if a
                # real recording turns out to contain something the agent had
                # only invented, that is a genuine witness and the phrase has
                # now been heard.  Without this the row stayed marked
                # generated for ever and the evidence was thrown away.
                origin, source_id = stored.origin, stored.source_id
                if provenance.may_be_learned_from(phrase.origin) \
                        and not provenance.may_be_learned_from(stored.origin):
                    origin, source_id = phrase.origin, phrase.source_id
                    log.info("phrase %s was generated and has now been heard "
                             "in %s", stored.id, source_id or "a source")
                with self._conn:
                    self._conn.execute(
                        "UPDATE phrases SET votes=?, confidence=?, origin=?,"
                        " source_id=? WHERE id=?",
                        (votes, confidence, origin, source_id, existing["id"]))
                stored.votes = votes
                stored.confidence = confidence
                stored.origin = origin
                stored.source_id = source_id
                return stored, False

            with self._conn:
                self._conn.execute(
                    "INSERT INTO phrases(id, raaga, swaras, midi, durations,"
                    " function, source_id, confidence, fingerprint, contour, tempo,"
                    " votes, rejected, learned_at, notes, origin)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (phrase.id, phrase.raaga, json.dumps(phrase.swaras),
                     json.dumps(phrase.midi), json.dumps(phrase.durations),
                     phrase.function, phrase.source_id, phrase.confidence,
                     phrase.fingerprint, phrase.contour, phrase.tempo, phrase.votes,
                     int(phrase.rejected), phrase.learned_at, phrase.notes,
                     phrase.origin))
            return phrase, True

    def phrases(self, raaga: str = "", min_confidence: float = 0.0,
                include_rejected: bool = False, limit: int = 500,
                function: str = "",
                origins: Optional[Sequence[str]] = None) -> List[Phrase]:
        """Every phrase, whatever produced it.

        This is the raw pool, and it is the right thing to read when the
        question is "what does this system have" - composing, answering the
        creator, indexing for originality, applying feedback.

        It is the *wrong* thing to read when the question is "what has this
        system learned".  Use ``learned_phrases`` for that; see
        ``core.provenance``.
        """
        with self._lock:
            clauses = ["confidence >= ?"]
            params: List[Any] = [min_confidence]
            if raaga:
                clauses.append("raaga = ?")
                params.append(raaga)
            if origins:
                clauses.append("origin IN (%s)" % ",".join("?" * len(origins)))
                params.extend(origins)
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

    def learned_phrases(self, raaga: str = "", min_confidence: float = 0.0,
                        include_rejected: bool = False, limit: int = 500,
                        function: str = "") -> List[Phrase]:
        """Only what was learned from somebody else's music.

        Training specification 2.4 and non-negotiable rule 2: the agent's
        own output is creative material, not evidence.  Everything that
        asks "is there enough to learn from", "what shall I practise from"
        or "what shall I be quizzed on" belongs here rather than in
        ``phrases``.
        """
        return self.phrases(raaga=raaga, min_confidence=min_confidence,
                            include_rejected=include_rejected, limit=limit,
                            function=function,
                            origins=provenance.LEARNED_FROM)

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

    def count_phrases(self, raaga: str = "", learned_only: bool = False) -> int:
        with self._lock:
            clauses = ["rejected = 0"]
            params: List[Any] = []
            if raaga:
                clauses.append("raaga = ?")
                params.append(raaga)
            if learned_only:
                clauses.append("origin IN (%s)"
                               % ",".join("?" * len(provenance.LEARNED_FROM)))
                params.extend(provenance.LEARNED_FROM)
            row = self._conn.execute(
                f"SELECT count(*) AS n FROM phrases WHERE {' AND '.join(clauses)}",
                params).fetchone()
            return int(row["n"])

    def count_phrases_by_origin(self, raaga: str = "",
                                learned_only: bool = True) -> Dict[str, int]:
        """How many phrases came from each origin, counted in the database.

        Counting by loading rows and tallying them in Python is only right
        while the rows fit under whatever limit the caller passed, and the
        answers that used it reported a display limit as a total.  One
        GROUP BY is bounded by the number of origins, not by the number of
        phrases.
        """
        with self._lock:
            clauses = ["rejected = 0"]
            params: List[Any] = []
            if raaga:
                clauses.append("raaga = ?")
                params.append(raaga)
            if learned_only:
                clauses.append("origin IN (%s)"
                               % ",".join("?" * len(provenance.LEARNED_FROM)))
                params.extend(provenance.LEARNED_FROM)
            rows = self._conn.execute(
                f"SELECT origin, count(*) AS n FROM phrases"
                f" WHERE {' AND '.join(clauses)} GROUP BY origin",
                params).fetchall()
            return {r["origin"]: int(r["n"]) for r in rows}

    def count_sources_by_status(self, raaga: str = ""
                                ) -> Dict[Tuple[str, str], int]:
        """Sources per (status, origin), counted in the database.

        Status matters to every answer about training: a source that is
        registered and still pending, or one whose analysis failed, is
        something the agent has available and not something it has learned
        from.  Keeping the two apart needs the status beside the origin.
        """
        with self._lock:
            if raaga:
                rows = self._conn.execute(
                    "SELECT status, origin, count(*) AS n FROM sources"
                    " WHERE raaga = ? GROUP BY status, origin",
                    (raaga,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT status, origin, count(*) AS n FROM sources"
                    " GROUP BY status, origin").fetchall()
            return {(r["status"], r["origin"]): int(r["n"]) for r in rows}

    @staticmethod
    def _row_to_phrase(row: sqlite3.Row) -> Phrase:
        return Phrase(
            id=row["id"], raaga=row["raaga"], swaras=json.loads(row["swaras"]),
            midi=json.loads(row["midi"]), durations=json.loads(row["durations"]),
            function=row["function"], source_id=row["source_id"],
            confidence=row["confidence"], fingerprint=row["fingerprint"],
            contour=row["contour"], tempo=row["tempo"], votes=row["votes"],
            rejected=bool(row["rejected"]), learned_at=row["learned_at"],
            notes=row["notes"], origin=row["origin"])

    # -- unresolved mood terms ---------------------------------------------
    def note_unknown_terms(self, terms: Sequence[str]) -> List[str]:
        """Record words a brief used that the engine could not read.

        Returns the terms that are still outstanding, so a caller can say
        honestly which parts of the request did not reach the ranking.  A
        term already resolved is not outstanding and is not reported.

        Repeated sightings share one investigation (the specification's
        "repeated occurrences can share an outstanding investigation") -
        the row is the same row, with its count and last-seen moved on.
        """
        outstanding: List[str] = []
        now = time.time()
        with self._lock:
            with self._conn:
                for raw in terms:
                    term = str(raw).strip().lower()
                    if not term:
                        continue
                    row = self._conn.execute(
                        "SELECT status FROM unresolved_terms WHERE term=?",
                        (term,)).fetchone()
                    if row is None:
                        self._conn.execute(
                            "INSERT INTO unresolved_terms(term, status,"
                            " first_seen, last_seen) VALUES (?,?,?,?)",
                            (term, vocabulary.PENDING, now, now))
                        outstanding.append(term)
                        continue
                    self._conn.execute(
                        "UPDATE unresolved_terms SET occurrences=occurrences+1,"
                        " last_seen=? WHERE term=?", (now, term))
                    # Resolved means understood; dismissed means judged and
                    # rejected.  Neither is outstanding - reporting a
                    # dismissed word as "still working out" would promise
                    # work that is deliberately not happening.
                    if row["status"] not in (vocabulary.RESOLVED,
                                             vocabulary.DISMISSED):
                        outstanding.append(term)
        if outstanding:
            log.info("brief used %d term(s) the engine cannot read: %s",
                     len(outstanding), ", ".join(outstanding))
        return outstanding

    def unresolved_terms(self, status: str = "", limit: int = 200
                         ) -> List[vocabulary.UnresolvedTerm]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM unresolved_terms WHERE status=?"
                    " ORDER BY last_seen DESC LIMIT ?",
                    (status, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM unresolved_terms"
                    " ORDER BY last_seen DESC LIMIT ?", (limit,)).fetchall()
            return [self._row_to_term(r) for r in rows]

    def unresolved_term(self, term: str
                        ) -> Optional[vocabulary.UnresolvedTerm]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM unresolved_terms WHERE term=?",
                (str(term).strip().lower(),)).fetchone()
            return self._row_to_term(row) if row else None

    def resolutions(self) -> Dict[str, List[str]]:
        """Every settled meaning, as a substitution table.

        This is what makes a resolution *do* something: the words go into
        the brief's text before it is scored, so every reader benefits at
        once without knowing this table exists.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT term, mapped_to FROM unresolved_terms"
                " WHERE status=?", (vocabulary.RESOLVED,)).fetchall()
        out: Dict[str, List[str]] = {}
        for row in rows:
            words = json.loads(row["mapped_to"] or "[]")
            if words:
                out[row["term"]] = words
        return out

    def dismiss_term(self, term: str, note: str = "") -> None:
        """Reject a meaning, and stop the system proposing it again.

        The reversal half of "evaluated, corrected, or reversed".  A
        dismissed term keeps its row - the history is the point - but drops
        out of ``resolutions``, so nothing it was influencing is influenced
        any more, and ``record_investigation`` will not quietly revive it.
        """
        term = str(term).strip().lower()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE unresolved_terms SET status=?, mapped_to='[]',"
                    " confidence=0, note=? WHERE term=?",
                    (vocabulary.DISMISSED, note, term))
        log.info("dismissed the reading of %r%s", term,
                 f": {note}" if note else "")

    def unconfirmed_readings(self) -> List[Tuple[str, List[str]]]:
        """Resolved meanings that no person has confirmed.

        Kept apart from ``resolutions`` because the two answer different
        questions: that one is "what should the engine read", this one is
        "what is the creator entitled to be told about".
        """
        out: List[Tuple[str, List[str]]] = []
        for term in self.unresolved_terms(status=vocabulary.RESOLVED,
                                          limit=500):
            if term.mapped_to and not provenance.may_be_learned_from(term.origin):
                out.append((term.term, list(term.mapped_to)))
        return out

    def set_term_status(self, term: str, status: str, *, note: str = "") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE unresolved_terms SET status=?, note=? WHERE term=?",
                    (status, note, str(term).strip().lower()))

    def record_investigation(self, term: str, mapped_to: Sequence[str],
                             confidence: float, evidence: str,
                             origin: str = provenance.GENERATED) -> str:
        """Store the outcome of one attempt at an unknown word.

        A mapping that named nothing usable is a failed attempt, not a
        resolution; after ``MAX_ATTEMPTS`` of those the term stops being
        retried and starts waiting for a person.
        """
        term = str(term).strip().lower()
        with self._lock:
            row = self._conn.execute(
                "SELECT attempts, status FROM unresolved_terms WHERE term=?",
                (term,)).fetchone()
            # A dismissed reading stays dismissed.  Without this the
            # investigator could propose the same rejected meaning on its
            # next sweep and quietly undo the rejection.
            if row is not None and row["status"] == vocabulary.DISMISSED:
                log.debug("%r was dismissed; not re-proposing", term)
                return vocabulary.DISMISSED
            attempts = int(row["attempts"] if row else 0) + 1
            words = list(mapped_to or ())
            if words:
                status = vocabulary.RESOLVED
            elif attempts >= vocabulary.MAX_ATTEMPTS:
                status = vocabulary.NEEDS_USER_INPUT
            else:
                status = vocabulary.PENDING
            with self._conn:
                self._conn.execute(
                    "UPDATE unresolved_terms SET status=?, mapped_to=?,"
                    " confidence=?, evidence=?, origin=?, attempts=?"
                    " WHERE term=?",
                    (status, json.dumps(words), float(confidence),
                     str(evidence)[:500], origin, attempts, term))
        log.info("investigated %r: %s%s", term, status,
                 f" -> {', '.join(words)}" if words else "")
        return status

    @staticmethod
    def _row_to_term(row: sqlite3.Row) -> vocabulary.UnresolvedTerm:
        return vocabulary.UnresolvedTerm(
            term=row["term"], status=row["status"],
            mapped_to=json.loads(row["mapped_to"] or "[]"),
            confidence=row["confidence"], evidence=row["evidence"],
            origin=row["origin"], occurrences=row["occurrences"],
            attempts=row["attempts"], first_seen=row["first_seen"],
            last_seen=row["last_seen"], note=row["note"])

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

    def overrule_facts(self, raaga: str, key: str, keep_value: str,
                       confidence: float = 0.3, note: str = "") -> int:
        """A ruling settled a disputed key: every other claim for it drops
        to ``confidence`` so it stops being used, but stays on record with
        the reason (document 04 section 6: corrected, not silently
        accumulated).  Returns how many claims were overruled."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, value, confidence, notes FROM raaga_facts"
                " WHERE raaga=? AND key=? AND value<>?",
                (raaga, key, keep_value)).fetchall()
            for row in rows:
                with self._conn:
                    self._conn.execute(
                        "UPDATE raaga_facts SET confidence=?, notes=? WHERE id=?",
                        (min(row["confidence"], confidence),
                         (f"{row['notes']}; " if row["notes"] else "")
                         + (note or "overruled"), row["id"]))
            if rows:
                self.log_event("fact.overruled",
                               f"{key}: {len(rows)} claim(s) give way to "
                               f"{keep_value[:40]}", raaga=raaga)
            return len(rows)

    def overrule_fact(self, raaga: str, key: str, value: str,
                      confidence: float = 0.3, note: str = "") -> bool:
        """Overrule one specific claim (see ``overrule_facts``)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, confidence, notes FROM raaga_facts"
                " WHERE raaga=? AND key=? AND value=?",
                (raaga, key, value)).fetchone()
            if row is None:
                return False
            with self._conn:
                self._conn.execute(
                    "UPDATE raaga_facts SET confidence=?, notes=? WHERE id=?",
                    (min(row["confidence"], confidence),
                     (f"{row['notes']}; " if row["notes"] else "")
                     + (note or "overruled"), row["id"]))
            self.log_event("fact.overruled", f"{key}={value[:40]}", raaga=raaga)
            return True

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
    # ==================================================================
    # selection feedback (Stage 1 pack document 05 section 6)
    # ==================================================================
    def record_selection_feedback(self, raaga: str, signal: float,
                                  dimensions: Optional[Dict[str, float]] = None,
                                  source: str = "") -> List[SelectionWeight]:
        """Fold one choice into what is known about this raaga's suitability.

        ``signal`` is the pack's own number - +1.0 accepted, +0.2 auditioned,
        -0.7 rejected - and ``dimensions`` is what the brief was actually
        asking for, so the lesson is learned *in context*: rejecting a raaga
        for a joyful brief must not sink it for a grieving one.  The overall
        ``"*"`` weight carries the part of the signal that is about the raaga
        however it was asked for.

        Nothing here touches ``raaga_facts``, ``phrases`` or anything else
        the library or the analysis pipeline writes.  That is the pack's rule
        and it is enforced by this method having no way to reach them.
        """
        if not raaga or not signal:
            return []
        now = time.time()
        shares: Dict[str, float] = {"*": signal * 0.4}
        for dimension, strength in (dimensions or {}).items():
            if strength >= 0.25:
                shares[dimension] = signal * 0.6 * strength

        stored: List[SelectionWeight] = []
        with self._lock, self._conn:
            for dimension, delta in shares.items():
                row = self._conn.execute(
                    "SELECT * FROM selection_weights WHERE raaga=? AND dimension=?",
                    (raaga, dimension)).fetchone()
                if row is None:
                    weight = SelectionWeight(
                        raaga=raaga, dimension=dimension,
                        weight=max(-WEIGHT_LIMIT, min(WEIGHT_LIMIT, delta)),
                        observations=1, at=now, last_at=now, source=source)
                    self._conn.execute(
                        "INSERT INTO selection_weights(id, raaga, dimension,"
                        " weight, observations, at, last_at, source, deprecated)"
                        " VALUES (?,?,?,?,?,?,?,?,0)",
                        (weight.id, weight.raaga, weight.dimension,
                         weight.weight, weight.observations, weight.at,
                         weight.last_at, weight.source))
                else:
                    total = max(-WEIGHT_LIMIT,
                                min(WEIGHT_LIMIT, row["weight"] + delta))
                    self._conn.execute(
                        "UPDATE selection_weights SET weight=?, observations=?,"
                        " last_at=?, source=?, deprecated=0 WHERE id=?",
                        (total, row["observations"] + 1, now, source, row["id"]))
                    weight = SelectionWeight(
                        id=row["id"], raaga=raaga, dimension=dimension,
                        weight=total, observations=row["observations"] + 1,
                        at=row["at"], last_at=now, source=source)
                stored.append(weight)
        self.log_event("selection.feedback",
                       f"{source or 'feedback'} {signal:+.1f} for {raaga}",
                       raaga=raaga)
        return stored

    def selection_weights(self, raaga: str = "") -> List[SelectionWeight]:
        """Everything learned about raaga selection, most recent first."""
        with self._lock:
            if raaga:
                rows = self._conn.execute(
                    "SELECT * FROM selection_weights WHERE raaga=? AND"
                    " deprecated=0 ORDER BY last_at DESC", (raaga,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM selection_weights WHERE deprecated=0"
                    " ORDER BY last_at DESC").fetchall()
        return [SelectionWeight(
            id=r["id"], raaga=r["raaga"], dimension=r["dimension"],
            weight=r["weight"], observations=r["observations"], at=r["at"],
            last_at=r["last_at"], source=r["source"],
            deprecated=bool(r["deprecated"])) for r in rows]

    def selection_weight_map(self) -> Dict[str, Dict[str, float]]:
        """``{raaga: {dimension: weight}}`` - one read for a whole ranking."""
        out: Dict[str, Dict[str, float]] = {}
        for weight in self.selection_weights():
            out.setdefault(weight.raaga, {})[weight.dimension] = weight.weight
        return out

    def reset_selection_weights(self, raaga: str = "") -> int:
        """Forget what was learned, for one raaga or for all of them.

        Deprecation rather than deletion (framework document 04 section 6):
        the row stays, so a creator can see that a preference was held and
        withdrawn rather than finding a gap where an explanation used to be.
        """
        with self._lock, self._conn:
            if raaga:
                cursor = self._conn.execute(
                    "UPDATE selection_weights SET deprecated=1 WHERE raaga=?"
                    " AND deprecated=0", (raaga,))
            else:
                cursor = self._conn.execute(
                    "UPDATE selection_weights SET deprecated=1 WHERE deprecated=0")
            count = cursor.rowcount
        self.log_event("selection.reset",
                       f"{count} learned selection weight(s) withdrawn"
                       + (f" for {raaga}" if raaga else ""), raaga=raaga)
        return count

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
                # Related knowledge accumulates: a phrase copied on the
                # second attempt is as much to avoid as the first one.
                related = json.loads(existing["related"]) if existing["related"] else []
                related += [r for r in lesson.related if r not in related]
                with self._conn:
                    self._conn.execute(
                        "UPDATE lessons SET last_at=?, attempt=?, result=?,"
                        " failure_reason=?, evidence=?, correction=?,"
                        " source_run=?, confidence=?, recurrences=?, related=?"
                        " WHERE id=?",
                        (lesson.last_at, lesson.attempt, lesson.result,
                         lesson.failure_reason, lesson.evidence, lesson.correction,
                         lesson.source_run, confidence, recurrences,
                         json.dumps(related), existing["id"]))
                stored = self._row_to_lesson(existing)
                stored.related = related
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
