"""The Agent Factory's durable store - document 04's SHARED KNOWLEDGE BASE
and TRAINING HISTORY in one sqlite file.

Follows the same shape as ``agent/knowledge.py``: one file, one connection
shared across threads, ``self._lock`` (an ``RLock``) held for the whole body
of every public method so no two statements run on the connection at once,
and a schema version in a ``meta`` table so a database written by a newer
build refuses to open under an older one rather than silently misreading it.

Enums are stored as their ``.value`` (an int for the ladders, a string for
the rest); lists and dicts are stored as JSON text; an ``AgentSpec`` travels
inside its profile row as JSON.  AGENT MEMORY - the agent's own history and
preferences - stays in ``agent/knowledge.py``; this file is what is shared
and reusable across agents.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging_setup import get_logger
from ..core.settings import config_dir
from .models import (AgentProfile, AgentSpec, Dispute, DisputeStatus,
                     KnowledgeClass, Lesson, MasteryLevel, MasteryRecord,
                     Maturity, Promotion, Reiteration, ReiterationCheck,
                     ReusableLesson, Ruling, Split, TestLevel, TestResult,
                     TestSpec, ValidationStatus, new_id)

log = get_logger("factory.store")

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS profiles (
    id TEXT PRIMARY KEY,
    name TEXT,
    role TEXT,
    domain TEXT,
    capabilities TEXT,
    maturity INTEGER DEFAULT 0,
    current_curriculum TEXT,
    current_lesson_id TEXT,
    strengths TEXT,
    weaknesses TEXT,
    knowledge_version TEXT,
    spec TEXT,
    created_at REAL,
    updated_at REAL);
CREATE INDEX IF NOT EXISTS profiles_by_domain ON profiles (domain);

CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    domain TEXT,
    concept TEXT,
    objective TEXT,
    prerequisites TEXT,
    source_knowledge TEXT,
    explanation TEXT,
    examples TEXT,
    counterexamples TEXT,
    practice_tasks TEXT,
    test_tasks TEXT,
    expected_behavior TEXT,
    common_errors TEXT,
    remediation TEXT,
    knowledge_class TEXT,
    confidence REAL DEFAULT 1.0,
    version INTEGER DEFAULT 1,
    origin TEXT,
    payload TEXT);
CREATE INDEX IF NOT EXISTS lessons_by_domain ON lessons (domain, concept);

CREATE TABLE IF NOT EXISTS reiterations (
    id TEXT PRIMARY KEY,
    lesson_id TEXT,
    agent_id TEXT,
    restate TEXT,
    explain TEXT,
    connect TEXT,
    example TEXT,
    counterexample TEXT,
    apply_summary TEXT,
    apply_score REAL DEFAULT 0.0,
    self_check TEXT,
    retest_due_at REAL DEFAULT 0.0,
    at REAL,
    check_json TEXT);
CREATE INDEX IF NOT EXISTS reiterations_by_agent ON reiterations (agent_id, lesson_id);

CREATE TABLE IF NOT EXISTS tests (
    id TEXT PRIMARY KEY,
    capability TEXT,
    level INTEGER DEFAULT 1,
    novelty REAL DEFAULT 0.0,
    difficulty REAL DEFAULT 0.5,
    ambiguity REAL DEFAULT 0.0,
    objective INTEGER DEFAULT 1,
    expected TEXT,
    acceptable_range TEXT,
    failure_mode_targeted TEXT,
    split TEXT,
    seed INTEGER DEFAULT 0,
    retired INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,
    author_agent_id TEXT,
    lesson_id TEXT,
    payload TEXT,
    created_at REAL);
CREATE INDEX IF NOT EXISTS tests_by_capability ON tests (capability, level, split);

CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    test_id TEXT,
    agent_id TEXT,
    lesson_id TEXT,
    level INTEGER DEFAULT 1,
    split TEXT,
    score REAL DEFAULT 0.0,
    passed INTEGER DEFAULT 0,
    student_claim TEXT,
    student_confidence REAL DEFAULT 0.0,
    trainer_claim TEXT,
    trainer_confidence REAL DEFAULT 0.0,
    failure_mode TEXT,
    evidence TEXT,
    duration_seconds REAL DEFAULT 0.0,
    judge_needed INTEGER DEFAULT 0,
    dispute_id TEXT,
    at REAL);
CREATE INDEX IF NOT EXISTS results_by_agent ON results (agent_id, at DESC);
CREATE INDEX IF NOT EXISTS results_by_test ON results (test_id);

CREATE TABLE IF NOT EXISTS disputes (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    test_id TEXT,
    lesson_id TEXT,
    question TEXT,
    student_claim TEXT,
    trainer_claim TEXT,
    evidence_student TEXT,
    evidence_trainer TEXT,
    student_confidence REAL DEFAULT 0.0,
    trainer_confidence REAL DEFAULT 0.0,
    shared_knowledge TEXT,
    applicable_rules TEXT,
    status TEXT,
    ruling_id TEXT,
    critical INTEGER DEFAULT 0,
    at REAL);
CREATE INDEX IF NOT EXISTS disputes_by_agent ON disputes (agent_id, status);

CREATE TABLE IF NOT EXISTS rulings (
    id TEXT PRIMARY KEY,
    dispute_id TEXT,
    ruling TEXT,
    accepted_claim TEXT,
    rejected_claim TEXT,
    rationale TEXT,
    confidence REAL DEFAULT 0.0,
    unresolved_issues TEXT,
    correction_student TEXT,
    correction_trainer TEXT,
    reusable_lesson_id TEXT,
    needs_external_evidence INTEGER DEFAULT 0,
    decided_by TEXT,
    at REAL);

CREATE TABLE IF NOT EXISTS reusable_lessons (
    id TEXT PRIMARY KEY,
    source_event TEXT,
    rule_or_procedure TEXT,
    knowledge_class TEXT,
    confidence REAL DEFAULT 0.5,
    validation_status TEXT,
    scope_domain TEXT,
    scope_concept TEXT,
    source_agent_id TEXT,
    version INTEGER DEFAULT 1,
    validations INTEGER DEFAULT 0,
    last_validated_at REAL DEFAULT 0.0,
    superseded_by TEXT,
    deprecated INTEGER DEFAULT 0,
    created_at REAL);
CREATE INDEX IF NOT EXISTS reusable_by_scope ON reusable_lessons (scope_domain, scope_concept, validation_status);

CREATE TABLE IF NOT EXISTS mastery (
    agent_id TEXT,
    concept TEXT,
    level INTEGER DEFAULT 0,
    evidence TEXT,
    failures_at_level INTEGER DEFAULT 0,
    updated_at REAL,
    PRIMARY KEY (agent_id, concept));

CREATE TABLE IF NOT EXISTS promotions (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    from_maturity INTEGER,
    to_maturity INTEGER,
    evidence TEXT,
    at REAL);
CREATE INDEX IF NOT EXISTS promotions_by_agent ON promotions (agent_id, at DESC);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    value REAL,
    agent_id TEXT,
    detail TEXT,
    at REAL);
CREATE INDEX IF NOT EXISTS metrics_by_name ON metrics (name, at DESC);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value)


def _loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return default


class FactoryStore:
    """The shared knowledge base and training history.

    Every public method holds ``self._lock`` for its whole body: rows are
    always materialised before the lock is released, exactly as in
    ``agent/knowledge.py``.
    """

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else config_dir() / "factory.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._closed = False
        self._migrate()

    # -- lifecycle -----------------------------------------------------
    def _migrate(self) -> None:
        # Called from __init__, before this object can have reached any
        # other thread - nothing to serialise against yet.
        with self._conn:
            self._conn.executescript(_SCHEMA)
            found = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if found is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES ('schema_version', ?)",
                    (str(self.SCHEMA_VERSION),))
                log.info("factory store created at %s", self.path)
            else:
                stored = int(found["value"])
                if stored > self.SCHEMA_VERSION:
                    raise RuntimeError(
                        f"factory.db was written by a newer version "
                        f"(schema {stored} > {self.SCHEMA_VERSION})")
                if stored < self.SCHEMA_VERSION:
                    log.info("migrating factory schema %d -> %d", stored,
                             self.SCHEMA_VERSION)
                    self._conn.execute(
                        "UPDATE meta SET value=? WHERE key='schema_version'",
                        (str(self.SCHEMA_VERSION),))

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

    # -- profiles --------------------------------------------------------
    def save_profile(self, profile: AgentProfile) -> None:
        with self._lock:
            spec_json = _dumps(_spec_to_dict(profile.spec)) if profile.spec else ""
            with self._conn:
                self._conn.execute(
                    "INSERT INTO profiles(id, name, role, domain, capabilities,"
                    " maturity, current_curriculum, current_lesson_id, strengths,"
                    " weaknesses, knowledge_version, spec, created_at, updated_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET name=excluded.name,"
                    " role=excluded.role, domain=excluded.domain,"
                    " capabilities=excluded.capabilities, maturity=excluded.maturity,"
                    " current_curriculum=excluded.current_curriculum,"
                    " current_lesson_id=excluded.current_lesson_id,"
                    " strengths=excluded.strengths, weaknesses=excluded.weaknesses,"
                    " knowledge_version=excluded.knowledge_version,"
                    " spec=excluded.spec, updated_at=excluded.updated_at",
                    (profile.id, profile.name, profile.role, profile.domain,
                     _dumps(profile.capabilities), int(profile.maturity),
                     profile.current_curriculum, profile.current_lesson_id,
                     _dumps(profile.strengths), _dumps(profile.weaknesses),
                     profile.knowledge_version, spec_json, profile.created_at,
                     profile.updated_at))

    def profile(self, agent_id: str) -> Optional[AgentProfile]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM profiles WHERE id=?",
                                     (agent_id,)).fetchone()
            return _row_to_profile(row) if row else None

    def profiles(self, domain: str = "") -> List[AgentProfile]:
        with self._lock:
            if domain:
                rows = self._conn.execute(
                    "SELECT * FROM profiles WHERE domain=? ORDER BY created_at",
                    (domain,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM profiles ORDER BY created_at").fetchall()
            return [_row_to_profile(r) for r in rows]

    # -- lessons -----------------------------------------------------------
    def save_lesson(self, lesson: Lesson) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO lessons(id, domain, concept, objective,"
                    " prerequisites, source_knowledge, explanation, examples,"
                    " counterexamples, practice_tasks, test_tasks,"
                    " expected_behavior, common_errors, remediation,"
                    " knowledge_class, confidence, version, origin, payload)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET domain=excluded.domain,"
                    " concept=excluded.concept, objective=excluded.objective,"
                    " prerequisites=excluded.prerequisites,"
                    " source_knowledge=excluded.source_knowledge,"
                    " explanation=excluded.explanation, examples=excluded.examples,"
                    " counterexamples=excluded.counterexamples,"
                    " practice_tasks=excluded.practice_tasks,"
                    " test_tasks=excluded.test_tasks,"
                    " expected_behavior=excluded.expected_behavior,"
                    " common_errors=excluded.common_errors,"
                    " remediation=excluded.remediation,"
                    " knowledge_class=excluded.knowledge_class,"
                    " confidence=excluded.confidence, version=excluded.version,"
                    " origin=excluded.origin, payload=excluded.payload",
                    (lesson.id, lesson.domain, lesson.concept, lesson.objective,
                     _dumps(lesson.prerequisites), _dumps(lesson.source_knowledge),
                     lesson.explanation, _dumps(lesson.examples),
                     _dumps(lesson.counterexamples), _dumps(lesson.practice_tasks),
                     _dumps(lesson.test_tasks), lesson.expected_behavior,
                     _dumps(lesson.common_errors), _dumps(lesson.remediation),
                     lesson.knowledge_class.value, lesson.confidence,
                     lesson.version, lesson.origin, _dumps(lesson.payload)))

    def lesson(self, lesson_id: str) -> Optional[Lesson]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM lessons WHERE id=?",
                                     (lesson_id,)).fetchone()
            return _row_to_lesson(row) if row else None

    def lessons(self, domain: str = "", concept: str = "") -> List[Lesson]:
        with self._lock:
            clauses, params = [], []
            if domain:
                clauses.append("domain=?")
                params.append(domain)
            if concept:
                clauses.append("concept=?")
                params.append(concept)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = self._conn.execute(
                f"SELECT * FROM lessons{where} ORDER BY id", params).fetchall()
            return [_row_to_lesson(r) for r in rows]

    def save_reiteration(self, r: Reiteration, check: ReiterationCheck) -> str:
        with self._lock:
            row_id = new_id("reiter")
            with self._conn:
                self._conn.execute(
                    "INSERT INTO reiterations(id, lesson_id, agent_id, restate,"
                    " explain, connect, example, counterexample, apply_summary,"
                    " apply_score, self_check, retest_due_at, at, check_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row_id, r.lesson_id, r.agent_id, r.restate, r.explain,
                     r.connect, r.example, r.counterexample, r.apply_summary,
                     r.apply_score, r.self_check, r.retest_due_at, r.at,
                     _dumps(_check_to_dict(check))))
            return row_id

    def reiterations(self, agent_id: str,
                     lesson_id: str = "") -> List[Tuple[Reiteration, ReiterationCheck]]:
        with self._lock:
            if lesson_id:
                rows = self._conn.execute(
                    "SELECT * FROM reiterations WHERE agent_id=? AND lesson_id=?"
                    " ORDER BY at", (agent_id, lesson_id)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM reiterations WHERE agent_id=? ORDER BY at",
                    (agent_id,)).fetchall()
            return [_row_to_reiteration_pair(r) for r in rows]

    # -- tests and results ---------------------------------------------------
    def save_test(self, test: TestSpec) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO tests(id, capability, level, novelty, difficulty,"
                    " ambiguity, objective, expected, acceptable_range,"
                    " failure_mode_targeted, split, seed, retired, version,"
                    " author_agent_id, lesson_id, payload, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET capability=excluded.capability,"
                    " level=excluded.level, novelty=excluded.novelty,"
                    " difficulty=excluded.difficulty, ambiguity=excluded.ambiguity,"
                    " objective=excluded.objective, expected=excluded.expected,"
                    " acceptable_range=excluded.acceptable_range,"
                    " failure_mode_targeted=excluded.failure_mode_targeted,"
                    " split=excluded.split, seed=excluded.seed,"
                    " retired=excluded.retired, version=excluded.version,"
                    " author_agent_id=excluded.author_agent_id,"
                    " lesson_id=excluded.lesson_id, payload=excluded.payload",
                    (test.id, test.capability, int(test.level), test.novelty,
                     test.difficulty, test.ambiguity, int(test.objective),
                     test.expected, test.acceptable_range,
                     test.failure_mode_targeted, test.split.value, test.seed,
                     int(test.retired), test.version, test.author_agent_id,
                     test.lesson_id, _dumps(test.payload), test.created_at))

    def test(self, test_id: str) -> Optional[TestSpec]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM tests WHERE id=?",
                                     (test_id,)).fetchone()
            return _row_to_test(row) if row else None

    def tests(self, capability: str = "", level: Optional[TestLevel] = None,
              split: Optional[Split] = None,
              include_retired: bool = False) -> List[TestSpec]:
        with self._lock:
            clauses, params = [], []
            if capability:
                clauses.append("capability=?")
                params.append(capability)
            if level is not None:
                clauses.append("level=?")
                params.append(int(level))
            if split is not None:
                clauses.append("split=?")
                params.append(split.value)
            if not include_retired:
                clauses.append("retired=0")
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = self._conn.execute(
                f"SELECT * FROM tests{where} ORDER BY created_at", params).fetchall()
            return [_row_to_test(r) for r in rows]

    def retire_test(self, test_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE tests SET retired=1, split=? WHERE id=?",
                    (Split.REGRESSION.value, test_id))

    def save_result(self, result: TestResult) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO results(id, test_id, agent_id, lesson_id, level,"
                    " split, score, passed, student_claim, student_confidence,"
                    " trainer_claim, trainer_confidence, failure_mode, evidence,"
                    " duration_seconds, judge_needed, dispute_id, at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET score=excluded.score,"
                    " passed=excluded.passed, student_claim=excluded.student_claim,"
                    " student_confidence=excluded.student_confidence,"
                    " trainer_claim=excluded.trainer_claim,"
                    " trainer_confidence=excluded.trainer_confidence,"
                    " failure_mode=excluded.failure_mode,"
                    " evidence=excluded.evidence, judge_needed=excluded.judge_needed,"
                    " dispute_id=excluded.dispute_id",
                    (result.id, result.test_id, result.agent_id, result.lesson_id,
                     int(result.level), result.split.value, result.score,
                     int(result.passed), result.student_claim,
                     result.student_confidence, result.trainer_claim,
                     result.trainer_confidence, result.failure_mode,
                     _dumps(result.evidence), result.duration_seconds,
                     int(result.judge_needed), result.dispute_id, result.at))

    def results(self, agent_id: str, capability: str = "",
               level: Optional[TestLevel] = None, split: Optional[Split] = None,
               limit: int = 500) -> List[TestResult]:
        with self._lock:
            clauses, params = ["results.agent_id=?"], [agent_id]
            join = ""
            if capability:
                join = " JOIN tests ON tests.id = results.test_id"
                clauses.append("tests.capability=?")
                params.append(capability)
            if level is not None:
                clauses.append("results.level=?")
                params.append(int(level))
            if split is not None:
                clauses.append("results.split=?")
                params.append(split.value)
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT results.* FROM results{join}"
                f" WHERE {' AND '.join(clauses)}"
                f" ORDER BY results.at DESC LIMIT ?", params).fetchall()
            return [_row_to_result(r) for r in rows]

    # -- disputes and rulings ----------------------------------------------
    def save_dispute(self, d: Dispute) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO disputes(id, agent_id, test_id, lesson_id,"
                    " question, student_claim, trainer_claim, evidence_student,"
                    " evidence_trainer, student_confidence, trainer_confidence,"
                    " shared_knowledge, applicable_rules, status, ruling_id,"
                    " critical, at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET status=excluded.status,"
                    " ruling_id=excluded.ruling_id",
                    (d.id, d.agent_id, d.test_id, d.lesson_id, d.question,
                     d.student_claim, d.trainer_claim,
                     _dumps(d.evidence_student), _dumps(d.evidence_trainer),
                     d.student_confidence, d.trainer_confidence,
                     _dumps(d.shared_knowledge), _dumps(d.applicable_rules),
                     d.status.value, d.ruling_id, int(d.critical), d.at))

    def dispute(self, dispute_id: str) -> Optional[Dispute]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM disputes WHERE id=?",
                                     (dispute_id,)).fetchone()
            return _row_to_dispute(row) if row else None

    def disputes(self, agent_id: str = "",
                status: Optional[DisputeStatus] = None) -> List[Dispute]:
        with self._lock:
            clauses, params = [], []
            if agent_id:
                clauses.append("agent_id=?")
                params.append(agent_id)
            if status is not None:
                clauses.append("status=?")
                params.append(status.value)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = self._conn.execute(
                f"SELECT * FROM disputes{where} ORDER BY at DESC", params).fetchall()
            return [_row_to_dispute(r) for r in rows]

    def save_ruling(self, r: Ruling) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO rulings(id, dispute_id, ruling, accepted_claim,"
                    " rejected_claim, rationale, confidence, unresolved_issues,"
                    " correction_student, correction_trainer, reusable_lesson_id,"
                    " needs_external_evidence, decided_by, at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (r.id, r.dispute_id, r.ruling, r.accepted_claim,
                     r.rejected_claim, r.rationale, r.confidence,
                     _dumps(r.unresolved_issues), r.correction_student,
                     r.correction_trainer, r.reusable_lesson_id,
                     int(r.needs_external_evidence), r.decided_by, r.at))
                if r.ruling == "unresolved":
                    status = DisputeStatus.UNRESOLVED
                elif r.ruling:
                    status = DisputeStatus.RESOLVED
                else:
                    status = None
                if status is not None:
                    self._conn.execute(
                        "UPDATE disputes SET ruling_id=?, status=? WHERE id=?",
                        (r.id, status.value, r.dispute_id))

    def ruling(self, ruling_id: str) -> Optional[Ruling]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM rulings WHERE id=?",
                                     (ruling_id,)).fetchone()
            return _row_to_ruling(row) if row else None

    # -- reusable lessons ----------------------------------------------------
    def save_reusable(self, r: ReusableLesson) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO reusable_lessons(id, source_event,"
                    " rule_or_procedure, knowledge_class, confidence,"
                    " validation_status, scope_domain, scope_concept,"
                    " source_agent_id, version, validations, last_validated_at,"
                    " superseded_by, deprecated, created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                    " ON CONFLICT(id) DO UPDATE SET source_event=excluded.source_event,"
                    " rule_or_procedure=excluded.rule_or_procedure,"
                    " knowledge_class=excluded.knowledge_class,"
                    " confidence=excluded.confidence,"
                    " validation_status=excluded.validation_status,"
                    " scope_domain=excluded.scope_domain,"
                    " scope_concept=excluded.scope_concept, version=excluded.version,"
                    " validations=excluded.validations,"
                    " last_validated_at=excluded.last_validated_at,"
                    " superseded_by=excluded.superseded_by,"
                    " deprecated=excluded.deprecated",
                    (r.id, r.source_event, r.rule_or_procedure,
                     r.knowledge_class.value, r.confidence,
                     r.validation_status.value, r.scope_domain, r.scope_concept,
                     r.source_agent_id, r.version, r.validations,
                     r.last_validated_at, r.superseded_by, int(r.deprecated),
                     r.created_at))

    def reusable(self, reusable_id: str) -> Optional[ReusableLesson]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reusable_lessons WHERE id=?",
                (reusable_id,)).fetchone()
            return _row_to_reusable(row) if row else None

    def reusable_lessons(self, domain: str = "", concept: str = "",
                         status: Optional[ValidationStatus] = None,
                         include_deprecated: bool = False) -> List[ReusableLesson]:
        with self._lock:
            clauses, params = [], []
            if domain:
                clauses.append("scope_domain=?")
                params.append(domain)
            if concept:
                clauses.append("scope_concept=?")
                params.append(concept)
            if status is not None:
                clauses.append("validation_status=?")
                params.append(status.value)
            if not include_deprecated:
                clauses.append("deprecated=0")
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = self._conn.execute(
                f"SELECT * FROM reusable_lessons{where} ORDER BY created_at",
                params).fetchall()
            return [_row_to_reusable(r) for r in rows]

    def validate_reusable(self, reusable_id: str, by_agent_id: str) -> ReusableLesson:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reusable_lessons WHERE id=?",
                (reusable_id,)).fetchone()
            if row is None:
                raise KeyError(reusable_id)
            current = _row_to_reusable(row)
            # The source agent's own validation is not evidence of anything:
            # one accidental success must not become a universal rule by the
            # author simply saying so again (document 04 section 5).
            if by_agent_id and by_agent_id == current.source_agent_id:
                return current
            validations = current.validations + 1
            status = current.validation_status
            if status == ValidationStatus.CANDIDATE and validations >= 2:
                status = ValidationStatus.VALIDATED
            elif status == ValidationStatus.VALIDATED and validations >= 3:
                status = ValidationStatus.SHARED
            now = time.time()
            with self._conn:
                self._conn.execute(
                    "UPDATE reusable_lessons SET validations=?,"
                    " last_validated_at=?, validation_status=? WHERE id=?",
                    (validations, now, status.value, reusable_id))
            current.validations = validations
            current.last_validated_at = now
            current.validation_status = status
            return current

    def deprecate_reusable(self, reusable_id: str, superseded_by: str = "") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE reusable_lessons SET deprecated=1, superseded_by=?,"
                    " validation_status=? WHERE id=?",
                    (superseded_by, ValidationStatus.DEPRECATED.value, reusable_id))

    # -- mastery and promotion ------------------------------------------------
    def mastery(self, agent_id: str, concept: str) -> MasteryRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM mastery WHERE agent_id=? AND concept=?",
                (agent_id, concept)).fetchone()
            if row is None:
                return MasteryRecord(agent_id=agent_id, concept=concept)
            return _row_to_mastery(row)

    def mastery_table(self, agent_id: str) -> Dict[str, MasteryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM mastery WHERE agent_id=?", (agent_id,)).fetchall()
            return {r["concept"]: _row_to_mastery(r) for r in rows}

    def save_mastery(self, record: MasteryRecord) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO mastery(agent_id, concept, level, evidence,"
                    " failures_at_level, updated_at) VALUES (?,?,?,?,?,?)"
                    " ON CONFLICT(agent_id, concept) DO UPDATE SET"
                    " level=excluded.level, evidence=excluded.evidence,"
                    " failures_at_level=excluded.failures_at_level,"
                    " updated_at=excluded.updated_at",
                    (record.agent_id, record.concept, int(record.level),
                     _dumps(record.evidence), record.failures_at_level,
                     record.updated_at))

    def save_promotion(self, p: Promotion) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO promotions(id, agent_id, from_maturity,"
                    " to_maturity, evidence, at) VALUES (?,?,?,?,?,?)",
                    (p.id, p.agent_id, int(p.from_maturity), int(p.to_maturity),
                     _dumps(p.evidence), p.at))

    def promotions(self, agent_id: str) -> List[Promotion]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM promotions WHERE agent_id=? ORDER BY at",
                (agent_id,)).fetchall()
            return [_row_to_promotion(r) for r in rows]

    # -- factory feedback (document 05 section 7) -----------------------------
    def record_metric(self, name: str, value: float, agent_id: str = "",
                      detail: str = "") -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO metrics(name, value, agent_id, detail, at)"
                    " VALUES (?,?,?,?,?)",
                    (name, value, agent_id, detail, time.time()))

    def metrics(self, name: str = "") -> List[Dict[str, Any]]:
        with self._lock:
            if name:
                rows = self._conn.execute(
                    "SELECT * FROM metrics WHERE name=? ORDER BY at DESC",
                    (name,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM metrics ORDER BY at DESC").fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            def count(table: str) -> int:
                return int(self._conn.execute(
                    f"SELECT count(*) AS n FROM {table}").fetchone()["n"])

            return {
                "profiles": count("profiles"),
                "lessons": count("lessons"),
                "reiterations": count("reiterations"),
                "tests": count("tests"),
                "results": count("results"),
                "disputes": count("disputes"),
                "rulings": count("rulings"),
                "reusable_lessons": count("reusable_lessons"),
                "mastery_records": count("mastery"),
                "promotions": count("promotions"),
                "metrics": count("metrics"),
            }


# --------------------------------------------------------------------------
# row <-> dataclass conversions
# --------------------------------------------------------------------------
def _spec_to_dict(spec: AgentSpec) -> Dict[str, Any]:
    return {
        "name": spec.name, "role": spec.role, "domain": spec.domain,
        "mission": spec.mission, "capabilities": spec.capabilities,
        "allowed_actions": spec.allowed_actions,
        "prohibited_actions": spec.prohibited_actions, "tools": spec.tools,
        "input_contract": spec.input_contract,
        "output_contract": spec.output_contract,
        "success_metrics": spec.success_metrics,
        "safety_constraints": spec.safety_constraints,
        "environment": spec.environment, "rollback": spec.rollback,
        "permissions": spec.permissions, "monitoring": spec.monitoring,
        "escalation": spec.escalation,
    }


def _dict_to_spec(data: Dict[str, Any]) -> AgentSpec:
    return AgentSpec(**data)


def _check_to_dict(check: ReiterationCheck) -> Dict[str, Any]:
    return {
        "restate_ok": check.restate_ok, "explain_ok": check.explain_ok,
        "connect_ok": check.connect_ok, "example_ok": check.example_ok,
        "counterexample_ok": check.counterexample_ok, "notes": check.notes,
    }


def _dict_to_check(data: Dict[str, Any]) -> ReiterationCheck:
    return ReiterationCheck(**data)


def _row_to_profile(row: sqlite3.Row) -> AgentProfile:
    spec_raw = row["spec"]
    spec = _dict_to_spec(json.loads(spec_raw)) if spec_raw else None
    return AgentProfile(
        id=row["id"], name=row["name"], role=row["role"], domain=row["domain"],
        capabilities=_loads(row["capabilities"], []),
        maturity=Maturity(row["maturity"]),
        current_curriculum=row["current_curriculum"],
        current_lesson_id=row["current_lesson_id"],
        strengths=_loads(row["strengths"], []),
        weaknesses=_loads(row["weaknesses"], []),
        knowledge_version=row["knowledge_version"], spec=spec,
        created_at=row["created_at"], updated_at=row["updated_at"])


def _row_to_lesson(row: sqlite3.Row) -> Lesson:
    return Lesson(
        id=row["id"], domain=row["domain"], concept=row["concept"],
        objective=row["objective"], prerequisites=_loads(row["prerequisites"], []),
        source_knowledge=_loads(row["source_knowledge"], []),
        explanation=row["explanation"], examples=_loads(row["examples"], []),
        counterexamples=_loads(row["counterexamples"], []),
        practice_tasks=_loads(row["practice_tasks"], []),
        test_tasks=_loads(row["test_tasks"], []),
        expected_behavior=row["expected_behavior"],
        common_errors=_loads(row["common_errors"], []),
        remediation=_loads(row["remediation"], []),
        knowledge_class=KnowledgeClass(row["knowledge_class"]),
        confidence=row["confidence"], version=row["version"], origin=row["origin"],
        payload=_loads(row["payload"], {}))


def _row_to_reiteration_pair(row: sqlite3.Row) -> Tuple[Reiteration, ReiterationCheck]:
    r = Reiteration(
        lesson_id=row["lesson_id"], agent_id=row["agent_id"],
        restate=row["restate"], explain=row["explain"], connect=row["connect"],
        example=row["example"], counterexample=row["counterexample"],
        apply_summary=row["apply_summary"], apply_score=row["apply_score"],
        self_check=row["self_check"], retest_due_at=row["retest_due_at"],
        at=row["at"])
    check = _dict_to_check(_loads(row["check_json"], {}))
    return r, check


def _row_to_test(row: sqlite3.Row) -> TestSpec:
    return TestSpec(
        id=row["id"], capability=row["capability"], level=TestLevel(row["level"]),
        novelty=row["novelty"], difficulty=row["difficulty"],
        ambiguity=row["ambiguity"], objective=bool(row["objective"]),
        expected=row["expected"], acceptable_range=row["acceptable_range"],
        failure_mode_targeted=row["failure_mode_targeted"],
        split=Split(row["split"]), seed=row["seed"], retired=bool(row["retired"]),
        version=row["version"], author_agent_id=row["author_agent_id"],
        lesson_id=row["lesson_id"], payload=_loads(row["payload"], {}),
        created_at=row["created_at"])


def _row_to_result(row: sqlite3.Row) -> TestResult:
    return TestResult(
        id=row["id"], test_id=row["test_id"], agent_id=row["agent_id"],
        lesson_id=row["lesson_id"], level=TestLevel(row["level"]),
        split=Split(row["split"]), score=row["score"], passed=bool(row["passed"]),
        student_claim=row["student_claim"],
        student_confidence=row["student_confidence"],
        trainer_claim=row["trainer_claim"],
        trainer_confidence=row["trainer_confidence"],
        failure_mode=row["failure_mode"], evidence=_loads(row["evidence"], []),
        duration_seconds=row["duration_seconds"],
        judge_needed=bool(row["judge_needed"]), dispute_id=row["dispute_id"],
        at=row["at"])


def _row_to_dispute(row: sqlite3.Row) -> Dispute:
    return Dispute(
        id=row["id"], agent_id=row["agent_id"], test_id=row["test_id"],
        lesson_id=row["lesson_id"], question=row["question"],
        student_claim=row["student_claim"], trainer_claim=row["trainer_claim"],
        evidence_student=_loads(row["evidence_student"], []),
        evidence_trainer=_loads(row["evidence_trainer"], []),
        student_confidence=row["student_confidence"],
        trainer_confidence=row["trainer_confidence"],
        shared_knowledge=_loads(row["shared_knowledge"], []),
        applicable_rules=_loads(row["applicable_rules"], []),
        status=DisputeStatus(row["status"]), ruling_id=row["ruling_id"],
        critical=bool(row["critical"]), at=row["at"])


def _row_to_ruling(row: sqlite3.Row) -> Ruling:
    return Ruling(
        id=row["id"], dispute_id=row["dispute_id"], ruling=row["ruling"],
        accepted_claim=row["accepted_claim"], rejected_claim=row["rejected_claim"],
        rationale=row["rationale"], confidence=row["confidence"],
        unresolved_issues=_loads(row["unresolved_issues"], []),
        correction_student=row["correction_student"],
        correction_trainer=row["correction_trainer"],
        reusable_lesson_id=row["reusable_lesson_id"],
        needs_external_evidence=bool(row["needs_external_evidence"]),
        decided_by=row["decided_by"], at=row["at"])


def _row_to_reusable(row: sqlite3.Row) -> ReusableLesson:
    return ReusableLesson(
        id=row["id"], source_event=row["source_event"],
        rule_or_procedure=row["rule_or_procedure"],
        knowledge_class=KnowledgeClass(row["knowledge_class"]),
        confidence=row["confidence"],
        validation_status=ValidationStatus(row["validation_status"]),
        scope_domain=row["scope_domain"], scope_concept=row["scope_concept"],
        source_agent_id=row["source_agent_id"], version=row["version"],
        validations=row["validations"], last_validated_at=row["last_validated_at"],
        superseded_by=row["superseded_by"], deprecated=bool(row["deprecated"]),
        created_at=row["created_at"])


def _row_to_mastery(row: sqlite3.Row) -> MasteryRecord:
    return MasteryRecord(
        agent_id=row["agent_id"], concept=row["concept"],
        level=MasteryLevel(row["level"]), evidence=_loads(row["evidence"], []),
        failures_at_level=row["failures_at_level"], updated_at=row["updated_at"])


def _row_to_promotion(row: sqlite3.Row) -> Promotion:
    return Promotion(
        id=row["id"], agent_id=row["agent_id"],
        from_maturity=Maturity(row["from_maturity"]),
        to_maturity=Maturity(row["to_maturity"]), evidence=_loads(row["evidence"], []),
        at=row["at"])
