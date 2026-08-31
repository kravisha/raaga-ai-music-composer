"""The Knowledge Base schema, and the rules about creating it.

Specification sections 2, 31, 32, 33 and 36.  Section 2 is the one this file
mostly exists to enforce: the Knowledge Base is created once and then grows.
Deleting learned knowledge must never be a side effect of starting up,
upgrading, reinstalling or training.

Two ideas keep that true here rather than by good intentions:

*A durable initialization marker.*  ``knowledge_base_initialized`` is written
once, with the moment it happened.  Every later open reads it and takes the
existing store.  Initialization and migration are separate code paths -
section 31 asks for exactly that - so a migration can never fall through into
"create it fresh".

*Views instead of duplicate tables.*  Section 33's ``entities``, ``claims``,
``procedures`` and ``examples`` are read-only views over ``knowledge_items``,
so the logical model can be queried by those names while there remains exactly
one row per piece of knowledge and one kind of id for relationships to point
at.  Writes go through the service, never through a view.

Full-text search is a contentless FTS5 index kept in step by triggers.  If the
SQLite build has no FTS5 the store still works and retrieval falls back to
LIKE; that is checked at open rather than assumed.
"""
from __future__ import annotations

SCHEMA_VERSION = 1

#: Written once, read at every open.  Section 31.
INITIALIZED_KEY = "knowledge_base_initialized"

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_metadata (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at REAL);

CREATE TABLE IF NOT EXISTS migrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_version INTEGER,
    to_version INTEGER,
    applied_at REAL,
    detail TEXT DEFAULT '',
    ok INTEGER DEFAULT 1);

-- One node table.  A claim is a row with subject and predicate filled in; an
-- entity is a row of an entity type.  One id space, so a relationship can
-- point at anything and nothing can drift out of step with itself.
CREATE TABLE IF NOT EXISTS knowledge_items (
    knowledge_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL DEFAULT '',
    knowledge_type TEXT NOT NULL,
    subject TEXT DEFAULT '',
    predicate TEXT DEFAULT '',
    object_value TEXT DEFAULT '',
    statement TEXT DEFAULT '',
    structured_value TEXT DEFAULT '{}',
    scope TEXT DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'candidate',
    confidence REAL DEFAULT 0.0,
    confidence_parts TEXT DEFAULT '{}',
    importance REAL DEFAULT 0.5,
    source_count INTEGER DEFAULT 0,
    created_at REAL,
    updated_at REAL,
    last_verified_at REAL DEFAULT 0,
    version INTEGER DEFAULT 1,
    tags TEXT DEFAULT '[]',
    valid_from TEXT DEFAULT '',
    valid_until TEXT DEFAULT '',
    language TEXT DEFAULT '',
    difficulty TEXT DEFAULT '',
    curriculum_level TEXT DEFAULT '',
    owner_or_creator TEXT DEFAULT '',
    review_state TEXT DEFAULT '',
    usage_count INTEGER DEFAULT 0,
    last_used_at REAL DEFAULT 0,
    learned_by TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    raga TEXT DEFAULT '',
    tala TEXT DEFAULT '',
    -- subject + predicate + raga, normalised: the duplicate-control key.
    identity TEXT NOT NULL DEFAULT '');

CREATE INDEX IF NOT EXISTS ki_identity ON knowledge_items(identity);
CREATE INDEX IF NOT EXISTS ki_raga ON knowledge_items(raga);
CREATE INDEX IF NOT EXISTS ki_tala ON knowledge_items(tala);
CREATE INDEX IF NOT EXISTS ki_type ON knowledge_items(knowledge_type);
CREATE INDEX IF NOT EXISTS ki_status ON knowledge_items(status);
CREATE INDEX IF NOT EXISTS ki_canonical ON knowledge_items(canonical_name);
CREATE INDEX IF NOT EXISTS ki_subject ON knowledge_items(subject);
CREATE INDEX IF NOT EXISTS ki_confidence ON knowledge_items(confidence);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,
    source_knowledge_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_knowledge_id TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    evidence TEXT DEFAULT '',
    created_at REAL,
    status TEXT DEFAULT 'accepted');

CREATE UNIQUE INDEX IF NOT EXISTS rel_unique
    ON relationships(source_knowledge_id, relation_type, target_knowledge_id);
CREATE INDEX IF NOT EXISTS rel_source ON relationships(source_knowledge_id);
CREATE INDEX IF NOT EXISTS rel_target ON relationships(target_knowledge_id);
CREATE INDEX IF NOT EXISTS rel_type ON relationships(relation_type);

CREATE TABLE IF NOT EXISTS aliases (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    knowledge_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    normalised TEXT NOT NULL,
    kind TEXT DEFAULT 'alias',
    UNIQUE(knowledge_id, normalised));
CREATE INDEX IF NOT EXISTS alias_norm ON aliases(normalised);

CREATE TABLE IF NOT EXISTS tags (
    tag TEXT PRIMARY KEY,
    created_at REAL);

CREATE TABLE IF NOT EXISTS knowledge_tags (
    knowledge_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (knowledge_id, tag));
CREATE INDEX IF NOT EXISTS kt_tag ON knowledge_tags(tag);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT DEFAULT '',
    title TEXT DEFAULT '',
    author_or_channel TEXT DEFAULT '',
    reference TEXT DEFAULT '',
    published_date TEXT DEFAULT '',
    acquired_date REAL,
    license_or_access_notes TEXT DEFAULT '',
    language TEXT DEFAULT '',
    checksum TEXT DEFAULT '',
    metadata TEXT DEFAULT '{}',
    training_source_id TEXT DEFAULT '',
    identity TEXT DEFAULT '');
CREATE INDEX IF NOT EXISTS src_identity ON sources(identity);
CREATE INDEX IF NOT EXISTS src_training ON sources(training_source_id);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    knowledge_id TEXT NOT NULL,
    source_segment TEXT DEFAULT '',
    timestamp_start REAL DEFAULT 0,
    timestamp_end REAL DEFAULT 0,
    transcript_excerpt TEXT DEFAULT '',
    feature_reference TEXT DEFAULT '',
    strength REAL DEFAULT 0.5,
    extraction_method TEXT DEFAULT 'inferred',
    run_id TEXT DEFAULT '',
    created_at REAL,
    supports INTEGER DEFAULT 1);
CREATE INDEX IF NOT EXISTS ev_knowledge ON evidence(knowledge_id);
CREATE INDEX IF NOT EXISTS ev_source ON evidence(source_id);
CREATE INDEX IF NOT EXISTS ev_run ON evidence(run_id);

CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    claim_a TEXT NOT NULL,
    claim_b TEXT NOT NULL,
    source_a TEXT DEFAULT '',
    source_b TEXT DEFAULT '',
    confidence_a REAL DEFAULT 0,
    confidence_b REAL DEFAULT 0,
    resolution_status TEXT DEFAULT 'unresolved',
    reviewer TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    created_at REAL,
    resolved_at REAL DEFAULT 0);
CREATE INDEX IF NOT EXISTS cft_status ON conflicts(resolution_status);
CREATE INDEX IF NOT EXISTS cft_a ON conflicts(claim_a);
CREATE INDEX IF NOT EXISTS cft_b ON conflicts(claim_b);

CREATE TABLE IF NOT EXISTS knowledge_versions (
    version_id TEXT PRIMARY KEY,
    knowledge_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    snapshot TEXT NOT NULL,
    changed_at REAL,
    reason TEXT DEFAULT '',
    caused_by_source_id TEXT DEFAULT '',
    caused_by_run_id TEXT DEFAULT '',
    changed_by TEXT DEFAULT 'system');
CREATE INDEX IF NOT EXISTS kv_knowledge ON knowledge_versions(knowledge_id);

-- 1:1 detail rows.  Not a second id space: the key is the item's own id.
CREATE TABLE IF NOT EXISTS procedure_details (
    knowledge_id TEXT PRIMARY KEY,
    goal TEXT DEFAULT '',
    prerequisites TEXT DEFAULT '[]',
    inputs TEXT DEFAULT '[]',
    steps TEXT DEFAULT '[]',
    optional_branches TEXT DEFAULT '[]',
    constraints TEXT DEFAULT '[]',
    failure_modes TEXT DEFAULT '[]',
    evaluation_criteria TEXT DEFAULT '[]');

CREATE TABLE IF NOT EXISTS example_details (
    knowledge_id TEXT PRIMARY KEY,
    concept_demonstrated TEXT DEFAULT '',
    notation TEXT DEFAULT '',
    swaras TEXT DEFAULT '[]',
    features TEXT DEFAULT '{}',
    source_id TEXT DEFAULT '',
    timestamp_start REAL DEFAULT 0,
    timestamp_end REAL DEFAULT 0,
    quality REAL DEFAULT 0.5,
    curriculum_stage TEXT DEFAULT '');

CREATE TABLE IF NOT EXISTS failure_lessons (
    lesson_id TEXT PRIMARY KEY,
    task TEXT DEFAULT '',
    attempted_method TEXT DEFAULT '',
    result TEXT DEFAULT '',
    failure_reason TEXT DEFAULT '',
    correction TEXT DEFAULT '',
    related_knowledge TEXT DEFAULT '[]',
    source_or_run TEXT DEFAULT '',
    confidence REAL DEFAULT 0.5,
    created_at REAL);

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    gap_id TEXT PRIMARY KEY,
    subject TEXT DEFAULT '',
    missing_information TEXT DEFAULT '',
    importance REAL DEFAULT 0.5,
    reason TEXT DEFAULT '',
    suggested_search TEXT DEFAULT '',
    curriculum_dependency TEXT DEFAULT '',
    status TEXT DEFAULT 'open',
    created_at REAL,
    closed_at REAL DEFAULT 0,
    identity TEXT DEFAULT '');
CREATE UNIQUE INDEX IF NOT EXISTS gap_identity ON knowledge_gaps(identity);
CREATE INDEX IF NOT EXISTS gap_status ON knowledge_gaps(status);

CREATE TABLE IF NOT EXISTS user_corrections (
    correction_id TEXT PRIMARY KEY,
    knowledge_id TEXT NOT NULL,
    action TEXT NOT NULL,
    explanation TEXT DEFAULT '',
    previous_value TEXT DEFAULT '',
    new_value TEXT DEFAULT '',
    previous_confidence REAL DEFAULT 0,
    new_confidence REAL DEFAULT 0,
    created_at REAL);
CREATE INDEX IF NOT EXISTS uc_knowledge ON user_corrections(knowledge_id);

-- What retrieval actually served, so "which knowledge did you use?" has an
-- answer after the fact and not only during the call.  Section 41.
CREATE TABLE IF NOT EXISTS retrieval_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL,
    task TEXT DEFAULT '',
    query TEXT DEFAULT '',
    knowledge_ids TEXT DEFAULT '[]',
    context_id TEXT DEFAULT '',
    item_count INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ru_context ON retrieval_usage(context_id);

CREATE TABLE IF NOT EXISTS kb_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL,
    kind TEXT,
    detail TEXT DEFAULT '',
    knowledge_id TEXT DEFAULT '',
    source_id TEXT DEFAULT '',
    run_id TEXT DEFAULT '',
    actor TEXT DEFAULT 'system');
CREATE INDEX IF NOT EXISTS audit_knowledge ON kb_audit(knowledge_id);
"""

#: Section 33's names, as read-only views over the one node table.
VIEWS = """
DROP VIEW IF EXISTS entities;
CREATE VIEW entities AS
    SELECT * FROM knowledge_items WHERE knowledge_type = 'entity';

DROP VIEW IF EXISTS claims;
CREATE VIEW claims AS
    SELECT knowledge_id AS claim_id, subject, predicate,
           object_value AS object_or_value, statement AS explanation,
           confidence, status, source_count, version, knowledge_id
    FROM knowledge_items
    WHERE subject <> '' AND predicate <> '';

DROP VIEW IF EXISTS procedures;
CREATE VIEW procedures AS
    SELECT k.*, d.goal, d.prerequisites, d.inputs, d.steps,
           d.optional_branches, d.constraints, d.failure_modes,
           d.evaluation_criteria
    FROM knowledge_items k JOIN procedure_details d
    ON d.knowledge_id = k.knowledge_id;

DROP VIEW IF EXISTS examples;
CREATE VIEW examples AS
    SELECT k.*, d.concept_demonstrated, d.notation, d.swaras, d.features,
           d.source_id AS example_source_id, d.timestamp_start,
           d.timestamp_end, d.quality, d.curriculum_stage
    FROM knowledge_items k JOIN example_details d
    ON d.knowledge_id = k.knowledge_id;
"""

#: Contentless FTS5 over the text a person would search.  Optional: some
#: SQLite builds ship without FTS5, and retrieval falls back to LIKE.
FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    canonical_name, subject, statement, object_value, notes,
    content='knowledge_items', content_rowid='rowid');

CREATE TRIGGER IF NOT EXISTS ki_fts_insert AFTER INSERT ON knowledge_items
BEGIN
    INSERT INTO knowledge_fts(rowid, canonical_name, subject, statement,
                              object_value, notes)
    VALUES (new.rowid, new.canonical_name, new.subject, new.statement,
            new.object_value, new.notes);
END;

CREATE TRIGGER IF NOT EXISTS ki_fts_delete AFTER DELETE ON knowledge_items
BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, canonical_name, subject,
                              statement, object_value, notes)
    VALUES ('delete', old.rowid, old.canonical_name, old.subject,
            old.statement, old.object_value, old.notes);
END;

CREATE TRIGGER IF NOT EXISTS ki_fts_update AFTER UPDATE ON knowledge_items
BEGIN
    INSERT INTO knowledge_fts(knowledge_fts, rowid, canonical_name, subject,
                              statement, object_value, notes)
    VALUES ('delete', old.rowid, old.canonical_name, old.subject,
            old.statement, old.object_value, old.notes);
    INSERT INTO knowledge_fts(rowid, canonical_name, subject, statement,
                              object_value, notes)
    VALUES (new.rowid, new.canonical_name, new.subject, new.statement,
            new.object_value, new.notes);
END;
"""

#: The taxonomy every Knowledge Base starts with - section 31 step 3.  These
#: are structural facts about Carnatic music as a subject, not claims learned
#: from anybody, so they are seeded as accepted and carry no evidence.
CORE_TAXONOMY = (
    ("Raga", "entity", "the melodic framework a piece is built on"),
    ("Tala", "entity", "the rhythmic cycle a piece is set in"),
    ("Swara", "entity", "a scale degree"),
    ("Gamaka", "entity", "an ornament applied to a swara or a movement"),
    ("Prayoga", "entity", "a characteristic phrase that identifies a raga"),
    ("Composition", "entity", "a composed piece"),
    ("Instrument", "entity", "something the music is played on"),
    ("Composer", "entity", "a person who composed"),
    ("Technique", "entity", "a way of doing something musical"),
    ("Curriculum Unit", "entity", "one step of a course of study"),
)
