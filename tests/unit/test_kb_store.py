"""The durable store - knowledge-base specification sections 2, 31, 34, 36.

Section 2 is what most of this file exists to defend: the Knowledge Base is
created once and grows, and losing learned knowledge must never be a side
effect of starting up, upgrading or training.  Several of these tests would
pass trivially against a store that recreated itself every run, so each one
checks the thing that would actually be destroyed.
"""
from __future__ import annotations

import sqlite3

import pytest

from raagacomposer.kb.schema import INITIALIZED_KEY, SCHEMA_VERSION
from raagacomposer.kb.store import (KnowledgeBaseCorrupt, KnowledgeBaseError,
                                    KnowledgeStore)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# creation and opening (sections 2, 31)
# --------------------------------------------------------------------------
def test_a_new_store_is_initialized_once_and_says_so(kb_path):
    store = KnowledgeStore(kb_path)
    try:
        assert store.initialized
        assert store.get_meta(INITIALIZED_KEY) == "true"
        assert store.initialized_at > 0
        assert store.schema_version == SCHEMA_VERSION
    finally:
        store.close()


def test_reopening_continues_rather_than_reinitializing(kb_path):
    """The rule the whole specification turns on."""
    first = KnowledgeStore(kb_path)
    stamp = first.initialized_at
    with first.transaction():
        first.execute(
            "INSERT INTO knowledge_items(knowledge_id, knowledge_type, "
            "canonical_name, statement, created_at, updated_at, identity) "
            "VALUES ('kn_test','fact','Kambhoji','something learned',1,1,'x')")
    first.close()

    second = KnowledgeStore(kb_path)
    try:
        assert second.initialized_at == stamp, "it was initialized again"
        assert second.count("knowledge_items") == 1, "learned knowledge went"
    finally:
        second.close()


def test_opening_a_missing_store_without_permission_is_refused(tmp_path):
    """A wrong path must look like a wrong path, not like total loss."""
    with pytest.raises(KnowledgeBaseError, match="not.*asked for"):
        KnowledgeStore(tmp_path / "nowhere.db", create=False)


def test_a_store_from_a_newer_version_is_refused_untouched(kb_path,
                                                           monkeypatch):
    KnowledgeStore(kb_path).close()
    import raagacomposer.kb.store as module
    monkeypatch.setattr(module, "SCHEMA_VERSION", 0)
    with pytest.raises(KnowledgeBaseError, match="newer version"):
        KnowledgeStore(kb_path)
    # ... and it is still there, unaltered.
    monkeypatch.undo()
    reopened = KnowledgeStore(kb_path)
    try:
        assert reopened.initialized
    finally:
        reopened.close()


def test_a_migration_is_recorded_and_changes_no_knowledge(kb_path,
                                                          monkeypatch):
    """Section 31: migration is its own path, never a recreation."""
    store = KnowledgeStore(kb_path)
    with store.transaction():
        store.execute(
            "INSERT INTO knowledge_items(knowledge_id, knowledge_type, "
            "canonical_name, statement, created_at, updated_at, identity) "
            "VALUES ('kn_a','fact','Kambhoji','a learned thing',1,1,'x')")
    store.set_meta("schema_version", "0")
    store.close()

    reopened = KnowledgeStore(kb_path)
    try:
        assert reopened.schema_version == SCHEMA_VERSION
        assert reopened.count("knowledge_items") == 1
        assert reopened.count("migrations") == 1
    finally:
        reopened.close()


def test_the_specifications_logical_tables_are_all_queryable(kb_path):
    """Section 33 names them; a caller must be able to use those names."""
    store = KnowledgeStore(kb_path)
    try:
        for name in ("kb_metadata", "knowledge_items", "entities", "claims",
                     "relationships", "aliases", "tags", "knowledge_tags",
                     "sources", "evidence", "conflicts", "knowledge_versions",
                     "procedures", "examples", "failure_lessons",
                     "knowledge_gaps", "user_corrections", "retrieval_usage",
                     "migrations"):
            store.query(f"SELECT * FROM {name} LIMIT 1")
    finally:
        store.close()


# --------------------------------------------------------------------------
# transactions (section 34)
# --------------------------------------------------------------------------
def test_a_failed_write_leaves_nothing_behind(kb_path):
    """Section 34: a failed learning run must not leave half-created
    canonical knowledge."""
    store = KnowledgeStore(kb_path)
    try:
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.execute(
                    "INSERT INTO knowledge_items(knowledge_id, "
                    "knowledge_type, canonical_name, statement, created_at, "
                    "updated_at, identity) VALUES "
                    "('kn_half','fact','Kambhoji','half a thing',1,1,'x')")
                raise RuntimeError("the run failed here")
        assert store.count("knowledge_items") == 0
    finally:
        store.close()


def test_transactions_can_follow_one_another(kb_path):
    """A plain write must not leave a transaction open behind it, or the next
    explicit one cannot begin."""
    store = KnowledgeStore(kb_path)
    try:
        for index in range(3):
            with store.transaction():
                store.execute(
                    "INSERT INTO knowledge_items(knowledge_id, "
                    "knowledge_type, canonical_name, statement, created_at, "
                    "updated_at, identity) VALUES (?,?,?,?,?,?,?)",
                    (f"kn_{index}", "fact", "Kambhoji", "a thing", 1, 1,
                     f"id{index}"))
            store.audit("test.write", f"item {index}")
        assert store.count("knowledge_items") == 3
    finally:
        store.close()


# --------------------------------------------------------------------------
# integrity and backup (section 36)
# --------------------------------------------------------------------------
def test_a_healthy_store_passes_its_integrity_check(kb_path):
    store = KnowledgeStore(kb_path)
    try:
        assert store.check_integrity()
        assert store.get_meta("last_integrity_check")
    finally:
        store.close()


def test_a_corrupt_store_is_preserved_rather_than_replaced(kb_path):
    """Section 36's line: a damaged Knowledge Base must not be silently
    replaced with an empty one."""
    KnowledgeStore(kb_path).close()
    # Overwrite the header so SQLite cannot read it as a database at all.
    with open(kb_path, "r+b") as handle:
        handle.write(b"this is not a database at all, not even close")

    with pytest.raises(KnowledgeBaseCorrupt):
        KnowledgeStore(kb_path)

    assert kb_path.exists(), "the damaged store was deleted"
    preserved = list(kb_path.parent.glob("*.damaged-*.db"))
    assert preserved, "no copy of the damaged store was kept"


def test_a_backup_is_a_readable_knowledge_base(kb_path, tmp_path):
    store = KnowledgeStore(kb_path)
    try:
        with store.transaction():
            store.execute(
                "INSERT INTO knowledge_items(knowledge_id, knowledge_type, "
                "canonical_name, statement, created_at, updated_at, identity) "
                "VALUES ('kn_b','fact','Kambhoji','a thing',1,1,'x')")
        target = store.backup(tmp_path / "copy.db")
    finally:
        store.close()

    restored = KnowledgeStore(target)
    try:
        assert restored.count("knowledge_items") == 1
        assert restored.initialized
    finally:
        restored.close()


def test_closing_checkpoints_rather_than_losing_recent_writes(kb_path):
    """Section 35: do not wait for shutdown, and do not lose what was in the
    write-ahead log when it comes."""
    store = KnowledgeStore(kb_path)
    with store.transaction():
        store.execute(
            "INSERT INTO knowledge_items(knowledge_id, knowledge_type, "
            "canonical_name, statement, created_at, updated_at, identity) "
            "VALUES ('kn_c','fact','Kambhoji','written just before close',"
            "1,1,'x')")
    store.close()

    reopened = KnowledgeStore(kb_path)
    try:
        assert reopened.count("knowledge_items") == 1
    finally:
        reopened.close()
