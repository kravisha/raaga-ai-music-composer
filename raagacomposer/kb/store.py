"""The durable Knowledge Base - opening it, writing to it, keeping it whole.

Specification sections 2, 31, 34, 35, 36 and 42.  This is the layer that makes
the persistence rule true rather than intended.

The rule is worth restating because everything here serves it: the Knowledge
Base is created once and grows.  It is never reinitialised as a side effect of
starting up, upgrading, reinstalling or training, and a damaged one is never
silently replaced with an empty one.

How that is enforced:

* opening is not creating.  :meth:`open` refuses to create a store unless it
  is asked to, so a wrong path is an error rather than a fresh empty database
  that looks like everything was lost.
* a durable marker records the moment of first initialization.  Every later
  open reads it and continues.
* a store written by a newer schema is refused outright.  Reading it anyway
  would silently misinterpret columns that had moved.
* a corrupt store stops destructive writes and keeps itself.  It is reported,
  a copy is preserved, and restoring is a deliberate act somewhere else.
* every knowledge write is one transaction, so a failed learning run leaves
  no half-created canonical knowledge (section 34).

Writes are serialised on a lock because the training queue works on its own
thread while the UI reads on another.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from ..core.logging_setup import get_logger
from ..core.settings import config_dir
from .schema import (CORE_SCHEMA, FTS_SCHEMA, INITIALIZED_KEY, SCHEMA_VERSION,
                     VIEWS)

log = get_logger("kb.store")


class KnowledgeBaseError(RuntimeError):
    """Something is wrong with the store itself, not with a piece of knowledge."""


class KnowledgeBaseCorrupt(KnowledgeBaseError):
    """Integrity check failed.  The store is kept; nothing destructive runs."""


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(text: Optional[str], fallback: Any) -> Any:
    try:
        return json.loads(text) if text else fallback
    except (TypeError, ValueError):
        return fallback


class KnowledgeStore:
    """The persistent store.  One file, opened once, grown for ever."""

    def __init__(self, path: Optional[Path] = None, *, create: bool = True,
                 check_integrity: bool = True) -> None:
        self.path = Path(path) if path else config_dir() / "knowledge_base.db"
        self._lock = threading.RLock()
        self._closed = False
        self.fts_available = False
        self.last_integrity_check = 0.0
        self.integrity_ok = True

        existed = self.path.exists()
        if not existed and not create:
            # Section 2: a missing store is reported, never quietly invented.
            raise KnowledgeBaseError(
                f"no Knowledge Base at {self.path}, and creating one was not "
                f"asked for. Learned knowledge is not recreated by accident.")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None turns off the driver's implicit transaction
        # management.  Without it an ordinary INSERT quietly opens a
        # transaction of its own and the explicit BEGIN in `transaction()`
        # fails with "cannot start a transaction within a transaction" - which
        # would make section 34's all-or-nothing write path unusable.
        try:
            self._conn = sqlite3.connect(str(self.path),
                                         check_same_thread=False,
                                         timeout=15.0, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError as exc:
            # Damaged badly enough that SQLite will not open it at all.  This
            # is still section 36's case and gets section 36's treatment:
            # stop, keep the file, report - never replace it with a fresh one.
            self._closed = True
            self.integrity_ok = False
            if not existed:
                raise
            preserved = self._preserve_damaged()
            raise KnowledgeBaseCorrupt(
                f"the Knowledge Base at {self.path} could not be opened "
                f"({exc}). Nothing has been written and nothing has been "
                f"replaced."
                + (f" A copy was preserved at {preserved}."
                   if preserved else "")) from exc

        if existed and check_integrity:
            self._check_integrity_or_refuse()

        self._create_or_open(existed)

    # ==================================================================
    # opening
    # ==================================================================
    def _create_or_open(self, existed: bool) -> None:
        with self._lock, self._conn:
            self._conn.executescript(CORE_SCHEMA)
            self._conn.executescript(VIEWS)
        self._install_fts()

        marker = self.get_meta(INITIALIZED_KEY)
        if marker is None:
            if existed and self._count("knowledge_items") > 0:
                # A store with knowledge but no marker predates the marker.
                # Adopt it; do not treat it as new.
                log.info("adopting an existing Knowledge Base that has no "
                         "initialization marker")
            self._initialize()
        else:
            stored = int(self.get_meta("schema_version") or 0)
            if stored > SCHEMA_VERSION:
                self.close()
                raise KnowledgeBaseError(
                    f"the Knowledge Base at {self.path} was written by a newer "
                    f"version of this application (schema {stored} > "
                    f"{SCHEMA_VERSION}). It has not been touched.")
            if stored < SCHEMA_VERSION:
                self._migrate(stored, SCHEMA_VERSION)
            log.info("Knowledge Base opened: %s (%d item(s))", self.path,
                     self._count("knowledge_items"))

    def _initialize(self) -> None:
        """Section 31.  Once, and recorded."""
        now = time.time()
        with self._lock, self._conn:
            self.set_meta(INITIALIZED_KEY, "true")
            self.set_meta("initialized_at", str(now))
            self.set_meta("schema_version", str(SCHEMA_VERSION))
            self.set_meta("created_by", "raagacomposer")
        self.audit("kb.initialized", f"created at {self.path}")
        log.info("Knowledge Base created at %s", self.path)

    def _migrate(self, from_version: int, to_version: int) -> None:
        """Section 31: migration is its own path, and never a recreation.

        There is nothing to do yet - this is schema 1 - but the ladder and its
        log exist now so that the first real migration cannot be written as
        "drop and rebuild".
        """
        log.info("migrating the Knowledge Base from schema %d to %d",
                 from_version, to_version)
        with self._lock, self._conn:
            # Future steps go here, each additive and each logged.
            self._conn.execute(
                "INSERT INTO migrations(from_version, to_version, applied_at, "
                "detail, ok) VALUES (?,?,?,?,1)",
                (from_version, to_version, time.time(),
                 "schema brought forward; no knowledge was altered"))
            self.set_meta("schema_version", str(to_version))
        self.audit("kb.migrated", f"{from_version} -> {to_version}")

    def _install_fts(self) -> None:
        try:
            with self._lock, self._conn:
                self._conn.executescript(FTS_SCHEMA)
            self.fts_available = True
        except sqlite3.Error as exc:
            # Not fatal: retrieval falls back to LIKE and says so.
            log.info("full-text search is unavailable (%s); retrieval will "
                     "use a slower substring match", exc)
            self.fts_available = False

    # ==================================================================
    # integrity (section 36)
    # ==================================================================
    def _check_integrity_or_refuse(self) -> None:
        try:
            row = self._conn.execute("PRAGMA integrity_check").fetchone()
            result = (row[0] if row else "unknown").lower()
        except sqlite3.DatabaseError as exc:
            result = f"unreadable: {exc}"
        self.last_integrity_check = time.time()
        if result == "ok":
            self.integrity_ok = True
            return

        # Section 36: stop, report, preserve.  Do not replace.
        self.integrity_ok = False
        preserved = self._preserve_damaged()
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
        self._closed = True
        raise KnowledgeBaseCorrupt(
            f"the Knowledge Base at {self.path} failed its integrity check "
            f"({result}). Nothing has been written and nothing has been "
            f"replaced."
            + (f" A copy was preserved at {preserved}." if preserved else ""))

    def _preserve_damaged(self) -> Optional[Path]:
        """Keep a copy of a damaged store before anything else happens."""
        target = self.path.with_name(
            f"{self.path.stem}.damaged-{time.strftime('%Y%m%d-%H%M%S')}"
            f"{self.path.suffix}")
        try:
            shutil.copy2(self.path, target)
        except OSError as exc:  # noqa: BLE001
            log.error("could not preserve the damaged Knowledge Base: %s", exc)
            return None
        log.error("the Knowledge Base at %s is damaged; a copy has been kept "
                  "at %s and nothing has been overwritten", self.path, target)
        return target

    def check_integrity(self) -> bool:
        """Run the check without refusing - used by the health report."""
        try:
            row = self._conn.execute("PRAGMA integrity_check").fetchone()
            self.integrity_ok = bool(row) and row[0].lower() == "ok"
        except sqlite3.DatabaseError:
            self.integrity_ok = False
        self.last_integrity_check = time.time()
        self.set_meta("last_integrity_check", str(self.last_integrity_check))
        return self.integrity_ok

    def backup(self, destination: Optional[Path] = None) -> Path:
        """Section 36.  A consistent copy, taken through SQLite itself."""
        target = Path(destination) if destination else self.path.with_suffix(
            f".backup-{time.strftime('%Y%m%d-%H%M%S')}.db")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            other = sqlite3.connect(str(target))
            try:
                self._conn.backup(other)
            finally:
                other.close()
        self.set_meta("last_backup", str(time.time()))
        self.audit("kb.backup", str(target))
        log.info("Knowledge Base backed up to %s", target)
        return target

    def checkpoint(self) -> None:
        """Section 35.  Do not wait for shutdown to make writes durable."""
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error as exc:  # noqa: BLE001
                log.debug("checkpoint failed: %s", exc)

    # ==================================================================
    # transactions (section 34)
    # ==================================================================
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """One unit of work.  A failure inside leaves nothing behind."""
        if self._closed:
            raise KnowledgeBaseError("the Knowledge Base is closed")
        if not self.integrity_ok:
            raise KnowledgeBaseCorrupt(
                "refusing to write to a Knowledge Base that failed its "
                "integrity check")
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def _count(self, table: str) -> int:
        try:
            row = self.one(f"SELECT COUNT(*) AS n FROM {table}")
            return int(row["n"]) if row else 0
        except sqlite3.Error:
            return 0

    def count(self, table: str) -> int:
        return self._count(table)

    # ==================================================================
    # metadata and audit
    # ==================================================================
    def get_meta(self, key: str) -> Optional[str]:
        row = self.one("SELECT value FROM kb_metadata WHERE key=?", (key,))
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kb_metadata(key, value, updated_at) "
                "VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET "
                "value=excluded.value, updated_at=excluded.updated_at",
                (key, value, time.time()))
            self._conn.commit()

    def audit(self, kind: str, detail: str = "", *, knowledge_id: str = "",
              source_id: str = "", run_id: str = "",
              actor: str = "system") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kb_audit(at, kind, detail, knowledge_id, "
                "source_id, run_id, actor) VALUES (?,?,?,?,?,?,?)",
                (time.time(), kind, detail[:2000], knowledge_id, source_id,
                 run_id, actor))
            self._conn.commit()

    def audit_trail(self, *, knowledge_id: str = "", limit: int = 200
                    ) -> List[Dict[str, Any]]:
        if knowledge_id:
            rows = self.query(
                "SELECT * FROM kb_audit WHERE knowledge_id=? "
                "ORDER BY id DESC LIMIT ?", (knowledge_id, limit))
        else:
            rows = self.query("SELECT * FROM kb_audit ORDER BY id DESC "
                              "LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # ==================================================================
    @property
    def initialized(self) -> bool:
        return self.get_meta(INITIALIZED_KEY) == "true"

    @property
    def initialized_at(self) -> float:
        return float(self.get_meta("initialized_at") or 0.0)

    @property
    def schema_version(self) -> int:
        return int(self.get_meta("schema_version") or 0)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Section 2's shutdown: finish writes, checkpoint, close cleanly."""
        if self._closed:
            return
        try:
            self.checkpoint()
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            self._closed = True
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


def open_knowledge_base(path: Optional[Path] = None, *, create: bool = True
                        ) -> KnowledgeStore:
    """Open the existing Knowledge Base, creating one only on a first run."""
    return KnowledgeStore(path, create=create)
