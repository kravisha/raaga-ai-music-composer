"""Project persistence: directory layout, atomic saves, SQLite metadata.

Spec sections 13 (SQLite + project directory), 15 (data model must survive
restart) and 18 (crash-safe, no silent data loss).

Every project is a self-contained folder::

    <projects_dir>/<slug>_<id>/
        project.json        canonical state, written atomically
        project.json.bak    previous good copy, used for recovery
        project.db          append-only SQLite log (history, conversation, ...)
        audio/  renders/  mixes/  exports/  voices/

The JSON file is the source of truth; the database is a crash-safe journal so
that a process killed mid-save still leaves a readable account of what
happened, and the registry lets the app list recent projects.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .logging_setup import get_logger
from .models import Project
from .serde import from_jsonable, to_jsonable
from .settings import Settings, config_dir

log = get_logger("persistence")

PROJECT_FILE = "project.json"
BACKUP_FILE = "project.json.bak"
DB_FILE = "project.db"
SUBDIRS = ("audio", "renders", "mixes", "exports", "voices")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY, at REAL, action TEXT, description TEXT, stage TEXT);
CREATE TABLE IF NOT EXISTS conversation (
    id TEXT PRIMARY KEY, at REAL, speaker TEXT, text TEXT, intent TEXT,
    interpretation TEXT, status TEXT, targets TEXT);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, at REAL, job_type TEXT, target TEXT, status TEXT,
    provider TEXT, output TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS errors (
    id TEXT PRIMARY KEY, at REAL, where_ TEXT, message TEXT, retries INTEGER,
    fallback TEXT);
CREATE TABLE IF NOT EXISTS artifacts (
    path TEXT PRIMARY KEY, at REAL, kind TEXT, note TEXT);
"""


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return s or "song"


class ProjectStore:
    """Reads and writes projects on disk."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings.load()

    # -- locations ---------------------------------------------------------
    @property
    def projects_dir(self) -> Path:
        p = Path(self.settings.projects_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def ensure_dirs(project_dir: Path) -> Path:
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        for sub in SUBDIRS:
            (project_dir / sub).mkdir(exist_ok=True)
        return project_dir

    @staticmethod
    def artifact_path(project_dir: Path, category: str, name: str) -> Path:
        d = Path(project_dir) / category
        d.mkdir(parents=True, exist_ok=True)
        return d / name

    # -- create / open / save ---------------------------------------------
    def create(self, title: str, directory: Optional[Path] = None) -> Tuple[Project, Path]:
        project = Project(title=title or "Untitled Song")
        if directory is None:
            directory = self.projects_dir / f"{slugify(project.title)}_{project.project_id[-6:]}"
        directory = self.ensure_dirs(Path(directory))
        self._init_db(directory)
        project.log_history("project.create", f"Created project {project.title!r}")
        self.save(project, directory)
        self.register(project, directory)
        log.info("created project %s at %s", project.project_id, directory)
        return project, directory

    def open(self, directory: Path) -> Project:
        directory = Path(directory)
        if directory.is_file():
            directory = directory.parent
        main = directory / PROJECT_FILE
        backup = directory / BACKUP_FILE
        project = self._read(main)
        if project is None:
            log.warning("project.json unreadable at %s, trying backup", directory)
            project = self._read(backup)
            if project is not None:
                self._record_error(directory, "open",
                                   "project.json was unreadable; recovered from backup")
        if project is None:
            raise FileNotFoundError(f"No readable project in {directory}")
        self.ensure_dirs(directory)
        self._init_db(directory)
        self.register(project, directory)
        return project

    @staticmethod
    def _read(path: Path) -> Optional[Project]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return from_jsonable(Project, data)
        except Exception as exc:  # noqa: BLE001
            log.error("failed to read %s: %s", path, exc)
            return None

    def save(self, project: Project, directory: Path) -> Path:
        directory = self.ensure_dirs(Path(directory))
        project.touch()
        main = directory / PROJECT_FILE
        tmp = directory / (PROJECT_FILE + ".tmp")
        payload = json.dumps(to_jsonable(project), indent=1, ensure_ascii=False)
        tmp.write_text(payload, encoding="utf-8")
        if main.exists():
            try:
                shutil.copy2(main, directory / BACKUP_FILE)
            except Exception:
                pass
        os.replace(tmp, main)
        try:
            self.sync_db(project, directory)
        except Exception as exc:  # noqa: BLE001 - journal must never block a save
            log.warning("db sync failed: %s", exc)
        return main

    # -- SQLite journal ----------------------------------------------------
    @staticmethod
    def _connect(directory: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(Path(directory) / DB_FILE), timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self, directory: Path) -> None:
        with self._connect(directory) as conn:
            conn.executescript(_SCHEMA)

    def sync_db(self, project: Project, directory: Path) -> None:
        with self._connect(directory) as conn:
            conn.executescript(_SCHEMA)
            conn.executemany(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                [("project_id", project.project_id),
                 ("title", project.title),
                 ("stage", project.current_stage.value),
                 ("modified_at", str(project.modified_at)),
                 ("language", project.brief.language),
                 ("raaga", project.raaga.selected),
                 ("duration_target", str(project.brief.duration_target))])
            conn.executemany(
                "INSERT OR REPLACE INTO history(id, at, action, description, stage)"
                " VALUES (?,?,?,?,?)",
                [(h.id, h.at, h.action, h.description, h.stage)
                 for h in project.history[-500:]])
            conn.executemany(
                "INSERT OR REPLACE INTO conversation"
                "(id, at, speaker, text, intent, interpretation, status, targets)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(c.id, c.at, c.speaker, c.text, c.intent, c.interpretation,
                  c.status, ",".join(c.targets)) for c in project.conversation[-500:]])
            conn.executemany(
                "INSERT OR REPLACE INTO jobs"
                "(id, at, job_type, target, status, provider, output, error)"
                " VALUES (?,?,?,?,?,?,?,?)",
                [(j.id, j.started_at, j.job_type, j.target, j.status, j.provider,
                  j.output, j.error) for j in project.jobs[-300:]])
            conn.executemany(
                "INSERT OR REPLACE INTO errors"
                "(id, at, where_, message, retries, fallback) VALUES (?,?,?,?,?,?)",
                [(e.id, e.at, e.where, e.message, e.retries, e.fallback)
                 for e in project.errors[-300:]])

    def _record_error(self, directory: Path, where: str, message: str) -> None:
        try:
            with self._connect(directory) as conn:
                conn.executescript(_SCHEMA)
                conn.execute(
                    "INSERT OR REPLACE INTO errors(id, at, where_, message, retries,"
                    " fallback) VALUES (?,?,?,?,?,?)",
                    (f"err_{int(time.time()*1000)}", time.time(), where, message, 0, ""))
        except Exception:
            pass

    def note_artifact(self, directory: Path, path: Path, kind: str, note: str = "") -> None:
        try:
            with self._connect(directory) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO artifacts(path, at, kind, note)"
                    " VALUES (?,?,?,?)", (str(path), time.time(), kind, note))
        except Exception:
            pass

    # -- registry ----------------------------------------------------------
    @staticmethod
    def registry_path() -> Path:
        return config_dir() / "projects.db"

    def register(self, project: Project, directory: Path) -> None:
        try:
            with sqlite3.connect(str(self.registry_path())) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS projects (project_id TEXT PRIMARY KEY,"
                    " title TEXT, directory TEXT, opened_at REAL, stage TEXT)")
                conn.execute(
                    "INSERT OR REPLACE INTO projects VALUES (?,?,?,?,?)",
                    (project.project_id, project.title, str(directory), time.time(),
                     project.current_stage.value))
        except Exception as exc:  # noqa: BLE001
            log.warning("registry update failed: %s", exc)
        self.settings.remember_project(str(directory))

    def recent(self, limit: int = 12) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        try:
            with sqlite3.connect(str(self.registry_path())) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS projects (project_id TEXT PRIMARY KEY,"
                    " title TEXT, directory TEXT, opened_at REAL, stage TEXT)")
                rows = conn.execute(
                    "SELECT title, directory, opened_at, stage FROM projects"
                    " ORDER BY opened_at DESC LIMIT ?", (limit,)).fetchall()
            for title, directory, opened_at, stage in rows:
                if Path(directory, PROJECT_FILE).exists():
                    out.append({"title": title, "directory": directory,
                                "opened_at": str(opened_at), "stage": stage})
        except Exception as exc:  # noqa: BLE001
            log.warning("registry read failed: %s", exc)
        return out

    # -- misc --------------------------------------------------------------
    def save_as(self, project: Project, source_dir: Optional[Path],
                target_dir: Path) -> Path:
        target_dir = self.ensure_dirs(Path(target_dir))
        if source_dir and Path(source_dir).exists():
            for sub in SUBDIRS:
                src = Path(source_dir) / sub
                if src.exists():
                    for f in src.iterdir():
                        if f.is_file():
                            try:
                                shutil.copy2(f, target_dir / sub / f.name)
                            except Exception:
                                pass
        project.log_history("project.save_as", f"Saved a copy to {target_dir}")
        self.save(project, target_dir)
        self.register(project, target_dir)
        return target_dir
