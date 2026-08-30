"""Logging and diagnostics (spec sections 12.32, 18)."""
from __future__ import annotations

import io
import json
import logging
import logging.handlers
import platform
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .settings import config_dir

_configured = False
_ring: "RingHandler" | None = None


class RingHandler(logging.Handler):
    """Keeps the last N formatted records in memory for the diagnostics pane."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self.capacity = capacity
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self.records.append(msg)
        if len(self.records) > self.capacity:
            del self.records[: len(self.records) - self.capacity]

    def tail(self, n: int = 300) -> List[str]:
        return self.records[-n:]


def log_dir() -> Path:
    p = config_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _configured, _ring
    root = logging.getLogger("raaga")
    if _configured:
        root.setLevel(getattr(logging, level.upper(), logging.INFO))
        return root

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-28s %(message)s", "%H:%M:%S")

    fh = logging.handlers.RotatingFileHandler(
        log_dir() / "raagacomposer.log", maxBytes=2_000_000, backupCount=5,
        encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    _ring = RingHandler()
    _ring.setFormatter(fmt)
    root.addHandler(_ring)

    root.propagate = False
    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"raaga.{name}")


def recent_log_lines(n: int = 300) -> List[str]:
    return _ring.tail(n) if _ring else []


def export_diagnostics(dest: Path, project_dir: Optional[Path] = None,
                       extra: Optional[dict] = None) -> Path:
    """Bundle logs + environment + project metadata into a support zip."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import PySide6
        env["PySide6"] = PySide6.__version__
    except Exception:
        pass
    if extra:
        env.update(extra)

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("environment.json", json.dumps(env, indent=2))
        buf = io.StringIO("\n".join(recent_log_lines(5000)))
        z.writestr("session.log", buf.getvalue())
        for f in sorted(log_dir().glob("raagacomposer.log*")):
            try:
                z.write(f, f"logs/{f.name}")
            except Exception:
                pass
        if project_dir and Path(project_dir).exists():
            for name in ("project.json", "project.json.bak"):
                f = Path(project_dir) / name
                if f.exists():
                    z.write(f, f"project/{name}")
    return dest
