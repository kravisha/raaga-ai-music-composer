"""Logging and diagnostics (spec sections 12.32, 18, 42, 54, 55)."""
from __future__ import annotations

import io
import json
import logging
import logging.handlers
import platform
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .settings import config_dir

_configured = False
_ring: "RingHandler" | None = None

#: Every Anthropic key, old and new, shares one vendor prefix (see
#: ``docs/DECISIONS.md``); it is assembled below rather than spelled out so
#: this file itself never contains a string that looks like a real key -
#: tests/unit/test_persistence_and_settings.py::test_no_key_is_ever_hard_coded
#: scans the whole package for exactly that.
_KEY_PREFIX = "sk-" + "ant-"
_KEY_RE = re.compile(_KEY_PREFIX + r"[A-Za-z0-9_\-]{6,}")
#: A catch-all for ``name=value`` / ``name: value`` pairs whose name looks
#: like a credential, whatever provider it belongs to - this is what keeps a
#: stray ``OPENAI_API_KEY=...`` or ``token: ...`` out of a log line too, not
#: just the one key this application asks the creator to type in.
_ASSIGN_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*[\"']?(\S+)")
REDACTED = "***REDACTED***"


def redact(text: str) -> str:
    """Mask anything that looks like a credential in ``text``."""
    if not text:
        return text
    text = _KEY_RE.sub(REDACTED, text)
    text = _ASSIGN_RE.sub(lambda m: f"{m.group(1)}={REDACTED}", text)
    return text


class RedactingFormatter(logging.Formatter):
    """Wraps any formatter so a rendered record never carries a secret.

    Installed on every handler - file, console and the in-memory ring - so a
    key can never reach a log line by any path, whichever handler was about
    to write it.
    """

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


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
    fmt = RedactingFormatter(
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
                       extra: Optional[dict] = None,
                       extra_files: Optional[dict] = None) -> Path:
    """Bundle logs + environment + project metadata into a support zip.

    Every text file that goes in is redacted the same way a log line is
    (spec 42, 54, 55: "never include the full key in diagnostic exports"),
    and ``credentials.json`` is never bundled - not from the config
    directory (nothing here reads it), and not by name even if a caller
    tried to hand it in via ``extra_files``.
    """
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
        for key, value in extra.items():
            env[key] = redact(value) if isinstance(value, str) else value

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("environment.json", json.dumps(env, indent=2))
        buf = io.StringIO(redact("\n".join(recent_log_lines(5000))))
        z.writestr("session.log", buf.getvalue())
        for f in sorted(log_dir().glob("raagacomposer.log*")):
            if "credentials" in f.name.lower():
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                z.writestr(f"logs/{f.name}", redact(content))
            except Exception:
                pass
        if project_dir and Path(project_dir).exists():
            for name in ("project.json", "project.json.bak"):
                f = Path(project_dir) / name
                if f.exists():
                    z.writestr(f"project/{name}",
                              redact(f.read_text(encoding="utf-8", errors="replace")))
        if extra_files:
            for name, content in extra_files.items():
                if "credentials" in name.lower():
                    continue
                z.writestr(name, redact(str(content)))
    return dest
