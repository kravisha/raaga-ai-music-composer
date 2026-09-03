"""Unit tests: secrets never survive into a log line or a diagnostics export
(spec sections 42, 54, 55).

``setup_logging`` configures one process-wide ``raaga`` logger the first time
it is called and never again (``_configured`` guards it), so a test that
wants to exercise it has to save and restore that global state itself -
otherwise handlers pointed at a ``tmp_path`` that pytest deletes afterwards
would go on leaking into every later test in the session.  ``isolated_logging``
below does exactly that.
"""
from __future__ import annotations

import logging
import zipfile

import pytest

from raagacomposer.core import logging_setup
from raagacomposer.core.logging_setup import (export_diagnostics, get_logger,
                                              recent_log_lines, redact,
                                              setup_logging)

pytestmark = pytest.mark.unit

FAKE_KEY = "sk-ant-api03-FAKEFAKEFAKEFAKE"


@pytest.fixture
def isolated_logging():
    root = logging.getLogger("raaga")
    orig_configured = logging_setup._configured
    orig_ring = logging_setup._ring
    orig_handlers = list(root.handlers)
    orig_level = root.level
    for h in orig_handlers:
        root.removeHandler(h)
    logging_setup._configured = False
    logging_setup._ring = None
    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        for h in orig_handlers:
            root.addHandler(h)
        root.setLevel(orig_level)
        logging_setup._configured = orig_configured
        logging_setup._ring = orig_ring


# --------------------------------------------------------------------------
# the pure function
# --------------------------------------------------------------------------
def test_redact_masks_a_claude_key():
    text = f"using key {FAKE_KEY} to call Claude"
    out = redact(text)
    assert FAKE_KEY not in out
    assert "***REDACTED***" in out


def test_redact_masks_generic_assignment_forms():
    assert "***REDACTED***" in redact("api_key=abcdef123456")
    assert "***REDACTED***" in redact('token: "abcdef123456"')
    assert "***REDACTED***" in redact("SECRET=abcdef123456")
    assert "***REDACTED***" in redact("password = hunter2value")


# --------------------------------------------------------------------------
# wired into the real handlers
# --------------------------------------------------------------------------
def test_key_is_masked_in_the_ring_buffer(tmp_path, monkeypatch, isolated_logging):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    setup_logging("INFO")
    log = get_logger("test-redaction")
    log.info("Claude key configured: %s", FAKE_KEY)
    tail = "\n".join(recent_log_lines(50))
    assert FAKE_KEY not in tail
    assert "***REDACTED***" in tail


def test_key_is_masked_in_the_log_file(tmp_path, monkeypatch, isolated_logging):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))
    setup_logging("INFO")
    log = get_logger("test-redaction-file")
    log.info("here is a key: %s", FAKE_KEY)
    for handler in logging.getLogger("raaga").handlers:
        handler.flush()
    log_file = logging_setup.log_dir() / "raagacomposer.log"
    content = log_file.read_text(encoding="utf-8")
    assert FAKE_KEY not in content
    assert "***REDACTED***" in content


# --------------------------------------------------------------------------
# the diagnostics bundle
# --------------------------------------------------------------------------
def test_diagnostics_export_never_contains_the_key_or_credentials_file(
        tmp_path, monkeypatch):
    monkeypatch.setenv("RAAGA_COMPOSER_HOME", str(tmp_path))

    # Simulate a key that reached a log line by some path this test does not
    # control - export_diagnostics must scrub it regardless of where it came
    # from, not merely rely on nothing upstream ever logging one.
    log_dir = logging_setup.log_dir()
    (log_dir / "raagacomposer.log").write_text(
        f"09:00:00 INFO some.module a stray key: {FAKE_KEY}\n", encoding="utf-8")

    # A credentials.json in the config directory must never be swept in.
    (tmp_path / "credentials.json").write_text(
        '{"anthropic_api_key": "' + FAKE_KEY + '"}', encoding="utf-8")

    out = export_diagnostics(
        tmp_path / "diag.zip",
        extra={"note": f"key in flight: {FAKE_KEY}"},
        extra_files={"providers.txt": f"claude detail: {FAKE_KEY}",
                     "credentials.json": '{"leak": "' + FAKE_KEY + '"}'})

    with zipfile.ZipFile(out) as z:
        names = z.namelist()
        assert not any("credentials" in n.lower() for n in names)
        assert any(n.startswith("logs/") for n in names)
        for name in names:
            data = z.read(name)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            assert FAKE_KEY not in text, f"key leaked into {name}"
