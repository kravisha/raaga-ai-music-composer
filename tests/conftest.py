"""Shared pytest configuration and fixtures.

Every suite runs against a throwaway configuration directory and throwaway
project folders, so tests never touch the creator's real settings, voice
profiles or projects.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before raagacomposer.core.settings is first imported.
_TEST_HOME = Path(tempfile.gettempdir()) / "raagacomposer-test-home"
_TEST_HOME.mkdir(parents=True, exist_ok=True)
os.environ["RAAGA_COMPOSER_HOME"] = str(_TEST_HOME)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raagacomposer.core.models import CreativeBrief          # noqa: E402
from raagacomposer.core.settings import Settings             # noqa: E402
from raagacomposer.music.melody import MelodyOptions, generate  # noqa: E402
from raagacomposer.raaga.library import RaagaLibrary, library   # noqa: E402

GOLDEN_DIR = Path(__file__).parent / "golden"


def pytest_configure(config: pytest.Config) -> None:
    for marker, description in (
        ("unit", "fast, isolated tests of one module"),
        ("integration", "several subsystems working together"),
        ("regression", "guards a defect that was found and fixed"),
        ("slow", "takes more than a few seconds"),
        ("ui", "builds real Qt widgets (offscreen)"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")


# --------------------------------------------------------------------------
# musical fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def raagas() -> RaagaLibrary:
    return library()


@pytest.fixture(scope="session")
def keeravani(raagas: RaagaLibrary):
    return raagas.require("Keeravani")


@pytest.fixture(scope="session")
def mohanam(raagas: RaagaLibrary):
    return raagas.require("Mohanam")


@pytest.fixture(scope="session")
def short_melody(keeravani):
    """A small deterministic tune reused by many tests."""
    return generate(keeravani, MelodyOptions(tempo_bpm=72, seed=7,
                                             duration_target=90))


@pytest.fixture
def brief() -> CreativeBrief:
    return CreativeBrief(situation="a man alone on a terrace after midnight",
                         mood="longing",
                         feel="lonely, late at night, but still warm",
                         language="Tamil", duration_target=90.0)


# --------------------------------------------------------------------------
# environment fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings.load()
    s.projects_dir = str(tmp_path / "projects")
    s.autosave_seconds = 5
    s.stt_provider = "none"
    # Each test gets its own memory: learning must never leak between tests.
    s.knowledge_db = str(tmp_path / "knowledge.db")
    s.learning_corpus_dir = ""
    s.learning_allow_web = False
    s.learning_autostart = False
    return s


@pytest.fixture
def app(settings: Settings):
    """A fully wired application controller with no window attached."""
    from raagacomposer.app import AppController

    controller = AppController(settings)
    try:
        yield controller
    finally:
        controller.close()


@pytest.fixture
def settle(app) -> Callable[..., None]:
    """Pump background jobs to completion the way the UI timer does."""

    def _settle(timeout: float = 240.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            app.pump()
            if not app.jobs.active_jobs():
                app.pump()
                if not app.jobs.active_jobs():
                    return
            time.sleep(0.02)
        active = [j.description for j in app.jobs.active_jobs()]
        raise TimeoutError(f"jobs did not finish: {active}")

    return _settle
