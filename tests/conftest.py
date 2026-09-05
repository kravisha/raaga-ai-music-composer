"""Shared pytest configuration and fixtures.

Every suite runs against a throwaway configuration directory and throwaway
project folders, so tests never touch the creator's real settings, voice
profiles or projects.
"""
from __future__ import annotations

import json
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
# Forces the credentials.json fallback for every SecretStore built with no
# explicit backend, so the suite can never read or write the machine's real
# Windows Credential Manager.  A test that exercises the keyring branch on
# purpose injects a fake backend object into SecretStore instead - see
# tests/unit/test_secrets.py.
os.environ["RAAGA_SECRET_BACKEND"] = "file"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# The suite must give the same answer on a machine with Ollama running and a
# key in the environment as on one with neither, so both are switched off
# here - in the one file every suite loads - and no test can reach a real
# model or spend real money.  The settings file is rewritten at the start of
# every session for the same reason: this directory outlives a run, and a
# stale projects_dir left in it once made a regression test depend on which
# of last week's temporary folders still existed.
os.environ.pop("ANTHROPIC_API_KEY", None)
(_TEST_HOME / "settings.json").write_text(json.dumps({
    "llm_provider": "off",
    "llm_routing": "off",
    "projects_dir": str(_TEST_HOME / "projects"),
    "recent_projects": [],
}, indent=2), encoding="utf-8")

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
# synthetic recordings
# --------------------------------------------------------------------------
# A real lesson is a teacher talking over a drone and occasionally singing.
# None of the three can be tested against a rendered exercise, because a
# rendered exercise has none of them, so they are built here instead.  The
# distinction the preprocessing turns on is pitch that is *held* (singing)
# against pitch that is always sliding (speech), so that is what these differ in.
ANALYSIS_SR = 22050


def drone_signal(seconds: float, sa_hz: float = 261.63,
                 sr: int = ANALYSIS_SR):
    """A tanpura or shruti box: the Pa below, Sa, and Sa's own harmonics."""
    import numpy as np

    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    for freq, amp in ((sa_hz * 2 / 3, 0.45), (sa_hz, 0.60),
                      (sa_hz * 2, 0.30), (sa_hz * 3, 0.15)):
        out += amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return out / (np.abs(out).max() + 1e-9)


def sung_signal(seconds: float, base_hz: float = 220.0,
                sr: int = ANALYSIS_SR, degrees=(0, 2, 3, 5, 7, 8, 7, 5)):
    """Singing: steps between scale degrees and *holds* each one."""
    import numpy as np

    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = np.zeros(n)
    step = max(1, n // len(degrees))
    for i, degree in enumerate(degrees):
        f[i * step:(i + 1) * step] = base_hz * 2 ** (degree / 12)
    f[len(degrees) * step:] = base_hz * 2 ** (degrees[-1] / 12)
    f *= 1 + 0.012 * np.sin(2 * np.pi * 5.5 * t)        # vibrato and gamaka
    phase = 2 * np.pi * np.cumsum(f) / sr
    out = (0.60 * np.sin(phase) + 0.25 * np.sin(2 * phase)
           + 0.10 * np.sin(3 * phase))
    return out.astype(np.float32)


def gamaka_signal(seconds: float, base_hz: float = 220.0,
                  sr: int = ANALYSIS_SR, swing_cents: float = 90.0):
    """Singing with heavy kampita: the pitch swings most of a semitone either
    way.  The raw contour is nowhere near flat, but it oscillates *around* each
    note rather than travelling, which is what makes it singing."""
    import numpy as np

    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = np.zeros(n)
    degrees = (0, 2, 3, 5, 7, 8, 7, 5)
    step = max(1, n // len(degrees))
    for i, degree in enumerate(degrees):
        f[i * step:(i + 1) * step] = base_hz * 2 ** (degree / 12)
    f[len(degrees) * step:] = base_hz * 2 ** (degrees[-1] / 12)
    f *= 2 ** ((swing_cents / 1200) * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    out = (0.60 * np.sin(phase) + 0.25 * np.sin(2 * phase)
           + 0.10 * np.sin(3 * phase))
    return out.astype(np.float32)


def speech_signal(seconds: float, base_hz: float = 115.0,
                  sr: int = ANALYSIS_SR):
    """Speech: pitch glides continuously and never settles, and consonants
    break the voicing up."""
    import numpy as np

    n = int(seconds * sr)
    t = np.arange(n) / sr
    f = base_hz * (1 + 0.45 * np.sin(2 * np.pi * 1.7 * t)
                   + 0.25 * np.sin(2 * np.pi * 3.9 * t + 1.0))
    phase = 2 * np.pi * np.cumsum(f) / sr
    out = (0.60 * np.sin(phase) + 0.30 * np.sin(2 * phase)
           + 0.20 * np.sin(3 * phase)).astype(np.float32)
    envelope = np.ones(n, dtype=np.float32)
    for i in range(0, n, int(0.28 * sr)):
        envelope[i:i + int(0.09 * sr)] = 0.0
    return out * envelope


def lesson_signal(talk_seconds: float = 4.0, sung_seconds: float = 6.0,
                  sa_hz: float = 261.63, sr: int = ANALYSIS_SR):
    """Talking, then singing, then talking - all over a drone.

    Returns (audio, sung_start, sung_end) so a test can say where the singing
    actually was.

    The singing is built on ``sa_hz``, the drone's own Sa.  It used to take
    ``sung_signal``'s default of 220 Hz while the drone sat at 261.63, three
    semitones adrift, so that relative to its own tanpura the "Keeravani
    lesson" sang G3 and D2 - two notes Keeravani does not contain.  Nothing
    caught it, because the analysis was told the raaga and snapped every
    pitch into it.  A singer and their drone agree about Sa; the fixture
    now does too.
    """
    import numpy as np

    parts = [speech_signal(talk_seconds, sr=sr),
             sung_signal(sung_seconds, base_hz=sa_hz, sr=sr),
             speech_signal(talk_seconds, sr=sr)]
    voice = np.concatenate(parts).astype(np.float32)
    mixed = 0.75 * voice + 0.55 * drone_signal(len(voice) / sr, sa_hz, sr)
    return mixed.astype(np.float32), talk_seconds, talk_seconds + sung_seconds


@pytest.fixture
def lesson_recording():
    """The shape of the problem: a lesson, not a rendered exercise."""
    return lesson_signal


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
    s.factory_db = str(tmp_path / "factory.db")
    s.learning_corpus_dir = ""
    s.learning_allow_web = False
    s.learning_autostart = False
    return s


@pytest.fixture
def kb_path(tmp_path: Path) -> Path:
    """A throwaway Knowledge Base file.

    Section 2 of the knowledge-base specification allows an isolated temporary
    database for a test, and forbids one anywhere else: every test here is
    explicitly asking for its own.
    """
    return tmp_path / "knowledge_base.db"


@pytest.fixture
def kb(kb_path: Path):
    """An empty Knowledge Base with the core taxonomy seeded."""
    from raagacomposer.kb.service import KnowledgeBaseService

    service = KnowledgeBaseService.initialize_if_needed(kb_path)
    try:
        yield service
    finally:
        if not service.store.closed:
            service.close()


@pytest.fixture
def kb_source(kb):
    """A source to hang evidence from."""
    from raagacomposer.kb.models import Source

    return kb.add_source(Source(
        source_type="video", title="A Kambhoji lesson",
        author_or_channel="a teacher",
        reference="https://youtu.be/AAAAAAAAAAA"))


@pytest.fixture
def kb_evidence(kb_source):
    """A supporting evidence record from the default source."""
    from raagacomposer.kb.models import Evidence, ExtractionMethod

    def _make(source=None, *, supports=True, strength=0.8,
              method=ExtractionMethod.AUDIO, run_id=""):
        return Evidence(
            source_id=(source or kb_source).source_id,
            source_segment="0:10-0:40", timestamp_start=10.0,
            timestamp_end=40.0, strength=strength,
            extraction_method=method, supports=supports, run_id=run_id)
    return _make


@pytest.fixture
def kb_claim():
    """A claim about Kambhoji, ready to commit."""
    from raagacomposer.kb import normalize
    from raagacomposer.kb.models import KnowledgeItem, KnowledgeType, Scope

    def _make(predicate="arohanam", value="S R2 G3 M1 P D2 S+",
              statement="", *, raga="Kambhoji",
              knowledge_type=KnowledgeType.FACT, tags=()):
        return KnowledgeItem(
            canonical_name=raga, knowledge_type=knowledge_type, subject=raga,
            predicate=predicate, object_value=value,
            statement=statement or f"{raga} {predicate} is {value}.",
            structured_value=normalize.structured_for(predicate, value),
            scope=[Scope.CARNATIC, Scope.RAGA], raga=raga, tags=list(tags))
    return _make


@pytest.fixture
def training_settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.training_db = str(tmp_path / "training.db")
    settings.training_allow_web = False
    settings.knowledge_base_db = str(tmp_path / "knowledge_base.db")
    return settings


@pytest.fixture
def agent_repo(tmp_path: Path):
    """The agent's own memory, so the training bridge has somewhere to land."""
    from raagacomposer.agent.knowledge import KnowledgeRepository

    repo = KnowledgeRepository(tmp_path / "knowledge.db")
    try:
        yield repo
    finally:
        repo.close()


@pytest.fixture
def training(training_settings: Settings, agent_repo):
    """A Training controller with its own store, wired to the agent."""
    from raagacomposer.training.controller import TrainingController

    controller = TrainingController(training_settings, agent_repo=agent_repo)
    try:
        yield controller
    finally:
        controller.close()


@pytest.fixture
def training_store(tmp_path: Path):
    from raagacomposer.training.store import TrainingStore

    store = TrainingStore(tmp_path / "training-only.db")
    try:
        yield store
    finally:
        store.close()


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
