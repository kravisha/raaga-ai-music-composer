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
    """
    import numpy as np

    parts = [speech_signal(talk_seconds, sr=sr),
             sung_signal(sung_seconds, sr=sr),
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
    s.learning_corpus_dir = ""
    s.learning_allow_web = False
    s.learning_autostart = False
    return s


@pytest.fixture
def training_settings(settings: Settings, tmp_path: Path) -> Settings:
    settings.training_db = str(tmp_path / "training.db")
    settings.training_allow_web = False
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
