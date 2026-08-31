"""Preparation inside the research agent, through the real ingestion path.

These go through ``ResearchAgent.ingest`` with real files on disk and the real
knowledge repository, because the point of the feature is not that the DSP
works in isolation - the unit tests cover that - but that a recording dropped
in the creator's learning folder ends up as trustworthy phrases in memory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.agent.research import LocalCorpusProvider, ResearchAgent
from raagacomposer.raaga.library import library
from tests.conftest import ANALYSIS_SR, lesson_signal, sung_signal

pytestmark = pytest.mark.integration


@pytest.fixture
def lesson_folder(tmp_path: Path):
    """A learning folder with one lesson recording in it, named for its raaga."""
    def _make(name: str = "Keeravani", **kwargs) -> Path:
        audio, _, _ = lesson_signal(**kwargs)
        folder = tmp_path / "learning" / name
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / f"{name}-lesson-01.wav", audio, ANALYSIS_SR)
        return tmp_path / "learning"
    return _make


def _agent(tmp_path: Path, settings, corpus: Path) -> ResearchAgent:
    settings.learning_corpus_dir = str(corpus)
    repo = KnowledgeRepository(str(tmp_path / "knowledge.db"))
    return ResearchAgent(repo, library(), settings)


def _ingest_all(agent: ResearchAgent, raaga):
    results = []
    for provider in agent.providers:
        if isinstance(provider, LocalCorpusProvider):
            for candidate in provider.find(raaga, "phrases", 5):
                results.append(agent.ingest(candidate))
    return results


def test_a_dropped_recording_is_prepared_before_it_is_heard(
        tmp_path, settings, lesson_folder, keeravani):
    agent = _agent(tmp_path, settings, lesson_folder())
    try:
        results = _ingest_all(agent, keeravani)
        assert results, "the file should have been found by its folder name"
        result = results[0]
        assert result.prepared is not None, "supplied audio must be prepared"
        assert result.prepared.drone.found
        assert result.prepared.silenced_seconds > 0, "the talking was kept"
        assert result.phrases_learned > 0
    finally:
        agent.repo.close()


def test_the_drone_becomes_the_tonic_rather_than_a_guess(
        tmp_path, settings, lesson_folder, keeravani):
    """G3. The tonic reaches the analysis as a fact from the tanpura instead of
    being inferred from the melody, which is the whole reason a tanpura is in
    the room."""
    agent = _agent(tmp_path, settings, lesson_folder(sa_hz=196.0))
    try:
        results = _ingest_all(agent, keeravani)
        assert results and results[0].result is not None
        assert abs(results[0].result.tonic_midi - 55.0) < 0.5
    finally:
        agent.repo.close()


def test_turning_preparation_off_restores_the_old_behaviour(
        tmp_path, settings, lesson_folder, keeravani):
    settings.learning_preprocess_recordings = False
    agent = _agent(tmp_path, settings, lesson_folder())
    try:
        results = _ingest_all(agent, keeravani)
        assert results and results[0].prepared is None
    finally:
        agent.repo.close()


def test_what_was_prepared_is_recorded_against_the_source(
        tmp_path, settings, lesson_folder, keeravani):
    """Memory has to say how a phrase was extracted, so that material ingested
    under an older pipeline can be told apart later."""
    agent = _agent(tmp_path, settings, lesson_folder())
    try:
        results = _ingest_all(agent, keeravani)
        source = agent.repo.source(results[0].source_id)
        assert "drone-notch-speech-gate" in source.extraction_version
        assert source.rights_status == "user-supplied"
    finally:
        agent.repo.close()


def test_clean_singing_with_no_drone_still_ingests(
        tmp_path, settings, keeravani):
    """Preparation must not be a requirement for material that needs none of
    it: a plain sung clip, no drone, nobody talking, still works."""
    folder = tmp_path / "learning" / "Keeravani"
    folder.mkdir(parents=True)
    sf.write(folder / "Keeravani-plain.wav", sung_signal(12.0), ANALYSIS_SR)
    agent = _agent(tmp_path, settings, tmp_path / "learning")
    try:
        results = _ingest_all(agent, keeravani)
        assert results
        assert not results[0].prepared.drone.found, "there is no drone here"
        assert results[0].analysed
    finally:
        agent.repo.close()


def test_a_prepared_ingest_reports_what_it_did(
        tmp_path, settings, lesson_folder, keeravani):
    """The creator is told what was taken out, not just what was learned."""
    agent = _agent(tmp_path, settings, lesson_folder())
    try:
        summary = _ingest_all(agent, keeravani)[0].summary()
        assert "drone at" in summary
        assert "silenced as speech" in summary
    finally:
        agent.repo.close()
