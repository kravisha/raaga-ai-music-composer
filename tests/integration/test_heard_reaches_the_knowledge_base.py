"""What the ears hear must reach the permanent Knowledge Base.

The agent has two memories.  ``knowledge.db`` is its own repository, which
``ResearchAgent`` fills and the composer reads.  ``knowledge_base.db`` is
the durable Knowledge Base, where a claim carries its evidence and a second
source agreeing attaches to the first rather than duplicating it.

Only the Training queue ever bridged into the second one, so everything the
agent learned by *listening* stopped at its own repository - a Knowledge
Base of 222 items with nothing in it from audio.  These tests hold the
other half of that bridge open.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.agent.research import LocalCorpusProvider, ResearchAgent
from raagacomposer.kb.service import KnowledgeBaseService
from raagacomposer.raaga.library import library
from tests.conftest import ANALYSIS_SR, lesson_signal

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(tmp_path):
    """A learning folder with one recording the agent can hear."""
    def _make(name: str = "Keeravani") -> Path:
        audio, _, _ = lesson_signal()
        folder = tmp_path / "learning" / name
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / f"{name}-01.wav", audio, ANALYSIS_SR)
        return tmp_path / "learning"
    return _make


def _ingest(tmp_path, settings, folder, kb, raaga, db_name="k.db"):
    settings.learning_corpus_dir = str(folder)
    repo = KnowledgeRepository(str(tmp_path / db_name))
    agent = ResearchAgent(repo, library(), settings, kb=kb)
    results = []
    try:
        for provider in agent.providers:
            if isinstance(provider, LocalCorpusProvider):
                for candidate in provider.find(raaga, "phrases", 5):
                    results.append(agent.ingest(candidate))
    finally:
        repo.close()
    return results


def _kb(tmp_path, name="kb.db"):
    return KnowledgeBaseService(path=tmp_path / name, create=True)


# --------------------------------------------------------------------------
def test_listening_writes_into_the_permanent_knowledge_base(
        tmp_path, settings, corpus, keeravani):
    kb = _kb(tmp_path)
    before = kb.store.count("knowledge_items")

    results = _ingest(tmp_path, settings, corpus(), kb, keeravani)
    assert results, "the recording should have been found"
    assert any(r.phrases_learned or r.facts_learned for r in results), \
        "nothing was heard, so this proves nothing about the bridge"

    after = kb.store.count("knowledge_items")
    assert after > before, "listening did not reach the Knowledge Base"
    assert sum(r.kb_items for r in results) > 0
    assert kb.store.count("evidence") > 0, "a claim arrived without evidence"


def test_what_was_heard_is_marked_as_heard(tmp_path, settings, corpus, keeravani):
    """A phrase from audio must not look like one read out of a book."""
    kb = _kb(tmp_path)
    _ingest(tmp_path, settings, corpus(), kb, keeravani)

    rows = kb.store.query("SELECT extraction_method FROM evidence")
    methods = {row[0] for row in rows}
    assert "audio_derived" in methods, methods

    heard = kb.store.query(
        "SELECT learned_by FROM knowledge_items WHERE learned_by = ?",
        ("listening",))
    assert heard, "nothing was recorded as learned by listening"


def test_no_knowledge_base_is_not_a_failure(tmp_path, settings, corpus, keeravani):
    """An agent built without one still learns into its own repository."""
    results = _ingest(tmp_path, settings, corpus(), None, keeravani)
    assert results
    assert all(r.error == "" for r in results)
    assert all(r.kb_items == 0 for r in results)


def test_a_second_recording_attaches_evidence_rather_than_duplicating(
        tmp_path, settings, corpus, keeravani):
    """The whole point of the Knowledge Base: agreement accumulates.

    Two recordings teaching the same phrase should leave one item carrying
    two pieces of evidence, not two items saying the same thing.
    """
    kb = _kb(tmp_path)
    folder = corpus()
    _ingest(tmp_path, settings, folder, kb, keeravani, db_name="one.db")
    items_after_first = kb.store.count("knowledge_items")
    evidence_after_first = kb.store.count("evidence")
    assert items_after_first > 0

    # The same audio again, as a different recording and a fresh repository,
    # so the agent has no memory of having heard it before.
    second = folder / "Keeravani" / "Keeravani-02.wav"
    second.write_bytes((folder / "Keeravani" / "Keeravani-01.wav").read_bytes())
    _ingest(tmp_path, settings, folder, kb, keeravani, db_name="two.db")

    assert kb.store.count("evidence") > evidence_after_first, \
        "the second hearing left no evidence behind"
    grew = kb.store.count("knowledge_items") - items_after_first
    assert grew < items_after_first, \
        "the same phrases were stored again instead of attaching evidence"
