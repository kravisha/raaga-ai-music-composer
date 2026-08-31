"""Unit tests: finding material, ingesting it, and practising."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.agent import analysis
from raagacomposer.agent.curriculum import CurriculumEngine
from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.agent.practice import PracticeEngine
from raagacomposer.agent.research import (LocalCorpusProvider, ReferenceProvider,
                                          ResearchAgent, WebLeadProvider,
                                          _collapse_repeats)
from raagacomposer.core.models import Note
from raagacomposer.music import instruments as catalog
from raagacomposer.music.synth import render_notes

pytestmark = pytest.mark.unit

SR = analysis.DEFAULT_SR
TONIC = 60


@pytest.fixture
def repo(tmp_path):
    repository = KnowledgeRepository(tmp_path / "knowledge.db")
    yield repository
    repository.close()


@pytest.fixture
def research(repo, raagas, settings):
    return ResearchAgent(repo, raagas, settings)


# --------------------------------------------------------------------------
# providers and rights
# --------------------------------------------------------------------------
def test_the_reference_provider_offers_the_raagas_own_material(raagas):
    provider = ReferenceProvider()
    found = provider.find(raagas.require("Keeravani"), "phrases", 6)
    assert found
    assert all(c.rights_status == "internally-generated" for c in found)
    assert any("arohanam" in c.title for c in found)
    assert all(c.audio_loader is not None for c in found)


def test_reference_material_actually_renders(raagas):
    candidate = ReferenceProvider().find(raagas.require("Keeravani"), "", 1)[0]
    audio, sr = candidate.audio_loader()
    assert sr == SR
    assert len(audio) > SR * 0.5
    assert float(np.abs(audio).max()) > 0.01


def test_the_corpus_provider_reads_the_creators_folder(tmp_path, raagas):
    folder = tmp_path / "corpus"
    folder.mkdir()
    audio = render_notes(
        [Note(swara="S", midi=60, start=0.0, duration=0.6)],
        catalog.get("flute"), SR, total_seconds=1.0)
    sf.write(str(folder / "Keeravani-alapana.wav"), audio, SR)
    sf.write(str(folder / "something-else.wav"), audio, SR)

    found = LocalCorpusProvider(folder).find(raagas.require("Keeravani"), "", 5)
    assert len(found) == 1
    assert found[0].rights_status == "user-supplied"
    assert "Keeravani" in found[0].title


def test_the_web_provider_is_off_unless_asked(raagas):
    assert WebLeadProvider(enabled=False).find(
        raagas.require("Keeravani"), "", 3) == []


def test_leads_are_recorded_but_never_fetched(repo, raagas, settings):
    from raagacomposer.agent.research import SourceCandidate
    agent = ResearchAgent(repo, raagas, settings)
    lead = SourceCandidate(locator="lead://Keeravani/x", title="a famous concert",
                           provider="web", raaga="Keeravani",
                           rights_status="external-unverified", ingestable=False)
    result = agent.ingest(lead)
    assert result.phrases_learned == 0
    assert "lead" in result.error
    assert repo.source(result.source_id).status == "lead_only"


def test_sources_are_ranked_by_rights_then_quality(research, tmp_path, raagas,
                                                   settings):
    folder = tmp_path / "mine"
    folder.mkdir()
    audio = render_notes([Note(swara="S", midi=60, start=0.0, duration=0.6)],
                         catalog.get("flute"), SR, total_seconds=1.0)
    sf.write(str(folder / "Keeravani-mine.wav"), audio, SR)
    settings.learning_corpus_dir = str(folder)
    agent = ResearchAgent(research.repo, raagas, settings)
    found = agent.find_sources("Keeravani", "phrases", 6)
    assert found[0].rights_status == "user-supplied"


def test_a_source_is_not_ingested_twice(research):
    candidate = research.find_sources("Keeravani", "phrases", 1)[0]
    first = research.ingest(candidate)
    assert first.analysed
    again = research.ingest(candidate)
    assert "already ingested" in again.error
    assert not research.find_sources("Keeravani", "phrases", 1) or \
        research.find_sources("Keeravani", "phrases", 1)[0].locator != \
        candidate.locator


# --------------------------------------------------------------------------
# ingestion and extraction
# --------------------------------------------------------------------------
def test_listening_to_a_source_produces_phrases_with_provenance(research, repo):
    candidate = next(c for c in research.find_sources("Keeravani", "phrases", 8)
                     if "prayoga" in c.title)
    result = research.ingest(candidate)
    assert result.analysed
    assert result.phrases_learned >= 1
    phrases = repo.phrases("Keeravani")
    assert phrases
    for phrase in phrases:
        assert phrase.source_id
        assert repo.source(phrase.source_id).rights_status == \
            "internally-generated"
        assert 0.0 < phrase.confidence <= 1.0


def test_only_notes_of_that_raaga_are_remembered(research, repo, raagas):
    for candidate in research.find_sources("Keeravani", "phrases", 8):
        research.ingest(candidate)
    allowed = set(raagas.require("Keeravani").allowed)
    from raagacomposer.raaga.library import parse_swara
    for phrase in repo.phrases("Keeravani"):
        assert all(parse_swara(s)[0] in allowed for s in phrase.swaras)


def test_a_held_note_is_not_remembered_as_a_phrase():
    swaras, durations = _collapse_repeats(["G2", "G2", "G2", "P"],
                                          [0.4, 0.4, 0.4, 0.5])
    assert swaras == ["G2", "P"]
    assert durations[0] == pytest.approx(1.2)


def test_an_unreadable_source_fails_safely(research, repo, tmp_path):
    from raagacomposer.agent.research import SourceCandidate

    def broken():
        raise OSError("the file went away")

    result = research.ingest(SourceCandidate(
        locator=str(tmp_path / "gone.wav"), title="missing", provider="corpus",
        raaga="Keeravani", audio_loader=broken))
    assert not result.analysed
    assert "could not read" in result.error
    assert repo.source(result.source_id).status == "failed"


def test_silence_is_not_learned_from(research, repo):
    from raagacomposer.agent.research import SourceCandidate
    result = research.ingest(SourceCandidate(
        locator="silence://", title="silence", provider="corpus",
        raaga="Keeravani",
        audio_loader=lambda: (np.zeros(SR, np.float32), SR)))
    assert result.phrases_learned == 0
    assert repo.source(result.source_id).status in ("empty", "failed")


def test_what_was_heard_is_written_down_with_its_source(research, repo):
    for candidate in research.find_sources("Keeravani", "phrases", 3):
        research.ingest(candidate)
    observed = repo.facts("Keeravani", "observed_ascent") + \
        repo.facts("Keeravani", "observed_resting_notes")
    assert observed
    assert all(f.source_id for f in observed)


def test_the_shipped_library_is_recorded_as_a_source(research, repo):
    written = research.seed_structural_knowledge("Keeravani")
    assert written >= 8
    assert repo.best_fact("Keeravani", "arohanam") is not None
    source_id = repo.best_fact("Keeravani", "arohanam").source_id
    assert repo.source(source_id).provider == "library"
    assert repo.source(source_id).rights_status == "internally-generated"


# --------------------------------------------------------------------------
# practice
# --------------------------------------------------------------------------
@pytest.fixture
def practice(repo, raagas, settings):
    return PracticeEngine(repo, raagas, settings)


@pytest.fixture
def curriculum(repo):
    return CurriculumEngine(repo, pilot_raaga="Keeravani")


@pytest.mark.parametrize("unit_id", ["a01.sound", "a02.pitch", "a03.tonic",
                                     "a04.swara", "a05.scale", "a06.variants",
                                     "a07.direction", "a08.interval",
                                     "a09.pulse", "a10.pattern_hear",
                                     "a12.pattern_make"])
def test_every_foundation_exercise_runs_and_scores(practice, curriculum,
                                                   unit_id):
    unit = curriculum.unit(unit_id)
    assert unit is not None
    report = practice.run(unit, "Keeravani", seed=7)
    assert report.exercises, unit_id
    assert 0.0 <= report.score <= 1.0
    assert all(e.name for e in report.exercises)


def test_the_ear_training_units_are_actually_passed(practice, curriculum):
    """These are the agent's ears: if they fail, nothing else can be trusted."""
    for unit_id in ("a03.tonic", "a04.swara", "a07.direction", "a10.pattern_hear"):
        report = practice.run(curriculum.unit(unit_id), "Keeravani", seed=3)
        assert report.passed, f"{unit_id} scored {report.score:.2f}"


def test_recall_fails_before_anything_is_learned(practice, curriculum):
    unit = curriculum.unit("b02.arohanam:Keeravani")
    report = practice.run(unit, "Keeravani")
    assert not report.passed
    assert any("nothing stored" in e.heard for e in report.exercises)


def test_recall_passes_once_the_fact_is_in_memory(practice, curriculum,
                                                  research):
    research.seed_structural_knowledge("Keeravani")
    report = practice.run(curriculum.unit("b02.arohanam:Keeravani"), "Keeravani")
    assert report.passed
    assert "S R2 G2 M1 P D1 N3" in report.exercises[0].heard


def test_phrase_recall_counts_what_was_heard(practice, curriculum, research):
    research.seed_structural_knowledge("Keeravani")
    unit = curriculum.unit("b06.prayogas:Keeravani")
    assert not practice.run(unit, "Keeravani").passed
    for candidate in research.find_sources("Keeravani", "phrases", 8):
        research.ingest(candidate)
    assert practice.run(unit, "Keeravani").passed


def test_the_agent_can_tell_a_valid_phrase_from_an_invalid_one(practice,
                                                               curriculum,
                                                               research):
    research.seed_structural_knowledge("Keeravani")
    report = practice.run(curriculum.unit("b10.grammar:Keeravani"), "Keeravani",
                          seed=5)
    assert report.score >= 0.8
    assert len(report.exercises) >= 10


def test_generated_patterns_are_marked_by_the_critic(practice, curriculum,
                                                     research):
    research.seed_structural_knowledge("Keeravani")
    report = practice.run(curriculum.unit("b12.micro:Keeravani"), "Keeravani",
                          seed=9)
    assert report.evaluation is not None
    assert report.artifacts
    assert all(len(a) >= 2 for a in report.artifacts)


def test_a_generated_section_respects_its_length(practice, curriculum,
                                                 research):
    research.seed_structural_knowledge("Keeravani")
    unit = curriculum.unit("b15.geetham:Keeravani")
    report = practice.run(unit, "Keeravani", seed=4)
    assert report.artifacts
    for notes in report.artifacts:
        span = max(n.start + n.duration for n in notes)
        assert span <= unit.params["seconds"] + 1.0


def test_an_unknown_skill_type_is_reported_not_crashed(practice, curriculum):
    unit = curriculum.unit("a01.sound")
    unit.skill_type = "does.not.exist"
    report = practice.run(unit, "Keeravani")
    assert not report.passed
    assert "no practice handler" in report.detail
