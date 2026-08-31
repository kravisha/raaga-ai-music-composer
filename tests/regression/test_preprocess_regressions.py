"""Defects found while making real recordings usable, each named.

The one that matters is the first: before preparation existed, a recording of
a lesson taught the agent phrases nobody had sung.  They went into permanent
memory with provenance and a confidence, were rebuilt into the raaga by
``learned_raaga`` and handed to the composer as its characteristic phrases.
Wrong phrases there are worse than no phrases at all, because everything
downstream is built to trust that store.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from raagacomposer.agent import preprocess as P
from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.agent.research import LocalCorpusProvider, ResearchAgent
from raagacomposer.core.settings import Settings
from raagacomposer.music import instruments, melody, synth
from raagacomposer.raaga.library import library as _library, parse_swara
from tests.conftest import (ANALYSIS_SR, drone_signal, speech_signal,
                            sung_signal)

pytestmark = pytest.mark.regression


def _lesson_of(raaga, sa_hz: float = 261.63, talk: float = 8.0):
    """A lesson recording whose sung melody we know note for note."""
    tune = melody.generate(raaga, melody.MelodyOptions(
        seed=5, duration_target=24.0, tonic_midi=60))
    voice = synth.render_notes(tune.notes, instruments.get("veena"), ANALYSIS_SR)
    voice = voice / (np.abs(voice).max() + 1e-9)
    mixed = np.concatenate([speech_signal(talk), voice,
                            speech_signal(talk)]).astype(np.float32)
    mixed = 0.75 * mixed + 0.55 * drone_signal(len(mixed) / ANALYSIS_SR, sa_hz)
    truth = " ".join(parse_swara(n.swara)[0] for n in tune.notes)
    return mixed.astype(np.float32), truth


def _learn_from(audio: np.ndarray, raaga, preprocess: bool):
    """Ingest one recording and return the phrases it produced."""
    folder = Path(tempfile.mkdtemp())
    (folder / raaga.name).mkdir()
    sf.write(folder / raaga.name / "lesson.wav", audio, ANALYSIS_SR)
    database = tempfile.mktemp(suffix=".db")
    repo = KnowledgeRepository(database)
    try:
        settings = Settings()
        settings.learning_corpus_dir = str(folder)
        settings.learning_preprocess_recordings = preprocess
        agent = ResearchAgent(repo, _library(), settings)
        for provider in agent.providers:
            if isinstance(provider, LocalCorpusProvider):
                for candidate in provider.find(raaga, "phrases", 5):
                    agent.ingest(candidate)
        return [[parse_swara(s)[0] for s in phrase.swaras]
                for phrase in repo.phrases(raaga=raaga.name, limit=500)]
    finally:
        repo.close()
        os.unlink(database)


def test_reg_a_lesson_recording_taught_phrases_nobody_sang(keeravani):
    """Talking over a drone was heard as melody and remembered as prayogas.

    The check is precision against ground truth, not how many phrases came
    back: the raw pipeline is happy to produce *more* phrases here, and almost
    all of them are inventions.
    """
    audio, truth = _lesson_of(keeravani)

    def precision(phrases) -> float:
        if not phrases:
            return 0.0
        real = sum(any(" ".join(p[i:i + 3]) in truth
                       for i in range(max(1, len(p) - 2)))
                   for p in phrases)
        return real / len(phrases)

    raw = _learn_from(audio, keeravani, preprocess=False)
    prepared = _learn_from(audio, keeravani, preprocess=True)

    assert raw, "the raw pipeline should still produce something to compare"
    assert precision(raw) < 0.5, (
        "this test is only meaningful while the unprepared pipeline really "
        "does invent phrases")
    assert precision(prepared) > 0.9, (
        f"prepared precision was {precision(prepared):.0%}; phrases that were "
        f"never sung are reaching permanent memory")


def test_reg_the_gate_is_not_applied_to_rendered_exercises(keeravani):
    """The agent's own reference material is one clean voice with no drone and
    nobody talking.  Preparing it could only take good phrases away, so it is
    left alone - a check that the decision is made on the source's rights
    status rather than on every source alike.
    """
    settings = Settings()
    settings.learning_preprocess_recordings = True
    database = tempfile.mktemp(suffix=".db")
    repo = KnowledgeRepository(database)
    try:
        agent = ResearchAgent(repo, _library(), settings)
        for provider in agent.providers:
            for candidate in provider.find(keeravani, "phrases", 3):
                should = agent._should_preprocess(candidate)
                if candidate.rights_status == "user-supplied":
                    assert should
                else:
                    assert not should, (
                        f"{candidate.rights_status} material should not be "
                        f"put through the gate")
    finally:
        repo.close()
        os.unlink(database)


def test_reg_silencing_speech_does_not_move_later_timestamps():
    """Splicing rejected stretches out would shift every note after the cut,
    and phrase segmentation reads the gaps between notes to find its
    boundaries.  The audio must stay the length it was.
    """
    audio = np.concatenate([
        speech_signal(4.0),
        0.8 * np.sin(2 * np.pi * 220 * np.arange(ANALYSIS_SR * 6) / ANALYSIS_SR),
        speech_signal(4.0)]).astype(np.float32)
    prepared = P.prepare(audio, ANALYSIS_SR, remove_drone=False)
    assert len(prepared.audio) == len(audio)


def test_reg_a_drone_is_not_reported_from_its_own_bin(keeravani):
    """The winning FFT bin is about 5 Hz wide, which near a low Sa is most of a
    semitone.  Reported unrefined, the tonic was up to 45 cents out and every
    swara measured from it inherited that error.
    """
    # The melody has to move.  A held tone is a drone by any definition worth
    # having, and asking the detector to prefer one steady pitch over another
    # would be asking it to read minds.
    for sa_hz in (146.83, 174.61, 233.08):
        mixed = (0.7 * sung_signal(6.0)
                 + 0.6 * drone_signal(6.0, sa_hz=sa_hz)).astype(np.float32)
        drone = P.detect_drone(mixed, ANALYSIS_SR)
        assert drone.found
        error = abs(1200.0 * np.log2(drone.hz / sa_hz))
        assert error < 25, f"tonic for Sa={sa_hz} was {error:.0f} cents out"


def test_reg_a_long_held_note_is_not_notched_out_as_a_drone():
    """A sustained note in an alapana is stationary, and the detector took it
    for a shruti box and filtered it out - removing the very thing the
    recording was worth listening to.

    What separates them is that a drone never stops and sounds a chord of
    partials, where a held note brings only itself.
    """
    from tests.conftest import gamaka_signal

    for seconds in (12.0, 20.0):
        assert not P.detect_drone(sung_signal(seconds), ANALYSIS_SR).found, \
            f"a {seconds:.0f}s held-note passage was taken for a drone"
    assert not P.detect_drone(gamaka_signal(12.0), ANALYSIS_SR).found

    # ... while a real drone is still found, including under singing.
    mixed = (0.7 * sung_signal(12.0)
             + 0.6 * drone_signal(12.0)).astype(np.float32)
    assert P.detect_drone(mixed, ANALYSIS_SR).found
