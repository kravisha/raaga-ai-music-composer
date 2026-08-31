"""Regression tests for the learning agent.

Each one guards a defect that was actually found while building it.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from raagacomposer.agent import analysis
from raagacomposer.agent.curriculum import CurriculumEngine
from raagacomposer.agent.knowledge import KnowledgeRepository
from raagacomposer.agent.music_agent import MusicAgent
from raagacomposer.agent.practice import PracticeEngine
from raagacomposer.agent.research import (ReferenceProvider, ResearchAgent,
                                          _collapse_repeats)
from raagacomposer.core.models import Note
from raagacomposer.music import instruments as catalog
from raagacomposer.music.synth import render_notes
from raagacomposer.speech.intent import interpret
from raagacomposer.speech.timeline_parser import TimeContext

pytestmark = pytest.mark.regression

SR = analysis.DEFAULT_SR
TONIC = 60


@pytest.fixture
def repo(tmp_path):
    repository = KnowledgeRepository(tmp_path / "k.db")
    yield repository
    repository.close()


# --------------------------------------------------------------------------
# REG-090  hearing
# --------------------------------------------------------------------------
def test_reg_090_naming_one_note_needs_a_given_tonic(raagas):
    """Estimating the tonic from a single note makes every note "S".

    Swara naming exercises rendered one note and let the pipeline guess the
    tonic; the guess was always the note itself, so the agent answered "S"
    every time and could never pass its fourth lesson.
    """
    raaga = raagas.require("Keeravani")
    audio = render_notes([Note(swara="P", midi=raaga.midi("P", TONIC),
                               start=0.0, duration=0.8)],
                         catalog.get("flute"), SR, total_seconds=1.2)
    given = analysis.analyse(audio, SR, raaga, fixed_tonic_midi=float(TONIC))
    assert given.notes[0].swara == "P"


def test_reg_091_a_flat_upper_sa_is_still_sa(raagas):
    """A note a little below the upper Sa was named as the leading note.

    nearest_swara only considered targets inside one octave, so 11.6 semitones
    above the tonic could not round up to S+.
    """
    raaga = raagas.require("Keeravani")
    for offset, expected in ((11.6, "S+"), (12.0, "S+"), (-0.4, "S"),
                             (23.6, "S++")):
        heard, _ = analysis.nearest_swara(TONIC + offset, TONIC, raaga)
        assert heard == expected, offset


def test_reg_092_the_descending_exercise_actually_descends(raagas, repo,
                                                           settings):
    """The direction lesson built its descending line from a note set with the
    octave marks stripped, so the "descending" example rose instead."""
    engine = PracticeEngine(repo, raagas, settings)
    curriculum = CurriculumEngine(repo)
    report = engine.run(curriculum.unit("a07.direction"), "Keeravani", seed=4)
    assert report.passed, report.summary()
    assert {e.expected for e in report.exercises} == {"up", "down"} or \
        len(report.exercises) < 4


# --------------------------------------------------------------------------
# REG-093  what is worth remembering
# --------------------------------------------------------------------------
def test_reg_093_a_held_note_is_not_a_phrase():
    """A sustained note arrives from the tracker as a run of identical swaras.

    Stored as-is it became the "phrase" D1 D1 D1 D1 D1 D1, which the composer
    then dutifully quoted back as six repeated notes.
    """
    swaras, durations = _collapse_repeats(["D1"] * 6, [0.3] * 6)
    assert swaras == ["D1"]
    assert durations == [pytest.approx(1.8)]


def test_reg_093b_phrases_need_more_than_one_note(repo, raagas, settings):
    agent = ResearchAgent(repo, raagas, settings)
    for candidate in agent.find_sources("Keeravani", "phrases", 8):
        agent.ingest(candidate)
    for phrase in repo.phrases("Keeravani"):
        assert len(phrase.swaras) >= 3
        assert len(set(phrase.swaras)) >= 3, phrase.swaras
        assert not all(s == phrase.swaras[0] for s in phrase.swaras)


# --------------------------------------------------------------------------
# REG-094  learning grammar, not songs
# --------------------------------------------------------------------------
def test_reg_094_quotations_are_short_and_in_range(repo, raagas, settings):
    """Whole learned phrases were being quoted verbatim, which failed the
    originality check, and quoting across octaves made the line leap."""
    from raagacomposer.agent.knowledge import Phrase
    from raagacomposer.agent.originality import PhraseIndex, check

    repo.add_phrase(Phrase(raaga="Keeravani",
                           swaras=["S+", "N3+", "D1+", "P+", "M1+", "G2+"],
                           confidence=0.9))
    engine = PracticeEngine(repo, raagas, settings)
    curriculum = CurriculumEngine(repo)
    report = engine.run(curriculum.unit("b13.short_phrase:Keeravani"),
                        "Keeravani", seed=6)
    index = PhraseIndex.from_repository(repo, "Keeravani")
    for notes in report.artifacts:
        swaras = [n.swara for n in notes]
        assert check(swaras, index).is_original, " ".join(swaras)
        span = max(n.midi for n in notes) - min(n.midi for n in notes)
        assert span <= 24, "the line jumps more than two octaves"


# --------------------------------------------------------------------------
# REG-095  the loop must always move
# --------------------------------------------------------------------------
def test_reg_095_research_does_not_spin_when_material_runs_out(tmp_path,
                                                               settings,
                                                               raagas):
    """A unit needing more phrases asked for material forever: the search
    returned nothing new, no attempt was made, and the loop never advanced."""
    settings.knowledge_db = str(tmp_path / "k.db")
    agent = MusicAgent(settings, raagas)
    try:
        seen = []
        for _ in range(60):
            step = agent.learn_step()
            seen.append((step.action, step.unit_id))
            if step.action == "idle":
                break
        # No unit may be researched more than a handful of times in a row.
        streak = longest = 0
        previous = None
        for action, unit_id in seen:
            if (action, unit_id) == previous:
                streak += 1
            else:
                streak = 1
            previous = (action, unit_id)
            longest = max(longest, streak)
        assert longest <= 4, f"the loop stalled: {seen[-6:]}"
        assert agent.curriculum.stage_summary()["overall_percent"] > 80
    finally:
        agent.close()


def test_reg_096_the_reference_provider_offers_all_it_can_play(raagas):
    """Truncating the provider's own list to the per-lesson limit meant the
    agent ran out of practice material after one lesson."""
    raaga = raagas.require("Keeravani")
    offered = ReferenceProvider().find(raaga, "phrases", 3)
    assert len(offered) > 3
    assert len(offered) >= 2 + len(raaga.prayogas)


def test_reg_098_a_rested_lesson_is_revisited_rather_than_idling(repo):
    """One hard-failing foundation unit blocked the whole curriculum."""
    curriculum = CurriculumEngine(repo, revisit_after=1e9)
    unit = curriculum.next_unit()
    for _ in range(unit.max_failures()):
        curriculum.record_attempt(unit, 0.3, False)
    revisited = curriculum.next_unit()
    assert revisited is not None and revisited.id == unit.id


# --------------------------------------------------------------------------
# REG-097  shutdown
# --------------------------------------------------------------------------
def test_reg_097_closing_waits_for_the_learner(tmp_path, settings, raagas):
    """Closing memory while the background thread was mid-query took the whole
    process down with an access violation."""
    settings.knowledge_db = str(tmp_path / "k.db")
    agent = MusicAgent(settings, raagas)
    agent.start_learning()
    time.sleep(0.6)
    assert agent.is_learning
    agent.close()                      # must not crash and must not hang
    assert not agent.is_learning
    assert agent.repo.closed
    names = [t.name for t in threading.enumerate()]
    assert "raaga-learning" not in names


def test_reg_097b_closing_twice_is_harmless(tmp_path, settings, raagas):
    settings.knowledge_db = str(tmp_path / "k.db")
    agent = MusicAgent(settings, raagas)
    agent.close()
    agent.close()
    assert agent.repo.closed


# --------------------------------------------------------------------------
# REG-099  instructions to the agent
# --------------------------------------------------------------------------
def test_reg_099_learn_a_raaga_is_not_read_as_choosing_one():
    """"Learn Keeravani" was classified as raaga.set because the raaga-name
    sweep ran after the rules and overwrote an unclassified intent."""
    ctx = TimeContext(duration=120.0)
    learn = interpret("Learn Keeravani.", ctx)
    assert learn.intent == "agent.learn"
    assert learn.raaga == "Keeravani"

    choose = interpret("Use raaga Keeravani.", ctx)
    assert choose.intent == "raaga.set"


def test_reg_099b_agent_instructions_do_not_shadow_the_composer():
    """The agent rules run first, so they must be specific enough not to
    swallow ordinary composing instructions."""
    ctx = TimeContext(duration=120.0)
    for text, expected in (("Add veena here.", "arrange.add"),
                           ("Play the first minute.", "transport.play"),
                           ("Give me the song without instruments.",
                            "voice.vocal_only"),
                           ("Replace violin with veena.", "arrange.replace"),
                           ("Make this part lighter.", "arrange.level")):
        assert interpret(text, ctx).intent == expected, text
